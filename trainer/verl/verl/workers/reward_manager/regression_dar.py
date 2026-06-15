# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import defaultdict

import re
import numpy as np
import pandas as pd
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

from .utils import parse_prediction, calculate_metrics, compute_training_metrics


@register("regression_dar")
class RegressionDistributionAwareRewardManager:
    """Reward manager with distribution-aware CRPS + leave-one-out credit assignment.
    - Uses CRPS over the empirical distribution of K rollouts per uid.
    - Per-trajectory reward r_k = CRPS(F_{-k}, y) - CRPS(F, y)
      (positive if adding rollout k improves the distribution).
    - Falls back to instance-wise negative MAE when grouping / parsing fails.
    """

    def __init__(self, 
                 tokenizer, 
                 num_examine, 
                 compute_score=None, 
                 reward_fn_key="data_source",
                 ) -> None:
        """
        Initialize the DistributionAwareRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        # Extra discount applied to invalid parses after copying the worst valid reward
        self.invalid_parse_penalty = 10.0
        finfo = torch.finfo(torch.float32)
        self._float32_max = float(finfo.max)
        self._float32_min = float(finfo.min)

    def __call__(self, data: DataProto, return_dict=False):
        """Compute distribution-aware rewards for a batch."""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        uid_arr = data.non_tensor_batch.get("uid", None)
        uids = []

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        last_token_pos = [0 for _ in range(len(data))]
        base_scores = [0.0 for _ in range(len(data))]     # instance-wise negative MAE
        preds = [None for _ in range(len(data))]          # numeric predictions
        targets = [None for _ in range(len(data))]        # numeric labels
        invalid_mask = [False for _ in range(len(data))]
        indices_per_uid = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            uid = uid_arr[i] if uid_arr is not None else None
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            last_token_pos[i] = int(valid_response_length.item() - 1)

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            parsed_result = parse_prediction(response_str)
            if isinstance(parsed_result, float):
                score = float(-np.abs(parsed_result - ground_truth))
            else:
                score = float("-inf")
            
            if isinstance(parsed_result, float):
                preds[i] = float(parsed_result)
                targets[i] = float(ground_truth)
            else:
                preds[i] = None
                targets[i] = float(ground_truth)
                invalid_mask[i] = True

            # base instance-wise reward (negative MAE)
            if isinstance(score, dict):
                base_reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                base_reward = score

            base_scores[i] = float(base_reward)

            if uid is not None:
                indices_per_uid[uid].append(i)
                uids.append(uid)

            # debug printing (base score)
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", base_reward)
        
        valid_idx = [i for i, p in enumerate(preds) if isinstance(p, float)]
        preds_clean = np.array([preds[i] for i in valid_idx], dtype=float)
        targets_clean = np.array([targets[i] for i in valid_idx], dtype=float)
        if uid_arr is not None:
            uids_clean = np.array([uids[i] for i in valid_idx])

        # Fill invalid parse rewards with the worst valid score per uid (or global)
        base_scores = self._apply_invalid_reward_penalty(
            base_scores, indices_per_uid, uid_arr, invalid_mask
        )

        if uid_arr is None: # validation
            metrics = calculate_metrics(preds_clean, targets_clean)
        else: # training
            metrics = compute_training_metrics(preds_clean, targets_clean, uids_clean)
        
        reward_extra_info.update(metrics)

        if uid_arr is None:
            for i in range(len(data)):
                reward_tensor[i, last_token_pos[i]] = self._clip_to_float32(base_scores[i])
            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor
        
        # -------- CRPS + leave-one-out credit assignment --------
        shaped_rewards = list(base_scores)

        # If only one rollout in group, just keep base score
        for uid, idxs in indices_per_uid.items():
            valid_idxs = [i for i in idxs if preds[i] is not None]
            if len(valid_idxs) <= 1:
                continue

            # Collect predictions and target for this group
            group_preds = [preds[i] for i in valid_idxs]
            group_targets = [targets[i] for i in valid_idxs]

            # inconsistent labels within group -> fallback
            y_val = group_targets[0]
            if any(abs(t - y_val) > 1e-8 for t in group_targets[1:]): 
                continue

            # Convert to tensors
            p = torch.tensor(group_preds, dtype=torch.float32)  # (K,)
            y = torch.tensor(float(y_val), dtype=torch.float32)

            K = p.shape[0]

            # |p_k - y|
            diff_y = torch.abs(p - y)  # (K,)
            term1_all = diff_y.mean()

            # pairwise |p_i - p_j|
            diff_mat = torch.abs(p.unsqueeze(0) - p.unsqueeze(1))  # (K, K)
            D_sum_all = diff_mat.sum()
            term2_all = D_sum_all / (2.0 * (K ** 2))

            crps_all = term1_all - term2_all  # scalar

            row_sums = diff_mat.sum(dim=1)  # (K,)

            # Leave-one-out CRPS for each k,
            # reward r_k = CRPS(F_{-k}, y) - CRPS(F, y)
            for local_idx, i in enumerate(valid_idxs):
                K_minus = K - 1
                if K_minus <= 0:
                    continue

                # term1 without k
                term1_minus = (diff_y.sum() - diff_y[local_idx]) / K_minus

                # sum of pairwise distances without row/col k
                D_sum_minus = D_sum_all - 2.0 * row_sums[local_idx]
                term2_minus = D_sum_minus / (2.0 * (K_minus ** 2))
                crps_minus = term1_minus - term2_minus

                # leave-one-out marginal credit
                r_k = (crps_minus - crps_all).item()
                shaped_rewards[i] = r_k

        shaped_rewards = self._apply_invalid_reward_penalty(
            shaped_rewards, indices_per_uid, uid_arr, invalid_mask
        )

        for i in range(len(data)):
            reward_tensor[i, last_token_pos[i]] = self._clip_to_float32(shaped_rewards[i])
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _apply_invalid_reward_penalty(self, scores, indices_per_uid, uid_arr, invalid_mask):
        """Push invalid rollouts below the worst valid reward within each uid (or globally)."""
        adjusted = list(scores)
        valid_scores = [score for score, invalid in zip(adjusted, invalid_mask) if not invalid]
        if valid_scores:
            global_min = min(valid_scores)
        else:
            global_min = 0.0  # neutral baseline if everything failed
        global_penalized = global_min - self.invalid_parse_penalty

        if uid_arr is not None:
            for uid, idxs in indices_per_uid.items():
                uid_valid = [adjusted[i] for i in idxs if not invalid_mask[i]]
                uid_min = min(uid_valid) if uid_valid else global_min
                uid_penalized = uid_min - self.invalid_parse_penalty
                for i in idxs:
                    if invalid_mask[i]:
                        adjusted[i] = uid_penalized
        else:
            for i in range(len(adjusted)):
                if invalid_mask[i]:
                    adjusted[i] = global_penalized

        return adjusted

    def _clip_to_float32(self, value: float) -> float:
        value = float(value)
        if math.isnan(value):
            return 0.0
        if value > self._float32_max:
            return self._float32_max
        if value < self._float32_min:
            return self._float32_min
        return value
