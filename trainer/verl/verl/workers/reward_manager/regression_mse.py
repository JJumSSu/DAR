from collections import defaultdict

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

from .utils import parse_prediction, calculate_metrics, compute_training_metrics


@register("regression_mse")
class RegressionMSERewardManager:
    """The reward manager."""

    PARSE_FAIL_SCORE = -5e10

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the RegressionMSERewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """Compute instance-wise negative MSE reward and calculate metrics for both training and validation phases.
        
        This reward manager is called during both "validation" and "training" phases.
        During validation, "uid" info is not available, whereas it is present during training (i.e., rollouts).
        """

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        uid_arr = data.non_tensor_batch.get("uid", None)
        preds = []
        targets = []
        uids = []

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem            
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            # scoring part
            parsed_result = parse_prediction(response_str)
            if isinstance(parsed_result, float):
                # Negative MSE: closer prediction yields value nearer to 0 (higher is better).
                squared_diff = np.square(parsed_result - ground_truth)
                score = -np.minimum(squared_diff, 1e10)  # cap at a large but safe value
                score = score.item()
                preds.append(parsed_result) # only append clean predictions
                targets.append(ground_truth)
                if uid_arr is not None:
                    uid = uid_arr[i]
                    uids.append(uid)
            else:
                score = self.PARSE_FAIL_SCORE

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

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
                    print("[score]", score)

        if uid_arr is None: # validation
            metrics = calculate_metrics(np.array(preds), np.array(targets))
            reward_extra_info.update(metrics)
        else: # training
            training_metrics = compute_training_metrics(np.array(preds), np.array(targets), np.array(uids))
            reward_extra_info.update(training_metrics)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
