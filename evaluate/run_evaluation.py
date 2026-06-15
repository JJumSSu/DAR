import os
import logging
import json
from typing import Optional

import numpy as np

from .arguments import get_args
from .utils import (
    load_kbss_evaluation_dataset,
    load_apps_evaluation_dataset,
    load_lipo_evaluation_dataset,
    load_esol_evaluation_dataset,
    load_freesolv_evaluation_dataset,
    parse_prediction,
    compute_metrics,
)


def main():
    args = get_args()

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO))
    logger = logging.getLogger(__name__)

    if args.task == 'kbss':
        eval_set = load_kbss_evaluation_dataset(args.root_dir)
    elif args.task == 'apps':
        eval_set = load_apps_evaluation_dataset(args.root_dir)
    elif args.task == 'lipo':
        eval_set = load_lipo_evaluation_dataset(args.root_dir)
    elif args.task == 'esol':
        eval_set = load_esol_evaluation_dataset(args.root_dir)
    elif args.task == 'freesolv':
        eval_set = load_freesolv_evaluation_dataset(args.root_dir)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    logger.info("Loaded %d evaluation examples for task: %s from %s", len(eval_set), args.task, args.root_dir)

    model_name = args.model
    logger.info("Loading vLLM model: %s", model_name)
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9, trust_remote_code=True, download_dir=args.cache_dir)
    tokenizer = llm.get_tokenizer()

    n_decode = max(1, int(getattr(args, "n_decode", 1)))
    temperature = getattr(args, "temperature", 1.0)
    top_p = getattr(args, "top_p", 1.0)

    def build_sampling_params(inference_seed: Optional[int]) -> SamplingParams:
        sp_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": getattr(args, "max_tokens", 16384),
            "n": n_decode,
        }
        if inference_seed is not None:
            sp_kwargs["seed"] = inference_seed
        return SamplingParams(**sp_kwargs)

    reasoning_mode = getattr(args, "reasoning", False)
    pretokenize = getattr(args, "pretokenize", False)
    batch_size = getattr(args, "batch_size", 8)
    eval_seeds = [1, 2, 3, 4, 5]

    logger.info(
        "Beginning batched inference on %d examples (batch_size=%d, pretokenize=%s, n_decode=%d, temperature=%s, top_p=%s)",
        len(eval_set), batch_size, pretokenize, n_decode, temperature, top_p
    )
    logger.info("Running evaluation across fixed seeds: %s", eval_seeds)

    def make_prompt_obj(system_prompt: str, input_prompt: str):
        if system_prompt is None:
            messages = [
                {"role": "user", "content": input_prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt},
            ]
        return tokenizer.apply_chat_template(messages, tokenize=pretokenize, add_generation_prompt=True)

    def run_single_seed(inference_seed: int) -> dict:
        predictions = []
        raw_predictions = []
        targets = []
        raw_outputs = []
        skipped = 0

        logger.info("Starting evaluation for seed=%d", inference_seed)

        for start in range(0, len(eval_set), batch_size):
            batch_examples = eval_set[start: start + batch_size]
            prompts = []
            targets_batch = []

            for ex in batch_examples:
                system_prompt = ex.get('system_prompt', None)
                input_prompt = ex['input_prompt']
                target = ex['response']
                try:
                    target_val = float(target)
                except Exception:
                    logger.info("Skipping example because target is not numeric: %s", target)
                    skipped += 1
                    continue

                prompt_obj = make_prompt_obj(system_prompt, input_prompt)
                prompts.append(prompt_obj)
                targets_batch.append(target_val)

            if len(prompts) == 0:
                continue

            sampling_params = build_sampling_params(inference_seed)

            try:
                results = llm.generate(prompts, sampling_params)
            except Exception as e:
                logger.error("vLLM generate error on batch starting at %d (seed=%d): %s", start, inference_seed, e)
                skipped += len(prompts)
                continue

            for res, tgt in zip(results, targets_batch):
                sample_predictions = []
                sample_outputs = []

                for out in (res.outputs or []):
                    full_output = (out.text or "").strip()

                    if reasoning_mode and "</think>" in full_output:
                        final_answer = full_output.split("</think>")[-1].strip()
                    else:
                        final_answer = full_output

                    if args.mode == 'rl' or args.mode == 'zero-shot':
                        parsed = parse_prediction(final_answer)
                    else:
                        parsed = final_answer

                    if parsed is None:
                        sample_outputs.append(final_answer)
                        continue

                    try:
                        sample_predictions.append(float(parsed))
                        sample_outputs.append(final_answer)
                    except Exception:
                        sample_outputs.append(final_answer)

                if len(sample_predictions) == 0:
                    logger.info("Could not parse any prediction from %d decode(s) for seed=%d", len(res.outputs or []), inference_seed)
                    skipped += 1
                    continue

                try:
                    # Keep existing behavior: average multiple decodes per prompt first.
                    pred_val = float(np.mean(sample_predictions))
                    target = float(tgt)
                except Exception:
                    logger.info("Could not convert prediction or target to float: %s, %s", sample_predictions, tgt)
                    skipped += 1
                    continue

                predictions.append(pred_val)
                raw_predictions.append(sample_predictions)
                targets.append(target)
                raw_outputs.append(sample_outputs)

        results_metrics = compute_metrics(predictions, targets)
        logger.info("Finished seed=%d. Successful=%d, Skipped=%d", inference_seed, len(predictions), skipped)

        return {
            "seed": inference_seed,
            "metrics": results_metrics,
            "predictions": predictions,
            "raw_predictions": raw_predictions,
            "targets": targets,
            "raw_outputs": raw_outputs,
            "skipped": skipped,
            "successful": len(predictions),
        }

    per_seed_results = []
    for seed in eval_seeds:
        per_seed_results.append(run_single_seed(seed))

    metric_names = sorted(per_seed_results[0]["metrics"].keys()) if per_seed_results else []
    metrics_mean_std = {}
    for metric_name in metric_names:
        vals = []
        for res in per_seed_results:
            val = res["metrics"].get(metric_name, np.nan)
            try:
                vals.append(float(val))
            except Exception:
                vals.append(np.nan)

        vals_np = np.asarray(vals, dtype=float)
        metrics_mean_std[metric_name] = {
            "mean": float(np.nanmean(vals_np)),
            "std": float(np.nanstd(vals_np)),
            "per_seed": vals,
        }

    logger.info("=" * 60)
    logger.info("Aggregated metrics across %d seeds", len(eval_seeds))
    for metric_name, stats in metrics_mean_std.items():
        logger.info(
            "%s: mean=%.6f std=%.6f per_seed=%s",
            metric_name.upper(),
            stats["mean"],
            stats["std"],
            stats["per_seed"],
        )
    logger.info("=" * 60)

    os.makedirs(args.pred_output_dir, exist_ok=True)
    out_dir = os.path.join(args.pred_output_dir, args.task)
    os.makedirs(out_dir, exist_ok=True)

    if model_name.count("/") > 2:
        global_step = model_name.split("/")[-2]
        model_clean = model_name.split("/")[-3]
        fpath = os.path.join(out_dir, f"{model_clean}-{global_step}.json")
    else:
        fpath = os.path.join(out_dir, f"{model_name}.json")
    
    with open(fpath, "w") as f:
        json.dump({
            "seeds": eval_seeds,
            "metrics_mean_std": metrics_mean_std,
            "per_seed_results": per_seed_results,
        }, f, indent=4)


if __name__ == "__main__":
    main()
