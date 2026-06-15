import argparse

parser = argparse.ArgumentParser(description="LLM Regression Analysis")
parser.add_argument("--task", type=str, default='kbss', choices=["kbss", "apps", "lipo", "esol", "freesolv"], help="Evaluation task/dataset")
parser.add_argument("--cache_dir", type=str, help="Path to the cache directory")
parser.add_argument("--loglevel", type=str, default='INFO', help="Logging level")
parser.add_argument("--pred_output_dir", type=str, help="Output directory")
parser.add_argument("--root-dir", type=str, default="/home/x-jpark38/DAR/data", help="Root directory containing evaluation parquet files")

parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Regression Model")
parser.add_argument("--mode", default="rl", choices=['rl', 'sft', 'zero-shot'])
parser.add_argument("--batch-size", type=int, default=8, help="Unused legacy argument kept for compatibility")
parser.add_argument("--max-tokens", type=int, default=4500, help="Max completion tokens for GPT Batch API generation")
parser.add_argument("--n-decode", type=int, default=1, help="Number of decoded samples per prompt; predictions are averaged over valid parses")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top-p")
parser.add_argument("--seed", type=int, default=42, help="Optional fixed seed; if omitted, a new seed is sampled for each inference batch")

def get_args():
	return parser.parse_args()
