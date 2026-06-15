import argparse
import os
import re
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset


def load_kbss_evaluation_dataset(root_dir: str):
    kbss_test_set_dir = os.path.join(root_dir, "KBSS_rl_test.parquet")
    ds = pd.read_parquet(kbss_test_set_dir)
    
    data = []
    for i in range(len(ds)):
        system_prompt = ds.iloc[i]['prompt'][0]['content']  # extract user content
        input_prompt = ds.iloc[i]['prompt'][1]['content']
        ground_truth = ds.iloc[i]['extra_info']['response']    
        data.append({'system_prompt': system_prompt, 'input_prompt': input_prompt, 'response': ground_truth})
    
    return data

def load_apps_evaluation_dataset(root_dir: str):
    apps_test_set_dir = os.path.join(root_dir, "APPS_rl_test.parquet")
    ds = pd.read_parquet(apps_test_set_dir)
    
    data = []
    for i in range(len(ds)):
        system_prompt = ds.iloc[i]['prompt'][0]['content']  # extract user content
        input_prompt = ds.iloc[i]['prompt'][1]['content']
        ground_truth = ds.iloc[i]['extra_info']['response']    
        data.append({'system_prompt': system_prompt, 'input_prompt': input_prompt, 'response': ground_truth})
    
    return data

def load_lipo_evaluation_dataset(root_dir: str):
    lipo_test_set_dir = os.path.join(root_dir, "LIPO_rl_test.parquet")
    ds = pd.read_parquet(lipo_test_set_dir)
    
    data = []
    for i in range(len(ds)):
        system_prompt = ds.iloc[i]['prompt'][0]['content']  # extract user content
        input_prompt = ds.iloc[i]['prompt'][1]['content']
        ground_truth = ds.iloc[i]['extra_info']['response']    
        data.append({'system_prompt': system_prompt, 'input_prompt': input_prompt, 'response': ground_truth})
    
    return data

def load_esol_evaluation_dataset(root_dir: str):
    esol_test_set_dir = os.path.join(root_dir, "ESOL_rl_test.parquet")
    ds = pd.read_parquet(esol_test_set_dir)
    
    data = []
    for i in range(len(ds)):
        system_prompt = ds.iloc[i]['prompt'][0]['content']  # extract user content
        input_prompt = ds.iloc[i]['prompt'][1]['content']
        ground_truth = ds.iloc[i]['extra_info']['response']    
        data.append({'system_prompt': system_prompt, 'input_prompt': input_prompt, 'response': ground_truth})
    
    return data

def load_freesolv_evaluation_dataset(root_dir: str):
    freesolv_test_set_dir = os.path.join(root_dir, "FREESOLV_rl_test.parquet")
    ds = pd.read_parquet(freesolv_test_set_dir)
    
    data = []
    for i in range(len(ds)):
        system_prompt = ds.iloc[i]['prompt'][0]['content']  # extract user content
        input_prompt = ds.iloc[i]['prompt'][1]['content']
        ground_truth = ds.iloc[i]['extra_info']['response']    
        data.append({'system_prompt': system_prompt, 'input_prompt': input_prompt, 'response': ground_truth})
    
    return data

def parse_prediction(text: str) -> Optional[float]:
    """Parse the first numeric value. Prefer the first value inside \boxed{...}; if no \boxed{} present, return None."""
    if not text:
        return None

    matches = re.findall(r"\\boxed\s*\{([^}]*)\}", text)
    if matches:
        candidate = matches[-1].strip()
    else:
        return None
        
    # strip common noise
    candidate = candidate.replace(",", "").replace("%", "").strip()

    # match integer/float with optional exponent
    num_re = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    mnum = re.search(num_re, candidate)
    if mnum:
        try:
            return float(mnum.group(1))
        except Exception:
            return None
    return None

def compute_metrics(preds: List[float], targets: List[float]) -> dict:
    preds_arr = np.asarray(preds, dtype=float)
    targets_arr = np.asarray(targets, dtype=float)
    mask = np.isfinite(preds_arr) & np.isfinite(targets_arr)
    preds_clean = preds_arr[mask]
    targets_clean = targets_arr[mask]

    print(f"Valid predictions: {preds_clean.size} / {preds_arr.size}")

    if preds_clean.size == 0:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
            "kendall_tau": float("nan"),
        }

    mae = float(np.mean(np.abs(preds_clean - targets_clean)))
    mse = float(np.mean((preds_clean - targets_clean) ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((targets_clean - preds_clean) ** 2))
    ss_tot = float(np.sum((targets_clean - np.mean(targets_clean)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    if preds_clean.size < 2:
        spearman = float("nan")
        kendall_tau = float("nan")
    else:
        # Spearman correlation = Pearson correlation of ranked values
        s_pred = pd.Series(preds_clean)
        s_tgt = pd.Series(targets_clean)
        pred_rank = s_pred.rank(method="average").to_numpy()
        tgt_rank = s_tgt.rank(method="average").to_numpy()

        with np.errstate(invalid="ignore"):
            corrmat = np.corrcoef(pred_rank, tgt_rank)
            spearman = float(corrmat[0, 1]) if corrmat.size >= 4 else float("nan")

        # Kendall tau-b
        n = preds_clean.size
        concordant = 0
        discordant = 0
        ties_pred = 0
        ties_tgt = 0

        for i in range(n - 1):
            for j in range(i + 1, n):
                dp = np.sign(preds_clean[j] - preds_clean[i])
                dt = np.sign(targets_clean[j] - targets_clean[i])

                if dp == 0 and dt == 0:
                    ties_pred += 1
                    ties_tgt += 1
                elif dp == 0:
                    ties_pred += 1
                elif dt == 0:
                    ties_tgt += 1
                elif dp == dt:
                    concordant += 1
                else:
                    discordant += 1

        denom = np.sqrt(
            (concordant + discordant + ties_pred) *
            (concordant + discordant + ties_tgt)
        )
        kendall_tau = float((concordant - discordant) / denom) if denom > 0 else float("nan")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
        "kendall_tau": kendall_tau,
    }

if __name__ == "__main__":
    text = [
        "To predict the kernel execution latency for the provided `triton_per_fused_add_div_mean_pow_sqrt_sub_0` kernel, we need to consider the following factors:\n\n1. **Kernel Parameters**:\n   - `in_ptr0`: Pointer to the input data.\n   - `out_ptr2`: Pointer to the output data.\n   - `xnumel`: Number of elements in the input data.\n   - `rnumel`: Number of elements in the reduction data.\n   - `XBLOCK`: Block size for the input data.\n   - `RBLOCK`: Block size for the reduction data.\n\n2. **Data Types**:\n   - Input data: `torch.float32`.\n   - Output data: `torch.float32`.\n\n3. **Thread Block and Grid Size**:\n   - `XBLOCK`: 16.\n   - `RBLOCK`: 16.\n   - `num_warps`: 2.\n   - `num_stages`: 1.\n\n4. **Memory Access**:\n   - The kernel loads and stores data using `tl.load` and `tl.store`.\n\n5. **Memory Bandwidth**:\n   - Assuming a memory bandwidth of 256 GB/s (this is a rough estimate).\n\nGiven these factors, we can estimate the latency using the formula:\n\\[ \\text{Latency (ms)} = \\frac{\\text{Bandwidth (GB/s)} \\times \\text{Data Size (GB)} \\times \\text{Bytes per Byte (1 Byte)} \\times \\text{Threads per Block} \\times \\text{Blocks per Grid} \\times \\text{Blocks per Stage} \\times \\text{Blocks per Warp} \\times \\text{Warp Size (Bytes)} \\times \\text{Instructions per Cycle} \\times \\text{Cycles per Cycle}}{\\text{Cycles per Clock}} \\]\n\n### Steps to Estimate Latency\n\n1. **Bandwidth Calculation**:\n   - Bandwidth = 256 GB/s.\n   - Data Size = 4 bytes (since `torch.float32` is 4 bytes).\n   - Bytes per Byte = 1.\n   - Threads per Block = 16.\n   - Blocks per Grid = 16.\n   - Blocks per Stage = 1.\n   - Blocks per Warp = 2.\n   - Warp Size = 32 bytes (since each warp has 32 threads).\n\n2. **Instructions per Cycle**:\n   - Assuming a 32-bit integer instruction per cycle.\n\n3. **Cycles per Cycle**:\n   - Assuming a 1 cycle per clock.\n\n### Calculation\n\n\\[ \\text{Latency (ms)} = \\frac{256 \\times 4 \\times 1 \\times 16 \\times 16 \\times 1 \\times 2 \\times 32 \\times 4 \\times 1}{1} \\]\n\n\\[ \\text{Latency (ms)} = \\frac{256 \\times 4 \\times 16 \\times 16 \\times 2 \\times 32 \\times 4}{1} \\]\n\n\\[ \\text{Latency (ms)} = 122880 \\]\n\n### Final Estimate\n\nThe estimated latency for the kernel execution is approximately **122.88 ms**.",
        "To predict the kernel execution latency for the provided `call` function, we need to consider the following factors:\n\n1. **Function Execution Time**: The function `call` is a CUDA kernel that performs computations on tensors.\n2. **Tensor Dimensions**: The tensors used in the function have specific dimensions.\n3. **CUDA Device and Stream**: The function is executed on a specific CUDA device with a specific stream.\n\nGiven these factors, we can estimate the latency by considering the execution time of the CUDA kernel and the overhead of the CUDA runtime.\n\n### Estimation Steps:\n\n1. **CUDA Kernel Execution Time**: The kernel `triton_poi_fused_avg_pool2d_0` is a simple averaging operation over a 4x4x4x4 tensor. The time taken for this operation can be estimated based on the size of the tensor and the number of elements processed.\n\n2. **Tensor Operations**: The function also performs operations on a 32x32x32x32 tensor using `triton_poi_fused_cat_1`.\n\n3. **CUDA Runtime Overhead**: The overhead of the CUDA runtime includes synchronization, memory transfers, and other runtime operations.\n\n### Estimation:\n\n1. **Kernel Execution Time**:\n   - The kernel `triton_poi_fused_avg_pool2d_0` processes 256 elements in a block of 128 elements. Assuming a single thread processes one element in a block, the time taken to process 256 elements can be estimated based on the number of threads and the number of blocks.\n   - For simplicity, let's assume the kernel takes 100 clock cycles per element, and there are 32 threads per block (4x8 grid).\n\n   \\[\n   \\text{Time per element} = \\frac{100 \\text{ cycles}}{256 \\text{ elements}} = 0.3906 \\text{ cycles}\n   \\]\n\n   \\[\n   \\text{Total time for 256 elements} = 256 \\times 0.3906 \\text{ cycles} = 99.1856 \\text{ cycles}\n   \\]\n\n2. **Tensor Operations**:\n   - The tensor `buf3` is processed in chunks of 128 elements using `triton_poi_fused_cat_1`. Assuming the time taken for each chunk is similar to the kernel execution time, we can estimate the time for 128 elements.\n\n   \\[\n   \\text{Time for 128 elements} = 99.1856 \\text{ cycles}\n   \\]\n\n3. **Total Execution Time**:\n   - The total execution time is the sum of the kernel execution time and the overhead of the CUDA runtime.\n\n   \\[\n   \\text{Total execution time} = 99.1856 \\text{ cycles} + 99.1856 \\text{ cycles} = 198.3712 \\text{ cycles}\n   \\]\n\n4. **Conversion to Latency**:\n   - Assuming a clock cycle time of 10 nanoseconds, the latency in nanoseconds is:\n\n   \\[\n   \\text{Latency (ns)} = \\frac{198.3712 \\text{ cycles}}{10 \\text{ cycles/ns}} = 19.83712 \\text{ ns}\n   \\]\n\n   Converting to milliseconds:\n\n   \\[\n   \\text{Latency (ms)} = \\frac{19.83712 \\text{ ns}}{1000 \\text{ ns/ms}} = 0.01983712 \\text{ ms}\n   \\]\n\n### Final Estimate:\n\n\\[\n\\boxed{0.0198}\n\\]",
        "To predict the kernel execution latency for the provided `AttentionNew` module, we need to analyze the code and understand the key components and operations involved. The module performs a weighted average over the second axis (axis=1) of a tensor. Given the complexity of the operations and the use of CUDA kernels, we can estimate the latency based on the provided code.\n\nHere's a step-by-step breakdown of the operations and their potential latencies:\n\n1. **Input Preprocessing**:\n   - The input tensor `input_0` is of size `[N, W, H]` and is padded to have the same size for each mini-batch. This involves some memory operations and data movement.\n\n2. **Attention Calculation**:\n   - The attention mechanism involves computing a weighted average over the second axis. This is typically done using matrix operations, which can be quite intensive.\n\n3. **Matrix Multiplication**:\n   - The operation `extern_kernels.bmm(reinterpret_tensor(buf4, (4, 1, 4), (4, 4, 1), 0), primals_1, out=buf5)` involves matrix multiplication. The `bmm` kernel is used for this operation, which can be relatively fast.\n\n4. **Softmax and Tanh**:\n   - The softmax and tanh operations are used to compute the attention weights and apply them to the input tensor, respectively. These operations are relatively fast once the matrix operations are done.\n\n5. **Output Calculation**:\n   - The output tensor is computed as a weighted sum of the input tensor, which involves element-wise operations and possibly some memory operations.\n\nGiven the complexity of the operations and the use of CUDA kernels, we can estimate the latency as follows:\n\n- **Input Preprocessing**: This involves some memory operations and data movement, which can be relatively slow but not the bottleneck.\n- **Matrix Multiplication**: The `extern_kernels.bmm` kernel is used for matrix multiplication, which is relatively fast.\n- **Softmax and Tanh**: These operations are relatively fast once the matrix operations are done.\n- **Output Calculation**: This involves element-wise operations and possibly some memory operations, which can be relatively slow but not the bottleneck.\n\nConsidering these factors, we can estimate the latency to be in the range of 1-2 milliseconds. Given the complexity of the operations and the use of CUDA kernels, we can estimate the latency to be around 1-2 milliseconds.\n\n\\[\n\\boxed{1-2}\n\\]",
        "To predict the kernel execution latency for the given `triton_poi_fused_mul_sigmoid_0` kernel, we need to consider the following:\n\n1. **Kernel Parameters**:\n   - `XBLOCK`: The block size, which is 128.\n   - `num_warps`: Number of warps, which is 4.\n   - `num_stages`: Number of stages, which is 1.\n\n2. **Memory Access**:\n   - The kernel loads 256 elements from `in_ptr0` and stores 256 elements in `out_ptr0`.\n\n3. **Instruction Count**:\n   - Each thread loads 1 element from `in_ptr0` and stores 1 element in `out_ptr0`.\n   - The kernel performs a single load and store operation per thread.\n\n4. **Latency Calculation**:\n   - Each thread takes 1 clock cycle to perform a load and store operation.\n   - The kernel is launched with a grid size of 256, which means there are 256 threads.\n   - The total number of operations per thread is 2 (load and store).\n\nGiven these parameters, we can estimate the latency as follows:\n\n1. **Total Operations per Thread**:\n   - 2 operations per thread.\n\n2. **Total Operations per Grid**:\n   - 256 threads * 2 operations per thread = 512 operations.\n\n3. **Latency Calculation**:\n   - Each operation takes 1 clock cycle.\n   - Therefore, the total latency is 512 clock cycles.\n\nSince the latency is in milliseconds, we convert the latency from clock cycles to milliseconds:\n\n\\[ \\text{Latency (ms)} = \\frac{512 \\text{ clock cycles}}{1000000 \\text{ clock cycles per second}} \\approx 0.000512 \\text{ ms} \\]\n\nThus, the estimated latency for the kernel execution is:\n\n\\[\\boxed{0.000512}\\]",
        "To predict the kernel execution latency for the given `triton_per_fused_add_div_mean_mul_std_sub_0` kernel, we need to consider the following factors:\n\n1. **Kernel Parameters**:\n   - `in_out_ptr0`: Pointer to the output buffer.\n   - `in_out_ptr1`: Pointer to the input buffer.\n   - `in_ptr0`: Pointer to the first input buffer.\n   - `in_ptr1`: Pointer to the second input buffer.\n   - `in_ptr2`: Pointer to the third input buffer.\n   - `out_ptr0`: Pointer to the output buffer.\n   - `xnumel`: Number of elements in the first input buffer.\n   - `rnumel`: Number of elements in the second input buffer.\n   - `XBLOCK`: Block size for the first input buffer.\n   - `RBLOCK`: Block size for the second input buffer.\n\n2. **Kernel Function**:\n   - The kernel performs element-wise operations and stores results in the output buffer.\n\n3. **Execution Context**:\n   - The kernel is executed on a CUDA device with a specific stream.\n\nGiven the structure and parameters of the kernel, we can estimate the latency by considering the number of operations and the memory access patterns.\n\n### Estimation Steps:\n\n1. **Element-wise Operations**:\n   - The kernel performs operations on 4 elements in the first input buffer (`in_ptr0`) and 64 elements in the second input buffer (`in_ptr1`).\n   - The operations include additions, subtractions, and divisions.\n\n2. **Memory Access**:\n   - The kernel accesses elements in the input buffers (`in_ptr0`, `in_ptr1`, `in_ptr2`) and stores results in the output buffer (`in_out_ptr0`).\n   - The memory access pattern is sequential.\n\n3. **Block Size**:\n   - The kernel uses a block size of `XBLOCK` for the first input buffer and `RBLOCK` for the second input buffer.\n\n### Latency Estimation:\n\nGiven the number of operations and the block size, we can estimate the latency as follows:\n\n- **Latency per Operation**:\n  - Each operation (addition, subtraction, division) typically takes a few cycles.\n  - For simplicity, let's assume each operation takes 10 cycles.\n\n- **Total Operations**:\n  - For the first input buffer (`in_ptr0`), there are 4 elements.\n  - For the second input buffer (`in_ptr1`), there are 64 elements.\n  - Total operations = 4 + 64 = 68 operations.\n\n- **Total Latency**:\n  - Total latency = 68 operations * 10 cycles/op = 680 cycles.\n\n- **Convert to Milliseconds**:\n  - 1 cycle = 100 nanoseconds = 0.001 milliseconds.\n  - Total latency = 680 cycles * 0.001 ms/cycle = 0.68 milliseconds.\n\n### Final Estimate:\n\nThe estimated kernel execution latency is **0.68 milliseconds**.",
        "To predict the kernel execution latency for the given `qdNew` module, we need to analyze the execution of the `triton_poi_fused_relu_0` kernel. This kernel performs a series of operations including loading, storing, and arithmetic operations on tensors.\n\n### Analysis of the Kernel Execution\n\n1. **Tensor Operations**:\n   - The kernel loads a tensor `in_ptr0` of size `XBLOCK` (128 elements) and stores it in `out_ptr0`.\n   - It then computes the maximum of the loaded tensor elements and stores the result in the same location.\n\n2. **Memory Access Patterns**:\n   - The tensor `in_ptr0` is loaded from memory, and the tensor `out_ptr0` is written back to memory.\n   - The kernel is designed to handle small tensors (128 elements), which suggests a small memory footprint.\n\n3. **Execution Time**:\n   - The kernel is designed to execute in a single block of threads.\n   - The number of threads in a block is `XBLOCK`, which is 128.\n\n### Estimating Latency\n\nTo estimate the latency, we need to consider the number of operations per thread and the number of threads per block. Since the kernel is designed to execute in a single block, we can assume that the latency is determined by the number of operations per thread.\n\nGiven that each thread performs a single load and store operation, and since the tensor is small (128 elements), the latency can be estimated as follows:\n\n- **Number of Operations per Thread**: 2 (load and store).\n- **Number of Threads per Block**: 128.\n\n### Calculation\n\nThe total number of operations per block is:\n\\[ \\text{Total Operations} = \\text{Number of Operations per Thread} \\times \\text{Number of Threads per Block} = 2 \\times 128 = 256. \\]\n\nSince the latency is determined by the number of operations per block, the latency can be estimated as:\n\\[ \\text{Latency} = \\frac{\\text{Total Operations}}{\\text{Number of Threads per Block}} = \\frac{256}{128} = 2 \\text{ operations per block}. \\]\n\nGiven that each operation takes a negligible amount of time (assuming a small kernel), the total latency can be estimated as:\n\\[ \\text{Latency} \\approx 2 \\text{ operations} \\approx 2 \\text{ cycles}. \\]\n\n### Final Estimate\n\nSince each operation is a small computation and assuming the latency is proportional to the number of operations, the estimated latency is approximately 2 cycles.\n\nThus, the estimated kernel execution latency is \\(\\boxed{2}\\) milliseconds.",
        "To predict the kernel execution latency for the provided code, we need to analyze the execution of the `call` function, which is the main function that calls the Triton JIT kernel functions.\n\nHere's a step-by-step breakdown of the execution:\n\n1. **Loading Data**:\n   - The `call` function loads data from tensors `primals_1`, `primals_2`, `primals_3`, and `primals_4`.\n   - `primals_1` and `primals_2` are loaded into buffers of shape `(8, 4)` and `(8, 4, 4, 4)`, respectively.\n   - `primals_3` and `primals_4` are loaded into buffers of shape `(64, 4, 4, 4)` and `(64, 4)`, respectively.\n\n2. **Kernel Execution**:\n   - The `triton_poi_fused_mul_0` kernel is executed with the parameters:\n     - `in_ptr0`: Buffer containing `primals_2`.\n     - `out_ptr0`: Buffer to store the result.\n     - `xnumel`: Number of elements in the buffer.\n     - `XBLOCK`: Block size for the kernel.\n   - The `triton_poi_fused_add_mul_1` kernel is executed with the parameters:\n     - `in_ptr0`: Buffer containing `primals_4`.\n     - `in_ptr1`: Buffer containing `primals_1`.\n     - `in_ptr2`: Buffer containing `primals_3`.\n     - `out_ptr0`: Buffer to store the result.\n     - `xnumel`: Number of elements in the buffer.\n     - `XBLOCK`: Block size for the kernel.\n   - The `triton_poi_fused_leaky_relu_backward_2` kernel is executed with the parameters:\n     - `in_ptr0`: Buffer containing `primals_1`.\n     - `in_ptr1`: Buffer containing `primals_2`.\n     - `out_ptr0`: Buffer to store the result.\n     - `xnumel`: Number of elements in the buffer.\n     - `XBLOCK`: Block size for the kernel.\n\n3. **Latency Estimation**:\n   - The latency of each kernel execution depends on the block size (`XBLOCK`) and the number of elements (`xnumel`).\n   - The number of elements in the buffers is `xnumel = 512` for `triton_poi_fused_mul_0`, `64` for `triton_poi_fused_add_mul_1`, and `4` for `triton_poi_fused_leaky_relu_backward_2`.\n\nGiven that the block size `XBLOCK` is `32` for `triton_poi_fused_mul_0`, `256` for `triton_poi_fused_add_mul_1`, and `256` for `triton_poi_fused_leaky_relu_backward_2`, we can estimate the latency as follows:\n\n- For `triton_poi_fused_mul_0`: \n  - Number of elements per block = 32\n  - Number of blocks per grid = 32\n  - Latency = Number of elements per block / Block size = 32 / 32 = 1 ms\n\n- For `triton_poi_fused_add_mul_1`: \n  - Number of elements per block = 64\n  - Number of blocks per grid = 4096\n  - Latency = Number of elements per block / Block size = 64 / 256 = 0.25 ms\n\n- For `triton_poi_fused_leaky_relu_backward_2`: \n  - Number of elements per block = 4\n  - Number of blocks per grid = 512\n  - Latency = Number of elements per block / Block size = 4 / 256 = 0.16 ms\n\nGiven that the number of elements in the buffers is relatively small compared to the block size, the latency for `triton_poi_fused_mul_0` is likely to be the highest due to the larger block size.\n\n**Latency Estimate**:\n- `triton_poi_fused_mul_0`: 1 ms\n- `triton_poi_fused_add_mul_1`: 0.25 ms\n- `triton_poi_fused_leaky_relu_backward_2`: 0.16 ms\n\nThe overall latency for the `call` function is dominated by the `triton_poi_fused_mul_0` kernel, so the estimated latency for the kernel execution is **1 ms**.",
        "To predict the kernel execution latency for the provided `call` function, we need to analyze the function's structure and identify the operations that are computationally intensive. Given the complexity of the operations, we will use the `triton` library's heuristic to estimate the latency.\n\nHere's a step-by-step approach to estimate the latency:\n\n1. **Identify the main operations**: The main operations in the `call` function are the forward passes through the custom kernels (`triton_per_fused_binary_cross_entropy_mean_0`, `triton_per_fused_binary_cross_entropy_mean_1`, `triton_per_fused_binary_cross_entropy_mean_2`, `triton_per_fused_binary_cross_entropy_mean_3`, `triton_per_fused_add_binary_cross_entropy_div_mean_mul_4`). We will focus on these operations.\n\n2. **Estimate the kernel execution time**: We can use the `triton` library's heuristic to estimate the latency. The heuristic takes into account the number of threads (`num_warps` and `num_stages`), the block size (`XBLOCK`), and the number of elements processed (`rnumel`).\n\n3. **Use the heuristic**: The `triton` library provides a method to estimate the latency based on the provided parameters.\n\nHere is the estimated latency for the given function:\n\n```python\nimport triton\n\n# Estimated latency for the provided function\nlatency_ms = triton.heuristics().estimate(\n    num_warps=2,\n    num_stages=1,\n    XBLOCK=1,\n    rnumel=16,\n    num_elements=4,\n    reduction='mean',\n    avg_factor=1\n)\n\nprint(f\"Estimated latency in milliseconds: {latency_ms:.2f}\")\n```\n\nThe estimated latency in milliseconds is approximately **1.00**.\n\nThis estimate assumes that the function is executed on a single CUDA stream and that the number of elements processed (`rnumel`) is 16. The `num_warps` and `num_stages` are set to 2 and 1, respectively, which are typical values for the operations involved. The `XBLOCK` is set to 1, and the number of elements processed (`num_elements`) is 4. The `reduction` and `avg_factor` are set to 'mean' and 1, respectively, which are common settings for the operations.",
        "To predict the kernel execution latency for the provided `call` function, we need to consider the following factors:\n\n1. **Function Complexity**: The `call` function is a high-level wrapper that calls multiple sub-functions in parallel. Each sub-function has its own set of parameters and operations.\n\n2. **Tensor Operations**: The function performs tensor operations such as loading, storing, and applying convolutional and layer normalization operations.\n\n3. **Kernel Execution**: Each sub-function is executed by a different kernel, and the latency of each kernel depends on its implementation details, including the number of warps, stages, and data dependencies.\n\n4. **CUDA Stream Management**: The function uses CUDA streams for parallel execution, which introduces additional overhead.\n\nGiven the complexity of the function, we can make some rough estimates:\n\n1. **Sub-function Latencies**:\n   - `triton_poi_fused_native_layer_norm_0`: This function performs layer normalization, which is relatively simple and likely has a low latency.\n   - `triton_poi_fused_convolution_leaky_relu_native_layer_norm_1`: This function performs a convolution followed by layer normalization, which is more complex and likely has a higher latency.\n   - `triton_poi_fused_add_convolution_mul_sigmoid_6`: This function performs a series of operations, including convolution, sigmoid, and addition, which is also complex but likely has a moderate latency.\n\n2. **Kernel Execution Latency**:\n   - The `call` function likely launches multiple kernels in parallel, which will introduce additional overhead. However, the exact number of threads and blocks can be inferred from the function signature.\n\n3. **Estimation**:\n   - Let's assume the latency of each sub-function:\n     - `triton_poi_fused_native_layer_norm_0`: 100 \u00b5s\n     - `triton_poi_fused_convolution_leaky_relu_native_layer_norm_1`: 1000 \u00b5s\n     - `triton_poi_fused_add_convolution_mul_sigmoid_6`: 1000 \u00b5s\n\n   - The `call` function launches multiple kernels in parallel, and we can estimate the total latency by summing up the latencies of these sub-functions.\n\n   - Total latency = 100 \u00b5s + 1000 \u00b5s + 1000 \u00b5s = 2100 \u00b5s\n\n   - Convert to milliseconds: 2100 \u00b5s = 0.21 ms\n\nGiven these assumptions and estimates, the estimated kernel execution latency is:\n\n```plaintext\n\\boxed{0.21}\n```",
        "To predict the kernel execution latency for the given `QNetworkNew` module, we need to consider the operations and their associated memory access patterns. Here's a step-by-step approach to estimate the latency:\n\n1. **Identify Key Operations**: The key operations in the `QNetworkNew` module are:\n   - Linear transformations (`fc1`, `fc2`, `fc3`)\n   - Element-wise operations (`addmm`)\n\n2. **Estimate Latency for Each Operation**:\n   - **Linear Transformations**: These are typically faster than element-wise operations. For a 2D matrix multiplication, the latency is approximately \\(O(n^2)\\) where \\(n\\) is the number of elements in the matrix.\n   - **Element-wise Operations**: These operations are generally faster than matrix multiplications. For example, `addmm` typically has a latency of \\(O(n)\\) where \\(n\\) is the number of elements in the matrix.\n\n3. **Estimate Latency for Each Kernel**:\n   - **`triton_poi_fused_cat_0`**: This kernel performs a matrix multiplication followed by a simple addition. The latency is approximately \\(O(n^2)\\) where \\(n\\) is the number of elements.\n   - **`triton_poi_fused_relu_1`**: This kernel performs element-wise addition and maximum operations. The latency is approximately \\(O(n)\\) where \\(n\\) is the number of elements.\n\n4. **Estimate Total Latency**:\n   - The total latency is the sum of the latencies of the individual operations.\n\nLet's break down the operations and their latencies:\n\n### Key Operations and Latencies\n\n1. **Linear Transformations**:\n   - **`fc1`**: \\(O(8 \\times 256 \\times 256) = O(5120000)\\)\n   - **`fc2`**: \\(O(256 \\times 256) = O(65536)\\)\n   - **`fc3`**: \\(O(256 \\times 1) = O(256)\\)\n\n2. **Element-wise Operations**:\n   - **`addmm`**: \\(O(256 \\times 1 \\times 1) = O(256)\\)\n\n### Total Latency\n\n- **Latency for `triton_poi_fused_cat_0`**: \\(O(32^2) = O(1024)\\)\n- **Latency for `triton_poi_fused_relu_1`**: \\(O(1024) = O(1024)\\)\n\n### Total Latency\n\n- **Total Latency**: \\(O(1024) + O(1024) = O(2048)\\)\n\nGiven the operations and their latencies, the estimated kernel execution latency for the `QNetworkNew` module is approximately **2048 milliseconds**.\n\n\\(\\boxed{2048}\\)",
    ]

    for t in text:
        print(parse_prediction(t))
