#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"

# Clean out conflicting core packages first
pip uninstall -y torch torchvision torchaudio vllm tensordict torchdata ray grpcio grpcio-status protobuf pyarrow flash-attn flash_attn flashinfer flashinfer-python || true

# PyTorch 2.7.0 + CUDA 12.8
pip install --no-cache-dir \
  torch==2.7.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# vLLM 0.9.2
pip install --no-cache-dir "vllm==0.9.2"

# Keep sglang, but remove the old torch2.6-specific flashinfer wheel source
if [ $USE_SGLANG -eq 1 ]; then
    pip install --no-cache-dir "sglang[all]==0.4.6.post1"
    pip install --no-cache-dir torch-memory-saver
fi

# Keep tensordict / torchdata install step
pip install --no-cache-dir "tensordict==0.6.2" torchdata

echo "2. install basic packages"

# Ray stack pinned to the versions that worked in your environment
pip install --no-cache-dir \
  "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
  "numpy<2.0.0" "pyarrow==23.0.0" pandas \
  "ray[default]==2.53.0" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
  pytest py-spy pyext pre-commit ruff

pip install --no-cache-dir \
  "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" \
  "grpcio==1.78.0" "protobuf==4.25.8"

echo "3. install FlashAttention and FlashInfer"

# FlashAttention: source build against the torch already installed in this env
pip install --no-cache-dir packaging ninja wheel setuptools

wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
pip install --no-cache-dir flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# FlashInfer: package-based install
pip install --no-cache-dir "flashinfer-python==0.3.1"

if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2
fi

echo "5. May need to fix opencv"
pip install --no-cache-dir opencv-python
pip install --no-cache-dir opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    pip install --no-cache-dir nvidia-cudnn-cu12==9.8.0.87
fi

echo "7. verify key versions"
python - <<'PY'
import torch, ray, grpc, pyarrow, google.protobuf
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("ray:", ray.__version__)
print("grpcio:", grpc.__version__)
print("pyarrow:", pyarrow.__version__)
print("protobuf:", google.protobuf.__version__)
try:
    import flash_attn
    print("flash_attn: ok")
except Exception as e:
    print("flash_attn import error:", e)
try:
    import flashinfer
    print("flashinfer: ok")
except Exception as e:
    print("flashinfer import error:", e)
PY

echo "Successfully installed all packages"