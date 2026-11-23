#!/usr/bin/env bash
# Runtime environment of every node task (sourced by runner.py before each command): the CWM environment's Python, CUDA
# libraries, offline model loading and thread counts.
CWM_ENV=@CD_NODE@/cwm_h3_reproduction_v1
ACCEL=@CD_NODE@/cwm_h3_acceleration_v1
export PY=$CWM_ENV/.venv/bin/python
export TORCHRUN=$CWM_ENV/.venv/bin/torchrun
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/compat:$CWM_ENV/cuda-12.8.1/lib64:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TRITON_CACHE_DIR="$ACCEL/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ACCEL/cache/inductor"
