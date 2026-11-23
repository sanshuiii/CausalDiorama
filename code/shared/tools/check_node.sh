#!/usr/bin/env bash
# One GPU node's environment for the Causal Diorama pipelines (queued by bin/cd_init.py --check-node). Prints one line per check.
N=@CD_NODE@; S=@CD_SHARED@; M=$N/cwm_h3_reproduction_v1/release/models; PY=$N/cwm_h3_reproduction_v1/.venv/bin/python; fail=0
chk(){ if [ -e "$1" ]; then echo "ok      $1"; else echo "MISSING $1"; fail=1; fi; }
for f in $PY $M/diffusion_models/minimax_h3_ref2va_bf16.safetensors $M/lora/minimax-h3-cwm.safetensors \
         $M/vae/minimax_h3_video_vae_fp16.safetensors $M/vae/minimax_h3_audio_vae_fp32.safetensors \
         $M/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors $M/processors/Qwen3-VL-32B-Instruct \
         $N/cwm_h3_acceleration_v1/code/optimizations.py $N/cwm_h3_reproduction_v1/code/verify_h20_runtime.py \
         $N/compbench_v1/pydeps $N/qwen_image21/model $N/qwen_image21/pydeps $N/compbench_v1/tools/node_env.sh \
         $S/models/grounding-dino-base $S/models/sam2.1-hiera-large $S/models/Depth-Anything-V2-Metric-Outdoor-Small-hf $S/models/Depth-Anything-V2-Metric-Indoor-Small-hf; do chk $f; done
echo "0e14ad22ae203f684ba6521e0b6fbf58f64668ed454fbce346acafd7f448fdb2  $M/lora/minimax-h3-cwm.safetensors" | sha256sum -c --quiet && echo "ok      LoRA sha256" || { echo "BAD     LoRA sha256"; fail=1; }
$PY -c "import torch, cwm_h3_inference, musubi_tuner, cv2, transformers; print('ok      imports (CWM venv): torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())" || { echo "BAD     CWM venv imports"; fail=1; }
PYTHONPATH=$N/qwen_image21/pydeps $PY -c "import diffusers, transformers; print('ok      imports (Qwen-Image pydeps): diffusers', diffusers.__version__, 'transformers', transformers.__version__)" || { echo "BAD     Qwen-Image pydeps"; fail=1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -8
[ $fail = 0 ] && echo "NODE READY" || echo "NODE NOT READY"
