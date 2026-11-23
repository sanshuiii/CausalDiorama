#!/usr/bin/env bash
# Grounding DINO + SAM 2.1 masks (masks_node.py) for the local first-frame edit on one GPU node, the job list split over the 8 GPUs
# (one process per GPU, as the runner's 'workers' tasks do).
#   bash masks_par.sh LIST LOGPREFIX          (LIST: one job JSON per line; shards LIST.shard<i>; logs LOGPREFIX.<i>.log)
LIST=$1; LOG=$2
PY=@CD_NODE@/cwm_h3_reproduction_v1/.venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
for m in grounding-dino-base sam2.1-hiera-large; do
  [ -d @CD_SHARED@/models/$m ] || { echo "missing model @CD_SHARED@/models/$m"; exit 2; }
done
mapfile -t L < <(grep -v '^[[:space:]]*$' "$LIST")
n=${#L[@]}; [ "$n" -eq 0 ] && { echo "empty list"; exit 0; }
k=$(( n < 8 ? n : 8 ))
pids=()
for i in $(seq 0 $((k - 1))); do
  sh="$LIST.shard$i"; : > "$sh"
  for j in $(seq "$i" "$k" $((n - 1))); do echo "${L[$j]}" >> "$sh"; done
  PYTHONPATH=@CD_NODE@/qwen_image21/pydeps CUDA_VISIBLE_DEVICES=$i LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat \
    "$PY" "$HERE/masks_node.py" "$sh" > "$LOG.$i.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "masks_par: $n jobs on $k GPUs, exit $rc"
exit $rc
