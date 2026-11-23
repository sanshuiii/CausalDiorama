"""Start-of-task GPU claim, run before every GPU task on a node (the `PRE` prefix of the node commands).

Holds 256 MB on every GPU for a few seconds and exits. On nodes where a background process takes idle GPUs and treats processes
that appear while it starts as its own, this short-lived process is the one it may adopt, and the job that starts afterwards runs
alone. Elsewhere it only costs the few seconds.

    python gpu_preclaim.py [SECONDS]      (default 15)
"""
import sys, time
import torch

n = torch.cuda.device_count()
bufs = [torch.empty(256 << 20, dtype=torch.uint8, device=f'cuda:{i}') for i in range(n)]
torch.cuda.synchronize()
time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 15)
print(f'gpu_preclaim: held 256 MB on {n} GPUs', flush=True)
