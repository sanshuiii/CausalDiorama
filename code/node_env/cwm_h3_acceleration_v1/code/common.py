import os
import time
from datetime import timedelta
import torch
import torch.distributed as dist


def setup():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    marker = torch.zeros(256*1024*1024, dtype=torch.uint8, device=device)
    started = time.monotonic()
    while torch.cuda.mem_get_info(device)[0] < 88*2**30:
        if time.monotonic()-started > 150:
            raise RuntimeError("GPU did not become available; existing workload was not modified")
        time.sleep(3)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
    del marker
    torch.cuda.empty_cache()
    return device


def sync():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def finish():
    if dist.is_initialized():
        dist.destroy_process_group()
