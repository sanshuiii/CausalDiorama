"""Merge original weights once on CPU, then broadcast identical tensors."""
import hashlib
import json
import time
import torch
import torch.distributed as dist
from common import sync
from optimizations import rank,world


@torch.no_grad()
def load_broadcast(config,device):
    # The original LoRA merger updates parameters in worker threads. Ordinary
    # tensors are required because inference_mode is thread-local.
    from cwm_h3_inference.runtime import strict_merge_lora
    from musubi_tuner.minimax_h3.model import load_h3_transformer,MiniMaxH3Model
    details={}
    if rank()==0:
        t=time.monotonic()
        model=load_h3_transformer(config.models.dit,device='cpu',dtype=torch.bfloat16,
                                  attn_mode='flash3',split_attn=False,disable_mmap=False)
        details['load_cpu_seconds']=time.monotonic()-t;t=time.monotonic()
        gate=strict_merge_lora(model,config.models.lora,config.expected_lora_modules)
        details['merge_cpu_seconds']=time.monotonic()-t
        info=[model.config,gate,details]
    else:
        model=None;info=[None,None,None]
    if world()>1:dist.broadcast_object_list(info,src=0,device=device)
    model_config,gate,details=info
    t=time.monotonic()
    if rank()!=0:
        # Meta construction preserves every original FP32/BF16 parameter dtype
        # while avoiding CPU checkpoint copies and random initialization work.
        model=MiniMaxH3Model(model_config,attn_mode='flash3',split_attn=False,dtype=torch.bfloat16,device='meta')
        model.to_empty(device=device)
    else:model.to(device)
    sync();details['allocate_gpu_seconds']=time.monotonic()-t
    tensors=model.state_dict()
    signature=hashlib.sha256(json.dumps([(name,list(value.shape),str(value.dtype)) for name,value in tensors.items()]).encode()).hexdigest()
    if world()>1:
        signatures=[None]*world();dist.all_gather_object(signatures,signature)
        if len(set(signatures))!=1:raise RuntimeError('rank model schemas differ before weight broadcast')
    t=time.monotonic()
    if world()>1:
        for tensor in tensors.values():dist.broadcast(tensor,src=0)
    sync()
    details.update(broadcast_seconds=time.monotonic()-t,model_tensor_count=len(tensors),
                   model_tensor_bytes=sum(x.numel()*x.element_size() for x in tensors.values()),
                   model_schema_sha256=signature,merge_rank=0)
    model.eval().requires_grad_(False)
    return model,gate,details
