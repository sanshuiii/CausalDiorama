"""Task-owned adaptation of the official H800 smoke check for the actual H20."""
import json,time,pathlib
import av,torch,torchvision
from flash_attn_interface import flash_attn_func
def reserve():
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('exactly one visible GPU required')
    torch.cuda.set_device(0)
    marker=torch.zeros(256*1024*1024,dtype=torch.uint8,device='cuda')
    torch.cuda.synchronize();started=time.monotonic()
    while True:
        free,total=torch.cuda.mem_get_info()
        if free>=88*1024**3:break
        if time.monotonic()-started>120:raise RuntimeError('GPU did not become available within 120 seconds; existing work was not stopped')
        print(json.dumps({'stage':'waiting_for_gpu','free_gib':round(free/1024**3,2)}),flush=True);time.sleep(3)
    print(json.dumps({'stage':'gpu_available','free_gib':round(free/1024**3,2),'wait_seconds':round(time.monotonic()-started,2)}),flush=True)
    return marker
def verify():
    gpu=torch.cuda.get_device_name(0);cap=tuple(torch.cuda.get_device_capability(0))
    observed={'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_runtime':torch.version.cuda}
    assert observed=={'torch':'2.9.1+cu128','torchvision':'0.24.1+cu128','cuda_runtime':'12.8'},observed
    assert cap==(9,0) and 'H20' in gpu,(gpu,cap)
    gen=torch.Generator(device='cpu').manual_seed(20260826)
    q,k,v=[torch.randn((1,128,4,128),generator=gen,dtype=torch.bfloat16).cuda() for _ in range(3)]
    with torch.inference_mode():out=flash_attn_func(q,k,v,causal=False)
    torch.cuda.synchronize()
    assert out.shape==q.shape and torch.isfinite(out).all().item()
    # A small FP32 reference provides an independent numeric check.
    ref=torch.nn.functional.scaled_dot_product_attention(q.float().transpose(1,2),k.float().transpose(1,2),v.float().transpose(1,2)).transpose(1,2)
    max_error=(out.float()-ref).abs().max().item()
    assert max_error<0.02,max_error
    from musubi_tuner.wan.modules.attention import flash_attention
    with torch.inference_mode():varlen=flash_attention([q,k,v],attn_mode='flash3',split_attn=False,dtype=torch.bfloat16)
    torch.cuda.synchronize()
    varlen_error=(varlen.float()-ref).abs().max().item()
    assert varlen.shape==q.shape and torch.isfinite(varlen).all().item() and varlen_error<0.02
    av.codec.Codec('libx264','w')
    return {'status':'pass',**observed,'gpu':gpu,'capability':list(cap),'fa3_shape':list(out.shape),'max_abs_error_vs_fp32':max_error,'musubi_varlen_max_abs_error':varlen_error,'libx264':True}
if __name__=='__main__':
    marker=reserve();result=verify()
    pathlib.Path('@CD_NODE@/cwm_h3_reproduction_v1/logs/runtime_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
