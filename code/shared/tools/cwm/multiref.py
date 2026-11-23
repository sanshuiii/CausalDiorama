"""Isolated native Ref2VA extension; original CWM modules remain unchanged."""
from pathlib import Path
import hashlib,json,gc

def spec(config):
    p=config.cache_dir.parent/'references.json'
    d=json.loads(p.read_text())
    assert 1<=len(d['images'])<=9
    assert Path(d['images'][0]['path']).resolve()==config.anchor_image.resolve()
    assert len({x['path'] for x in d['images']})==len(d['images'])
    return d

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def contract(config):
    s=spec(config)
    return {'schema':'study-multiref-v1','images':[dict(x,sha256=digest(x['path'])) for x in s['images']],
      'system_prompt':s['system_prompt'],'caption':config.windows[0].prompt,
      'conditions':[[p.name,digest(p)] for p in sorted(config.condition_root.glob('*')) if p.is_file()],
      'models':{k:str(v) for k,v in vars(config.models).items()},'encoding_version':1}

def validate_cache(config):
    m=json.loads((config.cache_dir/'multiref_manifest.json').read_text())
    assert m['status']=='complete' and m['contract']==contract(config),'cache input contract changed'
    for p,h in m['files'].items():assert digest(config.cache_dir/p)==h,p
    return {'status':'pass','image_count':len(m['contract']['images']),'files':len(m['files'])}

def extra_pixels(path,short_edge=768):
    import numpy as np
    from PIL import Image
    im=Image.open(path).convert('RGB');w,h=im.size
    size=tuple(max(32,round(v*short_edge/min(w,h)/32)*32) for v in (w,h))
    return np.array(im.resize(size,Image.Resampling.LANCZOS))

def prepare(config):
    import numpy as np,torch
    from safetensors.torch import save_file
    from cwm_h3_inference.cache import _prepare_visual_caches,_load_anchor,cache_paths
    from cwm_h3_inference.duv import load_duv_window
    from cwm_h3_inference.constants import FPS,QWEN_OFFSETS
    from musubi_tuner.minimax_h3.video_vae import load_video_vae,encode_video_condition
    from musubi_tuner.minimax_h3_generate_video import _prepare_pixels
    from musubi_tuner.minimax_h3.media import H3AudioSource,H3Record,H3Reference
    from musubi_tuner.minimax_h3.text_encoder import H3TextVisual,H3Presentation,build_presentation,encode_h3_presentation,load_h3_processor,load_h3_text_encoder
    assert torch.cuda.device_count()==1 and len(config.windows)==1
    cfg=spec(config);c=contract(config);root=config.cache_dir;root.mkdir(parents=True,exist_ok=True)
    planned=root/'multiref_inputs.json'
    if planned.exists():assert json.loads(planned.read_text())==c
    else:planned.write_text(json.dumps(c,indent=2))
    if (root/'multiref_manifest.json').exists():return validate_cache(config)
    _prepare_visual_caches(config)
    device=torch.device('cuda:0')
    pending=[(i,x) for i,x in enumerate(cfg['images']) if i and not (root/f'image_{i:02d}.safetensors').exists()]
    if pending:
        vae=load_video_vae(config.models.video_vae,device=device,dtype=torch.float16,disable_mmap=False).eval().requires_grad_(False)
        for i,x in pending:
            pixels=extra_pixels(x['path'],x.get('short_edge',768))
            with torch.inference_mode():lat=encode_video_condition(vae,_prepare_pixels(torch.from_numpy(pixels).unsqueeze(0)).to(device,torch.float16))[0].float().cpu().contiguous()
            assert lat.shape[0:2]==(24,1) and torch.isfinite(lat).all()
            save_file({'latent':lat},str(root/f'image_{i:02d}.safetensors'),metadata={'picture':str(i+1),'entity':x['entity'],'source_sha256':digest(x['path'])})
            del pixels,lat;torch.cuda.empty_cache()
        del vae;gc.collect();torch.cuda.empty_cache()
    qpath=cache_paths(config,0)['qwen']
    if not qpath.exists():
        processor=load_h3_processor(str(config.models.processor))
        model=load_h3_text_encoder(str(config.models.text_encoder),processor_path=str(config.models.processor),device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
        visuals={};refs=[]
        for i,x in enumerate(cfg['images']):
            key=Path(x['path']);px=_load_anchor(key) if i==0 else extra_pixels(key,x.get('short_edge',768))
            visuals[key]=H3TextVisual(torch.from_numpy(np.ascontiguousarray(px)).unsqueeze(0));refs.append(H3Reference(type='image',path=key))
        vkey=config.condition_root/'000000.semantic_id.png';duv=load_duv_window(config.condition_root,0,for_qwen=True)
        visuals[vkey]=H3TextVisual(torch.from_numpy(np.ascontiguousarray(duv)),tuple(i/FPS for i in QWEN_OFFSETS))
        refs.append(H3Reference(type='video',path=vkey,duration_seconds=124/FPS))
        placeholder=root/'.presentation-only'
        record=H3Record(video_path=placeholder,caption=config.windows[0].prompt,target_audio=H3AudioSource(path=placeholder,embedded=False),references=tuple(refs),jsonl_line=0)
        plain=build_presentation(record,'ref2va',visuals)
        chat=processor.apply_chat_template([{'role':'system','content':cfg['system_prompt']},{'role':'user','content':plain.processor_text}],tokenize=False,add_generation_prompt=False)
        for i in range(len(cfg['images'])):assert f'<Picture {i+1}>' in chat
        assert len(plain.images)==len(cfg['images']) and len(plain.videos)==1
        presentation=H3Presentation(text=plain.text,images=plain.images,videos=plain.videos,processor_text=chat)
        with torch.inference_mode():hidden,tags=encode_h3_presentation(processor,model,presentation)
        assert hidden.ndim==2 and hidden.shape[1]==5120 and tags.shape==(hidden.shape[0],) and torch.isfinite(hidden).all()
        save_file({'varlen_mmh3_hidden_states_bfloat16':hidden.to('cpu',torch.bfloat16).contiguous(),'varlen_mmh3_token_tags_int64':tags.to('cpu',torch.int64).contiguous()},str(qpath))
        (root/'presentation.txt').write_text(chat)
        del model,processor,visuals,hidden,tags;gc.collect();torch.cuda.empty_cache()
    files={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*.safetensors'))}
    (root/'multiref_manifest.json').write_text(json.dumps({'status':'complete','contract':c,'files':files},indent=2))
    return validate_cache(config)

def build_inputs(config,device):
    import torch
    from safetensors.torch import load_file
    from cwm_h3_inference.engine import Ref2VAEngine
    from cwm_h3_inference.constants import VIDEO_LATENT_SHAPE,AUDIO_LATENT_SHAPE
    from musubi_tuner.minimax_h3.sampling import initialize_target_latents
    from musubi_tuner.minimax_h3.packing import H3ReferenceGeometry,H3VideoGeometry,build_h3_layout
    engine=object.__new__(Ref2VAEngine);engine.config=config;engine.device=device;engine.torch=torch
    anchor,duv,hidden,tags=engine._load_window_cache(0)
    images=[anchor]+[load_file(str(config.cache_dir/f'image_{i:02d}.safetensors'))['latent'].unsqueeze(0) for i in range(1,len(spec(config)['images']))]
    geometry=[H3ReferenceGeometry('image',video=H3VideoGeometry(*v.shape[2:])) for v in images]
    geometry.append(H3ReferenceGeometry('video',video=H3VideoGeometry(*duv.shape[2:])))
    layout=build_h3_layout(task='ref2va',text_length=hidden.shape[1],target_video=H3VideoGeometry(*VIDEO_LATENT_SHAPE[2:]),target_audio_frames=AUDIO_LATENT_SHAPE[-1],references=tuple(geometry))
    v,a=initialize_target_latents(video_shape=VIDEO_LATENT_SHAPE,audio_shape=AUDIO_LATENT_SHAPE,seed=config.seed,device=device,video_dtype=torch.float32,audio_dtype=torch.float32)
    values=images+[duv];offsets=[3000]+[4000+i for i in range(1,len(images))]+[1000];refs=[]
    for value,offset in zip(values,offsets):
        value=value.to(device,torch.float32);g=torch.Generator(device='cpu').manual_seed(config.seed+offset)
        noise=torch.randn(tuple(value.shape),generator=g,dtype=torch.float32).to(device)
        refs.append(.999*value+.001*noise)
    fixed,mask=engine._load_input_video()
    return dict(layout=layout,text_hidden_states=hidden.to(device,torch.bfloat16),text_token_tags=tags.to(device),initial_video=v,initial_audio=a,steps=config.steps,video_shift=12.,audio_shift=3.,visual_condition_latents=tuple(refs),audio_condition_latents=(),visual_condition_clean=.999,audio_condition_clean=1.,input_video_latents=fixed,video_denoise_mask=mask,input_audio_latents=None,audio_denoise_mask=None)
