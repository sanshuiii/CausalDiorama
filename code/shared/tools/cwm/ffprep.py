"""Single-GPU worker: condition caches for free-first-frame H3 Ref2VA jobs (stage 1).

  $PY ffprep.py LIST_FILE        (one job.json path per line; finished caches are kept)

A job has no RGB anchor and no fixed first frame. Its pictures are per-subject reference images (environment
first) and, optionally, the plan's depth+semantic proxy video for the first `frames` frames:
  images[i] = {key, path[, video]}   path is a PNG; if it is missing, frame 0 of `video` is written there first
  condition_root                     render/conditions of the plan, or null (image-only job, e.g. appearance edit)
  frames                             17n+5 (22 = one chunk, 124 = one full window)
  system_prompt                      chat system block, or null for the base model's plain presentation
Writes cache_dir/{image_XX,duv,qwen}.safetensors, presentation.txt and manifest.json (status complete + digests).
Same encoders and conventions as cwm_h3_inference.cache / multiref.prepare.
"""
import gc, hashlib, json, os, sys, time, traceback
from pathlib import Path

sys.path.insert(0, '@CD_NODE@/cwm_h3_reproduction_v1/code')
FPS = 24


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def latent_frames(frames):
    assert frames >= 5 and (frames - 5) % 17 == 0, frames
    return 5 * ((frames - 5) // 17) + 2


def extract_frame0(video, png):
    import av
    png.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video)) as c:
        img = next(c.decode(video=0)).to_image()
    tmp = png.with_name(f'{png.stem}.{os.getpid()}.tmp.png')  # workers on other GPUs may extract the same picture
    img.save(tmp)
    tmp.replace(png)


def pixels(path, short_edge=768):
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert('RGB'); w, h = im.size
    size = tuple(max(32, round(v * short_edge / min(w, h) / 32) * 32) for v in (w, h))
    return np.array(im.resize(size, Image.Resampling.LANCZOS))


def duv_frames(root, frames, for_qwen):
    import numpy as np
    from cwm_h3_inference.duv import load_duv_frame
    offsets = range(0, frames, 12) if for_qwen else range(frames)
    return np.stack([load_duv_frame(Path(root), o, for_qwen=for_qwen) for o in offsets])


def contract(job):
    keep = {k: job[k] for k in ('frames', 'condition_root', 'system_prompt', 'caption', 'models')}
    keep['images'] = [dict(x, sha256=digest(x['path'])) for x in job['images']]
    if job['condition_root']:
        root = Path(job['condition_root'])
        keep['conditions'] = [[p.name, digest(p)] for p in sorted(root.glob('*')) if p.is_file() and int(p.name[:6]) < job['frames']]
    return keep


def prepare(job, vae_box, qwen_box):
    import numpy as np, torch
    from safetensors.torch import save_file
    from musubi_tuner.minimax_h3.video_vae import load_video_vae, encode_video_condition
    from musubi_tuner.minimax_h3_generate_video import _prepare_pixels
    from musubi_tuner.minimax_h3.media import H3AudioSource, H3Record, H3Reference
    from musubi_tuner.minimax_h3.text_encoder import (H3TextVisual, H3Presentation, build_presentation, encode_h3_presentation,
                                                      load_h3_processor, load_h3_text_encoder)
    root = Path(job['cache_dir']); root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists() and json.loads((root / 'manifest.json').read_text()).get('status') == 'complete':
        return 'kept'
    for x in job['images']:
        if not Path(x['path']).exists():
            if not x.get('video'):
                raise FileNotFoundError(x['path'])
            extract_frame0(x['video'], Path(x['path']))
    c = contract(job)
    (root / 'inputs.json').write_text(json.dumps(c, indent=1, ensure_ascii=False))
    device = torch.device('cuda:0')
    if vae_box.get('vae') is None:
        vae_box['vae'] = load_video_vae(job['models']['video_vae'], device=device, dtype=torch.float16, disable_mmap=False).eval().requires_grad_(False)
    vae = vae_box['vae']
    for i, x in enumerate(job['images']):
        px = pixels(x['path'], x.get('short_edge', 768))
        with torch.inference_mode():
            lat = encode_video_condition(vae, _prepare_pixels(torch.from_numpy(px).unsqueeze(0)).to(device, torch.float16))[0].float().cpu().contiguous()
        assert lat.shape[0:2] == (24, 1) and torch.isfinite(lat).all()
        save_file({'latent': lat}, str(root / f'image_{i:02d}.safetensors'), metadata={'picture': str(i + 1), 'key': x['key']})
    if job['condition_root']:
        duv = duv_frames(job['condition_root'], job['frames'], False)
        with torch.inference_mode():
            lat = encode_video_condition(vae, _prepare_pixels(torch.from_numpy(duv.copy())).to(device, torch.float16))[0].float().cpu().contiguous()
        assert tuple(lat.shape) == (24, latent_frames(job['frames']), 12, 21), lat.shape
        save_file({'latent': lat}, str(root / 'duv.safetensors'), metadata={'frames': str(job['frames'])})
    torch.cuda.empty_cache()
    if qwen_box.get('model') is None:
        m = job['models']
        qwen_box['processor'] = load_h3_processor(m['processor'])
        qwen_box['model'] = load_h3_text_encoder(m['text_encoder'], processor_path=m['processor'], device=device,
                                                 dtype=torch.bfloat16).eval().requires_grad_(False)
    processor, model = qwen_box['processor'], qwen_box['model']
    visuals, refs = {}, []
    for x in job['images']:
        key = Path(x['path'])
        visuals[key] = H3TextVisual(torch.from_numpy(np.ascontiguousarray(pixels(key, x.get('short_edge', 768)))).unsqueeze(0))
        refs.append(H3Reference(type='image', path=key))
    if job['condition_root']:
        vkey = Path(job['condition_root']) / '000000.semantic_id.png'
        q = duv_frames(job['condition_root'], job['frames'], True)
        visuals[vkey] = H3TextVisual(torch.from_numpy(np.ascontiguousarray(q)), tuple(i / FPS for i in range(0, job['frames'], 12)))
        refs.append(H3Reference(type='video', path=vkey, duration_seconds=job['frames'] / FPS))
    placeholder = root / '.presentation-only'
    record = H3Record(video_path=placeholder, caption=job['caption'], target_audio=H3AudioSource(path=placeholder, embedded=False),
                      references=tuple(refs), jsonl_line=0)
    plain = build_presentation(record, 'ref2va', visuals)
    if job['system_prompt']:
        text = processor.apply_chat_template([{'role': 'system', 'content': job['system_prompt']}, {'role': 'user', 'content': plain.processor_text}],
                                             tokenize=False, add_generation_prompt=False)
    else:
        text = plain.processor_text
    for i in range(len(job['images'])):
        assert f'<Picture {i + 1}>' in text
    assert len(plain.images) == len(job['images']) and len(plain.videos) == (1 if job['condition_root'] else 0)
    presentation = H3Presentation(text=plain.text, images=plain.images, videos=plain.videos, processor_text=text)
    with torch.inference_mode():
        hidden, tags = encode_h3_presentation(processor, model, presentation)
    assert hidden.ndim == 2 and hidden.shape[1] == 5120 and tags.shape == (hidden.shape[0],) and torch.isfinite(hidden).all()
    save_file({'hidden': hidden.to('cpu', torch.bfloat16).contiguous(), 'tags': tags.to('cpu', torch.int64).contiguous()}, str(root / 'qwen.safetensors'))
    (root / 'presentation.txt').write_text(text)
    files = {p.name: digest(p) for p in sorted(root.glob('*.safetensors'))}
    (root / 'manifest.json').write_text(json.dumps({'status': 'complete', 'text_rows': int(hidden.shape[0]), 'files': files,
                                                    'contract_sha256': hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()}, indent=1))
    del hidden, tags, visuals, plain, presentation
    gc.collect(); torch.cuda.empty_cache()
    return 'prepared'


def main():
    from verify_h20_runtime import reserve
    paths = [Path(x) for x in Path(sys.argv[1]).read_text().split() if x]
    marker = reserve()
    vae_box, qwen_box, bad = {}, {}, 0
    for p in paths:
        t = time.monotonic()
        try:
            r = prepare(json.loads(p.read_text()), vae_box, qwen_box)
            print('FFPREP', r, p, round(time.monotonic() - t, 1), flush=True)
        except Exception:
            bad += 1
            print('FFPREP_FAILED', p, traceback.format_exc(), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
