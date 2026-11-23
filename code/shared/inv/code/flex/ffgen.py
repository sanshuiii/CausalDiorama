# Geometry-aware copy of shared/tools/cwm/ffgen.py (editing): geometry from cwm_h3_inference.constants (patched by flex_geom.apply) instead of the fixed 124 / 1344x768.
"""Free-first-frame H3 Ref2VA on 1/2/4/8 H20s (stage 1): per-subject reference pictures, optional proxy video,
optional CWM LoRA, no RGB anchor and no fixed first frame.

  torchrun --standalone --nproc_per_node=8 ffgen.py --job JOB.json

Uses the caches of ffprep.py. Writes job['output'] (mp4, all frames), job['frame0'] (lossless PNG of frame 0, decoded
from the latents), metrics.json next to the output, and copies them to the shared results tree when job['export'] is set.
Existing outputs are never overwritten.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, pathlib, shutil, sys, time, traceback
from types import SimpleNamespace
sys.path.insert(0, '@CD_NODE@/cwm_h3_acceleration_v1/code')
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from common import setup, sync, finish
from optimizations import RuntimeState, install, precompute_adaln, rank, world

RESULTS = pathlib.Path(__file__).resolve().parents[3] / 'results'   # <root>/inv/code/flex/ffgen.py -> <root>/results


def build_inputs(job, device):
    from musubi_tuner.minimax_h3.media import video_latent_frames, audio_latent_frames
    from musubi_tuner.minimax_h3.packing import H3ReferenceGeometry, H3VideoGeometry, build_h3_layout
    from musubi_tuner.minimax_h3.sampling import initialize_target_latents
    root = pathlib.Path(job['cache_dir'])
    m = json.loads((root / 'manifest.json').read_text())
    assert m['status'] == 'complete'
    for name, h in m['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == h, name
    q = load_file(str(root / 'qwen.safetensors'))
    hidden, tags = q['hidden'].unsqueeze(0), q['tags'].unsqueeze(0)
    values = [load_file(str(root / f'image_{i:02d}.safetensors'))['latent'].unsqueeze(0) for i in range(len(job['images']))]
    geometry = [H3ReferenceGeometry('image', video=H3VideoGeometry(*v.shape[2:])) for v in values]
    offsets = [4000 + i for i in range(len(values))]
    if job['condition_root']:
        duv = load_file(str(root / 'duv.safetensors'))['latent']
        duv = torch.nn.functional.pad(duv.unsqueeze(0), (0, duv.shape[-1] % 2, 0, duv.shape[-2] % 2), value=0.0).contiguous()  # flex: height and width to even (16:9: width 21 -> 22 as before)  # width 21 -> 22, as the CWM release
        geometry.append(H3ReferenceGeometry('video', video=H3VideoGeometry(*duv.shape[2:])))
        values.append(duv); offsets.append(1000)
    frames = video_latent_frames(job['frames']); audio = audio_latent_frames(job['frames'])
    layout = build_h3_layout(task='ref2va', text_length=hidden.shape[1], target_video=H3VideoGeometry(frames, __import__('flex_geom').active()[2] // 16, __import__('flex_geom').active()[1] // 16),
                             target_audio_frames=audio, references=tuple(geometry))
    v, a = initialize_target_latents(video_shape=(1, 24, frames, __import__('flex_geom').active()[2] // 16, __import__('flex_geom').active()[1] // 16), audio_shape=(1, 32, 2, audio), seed=job['seed'], device=device,
                                     video_dtype=torch.float32, audio_dtype=torch.float32)
    refs = []
    for value, offset in zip(values, offsets):
        value = value.to(device, torch.float32); g = torch.Generator(device='cpu').manual_seed(job['seed'] + offset)
        noise = torch.randn(tuple(value.shape), generator=g, dtype=torch.float32).to(device)
        refs.append(.999 * value + .001 * noise)
    return dict(layout=layout, text_hidden_states=hidden.to(device, torch.bfloat16), text_token_tags=tags.to(device), initial_video=v,
                initial_audio=a, steps=job['steps'], video_shift=12., audio_shift=3., visual_condition_latents=tuple(refs),
                audio_condition_latents=(), visual_condition_clean=.999, audio_condition_clean=1.)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--job', type=pathlib.Path, required=True); args = p.parse_args()
    job = json.loads(args.job.read_text())
    out = pathlib.Path(job['output']); png = pathlib.Path(job['frame0']); runtime = out.parent
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic(); device = setup()
    result = dict(status='running', job=str(args.job), job_id=job['job_id'], frames=job['frames'], lora=job['lora'], world_size=world(),
                  images=[x['key'] for x in job['images']], proxy_video=bool(job['condition_root']), seed=job['seed'], steps=job['steps'],
                  fixed_first_frame=False, phases={}, step_seconds=[])

    def save():
        if rank() == 0:
            (runtime / 'metrics.json').write_text(json.dumps(result, indent=2))

    def phase(name, fn):
        sync(); t = time.monotonic(); value = fn(); sync()
        result['phases'][name] = time.monotonic() - t
        if rank() == 0:
            print(json.dumps({'phase': name, 'seconds': result['phases'][name]}), flush=True)
        save(); return value
    try:
        runtime.mkdir(parents=True, exist_ok=True)
        inputs = phase('load_conditions', lambda: build_inputs(job, device))
        result['packed_rows'] = inputs['layout'].row_count
        if job['lora']:
            from loading import load_broadcast
            cfg = SimpleNamespace(models=SimpleNamespace(dit=pathlib.Path(job['models']['dit']), lora=pathlib.Path(job['models']['lora'])),
                                  expected_lora_modules=200)
            model, result['lora_gate'], result['loading'] = phase('load_merge_broadcast', lambda: load_broadcast(cfg, device))
        else:
            raise ValueError('only H3 Ref2VA with the CWM LoRA is supported (job lora: true)')
        model.eval().requires_grad_(False)
        state = install(model, RuntimeState(job['steps']))
        result['adaln'] = phase('precompute_adaln', lambda: precompute_adaln(model, state, inputs['layout'], inputs['text_token_tags'], None,
                                                                         inputs['initial_video'].shape))
        state.fused = True; state.cache_threshold = 0.
        from musubi_tuner.minimax_h3.sampling import sample_joint_av
        sync(); last = [time.monotonic()]; torch.cuda.reset_peak_memory_stats()

        def callback(i, n):
            sync(); now = time.monotonic(); result['step_seconds'].append(now - last[0]); last[0] = now
            if rank() == 0:
                print(json.dumps({'step': i, 'total': n, 'seconds': result['step_seconds'][-1]}), flush=True)
        sample = phase('denoise', lambda: sample_joint_av(model, **inputs, step_callback=callback))
        assert torch.isfinite(sample.video).all() and torch.isfinite(sample.audio).all()
        assert not any(x['reuse'] for x in state.decisions)
        mem = torch.tensor([torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30], device=device)
        if world() > 1:
            dist.all_reduce(mem, op=dist.ReduceOp.MAX)
        result['peak_allocated_gib_per_rank'], result['peak_reserved_gib_per_rank'] = mem[0].item(), mem[1].item()
        if rank() == 0:
            save_file({'video': sample.video[0].float().cpu().contiguous()}, str(runtime / 'latents.safetensors'))
        sample = inputs = state = model = None   # drop the references (as del)
        gc.collect(); torch.cuda.empty_cache(); sync()
        if rank() == 0:
            import av
            from PIL import Image
            from musubi_tuner.minimax_h3.video_vae import load_video_vae
            t = time.monotonic()
            vae = load_video_vae(job['models']['video_vae'], device=device, dtype=torch.float16, disable_mmap=False).eval().requires_grad_(False)
            latents = load_file(str(runtime / 'latents.safetensors'))['video'].unsqueeze(0)
            with torch.inference_mode():
                decoded = vae.decode(latents.to(device, torch.float16))[0, :, :job['frames']].cpu()
            assert tuple(decoded.shape) == (3, job['frames'], __import__('flex_geom').active()[2], __import__('flex_geom').active()[1]), decoded.shape
            px = ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).contiguous()
            Image.fromarray(px[0].numpy()).save(png)
            tmp = out.with_suffix('.tmp.mp4')
            with av.open(str(tmp), 'w', options={'movflags': '+faststart'}) as c:
                s = c.add_stream('libx264', rate=24); s.width = __import__('flex_geom').active()[1]; s.height = __import__('flex_geom').active()[2]; s.pix_fmt = 'yuv420p'; s.options = {'crf': '18'}
                for fr in px:
                    for pk in s.encode(av.VideoFrame.from_ndarray(fr.numpy(), format='rgb24')):
                        c.mux(pk)
                for pk in s.encode():
                    c.mux(pk)
            os.replace(tmp, out)
            result['phases']['decode_write'] = time.monotonic() - t
            ex = job.get('export')
            if ex:
                dst = RESULTS / ex['arm'] / ex['category']; dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out, dst / f"{ex['name']}.mp4"); shutil.copy2(png, dst / f"{ex['name']}.png")
        sync(); result['status'] = 'pass'
    except BaseException as exc:
        result['status'] = 'failed'; result['error'] = str(exc); result['traceback'] = traceback.format_exc(); raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started; save()
        if rank() == 0 and job.get('export'):
            ex = job['export']; dst = RESULTS / ex['arm'] / ex['category']; dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runtime / 'metrics.json', dst / f"{ex['name']}.metrics.json")
    if rank() == 0:
        print(json.dumps({'status': result['status'], 'output': str(out), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    finish()


if __name__ == '__main__':
    main()
