# Geometry-aware copy of shared/tools/cwm/ffgen_many.py (editing): geometry from cwm_h3_inference.constants (patched by flex_geom.apply) instead of the fixed 124 / 1344x768.
"""Stage-1 (free first frame) H3 Ref2VA generation for MANY jobs in one torchrun process; the DiT + LoRA stays loaded.

  cd @CD_SHARED@/tools/cwm
  torchrun --standalone --nproc_per_node=8 @CD_SHARED@/tools_dev/cwm/ffgen_many.py \
      --jobs LIST.txt [--summary SUMMARY.json] [--results-root DIR] [--strict]

LIST.txt: one ffgen.py job JSON path per line (blank lines and lines starting with '#' are ignored).
The DiT + CWM LoRA is loaded, merged and broadcast once (by the first job that runs); the video VAE stays loaded on rank 0.
Every job writes exactly what `ffgen.py --job JOB.json` writes: job['output'] (mp4, all frames), job['frame0'] (lossless
PNG of frame 0), latents.safetensors and metrics.json next to the output, and - when job['export'] is set - the copies
<results-root>/<arm>/<category>/<name>.{mp4,png,metrics.json} (results-root defaults to the shared results tree).
metrics.json has the same fields and phases as ffgen.py; the model-loading phase appears only in the job that loaded the
model (later jobs get loading.resident = true), and a 'resident' block records the job index, the free GPU memory at the
start and whether the VAE was loaded in this job. Jobs whose output already exists are skipped (nothing is written).
A failing job gets metrics status 'failed' + error + traceback (and the metrics export, as in ffgen.py) and the loop
continues. Exit code 0 when the loop completes (failures are in the metrics and the summary), unless --strict.

--parallel-decode spreads the VAE tile decodes over all ranks (bit-identical pixels, see resident.parallel_clips);
--verify-decode additionally runs the serial decode on rank 0 and fails the job unless the pixels are identical.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append('@CD_SHARED@/tools/cwm')  # ffgen.py = the single-job reference (node copy)
import resident as R  # noqa: E402  (puts the acceleration code on sys.path)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from optimizations import rank, world  # noqa: E402
import ffgen  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[3] / 'results'   # default export root (flexrun passes --results-root)


def model_key(job):
    m = job['models']
    return (str(m['dit']), str(m['lora']) if job['lora'] else None)


def preflight(job):
    for p in [job['models']['dit']] + ([job['models']['lora']] if job['lora'] else []):
        if not pathlib.Path(p).exists():
            raise FileNotFoundError(p)


def loader(job, device):
    if job['lora']:
        from loading import load_broadcast
        cfg = SimpleNamespace(models=SimpleNamespace(dit=pathlib.Path(job['models']['dit']), lora=pathlib.Path(job['models']['lora'])),
                              expected_lora_modules=200)
        return load_broadcast(cfg, device)
    raise ValueError('only H3 Ref2VA with the CWM LoRA is supported (job lora: true)')


def run_job(index, path, res, args, started):
    device = res.device
    decision = None
    if rank() == 0:
        try:
            job = json.loads(pathlib.Path(path).read_text())
            decision = ['skip' if pathlib.Path(job['output']).exists() else 'run', job, None]
        except Exception:
            decision = ['bad', None, traceback.format_exc()]
    kind, job, error = R.broadcast(decision, device)
    if kind == 'bad':
        if rank() == 0:
            print(json.dumps({'job': path, 'status': 'failed', 'error': 'unreadable job'}), error, flush=True)
        return dict(index=index, job=path, status='failed', error=error)
    if kind == 'skip':
        if rank() == 0:
            print(json.dumps({'job': path, 'status': 'skipped', 'output': job['output']}), flush=True)
        return dict(index=index, job=path, job_id=job.get('job_id'), status='skipped', output=job['output'])

    out = pathlib.Path(job['output']); png = pathlib.Path(job['frame0']); runtime = out.parent
    result = dict(status='running', job=str(path), job_id=job['job_id'], frames=job['frames'], lora=job['lora'], world_size=world(),
                  images=[x['key'] for x in job['images']], proxy_video=bool(job['condition_root']), seed=job['seed'], steps=job['steps'],
                  fixed_first_frame=False, phases={}, step_seconds=[])
    result['resident'] = dict(script='ffgen_many.py', job_index=index, free_gib_min_at_start=R.free_gib(device))
    results_root = args.results_root

    def save():
        if rank() == 0:
            R.quiet(lambda: (runtime / 'metrics.json').write_text(json.dumps(result, indent=2)), 'metrics.json')

    def phase(name, fn):
        R.sync(); t = time.monotonic(); value = R.guarded(device, fn, name); R.sync()
        result['phases'][name] = time.monotonic() - t
        if rank() == 0:
            print(json.dumps({'job_id': job['job_id'], 'phase': name, 'seconds': result['phases'][name]}), flush=True)
        save(); return value

    inputs = sample = lat = slices = clips = None
    try:
        R.guarded(device, lambda: runtime.mkdir(parents=True, exist_ok=True), 'mkdir')
        inputs = phase('load_conditions', lambda: ffgen.build_inputs(job, device))  # unchanged single-job code
        result['packed_rows'] = inputs['layout'].row_count
        key = model_key(job)
        loaded_now = res.model is None
        if loaded_now:
            try:
                R.guarded(device, lambda: preflight(job), 'model files')
                phase('load_merge_broadcast' if job['lora'] else 'load_broadcast', lambda: res.load(key, lambda: loader(job, device), job['job_id']))
            except R.JobFailed:
                res.load_failed = True
                raise
        else:
            R.guarded(device, lambda: res.check_key(key), 'model check')
        if job['lora']:
            result['lora_gate'] = res.gate
        result['loading'] = dict(res.details, resident=not loaded_now, loaded_by_job=res.loaded_by)
        result['adaln'] = phase('precompute_adaln', lambda: res.prepare_adaln(job['job_id'], inputs['layout'], inputs['text_token_tags'], None,
                                                                             inputs['initial_video'].shape, job['steps'], True, 0.))
        state = res.state
        from musubi_tuner.minimax_h3.sampling import sample_joint_av
        R.sync(); last = [time.monotonic()]; torch.cuda.reset_peak_memory_stats()

        def callback(i, n):
            R.sync(); now = time.monotonic(); result['step_seconds'].append(now - last[0]); last[0] = now
            if rank() == 0:
                print(json.dumps({'job_id': job['job_id'], 'step': i, 'total': n, 'seconds': result['step_seconds'][-1]}), flush=True)
        sample = phase('denoise', lambda: sample_joint_av(res.model, **inputs, step_callback=callback))

        def check():
            assert torch.isfinite(sample.video).all() and torch.isfinite(sample.audio).all()
            assert not any(x['reuse'] for x in state.decisions)
        R.guarded(device, check, 'output check')
        result['peak_allocated_gib_per_rank'], result['peak_reserved_gib_per_rank'] = R.peak_memory(device)
        R.on_rank0(device, lambda: save_file({'video': sample.video[0].float().cpu().contiguous()}, str(runtime / 'latents.safetensors')),
                   'save latents')
        lat = sample.video.float().contiguous() if args.parallel_decode else None
        sample = inputs = None
        gc.collect(); torch.cuda.empty_cache(); R.sync()
        t_decode = time.monotonic()
        slices = clips = None
        if args.parallel_decode:  # every rank decodes a share of the VAE tiles; rank 0 replays them below
            if world() > 1:
                dist.broadcast(lat, src=0)
            result['resident']['vae_loaded_in_this_job'] = R.guarded(device, lambda: res.video_vae(job['models']['video_vae'])[1], 'load vae')
            slices, clips = R.guarded(device, lambda: R.parallel_clips(res.vae, lat.to(torch.float16), device), 'parallel decode')
            lat = None
            result['resident']['parallel_decode_seconds'] = time.monotonic() - t_decode

        def decode_write():
            import av
            from PIL import Image
            t = t_decode
            vae, loaded = res.video_vae(job['models']['video_vae'])
            if not args.parallel_decode:
                result['resident']['vae_loaded_in_this_job'] = loaded
            lat = load_file(str(runtime / 'latents.safetensors'))['video'].unsqueeze(0)
            with torch.inference_mode():
                if args.parallel_decode:
                    with R.ClipReplay(vae, slices, clips):
                        decoded = vae.decode(lat.to(device, torch.float16))[0, :, :job['frames']].cpu()
                else:
                    decoded = vae.decode(lat.to(device, torch.float16))[0, :, :job['frames']].cpu()
                if args.verify_decode:
                    tv = time.monotonic()
                    serial = vae.decode(lat.to(device, torch.float16))[0, :, :job['frames']].cpu()
                    result['resident']['decode_verified_equal'] = bool(torch.equal(serial, decoded))
                    result['resident']['verify_seconds'] = time.monotonic() - tv
                    t += result['resident']['verify_seconds']  # keep the verification out of decode_write
                    if not result['resident']['decode_verified_equal']:
                        raise RuntimeError('parallel decode differs from the serial decode')
                    del serial
            assert tuple(decoded.shape) == (3, job['frames'], __import__('flex_geom').active()[2], __import__('flex_geom').active()[1]), decoded.shape
            t_decoded = time.monotonic()
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
            result['resident']['decode_seconds'] = t_decoded - t
            result['resident']['png_mp4_seconds'] = time.monotonic() - t_decoded
            del decoded, px, lat
            torch.cuda.empty_cache()
            ex = job.get('export')
            if ex:
                dst = results_root / ex['arm'] / ex['category']; dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out, dst / f"{ex['name']}.mp4"); shutil.copy2(png, dst / f"{ex['name']}.png")
        R.on_rank0(device, decode_write, 'decode_write')
        slices = clips = None
        R.sync(); result['status'] = 'pass'
    except Exception as exc:
        result['status'] = 'failed'; result['error'] = str(exc); result['traceback'] = traceback.format_exc()
        if rank() == 0:
            print(json.dumps({'job_id': job['job_id'], 'status': 'failed', 'error': str(exc)[-3000:]}), flush=True)
        inputs = sample = lat = slices = clips = None
        gc.collect(); torch.cuda.empty_cache()
    finally:
        result['elapsed_seconds'] = time.monotonic() - started; save()
        if rank() == 0 and job.get('export'):
            def export_metrics():
                ex = job['export']; dst = results_root / ex['arm'] / ex['category']; dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(runtime / 'metrics.json', dst / f"{ex['name']}.metrics.json")
            R.quiet(export_metrics, 'metrics export')
    R.sync()
    if rank() == 0:
        print(json.dumps({'status': result['status'], 'job_id': job['job_id'], 'output': str(out), 'elapsed_seconds': result['elapsed_seconds']}),
              flush=True)
    return dict(index=index, job=path, job_id=job['job_id'], status=result['status'], output=str(out), frames=job['frames'],
                elapsed_seconds=result['elapsed_seconds'], phases=result['phases'],
                adaln_tables=(result.get('adaln') or {}).get('tables'), error=result.get('error'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobs', type=pathlib.Path, required=True)
    p.add_argument('--summary', type=pathlib.Path, help='JSON summary written by rank 0 after every job')
    p.add_argument('--results-root', type=pathlib.Path, default=RESULTS, help='root of the export tree (default: shared results)')
    p.add_argument('--strict', action='store_true', help='exit 1 if any job failed')
    p.add_argument('--parallel-decode', action='store_true', help='spread the VAE tile decodes over all ranks (VAE loaded on every rank)')
    p.add_argument('--verify-decode', action='store_true', help='also run the serial decode and require identical pixels (implies --parallel-decode)')
    args = p.parse_args()
    args.parallel_decode = args.parallel_decode or args.verify_decode
    t0 = time.monotonic()
    paths = R.read_list(args.jobs)
    device = R.setup()
    summary = dict(script='ffgen_many.py', jobs_file=str(args.jobs), world_size=world(), setup_seconds=time.monotonic() - t0,
                   parallel_decode=args.parallel_decode, verify_decode=args.verify_decode, jobs=[])

    def write_summary():
        if rank() == 0 and args.summary:
            summary['total_seconds'] = time.monotonic() - t0
            summary['adaln_table_builds'] = res.builds
            summary['counts'] = {s: sum(1 for j in summary['jobs'] if j['status'] == s) for s in ('pass', 'failed', 'skipped')}
            R.quiet(lambda: (args.summary.parent.mkdir(parents=True, exist_ok=True), args.summary.write_text(json.dumps(summary, indent=2))),
                    'summary')
    res = R.Resident(device)
    first = True
    for index, path in enumerate(paths):
        started = t0 if first else time.monotonic()  # the first job that runs includes setup + model loading, as in ffgen.py
        try:
            record = run_job(index, path, res, args, started)
        except Exception:  # only symmetric (data-driven) errors can get here; every rank holds the same job
            record = dict(index=index, job=path, status='failed', error=traceback.format_exc())
            if rank() == 0:
                print(json.dumps({'job': path, 'status': 'failed'}), record['error'], flush=True)
        if record['status'] != 'skipped':
            first = False
        summary['jobs'].append(record)
        write_summary()
        if res.load_failed:
            if rank() == 0:
                print('FFGEN_MANY_ABORT model loading failed; remaining jobs not run', flush=True)
            break
    write_summary()
    counts = {s: sum(1 for j in summary['jobs'] if j['status'] == s) for s in ('pass', 'failed', 'skipped')}
    if rank() == 0:
        print('FFGEN_MANY_DONE', json.dumps(dict(counts, total_seconds=time.monotonic() - t0)), flush=True)
    R.finish()
    sys.exit(1 if args.strict and (counts['failed'] or len(summary['jobs']) < len(paths)) else 0)


if __name__ == '__main__':
    main()
