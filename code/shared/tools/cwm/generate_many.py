"""Stage-2 (fixed first frame) CWM Ref2VA generation for MANY configs in one torchrun process; the DiT + LoRA stays loaded.

  cd @CD_SHARED@/tools/cwm
  torchrun --standalone --nproc_per_node=8 @CD_SHARED@/tools_dev/cwm/generate_many.py \
      --list LIST.txt [--summary SUMMARY.json] [--results-root DIR] [--no-fusion] [--cache-threshold X] [--strict]

LIST.txt lines: <h3_config.json> <output.mp4> [<arm> <category> <name>]   (blank lines and '#' lines are ignored)
Each line is what `generate.py --config <h3_config.json> --output <output.mp4>` does, with the same outputs: the mp4, its
receipt <output>.json, and <output stem>.runtime/{metrics.json, recovery/window_000.safetensors}. With the export fields,
the video and its metrics are copied to <results-root>/<arm>/<category>/<name>.mp4 / .metrics.json exactly as the node
runner does for a cwm_generate task (also when the output already existed and its metrics say pass).
The DiT + LoRA is loaded once (by the first config that runs) and the video VAE stays loaded on rank 0, so the phases
load_merge_broadcast and load_vae appear only in the job that loaded them (later jobs: loading.resident = true).
Outputs that exist are skipped; a config whose recovery latent exists without a video is refused (as generate.py does);
a failing config gets metrics status 'failed' + traceback and the loop continues. Exit code 0 when the loop completes,
unless --strict. Retake (arm B: --retake-latent/--retake-mask) is not supported here; use generate.py for it.
--parallel-decode spreads the VAE tile decodes over all ranks (bit-identical pixels, see resident.parallel_clips);
--verify-decode additionally runs the serial decode on rank 0 and fails the job unless the pixels are identical.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib
import shutil
import sys
import time
import traceback
from dataclasses import replace

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append('@CD_SHARED@/tools/cwm')  # multiref.py of generate.py (node copy)
import resident as R  # noqa: E402  (puts the acceleration code on sys.path)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from optimizations import rank, world  # noqa: E402
import multiref  # noqa: E402

RESULTS = pathlib.Path('@CD_SHARED@/results')


def sources():
    files = {p.name: p for p in sorted(HERE.glob('*.py'))}
    files['multiref.py'] = pathlib.Path(multiref.__file__)
    return {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()}


SOURCES = sources()  # hashed at start-up: the code this process runs


def model_files(config):
    for p in (config.models.dit, config.models.lora):
        if not pathlib.Path(p).exists():
            raise FileNotFoundError(p)


def parse(line):
    parts = line.split()
    if len(parts) not in (2, 5):
        raise ValueError(f'expected "<h3_config.json> <output.mp4> [<arm> <category> <name>]", got {line!r}')
    export = dict(zip(('arm', 'category', 'name'), parts[2:])) if len(parts) == 5 else None
    return pathlib.Path(parts[0]), pathlib.Path(parts[1]).resolve(), export


def export_result(results_root, export, output):
    """node runner, cwm_generate: copy the video and its metrics when the metrics say pass."""
    m = json.loads((output.with_suffix('.runtime') / 'metrics.json').read_text())
    if m.get('status') != 'pass':
        raise RuntimeError(f"metrics status is {m.get('status')}, not exported")
    dst = results_root / export['arm'] / export['category']
    dst.mkdir(parents=True, exist_ok=True)
    for src, name in ((output, f"{export['name']}.mp4"), (output.with_suffix('.runtime') / 'metrics.json', f"{export['name']}.metrics.json")):
        if pathlib.Path(src).exists():
            shutil.copy2(src, dst / name)


def run_job(index, line, res, args, started):
    device = res.device
    decision = None
    if rank() == 0:
        try:
            config_path, output, export = parse(line)
            runtime = output.with_suffix('.runtime')
            if output.exists():
                decision = ['skip', None]
            elif (runtime / 'recovery/window_000.safetensors').exists():
                decision = ['refuse', f'FileExistsError: {runtime} (recovery latent without a video; generate.py refuses it too)']
            else:
                decision = ['run', None]
        except Exception:
            decision = ['bad', traceback.format_exc()]
    kind, error = R.broadcast(decision, device)
    record = dict(index=index, line=line)
    if kind == 'bad':
        if rank() == 0:
            print(json.dumps({'line': line, 'status': 'failed'}), error, flush=True)
        return dict(record, status='failed', error=error)
    config_path, output, export = parse(line)
    runtime = output.with_suffix('.runtime')
    record.update(output=str(output), config=str(config_path))
    if kind == 'refuse':
        if rank() == 0:
            print(json.dumps({'output': str(output), 'status': 'failed', 'error': error}), flush=True)
        return dict(record, status='failed', error=error)
    if kind == 'skip':
        if rank() == 0:
            print(json.dumps({'output': str(output), 'status': 'skipped'}), flush=True)
            if export:
                R.quiet(lambda: export_result(args.results_root, export, output), f'export of existing {output}')
        R.sync()
        return dict(record, status='skipped')

    job_id = f'{index}:{output}'
    result = {'status': 'running', 'world_size': world(), 'config': str(config_path.resolve()), 'output': str(output),
              'condition_cache_reused': True, 'blocks_to_swap': 0, 'adaln_precompute': True, 'pointwise_fusion': not args.no_fusion,
              'communication_dtype': 'bf16', 'cache_threshold': args.cache_threshold, 'phases': {}, 'step_seconds': []}
    result['resident'] = dict(script='generate_many.py', job_index=index, free_gib_min_at_start=R.free_gib(device))

    def save():
        if rank() == 0:
            R.quiet(lambda: (runtime / 'metrics.json').write_text(json.dumps(result, indent=2)), 'metrics.json')

    def phase(name, fn):
        R.sync(); t = time.monotonic(); value = R.guarded(device, fn, name); R.sync()
        result['phases'][name] = time.monotonic() - t
        if rank() == 0:
            print(json.dumps({'output': str(output), 'phase': name, 'seconds': result['phases'][name]}), flush=True)
        save(); return value

    inputs = sample = lat = slices = clips = None
    try:
        from cwm_h3_inference.config import load_config
        from cwm_h3_inference.constants import RECOVERY_SCHEMA
        from cwm_h3_inference.engine import Ref2VAEngine
        from musubi_tuner.minimax_h3.sampling import sample_joint_av
        R.guarded(device, lambda: (output.parent.mkdir(parents=True, exist_ok=True), runtime.mkdir(parents=True, exist_ok=True)), 'mkdir')
        result['source_sha256'] = SOURCES

        def load():
            c = load_config(config_path, check_files=True)
            if len(c.windows) != 1:
                raise ValueError('this validated entry currently supports one CWM window (124 frames)')
            return c
        config = R.guarded(device, load, 'load_config')
        result.update(seed=config.seed, steps=config.steps, frames=config.output_frames, resolution=[1344, 768])
        phase('validate_cache', lambda: multiref.validate_cache(config))
        inputs = phase('load_conditions', lambda: multiref.build_inputs(config, device))
        key = (str(config.models.dit), str(config.models.lora), config.expected_lora_modules)
        loaded_now = res.model is None
        if loaded_now:
            from loading import load_broadcast
            try:
                R.guarded(device, lambda: model_files(config), 'model files')
                phase('load_merge_broadcast', lambda: res.load(key, lambda: load_broadcast(config, device), job_id))
            except R.JobFailed:
                res.load_failed = True
                raise
        else:
            R.guarded(device, lambda: res.check_key(key), 'model check')
        result['lora_gate'] = res.gate
        result['loading'] = dict(res.details, resident=not loaded_now, loaded_by_job=res.loaded_by)
        result['adaln'] = phase('precompute_adaln', lambda: res.prepare_adaln(
            job_id, inputs['layout'], inputs['text_token_tags'], inputs['video_denoise_mask'], inputs['initial_video'].shape, config.steps,
            not args.no_fusion, args.cache_threshold))
        state = res.state
        R.sync(); last = [time.monotonic()]; torch.cuda.reset_peak_memory_stats()

        def callback(i, n):
            R.sync(); now = time.monotonic(); seconds = now - last[0]; last[0] = now; result['step_seconds'].append(seconds)
            if rank() == 0:
                print(json.dumps({'output': str(output), 'step': i, 'total': n, 'seconds': seconds, 'cached': state.skip}), flush=True)
        sample = phase('denoise', lambda: sample_joint_av(res.model, **inputs, step_callback=callback))

        def check():
            assert torch.isfinite(sample.video).all() and torch.isfinite(sample.audio).all()
            error = (sample.video[:, :, 0] - inputs['input_video_latents'][:, :, 0]).abs().max().item()
            assert error < 1e-5, error
            return error
        result['fixed_first_latent_max_abs_error'] = R.guarded(device, check, 'output check')
        result['peak_allocated_gib_per_rank'], result['peak_reserved_gib_per_rank'] = R.peak_memory(device)
        result['cache_decisions'] = list(state.decisions)

        def save_recovery():
            recovery = runtime / 'recovery'; recovery.mkdir(exist_ok=True)
            save_file({'video': sample.video[0].float().cpu().contiguous(), 'audio': sample.audio[0].float().cpu().contiguous()},
                      str(recovery / 'window_000.safetensors'), metadata={'schema': RECOVERY_SCHEMA, 'window_index': '0'})
        R.on_rank0(device, save_recovery, 'save recovery latent')
        lat = sample.video.float().contiguous() if args.parallel_decode else None
        sample = inputs = None
        gc.collect(); torch.cuda.empty_cache(); R.sync()
        t_decode = time.monotonic()
        if args.parallel_decode:  # every rank decodes a share of the VAE tiles; rank 0 replays them in write_video
            if world() > 1:
                dist.broadcast(lat, src=0)
            t = time.monotonic()
            loaded_vae = R.guarded(device, lambda: res.video_vae(config.models.video_vae)[1], 'load vae')
            torch.cuda.synchronize()
            if loaded_vae:
                result['phases']['load_vae'] = time.monotonic() - t
            result['resident']['vae_loaded_in_this_job'] = loaded_vae
            t_decode = time.monotonic()
            slices, clips = R.guarded(device, lambda: R.parallel_clips(res.vae, lat.to(torch.float16), device), 'parallel decode')
            lat = None
            result['resident']['parallel_decode_seconds'] = time.monotonic() - t_decode

        def decode_write():
            t = time.monotonic()
            vae, loaded = res.video_vae(config.models.video_vae)
            torch.cuda.synchronize()
            if not args.parallel_decode:
                if loaded:
                    result['phases']['load_vae'] = time.monotonic() - t
                result['resident']['vae_loaded_in_this_job'] = loaded
            engine = object.__new__(Ref2VAEngine)
            engine.config = replace(config, cache_dir=runtime, output=output)
            engine.device = device; engine.torch = torch; engine.video_vae = vae; engine.lora_gate = result['lora_gate']
            extra = 0.0
            if args.verify_decode:  # serial decode of the saved latent vs the replayed parallel clips, outside the timing
                tv = time.monotonic()
                with torch.inference_mode():
                    video, _audio = engine._load_recovery(0)
                    serial = vae.decode(video.unsqueeze(0).to(device, torch.float16))[0, :, :124].cpu()
                    with R.ClipReplay(vae, slices, clips):
                        replayed = vae.decode(video.unsqueeze(0).to(device, torch.float16))[0, :, :124].cpu()
                result['resident']['decode_verified_equal'] = bool(torch.equal(serial, replayed))
                del serial, replayed, video, _audio
                extra = result['resident']['verify_seconds'] = time.monotonic() - tv
                if not result['resident']['decode_verified_equal']:
                    raise RuntimeError('parallel decode differs from the serial decode')
            t = t_decode if args.parallel_decode else time.monotonic()
            if args.parallel_decode:
                with R.ClipReplay(vae, slices, clips):
                    engine.write_video()
            else:
                engine.write_video()
            torch.cuda.synchronize()
            result['phases']['decode_write'] = time.monotonic() - t - extra
            torch.cuda.empty_cache()
        R.on_rank0(device, decode_write, 'decode_write')
        slices = clips = None
        R.sync(); result['status'] = 'pass'
    except Exception as exc:
        result['status'] = 'failed'; result['error'] = str(exc); result['traceback'] = traceback.format_exc()
        if rank() == 0:
            print(json.dumps({'output': str(output), 'status': 'failed', 'error': str(exc)[-3000:]}), flush=True)
        inputs = sample = lat = slices = clips = None
        gc.collect(); torch.cuda.empty_cache()
    finally:
        result['elapsed_seconds'] = time.monotonic() - started; save()
    exported = None
    if rank() == 0 and export and result['status'] == 'pass':
        exported = False
        def do_export():
            nonlocal exported
            export_result(args.results_root, export, output); exported = True
        R.quiet(do_export, f'export of {output}')
    R.sync()
    if rank() == 0:
        print(json.dumps({'status': result['status'], 'output': str(output), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    return dict(record, status=result['status'], elapsed_seconds=result['elapsed_seconds'], phases=result['phases'],
                adaln_tables=(result.get('adaln') or {}).get('tables'), exported=exported, error=result.get('error'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--list', type=pathlib.Path, required=True)
    p.add_argument('--summary', type=pathlib.Path, help='JSON summary written by rank 0 after every job')
    p.add_argument('--results-root', type=pathlib.Path, default=RESULTS, help='root of the export tree (default: shared results)')
    p.add_argument('--no-fusion', action='store_true')
    p.add_argument('--cache-threshold', type=float, default=0.0)
    p.add_argument('--strict', action='store_true', help='exit 1 if any job failed')
    p.add_argument('--parallel-decode', action='store_true', help='spread the VAE tile decodes over all ranks (VAE loaded on every rank)')
    p.add_argument('--verify-decode', action='store_true', help='also run the serial decode and require identical pixels (implies --parallel-decode)')
    args = p.parse_args()
    args.parallel_decode = args.parallel_decode or args.verify_decode
    if not 0 <= args.cache_threshold < 1:
        raise ValueError('cache threshold must be in [0,1)')
    t0 = time.monotonic()
    lines = R.read_list(args.list)
    device = R.setup()
    if world() not in (1, 2, 4, 8):
        raise ValueError('use 1, 2, 4, or 8 GPU processes')
    summary = dict(script='generate_many.py', list_file=str(args.list), world_size=world(), setup_seconds=time.monotonic() - t0,
                   parallel_decode=args.parallel_decode, verify_decode=args.verify_decode, jobs=[])
    res = R.Resident(device)

    def write_summary():
        if rank() == 0 and args.summary:
            summary['total_seconds'] = time.monotonic() - t0
            summary['adaln_table_builds'] = res.builds
            summary['counts'] = {s: sum(1 for j in summary['jobs'] if j['status'] == s) for s in ('pass', 'failed', 'skipped')}
            R.quiet(lambda: (args.summary.parent.mkdir(parents=True, exist_ok=True), args.summary.write_text(json.dumps(summary, indent=2))),
                    'summary')
    first = True
    for index, line in enumerate(lines):
        started = t0 if first else time.monotonic()  # the first job that runs includes setup + model loading, as in generate.py
        try:
            record = run_job(index, line, res, args, started)
        except Exception:  # only symmetric (data-driven) errors can get here; every rank holds the same line
            record = dict(index=index, line=line, status='failed', error=traceback.format_exc())
            if rank() == 0:
                print(json.dumps({'line': line, 'status': 'failed'}), record['error'], flush=True)
        if record['status'] != 'skipped':
            first = False
        summary['jobs'].append(record)
        write_summary()
        if res.load_failed:
            if rank() == 0:
                print('GENERATE_MANY_ABORT model loading failed; remaining configs not run', flush=True)
            break
    write_summary()
    counts = {s: sum(1 for j in summary['jobs'] if j['status'] == s) for s in ('pass', 'failed', 'skipped')}
    if rank() == 0:
        print('GENERATE_MANY_DONE', json.dumps(dict(counts, total_seconds=time.monotonic() - t0)), flush=True)
    R.finish()
    sys.exit(1 if args.strict and (counts['failed'] or len(summary['jobs']) < len(lines)) else 0)


if __name__ == '__main__':
    main()
