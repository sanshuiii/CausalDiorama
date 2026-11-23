#!/usr/bin/env python3
"""Causal Diorama initialisation: check the API, the workstation tools, the GPU nodes and their models.

    python3 bin/cd_init.py --check                 API key + one request, workstation Python modules, ffmpeg, runner heartbeats
    python3 bin/cd_init.py --check-node NODE       queue a check task on NODE (models, environments, imports, LoRA hash); waits
    python3 bin/cd_init.py --start-runner NODE     print the command that starts the runner of NODE (run it on that node)
    python3 bin/cd_init.py --downloads             print the download commands of every model (README "Models")
"""
import argparse, importlib, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
from cd import paths  # noqa: E402

DOWNLOADS = f'''# on each GPU node (huggingface_hub >= 0.30: `hf download`); N = {paths.NODE}
N={paths.NODE}; M=$N/cwm_h3_reproduction_v1/release/models
hf download Comfy-Org/MiniMax-H3 --revision 0543966fbdce5ba05709a8f2031c94bdba629b4a --local-dir $M \\
  --include diffusion_models/minimax_h3_ref2va_bf16.safetensors text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \\
            vae/minimax_h3_video_vae_fp16.safetensors vae/minimax_h3_audio_vae_fp32.safetensors
hf download Qwen/Qwen3-VL-32B-Instruct --revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb --local-dir $M/processors/Qwen3-VL-32B-Instruct \\
  --include chat_template.json config.json generation_config.json merges.txt preprocessor_config.json tokenizer_config.json tokenizer.json video_preprocessor_config.json vocab.json
hf download NTU-yiwen/awm-minimax-h3-new1344-lora-checkpoints --revision a3df90f6742748896457cec84b13dbedbce56fff --local-dir /tmp/cwm_lora \\
  --include q4_system_chat/rigid_3ep_2e5_to_1e6/minimax-h3-new1344-q4-system-chat-fresh-official-3ep-warmup100-cosine2e5to1e6-fa3-v1.safetensors
mkdir -p $M/lora && mv /tmp/cwm_lora/q4_system_chat/rigid_3ep_2e5_to_1e6/*.safetensors $M/lora/minimax-h3-cwm.safetensors
echo "0e14ad22ae203f684ba6521e0b6fbf58f64668ed454fbce346acafd7f448fdb2  $M/lora/minimax-h3-cwm.safetensors" | sha256sum -c
hf download Qwen/Qwen-Image-2.1 --local-dir $N/qwen_image21/model
# once, on the shared directory ({paths.SHARED}): detection, segmentation and depth models (perception and the first-frame masks)
hf download IDEA-Research/grounding-dino-base --local-dir {paths.SHARED}/models/grounding-dino-base
hf download facebook/sam2.1-hiera-large --local-dir {paths.SHARED}/models/sam2.1-hiera-large
hf download depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf --local-dir {paths.SHARED}/models/Depth-Anything-V2-Metric-Outdoor-Small-hf
hf download depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf --local-dir {paths.SHARED}/models/Depth-Anything-V2-Metric-Indoor-Small-hf'''


def check():
    ok = True
    for m in ('numpy', 'PIL', 'requests', 'scipy'):
        try:
            importlib.import_module(m); print('workstation module', m, 'ok')
        except Exception as e:
            ok = False; print('workstation module', m, 'MISSING:', e)
    r = subprocess.run([paths.FFMPEG, '-hide_banner', '-f', 'lavfi', '-i', 'color=black:s=64x64:d=0.1', '-vsync', '0', '-f', 'null', '-'],
                       capture_output=True, text=True)
    print('ffmpeg', paths.FFMPEG, 'ok' if r.returncode == 0 else 'FAILED: ' + r.stderr[-200:]); ok &= r.returncode == 0
    sys.path.insert(0, str(paths.ENGINE / 'llm'))
    import api_llm
    try:
        d, meta = api_llm.complete_native(dict(model=api_llm.MODEL, max_tokens=64, messages=[{'role': 'user', 'content': 'Reply with the single word ok.'}]),
                                          record=paths.WORK / 'scratch' / 'init_calls.jsonl', tag='init', tries=1, stream=False)
        print('LLM', api_llm.MODEL, 'at', api_llm.MESSAGES_URL, 'ok:', ''.join(b.get('text', '') for b in d['content'] if b.get('type') == 'text')[:40])
    except Exception as e:
        ok = False; print('LLM FAILED:', str(e)[:300])
    for n in paths.NODES:
        f = paths.QUEUE / 'status' / n / 'runner.json'
        age = (time.time() - f.stat().st_mtime) / 60 if f.exists() else None
        print('runner', n, 'heartbeat', f'{age:.1f} min ago' if age is not None else 'NONE (start it: bin/cd_init.py --start-runner ' + n + ')')
        ok &= age is not None and age < 10
    print('ALL OK' if ok else 'SOME CHECKS FAILED'); return ok


def check_node(node):
    tid = f'c00_check_{time.strftime("%m%d%H%M%S")}'
    q = paths.QUEUE / 'queue' / node / 'main'; q.mkdir(parents=True, exist_ok=True)
    out = paths.SHARED / 'status' / f'check_{node}.txt'; out.parent.mkdir(parents=True, exist_ok=True)
    (q / f'{tid}.json').write_text(json.dumps(dict(type='shell', retries=0, command=f'bash {paths.SHARED}/tools/check_node.sh > {out} 2>&1')))
    print('queued', tid, 'on', node, '; waiting ...')
    ev = paths.QUEUE / 'status' / node / 'events.log'
    for _ in range(240):
        t = ev.read_text() if ev.exists() else ''
        if f'DONE {tid}' in t or f'FAILED {tid}' in t:
            break
        time.sleep(15)
    print(out.read_text() if out.exists() else f'no report yet: {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true'); ap.add_argument('--check-node'); ap.add_argument('--start-runner'); ap.add_argument('--downloads', action='store_true')
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() else 1)
    if a.check_node:
        check_node(a.check_node)
    if a.start_runner:
        print(f'# on node {a.start_runner}:\nbash {paths.SHARED}/tools/start_runner.sh {a.start_runner}')
    if a.downloads:
        print(DOWNLOADS)
    if not any(vars(a).values()):
        print(__doc__)


if __name__ == '__main__':
    main()
