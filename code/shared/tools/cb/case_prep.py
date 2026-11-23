"""Prepare one full-method case on a node: anchor frame -> (optional) alignment -> condition render -> CWM config.

Batch usage (one worker per GPU via the runner): python case_prep.py LIST_FILE
  each line: <scm_json> <expand_dir> <case_dir> [noalign]
Outputs in <case_dir>: anchor.png, align/ (align.json, scm_aligned.json), scm_final.json, render/ (conditions/, instances/,
prompt.txt, render_receipt.json), references.json, h3_config.json, prep_receipt.json.
Never overwrites a finished case (prep_receipt.json present).
"""
import hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '@CD_NODE@/cwm_h3_reproduction_v1/sources/cwm/src')
import world

P = '@CD_NODE@'
REL = f'{P}/cwm_h3_reproduction_v1/release/models'
SYSTEM = Path(f'{P}/cwm_h3_reproduction_v1/sources/cwm/src/cwm_h3_inference/prompts/system_w0.txt')
SUFFIX = '\nMatch the depth, placement, scale and motion of Video 1.\n'
RULES = os.environ.get('CB_RULES', str(HERE / 'rules_v1.json'))


def extract_anchor(mp4, png):
    import av
    with av.open(str(mp4)) as c:
        frame = next(c.decode(video=0))
        frame.to_image().save(png)


def render_conditions(scm, out):
    from PIL import Image
    for k in ('conditions', 'instances'):
        (out / k).mkdir(parents=True, exist_ok=True)
    sc = world.Scene(scm)
    states = []
    for f in range(world.FRAMES):
        st, rgb, depth, sem, ins = sc.render(f)
        assert np.isfinite(depth).all() and sem.max() <= 11
        depth.tofile(out / 'conditions' / f'{f:06d}.depth.f32')
        Image.fromarray(sem).save(out / 'conditions' / f'{f:06d}.semantic_id.png')
        Image.fromarray(ins).save(out / 'instances' / f'{f:06d}.png')
        if f in (0, 62, 123):
            Image.fromarray(rgb).save(out / f'rgb_{f:03d}.png')
        states.append(st)
    (out / 'world_states.json').write_text(json.dumps(states))
    prompt = world.compile_prompt(scm)
    (out / 'prompt.txt').write_text(prompt)
    receipt = dict(status='pass', frames=world.FRAMES, scm_sha256=hashlib.sha256(json.dumps(scm, sort_keys=True).encode()).hexdigest())
    (out / 'render_receipt.json').write_text(json.dumps(receipt, indent=2))
    return prompt


def prep(scm_path, expand_dir, case_dir, align=True, device='cuda'):
    case = Path(case_dir); case.mkdir(parents=True, exist_ok=True)
    if (case / 'prep_receipt.json').exists():
        return json.loads((case / 'prep_receipt.json').read_text())
    t0 = time.time()
    anchor = case / 'anchor.png'
    if not anchor.exists():
        extract_anchor(Path(expand_dir) / 'generated.mp4', anchor)
    scm = json.loads(Path(scm_path).read_text())
    rep, level = None, 'none'
    if align:
        # Align to the anchor, but keep the plan's own requirements: re-check each alignment level with the plan
        # checker and fall back (full -> no rescale -> ease from anchor onto the plan within 1 s -> camera pitch only -> none).
        import align_anchor, scm_lint, re
        m = re.search(r'scm_(\w+)\.json$', str(scm_path))
        rp = HERE / f'rules_{m.group(1)}.json' if m else Path(RULES)
        rules = json.loads((rp if rp.exists() else Path(RULES)).read_text())['lint']
        for lv, kw in (('full', {}), ('noscale', {'scale': False}), ('blend', {'blend': 1.0}), ('pitch', {'move': False})):
            r = align_anchor.run(scm_path, str(anchor), str(case / f'align_{lv}'), device=device, **kw)
            cand = json.loads((case / f'align_{lv}' / 'scm_aligned.json').read_text())
            errs = scm_lint.lint(cand, rules)[0]
            r['lint_errors'] = [e['msg'] for e in errs]
            rep = rep or {}
            rep[lv] = {k: r.get(k) for k in ('candidates', 'detected', 'pitch_change_deg', 'mean_iou_before', 'mean_iou_after', 'undetected', 'error', 'lint_errors')}
            if not errs:
                scm, level = cand, lv
                break
    (case / 'scm_final.json').write_text(json.dumps(scm, indent=1))
    prompt = render_conditions(scm, case / 'render') + SUFFIX
    (case / 'references.json').write_text(json.dumps({'images': [{'entity': 'anchor', 'path': str(anchor)}],
                                                      'system_prompt': SYSTEM.read_text()}, indent=2))
    cfg = {'anchor_image': str(anchor), 'attention_mode': 'flash3', 'blocks_to_swap': 0, 'cache_dir': str(case / 'cache'),
           'condition_root': str(case / 'render' / 'conditions'), 'expected_lora_modules': 200,
           'models': {'audio_vae': f'{REL}/vae/minimax_h3_audio_vae_fp32.safetensors',
                      'dit': f'{REL}/diffusion_models/minimax_h3_ref2va_bf16.safetensors',
                      'lora': f'{REL}/lora/minimax-h3-cwm.safetensors',
                      'processor': f'{REL}/processors/Qwen3-VL-32B-Instruct',
                      'text_encoder': f'{REL}/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors',
                      'video_vae': f'{REL}/vae/minimax_h3_video_vae_fp16.safetensors'},
           'output': str(case / 'A' / 'generated.mp4'), 'seed': 9942001, 'steps': 20,
           'windows': [{'start_frame': 0, 'prompt': prompt}]}
    (case / 'h3_config.json').write_text(json.dumps(cfg, indent=2))
    r = dict(status='pass', aligned=bool(align), align_level=level, seconds=round(time.time() - t0, 1), align=rep,
             prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    (case / 'prep_receipt.json').write_text(json.dumps(r, indent=2))
    return r


def main():
    lines = [l.split() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    bad = 0
    for parts in lines:
        scm, expand, case = parts[:3]
        try:
            r = prep(scm, expand, case, align=not (len(parts) > 3 and parts[3] == 'noalign'))
            print('PREP', case, r.get('align_level'), flush=True)
        except Exception:
            bad += 1
            print('PREP_FAILED', case, traceback.format_exc(), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
