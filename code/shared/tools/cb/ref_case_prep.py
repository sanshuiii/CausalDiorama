"""Prepare one stage-2 case: anchor = frame 0 of the stage-1 (reference) generation, no alignment.

Batch usage (runner 'workers'): python ref_case_prep.py LIST_FILE   each line: <case_spec.json>
case_spec: case_dir, plan, render_dir, anchor_from, images [{key, path}], system_prompt, caption, seed.
Writes anchor.png, scm_final.json (the plan, unchanged), references.json (anchor + reference pictures), h3_config.json,
prep_receipt.json. The plan is NOT re-fitted to the anchor; OWLv2 only measures how well frame 0 matches the plan
(mean box IoU before alignment), reported in prep_receipt.json.
"""
import hashlib, json, shutil, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
P = '@CD_NODE@'
REL = f'{P}/cwm_h3_reproduction_v1/release/models'


def prep(spec):
    case = Path(spec['case_dir']); case.mkdir(parents=True, exist_ok=True)
    if (case / 'prep_receipt.json').exists():
        return json.loads((case / 'prep_receipt.json').read_text())
    t0 = time.time()
    anchor = case / 'anchor.png'
    shutil.copy2(spec['anchor_from'], anchor)
    plan = json.loads(Path(spec['plan']).read_text())
    (case / 'scm_final.json').write_text(json.dumps(plan, indent=1))
    report = None
    try:
        import align_anchor
        r = align_anchor.run(str(case / 'scm_final.json'), str(anchor), str(case / 'iou_check'), device='cuda')
        report = {k: r.get(k) for k in ('candidates', 'detected', 'undetected', 'mean_iou_before', 'error')}
    except Exception:
        report = {'error': traceback.format_exc()[-800:]}
    images = [{'entity': 'anchor', 'path': str(anchor)}] + [{'entity': x['key'], 'path': x['path'], 'short_edge': 768} for x in spec['images']]
    (case / 'references.json').write_text(json.dumps({'images': images, 'system_prompt': spec['system_prompt']}, indent=2))
    cfg = {'anchor_image': str(anchor), 'attention_mode': 'flash3', 'blocks_to_swap': 0, 'cache_dir': str(case / 'cache'),
           'condition_root': str(Path(spec['render_dir']) / 'conditions'), 'expected_lora_modules': 200,
           'models': {'audio_vae': f'{REL}/vae/minimax_h3_audio_vae_fp32.safetensors',
                      'dit': f'{REL}/diffusion_models/minimax_h3_ref2va_bf16.safetensors',
                      'lora': f'{REL}/lora/minimax-h3-cwm.safetensors',
                      'processor': f'{REL}/processors/Qwen3-VL-32B-Instruct',
                      'text_encoder': f'{REL}/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors',
                      'video_vae': f'{REL}/vae/minimax_h3_video_vae_fp16.safetensors'},
           'output': str(case / 'A' / 'generated.mp4'), 'seed': spec['seed'], 'steps': 20,
           'windows': [{'start_frame': 0, 'prompt': spec['caption']}]}
    (case / 'h3_config.json').write_text(json.dumps(cfg, indent=2))
    r = dict(status='pass', aligned=False, align_level='none (reference first frame)', seconds=round(time.time() - t0, 1), iou_check=report,
             anchor_sha256=hashlib.sha256(anchor.read_bytes()).hexdigest(), prompt_sha256=hashlib.sha256(spec['caption'].encode()).hexdigest())
    (case / 'prep_receipt.json').write_text(json.dumps(r, indent=2))
    return r


def main():
    bad = 0
    for line in Path(sys.argv[1]).read_text().split():
        try:
            r = prep(json.loads(Path(line).read_text()))
            print('PREP', line, (r.get('iou_check') or {}).get('mean_iou_before'), flush=True)
        except Exception:
            bad += 1
            print('PREP_FAILED', line, traceback.format_exc(), flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
