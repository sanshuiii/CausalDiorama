"""Prepare the generation jobs of planned prompts (CPU, on the machine that holds the plans).

For every prompt with a plan (<plans>/<id>/plan.json from plan.py):
  1. conditions: pv2.augment adds the reference-picture fields (one picture per entity, plus the empty place) and pv2.frame0
     measures what is visible at t = 0; a plan the executor cannot run is a planning failure (recorded, not re-run);
  2. reference pictures: one Qwen-Image-2.1 job per picture (40 steps, seed 9942001, the size the picture table gives);
  3. white model: the executor renders depth and semantic maps for 124 frames at 24 fps on the node (render line);
  4. stage 1: H3 Ref2VA + CWM LoRA with the pictures, the white model and the stage-1 caption, 124 frames, 20 steps; frame 0 kept;
  5. stage 2: the same model with the stage-1 frame 0 locked as the first frame, the pictures, the white model and the stage-2
     caption, 124 frames, 20 steps, 1344x768, 24 fps: the final video.

    python generation/prepare.py --plans plans --config config.json [--ids a,b] [--category NAME]

Writes the job files under <work>/jobs and a record per prompt in <work>/prepared.json. A prompt's results land in
<results>/<arm>/<category>/<name>.*, with category and name from its prompt.json (plan.py), else --category (default
"videos") and its id.
"""
import argparse, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARM_STAGE1, ARM_VIDEO, CB, FRAMES, PICTURE_STEPS, SEED, STAGE_STEPS, jdump, load_config  # noqa: E402

sys.path.insert(0, str(CB))
import pv2  # noqa: E402

SYS1 = (CB / 'system_stage1.txt').read_text().strip()
SYS2 = (CB / 'system_stage2.txt').read_text().strip()


def prepare_one(cfg, pid, plans, category):
    """One prompt -> its record (state, job files, render line, stage-2 config and generation line)."""
    work, node = Path(cfg['work_root']), cfg['node_root']; pics = work / 'pics' / pid
    d = Path(plans) / pid; out = dict(lint=None)
    meta = json.loads((d / 'prompt.json').read_text()) if (d / 'prompt.json').exists() else {}
    category, name = meta.get('category', category), meta.get('name', pid)
    try:
        out['lint'] = bool(json.loads((d / 'plan.lint' / 'lint.json').read_text()).get('passed'))
    except (OSError, ValueError):
        pass
    try:
        A2, _ = pv2.augment(json.loads((d / 'plan.json').read_text()))
    except FileNotFoundError as exc:
        return dict(out, state='failed', failed=f'plan file missing: {Path(exc.filename).name}')
    except Exception as exc:   # a plan the executor cannot run
        return dict(out, state='failed', failed=f'plan cannot be executed: {f"{type(exc).__name__}: {exc}"[:200]}')
    (d / 'plan_v2.json').write_text(json.dumps(A2, ensure_ascii=False, indent=1))
    shp = work / 'plans' / f'{pid}.json'; shp.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(d / 'plan_v2.json', shp)
    f0 = json.loads(json.dumps(pv2.frame0(A2)))
    plan = json.loads((d / 'plan_v2.json').read_text())
    pic_jobs = []
    for key, x in pv2.ref_table(plan).items():
        assert x['source'] == 't2v', (pid, key, x['source'])
        spec = dict(id=f'{pid}_{key}', prompt=x['prompt'], width=x['size'][0], height=x['size'][1], seed=SEED, steps=PICTURE_STEPS,
                    out=str(pics / f'{key}.png'))
        pic_jobs.append(jdump(work / 'jobs' / 'pics' / f'{pid}_{key}.json', spec))
    render = f'{node}/render/{pid}'
    cap1, keys1 = pv2.caption(plan, f0, 1); d1 = f'{node}/s1/{pid}'
    s1 = dict(job_id=f's1_{pid}', frames=FRAMES, seed=SEED, steps=STAGE_STEPS, lora=True, models=cfg['models'],
              condition_root=f'{render}/conditions', system_prompt=SYS1, caption=cap1,
              images=[{'key': kk, 'path': str(pics / f'{kk}.png')} for kk in keys1],
              cache_dir=f'{d1}/cache', output=f'{d1}/generated.mp4', frame0=f'{d1}/frame0.png',
              export={'arm': ARM_STAGE1, 'category': category, 'name': name})
    cap2, keys2 = pv2.caption(plan, f0, 2); case_dir = f'{node}/s2/{category}/{pid}'
    s2 = dict(case_dir=case_dir, plan=str(shp), render_dir=render, anchor_from=str(Path(cfg['results_root']) / ARM_STAGE1 / category / f'{name}.png'),
              seed=SEED, images=[{'key': kk, 'path': str(pics / f'{kk}.png')} for kk in keys2], system_prompt=SYS2, caption=cap2)
    return dict(out, state='prepared', category=category, name=name, frame0=f0, pics=pic_jobs, render_line=f'{shp} {render}',
                s1=jdump(work / 'jobs' / 's1' / f'{pid}.json', s1), s2=jdump(work / 'jobs' / 's2' / f'{pid}.json', s2),
                cfg=f'{case_dir}/h3_config.json', gen=f'{case_dir}/h3_config.json {case_dir}/A/generated.mp4 {ARM_VIDEO} {category} {name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--plans', required=True); ap.add_argument('--config', required=True); ap.add_argument('--ids', default='')
    ap.add_argument('--category', default='videos')
    a = ap.parse_args(); cfg = load_config(a.config)
    ids = [x for x in a.ids.split(',') if x] or sorted(p.name for p in Path(a.plans).iterdir() if (p / 'plan.json').exists())
    recf = Path(cfg['work_root']) / 'prepared.json'; rec = json.loads(recf.read_text()) if recf.exists() else {}
    for pid in ids:
        rec[pid] = prepare_one(cfg, pid, a.plans, a.category)
        print(pid, rec[pid]['state'], rec[pid].get('failed', ''), flush=True)
    jdump(recf, rec)


if __name__ == '__main__':
    main()
