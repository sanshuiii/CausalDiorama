"""Stage-2 case preparation (anchor = the chosen stage-1 frame 0, no alignment; OWLv2 only measures frame-0 IoU).
python pv2_case_prep.py LIST_FILE, lines = case-spec JSON paths (same spec as cb/ref_case_prep.py). Plans keep the camera in the
node C; the IoU measurement (align_anchor) reads a fixed `camera`, so a copy of the plan with camera = C's pose at t = 0 is used for
it. Cases whose anchor or render is missing are recorded in <case_dir>/prep_skipped.json and skipped; always exits 0.
A white-model render that is missing on this node (render_dir without render_receipt.json) is first rendered here from
spec['plan'] with the same function as e2/render.py (cb/case_prep.render_conditions; world.py is deterministic). A render error is
recorded in <render_dir>/render_failed.txt as e2/render.py does, and the case is then skipped."""
import json, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cb'))
import ref_case_prep, world

for line in Path(sys.argv[1]).read_text().split():
    spec = json.loads(Path(line).read_text()); case = Path(spec['case_dir']); case.mkdir(parents=True, exist_ok=True)
    rd = Path(spec['render_dir'])
    if Path(spec['anchor_from']).exists() and not (rd / 'render_receipt.json').exists():
        try:
            from case_prep import render_conditions
            rd.mkdir(parents=True, exist_ok=True)
            render_conditions(json.loads(Path(spec['plan']).read_text()), rd)
            print('RENDERED (missing on this node)', rd, flush=True)
        except Exception:
            rd.mkdir(parents=True, exist_ok=True); (rd / 'render_failed.txt').write_text(traceback.format_exc())
            print('RENDER_FAILED', rd, traceback.format_exc(), flush=True)
    miss = [p for p in (spec['anchor_from'], str(Path(spec['render_dir']) / 'render_receipt.json')) if not Path(p).exists()]
    if miss:
        (case / 'prep_skipped.json').write_text(json.dumps({'missing': miss}, indent=1)); print('PREP_SKIPPED', line, miss, flush=True); continue
    try:
        plan = json.loads(Path(spec['plan']).read_text())
        if 'C' in plan and 'camera' not in plan:
            plan['camera'] = world.Scene(plan).camera(0)
        (case / 'plan_for_prep.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1))
        r = ref_case_prep.prep(dict(spec, plan=str(case / 'plan_for_prep.json')))
        print('PREP', line, (r.get('iou_check') or {}).get('mean_iou_before'), flush=True)
    except Exception:
        (case / 'prep_failed.txt').write_text(traceback.format_exc()); print('PREP_FAILED', line, traceback.format_exc(), flush=True)
