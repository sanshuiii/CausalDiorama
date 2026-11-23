"""White-model conditions for a list of plans (CPU). python render.py LIST_FILE, lines '<plan_json> <render_dir>'.
Renderer: cb/case_prep.render_conditions; a failing plan is recorded in <render_dir>/render_failed.txt and skipped (exit 0)."""
import json, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cb'))
from case_prep import render_conditions

for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip():
        continue
    plan, out = line.split(); out = Path(out)
    try:
        if not (out / 'render_receipt.json').exists():
            out.mkdir(parents=True, exist_ok=True)
            render_conditions(json.loads(Path(plan).read_text()), out)
        print('RENDERED', out, flush=True)
    except Exception:
        out.mkdir(parents=True, exist_ok=True); (out / 'render_failed.txt').write_text(traceback.format_exc())
        print('RENDER_FAILED', out, traceback.format_exc(), flush=True)
