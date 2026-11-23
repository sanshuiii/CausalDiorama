"""CWM condition caches (multiref.prepare) for a list of h3_config.json files; configs that do not exist (skipped cases) are skipped and
failures are recorded without failing the task (exit 0), so one bad case never blocks the node's video generations."""
import json, pathlib, sys, time, traceback
sys.path.insert(0, '@CD_SHARED@/tools/cwm')
sys.path.insert(0, '@CD_NODE@/cwm_h3_reproduction_v1/code')
from verify_h20_runtime import reserve
from cwm_h3_inference.config import load_config
from multiref import prepare

paths = [pathlib.Path(x) for x in pathlib.Path(sys.argv[1]).read_text().split() if x]
paths = [p for p in paths if p.exists() or print('SKIP (no config)', p, flush=True)]
if paths:
    marker = reserve()
for p in paths:
    rec = p.parent / 'prepare_receipt.json'
    if rec.exists() and json.loads(rec.read_text()).get('status') == 'pass':
        print('KEEP', p, flush=True); continue
    t = time.monotonic()
    try:
        r = prepare(load_config(p, check_files=True))
        rec.write_text(json.dumps({'status': 'pass', 'seconds': round(time.monotonic() - t, 1), 'validation': r}, indent=2))
        print('PREPARED', p, round(time.monotonic() - t, 1), flush=True)
    except Exception:
        (p.parent / 'prepare_failed.txt').write_text(traceback.format_exc()); print('PREPARE_FAILED', p, traceback.format_exc(), flush=True)
