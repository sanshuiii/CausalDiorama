#!/usr/bin/env python3
"""Install a configured copy of this repository.

    python3 bin/configure.py --config config/my_site.json

The repository keeps every location as a token (@CD_CODE@, @CD_WORK@, ...). This script writes a copy with the tokens filled in
to `code`, copies the node-side code (shared/) to `shared`, writes generation/config.json (the generation path's config, from
generation/config.example.json), and creates the work, queue and status directories. Run everything from the configured copy. Config keys (see config/site.example.json):
  code       where the configured repository goes (the workstation runs bin/*.py from here)
  work       run data (one tree per run)
  shared     a directory every GPU node and the workstation can read and write: node code, results, models used by the nodes
  queue      the node queues and runner status (often the same as shared)
  node       the root of the GPU environments on each node (README "GPU environment")
  node_runs  node-local scratch (white-model renders, caches, case directories), e.g. <node>/runs
  ffmpeg     an ffmpeg binary on the workstation (static build 6.x; needs -vsync)
  nodes      GPU node names (gpu-0, gpu-1, ...), each served by one runner (bin/cd_init.py --start-runner prints the command)
"""
import argparse, json, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXT = {'.py', '.sh', '.md', '.txt', '.json'}
KEYS = {'code': '@CD_CODE@', 'work': '@CD_WORK@', 'shared': '@CD_SHARED@', 'queue': '@CD_QUEUE@', 'node_runs': '@CD_NODE_RUNS@',
        'node': '@CD_NODE@', 'ffmpeg': '@CD_FFMPEG@', 'nodes': '@CD_NODES@'}


def fill(src, dst, cfg):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in TEXT:
        s = src.read_text()
        for k, tok in KEYS.items():
            s = s.replace(tok, ','.join(cfg[k]) if k == 'nodes' else str(cfg[k]).rstrip('/'))
        dst.write_text(s)
        if src.stat().st_mode & 0o111:
            dst.chmod(dst.stat().st_mode | 0o755)
    else:
        shutil.copy2(src, dst)


def gen_config(cfg, code):
    """generation/config.json from generation/config.example.json: its {{...}} placeholders get the site's locations, and the
    queue, node tools and extra Python paths those of the runners (the mapping is in the package README, "Configure")"""
    n = str(cfg['node']).rstrip('/'); sh = str(cfg['shared']).rstrip('/')
    t = (code / 'generation' / 'config.example.json').read_text()
    for k, v in (('{{SHARED}}', sh), ('{{NODE_WORK}}', str(cfg['node_runs']).rstrip('/')), ('{{H3_ENV}}', f'{n}/cwm_h3_reproduction_v1/.venv'),
                 ('{{MODELS}}', f'{n}/cwm_h3_reproduction_v1/release/models'), ('{{ROOT}}', str(code))):
        t = t.replace(k, v)
    c = json.loads(t)
    c.update(queue_root=f"{str(cfg['queue']).rstrip('/')}/queue", node_tools=f'{n}/compbench_v1/tools', pydeps=f'{n}/compbench_v1/pydeps',
             qwen_pydeps=f'{n}/qwen_image21/pydeps')
    (code / 'generation' / 'config.json').write_text(json.dumps(c, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True); a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    miss = [k for k in KEYS if k not in cfg]
    if miss:
        raise SystemExit(f'config misses {miss}')
    code, shared = Path(cfg['code']), Path(cfg['shared'])
    if code.resolve() == REPO.resolve():
        raise SystemExit('`code` must be a new directory, not the template itself')
    n = 0
    for p in sorted(REPO.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.relative_to(REPO).parts[0] != 'build':
            fill(p, code / p.relative_to(REPO), cfg); n += 1
    for p in sorted((code / 'shared').rglob('*')):   # the node code goes where the nodes read it
        if p.is_file():
            d = shared / p.relative_to(code / 'shared'); d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, d)
    gen_config(cfg, code)
    for d in [Path(cfg['work']) / 'fr', Path(cfg['work']) / 'gen', Path(cfg['work']) / 'inputs', Path(cfg['work']) / 'scratch',
              shared / 'results', shared / 'models', shared / 'inv' / 'logs', shared / 'status', shared / 'generation' / 'work',
              shared / 'generation' / 'results']:
        d.mkdir(parents=True, exist_ok=True)
    for node in cfg['nodes']:
        for d in (Path(cfg['queue']) / 'queue' / node / 'main', Path(cfg['queue']) / 'status' / node):
            d.mkdir(parents=True, exist_ok=True)
    (code / 'config').mkdir(exist_ok=True); (code / 'config' / 'site.json').write_text(json.dumps(cfg, indent=1))
    print(f'{n} files written to {code}; node code copied to {shared}. Next: python3 {code}/bin/cd_init.py --check')


if __name__ == '__main__':
    main()
