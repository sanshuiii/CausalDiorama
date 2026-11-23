#!/usr/bin/env python3
"""Generation: a prompt -> a Causal Diorama and its video.

    python3 bin/cd_gen.py --prompt "A hot air balloon floating under a sheep" --run balloon --out out/balloon [--outputs both]

--outputs both     (default) the diorama (plan, pictures, first frame, video) in OUT/diorama/ and the video as OUT/video.mp4
          diorama  plan + reference pictures only (no GPU video generation; no first frame)
          video    only OUT/video.mp4
--node             the GPU node whose runner takes the tasks (default: the first configured node)
--plan FILE        skip the planning agent and use this plan (e.g. a diorama's scm edited by hand)
"""
import argparse, json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
from cd import paths, gen, diorama as D  # noqa: E402


def pv2_keys(A):
    """the picture keys of a plan (as gen.prepare draws them)"""
    A2, _ = gen.pv2.augment(A); return D.reference_keys(A2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prompt', required=True); ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--outputs', default='both', choices=('both', 'diorama', 'video')); ap.add_argument('--node', default=paths.NODES[0])
    ap.add_argument('--plan', default=None)
    ap.add_argument('--resume', action='store_true', help='continue an interrupted run: queue only the steps whose outputs are missing')
    ap.add_argument('--agent', choices=('api', 'claude-code'), default='claude-code' if os.environ.get('CD_AGENT_MODE') == 'claude_cli' else 'api',
                    help='api: Messages API (Claude Code subagent replica); claude-code: one isolated `claude -p` process per agent task')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    os.environ['CD_AGENT_MODE'] = 'claude_cli' if a.agent == 'claude-code' else 'api'   # read by the engine and its agent runner
    g = gen.gdir(a.run); start = 'a'
    if a.resume and (g / 'scm_t2v.json').exists():
        A = json.loads((g / 'scm_t2v.json').read_text()); o = gen.outputs(a.run)
        start = 'h' if o['frame0'].exists() else 'c' if (g / 'pics').exists() and all((g / 'pics' / f'{k}.png').exists() for k in pv2_keys(A)) else 'a'
    else:
        if g.exists():
            raise SystemExit(f'{g} exists: choose another --run name or remove it (or --resume)')
        g.mkdir(parents=True); (g / 'prompt.txt').write_text(a.prompt)
        A = json.loads(Path(a.plan).read_text()) if a.plan else gen.plan(a.run, a.prompt)
        if a.plan:
            (g / 'scm_t2v.json').write_text(json.dumps(A, ensure_ascii=False, indent=1))
    prep = gen.prepare(a.run, A)
    video = a.outputs != 'diorama'
    tids = gen.queue(a.run, a.node, prep, video=video, start=start)
    o = gen.outputs(a.run)
    gen.wait(a.node, tids, [o['video'], o['frame0']] if video else [p for p in (g / 'pics').glob('*.png')])
    if a.outputs in ('both', 'diorama'):
        m = gen.pack(a.run, out / 'diorama', a.prompt, prep, video=video); print('diorama:', out / 'diorama', '|', len(m['pictures']), 'pictures')
    if video:
        shutil.copy2(o['video'], out / 'video.mp4'); print('video:', out / 'video.mp4')


if __name__ == '__main__':
    main()
