#!/usr/bin/env python3
"""Revision (counterfactual edit of a generated diorama): a Causal Diorama + an edit instruction -> the edited Causal Diorama and its
video. The edit is made by the editing engine of bin/cd_edit.py (the same editor agent and rules, reference pictures, first-frame
route route_flex and empty-place rule), starting from the diorama's own plan A, pictures, first frame and video. Video B is then
generated anew under conditions B (plan B, its pictures, its white model, its first frame locked as frame 0) from the same initial
noise as video A: the seed recorded in the diorama (generation.seed), 124 frames, 20 steps.

    python3 bin/cd_revise.py --diorama out/balloon/diorama --instruction "Make the balloon blue" --run balloon_blue \\
        --out out/balloon_blue [--prompt "..."] [--outputs both]

--diorama          a diorama directory or zip from cd_gen.py (or one edited by hand); 124 frames at 1344x768. Without a video, the
                   video of plan A is made first (stage 1 + stage 2, as cd_gen.py) and the completed diorama A is written to OUT/diorama_A
--prompt           the scene description; default: the diorama's text.prompt
--outputs both     (default) OUT/diorama_B (edited, with the video) and OUT/video.mp4; diorama: diorama_B without the video;
                   video: only OUT/video.mp4
"""
import argparse, json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
from cd import paths, runs, gen, diorama as D  # noqa: E402


def video_for(dio, run, out, node):
    """plan A's video when the diorama has none: stage 1 + stage 2 from its plan and pictures (the pictures are not drawn again)"""
    p, m = D.load(dio); g = gen.gdir(run + '_A')
    if g.exists():
        raise SystemExit(f'{g} exists: choose another --run name or remove it')
    g.mkdir(parents=True)
    scm = json.loads(json.dumps(m['scm']))
    for node_ in [scm.get('background') or {}] + scm['E']:
        node_.pop('reference', None)
    prep = gen.prepare(run + '_A', scm, pictures_from={k: p / e['file'] for k, e in m['pictures'].items()})
    tids = gen.queue(run + '_A', node, prep, video=True); o = gen.outputs(run + '_A')
    gen.wait(node, tids, [o['video'], o['frame0']])
    return gen.pack(run + '_A', out / 'diorama_A', m.get('text', {}).get('prompt', ''), prep, video=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--diorama', required=True); ap.add_argument('--instruction', required=True); ap.add_argument('--prompt', default='')
    ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--outputs', default='both', choices=('both', 'diorama', 'video')); ap.add_argument('--node', default=paths.NODES[0])
    ap.add_argument('--timeout-hours', type=float, default=12)
    ap.add_argument('--resume', action='store_true', help='continue an interrupted run in its existing tree')
    ap.add_argument('--agent', choices=('api', 'claude-code'), default='claude-code' if os.environ.get('CD_AGENT_MODE') == 'claude_cli' else 'api',
                    help='api: Messages API (Claude Code subagent replica); claude-code: one isolated `claude -p` process per agent task')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    os.environ['CD_AGENT_MODE'] = 'claude_cli' if a.agent == 'claude-code' else 'api'   # read by the engine and its agent runner
    dio, m = D.load(a.diorama)
    if D.validate(dio):
        raise SystemExit(f'the diorama does not validate: {D.validate(dio)}')
    g = m['geometry']
    if (g['frames'], g['width'], g['height']) != (124, 1344, 768):
        raise SystemExit(f'revision needs a diorama of 124 frames at 1344x768 (the geometry of cd_gen.py), not {g}')
    if not m.get('video'):
        video_for(dio, a.run, out, a.node); dio, m = D.load(out / 'diorama_A'); print('plan A video made:', out / 'diorama_A')
    desc = a.prompt.strip() or (m.get('text') or {}).get('prompt', '')
    if a.resume and runs.root(a.run).exists():
        r = runs.root(a.run); (r / 'state' / 'llm_daemon.stop').unlink(missing_ok=True)
    else:
        r = runs.new_tree(a.run, dio / m['video']['file'], desc, a.instruction, no_video=a.outputs == 'diorama')
        runs.engine(a.run, 'flexrun.py', 'prep', str(runs.N))
        runs.seed_known_a(a.run, dio)
    ok = runs.daemon(a.run, lambda: runs.done(a.run, a.outputs), a.timeout_hours)
    if not ok:
        raise SystemExit(f'not finished within {a.timeout_hours} h; the tree is {r}')
    vid = runs.video_b(a.run)
    if a.outputs in ('both', 'diorama'):
        runs.pack_side(a.run, 'B', out / 'diorama_B', kind='edited', text=dict(prompt=desc, instruction=a.instruction),
                       parent=dict(id=m['id'], scm_sha256=D.scm_sha(m['scm'])), video=vid if a.outputs == 'both' else None,
                       video_source='regenerated under plan B with the seed of video A', pipeline='revision')
        print('diorama:', out / 'diorama_B')
    if a.outputs != 'diorama':
        shutil.copy2(vid, out / 'video.mp4'); print('video:', out / 'video.mp4')


if __name__ == '__main__':
    main()
