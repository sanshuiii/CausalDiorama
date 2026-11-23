#!/usr/bin/env python3
"""Editing a real video: (optional prompt) + source video + edit instruction -> the fitted Causal Diorama of the source, the edited
Causal Diorama and the edited video.

    python3 bin/cd_edit.py --video in.mp4 --instruction "Replace the dog with a white polar bear" --run dog1 --out out/dog1 \\
        [--prompt "A golden retriever runs into a lake at sunset ..."] [--outputs both]

--prompt           a description of the source video; empty: Claude Opus writes one from 6 frames of the source
--outputs both     (default) OUT/diorama_A (fitted to the source), OUT/diorama_B (edited, with the video) and OUT/video.mp4
          diorama  both dioramas, without the F4 inversion (no edited video; diorama_B holds its first frame)
          video    only OUT/video.mp4
The output length follows the source: 17n+5 frames at 24 fps, at most 362 frames (15.08 s); longer sources are cut there.
"""
import argparse, json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
from cd import runs, describe, diorama as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True); ap.add_argument('--instruction', required=True); ap.add_argument('--prompt', default='')
    ap.add_argument('--run', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--outputs', default='both', choices=('both', 'diorama', 'video')); ap.add_argument('--timeout-hours', type=float, default=12)
    ap.add_argument('--resume', action='store_true', help='continue an interrupted run in its existing tree')
    ap.add_argument('--agent', choices=('api', 'claude-code'), default='claude-code' if os.environ.get('CD_AGENT_MODE') == 'claude_cli' else 'api',
                    help='api: Messages API (Claude Code subagent replica); claude-code: one isolated `claude -p` process per agent task')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    os.environ['CD_AGENT_MODE'] = 'claude_cli' if a.agent == 'claude-code' else 'api'   # read by the engine and its agent runner
    if a.resume and runs.root(a.run).exists():
        r = runs.root(a.run); ck = json.loads((r / 'runs' / 'checklist.json').read_text())[0]
        desc, a.instruction = ck['original_description'], ck['editing_instruction']
        (r / 'state' / 'llm_daemon.stop').unlink(missing_ok=True)
    else:
        desc = a.prompt.strip()
        if not desc:
            dur = runs.probe(a.video)[0]; desc = describe.describe(a.video, dur, out / 'describe_calls.jsonl'); print('description:', desc)
        r = runs.new_tree(a.run, a.video, desc, a.instruction, no_video=a.outputs == 'diorama')
        runs.engine(a.run, 'flexrun.py', 'prep', str(runs.N))
    ok = runs.daemon(a.run, lambda: runs.done(a.run, a.outputs), a.timeout_hours)
    if not ok:
        raise SystemExit(f'not finished within {a.timeout_hours} h; the tree is {r}')
    vid = runs.video_b(a.run)
    if a.outputs in ('both', 'diorama'):
        A = runs.pack_side(a.run, 'A', out / 'diorama_A', kind='fitted', text=dict(prompt=desc),
                           video=r / 'src' / f'{runs.N}.mp4', video_source='the source video (24 fps, centre crop)', pipeline='edit')
        runs.pack_side(a.run, 'B', out / 'diorama_B', kind='edited', text=dict(prompt=desc, instruction=a.instruction),
                       parent=dict(id=A['id'], scm_sha256=D.scm_sha(A['scm'])), video=vid if a.outputs == 'both' else None,
                       video_source='F4 inversion of the source under plan B', pipeline='edit')
        print('dioramas:', out / 'diorama_A', out / 'diorama_B')
    if a.outputs != 'diorama':
        shutil.copy2(vid, out / 'video.mp4'); print('video:', out / 'video.mp4')


if __name__ == '__main__':
    main()
