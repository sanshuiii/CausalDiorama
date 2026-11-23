#!/usr/bin/env python3
"""Causal Diorama tools.

    python3 bin/cd_diorama.py info     DIORAMA            summary of the manifest (variables, pictures, video)
    python3 bin/cd_diorama.py validate DIORAMA [--render] hashes, completeness, and (--render) that the plan renders
    python3 bin/cd_diorama.py render   DIORAMA OUT        white-model depth / semantic images of frame 0, the middle and the end
    python3 bin/cd_diorama.py zip      DIORAMA [OUT.zip]  one file to move or share; every command also reads the zip
    python3 bin/cd_diorama.py unzip    DIORAMA.zip OUT
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine')); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine' / 'cb'))
from cd import diorama as D  # noqa: E402


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == 'info':
        p, m = D.load(path); s = m['scm']
        print(json.dumps(dict(id=m['id'], kind=m['kind'], geometry=m['geometry'], camera=s['C'].get('controller'),
                              entities=[e['id'] for e in s['E']], tracks=[x['id'] for x in s.get('S', [])], events=[x['id'] for x in s.get('R', [])],
                              dependencies=len(s.get('D', [])), pictures=sorted(m['pictures']), first_frame=bool(m.get('first_frame')),
                              video=bool(m.get('video')), text=m.get('text'), parent=m.get('parent')), ensure_ascii=False, indent=1))
    elif cmd == 'validate':
        errs = D.validate(path, render='--render' in sys.argv)
        print('\n'.join(errs) if errs else 'ok'); sys.exit(1 if errs else 0)
    elif cmd == 'render':
        print(D.render_preview(path, sys.argv[3]))
    elif cmd == 'zip':
        print(D.zip_dir(path, sys.argv[3] if len(sys.argv) > 3 else None))
    elif cmd == 'unzip':
        print(D.unzip(path, sys.argv[3]))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
