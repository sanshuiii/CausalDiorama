"""The perception check of the editing engine, a context-free agent stage between perception (perceive.py) and the fitting planner.

Inputs per clip (checks/percept/<n>/): one strip per entity (its detection box cropped on each sampled frame, frame / time / score
burnt in; "not detected" tiles), three full frames with every box and a large label, the entity list (id, short text, appearance)
and the agent's task file (prompts/agents/pcheck/). The agent writes verdict.json {entity: {frame: "ok" | "drop" | "is:<entity>"}}
for every detection; percept_apply writes percept/<n>_checked (boxes moved or dropped, masks renamed with them).
"""
import json, shutil, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import make_stage2 as M2  # noqa: E402  (frame_png)
sys.path.insert(0, '@CD_CODE@/engine/cb'); import prompt_tpl as PT  # noqa: E402  (prompts/)

R = M2.R; CK = R / 'checks'   # set by flexrun for the run tree
FONT = '@CD_CODE@/assets/fonts/NotoSansSC-Regular.otf'
COLORS = [(230, 25, 75), (60, 180, 75), (255, 200, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 220, 220), (240, 50, 230),
          (160, 200, 60), (0, 128, 128), (220, 190, 255), (170, 110, 40)]
FULL_FRAMES = (0, 60, 120)


def font(size):
    return ImageFont.truetype(FONT, size)


def clips(arg):
    return [int(x) for x in arg.split(',') if x]


def entities(n):
    return json.loads((R / 'plan' / str(n) / 'entities.json').read_text())


def percept_inputs(a):
    for n in clips(a.clips):
        src = R / 'percept' / str(n); per = json.loads((src / 'perception.json').read_text())
        ents = entities(n); d = CK / 'percept' / str(n); d.mkdir(parents=True, exist_ok=True)
        names = [e['name'] for e in per['entities']]; info = {e['id']: e for e in ents['entities']}
        frames = sorted(per['frames'], key=int)
        for k, name in enumerate(names):   # one strip per entity: 12 tiles, 6 per row
            tiles = []
            for f in frames:
                det = per['frames'][f].get(name); im = Image.open(M2.frame_png(n, int(f))).convert('RGB'); W, H = im.size
                if det:
                    b = det['box']; mx, my = (b[2] - b[0]) * 0.15, (b[3] - b[1]) * 0.15
                    crop = im.crop((int(max(0, b[0] - mx) * W), int(max(0, b[1] - my) * H), int(min(1, b[2] + mx) * W), int(min(1, b[3] + my) * H)))
                    crop.thumbnail((300, 260)); tile = Image.new('RGB', (300, 290), (245, 245, 245)); tile.paste(crop, ((300 - crop.width) // 2, 30))
                    txt = f'frame {int(f)} ({int(f) / 24:.2f}s)  score {det["score"]:.2f}'
                else:
                    tile = Image.new('RGB', (300, 290), (200, 200, 200)); txt = f'frame {int(f)} ({int(f) / 24:.2f}s)'
                    ImageDraw.Draw(tile).text((70, 140), 'not detected', fill=(90, 90, 90), font=font(26))
                ImageDraw.Draw(tile).text((6, 4), txt, fill=(0, 0, 0), font=font(18)); tiles.append(tile)
            rows = (len(tiles) + 5) // 6; strip = Image.new('RGB', (6 * 300 + 5 * 6, rows * 290 + (rows - 1) * 6 + 44), 'white')
            ImageDraw.Draw(strip).text((8, 6), f'{name}: detections of "{info.get(name, {}).get("text", name)}"', fill=(0, 0, 0), font=font(26))
            for i, t in enumerate(tiles):
                strip.paste(t, ((i % 6) * 306, 44 + (i // 6) * 296))
            strip.save(d / f'strip_{name}.jpg', quality=88)
        full = [str(f) for f in FULL_FRAMES if str(f) in per['frames']] if int(frames[-1]) <= 123 else [frames[0], frames[len(frames) // 2], frames[-1]]
        for f in full:   # full frames, every box with a large label (first / middle / last frame of a longer clip)
            im = Image.open(M2.frame_png(n, int(f))).convert('RGB'); W, H = im.size; dr = ImageDraw.Draw(im)
            for k, name in enumerate(names):
                det = per['frames'][f].get(name)
                if det:
                    b = det['box']; c = COLORS[k % len(COLORS)]; box = [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
                    dr.rectangle(box, outline=c, width=5); tw = dr.textlength(name, font=font(28))
                    y = max(0, box[1] - 38) if box[1] > 40 else box[1] + 4
                    dr.rectangle([box[0], y, box[0] + tw + 12, y + 36], fill=c); dr.text((box[0] + 6, y + 2), name, fill='white', font=font(28))
            im.save(d / f'frame_{int(f):03d}.jpg', quality=88)
        ent_lines = [PT.fill('agents/pcheck/entity_line.txt', ID=e['id'], TEXT=e.get('text', ''), APP=e.get('appearance', ''),
                             NOTES=PT.fill('agents/pcheck/entity_notes.txt', NOTES=e['notes']) if e.get('notes') else '') for e in ents['entities']]
        det_lines = [PT.fill('agents/pcheck/detection_line.txt', NAME=name, FRAMES=', '.join(f for f in frames if per['frames'][f].get(name))) for name in names]
        out = d / 'verdict.json'
        md = PT.lines('agents/pcheck/task.md', dict(ENT_LINES=ent_lines, DET_LINES=det_lines,
                      STRIPS=[PT.fill('agents/pcheck/strip_line.txt', PATH=d / ('strip_' + name + '.jpg'), NAME=name) for name in names],
                      FRAMES=[PT.fill('agents/pcheck/frame_line.txt', PATH=p) for p in sorted(d.glob('frame_*.jpg'))]), OUT=out, N4=f'{n:04d}')
        (d / 'agent_input.md').write_text(md + '\n'); print(n, len(names), 'entities ->', d / 'agent_input.md')


def percept_apply(a):
    for n in clips(a.clips):
        d = CK / 'percept' / str(n); v = json.loads((d / 'verdict.json').read_text())
        src = R / 'percept' / str(n); dst = R / 'percept' / f'{n}_checked'
        per = json.loads((src / 'perception.json').read_text()); new = json.loads(json.dumps(per)); log = []
        if dst.exists():
            shutil.rmtree(dst)
        (dst / 'masks').mkdir(parents=True); shutil.copytree(src / 'depth', dst / 'depth') if (src / 'depth').exists() else None
        if (src / 'sheet.jpg').exists():
            shutil.copy2(src / 'sheet.jpg', dst / 'sheet.jpg')  # the detector's overlay sheet (named in the planner prompt)
        names = [e['name'] for e in per['entities']]
        for f, row in per['frames'].items():
            moves = {}
            for name in names:
                if not row.get(name):
                    continue
                verdict = str(v.get(name, {}).get(str(int(f)), 'ok')).strip()
                if verdict == 'ok':
                    moves.setdefault(name, []).append(name)
                elif verdict.startswith('is:') and verdict[3:] in names:
                    moves.setdefault(verdict[3:], []).append(name); log.append(f'frame {f}: {name} -> {verdict[3:]}')
                else:
                    log.append(f'frame {f}: {name} dropped' + ('' if verdict == 'drop' else f' (unrecognised verdict {verdict!r})'))
            for name in names:   # a slot claimed twice keeps the detection that was its own, else the higher score
                cands = moves.get(name, [])
                pick = name if name in cands else (max(cands, key=lambda c: row[c]['score']) if cands else None)
                if len(cands) > 1:
                    log.append(f'frame {f}: {name} claimed by {cands}, kept {pick}')
                new['frames'][f][name] = row[pick] if pick else None
                mk = src / 'masks' / f'{int(f):03d}_{pick}.png'
                if pick and mk.exists():
                    shutil.copy2(mk, dst / 'masks' / f'{int(f):03d}_{name}.png')
        new['note'] = f'perception check agent ({d / "verdict.json"}): ' + (v.get('notes') or '') + f' | {len(log)} changes'
        (dst / 'perception.json').write_text(json.dumps(new, ensure_ascii=False, indent=1)); (d / 'applied.json').write_text(json.dumps(log, indent=1))
        print(n, len(log), 'changes', log[:12])
