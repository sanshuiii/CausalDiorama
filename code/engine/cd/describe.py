"""Source description for editing without a prompt (as for OpenVE-Bench, which has no source descriptions): Claude Opus sees 6 frames of the source only (not the instruction), sampled at the middle of 6 equal parts, each
at most 768 px (JPEG quality 88), and writes about 40 words in the style of three example descriptions
(prompts/agents/describe_source.md, verbatim). One request, max_tokens 1500."""
import base64, io, subprocess, sys
from pathlib import Path

from . import paths

sys.path.insert(0, str(paths.ENGINE / 'llm'))


def frames(video, dur, k=6):
    from PIL import Image
    out = []
    for j in range(k):
        t = dur * (j + 0.5) / k
        png = subprocess.run([paths.FFMPEG, '-loglevel', 'error', '-ss', f'{t:.3f}', '-i', str(video), '-frames:v', '1', '-f', 'image2pipe',
                              '-vcodec', 'png', '-'], capture_output=True).stdout
        im = Image.open(io.BytesIO(png)).convert('RGB'); im.thumbnail((768, 768)); buf = io.BytesIO(); im.save(buf, 'JPEG', quality=88)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def describe(video, dur, record):
    import api_llm
    text = (paths.PROMPTS / 'agents' / 'describe_source.md').read_text()
    content = [{'type': 'text', 'text': text}] + [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': b}}
                                                 for b in frames(video, dur)]
    body = {'model': api_llm.MODEL, 'max_tokens': 1500, 'messages': [{'role': 'user', 'content': content}]}
    d, meta = api_llm.complete_native(body, record=record, tag='describe')
    out = ''.join(b.get('text', '') for b in d.get('content', []) if b.get('type') == 'text').strip()
    if not out:
        raise SystemExit(f'the description request returned no text (stop {d.get("stop_reason")})')
    return out
