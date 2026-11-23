"""LLM calls through the Anthropic Messages API (shared by every agent step of the pipeline).

Every agent task is a fresh, self-contained conversation (no memory, no project context): isolation by construction.
Configuration (environment variables; bin/configure.py writes defaults into config/cd.env):
    ANTHROPIC_API_KEY    API key (or LLM_API_TOKEN)                                    required
    CD_LLM_MESSAGES_URL  Messages endpoint             (default https://api.anthropic.com/v1/messages)
    CD_LLM_AUTH          x-api-key | bearer            (default x-api-key; bearer for compatible endpoints that expect "Authorization: Bearer")
    LLM_MODEL            model name                    (default claude-opus-5-5)
    LLM_MAX_CONCURRENCY  threads one process may use   (default 8)
    LLM_TIMEOUT          seconds per HTTP request      (default 900)
    LLM_RETRIES          attempts per call             (default 4)
    LLM_SLOT_DIR, LLM_TOTAL_SLOTS   machine-wide cap on requests in flight (flock slots; default /tmp/llm_api_slots, 16)
The credential is read from the environment only, never written to any file, log or transcript.

    resp, meta = complete_native(body, record=..., tag=...)   # Messages API (adaptive thinking, tool use, images)
"""
import fcntl, json, os, threading, time
from contextlib import contextmanager
from pathlib import Path
import requests

MODEL = os.environ.get('LLM_MODEL', 'claude-opus-5-5')
TIMEOUT = int(os.environ.get('LLM_TIMEOUT', 900))
RETRIES = int(os.environ.get('LLM_RETRIES', 4))
SLOT_DIR = Path(os.environ.get('LLM_SLOT_DIR', '/tmp/llm_api_slots'))
TOTAL_SLOTS = int(os.environ.get('LLM_TOTAL_SLOTS', 16))
_BASE = os.environ.get('ANTHROPIC_BASE_URL', '').rstrip('/')   # the same variables as the Anthropic SDK (used by the generation path)
MESSAGES_URL = os.environ.get('CD_LLM_MESSAGES_URL') or (f'{_BASE}/v1/messages' if _BASE else 'https://api.anthropic.com/v1/messages')
AUTH = os.environ.get('CD_LLM_AUTH') or ('bearer' if os.environ.get('ANTHROPIC_AUTH_TOKEN') and not os.environ.get('ANTHROPIC_API_KEY') else 'x-api-key')
_LOCK = threading.Lock(); _TOKEN = None


def max_concurrency():
    return int(os.environ.get('LLM_MAX_CONCURRENCY', 8))


def _token():
    global _TOKEN
    if _TOKEN is None:
        tok = os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_AUTH_TOKEN') or os.environ.get('LLM_API_TOKEN')
        if not tok:
            raise RuntimeError('set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN for a compatible endpoint) in the environment, see README "LLM API"')
        _TOKEN = tok
    return _TOKEN


def _headers(native):
    h = {'Content-Type': 'application/json'}
    if native:
        h['anthropic-version'] = '2023-06-01'
    if AUTH == 'bearer' or not native:
        h['Authorization'] = f'Bearer {_token()}'
    else:
        h['x-api-key'] = _token()
    return h


@contextmanager
def _slot():
    """hold one of TOTAL_SLOTS machine-wide flock slots for the duration of one HTTP request"""
    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(TOTAL_SLOTS):
            f = open(SLOT_DIR / f'slot{i:02d}', 'a')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close(); continue
            try:
                yield i
            finally:
                fcntl.flock(f, fcntl.LOCK_UN); f.close()
            return
        time.sleep(1)


def complete_native(body, *, record, tag, tries=None, timeout=None, stream=True, idle=None):
    """One Anthropic Messages API call (native content blocks: thinking with signatures, tool_use / tool_result, images in tool
    results). body is the full request (model, max_tokens, messages, tools, thinking, output_config). Returns (response, meta);
    same slots, retries and per-attempt record as complete(). stream=True (default) reads the answer as server-sent events and
    rebuilds the same response object: an intermediary may cut a request that has sent nothing for minutes, which a
    long turn (thinking plus writing a large file in one tool call) exceeds; a stream keeps the connection busy.
    idle: optional seconds without a single byte after which a stream is given up and retried (streams only: a non-streamed reply
    sends nothing until it is complete; default None: the read timeout is
    `timeout`). A healthy stream sends events every few seconds, thinking included; a stalled request sends nothing at all."""
    tries = tries or RETRIES; timeout = timeout or TIMEOUT; last = None
    for attempt in range(tries):
        t0 = time.time(); meta = dict(tag=tag, attempt=attempt, protocol='anthropic', effort=(body.get('output_config') or {}).get('effort'),
                                      max_tokens=body.get('max_tokens'), stream=stream)
        try:
            with _slot():
                r = requests.post(MESSAGES_URL, headers=_headers(True), data=json.dumps(dict(body, stream=True) if stream else body),
                                  timeout=(60, min(timeout, idle) if idle and stream else timeout), stream=stream)
                meta.update(status=r.status_code)
                if r.status_code == 200:
                    d = _read_stream(r, timeout) if stream else r.json()
            meta.update(seconds=round(time.time() - t0, 1))
            if r.status_code == 200:
                meta.update(ok=bool(d.get('content')), id_prefix=str(d.get('id', ''))[:8], backend=d.get('model'), finish=d.get('stop_reason'),
                            usage=d.get('usage'))
                if meta['ok']:
                    _log(record, meta); return d, meta
                last = f"empty reply (stop {d.get('stop_reason')}{', ' + d['_error'] if d.get('_error') else ''})"
            else:
                last = f'HTTP {r.status_code}: {r.text[:300]}'
        except Exception as e:
            meta.update(seconds=round(time.time() - t0, 1)); last = f'{type(e).__name__}: {e}'[:300]
        meta.update(ok=False, error=last); _log(record, meta)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'{tag}: no reply after {tries} attempts: {last}')


def _read_stream(r, timeout):
    """rebuild a Messages response from its server-sent events (message_start, content_block_start / _delta / _stop,
    message_delta, message_stop); an error event or a cut stream gives an empty content (the caller retries)"""
    msg, blocks, parts = {}, {}, {}; t0 = time.time()
    r.encoding = 'utf-8'   # text/event-stream without a charset would be decoded as ISO-8859-1 and break non-ASCII text and thinking
    for raw in r.iter_lines(decode_unicode=True):
        if time.time() - t0 > timeout:
            raise TimeoutError(f'stream longer than {timeout} s')
        if not raw or not raw.startswith('data:'):
            continue
        ev = json.loads(raw[5:].strip()); kind = ev.get('type')
        if kind == 'message_start':
            msg = ev['message']; msg['content'] = []
        elif kind == 'content_block_start':
            blocks[ev['index']] = dict(ev['content_block']); parts[ev['index']] = []
        elif kind == 'content_block_delta':
            b, dl = blocks[ev['index']], ev['delta']; t = dl.get('type')
            if t == 'text_delta':
                b['text'] = b.get('text', '') + dl['text']
            elif t == 'thinking_delta':
                b['thinking'] = b.get('thinking', '') + dl['thinking']
            elif t == 'signature_delta':
                b['signature'] = b.get('signature', '') + dl['signature']
            elif t == 'input_json_delta':
                parts[ev['index']].append(dl.get('partial_json', ''))
        elif kind == 'content_block_stop':
            b = blocks[ev['index']]
            if b.get('type') == 'tool_use':
                js = ''.join(parts[ev['index']])
                try:
                    b['input'] = json.loads(js) if js.strip() else (b.get('input') or {})
                except ValueError:   # cut at max_tokens: the caller sees stop_reason and does not run it (no extra field: blocks go back as they are)
                    b['input'] = {}
        elif kind == 'message_delta':
            msg.update({k: v for k, v in (ev.get('delta') or {}).items() if v is not None})
            msg['usage'] = dict(msg.get('usage') or {}, **(ev.get('usage') or {}))
        elif kind == 'error':
            msg['_error'] = str(ev.get('error'))[:200]; break
        elif kind == 'message_stop':
            break
    if msg.get('_error') or not msg.get('stop_reason'):
        msg['content'] = []; return msg   # incomplete: treated as an empty reply and retried
    msg['content'] = [blocks[i] for i in sorted(blocks)]
    return msg


def _log(record, meta):
    with _LOCK:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        with open(record, 'a') as f:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')


def image_part_native(path):
    """One image content block in the native Messages format (base64); png / jpg / webp."""
    import base64
    p = Path(path); mime = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.webp': 'webp'}.get(p.suffix.lower(), 'png')
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': f'image/{mime}', 'data': base64.b64encode(p.read_bytes()).decode()}}


