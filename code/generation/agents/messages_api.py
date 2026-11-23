"""Run one context-free agent task over the Anthropic Messages API (official Python SDK).

The conversation starts with a single user message (the task text points to task.md) and no system prompt. The model gets the
four tools of sandbox.py, adaptive thinking and the given effort; every assistant turn, thinking blocks included, is sent back
unchanged. Images the agent reads are returned inside the tool result. A turn cut at max_tokens does not run its last
(incomplete) tool call; the model is told so. Credentials come from the environment (ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN
with ANTHROPIC_BASE_URL for a compatible endpoint). A full transcript is written as JSON (images as path + sha256).
"""
import hashlib, json, time
from pathlib import Path

import anthropic

from .sandbox import Sandbox, TOOL_SPECS, image_block, run_tool

MODEL = 'claude-opus-5-5'


def _log_block(block):
    """A content block for the transcript: thinking signatures by length only, images by path and hash."""
    if block.get('type') == 'thinking':
        return dict(type='thinking', thinking=block.get('thinking', ''), signature=f"<{len(block.get('signature') or '')} chars>")
    if block.get('type') == 'redacted_thinking':
        return dict(type='redacted_thinking', data=f"<{len(block.get('data') or '')} chars>")
    return block


def _image_ref(path):
    return {'type': 'image_ref', 'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def run(task, *, workdir, read_roots, write_roots, bash_prefixes, transcript, bash_env=None, model=MODEL, effort='medium',
        max_tokens=32000, max_turns=400):
    """Returns (final text, info). Raises RuntimeError on a refusal or when no final answer comes within max_turns; API errors
    propagate after the SDK's retries."""
    client = anthropic.Anthropic(max_retries=4)
    sb = Sandbox(workdir, read_roots, write_roots, bash_prefixes, bash_env)
    messages = [{'role': 'user', 'content': task}]; log = [{'role': 'user', 'content': task}]
    turns, usage, t0 = [], {}, time.time()

    def save(status):
        Path(transcript).parent.mkdir(parents=True, exist_ok=True)
        Path(transcript).write_text(json.dumps(dict(status=status, model=model, effort=effort, seconds=round(time.time() - t0, 1), usage=usage,
                                                    denied=sb.denied, turns=turns, messages=log), ensure_ascii=False, indent=1))

    for turn in range(max_turns):
        req = dict(model=model, max_tokens=max_tokens, messages=messages, tools=TOOL_SPECS, thinking={'type': 'adaptive'},
                   output_config={'effort': effort}, cache_control={'type': 'ephemeral'})
        t1 = time.time()
        try:
            with client.messages.stream(**req) as stream:
                resp = stream.get_final_message()
        except anthropic.APIError:
            save('api_error'); raise
        for k, v in resp.usage.model_dump().items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        content = [b.model_dump(exclude_none=True) for b in resp.content]
        messages.append({'role': 'assistant', 'content': resp.content}); log.append({'role': 'assistant', 'content': [_log_block(b) for b in content]})
        text = ''.join(b.get('text', '') for b in content if b.get('type') == 'text'); calls = [b for b in content if b.get('type') == 'tool_use']
        rec = dict(turn=turn, seconds=round(time.time() - t1, 1), model=resp.model, stop_reason=resp.stop_reason, usage=resp.usage.model_dump(),
                   thinking_blocks=sum(b.get('type') in ('thinking', 'redacted_thinking') for b in content), text=text[:500], calls=[])
        if resp.stop_reason == 'refusal':
            turns.append(rec); save('refusal')
            raise RuntimeError(f'the model declined the task (stop_details: {resp.stop_details})')
        if not calls:
            turns.append(rec); save('done')
            return text, dict(turns=turn + 1, usage=usage, denied=sb.denied, seconds=round(time.time() - t0, 1), stop_reason=resp.stop_reason,
                              tools={k: sum(1 for t in turns for c in t['calls'] if c['name'] == k) for k in ('Read', 'Write', 'Edit', 'Bash')})
        results, logged = [], []
        cut = resp.stop_reason == 'max_tokens'
        for c in calls:
            if cut and c is calls[-1]:
                out, img = 'Error: the reply reached the output token limit (max_tokens) before this tool call was complete, so it was not run.', None
            else:
                out, img = run_tool(sb, c['name'], c.get('input'))
            if img is not None:
                results.append({'type': 'tool_result', 'tool_use_id': c['id'], 'content': [{'type': 'text', 'text': out}, image_block(img)]})
                logged.append({'type': 'tool_result', 'tool_use_id': c['id'], 'content': [{'type': 'text', 'text': out}, _image_ref(img)]})
            else:
                results.append({'type': 'tool_result', 'tool_use_id': c['id'], 'content': out}); logged.append(results[-1])
            args = c.get('input') if isinstance(c.get('input'), dict) else {}
            rec['calls'].append(dict(name=c['name'], args={k: (v if len(str(v)) < 300 else str(v)[:300] + '...') for k, v in args.items()},
                                     result=str(out)[:300]))
        messages.append({'role': 'user', 'content': results}); log.append({'role': 'user', 'content': logged})
        turns.append(rec); save('running')
    save('max_turns')
    raise RuntimeError(f'no final answer after {max_turns} turns')
