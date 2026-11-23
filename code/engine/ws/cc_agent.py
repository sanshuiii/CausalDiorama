"""Replica of a Claude Code general-purpose subagent on the Anthropic Messages API (CD_AGENT_MODE=api, the default agent mode).

It sends what a Claude Code subagent receives (prompts/cc/profile.json, prompts/cc/tools.json):
  - system: Claude Code's four system-prompt parts (prompt_snapshot), the last one a <total_tokens> line
  - first user message: the spawn message, then the rendered context attachments in their order (deferred-tools list, the
    SubagentHandback reminder, environment, model, skill listing, date, attribution note); after the first tool results the agent
    listing, MCP instructions and the auto-mode note; a <total_tokens> reminder after every tool-result turn
    (left out: session_context and credential_org, personal / account data that is never sent)
  - tools: Bash / Edit / Read / Write with Claude Code's descriptions and schemas, plus SubagentHandback (the run ends with it)
  - tool results in Claude Code's formats: Read text as "N<TAB>line", images alone (a PNG over 1 MB as a JPEG, quality 90, 4:2:0,
    the same bytes Claude Code sent), Write / Edit with "(file state is current ...)", errors as <tool_use_error> with is_error,
    Bash failures as "Exit code N" with is_error, "(Bash completed with no output)"
  - native Messages API, adaptive thinking (signed thinking blocks sent back unchanged), the run's effort, prompt cache (5 min)
Tools run in api_agent's Sandbox (the same read / write roots and Bash whitelist as the other API agents).
Transport: engine/llm/api_llm.py (endpoint, credential, concurrency from the environment).
"""
import copy, hashlib, io, json, os, re, sys, time
from pathlib import Path

import api_llm, api_agent

HERE = Path(__file__).resolve().parent
PROFILE = json.loads((Path('@CD_CODE@/prompts/cc') / 'profile.json').read_text())
TOOLS = [{k: ({kk: vv for kk, vv in v.items() if kk != '$schema'} if k == 'input_schema' else v) for k, v in t.items()}
         for t in json.loads((Path('@CD_CODE@/prompts/cc') / 'tools.json').read_text())['tools']]
SUFFIX = PROFILE['write_suffix']
IDLE = 300   # s without a byte: a stalled stream is retried (healthy streams send events every few seconds)
sys.path.insert(0, '@CD_CODE@/engine/cb'); import prompt_tpl as PT  # noqa: E402  (prompts/)
ENFORCE = PT.text('cc/handback_enforce.txt')   # Claude Code's reminder text, verbatim
PNG_TO_JPEG = 1 << 20   # bytes: larger PNGs are sent as JPEG, as Claude Code does


def _text(t):
    return {'type': 'text', 'text': t}


def _image(path):
    """the image block Claude Code's Read returns: large PNGs re-encoded as JPEG (quality 90, 4:2:0), everything else as is"""
    import base64
    p = Path(path); data = p.read_bytes(); ext = p.suffix.lower()
    mime = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}.get(ext, 'image/png')
    if ext == '.png' and len(data) > PNG_TO_JPEG:
        from PIL import Image
        b = io.BytesIO(); Image.open(io.BytesIO(data)).convert('RGB').save(b, 'JPEG', quality=90); data = b.getvalue(); mime = 'image/jpeg'
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': mime, 'data': base64.b64encode(data).decode()}}


def _err(msg):
    return f'<tool_use_error>{msg}</tool_use_error>', True


def _tool(sb, name, args):
    """one tool call -> (content for the tool_result, is_error, image path or None, ends the run)"""
    if not isinstance(args, dict):
        return (*_err('Error: tool input must be an object'), None, False)
    if name == 'SubagentHandback':
        return PROFILE['handback_result'], False, None, True
    if name == 'Read':
        out, img = sb.Read(args.get('file_path', ''), args.get('offset'), args.get('limit'))
        if img is not None:
            return [_image(img)], False, img, False
        if out.startswith('Error: '):
            m = out[7:]
            return (*_err('File does not exist.' if m.startswith('file does not exist') else m), None, False)
        return re.sub(r'(?m)^ +(\d+)\t', r'\1\t', out), False, None, False
    if name in ('Write', 'Edit'):
        p = sb._abs(args.get('file_path', '')); existed = p.exists()
        keys = ('file_path', 'content') if name == 'Write' else ('file_path', 'old_string', 'new_string', 'replace_all')
        try:
            out = getattr(sb, name)(**{k: v for k, v in args.items() if k in keys})
        except Exception as e:
            return (*_err(f'{type(e).__name__}: {e}'), None, False)
        if out.startswith('Error: '):
            return (*_err(out[7:]), None, False)
        if name == 'Write' and existed:
            out = f'The file {p} has been updated successfully.'
        return out + SUFFIX, False, None, False
    if name == 'Bash':
        if args.get('run_in_background') or args.get('dangerouslyDisableSandbox'):
            return (*_err('This environment runs commands in the foreground and sandboxed only.'), None, False)
        out = sb.Bash(args.get('command', ''), args.get('timeout'), args.get('description'))
        if out.startswith('Error: refused by the sandbox'):
            return (*_err(out[7:]), None, False)
        m = re.search(r'\n\(exit code (\d+)\)$', out)
        if m:
            return f'Exit code {m.group(1)}\n' + out[:m.start()], True, None, False
        if out == '(no output)':
            return '(Bash completed with no output)', False, None, False
        return out, False, None, False
    return (*_err(f'Error: No such tool available: {name}'), None, False)


def _with_cache(system, messages, ttl):
    """cache breakpoints as Claude Code sets them: the end of the system prompt and the end of the last two user turns"""
    cc = {'type': 'ephemeral', 'ttl': ttl}
    sysb = [dict(b) for b in system]; sysb[-1]['cache_control'] = cc
    out = list(messages); k = 0
    for i in range(len(out) - 1, -1, -1):
        if out[i]['role'] == 'user' and k < 2:
            c = [dict(b) for b in out[i]['content']]; c[-1] = dict(c[-1], cache_control=cc); out[i] = dict(out[i], content=c); k += 1
    return sysb, out


def run_agent(spawn, *, workdir, read_roots, write_roots, transcript, record, tag, bash_allow, py_scripts, effort, max_tokens=128000,
              cache='5m', max_turns=400, bash_prefixes=None, bash_env=None):
    sb = api_agent.Sandbox(workdir, read_roots, write_roots, bash_allow, py_scripts, True, bash_prefixes, bash_env)
    left = int(re.search(r'\d+', PROFILE['total_tokens_reminder']).group())
    system = [_text(p) for p in PROFILE['system']]
    first = [_text(spawn)] + [_text(a['text'] if a['kind'] != 'date' else f"<system-reminder>\nToday's date is {time.strftime('%Y-%m-%d')}.\n</system-reminder>")
                              for a in PROFILE['first_user']]
    messages = [{'role': 'user', 'content': first}]; log = [{'role': 'system', 'content': PROFILE['system']}, {'role': 'user', 'content': first}]
    turns = []; usage = {}; t0 = time.time(); report = None; nudged = False

    def save(status):
        Path(transcript).parent.mkdir(parents=True, exist_ok=True)
        s = json.dumps(dict(tag=tag, status=status, protocol='anthropic', profile='claude_code', seconds=round(time.time() - t0, 1), usage=usage,
                            denied=sb.denied, turns=turns, messages=log), ensure_ascii=False, indent=1)
        tok = api_llm._TOKEN or os.environ.get('LLM_API_TOKEN')
        Path(transcript).write_text(s.replace(tok, '[REDACTED]') if tok else s)

    def info(meta, n):
        return dict(turns=n, usage=usage, denied=sb.denied, seconds=round(time.time() - t0, 1), finish=meta.get('finish'), report=report,
                    tools={k: sum(1 for t in turns for c in t['calls'] if c['name'] == k) for k in ('Read', 'Write', 'Edit', 'Bash', 'SubagentHandback')})

    for turn in range(max_turns):
        sysb, msgs = _with_cache(system, messages, cache) if cache else (system, messages)
        body = dict(model=api_llm.MODEL, max_tokens=max_tokens, system=sysb, messages=msgs, tools=TOOLS, thinking={'type': 'adaptive'},
                    output_config={'effort': effort})
        try:
            d, meta = api_llm.complete_native(body, record=record, tag=tag, idle=IDLE)
        except Exception:
            save('api_error'); raise
        u = meta.get('usage') or {}
        for k, v in u.items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        left -= sum(u.get(k) or 0 for k in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'))
        content = d.get('content') or []
        messages.append({'role': 'assistant', 'content': content}); log.append({'role': 'assistant', 'content': [api_agent._native_log(b) for b in content]})
        text = ''.join(b.get('text', '') for b in content if b.get('type') == 'text'); calls = [b for b in content if b.get('type') == 'tool_use']
        rec = dict(turn=turn, seconds=meta.get('seconds'), backend=meta.get('backend'), finish=meta.get('finish'), usage=u,
                   thinking_blocks=sum(b.get('type') in ('thinking', 'redacted_thinking') for b in content), text=text[:500], calls=[])
        if not calls:
            turns.append(rec)
            if not nudged and d.get('stop_reason') == 'end_turn':   # Claude Code's reminder when a subagent ends without its handback
                nudged = True; messages.append({'role': 'user', 'content': [_text(ENFORCE)]}); log.append(messages[-1]); save('running'); continue
            save('done')
            return text, info(meta, turn + 1)
        results, logged, ends = [], [], False
        cut = d.get('stop_reason') == 'max_tokens'
        for c in calls:
            if cut and c is calls[-1]:
                out, is_err, img, end = (*_err('The reply reached the output token limit before this tool call was complete; it was not run.'), None, False)
            else:
                out, is_err, img, end = _tool(sb, c['name'], c.get('input'))
            if end:
                report = (c.get('input') or {}).get('message'); ends = True
            r = {'type': 'tool_result', 'tool_use_id': c['id'], 'content': out}
            if is_err:
                r['is_error'] = True
            results.append(r)
            logged.append(dict(r, content=[api_agent._img_ref(None, img)]) if img is not None else r)
            rec['calls'].append(dict(name=c['name'], args={k: (v if len(str(v)) < 300 else str(v)[:300] + '...') for k, v in (c.get('input') if isinstance(c.get('input'), dict) else {}).items()},
                                     result=(str(out)[:300] if img is None else f'[image {img}]'), error=is_err))
        turns.append(rec)
        if ends:   # SubagentHandback ends the run (Claude Code: toolEndsTurn), its report is the result
            messages.append({'role': 'user', 'content': results}); log.append({'role': 'user', 'content': logged}); save('done')
            return report or text, info(meta, turn + 1)
        extra = [a['text'] for a in PROFILE['after_first_tool_results']] if turn == 0 else []
        extra.append(f'<system-reminder>\n<total_tokens>{left} tokens left</total_tokens>\n</system-reminder>')
        messages.append({'role': 'user', 'content': results + [_text(x) for x in extra]})
        log.append({'role': 'user', 'content': logged + [_text(x) for x in extra]}); save('running')
    save('max_turns')
    raise RuntimeError(f'{tag}: no final answer after {max_turns} turns')
