"""Run one context-free agent task as an isolated Claude Code process (`claude -p`).

The process starts in the task's own directory with the task text pointing to task.md in that directory, the tools Read, Write,
Edit and Bash, writes and edits allowed only inside the directory, Bash allowed only for the given command prefix, reads of the
listed directories refused, no MCP servers, and an environment without the parent's session variables. The CLI is taken from
CLAUDE_CLI (default `claude` on PATH).
"""
import json, os, subprocess, time

MODEL = 'claude-opus-5-5'


def child_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDE') and k not in ('PWD', 'OLDPWD', 'PYTHONPATH')}
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', DISABLE_AUTOUPDATER='1')
    return env


def run(ask, *, workdir, bash_prefix, deny_read, model=MODEL, effort='medium', timeout=3600):
    """Returns the CLI's JSON result (session_id, is_error, num_turns, result, usage, total_cost_usd) plus seconds."""
    d = str(workdir)
    cmd = [os.environ.get('CLAUDE_CLI', 'claude'), '-p', ask, '--model', model, '--effort', effort, '--tools', 'Read', 'Write', 'Edit', 'Bash',
           '--allowedTools', f'Write(/{d}/**)', f'Edit(/{d}/**)', f'Bash({bash_prefix}:*)',
           '--disallowedTools', *[f'Read(/{p}/**)' for p in deny_read], '--strict-mcp-config', '--output-format', 'json']
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout, env=child_env())
        res = json.loads(p.stdout) if p.stdout.strip().startswith('{') else {'is_error': True, 'result': (p.stdout + p.stderr)[-600:]}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        res = {'is_error': True, 'result': f'{type(e).__name__}: {e}'}
    res['seconds'] = round(time.time() - t0, 1)
    return res
