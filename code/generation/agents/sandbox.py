"""Tools of a context-free planning agent, with Claude Code semantics.

    Read   file_path, offset, limit   text as `cat -n` (2000 lines by default, long lines cut); png / jpg / webp come back as images
    Write  file_path, content         a file that already exists must have been read first
    Edit   file_path, old_string, new_string, replace_all   exact replacement, old_string unique unless replace_all; read first
    Bash   command, timeout (ms)      only commands that start with one of the allowed prefixes (the checker command) and contain
                                      none of ; & | ` $ ( ) < > or a newline; output cut at 30000 characters

Reads are limited to the readable roots (which include the writable ones), writes and edits to the writable roots. A refused call
returns an error text to the model and is recorded in `denied`.
"""
import base64, os, re, shlex, signal, site, subprocess
from pathlib import Path

IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.webp'}

TOOL_SPECS = [
    {'name': 'Read', 'description': 'Reads a file from the local filesystem. file_path must be absolute. Text is returned in cat -n format '
     '(line numbers from 1), up to 2000 lines by default; use offset and limit for long files. Images (png, jpg, webp) are shown to you.',
     'input_schema': {'type': 'object', 'properties': {
         'file_path': {'type': 'string', 'description': 'The absolute path to the file to read'},
         'offset': {'type': 'integer', 'description': 'The line number to start reading from'},
         'limit': {'type': 'integer', 'description': 'The number of lines to read'}}, 'required': ['file_path']}},
    {'name': 'Write', 'description': 'Writes a file to the local filesystem, overwriting it if it exists. file_path must be absolute. '
     'An existing file must be read with Read first.',
     'input_schema': {'type': 'object', 'properties': {
         'file_path': {'type': 'string', 'description': 'The absolute path to the file to write'},
         'content': {'type': 'string', 'description': 'The content to write to the file'}}, 'required': ['file_path', 'content']}},
    {'name': 'Edit', 'description': 'Performs exact string replacement in a file. The file must have been read with Read first. old_string '
     'must match the file exactly and be unique, unless replace_all is true.',
     'input_schema': {'type': 'object', 'properties': {
         'file_path': {'type': 'string', 'description': 'The absolute path to the file to modify'},
         'old_string': {'type': 'string', 'description': 'The text to replace'},
         'new_string': {'type': 'string', 'description': 'The text to replace it with (must be different from old_string)'},
         'replace_all': {'type': 'boolean', 'description': 'Replace all occurrences of old_string (default false)'}},
         'required': ['file_path', 'old_string', 'new_string']}},
    {'name': 'Bash', 'description': 'Executes a bash command and returns its output. The working directory persists between calls. '
     'timeout is in milliseconds (default 120000, max 600000).',
     'input_schema': {'type': 'object', 'properties': {
         'command': {'type': 'string', 'description': 'The command to execute'},
         'timeout': {'type': 'number', 'description': 'Optional timeout in milliseconds (max 600000)'},
         'description': {'type': 'string', 'description': 'What the command does'}}, 'required': ['command']}},
]
TOOL_ARGS = {'Read': ('file_path', 'offset', 'limit'), 'Write': ('file_path', 'content'),
             'Edit': ('file_path', 'old_string', 'new_string', 'replace_all'), 'Bash': ('command', 'timeout', 'description')}


def clean_env(extra=None):
    """Environment of the agent's commands: the parent's without credentials and session variables (CLAUDE*, ANTHROPIC*, names
    holding TOKEN / KEY / SECRET / PASSWORD, PYTHONPATH, PWD, OLDPWD). HOME moves to the task directory, so PYTHONUSERBASE is pinned
    to the parent's user base to keep user-installed packages importable."""
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith('CLAUDE') or k.startswith('ANTHROPIC') or any(w in k.upper() for w in ('TOKEN', 'KEY', 'SECRET', 'PASSWORD'))
                   or k in ('PYTHONPATH', 'PWD', 'OLDPWD'))}
    env.setdefault('PYTHONUSERBASE', site.getuserbase())
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def image_block(path):
    """An image content block (base64) for a Messages API request."""
    p = Path(path); mime = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.webp': 'webp'}.get(p.suffix.lower(), 'png')
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': f'image/{mime}', 'data': base64.b64encode(p.read_bytes()).decode()}}


class Sandbox:
    def __init__(self, workdir, read_roots, write_roots, bash_prefixes, bash_env=None):
        self.cwd = Path(workdir).resolve(); self.cwd.mkdir(parents=True, exist_ok=True)
        self.write = [Path(p).resolve() for p in write_roots]
        self.read = [Path(p).resolve() for p in read_roots] + self.write
        self.prefixes = list(bash_prefixes); self.seen = set(); self.denied = []
        self.env = clean_env(dict({'HOME': str(self.cwd)}, **(bash_env or {})))

    def _under(self, p, roots):
        p = Path(p).resolve()
        return any(p == r or r in p.parents for r in roots)

    def _abs(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.cwd / p

    def deny(self, tool, why):
        self.denied.append(dict(tool=tool, why=why[:300])); return f'Error: refused by the sandbox: {why}'

    def Read(self, file_path, offset=None, limit=None):
        p = self._abs(file_path)
        if not self._under(p, self.read):
            return self.deny('Read', f'{p} is outside the readable paths'), None
        if p.is_dir():
            return f'Error: {p} is a directory', None
        if not p.exists():
            return f'Error: file does not exist: {p}', None
        self.seen.add(str(p.resolve()))
        if p.suffix.lower() in IMAGE_EXT:
            return f'[image {p} shown below]', p
        lines = p.read_text(errors='replace').splitlines()
        if not lines:
            return '<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>', None
        start = max(int(offset or 1), 1) - 1; n = int(limit or 2000)
        return ''.join(f'{i + 1:6d}\t{l[:2000]}\n' for i, l in enumerate(lines[start:start + n], start)), None

    def Write(self, file_path, content):
        p = self._abs(file_path)
        if not self._under(p, self.write):
            return self.deny('Write', f'{p} is outside the writable paths')
        if p.exists() and str(p.resolve()) not in self.seen:
            return 'Error: File has not been read yet. Read it first before writing to it.'
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content); self.seen.add(str(p.resolve()))
        return f'File created successfully at: {p}'

    def Edit(self, file_path, old_string, new_string, replace_all=False):
        p = self._abs(file_path)
        if not self._under(p, self.write):
            return self.deny('Edit', f'{p} is outside the writable paths')
        if str(p.resolve()) not in self.seen:
            return 'Error: File has not been read yet. Read it first before writing to it.'
        if old_string == new_string:
            return 'Error: old_string and new_string are exactly the same.'
        s = p.read_text(); k = s.count(old_string)
        if k == 0:
            return f'Error: String to replace not found in file.\nString: {old_string}'
        if k > 1 and not replace_all:
            return f'Error: Found {k} matches of the string to replace, but replace_all is false. Provide more context to make it unique.'
        p.write_text(s.replace(old_string, new_string) if replace_all else s.replace(old_string, new_string, 1))
        return f'The file {p} has been updated successfully.'

    def Bash(self, command, timeout=None, description=None):
        c = command.strip()
        if not any(c.startswith(x) for x in self.prefixes):
            return self.deny('Bash', f'only commands starting with {self.prefixes} are allowed')
        if re.search(r'[;&|`$()<>\n]', c):
            return self.deny('Bash', 'the command may not contain ; & | ` $ ( ) < > or a newline')
        t = min(float(timeout or 120000), 600000) / 1000; cwdf = self.cwd / '.agent_cwd'
        wrapped = f'cd {shlex.quote(str(self.cwd))} && {{\n{command}\n}}; __rc=$?; pwd > {shlex.quote(str(cwdf))}; exit $__rc'
        pr = subprocess.Popen(['bash', '-c', wrapped], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace',
                              env=self.env, start_new_session=True)   # own process group: a timeout also stops the checker's children
        try:
            out, _ = pr.communicate(timeout=t)
            if pr.returncode:
                out += f'\n(exit code {pr.returncode})'
        except subprocess.TimeoutExpired:
            os.killpg(pr.pid, signal.SIGKILL); out, _ = pr.communicate(); out = (out or '') + f'\nCommand timed out after {t:.0f} s'
        try:
            nc = Path(cwdf.read_text().strip())
            if self._under(nc, self.read):
                self.cwd = nc
            cwdf.unlink()
        except OSError:
            pass
        out = out if len(out) <= 30000 else out[:30000] + f'\n... (output truncated, {len(out)} characters in total)'
        return out or '(no output)'


def run_tool(sb, name, args):
    """One tool call -> (result text, image path or None); any error goes back to the model as the result, as in Claude Code."""
    keys = TOOL_ARGS.get(name); img = None
    try:
        if keys is None:
            out = f'Error: no such tool {name}'
        elif not isinstance(args, dict):
            out = 'Error: tool input must be an object'
        elif name == 'Read':
            out, img = sb.Read(**{k: v for k, v in args.items() if k in keys})
        else:
            out = getattr(sb, name)(**{k: v for k, v in args.items() if k in keys})
    except Exception as e:
        out = f'Error: {type(e).__name__}: {e}'
    if not str(out).strip():   # an empty tool result is rejected by the API
        out = '(no output)'
    return out, img
