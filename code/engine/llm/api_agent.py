"""Context-free tool-using agent on the Anthropic Messages API (api_llm.py).

Each task is one fresh conversation whose first user message is the task text
(for example "Your whole task is written in <file>. Read that file and follow it exactly."), with no system prompt and no
other context. The model gets four tools with Claude Code semantics:
    Read   file_path, offset, limit   text as `cat -n` (2000 lines by default, long lines cut); png / jpg / webp are shown as images
                                      in a user message right after the tool results
    Write  file_path, content         a file that already exists must have been read first
    Edit   file_path, old_string, new_string, replace_all   exact replacement, old_string unique unless replace_all; read first
    Bash   command, timeout (ms)      bash in the task's working directory (the directory persists between calls), output cut at
                                      30000 characters, timeout 120 s by default, 600 s at most
Isolation: reads only under read_roots or write_roots, writes / edits only under write_roots.
Bash runs only whitelisted programs (`bash_allow`); every path in a command (absolute, or relative to the current directory,
`cd` targets included) must lie under the readable paths, and redirection targets (`> f`) under the writable ones. python3 runs
only the scripts in `py_scripts`: a bare name runs a script of that name lying under the readable paths; an absolute path runs
exactly that file even when it is not readable (run but not read, e.g. a neutral checker copy). With py_inline=True (default)
`python3 -c`, `python3 -` and heredocs are allowed as well; py_inline=False refuses them so that only the listed scripts run.
bash_env adds variables to the Bash environment, which is the parent's minus credentials and session plumbing (clean_env).
bash_prefixes (default None) switches Bash to Claude Code prefix rules instead, e.g. `Bash(python3 /x/scm_lint.py:*)`: the
stripped command must start with one of the prefixes and contain none of ; & | ` $ ( ) < > or a newline; the program whitelist
and the path checks are then not applied (the prefix fixes what runs).
NOTE: with py_inline=True (or any program that can open files by computed names) these checks are a policy constraint, not a
security boundary; they are audited afterwards through the logged refusals (`denied`) and the transcripts.
A refused call returns an error to the model and is logged.
Every task keeps a full transcript (JSON, images replaced by path + sha256, no credential) with each turn's tool calls,
result summaries, latency, usage and the backend that served it.

    from api_agent import run_agent
    text, info = run_agent(task_text, workdir=..., read_roots=[...], write_roots=[...], bash_allow=BASIC_BASH | {...},
                           py_scripts={'scm_lint.py', '/abs/neutral/compile_world.py'}, py_inline=True,   # or bash_prefixes=[...]
                           transcript=path, record=path, tag='planner 0042',
                           effort='medium', max_tokens=32000)
"""
import hashlib, json, os, re, shlex, signal, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import api_llm  # noqa: E402

IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.webp'}
BASIC_BASH = {'cat', 'ls', 'head', 'tail', 'wc', 'echo', 'printf', 'grep', 'sed', 'sort', 'uniq', 'cut', 'find', 'cd', 'pwd', 'true',
              'false', 'test', '[', 'cp', 'mkdir', 'python3', 'python', 'tr', 'basename', 'dirname', 'seq', 'diff', 'paste', 'realpath'}
# not awk (system(), -f / -i files and gawk's indirect calls run commands unchecked); sed / find / cp / mkdir / sort / uniq are
# restricted below (sed e / w / s///e / s///w / -f, find -exec / -delete / -fprint..., write targets)
SHELL_WORDS = {'for', 'do', 'done', 'if', 'then', 'else', 'elif', 'fi', 'while', 'until', 'in', '{', '}', '(', ')', '!', 'case', 'esac',
               'export', 'set', 'read', 'local', 'unset', 'shift', 'return', 'break', 'continue', 'function'}   # shell syntax / builtins

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'Read', 'description': 'Reads a file from the local filesystem. file_path must be absolute. Text is returned in cat -n format '
        '(line numbers from 1), up to 2000 lines by default; use offset and limit for long files. Images (png, jpg, webp) are shown to you.',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string', 'description': 'The absolute path to the file to read'},
            'offset': {'type': 'integer', 'description': 'The line number to start reading from'},
            'limit': {'type': 'integer', 'description': 'The number of lines to read'}}, 'required': ['file_path']}}},
    {'type': 'function', 'function': {
        'name': 'Write', 'description': 'Writes a file to the local filesystem, overwriting it if it exists. file_path must be absolute. '
        'An existing file must be read with Read first.',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string', 'description': 'The absolute path to the file to write'},
            'content': {'type': 'string', 'description': 'The content to write to the file'}}, 'required': ['file_path', 'content']}}},
    {'type': 'function', 'function': {
        'name': 'Edit', 'description': 'Performs exact string replacement in a file. The file must have been read with Read first. old_string '
        'must match the file exactly and be unique, unless replace_all is true.',
        'parameters': {'type': 'object', 'properties': {
            'file_path': {'type': 'string', 'description': 'The absolute path to the file to modify'},
            'old_string': {'type': 'string', 'description': 'The text to replace'},
            'new_string': {'type': 'string', 'description': 'The text to replace it with (must be different from old_string)'},
            'replace_all': {'type': 'boolean', 'description': 'Replace all occurrences of old_string (default false)'}},
            'required': ['file_path', 'old_string', 'new_string']}}},
    {'type': 'function', 'function': {
        'name': 'Bash', 'description': 'Executes a bash command and returns its output. The working directory persists between calls. '
        'timeout is in milliseconds (default 120000, max 600000).',
        'parameters': {'type': 'object', 'properties': {
            'command': {'type': 'string', 'description': 'The command to execute'},
            'timeout': {'type': 'number', 'description': 'Optional timeout in milliseconds (max 600000)'},
            'description': {'type': 'string', 'description': 'What the command does'}}, 'required': ['command']}}},
]


class Sandbox:
    def __init__(self, workdir, read_roots, write_roots, bash_allow, py_scripts, py_inline=True, bash_prefixes=None, bash_env=None):
        self.cwd = Path(workdir).resolve(); self.cwd.mkdir(parents=True, exist_ok=True)
        self.write = [Path(p).resolve() for p in write_roots]
        self.read = [Path(p).resolve() for p in read_roots] + self.write
        self.allow = set(bash_allow); self.seen = set(); self.denied = []; self.py_inline = py_inline; self.prefixes = bash_prefixes
        self.env = clean_env(dict({'HOME': str(self.cwd)}, **(bash_env or {})))   # ~ / $HOME point at the task directory
        self.py_names = {s for s in py_scripts if '/' not in s}                        # by name, under the readable paths
        self.py_paths = {str(Path(s).resolve()) for s in py_scripts if '/' in s}        # by absolute path: run, not read

    def _under(self, p, roots):
        p = Path(p).resolve()
        return any(p == r or r in p.parents for r in roots)

    def _abs(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.cwd / p

    def deny(self, tool, why):
        self.denied.append(dict(tool=tool, why=why[:300])); return f'Error: refused by the sandbox: {why}'

    # ---- tools ----
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
        out = ''.join(f'{i + 1:6d}\t{l[:2000]}\n' for i, l in enumerate(lines[start:start + n], start))
        return out, None

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

    def _check(self, command):
        """None if the command may run, else the reason it is refused (see the module docstring)"""
        if self.prefixes is not None:   # Claude Code prefix rules
            c = command.strip()
            if not any(c.startswith(x) for x in self.prefixes):
                return f'only commands starting with {self.prefixes} are allowed'
            if re.search(r'[;&|`$()<>\n]', c):
                return 'the command may not contain ; & | ` $ ( ) < > or a newline'
            return None
        body = re.sub(r"<<-?\s*['\"]?(\w+)['\"]?(.*?)\n(.*?)\n\s*\1\s*(\n|$)", lambda m: ' <<HEREDOC' + m.group(2) + '\n', command, flags=re.S)
        for path in re.findall(r"(?<![\w.$/)\]])(/[A-Za-z_.][A-Za-z0-9_.@%+=:,/-]*)", command):   # not /0.5 (arithmetic)
            if path not in ('/dev/null', '/dev/stdout', '/dev/stderr') and str(Path(path).resolve()) not in self.py_paths \
                    and not self._under(path, self.read):
                return f'path {path} is outside the allowed paths'
        body = '\n'.join(l for l in body.split('\n') if not l.lstrip().startswith('#'))   # comment lines (may hold stray quotes)
        quoted = []   # quoted strings (may span lines, e.g. python3 -c "...") become single placeholder words

        def q(m):
            quoted.append(m.group(0)[1:-1]); return f'QSTR{len(quoted) - 1}Q'
        body = re.sub(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"", q, body, flags=re.S)
        unq = lambda a: re.sub(r'QSTR(\d+)Q', lambda m: quoted[int(m.group(1))], a)
        env = {}; cwd = self.cwd; funcs = set()   # shell functions defined in this command may be called later in it
        for seg in re.split(r'&&|\|\||;|\||\n|`|\$\(|<\(|>\(', body):
            w = seg.split()
            w = [unq(x) for x in (w[:next((i for i, x in enumerate(w) if x.startswith('#')), len(w))])]
            w, why = self._redirects(w, env, cwd)   # every < / > target, wherever it stands, is checked and taken out
            if why:
                return why
            if any(x.startswith('~') for x in w):
                return 'words starting with ~ (home directory expansion) are not allowed'
            while w and (w[0] in SHELL_WORDS or re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', w[0])):
                if w[0] in ('for', 'case', 'select', 'set', 'export', 'unset', 'local', 'read'):
                    w = []; break   # `for X in <words>`: only literals here; the commands follow `do`
                if '=' in w[0] and w[0] not in SHELL_WORDS:
                    k, v = w[0].split('=', 1); env[k] = v
                w = w[1:]
            if w and re.match(r'^\w+\(\)\{?$', w[0]):   # `name(){ ...`: the body is checked as its own segments
                funcs.add(w[0].split('(')[0]); w = [x for x in w[1:] if x != '{']
            if w and w[0] in funcs:
                continue
            if w and w[0].startswith('$'):   # a command held in a variable set earlier in the same call, e.g. G="python3 .../geomrun.py"
                v = env.get(w[0].strip('${}'))
                if v is None:
                    return f'cannot resolve the command {w[0]!r}'
                w = v.split() + w[1:]
            if not w or w[0].startswith('<') or w[0].startswith('>') or w[0].startswith('QSTR'):
                continue
            prog = w[0].rsplit('/', 1)[-1]
            if prog in SHELL_WORDS:
                continue
            if prog == 'sed' and w[0] != 'sed':
                return 'call sed by its bare name'
            if prog not in self.allow:
                return f'program {prog!r} is not allowed (allowed: {", ".join(sorted(self.allow))})'
            args = [re.sub(r'\$\{?(\w+)\}?', lambda m: env.get(m.group(1), m.group(0)), a) for a in w[1:]]
            for a in args:   # path-like words (and what globs expand to), resolved here, must be readable
                a2 = a.split('=', 1)[1] if a.startswith('-') and '=' in a else a
                if not a2 or any(c in a2 for c in ' \n$') or a2.startswith('<') or a2 == '<<HEREDOC':
                    continue
                if any(c in a2 for c in '*?['):   # a glob: what it expands to must be readable
                    import glob as _g
                    hits = _g.glob(a2 if a2.startswith('/') else str(cwd / a2))
                    if any(not self._under(h, self.read) for h in hits):
                        return f'glob {a2} reaches outside the allowed paths'
                    continue
                if '/' in a2 or a2.startswith('.'):
                    p = (Path(a2) if a2.startswith('/') else cwd / a2).resolve()
                    if str(p) not in self.py_paths and not self._under(p, self.read):
                        return f'path {a2} (resolves to {p}) is outside the allowed paths'
            if prog == 'cd':
                tgt = args[0] if args else str(self.cwd); p = Path(tgt) if tgt.startswith('/') else cwd / tgt
                if not self._under(p, self.read):
                    return f'cd target {tgt} is outside the allowed paths'
                cwd = p.resolve(); continue
            why = self._program_rules(prog, args, cwd)
            if why:
                return why
            if prog in ('python3', 'python'):
                first = next((a for a in args if a not in ('-u', '-B')), '')
                if first in ('-c', '-', '<<HEREDOC', '') or first.startswith('<'):
                    if not self.py_inline:
                        return 'inline python (-c, stdin, heredoc) is not allowed here; only the listed scripts can run'
                    continue
                scripts = [a for a in args if a.endswith('.py')]
                if not scripts:
                    return f'python {first!r} is not allowed: only the listed scripts can run'
                for x in scripts:
                    p = (Path(x) if x.startswith('/') else cwd / x).resolve()
                    if str(p) in self.py_paths or (p.name in self.py_names and self._under(p, self.read)):
                        continue
                    return f'python script {x} is not allowed (allowed: {", ".join(sorted(self.py_names | self.py_paths))})'
        return None

    def _redirects(self, w, env, cwd):
        """check and drop redirections: < f (input) must be readable, > f / >> f / 2> f (output) writable; returns (words, reason)"""
        out = []; i = 0
        sub = lambda a: re.sub(r'\$\{?(\w+)\}?', lambda m: env.get(m.group(1), m.group(0)), a)
        while i < len(w):
            x = w[i]
            if x in ('<<HEREDOC', '<<<') or x.startswith('<<'):
                out.append(x); i += 1; continue
            m = re.match(r'^(\d?)(<|>>?|&>>?)(.*)$', x)
            if m and not m.group(3).startswith('&'):
                tgt = m.group(3) or (w[i + 1] if i + 1 < len(w) else ''); i += 1 if m.group(3) else 2
                tgt = sub(tgt)
                if not tgt or tgt == '/dev/null':
                    continue
                if any(c in tgt for c in '$*?[~') or tgt.startswith('~'):
                    return out, f'redirection target {tgt} must be a plain path'
                p = Path(tgt) if tgt.startswith('/') else cwd / tgt
                if m.group(2) == '<' and not self._under(p, self.read):
                    return out, f'input {tgt} is outside the readable paths'
                if m.group(2) != '<' and not self._under(p, self.write):
                    return out, f'redirection target {tgt} is outside the writable paths'
                continue
            out.append(x); i += 1
        return out, None

    def _program_rules(self, prog, args, cwd):
        """per-program limits in whitelist mode: commands that can run other commands or write files"""
        wr = lambda x: x == '/dev/null' or self._under(Path(x) if x.startswith('/') else cwd / x, self.write)
        pos = [x for x in args if not x.startswith('-')]
        if prog == 'sed':   # also run as `sed --sandbox` by the Bash wrapper (GNU sed then refuses e / w / r itself)
            opts = [x for x in args if x.startswith('-')]
            if any(o in ('-f', '--file') or o.startswith('--file=') or (re.match(r'^-[a-zA-Z]*f', o) and not o.startswith('--')) for o in opts):
                return 'sed -f (a script file) is not allowed'
            scripts = [args[i + 1] for i, x in enumerate(args[:-1]) if x in ('-e', '--expression')] + \
                      [x.split('=', 1)[1] for x in args if x.startswith('--expression=')] or pos[:1]
            for sc in scripts:
                # linear-time forms (no exponential backtracking on runs of backslashes or whitespace); an escaped delimiter is
                # read as sed reads it. A newline inside the address run is itself a command start, hence [^\S\n].
                if re.search(r'(^|[;\n{}])(?:[0-9$,/!~+]|[^\S\n])*[eEwWrR](\s|$|;)', sc) or re.search(r's(.)(?:\\.|(?!\1)[^\\])*\1(?:\\.|(?!\1)[^\\])*\1[gpIiMm0-9]*[ew]', sc):
                    return 'sed scripts with e / w / W / r / R commands or s///e, s///w flags are not allowed'
            if any(o == '-i' or o.startswith('-i') or o.startswith('--in-place') or (re.match(r'^-[a-zA-Z]*i', o) and not o.startswith('--')) for o in opts):
                files = pos[0 if any(x in ('-e', '--expression') for x in args) else 1:]
                if any(not wr(f) for f in files):
                    return 'sed -i may only edit files under the writable paths'
        if prog == 'find' and any(x in ('-exec', '-execdir', '-ok', '-okdir', '-delete', '-fprint', '-fprint0', '-fprintf', '-fls') for x in args):
            return 'find -exec / -execdir / -ok / -okdir / -delete / -fprint* / -fls are not allowed'
        if prog == 'cp' and pos and not wr(pos[-1]):
            return f'cp target {pos[-1]} is outside the writable paths'
        if prog == 'mkdir' and any(not wr(x) for x in pos):
            return 'mkdir only under the writable paths'
        if prog == 'sort' and any(x.startswith(('--compress-program', '--files0-from', '--temporary-directory')) or x.startswith('-T') for x in args):
            return 'sort --compress-program / --files0-from / -T are not allowed'
        if prog == 'sort':
            outs = [args[i + 1] for i, x in enumerate(args[:-1]) if x == '-o'] + [x.split('=', 1)[1] for x in args if x.startswith('--output=')] + \
                   [x[2:] for x in args if x.startswith('-o') and len(x) > 2]
            if any(not wr(o) for o in outs):
                return 'sort -o only under the writable paths'
        if prog == 'uniq' and len(pos) >= 2 and not wr(pos[1]):
            return 'uniq output file outside the writable paths'
        return None

    def Bash(self, command, timeout=None, description=None):
        why = self._check(command)
        if why:
            return self.deny('Bash', why)
        t = min(float(timeout or 120000), 600000) / 1000; cwdf = self.cwd / '.agent_cwd'
        pre = '' if self.prefixes is not None else 'sed(){ command sed --sandbox "$@"; }\n'   # whitelist mode: GNU sed refuses e / w / r
        wrapped = f'{pre}cd {shlex.quote(str(self.cwd))} && {{\n{command}\n}}; __rc=$?; pwd > {shlex.quote(str(cwdf))}; exit $__rc'
        pr = subprocess.Popen(['bash', '-c', wrapped], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace',
                              env=self.env, start_new_session=True)   # own process group: a timeout kills the checker's children too
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
        except Exception:
            pass
        out = out if len(out) <= 30000 else out[:30000] + f'\n... (output truncated, {len(out)} characters in total)'
        return out or '(no output)'


def clean_env(extra=None):
    """the environment of Bash commands: the parent's minus credentials and session plumbing (CLAUDE*, LLM_*, names holding TOKEN /
    KEY / SECRET / PASSWORD, PYTHONPATH, PWD, OLDPWD), with PYTHONUSERBASE pinned to the parent's user base (HOME is moved to the
    task directory, and user-installed packages such as numpy must stay importable), plus the caller's extra variables"""
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith('CLAUDE') or k.startswith('LLM_') or any(w in k.upper() for w in ('TOKEN', 'KEY', 'SECRET', 'PASSWORD'))
                   or k in ('PYTHONPATH', 'PWD', 'OLDPWD'))}
    if 'PYTHONUSERBASE' not in env:   # HOME is moved to the task directory: keep Python's user packages (~/.local of the parent)
        import site
        env['PYTHONUSERBASE'] = site.getuserbase()
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def _img_ref(part, path):  # transcript placeholder for an image (path + hash instead of base64)
    return {'type': 'image_ref', 'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _run_tool(sb, name, args):
    """one tool call -> (result text, image path or None); any error goes back to the model as the result, as in Claude Code"""
    keys = {'Read': ('file_path', 'offset', 'limit'), 'Write': ('file_path', 'content'),
            'Edit': ('file_path', 'old_string', 'new_string', 'replace_all'), 'Bash': ('command', 'timeout', 'description')}.get(name)
    img = None
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
    if not str(out).strip():   # a whitespace-only tool result (e.g. `echo`) makes the provider reject the whole conversation (400, 4008)
        out = '(no output)'
    return out, img


def run_agent(task, *, workdir, read_roots, write_roots, transcript, record, tag, bash_allow=BASIC_BASH, py_scripts=(), py_inline=True,
              bash_prefixes=None, bash_env=None, effort=None, max_tokens=None, max_turns=400, protocol='anthropic', system=None,
              cache=None):
    """Run one context-free task to the end. Returns (final text, info). Raises RuntimeError when the API gives up.
    The Messages API with adaptive thinking: every assistant turn (thinking blocks with their signatures included) is sent back
    unchanged, as Claude Code does, and images go inside tool results (needs max_tokens).
    system: optional system prompt (default none).
    cache: None (default, no cache_control sent) or a prompt-cache TTL ('5m' / '1h'): the conversation prefix is cached between
    turns as Claude Code does."""
    if protocol != 'anthropic':
        raise ValueError("only protocol='anthropic' (the Messages API) is supported")
    sb = Sandbox(workdir, read_roots, write_roots, bash_allow, py_scripts, py_inline, bash_prefixes, bash_env)
    return _run_native(task, sb, transcript, record, tag, effort, max_tokens, max_turns, system, cache)


TOOLS_NATIVE = [{'name': t['function']['name'], 'description': t['function']['description'], 'input_schema': t['function']['parameters']} for t in TOOLS]


def _native_log(block):
    """a content block for the transcript: images as path + hash, thinking signatures by length only"""
    if block.get('type') == 'thinking':
        return dict(type='thinking', thinking=block.get('thinking', ''), signature=f"<{len(block.get('signature') or '')} chars>")
    if block.get('type') == 'redacted_thinking':
        return dict(type='redacted_thinking', data=f"<{len(block.get('data') or '')} chars>")
    return block


def _cached(messages, ttl):
    """the request's messages with prompt-cache breakpoints on the last block of the last two user messages (as Claude Code
    caches its conversation): the prefix up to them is read from the cache on the next turn instead of being processed again.
    The model sees the same content; `messages` itself is not changed."""
    out = list(messages); marked = 0
    for i in range(len(out) - 1, -1, -1):
        if out[i]['role'] != 'user' or marked == 2:
            continue
        c = out[i]['content']; c = [{'type': 'text', 'text': c}] if isinstance(c, str) else [dict(b) for b in c]
        c[-1]['cache_control'] = {'type': 'ephemeral', 'ttl': ttl}; out[i] = dict(out[i], content=c); marked += 1
    return out


def _run_native(task, sb, transcript, record, tag, effort, max_tokens, max_turns, system=None, cache=None):
    if not max_tokens:
        raise ValueError('the native protocol needs max_tokens')
    messages = [{'role': 'user', 'content': task}]; log = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': task}]
    turns = []; usage = {}; t0 = time.time()

    def save(status):
        Path(transcript).parent.mkdir(parents=True, exist_ok=True)
        s = json.dumps(dict(tag=tag, status=status, protocol='anthropic', seconds=round(time.time() - t0, 1), usage=usage, denied=sb.denied,
                            turns=turns, messages=log), ensure_ascii=False, indent=1)
        tok = api_llm._TOKEN or os.environ.get('LLM_API_TOKEN')
        Path(transcript).write_text(s.replace(tok, '[REDACTED]') if tok else s)

    for turn in range(max_turns):
        body = dict(model=api_llm.MODEL, max_tokens=max_tokens, messages=_cached(messages, cache) if cache else messages, tools=TOOLS_NATIVE,
                    thinking={'type': 'adaptive'})
        if system:
            body['system'] = system
        if effort is not None:
            body['output_config'] = {'effort': effort}
        try:
            d, meta = api_llm.complete_native(body, record=record, tag=tag)
        except Exception:
            save('api_error'); raise
        for k, v in (meta.get('usage') or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        content = d.get('content') or []
        messages.append({'role': 'assistant', 'content': content}); log.append({'role': 'assistant', 'content': [_native_log(b) for b in content]})
        text = ''.join(b.get('text', '') for b in content if b.get('type') == 'text'); calls = [b for b in content if b.get('type') == 'tool_use']
        rec = dict(turn=turn, seconds=meta.get('seconds'), backend=meta.get('backend'), finish=meta.get('finish'), usage=meta.get('usage'),
                   thinking_blocks=sum(b.get('type') in ('thinking', 'redacted_thinking') for b in content), text=text[:500], calls=[])
        if not calls:
            turns.append(rec); save('done')
            return text, dict(turns=turn + 1, usage=usage, denied=sb.denied, seconds=round(time.time() - t0, 1), finish=meta.get('finish'),
                              tools={k: sum(1 for t in turns for c in t['calls'] if c['name'] == k) for k in ('Read', 'Write', 'Edit', 'Bash')})
        results, logged = [], []
        cut = d.get('stop_reason') == 'max_tokens'
        for c in calls:
            if cut and c is calls[-1]:   # the last tool call of a reply cut at max_tokens is incomplete: not run, and the model is told so
                out, img = 'Error: the reply reached the output token limit (max_tokens) before this tool call was complete, so it was not run.', None
            else:
                out, img = _run_tool(sb, c['name'], c.get('input'))
            if img is not None:   # the image inside the tool result, as Claude Code returns it
                part = api_llm.image_part_native(img)
                results.append({'type': 'tool_result', 'tool_use_id': c['id'], 'content': [{'type': 'text', 'text': out}, part]})
                logged.append({'type': 'tool_result', 'tool_use_id': c['id'], 'content': [{'type': 'text', 'text': out}, _img_ref(part, img)]})
            else:
                results.append({'type': 'tool_result', 'tool_use_id': c['id'], 'content': out}); logged.append(results[-1])
            rec['calls'].append(dict(name=c['name'], args={k: (v if len(str(v)) < 300 else str(v)[:300] + '...') for k, v in (c.get('input') if isinstance(c.get('input'), dict) else {}).items()},
                                     result=str(out)[:300]))
        messages.append({'role': 'user', 'content': results}); log.append({'role': 'user', 'content': logged})
        turns.append(rec); save('running')
    save('max_turns')
    raise RuntimeError(f'{tag}: no final answer after {max_turns} turns')
