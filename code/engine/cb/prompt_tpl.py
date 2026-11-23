"""Prompt templates (prompts/): every text the pipeline sends to a model is a file there.

    text(name)              the file prompts/<name> with one trailing newline removed
    fill(name, **kw)        text(name).format(**kw)  (literal braces are doubled in the files)
    lines(name, lists, **kw)   for multi-line task files: a line that is only "{NAME}" with NAME in `lists` expands to that list's
                            lines (none for an empty list); every other line is formatted with kw; joined with newlines
"""
from functools import lru_cache
from pathlib import Path

DIR = Path('@CD_CODE@/prompts')


@lru_cache(maxsize=None)
def text(name):
    s = (DIR / name).read_text()
    return s[:-1] if s.endswith('\n') else s


def fill(name, **kw):
    return text(name).format(**kw)


def lines(name, lists=None, **kw):
    out = []
    for line in text(name).split('\n'):
        k = line.strip()
        if lists and k.startswith('{') and k.endswith('}') and k[1:-1] in lists:
            out += list(lists[k[1:-1]])
        else:
            out.append(line.format(**kw))
    return '\n'.join(out)
