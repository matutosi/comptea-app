"""文書がコードとずれていないか (2026-09-17)

09-17 の文書の点検で，3 日で約 10 件の名前のずれ (移った関数・消えた関数) と，
行き先の無いリンクが見つかった．名前を変えたときに文書を直し忘れても気づけるよう，
**文書の字面**で確かめる (`tests/test_no_local_paths.py` と同じ考え方)．

- backtick で書いた `mod.name` は，そのモジュールに定義があること
- backtick で書いた `cli/xxx.py` や `xxx.py` は，ファイルがあること
- Markdown のリンクは，ファイルと見出し (GitHub の slug) があること

経緯の記録 (`docs/lessons*.md`) は，消した名前を書くのが正しいので名前は見ない
(リンクだけ見る)．
"""
import glob
import io
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rel(pats):
    out = []
    for pat in pats:
        for p in glob.glob(os.path.join(ROOT, pat), recursive=True):
            r = os.path.relpath(p, ROOT)
            if 'worktrees' not in r:
                out.append(r.replace(os.sep, '/'))
    return sorted(set(out))


# 今の姿を書く文書 (名前とリンクを見る)
CURRENT = _rel(['README.md', 'docs/architecture.md', 'docs/pipeline.md',
                'docs/vegetation_science.md', 'eval/README.md',
                '.claude/skills/**/*.md'])
# リンクだけ見る文書
ALL_MD = _rel(['*.md', 'docs/**/*.md', 'eval/**/*.md', '.claude/skills/**/*.md'])

TICK = re.compile(r'`([^`\n]+)`')
DOTTED = re.compile(r'^([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*)\.([A-Za-z_]\w*)(?:\(\))?$')
PATH = re.compile(r'^(?:comptea|cli|eval|apps|tests)/[\w/.\-]+\.(?:py|txt|pt)$')
PYFILE = re.compile(r'^[\w\-]+\.py$')
DATAFILE = re.compile(r'^[\w\-]+\.(?:csv|txt|md|pt|png|json|tsv|xlsx|jpg|pdf)$')
CODE_DIRS = ('comptea', 'cli', 'eval', 'apps', 'tests')


def _text(rel):
    return io.open(os.path.join(ROOT, rel), encoding='utf-8').read()


def _module_file(mod):
    parts = mod.split('.')
    if parts[0] == 'comptea':
        parts = parts[1:]
    for base in ('comptea', 'eval', 'cli'):
        p = os.path.join(ROOT, base, *parts) + '.py'
        if os.path.exists(p):
            return p
    return None                               # df.x のような属性や列名は見ない


def _defines(path, name):
    src = io.open(path, encoding='utf-8').read()
    n = re.escape(name)
    return re.search(rf'^\s*(?:async\s+)?(?:def|class)\s+{n}\b|^\s*{n}\s*[:=]'
                     rf'|^\s*(?:from\s+\S+\s+)?import\s+.*\b{n}\b', src, re.M) is not None


def name_problems(text):
    """文書の本文から，コードに無い名前を (種類, 名前, 行) で返す．見た数も返す"""
    bad, seen = [], 0
    for i, line in enumerate(text.splitlines(), 1):
        for tok in TICK.findall(line):
            tok = tok.strip()
            if PATH.match(tok):
                seen += 1
                if not glob.glob(os.path.join(ROOT, tok)):
                    bad.append(('path', tok, i))
            elif PYFILE.match(tok):
                seen += 1
                if not any(glob.glob(os.path.join(ROOT, b, '**', tok), recursive=True)
                           for b in CODE_DIRS):
                    bad.append(('file', tok, i))
            elif DATAFILE.match(tok):
                continue
            else:
                m = DOTTED.match(tok)
                if not m:
                    continue
                f = _module_file(m.group(1))
                if f is None:
                    continue
                seen += 1
                if not _defines(f, m.group(2)):
                    bad.append(('name', tok, i))
    return bad, seen


def slug(heading):
    """GitHub の見出しの slug (小文字・記号を落とす・空白を - に)"""
    s = re.sub(r'[^\w\- ]', '', heading.strip().lower())
    return s.replace(' ', '-')


def _anchors(rel):
    body = re.sub(r'```.*?```', '', _text(rel), flags=re.S)
    return {slug(h) for h in re.findall(r'^#+\s+(.*)$', body, flags=re.M)}


LINK = re.compile(r'\]\(([^)\s]+)\)')


def link_problems(rel):
    bad = []
    body = re.sub(r'```.*?```', '', _text(rel), flags=re.S)
    for i, line in enumerate(body.splitlines(), 1):
        for target in LINK.findall(line):
            if re.match(r'^[a-z]+:', target):          # http: mailto: など
                continue
            path, _, anchor = target.partition('#')
            dest = (os.path.normpath(os.path.join(os.path.dirname(rel), path))
                    if path else os.path.normpath(rel))
            full = os.path.join(ROOT, dest)
            if not os.path.exists(full):
                bad.append(f'{i}: {target} (ファイルが無い)')
            elif anchor and dest.endswith('.md') and anchor.lower() not in _anchors(dest):
                bad.append(f'{i}: {target} (見出しが無い)')
    return bad


@pytest.mark.parametrize('rel', CURRENT)
def test_文書の名前がコードにある(rel):
    bad, _ = name_problems(_text(rel))
    assert not bad, ('文書の名前がコードに無い (名前を変えたら文書も直す): '
                     + ' / '.join(f'{i}: {t}' for _, t, i in bad))


@pytest.mark.parametrize('rel', ALL_MD)
def test_リンクと見出しの行き先がある(rel):
    bad = link_problems(rel)
    assert not bad, '行き先の無いリンク: ' + ' / '.join(bad)


def test_名前の検査は実際に名前を見ている():
    """抜き出しが壊れて 0 件を見て通る，を防ぐ"""
    seen = sum(name_problems(_text(rel))[1] for rel in CURRENT)
    assert seen >= 200, seen


def test_消えた名前と無いファイルを拾う():
    bad, _ = name_problems('`pipeline.read.push_left_rule` と `cli/no_such.py` と '
                           '`no_such_tool.py`，`pipeline.read.move_off_rules` はある')
    assert [k for k, _, _ in bad] == ['name', 'path', 'file']


def test_slugはGitHubに合わせる():
    assert slug('11. 続きのページをつなぐ') == '11-続きのページをつなぐ'
    assert slug('外の読み手 (任意)') == '外の読み手-任意'
