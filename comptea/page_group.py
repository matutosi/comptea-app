"""ファイル名の枝番から，同じ表に属するページをまとめる

組成表の下の流し込み(出現1回の種・調査地・調査年月日・出典)は，紙面に
収まらないと**次のページへあふれる**．どのページが続きかは，ふつうノンブル
(紙面のページ番号)の順で分かるが，**紙面の配置の都合で崩れることがある**
(`s01114_kinki_081` = p.383 と `s01114_kinki_082` = p.382 の実例)．

そこで**ユーザがファイル名の枝番で明示する**(2026-09-08 決定)．

    組成表だけ            xxx.jpg
    組成表と続きがある    xxx-1.jpg(組成表)  xxx-2.jpg(続き)

枝番は 1 から始まる通し番号で，**読む順**に振る．
組成表が 2 ページ以上に渡ることもあるので，枝番のどれが表かは決めない
(表のあるページは格子ができ，続きのページはできない．工程が自分で見分ける)．

**枝番はページのまとまりだけを表す**．工程が後ろに足す `_t1`・`_p2`・`_s1`
(1ページに表が2つ・折り込みを切った・左右に並んだ表)とは別の話なので，
枝番より後ろに付いたものはそのまま持ち回る．
"""
import re

# `<表>-<枝番>` に，工程が足す接尾辞が続くことがある(`-1_t2`)．
# 枝番は 1 桁以上の数．表の名前に `-数字` が入っていても，**末尾のものだけ**を見る
_PART = re.compile(r'^(?P<stem>.+)-(?P<part>\d+)(?P<rest>(?:_[A-Za-z]\w*)?)$')


def split_name(name: str):
    """`<表>-<枝番><接尾辞>` を (表, 枝番, 接尾辞) に分ける

    枝番が無ければ (名前, None, '') を返す．

    >>> split_name('s01114_kinki_081-1')
    ('s01114_kinki_081', 1, '')
    >>> split_name('s01114_kinki_081-2_t1')
    ('s01114_kinki_081', 2, '_t1')
    >>> split_name('s01114_kinki_045')
    ('s01114_kinki_045', None, '')
    """
    m = _PART.match(name)
    if not m:
        return (name, None, '')
    return (m.group('stem'), int(m.group('part')), m.group('rest'))


def table_key(name: str) -> str:
    """そのページが属する表の名前(枝番を落としたもの)"""
    stem, part, rest = split_name(name)
    return stem + rest if part is not None else name


def group_parts(names):
    """ページの名前を表ごとにまとめる

    Args:
        names: 置き場や画像の名前(拡張子は付けない)
    Returns:
        {表の名前: [(枝番, 名前), ...]}．枝番の順に並ぶ．
        枝番の無いページは 1 つだけの組にし，枝番を None にする
    """
    groups = {}
    for name in names:
        stem, part, rest = split_name(name)
        key = stem + rest if part is not None else name
        groups.setdefault(key, []).append((part, name))
    for key, items in groups.items():
        # 枝番の無いものは並べ替えの基準が無いので，名前の順に置く
        items.sort(key=lambda t: (t[0] is None, t[0], t[1]))
    return groups


def missing_parts(items):
    """枝番の抜けを返す(`-1` と `-3` があって `-2` が無い，など)

    抜けたまま繋ぐと，**気づかないうちに種が落ちる**ので知らせる．
    """
    parts = sorted(p for p, _ in items if p is not None)
    if not parts:
        return []
    return [n for n in range(1, parts[-1] + 1) if n not in parts]
