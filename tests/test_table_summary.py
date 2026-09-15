"""段階 3 のまとめで，要確認を「見るべき」と「印で外せる」に分ける

2026-09-15 に足した．**流し込み・「1 回出現の種」・見出しの行は，読めなくて
当たり前** (種名の文章が行に入っている)．全 147 表で要確認 1,027 件のうち
**379 件 (37%)** がこれで，混ぜて数えると「まだこんなに残っている」と見える．
"""
import pandas as pd
import pytest

from comptea.pipeline import table as table_mod


def _long(rows):
    return pd.DataFrame(
        [{'source_image': 'a.png', 'source': 'body', 'plot': p, 'row_no': r,
          'j_name': j, 's_name': '', 'layer': '', 'cover': c,
          'sociability': '', 'comp_raw': c, 'status': s, 'note': n}
         for p, r, j, c, s, n in rows])


def _sum(df_long, df_plot=None):
    """検査の文言をまとめて返す

    `check(df_long, df_plot, out)` は `out` に足していく形 (返り値ではない)．
    """
    out = []
    table_mod.check(df_long, pd.DataFrame() if df_plot is None else df_plot,
                    out)
    return '\n'.join(out)


def test_印の無い要確認はそのまま数える():
    got = _sum(_long([(1, 1, 'アカマツ', 'ど', 'Need Check', '')]))
    assert '読めない形 1 件' in got
    assert '見なくてよい' not in got


def test_流し込みの行は見なくてよいと分ける():
    got = _sum(_long([(1, 1, 'アカマツ', 'ど', 'Need Check', ''),
                      (1, 2, '', 'Oru', 'Need Check', 'flow'),
                      (1, 3, '', 'ens', 'Need Check', 'flow')]))
    assert '読めない形 3 件' in got
    assert '2 件は流し込み' in got
    assert '見るべきは 1 件' in got


def test_1回出現の種と見出しも外す():
    got = _sum(_long([(1, 1, '', 'x', 'Need Check', 'once'),
                      (1, 2, '', 'y', 'Need Check', 'heading')]))
    assert '2 件は流し込み' in got
    assert '見るべきは 0 件' in got


def test_見るべき件だけを並べる():
    """一覧に出すのは**見るべき**セルだけ (印の付いた行は並べない)"""
    got = _sum(_long([(1, 1, 'アカマツ', 'ど', 'Need Check', ''),
                      (1, 2, 'ススキ', 'Oru', 'Need Check', 'flow')]))
    assert 'アカマツ' in got
    assert 'ススキ' not in got


def test_要確認が無ければ何も分けない():
    got = _sum(_long([(1, 1, 'アカマツ', '2・2', 'OK', '')]))
    assert '読めない形 0 件' in got
