"""ファイル名の枝番から，同じ表に属するページをまとめる

どのページが続きかはノンブルの順で分かるはずだが，**紙面の配置の都合で
崩れる**(`s01114_kinki_081` = p.383 と `s01114_kinki_082` = p.382)．
そこでユーザが枝番で明示する(2026-09-08 決定)．
"""
import pytest

from comptea import page_group as pg


@pytest.mark.parametrize("name, expected", [
    # 枝番あり
    ('s01114_kinki_081-1', ('s01114_kinki_081', 1, '')),
    ('s01114_kinki_081-2', ('s01114_kinki_081', 2, '')),
    # 3 ページ以上に渡ることもある
    ('s01114_kinki_081-10', ('s01114_kinki_081', 10, '')),
    # 枝番なし(組成表だけのページ)
    ('s01114_kinki_045', ('s01114_kinki_045', None, '')),
    # 工程が後ろに足す接尾辞は，そのまま持ち回る
    ('s01114_kinki_081-2_t1', ('s01114_kinki_081', 2, '_t1')),
    ('s01115_01-1_p2', ('s01115_01', 1, '_p2')),
    # 表の名前に `-数字` が入っていても，末尾の枝番だけを見る
    ('tab-4_sz1856-1', ('tab-4_sz1856', 1, '')),
])
def test_split_name(name, expected):
    assert pg.split_name(name) == expected


def test_table_key():
    assert pg.table_key('s01114_kinki_081-2') == 's01114_kinki_081'
    assert pg.table_key('s01114_kinki_045') == 's01114_kinki_045'
    # 接尾辞は表の別を表すので残す(1ページに表が2つ)
    assert pg.table_key('s01114_kinki_081-1_t2') == 's01114_kinki_081_t2'


def test_group_parts_枝番でまとまる():
    got = pg.group_parts(['s01114_kinki_081-2', 's01114_kinki_045',
                          's01114_kinki_081-1'])
    assert got == {
        's01114_kinki_081': [(1, 's01114_kinki_081-1'),
                             (2, 's01114_kinki_081-2')],
        's01114_kinki_045': [(None, 's01114_kinki_045')],
    }


def test_group_parts_接尾辞が違えば別の表():
    """1ページに表が2つあるときは `_t1`・`_t2` で分かれる"""
    got = pg.group_parts(['k_081-1_t1', 'k_081-2_t1', 'k_081-1_t2'])
    assert sorted(got) == ['k_081_t1', 'k_081_t2']
    assert got['k_081_t1'] == [(1, 'k_081-1_t1'), (2, 'k_081-2_t1')]


def test_つなぐと種名とページをまたいだ地点が復元される():
    """1 ページずつ解析すると解けないものが，つなぐと解ける

    `kinki_081`(p.383)の末尾 `Lysimachia japonica f. subsessilis` と
    `kinki_082`(p.382)の頭 `コナスビ` は同じ 1 件で，地点は前ページの `in 4:`．
    """
    from comptea import parse_text

    part1 = ('in 4: Hypericum erectum オトギリソウ +, '
             'Lysimachia japonica f. subsessilis')
    part2 = 'コナスビ +, Rubus microphyllum ニガイチゴ +.'
    # ページごとに解析すると，割れた種名は拾えず，地点も引き継げない
    assert len(parse_text.parse_once_species(part2)) == 0
    # つないでから 1 度だけ解析する
    rows = parse_text.parse_once_species(f'{part1} {part2}')
    assert [(r['plot'], r['s_name'], r['j_name']) for r in rows] == [
        (4, 'Hypericum erectum', 'オトギリソウ'),
        (4, 'Lysimachia japonica f. subsessilis', 'コナスビ'),
        (4, 'Rubus microphyllum', 'ニガイチゴ'),
    ]


def test_missing_parts():
    assert pg.missing_parts([(1, 'a-1'), (3, 'a-3')]) == [2]
    assert pg.missing_parts([(1, 'a-1'), (2, 'a-2')]) == []
    assert pg.missing_parts([(None, 'a')]) == []
    # 1 から始まらないときも抜けとして知らせる
    assert pg.missing_parts([(2, 'a-2')]) == [1]
