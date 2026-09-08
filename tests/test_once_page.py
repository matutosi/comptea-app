"""ページをまたいだ「1回出現種」と，表の下の注記

組成表の下の流し込みが紙面に収まらないと，次のページへあふれる．
あふれた先は本文や写真だけのページなので，これまでページごと捨てていた
(2026-09-07 にユーザ指摘で判明)．**文字列はすべて実物から起こしたもの**．
"""
import pytest

from comptea import once_page, parse_text as pt


# 実物 --------------------------------------------------------------------
# kinki_046(p.249)．前ページの表の続きで，目印から始まる
K046 = ('出現1回の種 Außerdem je einmal in Lfd. Nr. 1: Cocculus orbiculatus '
        'アオツヅラフジ S—+・2, in 2: Rosa multiflora ノイバラ S—+, '
        'Euonymus sieboldianus マユミ S—+, Aristolochia kaempferi '
        'オオバウマノスズクサ S—+, Cryptomeria japonica スギ B—2・3, '
        'Prunus jamasakura ヤマザクラ B—+, Wisteria brachybotrys ヤマフジ S—+.')
K046_NOTE = ('Lage d. Aufn. 調査地: Kitazono, Stadt Itami Präf. Osaka '
             '大阪府伊丹市北園. Datum d. Aufn. 調査年月日: 6 Nov. 1973')
# kinki_005(p.155)．目印は前ページにあり，種が1つだけ残っている
K005_TAIL = 'serratum トウゲシバ K—+.'
K005_NOTE = ('調査地 Lage: Lfd. Nr. 1~5: Suibara, Kokawa-cho, Naga-gun. '
             'Präf. Wakayama 和歌山県那賀郡粉河町杉原龍門山. '
             '3—5: Original 原調査資料, 2: Präf. Wakayama 和歌山県 1978.')
# kinki_062(p.—)．地点ごとに調査地が並ぶ形
K062_NOTE = ('Lage d. Aufn. 調査地: Lfd. Nr.1,2: Ryujin-mura, Hidaka-gun '
             '日高郡龍神村, 3: Tsuchiyama-cho, Koga-gun 甲賀郡土山町, '
             '8,10: Hioki, Stadt Miyazu 宮津市日置')
# 本文(kinki_082)．カタカナは多いが被度が並ばない
PROSE = ('土壌の浅い立地に生育している．クロヅルーノリウツギ群集の自然性の'
         '生育地はブナ林が風衝地，岩角地の低木林に移行する林縁部にみられる．')


# --- 種の並びかどうかの見分け --------------------------------------------

@pytest.mark.parametrize("text, expected", [
    (K046, True),
    ('Dumasia truncata ノササゲ K—+, Dioscorea japonica ヤマノイモ K—+', True),
    # 本文はカタカナが続いても被度が並ばない
    (PROSE, False),
    ('26. イブキシモツケ群落', False),
    ('', False),
])
def test_looks_like_entries(text, expected):
    assert once_page.looks_like_entries(text) is expected


def test_tail_of_entries():
    """最後の1行は種が1つだけのことがある"""
    assert once_page._tail_of_entries(K005_TAIL)
    assert not once_page._tail_of_entries(PROSE)


# --- 塊の範囲 -------------------------------------------------------------

def test_find_once_span_目印から始まる():
    texts = ['118. クサイチゴータラノキ群集', K046, K046_NOTE]
    assert once_page.find_once_span(texts) == (1, 2)


def test_find_once_span_前のページからの続き():
    """目印が無くても，ページの頭が種の並びなら続きとみなす"""
    texts = [K005_TAIL, K005_NOTE, 'Fig. 54. 蛇紋岩地に生育する…']
    assert once_page.find_once_span(texts) == (0, 1)


def test_find_once_span_本文だけのページ():
    texts = ['26. イブキシモツケ群落', PROSE, PROSE]
    assert once_page.find_once_span(texts) is None


def test_note_text():
    texts = [K046, K046_NOTE, '119. つぎの群集']
    lines = [{'text': t, 'x1': 0, 'x2': 10, 'y1': i * 10, 'y2': i * 10 + 5}
             for i, t in enumerate(texts)]
    span = once_page.find_once_span(texts)
    assert once_page.note_text(lines, span) == pt.normalize(K046_NOTE)


def test_note_text_柱を読み飛ばして注記を拾う():
    """塊の無いページでは 0 行目から見るので，紙面の柱で止めてはいけない

    `s01114_kinki_010-2` で注記が丸ごと落ちていた(2026-09-08 に実データで発覚)．
    注記が**始まってから**の見出しは，これまでどおり区切りとして使う．
    """
    texts = ['37. ハマエンドウーテリハノイバラ群落',
             'テリハノイバラの優占しているマント群落は，海岸砂丘…',
             '7: Koza-cho, Higashimuro-gun 東牟婁郡古座町田原 (29. Apr. 1983)',
             'Lfd. Nr. 1—9: Original 原調査資料.',
             '38. つぎの群落']
    lines = [{'text': t, 'x1': 0, 'x2': 10, 'y1': i * 10, 'y2': i * 10 + 5}
             for i, t in enumerate(texts)]
    got = once_page.note_text(lines, None)
    assert '古座町田原' in got and '原調査資料' in got
    assert '38.' not in got          # 始まったあとの見出しでは止める
    assert 'マント群落' not in got     # 本文は拾わない


def test_block_box():
    lines = [{'text': K046, 'x1': 10, 'x2': 900, 'y1': 100, 'y2': 160}]
    assert once_page.block_box(lines, (0, 1), pad=5) == (5, 95, 905, 165)


# --- 1回出現種の読み取り --------------------------------------------------

def test_parse_once_species_実物():
    rows = pt.parse_once_species(K046)
    assert len(rows) == 7
    assert rows[0] == {'plot': 1, 'j_name': 'アオツヅラフジ',
                       's_name': 'Cocculus orbiculatus',
                       'layer': 'S', 'comp_raw': '+;2', 'constancy': None}
    # 地点2は6件．`B—2・3` は被度と群度に分ける
    assert [r['plot'] for r in rows[1:]] == [2] * 6
    assert rows[4]['comp_raw'] == '2;3'


def test_parse_once_species_続きは地点を引き継ぐ():
    """目印より前の断片は，前のページの最後の地点に入れる"""
    rows = pt.parse_once_species(K005_TAIL, start_plot=5)
    assert rows == [{'plot': 5, 'j_name': 'トウゲシバ', 's_name': 'serratum',
                     'layer': 'K', 'comp_raw': '+', 'constancy': None}]


def test_parse_once_species_地点を渡さなければ捨てる():
    """これまでどおり．目印の無い断片を勝手に地点1へ入れない"""
    assert pt.parse_once_species(K005_TAIL) == []


def test_parse_once_species_区切りのカンマが印字で落ちていても切る():
    """`ゼンマイ K—+Viola kusanoana …` の実例(072-2．2026-09-08)

    切れないと，2 件が 1 件にまとまって**後ろの種が黙って落ちる**．
    """
    text = ('in 1: Osmunda japonica ゼンマイ K—+Viola kusanoana '
            'オオタチツボスミレ K—+')
    rows = pt.parse_once_species(text)
    assert [(r['j_name'], r['s_name']) for r in rows] == [
        ('ゼンマイ', 'Osmunda japonica'),
        ('オオタチツボスミレ', 'Viola kusanoana'),
    ]


def test_parse_once_species_階層が続く形は切らない():
    """`S—+, K—+・2` は同じ種の別の階層．属名が始まらないので切れない"""
    rows = pt.parse_once_species(
        'in 4: Fraxinus lanuginosa アオダモ S—+, K—+・2')
    assert [(r['j_name'], r['layer']) for r in rows] == [
        ('アオダモ', 'S'), ('アオダモ', 'K')]


def test_parse_once_species_常在度表の形():
    """常在度表の1回出現種は `常在度(被度)` で書かれる(kinki_064)"""
    text = ('in 6: Oplismenus undulatifolius ケチヂミザサ II(+), '
            'Desmodium fallax ケヤブハギ I(1), '
            'Salvia japonica アキノタムラソウ I(2)')
    rows = pt.parse_once_species(text)
    assert [(r['constancy'], r['comp_raw']) for r in rows] == [
        ('II', '+'), ('I', '1'), ('I', '2')]
    assert all(r['plot'] == 6 and r['layer'] is None for r in rows)


def test_parse_once_species_被度の形は常在度にしない():
    """`B—2・3` を `2(3)` と取り違えない"""
    rows = pt.parse_once_species(
        'in 2: Cryptomeria japonica スギ B—2・3, Rosa multiflora ノイバラ S—+')
    assert [(r['layer'], r['comp_raw'], r['constancy']) for r in rows] == [
        ('B', '2;3', None), ('S', '+', None)]


def test_looks_like_entries_常在度の並び():
    assert once_page.looks_like_entries(
        'Oplismenus undulatifolius ケチヂミザサ II(+), '
        'Desmodium fallax ケヤブハギ I(1)')


def test_parse_once_species_続きのあとに目印が来る():
    text = K005_TAIL + ' in 2: Rosa multiflora ノイバラ S—+'
    rows = pt.parse_once_species(text, start_plot=5)
    assert [r['plot'] for r in rows] == [5, 2]


# --- 注記(調査地・調査年月日・出典) ---------------------------------------

def test_parse_site_notes_1地点():
    """地点の指定が無ければ plot は None"""
    got = pt.parse_site_notes(K046_NOTE)
    assert got == [
        {'plot': None, 'field': 'locality',
         'value': 'Kitazono, Stadt Itami Präf. Osaka 大阪府伊丹市北園'},
        {'plot': None, 'field': 'date', 'value': '6 Nov. 1973'},
    ]


def test_parse_site_notes_範囲と飛び番():
    got = pt.parse_site_notes(K062_NOTE)
    assert [(r['plot'], r['value']) for r in got] == [
        (1, 'Ryujin-mura, Hidaka-gun 日高郡龍神村'),
        (2, 'Ryujin-mura, Hidaka-gun 日高郡龍神村'),
        (3, 'Tsuchiyama-cho, Koga-gun 甲賀郡土山町'),
        (8, 'Hioki, Stadt Miyazu 宮津市日置'),
        (10, 'Hioki, Stadt Miyazu 宮津市日置'),
    ]
    assert {r['field'] for r in got} == {'locality'}


def test_parse_site_notes_出典は値の形で見分ける():
    """`Nachweis` の見出しが前のページに残ることがある(kinki_005)"""
    got = pt.parse_site_notes(K005_NOTE)
    loc = [r for r in got if r['field'] == 'locality']
    src = [r for r in got if r['field'] == 'source_ref']
    assert [r['plot'] for r in loc] == [1, 2, 3, 4, 5]
    assert [(r['plot'], r['value']) for r in src] == [
        (3, 'Original 原調査資料'),
        (4, 'Original 原調査資料'),
        (5, 'Original 原調査資料'),
        (2, 'Präf. Wakayama 和歌山県 1978'),
    ]


def test_parse_site_notes_調査地の括弧の日付は出典にしない():
    """調査地に年が入るのは括弧の中だけ(kinki_011)"""
    text = ('調査地: 4—6: Kumihama-cho, Kumano-gun 熊野郡久美浜町箱石 '
            '(6. Juli 1983), 7: Koza-cho 古座町田原 (29. Apr. 1983)')
    got = pt.parse_site_notes(text)
    assert {r['field'] for r in got} == {'locality'}
    assert [r['plot'] for r in got] == [4, 5, 6, 7]


@pytest.mark.parametrize("head", [
    'Nachweis d. Vegetationsaufnahmen 既発表資料',
    'Nachweis d. Vegetationsaufnahme 既発表資料名',
    # 印字の誤植(kinki_082)．`aufnahmen` が `anfnahmen` になっている
    'Nachweis d. Vegetationsanfnahmen 既発表資料',
])
def test_parse_site_notes_出典の見出しの揺れ(head):
    got = pt.parse_site_notes(f'{head}: 1—2: Original 原調査資料')
    assert [(r['plot'], r['field']) for r in got] == [
        (1, 'source_ref'), (2, 'source_ref')]


def test_parse_site_notes_見出しが無ければ空():
    assert pt.parse_site_notes(PROSE) == []
