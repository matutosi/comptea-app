"""表の下の注記から地点情報を取る (site_notes.py)

レイアウト解析は**1 つの注記を複数の段落に割る**ことがある．
s01115_19_p3 の注記は
    「調査地 Lage: Lfd. Nr.1-10: … 神戸市山田町菊水山 Juni 1983), 11-15: Berg Nagam」
    「obanoyama-cho, Stadt Kobe 神戸市篠原伯母野山町長峰山」
    「(25, Juni 1983).」
の 3 段落に割れていた．続きの段落には見出し語が無いので，見出しだけで選ぶと
**日付が欠け，地名も途中で切れる**．
"""
import pandas as pd
import pytest

from comptea import site_notes


def _p(y, text, x1=0, x2=1000, h=40):
    return {'box': (x1, y, x2, y + h), 'text': text}


def test_見出しのある段落を注記とみなす():
    got = site_notes.pick([_p(0, '調査地 Lage: Lfd. Nr.1: 神戸市山田町'),
                           _p(100, 'カワラヨモギ群落は、植生の高さが15~20cmで')])
    assert len(got) == 1 and got[0].startswith('調査地')


def test_続きの段落をつなぐ():
    """見出しの無い続きも，すぐ下にあれば同じ注記"""
    got = site_notes.pick([_p(0, '調査地 Lage: Lfd. Nr.1-10: 神戸市山田町菊水山'),
                           _p(45, 'obanoyama-cho, Stadt Kobe 神戸市篠原'),
                           _p(90, '(25, Juni 1983).')])
    assert len(got) == 1
    assert '神戸市篠原' in got[0] and '1983' in got[0]


def test_離れた段落はつながない():
    got = site_notes.pick([_p(0, '調査地 Lage: Lfd. Nr.1: 神戸市山田町'),
                           _p(600, 'この群落は海岸砂丘に成立している。')])
    assert len(got) == 1 and '海岸砂丘' not in got[0]


def test_次の見出しで切る():
    got = site_notes.pick([_p(0, '調査地 Lage: Lfd. Nr.1: 神戸市'),
                           _p(45, 'Nachweis d. Vegetationsaufnahmen 既発表資料名: Original')])
    assert len(got) == 2


def test_出現1回の種も拾える():
    got = site_notes.pick([_p(0, '出現1回の種 Außerdem je einmal in Lfd. Nr. 1 : Acer')],
                          kinds=('once',))
    assert len(got) == 1


def test_見出しが無ければ何も拾わない():
    assert site_notes.pick([_p(0, 'この群落は海岸砂丘に成立している。')]) == []


def test_地点ごとの情報にする():
    """つないだ注記を `parse_text.parse_site_notes` に通す"""
    got = site_notes.site_info(
        [_p(0, '調査地 Lage d. Aufn.: Lfd. Nr. 1,2: Ryujin-mura 日高郡龍神村'),
         _p(45, 'Datum d. Aufn. 調査年月日: 6 Nov. 1973.')])
    fields = {r['field'] for r in got}
    assert 'locality' in fields
    assert any(r['plot'] == 1 for r in got)


# --- 調査地の文に埋もれた日付を分ける ------------------------------------
#
# 日付は「調査地」の文の中に**括弧付きで**書かれるのが普通で，独立した
# 「調査年月日」の見出しは少ない (注記画像 10 枚で date は 4 件しか出なかった)．
#   「Berg Kikusui, Shimada-cho, Stadt Kobe 神戸市山田町菊水山 (23. Juni 1983)」

def test_括弧の中の日付を取り出す():
    got = site_notes.split_date('神戸市山田町菊水山 (23. Juni 1983)')
    assert got == ('神戸市山田町菊水山', '23. Juni 1983')


def test_括弧が無くても末尾の日付を取り出す():
    got = site_notes.split_date('日高郡龍神村 6 Nov. 1973')
    assert got == ('日高郡龍神村', '6 Nov. 1973')


def test_日付が無ければそのまま():
    assert site_notes.split_date('日高郡龍神村') == ('日高郡龍神村', None)


def test_年だけでも日付とみなす():
    got = site_notes.split_date('滋賀県 1978')
    assert got == ('滋賀県', '1978')


def test_地名の中の数字は日付にしない():
    """「2 丁目」のような数字を日付と取らない"""
    assert site_notes.split_date('山田町2丁目')[1] is None


def test_地点情報の日付を分けて足す():
    recs = [{'plot': 1, 'field': 'locality', 'value': '神戸市山田町 (23. Juni 1983)'}]
    got = site_notes.with_dates(recs)
    loc = [r for r in got if r['field'] == 'locality'][0]
    dates = [r for r in got if r['field'] == 'date']
    assert loc['value'] == '神戸市山田町'
    assert len(dates) == 1 and dates[0]['plot'] == 1
    assert dates[0]['value'] == '23. Juni 1983'


def test_すでに日付がある地点には足さない():
    recs = [{'plot': 1, 'field': 'locality', 'value': '神戸市 (23. Juni 1983)'},
            {'plot': 1, 'field': 'date', 'value': '6 Nov. 1973'}]
    got = site_notes.with_dates(recs)
    assert len([r for r in got if r['field'] == 'date']) == 1


def test_長すぎるものは日付にしない():
    """「出現 1 回の種」の列挙が巻き添えで日付になるのを防ぐ"""
    v = ('神戸市 (Microstegium vimineum ++2, Polygomum nodosum オオイヌタデ +, '
         'Galium kikunugura キクムグラ + 1983)')
    assert site_notes.split_date(v)[1] is None


# --- 表頭の表 (`plot_table`) へ差し込む ------------------------------------
#
# 表頭から取れる項目は紙面によって欠ける (表頭の無い表もある)．
# **注記は欠けた所を埋めるためのもの**なので，**表頭の値は上書きしない**．

def _plots(rows):
    return pd.DataFrame(rows)


def test_空いている項目を注記で埋める():
    df = _plots([{'plot': 1, 'locality': None, 'date': None},
                 {'plot': 2, 'locality': None, 'date': None}])
    recs = [{'plot': 1, 'field': 'locality', 'value': '神戸市山田町'},
            {'plot': 2, 'field': 'date', 'value': '6 Nov. 1973'}]
    got = site_notes.merge_plots(df, recs)
    assert got.loc[got['plot'] == 1, 'locality'].iloc[0] == '神戸市山田町'
    assert got.loc[got['plot'] == 2, 'date'].iloc[0] == '6 Nov. 1973'


def test_表頭の値は上書きしない():
    df = _plots([{'plot': 1, 'locality': '六甲山', 'date': None}])
    recs = [{'plot': 1, 'field': 'locality', 'value': '神戸市山田町'},
            {'plot': 1, 'field': 'date', 'value': '6 Nov. 1973'}]
    got = site_notes.merge_plots(df, recs)
    assert got.loc[0, 'locality'] == '六甲山'          # 表頭が正
    assert got.loc[0, 'date'] == '6 Nov. 1973'         # 空いていた所は埋まる


def test_無い項目の列は足す():
    """`source_ref` (出典) は表頭には無く，注記にしかない"""
    df = _plots([{'plot': 1, 'locality': '六甲山'}])
    recs = [{'plot': 1, 'field': 'source_ref', 'value': '原調査資料 Original'}]
    got = site_notes.merge_plots(df, recs)
    assert got.loc[0, 'source_ref'] == '原調査資料 Original'


def test_表に無い地点は入れない():
    """注記の地点番号の読み違いで，存在しない地点が増えるのを防ぐ"""
    df = _plots([{'plot': 1, 'locality': None}])
    recs = [{'plot': 9, 'field': 'locality', 'value': '神戸市'}]
    got = site_notes.merge_plots(df, recs)
    assert len(got) == 1
    assert got.attrs['warnings']


def test_地点の無い注記は全地点に当てる():
    """1 地点ぶんしか書かれていない注記は `plot` が None になる"""
    df = _plots([{'plot': 1, 'locality': None}, {'plot': 2, 'locality': None}])
    recs = [{'plot': None, 'field': 'locality', 'value': '神戸市山田町'}]
    got = site_notes.merge_plots(df, recs)
    assert list(got['locality']) == ['神戸市山田町', '神戸市山田町']


def test_空の表はそのまま():
    df = _plots([])
    assert len(site_notes.merge_plots(df, [{'plot': 1, 'field': 'date',
                                            'value': '1983'}])) == 0


def test_注記が無ければ何もしない():
    df = _plots([{'plot': 1, 'locality': '六甲山'}])
    got = site_notes.merge_plots(df, [])
    assert got.loc[0, 'locality'] == '六甲山'


# --- 工程につなぐ ----------------------------------------------------------
#
# 注記の画像は，表の画像の隣に `<表の画像>_note.png` として置かれる
# (`split_sheet.note_boxes`)．続きのページでは置き場の `note_block.png`．

class _FakeReader:
    """段落を返すだけの読み手"""

    def __init__(self, paras):
        self.paras = paras
        self.calls = 0

    def available(self):
        return True

    def read_paragraphs(self, img, box=None):
        self.calls += 1
        return self.paras


def test_表の画像の隣の注記の画像を見つける(tmp_path):
    img = tmp_path / 's01115_02_p1.png'
    img.write_bytes(b'')
    note = tmp_path / 's01115_02_p1_note.png'
    note.write_bytes(b'')
    assert site_notes.find_note_image(str(img)) == str(note)


def test_置き場の_note_block_も見る(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    note = work / 'note_block.png'
    note.write_bytes(b'')
    assert site_notes.find_note_image(None, work=str(work)) == str(note)


def test_注記の画像が無ければ_None(tmp_path):
    img = tmp_path / 'a.png'
    img.write_bytes(b'')
    assert site_notes.find_note_image(str(img), work=str(tmp_path)) is None


def test_読み手が無ければ読まない(tmp_path):
    img = tmp_path / 'a_note.png'
    img.write_bytes(b'')
    assert site_notes.read_site_info(str(img), reader=None, use_yomi=False) == []


def test_注記の画像を読んで地点情報にする(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    note = tmp_path / 'a_note.png'
    Image.new('RGB', (60, 40), 'white').save(note)
    reader = _FakeReader([_p(0, '調査地 Lage d. Aufn.: Lfd. Nr. 1,2: 日高郡龍神村'),
                          _p(45, 'Datum d. Aufn. 調査年月日: 6 Nov. 1973.')])
    got = site_notes.read_site_info(str(note), reader=reader)
    assert reader.calls == 1
    assert {r['field'] for r in got} >= {'locality'}


def test_工程に差し込む(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'tab.png'
    Image.new('RGB', (60, 40), 'white').save(img)
    Image.new('RGB', (60, 40), 'white').save(tmp_path / 'tab_note.png')
    df = _plots([{'plot': 1, 'locality': None, 'date': None},
                 {'plot': 2, 'locality': None, 'date': None}])
    reader = _FakeReader([_p(0, '調査地 Lage d. Aufn.: Lfd. Nr. 1,2: 日高郡龍神村')])
    got, info = site_notes.apply_notes(df, image=str(img), reader=reader)
    assert '日高郡龍神村' in str(got.loc[0, 'locality'])
    assert '注記' in info


def test_注記が無ければ表はそのまま(tmp_path):
    df = _plots([{'plot': 1, 'locality': None}])
    got, info = site_notes.apply_notes(df, image=str(tmp_path / 'none.png'))
    assert got.loc[0, 'locality'] is None
    assert info == ''


# --- 本のページは注記が切り出されていない ----------------------------------
#
# 折込は `split_sheet.note_boxes` が `<画像>_note.png` を書き出すが，
# **本のページ (s01114) は注記がページの下にあり，切り出されていない**．
# 2026-09-13 の実測: 80 表のうち 68 表 (85%) の注記がページを読めば取れる．
# ただし**1 枚に 2 表ある紙面では，どちらの表の注記か分けられない**ので使わない．

def test_注記の画像が無ければページを読む(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'kinki_002.png'
    Image.new('RGB', (60, 40), 'white').save(img)
    df = _plots([{'plot': 1, 'locality': None}])
    reader = _FakeReader([_p(0, '調査地 Lage d. Aufn.: Lfd. Nr. 1: 日高郡龍神村')])
    got, info = site_notes.apply_notes(df, image=str(img), reader=reader, page=True)
    assert reader.calls == 1
    assert '日高郡龍神村' in str(got.loc[0, 'locality'])
    assert 'kinki_002' in info


def test_ページは頼まれなければ読まない(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'kinki_002.png'
    Image.new('RGB', (60, 40), 'white').save(img)
    reader = _FakeReader([_p(0, '調査地 Lage: Lfd. Nr. 1: 日高郡龍神村')])
    got, info = site_notes.apply_notes(_plots([{'plot': 1, 'locality': None}]),
                                       image=str(img), reader=reader)
    assert reader.calls == 0 and info == ''


def test_切り出した注記の方を先に使う(tmp_path):
    """`_note.png` があればページは読まない (注記だけの方が確か)"""
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'tab.png'
    Image.new('RGB', (60, 40), 'white').save(img)
    Image.new('RGB', (60, 40), 'white').save(tmp_path / 'tab_note.png')
    reader = _FakeReader([_p(0, '調査地 Lage: Lfd. Nr. 1: 日高郡龍神村')])
    _got, info = site_notes.apply_notes(_plots([{'plot': 1, 'locality': None}]),
                                        image=str(img), reader=reader, page=True)
    assert 'tab_note.png' in info


# --- 「1 回出現の種」も同じ注記から取る ------------------------------------
#
# 折込の注記画像には，地点情報と「1 回出現の種」が並んでいる．
# 2026-09-13 の実測 (注記画像 10 枚): 地点情報 370 件・出現 1 回の種 275 件．
# **同じ (地点, 和名, 階層, 被度) が重なって出る** (長い文を解析するため)．

ONCE = ('出現1回の種 Außerdem je einmal in Lfd. Nr. 3 : '
        'Asplenium oligophlebium カミガモシダ K +, '
        'Lindera strychnifolia テンダイウヤク K +')


def test_出現1回の種を取る():
    got = site_notes.once_species([_p(0, ONCE)])
    names = [r['j_name'] for r in got]
    assert 'カミガモシダ' in names and 'テンダイウヤク' in names
    assert all(r['plot'] == 3 for r in got)


def test_同じものは1件にする():
    """同じ文が 2 つの段落に割れて読まれることがある"""
    got = site_notes.once_species([_p(0, ONCE), _p(45, ONCE)])
    key = [(r['plot'], r['j_name'], r['layer'], r['comp_raw']) for r in got]
    assert len(key) == len(set(key))


def test_階層が違えば別の行():
    """同じ種が高木層と低木層に出るのは正しい"""
    t = ('出現1回の種 Außerdem je einmal in Lfd. Nr. 2 : '
         'Pterostyrax corymbosa アサガラ B2 2, '
         'Pterostyrax corymbosa アサガラ S +')
    got = site_notes.once_species([_p(0, t)])
    assert len({(r['layer'], r['comp_raw']) for r in got}) == 2


def test_注記が無ければ空():
    assert site_notes.once_species([_p(0, 'この群落は海岸砂丘に成立している。')]) == []


# --- 続きのページの切り出しをレイアウト解析で読む --------------------------
#
# 枝番 `-1` のページは注記が次のページ (`-2`) にある (本のページ 80 表のうち
# 12 表)．`cli/link_pages.py` がつないでいるが，読んでいるのは EasyOCR の
# `note.txt` だった．**切り出した `note_block.png` をレイアウト解析で読む**．

def _block(tmp_path, name):
    from PIL import Image

    Image.new('RGB', (60, 40), 'white').save(tmp_path / name)


def test_置き場の注記の切り出しを読む(tmp_path):
    pytest.importorskip('PIL')
    _block(tmp_path, 'note_block.png')
    # **近い段落は「続き」としてつなぐ**のが設計なので，本文は離して置く
    reader = _FakeReader([_p(0, '調査地 Lage: Lfd. Nr. 1: 日高郡龍神村'),
                          _p(600, 'この群落は海岸砂丘に成立している。')])
    got = site_notes.block_text(tmp_path, reader=reader)
    assert '日高郡龍神村' in got
    assert '海岸砂丘' not in got          # 離れた本文は外す


def test_読みは残して二度読まない(tmp_path):
    pytest.importorskip('PIL')
    _block(tmp_path, 'note_block.png')
    reader = _FakeReader([_p(0, '調査地 Lage: Lfd. Nr. 1: 日高郡龍神村')])
    a = site_notes.block_text(tmp_path, reader=reader)
    b = site_notes.block_text(tmp_path, reader=reader)
    assert a == b and reader.calls == 1


def test_出現1回の切り出しも読める(tmp_path):
    pytest.importorskip('PIL')
    _block(tmp_path, 'once_block.png')
    reader = _FakeReader([_p(0, '出現1回の種 Außerdem je einmal in Lfd. Nr. 3 : '
                               'Asplenium oligophlebium カミガモシダ K +')])
    got = site_notes.block_text(tmp_path, name='once_block.png',
                                kinds=('once',), reader=reader)
    assert 'カミガモシダ' in got


def test_切り出しが無ければ空(tmp_path):
    assert site_notes.block_text(tmp_path, reader=_FakeReader([])) == ''


def test_読み手が無ければ空(tmp_path):
    pytest.importorskip('PIL')
    _block(tmp_path, 'note_block.png')
    assert site_notes.block_text(tmp_path, reader=None, use_yomi=False) == ''


# --- 「1 回出現の種」の名前を辞書で直す ------------------------------------
#
# 2026-09-13 の実測 (折込 10 枚・254 件): 形が整っているのは 241 件 (95%) で，
# 崩れは 13 件しかない．**弱点は形でなく字の読み違い**
# (`ASPlenium oligophlebium`・`Lindera strychnl`)．
# 工程の本体と同じ `correct_text.correct_name` に通す．

def test_和名を辞書で直す():
    recs = [{'plot': 1, 'j_name': 'ススキ', 's_name': 'Miscanthus sinensis',
             'layer': 'K', 'comp_raw': '+'}]
    got = site_notes.correct_once(recs)
    assert got[0]['j_name'] == 'ススキ'
    assert got[0]['status'] == 'OK'


def test_読み違いは候補に直す():
    """`correct_name` が当てられるものは直る"""
    recs = [{'plot': 1, 'j_name': 'スス キ', 's_name': '', 'layer': None,
             'comp_raw': '+'}]
    got = site_notes.correct_once(recs)
    assert got[0]['j_name'] == 'ススキ'


def test_学名が空なら和名から引く():
    """**印字されている学名は置き換えない** (2026-09-01 の決定)"""
    recs = [{'plot': 1, 'j_name': 'ススキ', 's_name': '', 'layer': None,
             'comp_raw': '+'}]
    got = site_notes.correct_once(recs)
    assert got[0]['s_name']
    assert got[0].get('note') == '学名は和名から引いた'


def test_印字された学名は置き換えない():
    recs = [{'plot': 1, 'j_name': 'イタドリ', 's_name': 'Polygonum cuspidatum',
             'layer': None, 'comp_raw': '+'}]
    got = site_notes.correct_once(recs)
    assert got[0]['s_name'].startswith('Polygonum')


def test_直せないものは印字を残す():
    recs = [{'plot': 1, 'j_name': 'ヌヌヌヌヌヌ', 's_name': 'Zzzxx qqqvvv',
             'layer': None, 'comp_raw': '+'}]
    got = site_notes.correct_once(recs)
    assert got[0]['j_name'] == 'ヌヌヌヌヌヌ'
    assert got[0]['status'] == 'Need Check'


def test_元の項目は残る():
    recs = [{'plot': 3, 'j_name': 'ススキ', 's_name': '', 'layer': 'K',
             'comp_raw': '+', 'constancy': None}]
    got = site_notes.correct_once(recs)
    assert got[0]['plot'] == 3 and got[0]['layer'] == 'K'
    assert got[0]['comp_raw'] == '+'


# --- 表のページ側の注記も読む ----------------------------------------------
#
# 枝番 `-2` の注記は**文の途中から始まる続き**のことがある (010-2 は
# 「4-6 : Kumihama-cho …」で始まり，「調査地」の見出しは 010-1 側にある)．
# `link_pages` が両方をつなぐには，**表のページ側の注記も読む**必要がある．
# 表のページには `note_block.png` が無いので，ページ自身を読む．

def _located(work, image):
    (work / 'located.csv').write_text(
        'obj_name,source_image\ncomp,' + str(image).replace('\\', '/') + '\n',
        encoding='utf-8')


def test_置き場に切り出しが無ければページを読む(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'kinki_010-1.png'
    Image.new('RGB', (60, 40), 'white').save(img)
    work = tmp_path / 'work'
    work.mkdir()
    _located(work, img)
    reader = _FakeReader([_p(0, '調査地 Lage: Lfd. Nr. 1-3: 神戸市山田町')])
    got = site_notes.work_note_text(work, reader=reader)
    assert '神戸市山田町' in got


def test_切り出しがあればそちらを使う(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    work = tmp_path / 'work'
    work.mkdir()
    Image.new('RGB', (60, 40), 'white').save(work / 'note_block.png')
    img = tmp_path / 'kinki_010-2.png'
    Image.new('RGB', (60, 40), 'white').save(img)
    _located(work, img)
    reader = _FakeReader([_p(0, '調査地 Lage: Lfd. Nr. 4-6: 久美浜町箱石')])
    got = site_notes.work_note_text(work, reader=reader)
    assert '久美浜町箱石' in got
    assert reader.calls == 1              # ページは読まない


def test_1枚に2表ある紙面ではページを読まない(tmp_path):
    """どちらの表の注記か分けられない"""
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'kinki_037.png'
    Image.new('RGB', (60, 40), 'white').save(img)
    work = tmp_path / 'work'
    work.mkdir()
    _located(work, img)
    (work / 'table.txt').write_text('1/2', encoding='utf-8')
    reader = _FakeReader([_p(0, '調査地 Lage: Lfd. Nr. 1: 神戸市')])
    assert site_notes.work_note_text(work, reader=reader) == ''


def test_ページも切り出しも無ければ空(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    assert site_notes.work_note_text(work, reader=_FakeReader([])) == ''


# --- 学名が途中で切れた読みを補う ------------------------------------------
#
# 2026-09-13 の実測 (折込 254 件): **和名は辞書に当たるのに学名が当たらない**
# のが 68 件あり，そのすべてで和名から学名を引ける．しかし中を見ると
# **58 件は本当に別の名前**だった (古い資料の印字と現在の分類の違い．
# `Acanthopanax sclado` → `Chengiopanax sciadophylloides`)．
# **補ってよいのは，印字の読みが引いた名前の頭に収まる 10 件だけ**
# (`Hymenophyllum bar` → `Hymenophyllum barbatum`)．

def test_途中で切れた学名は補う():
    recs = [{'plot': 1, 'j_name': 'コケシノブ', 's_name': 'Hymenophyllum wrig',
             'layer': None, 'comp_raw': '+'}]
    got = site_notes.correct_once(recs)
    assert got[0]['s_name'].startswith('Hymenophyllum')
    assert len(got[0]['s_name']) > len('Hymenophyllum wrig')


def test_別の名前なら印字を残して控える():
    """**印字されている学名は置き換えない** (2026-09-01 の決定)"""
    recs = [{'plot': 1, 'j_name': 'テンダイウヤク', 's_name': 'Lindera strychnl',
             'layer': None, 'comp_raw': '+'}]
    got = site_notes.correct_once(recs)
    assert got[0]['s_name'] == 'Lindera strychnl'
    assert got[0].get('s_name_ref')            # 引いた名前は別に控える
    assert '食い違' in (got[0].get('note') or '')


def test_印字の方が詳しければ触らない():
    """`Euonymus fortunei var. radicans` を `Euonymus fortunei` にはしない"""
    recs = [{'plot': 1, 'j_name': 'ツルマサキ',
             's_name': 'Euonymus fortunei var. radicans',
             'layer': None, 'comp_raw': '+'}]
    got = site_notes.correct_once(recs)
    assert got[0]['s_name'] == 'Euonymus fortunei var. radicans'


# --- 値の末尾の頁番号を落とす ----------------------------------------------
#
# 注記は紙面のいちばん下にあり，**その下の頁番号まで一緒に読まれる**
# (`Original 原調査資料. 151`・`12. Mai, 1983. 113`)．

def test_末尾の頁番号を落とす():
    assert site_notes.drop_page_no('Original 原調査資料. 151') == 'Original 原調査資料'
    assert site_notes.drop_page_no('12. Mai, 1983. 113') == '12. Mai, 1983'


def test_年は落とさない():
    assert site_notes.drop_page_no('6 Nov. 1973') == '6 Nov. 1973'
    assert site_notes.drop_page_no('Juni 1983') == 'Juni 1983'


def test_地名の中の数字は落とさない():
    assert site_notes.drop_page_no('山田町2丁目') == '山田町2丁目'


def test_数字だけなら残す():
    assert site_notes.drop_page_no('151') == '151'


def test_地点情報から頁番号が消える():
    recs = [{'plot': 1, 'field': 'source_ref', 'value': 'Original 原調査資料. 151'}]
    got = site_notes.with_dates(recs)
    assert got[0]['value'] == 'Original 原調査資料'


# --- 表頭の値が壊れているときは，注記で埋める ------------------------------
#
# 2026-09-13 に工程を通して見つかった．kinki_002 の `locality` には表頭から
# `26`・`20/N`・`nan/耳` が入っており (項目の取り違え)，**「表頭は上書き
# しない」規則のせいで，注記の正しい地名が入れなかった**．
# **値がその項目の形をしていなければ，空とみなす**．

def test_地名の形をしていない値は空とみなす():
    assert site_notes.usable_value('locality', '26') is False
    # **`nan` が混じった値**も空とみなす (04_p3 の `nan/M`)．
    # 字数だけで見ると `nan` の 3 字で通ってしまう
    assert site_notes.usable_value('locality', 'nan/M') is False
    assert site_notes.usable_value('locality', 'Nantan-cho 南丹市') is True
    assert site_notes.usable_value('locality', '20/N') is False
    assert site_notes.usable_value('locality', 'nan') is False
    assert site_notes.usable_value('locality', '日高郡龍神村') is True
    assert site_notes.usable_value('locality', 'Ryujin-mura') is True


def test_日付は数字があればよい():
    assert site_notes.usable_value('date', "'83/6/8") is True
    assert site_notes.usable_value('date', '6 Nov. 1973') is True
    assert site_notes.usable_value('date', '-') is False


def test_壊れた表頭の値は注記で埋める():
    # 日付は**そろっている表頭を残す** ('83/6/8 は年月日がそろっている)
    df = _plots([{'plot': 1, 'locality': '26', 'date': "'83/6/8"}])
    recs = [{'plot': 1, 'field': 'locality', 'value': '日高郡龍神村'},
            {'plot': 1, 'field': 'date', 'value': '6 Nov. 1973'}]
    got = site_notes.merge_plots(df, recs)
    assert got.loc[0, 'locality'] == '日高郡龍神村'      # 壊れていたので入る
    assert got.loc[0, 'date'] == "'83/6/8"              # そろっているので残す


def test_正しい表頭の値は残す():
    df = _plots([{'plot': 1, 'locality': '六甲山'}])
    recs = [{'plot': 1, 'field': 'locality', 'value': '日高郡龍神村'}]
    got = site_notes.merge_plots(df, recs)
    assert got.loc[0, 'locality'] == '六甲山'


# --- 日付は「どちらが確かか」で選ぶ ----------------------------------------
#
# 2026-09-13 に折込 (19_p3) で工程を通して見つかった．表頭の `date` に
# 調査番号が混じった `49/83/6`・`6o/83` が入っており，**数字があるので
# 「使える」と見なされて**，注記の `Juni 1983` が入れなかった．
# 表頭の値が**日付として読めない**とき，注記に月か 4 桁の年があれば，
# 注記の方を採る．

def test_日付として読めない表頭は注記で置き換える():
    assert site_notes.better_date('Juni 1983', '49/83/6') is True
    assert site_notes.better_date('Juni 1983', '6o/83') is True


def test_日付として読める表頭は残す():
    assert site_notes.better_date('Juni 1983', "'83/6/8") is False


def test_注記に月も年も無ければ置き換えない():
    assert site_notes.better_date('6', '49/83/6') is False


def test_壊れた日付が注記で入れ替わる():
    df = _plots([{'plot': 1, 'date': '49/83/6', 'locality': '六甲山'}])
    recs = [{'plot': 1, 'field': 'date', 'value': 'Juni 1983'}]
    got = site_notes.merge_plots(df, recs)
    assert got.loc[0, 'date'] == 'Juni 1983'
    assert got.loc[0, 'locality'] == '六甲山'      # 地名はそのまま


# --- 文の途中の括弧の日付も分ける ------------------------------------------
#
# 2026-09-13 に kinki_006 で工程を通して見つかった．地名が次の形で，
# **括弧が末尾でない**ため日付が分けられず，地名に残っていた．
#
#   'Ha- masaka-cho. Mikata-gun,美方郡浜坂町釜屋(5. Juni 1983), in'

def test_途中の括弧の日付を取り出す():
    v = 'Hamasaka-cho, Mikata-gun 美方郡浜坂町釜屋(5. Juni 1983), in'
    head, date = site_notes.split_date(v)
    assert date == '5. Juni 1983'
    assert '1983' not in head
    assert head.endswith('釜屋')          # 末尾の 「, in」 も落とす


def test_途中の括弧が日付でなければ残す():
    v = '神戸市山田町 (北向きの斜面) の林'
    head, date = site_notes.split_date(v)
    assert date is None
    assert '北向き' in head


# --- 大きすぎる紙面はページとして読まない ----------------------------------
#
# 2026-09-13 に折込 (09_p2) の通しで気づいた．注記の切り出しが無い表では
# ページ自身を読むが，**折込は 1782〜12712 px (中央 5872) と大きく**，
# 丸ごと渡すと縮小されて読みが崩れる (既知: A0 の紙面は「"ry AND 1000」)．
# **本のページは 3498〜3510 px** なので，4000 px を境にする．
# 折込の注記は `<画像>_note.png` に切り出されているので，外しても困らない．

def test_大きすぎる画像は丸ごと読まない(tmp_path):
    """**丸ごとは読まず，下の帯だけ読む** (2026-09-13 に方針を詰めた)

    最初は「大きければ読まない」にしたが，実データを見ると**切り出しの無い
    折込にも紙面の下に注記があった** (04_p3 で 12 件・19_p1 で 54 件)．
    """
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'huge.png'
    Image.new('RGB', (3000, 9000), 'white').save(img)
    assert site_notes._page_ok(str(img)) is False


def test_本のページの大きさなら読む(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'page.png'
    Image.new('RGB', (2400, 3510), 'white').save(img)   # 本のページの実寸
    reader = _FakeReader([_p(0, '調査地 Lage: Lfd. Nr. 1: 神戸市山田町')])
    got, _info = site_notes.apply_notes(_plots([{'plot': 1, 'locality': None}]),
                                        image=str(img), reader=reader, page=True)
    assert '神戸市山田町' in str(got.loc[0, 'locality'])


def test_切り出した注記は大きくても読む(tmp_path):
    """`_note.png` は注記だけを切り出したもの．大きさで外さない"""
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'tab.png'
    Image.new('RGB', (3000, 9000), 'white').save(img)
    Image.new('RGB', (3757, 1612), 'white').save(tmp_path / 'tab_note.png')
    reader = _FakeReader([_p(0, '調査地 Lage: Lfd. Nr. 1: 神戸市山田町')])
    _got, info = site_notes.apply_notes(_plots([{'plot': 1, 'locality': None}]),
                                        image=str(img), reader=reader, page=True)
    assert 'tab_note.png' in info


# --- 大きい紙面は「下の帯」だけ読む ----------------------------------------
#
# 2026-09-13 に折込の通しで分かった．**注記の切り出しが無い表にも，
# 紙面の下に注記がある** (04_p3 で 12 件・03_p1 で 27 件・19_p1 で 54 件)．
# 丸ごと渡すと縮小されて崩れるが，**下の帯だけなら読める**
# (高さが 1/5 になるので縮尺が戻る)．

class _SizeReader(_FakeReader):
    """渡された画像の大きさを憶える読み手"""

    def __init__(self, paras):
        super().__init__(paras)
        self.sizes = []

    def read_paragraphs(self, img, box=None):
        self.sizes.append(img.size)
        return super().read_paragraphs(img, box=box)


def test_大きい紙面は下の帯だけ読む(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'tall.png'
    Image.new('RGB', (3000, 9000), 'white').save(img)
    reader = _SizeReader([_p(0, '調査地 Lage: Lfd. Nr. 1: 神戸市山田町')])
    got, info = site_notes.apply_notes(_plots([{'plot': 1, 'locality': None}]),
                                       image=str(img), reader=reader, page=True)
    assert reader.calls == 1
    assert reader.sizes[0] == (3000, int(9000 * site_notes.STRIP))
    assert '神戸市山田町' in str(got.loc[0, 'locality'])
    assert '下の帯' in info


def test_小さい紙面は丸ごと読む(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'page.png'
    Image.new('RGB', (2400, 3510), 'white').save(img)
    reader = _SizeReader([_p(0, '調査地 Lage: Lfd. Nr. 1: 神戸市山田町')])
    site_notes.apply_notes(_plots([{'plot': 1, 'locality': None}]),
                           image=str(img), reader=reader, page=True)
    assert reader.sizes[0] == (2400, 3510)


def test_切り出しがあれば帯は読まない(tmp_path):
    pytest.importorskip('PIL')
    from PIL import Image

    img = tmp_path / 'tab.png'
    Image.new('RGB', (3000, 9000), 'white').save(img)
    Image.new('RGB', (3000, 900), 'white').save(tmp_path / 'tab_note.png')
    reader = _SizeReader([_p(0, '調査地 Lage: Lfd. Nr. 1: 神戸市山田町')])
    _got, info = site_notes.apply_notes(_plots([{'plot': 1, 'locality': None}]),
                                        image=str(img), reader=reader, page=True)
    assert reader.sizes[0] == (3000, 900)
    assert 'tab_note.png' in info
