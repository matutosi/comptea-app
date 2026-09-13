"""表の下の注記から地点情報を取る (site_notes.py)

レイアウト解析は**1 つの注記を複数の段落に割る**ことがある．
s01115_19_p3 の注記は
    「調査地 Lage: Lfd. Nr.1-10: … 神戸市山田町菊水山 Juni 1983), 11-15: Berg Nagam」
    「obanoyama-cho, Stadt Kobe 神戸市篠原伯母野山町長峰山」
    「(25, Juni 1983).」
の 3 段落に割れていた．続きの段落には見出し語が無いので，見出しだけで選ぶと
**日付が欠け，地名も途中で切れる**．
"""
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
