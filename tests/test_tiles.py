"""大きな紙面を分割して読み，元の座標へ戻す (tiles.py)

A0 級の紙面をそのまま読み手に渡すと**縮小されて読みが崩れる**
(s01115_16_p2 の紙面全体は「"ry AND 1000」になった)．A3〜A4 ほどに分けて読み，
**切り出しの原点を足して元の座標へ戻す**．重なった読みは**確信度の大きい方**を採る．
"""
import pytest

from comptea import tiles


def test_紙面を重なり付きで分ける():
    got = tiles.split((1000, 500), size=400, overlap=0.25)
    assert got[0] == (0, 0, 400, 400)
    # 重なりのぶんだけ進む (400 - 100 = 300)
    assert (300, 0, 700, 400) in got
    assert all(x2 <= 1000 and y2 <= 500 for _x, _y, x2, y2 in got)


def test_小さい紙面は1枚のまま():
    assert tiles.split((300, 200), size=400) == [(0, 0, 300, 200)]


def test_端は紙面に収める():
    got = tiles.split((500, 300), size=400, overlap=0.25)
    assert all(x2 <= 500 and y2 <= 300 for _x, _y, x2, y2 in got)
    # 端のタイルも十分な大きさを保つ (紙面の端から逆算する)
    assert any(x1 == 100 for x1, _y, _x2, _y2 in got) or len(got) == 1


def test_読みを元の座標へ戻す():
    got = tiles.shift([((10, 20, 60, 40), 'あ', 0.9)], origin=(100, 200))
    assert got == [((110, 220, 160, 240), 'あ', 0.9)]


def test_重なった読みは確信度の大きい方を採る():
    a = ((100, 100, 200, 140), 'アカマツ', 0.6)
    b = ((102, 101, 198, 139), 'アカマシ', 0.9)
    got = tiles.merge([a, b])
    assert [t for _b, t, _c in got] == ['アカマシ']


def test_離れた読みは両方残す():
    a = ((100, 100, 200, 140), 'あ', 0.6)
    b = ((500, 500, 600, 540), 'い', 0.9)
    assert len(tiles.merge([a, b])) == 2


def test_重なりの判定は面積の割合で見る():
    """少し触れただけの別の語は残す"""
    a = ((100, 100, 200, 140), 'あ', 0.6)
    b = ((195, 100, 295, 140), 'い', 0.9)
    assert len(tiles.merge([a, b], least=0.5)) == 2


class FakeReader:
    """タイルごとに，その中の決まった位置の読みを返す偽の読み手"""

    def __init__(self):
        self.boxes = []

    def read_boxes(self, img, box=None):
        self.boxes.append(box)
        # タイルの左上から 10 px の所に読みがあるものとする
        return [((10, 10, 60, 30), f'T{len(self.boxes)}', 0.8)]


def test_タイルごとに読んで元の座標で返す():
    PIL = pytest.importorskip('PIL')
    from PIL import Image
    im = Image.new('RGB', (900, 400), 'white')
    r = FakeReader()
    got = tiles.read_tiled(im, r, size=400, overlap=0.25)
    assert len(r.boxes) >= 2                      # 分けて読んだ
    # どの読みも紙面の中にある
    assert all(0 <= b[0] and b[2] <= 900 for b, _t, _c in got)
    # 2 枚目のタイルの読みは，原点のぶんだけ右にある
    assert any(b[0] > 100 for b, _t, _c in got)
