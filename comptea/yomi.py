"""yomitoku を読み手として使う (yomi.py)

yomitoku (https://github.com/kotaro-kinoshita/yomitoku) を**外部プロセス**として
呼びます．torch で動くので，**主環境を汚さないよう別の環境に入れて**そこの
python を指します．

## 入れ方 (2026-09-14 に正式導入)

    python -m venv --system-site-packages <置き場>/venv_yomi
    <置き場>/venv_yomi/Scripts/pip install yomitoku==0.14.0

`--system-site-packages` にするのは，**主環境の torch を使い回す**ためです
(入れ直すと数 GB になる)．**置き場は環境変数 `COMPTEA_YOMI_PY` で指します**
(`setx COMPTEA_YOMI_PY <置き場>\\venv_yomi\\Scripts\\python.exe` のように
恒久設定しておけば，呼ぶたびに指す必要はありません)．
**公開リポジトリなので，手元の実際のパスはここに書きません**．
**入っていなければ黙って飛ばす**ので，入れていない環境でも工程は動きます
(`--reader multi` が yomitoku を使わなくなるだけ)．
重みは初回の呼び出しのときに取りに行きます (以後は使い回す)．

NDLOCR-Lite (`comptea.ndl`) が**行単位**で返すのに対し，yomitoku は**語単位**で
返すので，格子のセルに割り当てるのに向きます．

2026-09-12 の実測 (真値と突き合わせた一致数．工程と同じ補正をかけたうえで):

    kinki_010-1 の学名   EasyOCR 13 → yomitoku 24 (セル 27)
    kinki_047   の学名   EasyOCR 11 → yomitoku 31 (セル 61)
    kinki_047   の和名   EasyOCR 43・yomitoku 42 → **合わせて 45**

**読み手ごとに落とすセルが違う**ので，重ねると増えます．
"""
import json
import os
import subprocess
import tempfile

ENV_PY = 'COMPTEA_YOMI_PY'
# 決まった置き場 (2026-09-14 に正式導入．NDLOCR-Lite の隣に置いた)．
# `COMPTEA_YOMI_PY` を指していれば，そちらが優先される．
# `ndl.DEFAULT_DIRS` と同じ考え方: **環境変数なしでも動くようにする**
# **手元の実際の置き場は書かない** (公開リポジトリのため．2026-09-14 ユーザ決定)．
# 場所は環境変数 `COMPTEA_YOMI_PY` で指す．ここは**自分の環境だけで**
# 足したいときの逃げ道として残してある (足しても公開の repo には出さない)
DEFAULT_PYS = ()
# **GPU があるとは限らない**．`comptea.device` が決める (`COMPTEA_DEVICE`)

WORKER = '''
import json
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from yomitoku import OCR

src, dst, device = sys.argv[1], sys.argv[2], sys.argv[3]
res, _ = OCR(visualize=False, device=device)(
    np.asarray(Image.open(src).convert('RGB'))[:, :, ::-1])
out = []
for w in (getattr(res, 'words', None) or []):
    pts = getattr(w, 'points', None) or []
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    if not xs:
        continue
    out.append([[min(xs), min(ys), max(xs), max(ys)],
                getattr(w, 'content', ''),
                float(getattr(w, 'rec_score', 0) or 0)])
json.dump(out, open(dst, 'w', encoding='utf-8'), ensure_ascii=False)
'''


# 表の下の注記は**文章**なので，語ではなく**段落**で受け取る．
# `DocumentAnalyzer` は (結果, 図, 表) の 3 つを返すことがあるので先頭を採る．
PARA_WORKER = """
import json
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from yomitoku import DocumentAnalyzer

src, dst, device = sys.argv[1], sys.argv[2], sys.argv[3]
got = DocumentAnalyzer(visualize=False, device=device)(
    np.asarray(Image.open(src).convert('RGB'))[:, :, ::-1])
res = got[0] if isinstance(got, (list, tuple)) else got
out = []
for p in (getattr(res, 'paragraphs', None) or []):
    box = getattr(p, 'box', None) or []
    if len(box) < 4:
        continue
    out.append([[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                getattr(p, 'contents', '') or '',
                float(getattr(p, 'role', None) is None)])
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
"""


def find_python():
    """yomitoku を入れた環境の python．無ければ None

    (1) `COMPTEA_YOMI_PY` (2) `DEFAULT_PYS` の順に見ます．**環境変数を
    指していなくても，決まった置き場にあれば使えます** (2026-09-14)．
    """
    for p in (os.environ.get(ENV_PY),) + DEFAULT_PYS:
        if p and os.path.isfile(p):
            return p
    return None


class YomiReader:
    """`read_boxes(img, box)` で読む (`read_region` が使う形)"""

    def __init__(self, python=None, device=None):
        from . import device as _device

        self.python = python or find_python()
        self.device = _device.pick(device)
        self.calls = 0

    def available(self):
        return self.python is not None

    def read_boxes(self, img, box=None):
        """`img` の `box` の範囲を読み，**元の画像の座標**で返す"""
        if not self.available():
            return []
        self.calls += 1
        x1, y1 = (int(box[0]), int(box[1])) if box else (0, 0)
        crop = img.crop((x1, y1, int(box[2]), int(box[3]))) if box else img
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'a.png')
            dst = os.path.join(tmp, 'a.json')
            work = os.path.join(tmp, 'w.py')
            with open(work, 'w', encoding='utf-8') as f:
                f.write(WORKER)
            crop.convert('RGB').save(src)
            subprocess.run([self.python, work, src, dst, self.device],
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace')
            if not os.path.isfile(dst):
                return []
            with open(dst, encoding='utf-8') as f:
                got = json.load(f)
        return [((b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1), t)
                for b, t, _c in got if (t or '').strip()]

    def read_paragraphs(self, img, box=None):
        """段落で読む (レイアウト解析)．[{'box', 'text'}] を返す

        表の下の注記は文章なので，語ではなく段落で受け取ります
        (`site_notes.pick` が見出しと続きをつなぐ)．
        """
        got = self._run(img, box, PARA_WORKER)
        x1, y1 = (int(box[0]), int(box[1])) if box else (0, 0)
        return [{'box': (b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1),
                 'text': t}
                for b, t, _c in got if (t or '').strip()]

    def _run(self, img, box, worker):
        """別の環境の python で `worker` を走らせ，読みの一覧を受け取る"""
        if not self.available():
            return []
        self.calls += 1
        crop = img.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3]))) \
            if box else img
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'a.png')
            dst = os.path.join(tmp, 'a.json')
            work = os.path.join(tmp, 'w.py')
            with open(work, 'w', encoding='utf-8') as f:
                f.write(worker)
            crop.convert('RGB').save(src)
            subprocess.run([self.python, work, src, dst, self.device],
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace')
            if not os.path.isfile(dst):
                return []
            with open(dst, encoding='utf-8') as f:
                return json.load(f)
