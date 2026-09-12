"""yomitoku を読み手として使う (yomi.py)

yomitoku (https://github.com/kotaro-kinoshita/yomitoku) を**外部プロセス**として
呼びます．torch で動くので，**主環境を汚さないよう別の環境に入れて**そこの
python を指します (`COMPTEA_YOMI_PY`)．

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
DEFAULT_DEVICE = 'cuda'

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


def find_python():
    """yomitoku を入れた環境の python．無ければ None"""
    p = os.environ.get(ENV_PY)
    return p if p and os.path.isfile(p) else None


class YomiReader:
    """`read_boxes(img, box)` で読む (`read_region` が使う形)"""

    def __init__(self, python=None, device=DEFAULT_DEVICE):
        self.python = python or find_python()
        self.device = device
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
