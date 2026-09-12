"""NDLOCR-Lite を読み手として使う (ndl.py)

国立国会図書館の NDLOCR-Lite (https://github.com/ndl-lab/ndlocr-lite) を，
**外部プロセス**として呼びます．ONNX で動き **GPU が要りません**．

2026-09-12 の実測 (s01115_16_p2 の表頭の切り出し):

    EasyOCR      項目名 5 個   8.8 秒   (セルごとに読む今のやり方)
    NDLOCR-Lite  項目名 10 個  1.6 秒   (領域を 1 回読む)

**主環境を汚さないため，外部プロセスのまま使います**．領域単位で呼ぶので
呼び出し回数は少なく，遅さは問題になりません (紙面全体をタイルごとに呼ぶと
336 秒かかったので，そういう使い方はしません)．

置き場は環境変数 `COMPTEA_NDLOCR` で指すか，既定の場所を探します．
"""
import json
import os
import subprocess
import sys
import tempfile

DEFAULT_DIRS = (r'D:\pf\dos\ndlocr-lite\src',)
ENV = 'COMPTEA_NDLOCR'


def find_dir():
    """NDLOCR-Lite の `src` の場所．見つからなければ None"""
    p = os.environ.get(ENV)
    cands = ([p, os.path.join(p or '', 'src')] if p else []) + list(DEFAULT_DIRS)
    for d in cands:
        if d and os.path.isfile(os.path.join(d, 'ocr.py')):
            return d
    return None


def parse(doc):
    """NDLOCR-Lite の JSON から [((x1, y1, x2, y2), 文字列, 確信度)] を作る"""
    out = []
    for page in (doc or {}).get('contents', []) or []:
        for c in page or []:
            b = c.get('boundingBox') or []
            t = (c.get('text') or '').strip()
            if len(b) < 4 or not t:
                continue
            xs = [float(p[0]) for p in b]
            ys = [float(p[1]) for p in b]
            out.append(((min(xs), min(ys), max(xs), max(ys)), t,
                        float(c.get('confidence', 1.0))))
    return out


class NdlReader:
    """`read_boxes(img, box)` で読む (`read_region` が使う形)

    Args:
        device: 'cpu' か 'cuda'．既定は cpu (GPU が要らないのが取り柄)
    """

    def __init__(self, ndl_dir=None, python=None, device='cpu'):
        self.dir = ndl_dir or find_dir()
        self.python = python or sys.executable
        self.device = device
        self.calls = 0

    def available(self):
        return self.dir is not None

    def read_boxes(self, img, box=None):
        """`img` の `box` の範囲を読み，**元の画像の座標**で返す"""
        if not self.available():
            return []
        self.calls += 1
        x1, y1 = (int(box[0]), int(box[1])) if box else (0, 0)
        crop = img.crop((x1, y1, int(box[2]), int(box[3]))) if box else img
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, 'in'), os.path.join(tmp, 'out')
            os.makedirs(src)
            os.makedirs(dst)
            crop.convert('RGB').save(os.path.join(src, 'a.png'))
            subprocess.run(
                [self.python, 'ocr.py', '--sourcedir', src, '--output', dst,
                 '--device', self.device],
                cwd=self.dir, capture_output=True, text=True,
                encoding='utf-8', errors='replace')
            f = os.path.join(dst, 'a.json')
            if not os.path.isfile(f):
                return []
            with open(f, encoding='utf-8') as fh:
                doc = json.load(fh)
        return [((b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1), t)
                for b, t, _c in parse(doc)]
