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

    def __init__(self, ndl_dir=None, python=None, device=None):
        self.dir = ndl_dir or find_dir()
        self.python = python or sys.executable
        from . import device as _device

        # ONNX なので **cpu でも速い**．指定が無ければ環境に合わせる
        self.device = _device.pick(device)
        self.calls = 0

    def available(self):
        return self.dir is not None

    def read_crops(self, img, boxes, pad=0):
        """**セルをまとめて読む** (1 回のプロセスで)

        2026-09-14 の実測: 読めなかった組成のセルは **66 個中 44 個 (67%)**
        を NDLOCR-Lite が読めた (EasyOCR は 5 個)．ただし 1 セルずつ呼ぶと
        **1 個 15 秒**かかる (別プロセスの起動と模型の読み込み)．
        NDLOCR-Lite は**ディレクトリを丸ごと読む**ので，セルを 1 枚ずつ
        画像にして並べれば 1 回で済む．

        **領域をまとめて読む形は駄目だった** (488 セル中 53 個しか
        割り当たらない)．縦や格子に並べた 1 枚も駄目．**1 セル 1 画像**が要る．

        Returns:
            箱と同じ並びの文字列 (読めなければ空文字)
        """
        boxes = list(boxes or [])
        if not boxes:
            return []
        if not self.available():
            return [''] * len(boxes)
        self.calls += 1
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, 'in'), os.path.join(tmp, 'out')
            os.makedirs(src)
            os.makedirs(dst)
            names = []
            for i, b in enumerate(boxes):
                x1, y1 = int(b[0]) - pad, int(b[1]) - pad
                x2, y2 = int(b[2]) + pad, int(b[3]) + pad
                name = f'c{i:06d}'
                names.append(name)
                img.crop((max(0, x1), max(0, y1), x2, y2)).convert('RGB').save(
                    os.path.join(src, name + '.png'))
            subprocess.run(
                [self.python, 'ocr.py', '--sourcedir', src, '--output', dst,
                 '--device', self.device],
                cwd=self.dir, capture_output=True, text=True,
                encoding='utf-8', errors='replace')
            out = []
            for name in names:
                f = os.path.join(dst, name + '.json')
                if not os.path.isfile(f):
                    out.append('')
                    continue
                try:
                    with open(f, encoding='utf-8') as fh:
                        doc = json.load(fh)
                except Exception:
                    out.append('')
                    continue
                out.append(''.join(t for _b, t, _c in parse(doc)))
        return out

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
