"""段階 1 (格子) を，保存した検出で通す

検出器 (ultralytics) を差し替え，見本の zip の `detect.csv` を返させる．
CI には検出器も EasyOCR も入れないので，これで段階 1 の本体
(`build_one`・`locate_items`・列の境の直し・overlay) が毎回通る．
網羅が 29% だった `pipeline/grid.py` の歯止め．
"""
import shutil
import zipfile

import pandas as pd
import pytest

import conftest
from comptea import header_lines
from comptea.pipeline import grid


@pytest.fixture
def saved(tmp_path, monkeypatch):
    with zipfile.ZipFile(f'{conftest.ROOT}/examples/sample_grid.zip') as z:
        det = pd.read_csv(z.open('detect.csv'))
    img = tmp_path / 'sample.jpg'
    shutil.copy(conftest.SAMPLE, img)

    def fake_detect(image, weights, conf, by_class, imgsz, source=None):
        d = det.copy()
        d['source_image'] = str(source if source is not None else image).replace('\\', '/')
        return d

    monkeypatch.setattr(grid, 'detect', fake_detect)
    # 表頭の字の箱は EasyOCR が要る．無い場と同じにする (帯は黒画素から作る)
    monkeypatch.setattr(header_lines, '_reader', lambda: None)
    return img, tmp_path / 'work'


def test_保存した検出から格子と絵ができる(saved):
    img, work = saved
    code = grid.main([str(img), '--workdir', str(work), '--imgsz', '1280'])
    assert code in (0, None)
    loc = pd.read_csv(work / 'located.csv')
    comp = loc[loc['obj_name'] == 'comp']
    assert comp['col'].nunique() == 6             # 見本は 6 地点
    assert 25 <= comp['row'].nunique() <= 30      # 種の行 (見出しの行を含む)
    assert (work / 'overlay.png').is_file()
    assert (work / 'summary.txt').is_file()
    # 出力に手元の置き場を書かない (重みは名前だけ)
    assert set(loc['model'].dropna()) <= {'comptea.pt'}
