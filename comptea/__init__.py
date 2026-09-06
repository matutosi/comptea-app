"""comptea — 植生学の組成表を，構造化データにする

スキャンした組成表の画像から，縦持ちの表 (1 行 = 1 地点 × 1 種) を取り出す．

    from comptea import correct_text
    correct_text.correct_comp('5・5')      # {'corrected': '5;5', 'status': 'OK'}

**ここでは重いものを読み込まない**．`ultralytics`・`easyocr`・`torch` は，
使う関数の中で読む(検出も読み取りもしない工程では，読み込みだけで数秒かかる
ものを抱えたくない)．

工程の流れは docs/pipeline.md，コードの構成は docs/architecture.md を見る．
"""
import os

__version__ = '0.1.0'

# 辞書と重みの置き場．**パッケージの中を基準にする**(2026-09-07)．
# 以前は相対パスで開いていたので，`comptea/` を作業ディレクトリにしないと
# 動かなかった．どこから呼ばれても開けるようにする
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(DATA_DIR, 'weights', 'comptea.pt')


def data_path(name):
    """パッケージに同梱したファイルの場所を返す(辞書・重み)

    渡された名前がすでに絶対パスか，手元にあるファイルならそのまま返す．
    利用者が自前の辞書を渡せるようにするため．
    """
    if os.path.isabs(name) or os.path.isfile(name):
        return name
    return os.path.join(DATA_DIR, name)
