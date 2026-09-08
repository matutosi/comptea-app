"""物差しが使うデータの置き場を，環境変数で指す

`eval/` の 6 本は，ラベルを付けたスキャン(`labelme_data/`)と
人が書き起こした正解表(`truth/`)が要る．どちらも著作権のため公開しておらず，
**このリポジトリの外**にある．

これまでは「データの置き場を作業ディレクトリにして呼ぶ」約束だったが，
置き場が動くと呼び方も変わる．**環境変数 `COMPTEA_DATA` で指せる**ようにした
(2026-09-08)．設定してあれば，どこから呼んでも同じように動く．

    set COMPTEA_DATA=<データの置き場>      (Windows)
    export COMPTEA_DATA=<データの置き場>   (bash)

**実際の場所はここに書かない**．人によって違い，公開する情報でもない．
"""
import os
from pathlib import Path

ENV = 'COMPTEA_DATA'
# データの置き場だと見分ける目印．どちらかがあれば置き場とみなす
MARKS = ('labelme_data', 'truth')


def data_dir():
    """データの置き場(環境変数が無ければ，いまいる場所)"""
    env = os.environ.get(ENV)
    return Path(env).resolve() if env else Path.cwd()


def looks_like_data(d) -> bool:
    """そこがデータの置き場に見えるか"""
    d = Path(d)
    return any((d / m).is_dir() for m in MARKS)


def use_data_dir():
    """データの置き場へ移る(環境変数が設定されているときだけ)

    物差しの既定値は `labelme_data/…`・`truth/…` のような相対の場所なので，
    **置き場へ移ってから**動かす．設定が無ければ，これまでどおり
    いまいる場所を使う(呼び方は変わらない)．

    Returns:
        移った先の場所(移らなかったときは，いまいる場所)
    """
    env = os.environ.get(ENV)
    if not env:
        return Path.cwd()
    d = Path(env).resolve()
    if not d.is_dir():
        raise SystemExit(f'{ENV} の場所が無い: {d}')
    if not looks_like_data(d):
        raise SystemExit(
            f'{ENV} にデータが見あたらない: {d}\n'
            f'  {" / ".join(MARKS)} のどれかがある場所を指す')
    os.chdir(d)
    return d
