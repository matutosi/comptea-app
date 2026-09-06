"""各スクリプトが共通で使う準備

中核のモジュールは相対 import と相対パスの辞書(`j_name.txt` など)に
依存しているため，**必ず中核の場所を作業ディレクトリにしてから** import する．
利用者から受け取ったパスは，chdir する前に絶対パスへ直しておく．
"""
import os
import sys
from pathlib import Path

# 中核の置き場．`comptea/` が正で，`yolo/` は古い名前(2026-09 に改めた)
CORE_DIRS = ('comptea', 'yolo')


def yolo_dir() -> Path:
    """中核のモジュールの場所を返す

    環境変数 COMPTEA_YOLO があればそれを使う．
    無ければこのファイルの位置から遡って探す．
    """
    env = os.environ.get('COMPTEA_YOLO')
    if env:
        d = Path(env).resolve()
        if (d / 'locate.py').is_file():
            return d
        raise SystemExit(f'COMPTEA_YOLO に locate.py が無い: {d}')
    here = Path(__file__).resolve()
    for parent in here.parents:
        for name in CORE_DIRS:
            d = parent / name
            if (d / 'locate.py').is_file():
                return d
    raise SystemExit('中核のモジュールが見つからない．'
                     'COMPTEA_YOLO で場所を指定する')


def setup(paths=()):
    """yolo/ へ chdir し，import できるようにする

    Args:
        paths: 利用者から受け取ったパス(chdir 前に絶対パスへ直す)
    Returns:
        絶対パスにした paths のリスト
    """
    # Windows の既定(cp932)だと，出力を渡したときに日本語が化ける
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass
    resolved = [str(Path(p).resolve()) if p else p for p in paths]
    d = yolo_dir()
    sys.path.insert(0, str(d))
    os.chdir(d)
    return resolved


def workdir(image: str, workdir_opt: str = None, make: bool = True) -> Path:
    """中間物を置く場所

    既定は `yolo/work/<画像名>/`．画像1枚ぶんをひとまとめにする．

    Args:
        make: 作らずに場所だけ返すなら False．
              1ページに表が2つ以上あるときは `_t1`・`_t2` を後ろに足すので，
              ここで作ると**中身の無い置き場**が残り，次の工程が迷う
    """
    if workdir_opt:
        d = Path(workdir_opt).resolve()
    else:
        d = yolo_dir() / 'work' / Path(image).stem
    if make:
        d.mkdir(parents=True, exist_ok=True)
    return d


def need_file(path: Path, how: str):
    """前の工程の出力が無いときは，何を先に実行するかを言って止まる"""
    if not Path(path).is_file():
        raise SystemExit(f'{path} が無い．先に {how} を実行する')


# 目で確かめる必要が高い警告の目印(2026-09-01 に全88枚を通して決めた)．
# 1枚あたり警告は中央値 5 件あり，全部を並べると本当に見るべきものが埋もれる．
MUST_CHECK = (
    '行ぶん下まで伸びている', '行ぶん上まで伸びている',   # 行が丸ごと落ちている
    '1件も検出されなかった', '出力できない', '出力しない',  # 取れなかったもの
    '数を決められない',                                    # 行数・列数が定まらない
    'まま', '割れて',                                      # 字に重なったまま
    '拾えていない項目',                                    # 表頭の抜け
    '別々の表', '種名の列が拾えていない',                  # 1ページに表が2つ以上
    '拾い直した',                                          # 閾値を下げて取り直した
    '短冊',                                                # 横長の表を分けて検出した
    '地点の隙間から',                                      # 列の境が印字と合っていない
    '種名の列には字が', '組成部には字の行が',              # 行が丸ごと落ちている
    '行の高さが極端に',                                    # 表頭を 1 行に飲み込んだ
    '表頭の値の行',                                        # 表頭が組成部まで下りた
)


def sort_warnings(warnings):
    """警告を「要確認」と「参考」に分ける

    Returns:
        (要確認のリスト, 参考のリスト)
    """
    must, note = [], []
    for w in list(warnings or []):
        (must if any(k in str(w) for k in MUST_CHECK) else note).append(w)
    return must, note


def show_warnings(warnings, head='--- warnings ---'):
    """attrs['warnings'] を，黙って捨てずに表示する

    **要確認のものを先に出す**．どれも情報としては要るが，
    全部を同じ重みで並べると関門で選別できない．
    """
    warnings = list(warnings or [])
    if not warnings:
        return
    must, note = sort_warnings(warnings)
    print(head)
    for w in must:
        print(f'  !! {w}')
    for w in note:
        print(f'  ! {w}')
    if must:
        print(f'  ({len(must)} 件が要確認 / {len(note)} 件は参考)')
