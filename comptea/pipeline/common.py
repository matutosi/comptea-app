"""各スクリプトが共通で使う準備

中核は `comptea` パッケージ．**入れてあればそのまま読め**，入れていなければ
このリポジトリの置き場を `sys.path` に足す(`pip install -e .` をしていない
利用者のため)．

**作業ディレクトリは変えない**(2026-09-07)．以前は中核が相対 import と
相対パスの辞書に頼っていたので `comptea/` へ移る必要があったが，
パッケージにしたのでどこから呼んでもよくなった．
"""
import os
import sys
from pathlib import Path


def package_dir() -> Path:
    """中核 (`comptea` パッケージ) の場所を返す

    環境変数 COMPTEA_CORE があればそれを使う(古い名前 COMPTEA_YOLO も見る)．
    無ければ入っているものを探し，それも無ければこのファイルの位置から遡る．
    """
    env = os.environ.get('COMPTEA_CORE') or os.environ.get('COMPTEA_YOLO')
    if env:
        d = Path(env).resolve()
        if (d / 'locate.py').is_file():
            return d
        raise SystemExit(f'COMPTEA_CORE に locate.py が無い: {d}')
    here = Path(__file__).resolve()
    for parent in here.parents:
        for name in ('comptea', 'yolo'):
            d = parent / name
            if (d / 'locate.py').is_file():
                return d
    try:
        import comptea
        return Path(comptea.__file__).parent
    except ImportError:
        raise SystemExit('中核のモジュールが見つからない．'
                         'COMPTEA_CORE で場所を指定するか，pip install する')


def setup(paths=()):
    """`comptea` を読めるようにする

    Args:
        paths: 利用者から受け取ったパス(絶対パスへ直して返す)
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
    try:
        import comptea                                        # noqa: F401
    except ImportError:
        root = str(package_dir().parent)
        if root not in sys.path:
            sys.path.insert(0, root)
    return resolved


def workdir(image: str, workdir_opt: str = None, make: bool = True) -> Path:
    """中間物を置く場所

    既定は**いまいる場所**の `work/<画像名>/`．画像1枚ぶんをひとまとめにする．
    (2026-09-07 まではパッケージの中の `work/` だった．入れて使うと
     書き込めない場所になるので，呼んだ場所を基準にする)

    Args:
        make: 作らずに場所だけ返すなら False．
              1ページに表が2つ以上あるときは `_t1`・`_t2` を後ろに足すので，
              ここで作ると**中身の無い置き場**が残り，次の工程が迷う
    """
    if workdir_opt:
        d = Path(workdir_opt).resolve()
    else:
        d = Path.cwd() / 'work' / Path(image).stem
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
    全部を同じ重みで並べると，目で確かめるときに選別できない．
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


def code_version(root=None):
    """いまのコードの版 (コミットの短い名と，**未コミットの変更があるか**)

    2026-09-16 に足した．保存した通しの結果が**コミットしていない作業ツリーの
    状態**で回されていたのに気づかず，「17_p1 が 22 件後退した」と取り違えた
    (実際は読み直しの余白が 3 ではなく広いままだった)．版と変更の有無が
    結果に残っていれば，比べる前に 1 行で気づける．

    Returns:
        {'commit': 短い名, 'dirty': bool} か，git が無い環境では None
    """
    import subprocess
    root = Path(root) if root else package_dir().parent
    try:
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return None
        # **中核のコードだけ**を見る (文書や的の変更は結果を変えない)
        st = subprocess.run(['git', '-C', str(root), 'status', '--porcelain',
                             '--', 'comptea'],
                            capture_output=True, text=True, timeout=10)
        return {'commit': head.stdout.strip(),
                'dirty': bool(st.stdout.strip()) if st.returncode == 0 else None}
    except Exception:                                   # noqa: BLE001
        return None                                     # git が無い・時間切れ


RUN_INFO = 'run_info.json'


def write_run_info(work, stage, settings=None):
    """作業ディレクトリの `run_info.json` に，その段の版と設定を書き足す

    段ごとに上書きする (段階 2 をやり直せば段階 2 の記録だけが新しくなる)．
    """
    import json
    from datetime import datetime
    p = Path(work) / RUN_INFO
    info = {}
    if p.is_file():
        try:
            info = json.loads(p.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            info = {}
    info[stage] = {'time': datetime.now().isoformat(timespec='seconds'),
                   'version': code_version(),
                   'settings': settings or {}}
    p.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    return info[stage]


def run_info_line(work):
    """`run_info.json` を 1〜2 行にまとめる (`checks.txt` の先頭に出す)．無ければ ''"""
    import json
    p = Path(work) / RUN_INFO
    if not p.is_file():
        return ''
    try:
        info = json.loads(p.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return ''
    parts, dirty = [], False
    for stage in ('grid', 'read', 'table'):
        v = (info.get(stage) or {}).get('version')
        if not v:
            continue
        mark = '*' if v.get('dirty') else ''
        dirty = dirty or bool(v.get('dirty'))
        parts.append(f'{stage} {v.get("commit")}{mark}')
    lines = []
    if parts:
        lines.append('版     : ' + ' / '.join(parts)
                     + ('  (* は未コミットの変更あり)' if dirty else ''))
    rs = (info.get('read') or {}).get('settings') or {}
    if rs:
        lines.append('設定   : ' + ' '.join(f'{k}={v}' for k, v in sorted(rs.items())))
    return '\n'.join(lines)
