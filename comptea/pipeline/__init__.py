"""工程の段(格子・読み取り・組み上げ)

CLI からもアプリからも**同じものを呼ぶ**(2026-09-07)．前はアプリが
`cli/*.py` を subprocess で起こしており，そのたびに torch を読み込んでいた．

    from comptea.pipeline import grid, read, table
    grid.run(['画像.png', '--workdir', 'work/x'])

`run()` は CLI と同じ引数の並びを受け取り，何も渡さなければ `sys.argv` を使う．
"""
from . import common                                # noqa: F401


def run(stage, argv, capture=True):
    """段を**同じプロセスで**動かし，画面に出た文字を返す

    アプリはこれを呼ぶ．前は `cli/*.py` を subprocess で起こしており，
    そのたびに torch や easyocr を読み込んでいた(検出で 5 秒・読み取りで 7 秒)．
    測ると，CPU 版の torch なら検出 432 MB・読み取り 507 MB で，
    Streamlit の無料枠(1 GB ほど)に収まる(2026-09-07)．

    Args:
        stage: 'grid' / 'read' / 'table'
        argv: CLI と同じ引数の並び
        capture: 標準出力を集めて返す(False なら素通し)
    Returns:
        (終了コード, 画面に出た文字)．例外は SystemExit も含めて捕まえる
    """
    import contextlib
    import importlib
    import io

    mod = importlib.import_module(f'{__name__}.{stage}')
    buf = io.StringIO()
    try:
        if capture:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                mod.main(list(argv))
        else:
            mod.main(list(argv))
    except SystemExit as e:                 # 段は「できない理由」を SystemExit で返す
        code = e.code if isinstance(e.code, int) else 1
        if not isinstance(e.code, int) and e.code:
            buf.write(str(e.code) + '\n')
        return code, buf.getvalue()
    except Exception:                       # noqa: BLE001  画面に出して知らせる
        import traceback

        buf.write(traceback.format_exc())
        return 1, buf.getvalue()
    return 0, buf.getvalue()
