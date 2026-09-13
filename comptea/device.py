"""読み手を動かす装置 (CPU / GPU) を選ぶ (device.py)

**GPU のある環境で作ったので，既定が `cuda` になっていた** (`yomi.py`)．
GPU の無い環境ではそのまま落ちるので，選び方をここ 1 か所にまとめます．

    from comptea import device
    device.pick()            # 'cuda' か 'cpu'
    device.pick('cpu')       # 指したものをそのまま使う

決め方は上から順に:

1. 呼ぶ側が指したもの (`--device` など)
2. 環境変数 `COMPTEA_DEVICE` (`cpu` / `cuda`)
3. 自動 — `torch.cuda.is_available()`．**読み手は別の環境に入れてある**ので
   こちらに torch が無いことがあり，そのときは `nvidia-smi` があるかで見る

**GPU の無い環境でも読めます**．NDLOCR-Lite (`comptea.ndl`) は ONNX で
GPU が要らず，表頭の切り出しを CPU で 1.6 秒で読みました (2026-09-12 の実測)．
yomitoku も `cpu` で動きます (遅くはなる)．EasyOCR は GPU が無ければ
自分で CPU に落ちます．
"""
import os
import shutil

ENV = 'COMPTEA_DEVICE'
KNOWN = ('cpu', 'cuda')
_auto = None            # 自動の判定の答え (一度だけ調べる)


def _torch_cuda():
    """torch で GPU を見る．torch が無ければ None (分からない)"""
    try:
        import torch
    except Exception:
        return None
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return None


def _has_nvidia_smi():
    return shutil.which('nvidia-smi') is not None


def forget():
    """自動の判定をやり直す (試験用)"""
    global _auto
    _auto = None


def _detect():
    global _auto
    if _auto is None:
        got = _torch_cuda()
        if got is None:                       # torch が無い．道具の有無で見る
            got = _has_nvidia_smi()
        _auto = 'cuda' if got else 'cpu'
    return _auto


def pick(want=None):
    """使う装置の名前 (`'cpu'` か `'cuda'`)"""
    for v in (want, os.environ.get(ENV)):
        v = (v or '').strip().lower()
        if v in KNOWN:
            return v
    return _detect()


def is_cpu(want=None):
    """GPU を使わないか"""
    return pick(want) == 'cpu'
