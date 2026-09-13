"""読み手を動かす装置 (CPU / GPU) の選び方 (device.py)

**GPU のある環境で作ったので，既定が `cuda` になっていた** (`yomi.py`)．
GPU の無い環境ではそのままでは落ちるので，選び方を 1 か所にまとめる．
判定の中身は差し替えて確かめる (torch も GPU も要らない)．
"""
import pytest

from comptea import device


@pytest.fixture(autouse=True)
def _fresh():
    """**自動の判定は一度しか調べない**ので，的ごとにやり直す"""
    device.forget()
    yield
    device.forget()


def test_環境変数で指せる(monkeypatch):
    monkeypatch.setenv(device.ENV, 'cpu')
    assert device.pick() == 'cpu'
    monkeypatch.setenv(device.ENV, 'CUDA')
    assert device.pick() == 'cuda'


def test_環境変数の前後の空白は落とす(monkeypatch):
    monkeypatch.setenv(device.ENV, '  cpu  ')
    assert device.pick() == 'cpu'


def test_知らない値は自動に落とす(monkeypatch):
    """打ち間違いで落とさない．自動の判定に回す"""
    monkeypatch.setenv(device.ENV, 'gpu')
    monkeypatch.setattr(device, '_torch_cuda', lambda: False)
    monkeypatch.setattr(device, '_has_nvidia_smi', lambda: False)
    assert device.pick() == 'cpu'


def test_torch_が_GPU_を見つければ_cuda(monkeypatch):
    monkeypatch.delenv(device.ENV, raising=False)
    monkeypatch.setattr(device, '_torch_cuda', lambda: True)
    assert device.pick() == 'cuda'


def test_torch_が_GPU_を見つけなければ_cpu(monkeypatch):
    monkeypatch.delenv(device.ENV, raising=False)
    monkeypatch.setattr(device, '_torch_cuda', lambda: False)
    monkeypatch.setattr(device, '_has_nvidia_smi', lambda: True)
    assert device.pick() == 'cpu'          # torch の答えを優先する


def test_torch_が無ければ_nvidia_smi_で見る(monkeypatch):
    """読み手は別の環境に入れてあるので，こちらに torch が無いことがある"""
    monkeypatch.delenv(device.ENV, raising=False)
    monkeypatch.setattr(device, '_torch_cuda', lambda: None)
    monkeypatch.setattr(device, '_has_nvidia_smi', lambda: True)
    assert device.pick() == 'cuda'


def test_どちらも無ければ_cpu(monkeypatch):
    monkeypatch.delenv(device.ENV, raising=False)
    monkeypatch.setattr(device, '_torch_cuda', lambda: None)
    monkeypatch.setattr(device, '_has_nvidia_smi', lambda: False)
    assert device.pick() == 'cpu'


def test_指定があれば判定しない(monkeypatch):
    called = []
    monkeypatch.delenv(device.ENV, raising=False)
    monkeypatch.setattr(device, '_torch_cuda', lambda: called.append(1) or True)
    assert device.pick('cpu') == 'cpu'
    assert not called


def test_GPU_が無いかを聞ける(monkeypatch):
    monkeypatch.setenv(device.ENV, 'cpu')
    assert device.is_cpu() is True
    monkeypatch.setenv(device.ENV, 'cuda')
    assert device.is_cpu() is False


def test_判定は一度だけ(monkeypatch):
    """auto の判定は毎回やらない (torch の読み込みは重い)"""
    n = []
    monkeypatch.delenv(device.ENV, raising=False)
    monkeypatch.setattr(device, '_torch_cuda', lambda: n.append(1) or False)
    monkeypatch.setattr(device, '_has_nvidia_smi', lambda: False)
    assert device.pick() == 'cpu'
    assert device.pick() == 'cpu'
    assert len(n) == 1
