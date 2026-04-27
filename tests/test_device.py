from pathvlm_litebench.models import resolve_device


def test_resolve_device_auto():
    assert resolve_device("auto") is None


def test_resolve_device_none():
    assert resolve_device(None) is None


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_invalid():
    try:
        resolve_device("tpu")
    except ValueError as exc:
        assert "allowed values" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid device.")


def test_resolve_device_cuda_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    try:
        resolve_device("cuda")
    except ValueError as exc:
        assert "CUDA was requested but is not available" in str(exc)
    else:
        raise AssertionError("Expected ValueError when CUDA is unavailable.")
