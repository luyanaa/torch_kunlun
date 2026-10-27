import pytest
import torch

torch_kunlun = pytest.importorskip("torch_kunlun")
kunlun = torch_kunlun.kunlun

pytestmark = pytest.mark.skipif(
    not kunlun.is_available(),
    reason="Kunlun K200 device is not available",
)


def test_stream_and_event():
    stream = kunlun.Stream()
    event = kunlun.Event()
    with kunlun.stream(stream):
        x = torch.empty(256, dtype=torch.float16, device="kunlun")
        x.fill_(2.0)
        event.record()
    event.synchronize()
    stream.synchronize()
    assert torch.all(x.cpu() == 2.0)


def test_cross_device_copy():
    if kunlun.device_count() < 2:
        pytest.skip("need at least two XPU devices for the cross-device copy test")
    src = torch.ones(1024, dtype=torch.float16, device="kunlun:0")
    dst = torch.empty_like(src, device="kunlun:1")
    dst.copy_(src)
    kunlun.synchronize()
    assert torch.all(dst.cpu() == 1.0)


def test_stream_is_torch_stream_and_record_stream():
    stream = kunlun.Stream()
    assert isinstance(stream, torch.Stream)
    x = torch.empty(64, dtype=torch.float16, device="kunlun")
    x.record_stream(stream)
    stream.synchronize()
