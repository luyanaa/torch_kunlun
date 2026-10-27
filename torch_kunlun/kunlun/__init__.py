from __future__ import annotations

from collections import namedtuple
from typing import List, Optional, Tuple

import torch

from .. import _C

DeviceProperties = namedtuple(
    "DeviceProperties",
    ["name", "total_memory", "multi_processor_count", "warp_size", "major", "minor", "model"],
)


def is_available() -> bool:
    return device_count() > 0


def device_count() -> int:
    return int(_C.device_count())


def current_device() -> int:
    return int(_C.current_device())


def set_device(device: int) -> None:
    _C.set_device(int(device))


def synchronize(device: Optional[int] = None) -> None:
    if device is not None:
        set_device(device)
    _C.synchronize()


def get_device_properties(device=None) -> DeviceProperties:
    if device is None:
        index = current_device()
    elif isinstance(device, torch.device):
        if device.type != "kunlun":
            raise ValueError(f"expected a kunlun device, got {device}")
        index = current_device() if device.index is None else device.index
    else:
        index = int(device)
    return DeviceProperties(**_C.get_device_properties(index))


def get_device_capability(device=None) -> Tuple[int, int]:
    props = get_device_properties(device)
    return props.major, props.minor


def memory_allocated(device: Optional[int] = None) -> int:
    if device is not None:
        set_device(device)
    return int(_C.memory_allocated())


def memory_reserved(device: Optional[int] = None) -> int:
    if device is not None:
        set_device(device)
    return int(_C.memory_reserved())


def empty_cache() -> None:
    _C.empty_cache()


def _is_in_bad_fork() -> bool:
    return False


def manual_seed_all(seed: int) -> None:
    # The basic runtime has no XPU RNG yet.
    return None


def get_amp_supported_dtype() -> List[torch.dtype]:
    return [torch.float16]


class _DeviceGuard:
    def __init__(self, device: int):
        self.device = int(device)
        self.prev = None

    def __enter__(self):
        self.prev = current_device()
        set_device(self.device)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.prev is not None:
            set_device(self.prev)
        return False


# `torch.Stream` is the object accepted by `Tensor.record_stream()` and by
# PyTorch's PrivateUse1 guard.  Keep our handle as a real `torch.Stream`
# subclass rather than a lookalike wrapper.
_PRIVATEUSE1_DEVICE_TYPE = int(torch._C._autograd.DeviceType.PrivateUse1.value)
_OWNED_STREAMS = set()


class Stream(torch.Stream):
    """An XPU stream that is also a real ``torch.Stream``.

    Construction with ``stream_id`` wraps an existing handle (current/default
    stream); construction without it creates and owns a new XPU stream.
    """

    def __new__(cls, device=None, stream_id=None, priority: int = 0):
        index = current_device() if device is None else _device_index(device)
        owned = stream_id is None
        raw_id = int(_C.create_stream(index)) if owned else int(stream_id)
        obj = super().__new__(cls, raw_id, index, _PRIVATEUSE1_DEVICE_TYPE)
        if owned:
            _OWNED_STREAMS.add(raw_id)
        return obj

    def __init__(self, device=None, stream_id=None, priority: int = 0):
        # All state lives in the C++ torch.Stream base object; pybind types do
        # not necessarily allow arbitrary Python attributes.
        pass

    def wait_event(self, event: "Event") -> None:
        _C.stream_wait_event(self.stream_id, event.event_id)

    def wait_stream(self, other: "Stream") -> None:
        event = Event()
        event.record(other)
        event.wait(self)

    def synchronize(self) -> None:
        _C.synchronize_stream(self.stream_id)

    def __del__(self):
        try:
            raw = int(self.stream_id)
        except Exception:
            return
        if raw in _OWNED_STREAMS:
            _OWNED_STREAMS.discard(raw)
            try:
                _C.destroy_stream(raw)
            except Exception:
                pass


class Event:
    """An XPU event. elapsed_time is unavailable in XPURT."""

    def __init__(self, enable_timing: bool = False):
        self.enable_timing = enable_timing
        self.event_id = int(_C.event_create())

    def record(self, stream: Optional[Stream] = None) -> None:
        target = stream if stream is not None else current_stream()
        _C.event_record(self.event_id, target.stream_id)

    def wait(self, stream: Optional[Stream] = None) -> None:
        target = stream if stream is not None else current_stream()
        _C.stream_wait_event(target.stream_id, self.event_id)

    def query(self) -> bool:
        # XPURT has no non-blocking query; this waits.
        return bool(_C.event_query(self.event_id))

    def synchronize(self) -> None:
        _C.event_synchronize(self.event_id)

    def elapsed_time(self, other: "Event") -> float:
        raise NotImplementedError("XPURT has no event elapsed-time API")

    def __del__(self):
        if getattr(self, "event_id", None):
            try:
                _C.event_destroy(self.event_id)
            except Exception:
                pass


def _device_index(device) -> int:
    if isinstance(device, torch.device):
        if device.type not in ("kunlun", "privateuseone"):
            raise ValueError(f"expected a kunlun device, got {device}")
        return current_device() if device.index is None else int(device.index)
    return int(device)


def current_stream(device=None) -> Stream:
    index = current_device() if device is None else _device_index(device)
    return Stream(index, stream_id=int(_C.current_stream(index)))


def default_stream(device=None) -> Stream:
    index = current_device() if device is None else _device_index(device)
    return Stream(index, stream_id=int(_C.default_stream(index)))


def set_stream(stream: Stream) -> None:
    _C.set_stream(stream.stream_id)


class _StreamContext:
    def __init__(self, stream: Stream):
        self.stream = stream
        self.prev = None

    def __enter__(self):
        index = self.stream.device.index
        self.prev = current_stream(current_device() if index is None else index)
        set_stream(self.stream)
        return self.stream

    def __exit__(self, exc_type, exc_value, traceback):
        if self.prev is not None:
            set_stream(self.prev)
        return False


def stream(stream_obj: Stream) -> _StreamContext:
    return _StreamContext(stream_obj)


from .attention import scaled_dot_product_attention

__all__ = [
    "Stream",
    "Event",
    "scaled_dot_product_attention",
]
