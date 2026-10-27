"""Single-node BKCL collectives and a torch.distributed ProcessGroup.

The raw functions are the low-level path.  `ProcessGroupXCCL` is a thin
Python ProcessGroup built on the same BKCL context, intended for single-node
multi-card DataParallel/DDP.  It is deliberately **not** a multi-node backend:
there is no IB/RDMA bootstrap, and the unique id must be exchanged on the same
host (the default TCPStore does this).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from . import _C


def get_unique_id() -> bytes:
    return _C.xccl_get_unique_id()


def init_rank(rank: int, nranks: int, unique_id: bytes, device: Optional[int] = None) -> None:
    """Initialize the process-global BKCL context.

    Single-node only.  Prefer `dist.init_process_group("bkcl", ...)` for DDP;
    this function remains for manual collectives and probes.
    """
    if device is None:
        device = rank
    _C.xccl_init_rank(int(rank), int(nranks), unique_id, int(device))


def destroy() -> None:
    _C.xccl_destroy()


def rank() -> int:
    return int(_C.xccl_rank())


def nranks() -> int:
    return int(_C.xccl_nranks())


def _device_of(tensor: torch.Tensor) -> int:
    if tensor.device.type != "kunlun":
        raise ValueError("xccl requires a kunlun tensor")
    return int(tensor.device.index or 0)


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> None:
    _C.xccl_all_reduce(tensor, op, _device_of(tensor))


def broadcast(tensor: torch.Tensor, root: int) -> None:
    _C.xccl_broadcast(tensor, int(root), _device_of(tensor))


def all_gather(out: torch.Tensor, tensor: torch.Tensor) -> None:
    _C.xccl_all_gather(out, tensor, _device_of(tensor))


def reduce_scatter(out: torch.Tensor, tensor: torch.Tensor, op: str = "sum") -> None:
    _C.xccl_reduce_scatter(out, tensor, op, _device_of(tensor))


def barrier(device: int = 0) -> None:
    _C.xccl_barrier(int(device))


def _op_name(op) -> str:
    if op == dist.ReduceOp.SUM:
        return "sum"
    avg = getattr(dist.ReduceOp, "AVG", None)
    if avg is not None and op == avg:
        raise NotImplementedError("BKCL does not provide ReduceOp.AVG; use SUM and divide")
    if op == dist.ReduceOp.PRODUCT:
        return "prod"
    if op == dist.ReduceOp.MIN:
        return "min"
    if op == dist.ReduceOp.MAX:
        return "max"
    raise NotImplementedError(f"unsupported BKCL reduce op: {op}")


class _XCCLWork(dist.Work):
    """Completed work handle; XPURT has no non-blocking work query."""

    def wait(self, timeout=None):
        # BKCL is enqueued on the current XPU stream.  Waiting the device here
        # makes the Python Work contract conservative and DDP-safe.
        _C.synchronize()
        return True

    def is_completed(self) -> bool:
        return True

    def get_future(self):
        fut = torch.futures.Future()
        fut.set_result(None)
        return fut


class ProcessGroupXCCL(dist.ProcessGroup):
    """Single-node BKCL ProcessGroup.

    DDP-compatible subset: broadcast, allreduce, barrier, all_gather_base and
    reduce_scatter.  No send/recv, no multi-node, no process-group
    reconfiguration.
    """

    def __init__(self, store, rank: int, size: int, timeout=None):
        if store is None:
            raise ValueError("ProcessGroupXCCL requires a c10d Store")
        super().__init__(store, int(rank), int(size))
        self._rank = int(rank)
        self._size = int(size)
        if int(rank) == 0:
            uid = get_unique_id()
            store.set("kunlun_bkcl_uid", uid)
        else:
            uid = store.get("kunlun_bkcl_uid")
        # One process per rank, one PD per rank.  On a 2-card host this is
        # normally /dev/xpu0..3 (world_size 2 or 4).
        init_rank(int(rank), int(size), uid, device=int(rank))

    def getBackendName(self) -> str:
        return "bkcl"

    def getRank(self) -> int:
        return self._rank

    def getSize(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size

    def _device(self) -> int:
        return self._rank

    def broadcast(self, tensors, opts=None):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        root = int(getattr(opts, "rootRank", 0) if opts is not None else 0)
        for tensor in tensors:
            _C.xccl_broadcast(tensor, root, _device_of(tensor))
        return _XCCLWork()

    def allreduce(self, tensors, opts=None):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        op = _op_name(getattr(opts, "reduceOp", dist.ReduceOp.SUM))
        for tensor in tensors:
            _C.xccl_all_reduce(tensor, op, _device_of(tensor))
        return _XCCLWork()

    def allreduce_coalesced(self, tensors, opts=None):
        return self.allreduce(tensors, opts)

    def _allgather_base(self, output_buffer, input_buffer, opts=None):
        _C.xccl_all_gather(output_buffer, input_buffer, _device_of(input_buffer))
        return _XCCLWork()

    def all_gather(self, output_tensors, input_tensors, opts=None):
        # Non-coalesced fallback: one contiguous BKCL all-gather per input,
        # then scatter the flat result into the caller-provided output tensors.
        world = self._size
        for out_list, inp in zip(output_tensors, input_tensors):
            if len(out_list) != world:
                raise ValueError("all_gather output list must have world_size entries")
            flat = torch.empty(
                inp.numel() * world, dtype=inp.dtype, device=inp.device
            )
            _C.xccl_all_gather(flat, inp, _device_of(inp))
            for r, out in enumerate(out_list):
                out.copy_(flat.narrow(0, r * inp.numel(), inp.numel()).view_as(out))
        return _XCCLWork()

    def reduce_scatter(self, output, input, opts=None):
        op = _op_name(getattr(opts, "reduceOp", dist.ReduceOp.SUM))
        _C.xccl_reduce_scatter(output, input, op, _device_of(output))
        return _XCCLWork()

    def barrier(self, opts=None):
        _C.xccl_barrier(self._device())
        return _XCCLWork()

    def shutdown(self):
        destroy()


def _create_process_group(store, rank, size, timeout=None):
    return ProcessGroupXCCL(store, rank, size, timeout)


def register_backend() -> None:
    """Register `bkcl` with torch.distributed (single-node only)."""
    if "bkcl" in dist.Backend.backend_list:
        return
    dist.Backend.register_backend(
        "bkcl", _create_process_group, devices=["kunlun"]
    )


register_backend()
