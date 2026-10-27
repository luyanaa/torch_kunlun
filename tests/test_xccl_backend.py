import pytest
import torch

torch_kunlun = pytest.importorskip("torch_kunlun")
import torch.distributed as dist


def test_bkcl_backend_registered():
    assert "bkcl" in dist.Backend.backend_list
    assert dist.is_backend_available("bkcl")
