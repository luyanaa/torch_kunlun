from __future__ import annotations

import torch
from torch.utils.backend_registration import (
    generate_methods_for_privateuse1_backend,
    rename_privateuse1_backend,
)

from . import _C  # noqa: F401  (registers PrivateUse1, allocator, hooks)
from . import kunlun
from . import xccl
from . import optim

__version__ = "0.1.0"

# The C++ extension registers the same name during static initialization;
# calling the Python API makes torch.device("kunlun") and torch.kunlun visible.
rename_privateuse1_backend("kunlun")
torch._register_device_module("kunlun", kunlun)
generate_methods_for_privateuse1_backend()

scaled_dot_product_attention = kunlun.scaled_dot_product_attention

__all__ = ["kunlun", "xccl", "optim", "scaled_dot_product_attention"]
