# -*- coding: utf-8 -*-
"""Configuration.

This module contains flags to turn on and off optional modules.

"""
import warnings
from importlib import util
from typing import Literal

cupy_enabled = util.find_spec("cupy") is not None
if cupy_enabled:
    try:
        import cupy  # noqa
    except ImportError as e:
        warnings.warn(
            f"Importing cupy failed. "
            f"For more details, see the error stack below:\n{e}"
        )
        cupy_enabled = False

if cupy_enabled:  # pragma: no cover
    try:
        cudnn_enabled = util.find_spec("cupy.cuda.cudnn") is not None
        if cudnn_enabled:
            from cupy import cudnn  # noqa: F401
    except ImportError as e:
        warnings.warn(
            f"Importing cupy.cuda.cudnn failed. "
            f"For more details, see the error stack below:\n{e}"
        )
        cudnn_enabled = False
    try:
        nccl_enabled = util.find_spec("cupy.cuda.nccl") is not None
        if nccl_enabled:
            from cupy.cuda import nccl  # noqa: F401
    except ImportError as e:
        warnings.warn(
            f"Importing cupy.cuda.nccl failed. "
            f"For more details, see the error stack below:\n{e}"
        )
        nccl_enabled = False
else:
    cudnn_enabled = False
    nccl_enabled = False

mpi4py_enabled = util.find_spec("mpi4py") is not None

# This is to catch an import error when the cudnn in cupy (system) and pytorch
# (built in) are in conflict.

pytorch_enabled = util.find_spec("torch") is not None
if pytorch_enabled:
    try:
        import torch  # noqa
    except ImportError as e:
        warnings.warn(
            f"Importing torch failed. "
            f"For more details, see the error stack below:\n{e}"
        )
        pytorch_enabled = False
else:
    pytorch_enabled = False

preferred_backend = "numpy"
def set_backend_preference(backend: Literal["numpy", "cupy", "torch", "auto"]="auto") -> None:
    """Set which backend to prefer for CPU/GPU operations.

    Args:
        backend: 'numpy', 'cupy', 'torch', or 'auto'
    """
    global preferred_backend
    if backend in ["numpy", "cupy", "torch", "auto"]:
        if backend == "auto": 
            if pytorch_enabled:
                preferred_backend = "torch"
            elif cupy_enabled:
                preferred_backend = "cupy"
            else:
                preferred_backend = "numpy"
        else:
            preferred_backend = backend
    else:
        raise ValueError("Backend must be 'numpy', 'cupy', 'torch', or 'auto'")
