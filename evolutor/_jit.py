"""
JIT
---

Provides a simple decorator to JIT-compile the function using
numba if the library is installed, and do nothing otherwise.
"""
from __future__ import annotations

from typing import Callable


def maybe_jit(func: Callable, **kwargs) -> Callable:
    """
    A numba.jit decorator that does nothing if numba is not installed.
    """
    try:
        from numba import jit

        return jit(func, **kwargs)
    except ImportError:
        return func
