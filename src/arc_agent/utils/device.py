"""Device and CUDA utility helpers."""

from __future__ import annotations

from typing import Any


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def get_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as error:
        raise RuntimeError(
            "PyTorch is required for neural training. Install with `pip install -e .[train]` "
            "or use the non-neural mock evaluation path."
        ) from error


def select_device(requested: str = "auto") -> str:
    if requested == "cpu":
        return "cpu"
    if not torch_available():
        return "cpu"
    import torch

    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def cuda_memory_gb() -> float:
    if not torch_available():
        return 0.0
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated() / (1024**3))
