"""CPU / CUDA device resolution."""

from __future__ import annotations


def resolve_device(requested: str = "auto"):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] CUDA requested ({requested}) but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def device_info() -> dict:
    import platform

    import torch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        info["cuda_version"] = torch.version.cuda
    return info
