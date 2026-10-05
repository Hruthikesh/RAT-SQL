"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (if installed).

    ``deterministic=True`` additionally requests deterministic cuDNN kernels.
    Full bitwise determinism on CUDA is not guaranteed (e.g. scatter_add /
    index_put backward use atomics); we document this in the report.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass
