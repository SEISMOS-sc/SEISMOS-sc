"""Global seeding. Seeds fix the experiment, not bit-identical floats across GPUs or batch sizes."""

from __future__ import annotations

import os
import random
from typing import Optional

from .logging import get_logger


def set_global_seed(
    seed: int,
    strict_determinism: bool = False,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> None:
    log = get_logger()

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    try:
        import torch
    except ImportError:  # pragma: no cover
        log.warning("torch not available; skipped torch seeding")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    except Exception:  # pragma: no cover - cpu-only builds
        pass

    if strict_determinism:
        # Required by cuBLAS for deterministic GEMMs; must be set before the
        # first CUDA context is created to take effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            log.info("strict determinism enabled (warn_only=True)")
        except Exception as exc:  # pragma: no cover
            log.warning("could not enable deterministic algorithms: %s", exc)

    log.info(
        "seeded: seed=%d strict=%s cudnn_deterministic=%s",
        seed,
        strict_determinism,
        cudnn_deterministic,
    )


def sample_indices(n_total: int, n_wanted: Optional[int], seed: int) -> list[int]:
    """Seeded subsample of ``range(n_total)``, sorted."""
    if n_wanted is None or n_wanted >= n_total:
        return list(range(n_total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), n_wanted))
