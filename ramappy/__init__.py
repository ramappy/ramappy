"""ramappy package.

Importing `ramappy` enables optional ``sklearnex`` acceleration by
default when available.

Activation is done silently and will never raise if the optional dependency is
not installed.

Set ``RAMAPPY_AUTO_SKLEARNEX=0`` to disable automatic ``sklearnex`` patching.
"""

from __future__ import annotations

import contextlib
import os

from ramappy._version import __version__

_accel = None

# Importing this module is lightweight; sklearnex is still imported lazily
# inside `enable_sklearnex()`.
with contextlib.suppress(Exception):  # pragma: no cover
    import ramappy.accel as _accel  # type: ignore[assignment]


def _auto_enable_accelerations() -> bool:
    """Attempt to enable optional runtime accelerations.

    Returns
    -------
    bool
        True if any acceleration was enabled.
    """

    if os.getenv("RAMAPPY_AUTO_SKLEARNEX", "1") in {"0", "False", "NO", "false", "no"}:
        return False

    if _accel is None:
        return False

    try:
        enable = getattr(_accel, "enable_sklearnex", None)
        if not callable(enable):
            return False
        return bool(enable(verbose=False))
    except Exception:
        return False


SKLEARNEX_ENABLED = _auto_enable_accelerations()

# High-level imports
from ramappy.core import SpectralMap, Spectrum  # noqa: E402
from ramappy.pipeline import Pipeline  # noqa: E402
from ramappy.units import Quantity, QuantityUnit, Unit  # noqa: E402

__all__ = [
    "SKLEARNEX_ENABLED",
    "Pipeline",
    "Quantity",
    "QuantityUnit",
    "SpectralMap",
    "Spectrum",
    "Unit",
    "__version__",
]
