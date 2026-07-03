"""Optional acceleration hooks.

Nothing in this module should have import-time side effects.

Currently supported:
- Intel/oneAPI `sklearnex` patching for scikit-learn.
"""

from __future__ import annotations

from typing import Any


def enable_sklearnex(*, verbose: bool = False, **patch_kwargs: Any) -> bool:
    """Enable sklearnex patching if available.

    Returns
    -------
    bool
        True if sklearnex was successfully enabled, otherwise False.

    Notes
    -----
    This is intentionally opt-in (no import-time side effects).
    """
    try:
        from sklearnex import patch_sklearn  # type: ignore[import-not-found]

        patch_sklearn(**patch_kwargs)
        if verbose:
            print("Patched scikit-learn with sklearnex.")
        return True
    except Exception:
        return False
