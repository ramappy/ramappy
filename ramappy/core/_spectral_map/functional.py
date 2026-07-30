from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ramappy.core.pipeline import SpectralData

_CPU_COUNT: int = os.cpu_count() or 1
_IDENTITY: Callable = lambda x: x  # noqa: E731  # module-level identity avoids per-call closure
_PARALLEL_MIN_ITEMS: int = max(256, 16 * _CPU_COUNT)  # joblib threshold: below this, assume serial is faster


def _resolve_parallel_config(
    parallel: bool | Literal["threads", "processes"] | None, instance_parallel: bool | None = None
) -> tuple[bool, str | None]:
    """Resolve the *parallel* argument into ``(use_parallel, backend)``."""
    if parallel in {"threads", "processes"}:
        return True, parallel
    if parallel is None:
        enabled = instance_parallel is True
        return enabled, "threads" if enabled else None
    enabled = bool(parallel)
    return enabled, "threads" if enabled else None


def _apply_rows(
    func: Callable,
    data: np.ndarray,
    use_parallel: bool,
    parallel_backend: str | None,
    progress_wrapper: Callable,
    output_dtype: np.dtype | None,
    kwargs: dict,
) -> np.ndarray:
    """Apply *func* to each row of *data*, with optional joblib parallelism."""
    n_items = len(data)
    if n_items == 0:
        return np.array([], dtype=output_dtype)

    # Heuristic: joblib dispatch overhead dominates for small workloads.
    # Keep the threshold conservative so expensive per-item functions still benefit.
    do_parallel = use_parallel and n_items >= _PARALLEL_MIN_ITEMS

    if do_parallel:
        parallel_exec = Parallel(n_jobs=-1, prefer=parallel_backend)
        results = parallel_exec(delayed(func)(d, **kwargs) for d in progress_wrapper(data))
        return np.array(results, dtype=output_dtype)

    # Serial path: probe the first item to learn output shape, then pre-allocate.
    it = iter(progress_wrapper(data))
    first_res = np.asarray(func(next(it), **kwargs))
    if output_dtype is not None:
        first_res = first_res.astype(output_dtype, copy=False)

    out = np.empty((n_items, *first_res.shape), dtype=first_res.dtype)
    out[0] = first_res
    for i, d in enumerate(it, start=1):
        res = np.asarray(func(d, **kwargs))
        if output_dtype is not None:
            res = res.astype(first_res.dtype, copy=False)
        out[i] = res

    return out


def _apply_rows_standalone(
    func: Callable,
    data: np.ndarray,
    *,
    parallel: bool | str | None = None,
    output_dtype: np.dtype | None = None,
    kwargs: dict | None = None,
) -> np.ndarray:
    """Apply *func* to each row of *data* without requiring a SpectralMap instance."""
    use_parallel, parallel_backend = _resolve_parallel_config(parallel)
    return _apply_rows(func, data, use_parallel, parallel_backend, _IDENTITY, output_dtype, kwargs or {})


class _SpectralMapFunctionalMixin:
    """Functional helpers (apply a function across pixels/wavenumbers/maps)."""

    parallel: bool | None
    track_progress: Callable | str | None
    data: np.ndarray
    # SpectralMap provides these; declaring here keeps type-checkers happy.
    img_height: int
    img_width: int

    @property
    def x_size(self) -> int:
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_func(
        self,
        f: Callable,
        *,
        data: SpectralData | None = None,
        preserve_input_dtype: bool = True,
        by: Literal["pixel", "wavenumber", "map"] = "pixel",
        flatten: bool = True,  # only for `by='map'`
        parallel: bool | Literal["threads", "processes"] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply a function to SpectralMap data with optional parallelization."""

        spectral_data = self.data if data is None else data

        # Normalize to ndarray: downstream logic relies on ndarray attributes like dtype/T.
        spectral_data = np.asarray(spectral_data)

        use_parallel, parallel_backend = self._get_parallel_config(parallel)
        progress_wrapper = self._get_progress_wrapper()
        output_dtype = spectral_data.dtype if preserve_input_dtype else None

        if by == "wavenumber":
            result = _apply_rows(
                f, spectral_data.T, use_parallel, parallel_backend, progress_wrapper, output_dtype, kwargs
            )
            return result.T
        if by == "pixel":
            return _apply_rows(f, spectral_data, use_parallel, parallel_backend, progress_wrapper, output_dtype, kwargs)
        if by == "map":
            cube = spectral_data.cube
            # Iterate over wavenumber slices: (W, H, N) -> iterate N slices of shape (W, H)
            transposed_cube = np.transpose(cube, axes=[2, 0, 1])
            result = _apply_rows(
                f, transposed_cube, use_parallel, parallel_backend, progress_wrapper, output_dtype, kwargs
            )
            result = np.transpose(result, axes=(1, 2, 0))
            if flatten:
                return result.reshape(self.img_height * self.img_width, self.x_size)
            return result

        raise ValueError(f"Invalid value for 'by': {by}. Must be one of 'pixel', 'wavenumber', 'map'")

    def _get_parallel_config(self, parallel: bool | str | None) -> tuple[bool, str | None]:
        return _resolve_parallel_config(parallel, getattr(self, "parallel", None))

    def _get_progress_wrapper(self) -> Callable:
        track_progress = getattr(self, "track_progress", None)
        if track_progress == "tqdm":
            return tqdm
        return _IDENTITY
