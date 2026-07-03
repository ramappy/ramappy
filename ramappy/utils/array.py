"""Array manipulation utilities."""

from typing import Literal

import numpy as np
from numba import jit
from scipy.stats import trim_mean

from ramappy import const

_AGG_FUNCS: dict[str, object] = {
    "mean": np.mean,
    "median": np.median,
    "mean95": lambda x, axis: trim_mean(x, proportiontocut=0.025, axis=axis),
    "std": np.std,
    "max": np.max,
    "min": np.min,
    "p5": lambda x, axis: np.percentile(x, 5, axis=axis),
    "p95": lambda x, axis: np.percentile(x, 95, axis=axis),
    "p5p95": lambda x, axis: np.percentile(x, [5, 95], axis=axis),
    "p25": lambda x, axis: np.percentile(x, 25, axis=axis),
    "p75": lambda x, axis: np.percentile(x, 75, axis=axis),
    "p25p75": lambda x, axis: np.percentile(x, [25, 75], axis=axis),
}


@jit(cache=True)
def is_sorted(a: np.ndarray) -> bool:
    """Check if an array is sorted.

    Parameters
    ----------
    a : np.ndarray
        Input array.

    Returns
    -------
    bool
        True if sorted, False otherwise.

    Raises
    ------
    ValueError
        If the array has repeated values.
    """
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
        if a[i + 1] == a[i]:
            raise ValueError("X-axis has repeated values")
    return True


@jit(cache=True)
def _minmax(a: np.ndarray) -> tuple[float, float]:
    """Find min and max values of an array (numba optimized)."""
    max_val = a[0]
    min_val = a[0]

    for i in a[1:]:
        if i > max_val:
            max_val = i
        elif i < min_val:
            min_val = i
    return (min_val, max_val)


def minmax(a: np.ndarray | np.ma.MaskedArray) -> tuple[float, float]:
    """Find min and max values of an array.

    Parameters
    ----------
    a : np.ndarray or np.ma.MaskedArray
        Input array.

    Returns
    -------
    tuple[float, float]
        (min, max) values.
    """
    if isinstance(a, np.ma.MaskedArray):
        return _minmax(a.compressed().ravel())  # type: ignore
    if isinstance(a, np.ndarray):
        return _minmax(a.ravel())
    raise TypeError("Input must be a numpy array or masked array")


@jit(cache=True)
def merge_arrays(x: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Merge two sorted arrays.

    Parameters
    ----------
    x : np.ndarray
        First sorted array.
    x_grid : np.ndarray
        Second sorted array.

    Returns
    -------
    tuple
        Merged array and a tuple of indices (old_roi_idx, grid_idx).
    """
    # assume x and x_grid are sorted
    n_x, n_grid = len(x), len(x_grid)
    x_merge = np.empty(n_x + n_grid, dtype=x.dtype)
    old_roi_idx = np.empty(n_x, dtype=np.int64)
    grid_idx = np.empty(n_grid, dtype=np.int64)

    i, j, k = 0, 0, 0
    while i < n_x and j < n_grid:
        if x[i] < x_grid[j]:
            x_merge[k] = x[i]
            old_roi_idx[i] = k
            i += 1
        elif x[i] > x_grid[j]:
            x_merge[k] = x_grid[j]
            grid_idx[j] = k
            j += 1
        else:
            x_merge[k] = x[i]
            old_roi_idx[i] = k
            grid_idx[j] = k
            i += 1
            j += 1
        k += 1

    while i < n_x:
        x_merge[k] = x[i]
        old_roi_idx[i] = k
        i += 1
        k += 1

    while j < n_grid:
        x_merge[k] = x_grid[j]
        grid_idx[j] = k
        j += 1
        k += 1

    return x_merge[:k], (old_roi_idx, grid_idx)


def atleast_2d(ary: np.ndarray, axis: int = 1) -> np.ndarray:
    """View inputs as arrays with at least two dimensions.

    Expands on numpy own function by adding an axis parameter.

    Parameters
    ----------
    ary : np.ndarray
        Input array.
    axis : int, optional
        Axis to expand, by default 1.

    Returns
    -------
    np.ndarray
        Array with at least 2 dimensions.
    """
    if ary.ndim == 0:
        return ary.reshape(1, 1)
    if ary.ndim == 1:
        if axis == 0:
            return ary[np.newaxis, :]
        if axis == 1:
            return ary[:, np.newaxis]
        raise ValueError("Cannot atleast_2d with axis >= 2 / negative")
    return ary


def aggregate(
    intensities: np.ndarray,
    agg: Literal["mean", "median", "mean95", "std", "max", "min", "p5", "p95", "p5p95", "p25", "p75", "p25p75"]
    | None = "mean",
) -> np.ndarray:
    """Aggregate intensities along the pixel axis.

    Parameters
    ----------
    intensities : np.ndarray
        Input intensities.
    agg : str, optional
        Aggregation method, by default "mean".

    Returns
    -------
    np.ndarray
        Aggregated intensities.
    """
    if agg is None or intensities.shape[const.Axis.PIXEL] == 1:
        return intensities

    if agg not in _AGG_FUNCS:
        raise ValueError(f"Unknown aggregation method {agg}")

    return _AGG_FUNCS[agg](intensities, axis=const.Axis.PIXEL)  # type: ignore
