"""ROI utilities."""

import numpy as np
import numpy.typing as npt
from numba import jit


def find_nearest_x_many(x: np.ndarray, roi_x: np.ndarray) -> np.ndarray:
    """Vectorized nearest-index search for sorted `x`.

    Parameters
    ----------
    x:
        Sorted 1D array (ascending).
    roi_x:
        Scalar or array of query values.

    Returns
    -------
    np.ndarray
        Indices into `x` (same shape as `roi_x`).

    Notes
    -----
    This is significantly faster than calling `find_nearest_x` repeatedly from Python.
    """
    x = np.asarray(x)
    values = np.asarray(roi_x)

    if x.size == 1:
        return np.zeros_like(values, dtype=np.int64)

    idx = np.searchsorted(x, values, side="left")

    # Clip to valid right-index candidates.
    right = np.clip(idx, 0, x.size - 1)
    left = np.clip(right - 1, 0, x.size - 1)

    # Pick the closest between left/right; on ties, match find_nearest_x (prefer right).
    choose_left = (idx > 0) & (idx < x.size) & (np.abs(x[right] - values) > np.abs(x[left] - values))
    return np.where(choose_left, left, right).astype(np.int64)


@jit(nopython=True, cache=True)
def _common_roi_overlap_core(roi_A: np.ndarray, roi_B: np.ndarray) -> np.ndarray:
    """Numba core for `common_roi_overlap`. Expects `roi_A` and `roi_B` to already share a dtype."""
    n_a = roi_A.shape[0]
    n_b = roi_B.shape[0]

    # Preallocate: there can never be more overlaps than min(n_a, n_b), but
    # sizing on n_a + n_b keeps the bookkeeping trivial and the cost negligible.
    out = np.empty((n_a + n_b, 2), dtype=roi_A.dtype)
    count = 0
    i = 0
    j = 0

    while i < n_a and j < n_b:
        l_end = max(roi_A[i, 0], roi_B[j, 0])
        r_end = min(roi_A[i, 1], roi_B[j, 1])
        if l_end < r_end:
            out[count, 0] = l_end
            out[count, 1] = r_end
            count += 1

        if roi_A[i, 1] < roi_B[j, 1]:
            i += 1
        else:
            j += 1

    return out[:count].copy()


def common_roi_overlap(roi_A: np.ndarray, roi_B: np.ndarray) -> np.ndarray:
    """Find the common overlap between two sets of ROIs.

    Both ROI arrays must be sorted by their start column and contain
    non-overlapping intervals within themselves (the usual output of
    `get_x_regions`/`clean_roi`).

    Parameters
    ----------
    roi_A : np.ndarray, shape (N, 2)
        First set of ROIs.
    roi_B : np.ndarray, shape (M, 2)
        Second set of ROIs.

    Returns
    -------
    np.ndarray, shape (K, 2)
        Common ROIs. Always 2D, even when K is 0 (no overlap) so callers
        can rely on `.shape[0]` / `.shape[1]` without special-casing.

    Notes
    -----
    Both inputs are run through `clean_roi` first, so they are reshaped to (N, 2) and (M, 2) if necessary.

    `roi_A` and `roi_B` are also cast to a shared dtype (via
    `np.result_type`) before entering the jitted core.
    """
    roi_A = clean_roi(roi_A)
    roi_B = clean_roi(roi_B)
    dtype = np.result_type(roi_A.dtype, roi_B.dtype)
    return _common_roi_overlap_core(roi_A.astype(dtype), roi_B.astype(dtype))


def clean_roi(region_of_interest: list | np.ndarray) -> np.ndarray:
    """Reshape ROI to (N, 2).

    Parameters
    ----------
    region_of_interest : list or np.ndarray
        Input ROI.

    Returns
    -------
    np.ndarray
        Reshaped ROI.
    """
    return np.array(region_of_interest).reshape(-1, 2)


@jit(nopython=True, cache=True)
def find_nearest_x(x: np.ndarray, roi_x: float) -> int:
    """Find nearest index in x for a given value roi_x.

    Parameters
    ----------
    x : np.ndarray
        Sorted array.
    roi_x : float
        Value to find.

    Returns
    -------
    int
        Index of nearest value.
    """
    # assumes x is sorted
    # if outside range, returns first (`roi_x` below minimum) or last (`roi_x` above maximum) index
    if x.size == 1:
        return 0
    i = 0
    for i in range(x.size):
        if x[i] > roi_x:
            break
    if np.abs(x[i] - roi_x) > np.abs(x[i - 1] - roi_x):
        return i - 1
    return i


def get_x_regions(wn: np.ndarray, threshold: float = 10) -> np.ndarray:
    """Return separate connected intervals in the wn axis, as an array of array of endpoints.

    Use a heuristic method to determine if there is more than one region.

    Parameters
    ----------
    wn : array_like, shape (M,)
        Raman shift
    threshold : int or float, default=2
        The threshold used to find connected regions

    Returns
    -------
    ndarray, shape (N,2)
        Array containing arrays of endpoints

    """
    wn = np.asarray(wn)
    endpoints = np.array([wn[0], wn[-1]])
    wn_d = np.diff(wn)
    sep = np.argwhere(np.abs(wn_d - wn_d.mean()) > threshold * wn_d.std()).ravel()
    if len(sep):
        endpoints = np.sort(np.unique(np.concatenate((endpoints, wn[sep], wn[sep + 1]), 0)))
    return endpoints.reshape(-1, 2)


def select_x_indices(x: np.ndarray, roi_x: npt.ArrayLike | int | float | None, keep_regions: bool = False):
    """Select the logic indices in the x array corresponding to the provided roi_x.

    Parameters
    ----------
    x : array_like, shape (N,)
        Raman shift
    roi_x : array_like, shape (M,2) or int or float
        Wavenumber region to extract from wn.
        Example: ``region_of_interest = [[600, 1800], [2800, 3030]]`` or ``region_of_interest = [600, 3030]``
    keep_regions : bool, default=False
        If True, returns a list of indices for each region.

    Returns
    -------
    indices : array_like, shape (K,)
        The logic indices in the wn array corresponding to the provided region_of_interest.

    """
    if isinstance(roi_x, (float, int)):
        # just a single wavenumber, find nearest available wn
        return int(find_nearest_x_many(x, np.asarray(roi_x)))
    roi_x_arr: np.ndarray | None = np.asarray(roi_x) if roi_x is not None else None
    if roi_x_arr is None or len(roi_x_arr) == 0:
        return slice(None)
    roi_x_arr = clean_roi(roi_x_arr)

    # One combined nearest-index search for both the start and end columns
    n_regions = roi_x_arr.shape[0]
    combined_idx = find_nearest_x_many(x, np.concatenate((roi_x_arr[:, 0], roi_x_arr[:, 1])))
    starts, ends = combined_idx[:n_regions], combined_idx[n_regions:]

    # Ensure each interval is increasing in index space.
    lo = np.minimum(starts, ends)
    hi = np.maximum(starts, ends)

    # Common fast path: a single contiguous region can be represented as a slice,
    # avoiding allocating an explicit index array.
    if roi_x_arr.shape[0] == 1 and not keep_regions:
        return slice(int(lo[0]), int(hi[0]) + 1)

    if keep_regions:
        return [slice(int(s), int(e) + 1) for s, e in zip(lo, hi, strict=False)]

    indices = [np.arange(s, e + 1) for s, e in zip(lo, hi, strict=False)]
    return np.hstack(indices)
