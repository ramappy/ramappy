"""Smoothing utilities."""

from functools import lru_cache

import numba as nb
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.interpolate import make_interp_spline
from scipy.sparse.linalg import splu

from ramappy import const

# Mapping from rubberband interpolation kind to B-spline order for make_interp_spline.
# The 'linear'/'slinear' fast path in rubberband() bypasses this table.
_RUBBERBAND_KIND_TO_K: dict[str, int] = {
    "nearest": 1,
    "zero": 1,
    "slinear": 1,
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
}


def diffmat(n_points: int, order: int, dtype: npt.DTypeLike = np.float32) -> sparse.csc_matrix:
    """Finite-difference matrix.

    Parameters
    ----------
    n_points : int
        Size of the matrix.
    order : int
        Order of differences.

    Returns
    -------
    sparse.csc_matrix
        Finite-difference matrix.
    """
    # Finite-difference coefficients as floating-point; dtype is propagated to the sparse matrix.
    diags = np.zeros(2 * order + 1, dtype=dtype)
    diags[order] = 1.0
    for _ in range(order):
        diags = diags[:-1] - diags[1:]
    # can't use 'dia' since D.T * D returns 'csc' anyway
    diff_mat = sparse.diags(diags, np.arange(order + 1), shape=(n_points - order, n_points), format="csc")
    return diff_mat


@lru_cache(maxsize=64)
def _cached_diffmat(n_points: int, order: int, dtype: npt.DTypeLike = np.float64) -> sparse.csc_matrix:
    # diffmat is pure (no mutation) and can be safely cached.
    return diffmat(n_points, order, dtype=dtype)


@lru_cache(maxsize=32)
def _cached_whittaker_lu(m: int, order: int, lambda_: float, dtype: npt.DTypeLike = np.float64):
    """Cached LU factorization for the uniform-x, unit-weight Whittaker system."""
    diff_mat = _cached_diffmat(m, order, dtype)
    penalty_mat = sparse.csc_matrix(np.dtype(dtype).type(lambda_) * diff_mat.T * diff_mat)
    penalty_mat.setdiag(penalty_mat.diagonal() + 1)
    return splu(penalty_mat, options={"SymmetricMode": True})


def diff(matrix: sparse.csc_matrix, order: int = 1) -> sparse.csc_matrix:
    """Explicit finite-difference of a given matrix (along axis 0).

    Parameters
    ----------
    matrix : sparse.csc_matrix
        Input matrix.
    order : int, optional
        Order of differences, by default 1.

    Returns
    -------
    sparse.csc_matrix
        Difference matrix.
    """
    for _ in range(order):
        matrix = matrix[1:, :] - matrix[:-1, :]
    return matrix


def divdiffmat(x: np.ndarray, order: int, dtype: np.dtype | None = None) -> sparse.csc_matrix:
    """Divided finite-difference matrix (for non-uniform x's).

    Parameters
    ----------
    x : np.ndarray
        Sampling positions.
    order : int
        Order of differences.

    Returns
    -------
    sparse.csc_matrix
        Divided finite-difference matrix.
    """
    n_points = len(x)
    _dtype = x.dtype if dtype is None else np.dtype(dtype)
    if order == 0:
        return sparse.identity(n_points, format="csc", dtype=_dtype)

    dx = (x[order:] - x[:-order]).astype(_dtype)
    inv_dx = sparse.diags(1.0 / dx, 0, shape=(n_points - order, n_points - order), format="csc")
    return inv_dx * diff(divdiffmat(x, order - 1, dtype=_dtype))


def whittaker_smooth(
    y: np.ndarray, lambda_: float, order: int = 2, w: np.ndarray | int = 1, x: np.ndarray | None = None
) -> np.ndarray:
    """Whittaker-Eilers smoother.

    Minimises the penalised least-squares criterion [Eilers2003]_

    .. math::

        \\sum_i w_i (y_i - \\hat{y}_i)^2 +
        \\lambda \\sum_i (\\Delta^d \\hat{y}_i)^2

    where :math:`\\Delta^d` is the *d*-th order finite-difference operator and
    :math:`\\lambda` controls the trade-off between fidelity and smoothness.

    Parameters
    ----------
    y : ndarray, shape (m,) or (n, m)
        Data series along the last axis. Unless *x* is given, the series is
        assumed to be uniformly sampled.
    lambda_ : float
        Smoothing parameter :math:`\\lambda`. Larger values give smoother results.
    order : int, default=2
        Order *d* of the difference penalty.
    w : ndarray, shape (m,) or int, default=1
        Observation weights. Use ``0`` to mark missing values, ``1`` for
        observed. The default (scalar ``1``) is treated as an all-ones vector.
    x : ndarray, shape (m,) or None, default=None
        Sampling positions, must be strictly increasing. If given, the
        divided-difference matrix is used instead of the uniform-grid variant,
        allowing non-uniform spacing.

    Returns
    -------
    ndarray
        Smoothed series, same shape and dtype as *y*.

    References
    ----------
    .. [Eilers2003] Eilers, Paul H. C. 2003. "A Perfect Smoother." Analytical Chemistry 75 (14): 3631-36.
           https://doi.org/10.1021/ac034173t
    """
    m = y.shape[const.Axis.SPECTRAL]
    if m <= order:
        raise ValueError(f"Data must have at least {order} points")

    use_cached_solver = x is None and isinstance(w, int) and w == 1

    diff_mat = _cached_diffmat(m, order, y.dtype) if x is None else divdiffmat(x, order, dtype=y.dtype)

    if not (isinstance(w, int) and w == 1):
        y = y * w  # do not mutate caller's array
    if use_cached_solver:
        lu = _cached_whittaker_lu(m, order, float(lambda_), y.dtype)
        return lu.solve(y.T).astype(y.dtype).T

    penalty_mat = sparse.csc_matrix(np.dtype(y.dtype).type(lambda_) * diff_mat.T * diff_mat)
    penalty_mat.setdiag(penalty_mat.diagonal() + w)

    return splu(penalty_mat, options={"SymmetricMode": True}).solve(y.T).astype(y.dtype).T


def _lower_hull_indices_sorted_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return indices of the lower convex hull for points (x, y) with strictly increasing x.

    Uses the monotone chain algorithm in O(n). The returned indices are increasing.

    Notes
    -----
    For the rubberband baseline we want the *lower* envelope (the band stretched under the spectrum).
    """
    n = x.size
    if n <= 2:
        return np.arange(n, dtype=np.int64)

    hull: list[int] = []

    def cross(o: int, a: int, b: int) -> float:
        return (x[a] - x[o]) * (y[b] - y[o]) - (y[a] - y[o]) * (x[b] - x[o])

    for i in range(n):
        while len(hull) >= 2 and cross(hull[-2], hull[-1], i) <= 0:
            hull.pop()
        hull.append(i)

    return np.asarray(hull, dtype=np.int64)


def rubberband(
    y: np.ndarray,
    x: np.ndarray,
    kind: str = "slinear",
    *,
    max_points: int | None = None,
) -> np.ndarray:
    """Rubberband method for baseline correction [RubberbandRef]_.

    This implementation avoids the expensive full convex hull (Qhull) and instead computes the
    lower hull in O(n) given sorted x.

    Parameters
    ----------
    y : np.ndarray
        Intensities.
    x : np.ndarray
        Wavenumbers (must be increasing).
    kind : str, optional
        Interpolation kind, by default "slinear".
    max_points : int | None, optional
        If provided and the spectrum has more than ``max_points`` points, compute the rubberband
        baseline on a uniformly downsampled grid and interpolate back to the full grid.
        This is an opt-in speed/accuracy tradeoff useful for very long spectra.

    Returns
    -------
    np.ndarray
        Baseline.

    References
    ----------
    .. [RubberbandRef] https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if max_points is not None and x.size > max_points >= 3:
        # Uniformly downsample indices (ensure endpoints are included).
        # Avoid np.unique (alloc + sort): the integer mapping below is monotone and unique
        # as long as x.size > max_points.
        idx = (np.arange(int(max_points), dtype=np.int64) * (x.size - 1)) // (int(max_points) - 1)
        idx[0] = 0
        idx[-1] = x.size - 1

        x_sub = x[idx]
        y_sub = y[idx]
        v = _lower_hull_indices_sorted_x(x_sub, y_sub)

        if kind in {"linear", "slinear"}:
            baseline = np.interp(x, x_sub[v], y_sub[v]).astype(y.dtype, copy=False)
            np.minimum(baseline, y, out=baseline)
            return baseline

        k = min(_RUBBERBAND_KIND_TO_K.get(kind, 3), len(v) - 1)
        baseline = make_interp_spline(x_sub[v], y_sub[v], k=k, check_finite=False)(x).astype(y.dtype, copy=False)
        np.minimum(baseline, y, out=baseline)
        return baseline

    v = _lower_hull_indices_sorted_x(x, y)

    # Fast path for linear interpolation.
    if kind in {"linear", "slinear"}:
        baseline = np.interp(x, x[v], y[v]).astype(y.dtype, copy=False)
        np.minimum(baseline, y, out=baseline)
        return baseline

    k = min(_RUBBERBAND_KIND_TO_K.get(kind, 3), len(v) - 1)
    baseline = make_interp_spline(x[v], y[v], k=k, check_finite=False)(x).astype(y.dtype, copy=False)
    np.minimum(baseline, y, out=baseline)
    return baseline


@nb.njit(cache=True)
def _lower_hull_indices_sorted_x_nb(x: np.ndarray, y: np.ndarray, hull_out: np.ndarray) -> int:
    """Lower hull indices for strictly increasing x (numba implementation).

    Parameters
    ----------
    x, y:
        1D arrays (same length), x must be increasing.
    hull_out:
        Preallocated int64 array with length >= x.size.

    Returns
    -------
    int
        Number of hull points written into hull_out.
    """
    n = x.size
    if n <= 2:
        for i in range(n):
            hull_out[i] = i
        return n

    m = 0
    for i in range(n):
        while m >= 2:
            o = hull_out[m - 2]
            a = hull_out[m - 1]
            # cross((o)->(a), (o)->(i))
            cross = (x[a] - x[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (x[i] - x[o])
            if cross <= 0.0:
                m -= 1
            else:
                break
        hull_out[m] = i
        m += 1
    return m


@nb.njit(cache=True)
def _rubberband_linear_1d_nb(x: np.ndarray, y: np.ndarray, baseline_out: np.ndarray) -> None:
    """Compute rubberband baseline with linear interpolation (numba).

    Produces baseline_out in-place and clamps baseline_out <= y.
    """
    n = x.size
    if n == 0:
        return

    hull = np.empty(n, dtype=np.int64)
    m = _lower_hull_indices_sorted_x_nb(x, y, hull)

    if m <= 1:
        v = y[0]
        for i in range(n):
            baseline_out[i] = v
        return

    # Piecewise linear interpolation between successive hull vertices.
    for j in range(m - 1):
        a = hull[j]
        b = hull[j + 1]
        xa = x[a]
        xb = x[b]
        ya = y[a]
        yb = y[b]
        inv_dx = 1.0 / (xb - xa)
        for i in range(a, b + 1):
            baseline_out[i] = ya + (yb - ya) * (x[i] - xa) * inv_dx

    # Clamp baseline to never exceed y.
    for i in range(n):
        baseline_out[i] = min(baseline_out[i], y[i])


@nb.njit(parallel=True, cache=True)
def rubberband_batch(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Batched rubberband baseline (linear only).

    Parameters
    ----------
    y:
        2D array (n_pixels, n_points).
    x:
        1D array (n_points,), increasing.

    Returns
    -------
    np.ndarray
        Baseline array with same shape as y.

    Notes
    -----
    This is intended for the default rubberband mode (linear/slinear). For non-linear
    interpolation kinds or the downsampled ``max_points`` mode, use :func:`rubberband`.
    """
    n_pix = y.shape[0]
    n_pts = y.shape[1]
    out = np.empty((n_pix, n_pts), dtype=y.dtype)
    for p in nb.prange(n_pix):  # type: ignore
        _rubberband_linear_1d_nb(x, y[p], out[p])
    return out
