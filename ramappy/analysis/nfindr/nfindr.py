"""N-FINDR core routines adapted from `pyspc-unmix`.

Source project: https://github.com/r-hyperspec/pyspc-unmix (snapshot at tag `0.2.0`)
Original author: Rustam Guliev
License declaration at source: `license = "MIT"` in
https://github.com/r-hyperspec/pyspc-unmix/blob/0.2.0/pyproject.toml. Note: the `0.2.0` tag does not include a standalone LICENSE file text.

This module contains local adaptations and numba-based optimizations.
"""

import warnings

import numba as nb
import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted


@nb.jit("int32(int32)", cache=True)
def _factorial(n: int) -> int:
    f: int = 1
    for i in range(1, n + 1):
        f *= i
    return f


@nb.jit(cache=True)
def _inner_simplex_points(vertices: np.ndarray, high: float | list[float] = 1.0, n: int = 100) -> np.ndarray:
    """Generate inner simplex points.

    Parameters
    ----------
    vertices : np.ndarray
        List of vertex points. If a matrix, then vertex points are in rows,
        i.e., the size must be (N+1)xN
    high : float, optional
        Maximum possible coefficient for a vertex point, by default 1.0
    n : int, optional
        Number of points to generate, by default 100

    Returns
    -------
    np.ndarray
        nxN matrix of `n` points inside of the N-dimensional simplex defined
        by the vertices

    """
    p = len(vertices)
    coefficients = np.empty((n, p), dtype=np.float32)

    filled = 0
    while filled < n:
        need = n - filled
        new_coefficients = np.random.uniform(high=high, size=(3 * need, p)).astype(np.float32)  # noqa: NPY002
        new_coefficients = new_coefficients[np.sum(new_coefficients, axis=1) <= 1, :]

        take = min(need, new_coefficients.shape[0])
        if take:
            coefficients[filled : filled + take] = new_coefficients[:take, :]
            filled += take

    return coefficients @ vertices


@nb.jit(cache=True)
def _pad_ones(x: np.ndarray) -> np.ndarray:
    """Add column of ones to a 2D matrix.

    Parameters
    ----------
    x : np.ndarray
        2D matrix of size (N, M)

    Returns
    -------
    np.ndarray
        2D matrix of size (N, M+1) where the first column is a vector of 1s
        and the rest of the matrix is `x`
    """
    ones_column = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.hstack((ones_column, x))


@nb.jit(cache=True)
def _simplex_E(x: np.ndarray, indices: list[int] | np.ndarray | slice | None = None) -> np.ndarray:
    """Generate a simplex volume matrix.

    Simple helper function for generating a simplex volume matrix E
    (i.e. volume of simplex = det(E)/p-1!) of the following structure:
    |   1   1   ... 1 |
    | e_1 e_2 ... e_p |
    Where e_i is an i-th vertex point of the simplex.

    Parameters
    ----------
    x : np.ndarray
        Matrix whose rows will be included in the simplex. This
        matrix should be reduced using PCA or some other process
        so that it has p-1 columns before calling this function.
    indices : Optional[List[int]], optional
        Locations of the rows in the dataset to use as simplex vertices, by default None

    Returns
    -------
    np.ndarray
        A simplex volume matrix E, a p x p matrix whose first row contains only 1s

    """
    if indices is None:
        # indices = range(x.shape[0])
        indices = np.arange(x.shape[0])
    elif isinstance(indices, list):
        indices = np.asarray(indices)
    return np.transpose(_pad_ones(x[indices, :]))


@nb.jit(cache=True)
def _simplex_E_padded(x_padded: np.ndarray, indices: list[int] | np.ndarray | slice | None = None) -> np.ndarray:
    """Generate simplex volume matrix from a pre-padded matrix.

    This avoids rebuilding ``_pad_ones(x[...])`` repeatedly in tight loops.
    """
    if indices is None:
        indices = np.arange(x_padded.shape[0])
    elif isinstance(indices, list):
        indices = np.asarray(indices)
    return np.transpose(x_padded[indices, :])


# ["float32(float32[:,::1], bool)", "float32(float32[:,::1], Omitted(True))"],
@nb.jit(cache=True)
def simplex_volume(x: np.ndarray, factorial: bool = True) -> float:
    """Simplex volume.

    Calculates a simplex volume based on determinant formula

    Parameters
    ----------
    x : np.ndarray
        A 2D matrix containing simplex vertices in rows. The number of rows expected
        to be 1 more than number of columns. As, for example, 2D simplex would be a
        trianle with three points
    factorial : bool, optional
        Whether to divide the matrix determinant by factorial, by default True

    Returns
    -------
    float
        The N-D volume of the provided simplex

    Raises
    ------
    ValueError
        The simplex matrix does not have Nx(N-1) shape
    """
    if x.shape[0] != (x.shape[1] + 1):
        raise ValueError("Unexpected array size. The simplex matrix must be of size Nx(N-1)")
    volume = np.abs(np.linalg.det(_pad_ones(x)))
    if factorial:
        volume /= _factorial(x.shape[1])
    return volume


@nb.jit(cache=True)
def cart2bary(x: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Conversion of Cartesian to Barycentric coordinates.

    Parameters
    ----------
    x : np.ndarray of shape (M, N)
        2d matrix of Cartesian coordinates to be converted to barycentric. One row corresponds to one line.
    vertices : np.ndarray of shape (N+1, N)
        Vertex points of the simplex with respect to which barycentric coordinates should be computed

    Returns
    -------
    np.ndarray of shape (M,N+1)
        Barycentric coordinates of the points with respect to provided vertex points.

    Raises
    ------
    ValueError
        Dimensions of given points and vertices mismatch
    """
    # Ensure contiguity for faster matmul inside numba (and to avoid NumbaPerformanceWarning).
    x = np.ascontiguousarray(x)
    vertices = np.ascontiguousarray(vertices)

    _m, n = x.shape
    if vertices.shape[1] != n:
        raise ValueError("Vertices must have same number of columns as the point matrix")
    A = vertices[:-1, :] - vertices[-1, :]
    Ainv = np.linalg.inv(A.T)
    bary_coefs = (x @ Ainv.T) - (vertices[-1, :] @ Ainv.T)
    bary_coefs = np.hstack((bary_coefs, 1 - bary_coefs.sum(axis=1).reshape((-1, 1))))
    return bary_coefs


@nb.jit(cache=True)
def _estimate_volume_change(
    x: np.ndarray,
    indices: list[int],
    endmembers: int | list[int] | np.ndarray | None = None,
    new_indices: int | list[int] | np.ndarray | None = None,
    Einv: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate volume change using Cramer's rule.

    Parameters
    ----------
    x : np.ndarray
        Matrix of M points in N-dimensional space
    indices : List[int]
        N+1 indices of the initial endmembers
    endmembers : Optional[Union[int, List[int]]], optional
        One or list of endmember indices for replacement, by default all endmembers,
        i.e. `range(N+1)`
    new_indices : Optional[Union[int, List[int]]], optional
        One or list of point indices for replacement, by default all points,
        i.e. `range(M)`
    Einv : Optional[np.ndarray], optional
        Pre-calculated inversed E matrix for faster calculation, by default None

    Returns
    -------
    np.ndarray
        LxK matrix (V), where L is the length of `new_indices` and K is the length of
        `endmembers`, where `Vij` estimates how would the simplex volume chage if the
        j-th endmember would be replaced by i-th point. The calculated value is the
        new volume divided by old (initial) volume.
    """
    if Einv is None:
        E = _simplex_E(x, indices)
        Einv = np.linalg.inv(E)

    endmembers_: np.ndarray
    if endmembers is None:
        endmembers_ = np.arange(len(indices))
    elif isinstance(endmembers, int):
        endmembers_ = np.asarray([endmembers])
    else:
        endmembers_ = np.asarray(endmembers)

    new_indices_: np.ndarray
    if new_indices is None:
        new_indices_ = np.arange(x.shape[0])
    elif isinstance(new_indices, int):
        new_indices_ = np.asarray([new_indices])
    else:
        new_indices_ = np.asarray(new_indices)

    ratios = _pad_ones(x[new_indices_, :]) @ Einv.T[:, endmembers_]

    return np.abs(ratios)


@nb.jit(cache=True)
def nfindr(
    x: np.ndarray,
    indices: list[int] | None = None,
    iter_max: int = 10,
    keep_replacements: bool = False,
    threshold: float | None = None,
) -> tuple[list[int], list[list[int]] | None, int, bool]:
    """Run N-FINDR algorithm.

    The implementation corresponds to iter="points", estimator="Cramer"
    from the `unmixR` R package.

    Parameters
    ----------
    x : np.ndarray
        N-dimensional data matrix
    indices : Optional[List[int]], optional
        List of the initial points indices, by default generated randomly
    iter_max : int, optional
        Maximum number of outer loops, by default 10
    keep_replacements : bool, optional
        Return list of replacements as well as the list of the best indices,
        by default False
    threshold : float or None, optional
        Minimum volume-ratio improvement required to accept a vertex swap.
        ``None`` (default) auto-computes ``1 + max(1.5e-8, 10 * eps)`` where
        *eps* is the machine epsilon of ``x.dtype``, keeping the threshold
        above the floating-point noise floor for both float32 and float64.
        Pass an explicit value to override.

    Returns
    -------
    endmember_indices: List[int]
        List of indices giving the largest volume, i.e. the found endmember points.
        The list is sorted so the output would be more stable.
    replacements: list[list[int]] or None
        Candidate endmember index sets per iteration; only when ``keep_replacements=True``.
    n_iters : int
        Number of iterations performed before stopping.
    is_replacement : bool
        ``True`` if a replacement was made in the last iteration (i.e. the loop
        was stopped by ``iter_max``, not by convergence).
    """
    # Prepare data matrix
    m, n = x.shape
    p = n + 1

    # Get initial indices
    if indices is None:
        indices = np.random.choice(np.arange(m), p, replace=False)  # noqa: NPY002

    n_iters = 0
    is_replacement = True
    indices_best = indices.copy()
    replacements = [indices_best.copy()]
    x_padded = _pad_ones(x)
    Einv = np.linalg.inv(_simplex_E_padded(x_padded, indices_best))

    # Auto-compute threshold above the dtype noise floor when not provided.
    # For float32 (eps ~1.2e-7): threshold ~1 + 1.2e-6 (above quantisation noise).
    # For float64 (eps ~2.2e-16): max(1.5e-8, 2.2e-15) = 1.5e-8, preserving
    # the original float64 behaviour.
    if threshold is None:
        threshold = 1.0 + max(1.5e-8, 10.0 * np.finfo(x.dtype).eps)

    while (n_iters < iter_max) and is_replacement:
        n_iters += 1
        is_replacement = False
        for j in range(p):
            # Equivalent to _estimate_volume_change(..., endmembers=j), but avoids
            # repeated padding/slicing allocations in the hot path.
            row = np.ascontiguousarray(Einv[j, :])
            estimates = np.abs(x_padded @ row)
            i = int(np.argmax(estimates))
            if estimates[i] > threshold:
                # Update current simplex vertices
                indices_best[j] = i
                Einv = np.linalg.inv(_simplex_E_padded(x_padded, indices_best))
                # Mark that a replacement took place
                is_replacement = True
                # For debugging
                if keep_replacements:
                    replacements.append(indices_best.copy())

    # Sort the values to have same output if the endmebers are the same
    indices_best.sort()

    if keep_replacements:
        return indices_best, replacements, n_iters, is_replacement

    return indices_best, None, n_iters, is_replacement


class NFINDR(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """NFINDR unmixing algorithm.

    Finds the endmembers using the N-FINDR algorithm. Given the endmembers, decompose
    the data to the endmember coefficients using non-negative least squares (NNLS).
    The data is expected to already have reduced dimensionality.

    Parameters
    ----------
    n_endmembers : int, default=None
        Number of endmembers to find.

    initial_indices : List[int], default=None
        List of row indices to be used as initial points for NFINDR

    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
        Works the same as random_state in `sklearn.decomposition.PCA`

    work_dtype : {"float32", "float64"} or None, default=None
        Floating-point precision used for the simplex search.
        ``None`` (default) preserves the dtype of the input array.
        ``"float64"`` upcasts to double precision, which is useful when the
        input is already float32 but higher numerical accuracy is required.
        ``"float32"`` downcasts to single precision to reduce memory use.

    Attributes
    ----------
    endmembers_ : ndarray of shape (n_endmembers, n_endmembers-1)
        Matrix of vertex points found by NFINDR algorithm

    initial_indices_ : List[int] of len (n_endmembers,)
        List of initial points indices.

    endmember_indices_ : List[int] of len (n_endmembers,)
        List of final endmember points indices.

    n_samples_ : int
        Number of samples in the training data.

    n_endmembers_ : int
        Number of endmembers estimated during the training. I.e. either number of
        columns in the training data + 1 or explicitly provided `n_endmembers`

    volume_ : float
        The volume of the simplex formed by `endmembers_` vertex points.

    n_iters_ : int
        Number of iterations performed by the N-FINDR algorithm.

    converged_ : bool
        ``True`` if the algorithm converged before reaching the iteration limit,
        ``False`` if it was stopped early due to `iter_max`.

    Examples
    --------
    >>> import numpy as np
    >>> from ramappy.analysis.nfindr.nfindr import NFINDR
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> nf = NFINDR()
    >>> nf.fit(X)
    NFINDR()
    >>> print(nf.endmembers_)
    [[-1. -1.]
     [-2. -1.]
     [ 3.  2.]]
    >>> print(nf.transform(X))
    [[1.00000000e+00 0.00000000e+00 7.85046229e-17]
     [0.00000000e+00 1.00000000e+00 0.00000000e+00]
     [1.00000000e+00 1.00000000e+00 0.00000000e+00]
     [0.00000000e+00 1.00000000e+00 1.00000000e+00]
     [1.00000000e+00 0.00000000e+00 1.00000000e+00]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    """

    def __init__(
        self,
        n_endmembers=None,
        initial_indices=None,
        random_state=None,
        work_dtype=None,
    ) -> None:
        self.n_endmembers = n_endmembers
        self.initial_indices = initial_indices
        self.random_state = random_state
        self.work_dtype = work_dtype

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.

        Raises
        ------
        ValueError
            If the format is not recognized.
        """
        X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True)
        n_samples, n_features = X.shape
        n_endmembers = self.n_endmembers or (n_features + 1)

        if n_endmembers > n_features + 1:
            raise ValueError(
                "Dimension of data is too high. Please, reduce it (e.g., by PCA) or use `fit_transform` to directly reduce the dimensionality and apply NFINDR"
            )
        elif n_endmembers < n_features + 1:
            raise ValueError("Dimension of the data is too low. Please consider reducing the number of components")

        if self.initial_indices is None:
            random_state = check_random_state(self.random_state)
            initial_indices = random_state.choice(np.arange(n_samples), n_endmembers, replace=False)
        else:
            initial_indices = np.array(self.initial_indices)

        # Determine working dtype. None = preserve input dtype.
        # For float32 inputs, the convergence threshold is raised above
        # the float32 noise floor; pass work_dtype="float64" to force
        # double-precision arithmetic when higher accuracy is needed.
        np_dtype = {"float32": np.float32, "float64": np.float64}.get(self.work_dtype, X.dtype)  # type: ignore
        X_reduced = np.ascontiguousarray(X[:, : (n_endmembers - 1)], dtype=np_dtype)
        endmember_indices, _, n_iters, is_replacement = nfindr(X_reduced, initial_indices)
        if is_replacement:
            warnings.warn(
                "N-FINDR: the maximum number of iterations was reached. The algorithm may not have fully converged.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.endmember_indices_ = endmember_indices
        self.n_iters_ = n_iters
        self.converged_ = not is_replacement
        self.endmembers_ = X[endmember_indices, :]
        self.n_endmembers_ = n_endmembers
        self.initial_indices_ = initial_indices.tolist()
        self.n_samples_ = n_samples
        self.volume_ = simplex_volume(self.endmembers_)

        return self

    def transform(self, X, method="barycentric"):
        """Transform X to endmembers coefficients.

        X is converted to coefficients of previously found endmembers

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_endmembers)
            Decomposition of X to the endmember coefficients, where `n_samples`
            is the number of samples and `n_endmembers` is the number of the endmembers

        Raises
        ------
        ValueError
            If the method is not recognized.

        Notes
        -----
        The same pre-treatment (e.g., PCA) must be applied to the X as it was for
        the data used for fitting.
        """
        check_is_fitted(self)

        X = check_array(X, dtype=[np.float64, np.float32])  # , reset=False)

        if method == "barycentric":
            X_transformed = cart2bary(X[:, : (self.n_endmembers_ - 1)], self.endmembers_)
        elif method == "nnls":
            X_transformed = np.array([nnls(self.endmembers_.T, x)[0] for x in X[:, : (self.n_endmembers_ - 1)]])
        else:
            raise ValueError(f"Unexpected method '{method}'. Must be either 'barycentric' or 'nnls'.")

        return X_transformed

    def inverse_transform(self, X):
        """Transform data back to its original space.

        In other words, return an input `X_original` whose transform would be X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_endmembers)
            New data, where `n_samples` is the number of samples
            and `n_endmembers` is the number of endmembers.

        Returns
        -------
        X_original array-like of shape (n_samples, n_features)
            Original data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        """
        return np.array(X) @ self.endmembers_

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model with X and apply unmixing on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Ignored.

        **fit_params : dict
            For API compatibility (ignored).

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_endmembers)
            Transformed values.
        """
        self.fit(X)
        return self.transform(X)

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.endmembers_.shape[0]
