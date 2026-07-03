"""SVD-based denoising step and related utilities."""

from typing import Literal

import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import svds

from ramappy.core import SpectralMap, Spectrum
from ramappy.core.pipeline import (
    ParamsSeparateRegions,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)


def compute_spatial_signal_ratio(Vt, shape: tuple[int, int], roi_threshold: float = 0.5):
    """
    Calculates the spatial signal ratio for each SV of Vt, using the data of the hyperspectral image.
    The specified threshold, if present, determines the size of the region of interest 'boundary'.

    Parameters:
    -----------
    Vt: ndarray
        The Vt matrix obtained through SVD of the hyperspectral image data
    shape: (int, int)
        Shape of the 2-D map
    roi_threshold: float
        The threshold (0,1) to define the region of interest.

    Returns:
    --------
    spatial_ratios: ndarray
        An array containing the spatial signal ratio for each singular value of Vt
    """

    if not (0 < roi_threshold <= 1):
        raise ValueError("'roi_threshold' must be in the interval (0, 1].")

    n_rows, n_cols = shape
    num_pixels = n_rows * n_cols
    if Vt.shape[1] != num_pixels:
        raise ValueError(f"Vt shape {Vt.shape} is incompatible with map shape {shape}.")

    Vcube = Vt.T.reshape(n_rows, n_cols, -1)
    Vcube -= Vcube.mean(axis=(0, 1), keepdims=True)

    Vz = np.abs(np.fft.fftshift(np.fft.fft2(Vcube, axes=(0, 1)), axes=(0, 1)))

    # get inner ("signal") region, according to the threshold (0 is no pixels, 1 is whole 2-D map)
    region_row = int(np.round(n_rows * (1 - roi_threshold) / 2))
    region_col = int(np.round(n_cols * (1 - roi_threshold) / 2))
    rows = (region_row, n_rows - region_row)
    cols = (region_col, n_cols - region_col)

    inner_pixels_sum = Vz[slice(*rows), slice(*cols), :].sum(axis=(0, 1))
    outer_pixels_sum = np.fmax(np.spacing(1), Vz.sum(axis=(0, 1)) - inner_pixels_sum)
    # outer_pixels_sum = Vz.sum(axis=(0,1)) - inner_pixels_sum

    num_inner_pixels = np.diff(rows)[0] * np.diff(cols)[0]
    if num_inner_pixels <= 0:
        raise ValueError("Computed empty ROI; please increase roi_threshold or check map shape.")
    scale = (num_pixels - num_inner_pixels) / num_inner_pixels
    # scale = (1 - threshold**2) / threshold**2  # up to rounding error with region_row, region_col...

    spatial_ratio = scale * inner_pixels_sum / outer_pixels_sum - 1

    return spatial_ratio


class StepSmoothSVDParams(StepParams, ParamsSeparateRegions):
    """Parameters for `smooth_svd`."""

    method: Literal["manual", "spatial_ratio"] = "spatial_ratio"
    """The singular value selection method. Options are 'manual' or 'spatial_ratio'."""

    num_sv: int | None = None
    """The number of singular values to retain. Only used when `method == 'manual'`."""

    roi_threshold: float = 0.5
    """Region-of-interest size used in spatial-ratio mode. Must be in (0, 1]."""

    spatial_ratio_threshold_multiplier: float = 3.5
    """Multiplier applied to `std(spatial_signal_ratio)` to select singular values in spatial-ratio mode."""


@pipeline_step(
    step_name="smooth_svd",
    params_validator=SingleModelParamsValidator(StepSmoothSVDParams),
    friendly_name="Smooth (SVD)",
    step_category=StepClass.PROCESSING,
)
def smooth_svd(
    spectral_map: SpectralMap | Spectrum,
    *,
    method: Literal["manual", "spatial_ratio"] = "spatial_ratio",
    num_sv: int | None = None,
    roi_threshold: float = 0.5,
    spatial_ratio_threshold_multiplier: float = 3.5,
):
    """Calculate the Singular Value Decomposition.

    Parameters
    ----------
    method: `manual` or `spatial_ratio`, default=`spatial_ratio`
        The singular value selection method.
    num_sv: int
        The number of singular value to retain. Only used when `method == 'manual'`.
    roi_threshold: float, default=0.5
        Region-of-interest size used in spatial-ratio mode. Must be in (0, 1].
    spatial_ratio_threshold_multiplier: float, default=3.5
        Multiplier applied to `std(spatial_signal_ratio)` to select singular values in spatial-ratio mode.

    Raises
    ------
    ValueError
        If the method is not recognized.
    """
    intensities = spectral_map.data

    if method not in {"manual", "spatial_ratio"}:
        raise ValueError(f"Unknown SVD selection method: {method!r}")

    if not (0 < roi_threshold <= 1):
        raise ValueError("'roi_threshold' must be in the interval (0, 1].")

    if spatial_ratio_threshold_multiplier <= 0:
        raise ValueError("'spatial_ratio_threshold_multiplier' must be > 0.")

    matrix = intensities.T
    if method == "manual":
        if num_sv is None:
            raise ValueError("'num_sv' must be provided when method='manual'.")

        min_dim = min(matrix.shape)
        if not 1 <= num_sv <= min_dim:
            raise ValueError(f"'num_sv' must be in [1, {min_dim}] for matrix shape {matrix.shape}; got {num_sv}.")

        if num_sv == min_dim:
            U, s, Vt = svd(matrix, full_matrices=False)
            intensities = U * s @ Vt
        else:
            U, s, Vt = svds(matrix, k=num_sv, which="LM")
            sort_idx = np.argsort(s)[::-1]
            U = U[:, sort_idx]
            s = s[sort_idx]
            Vt = Vt[sort_idx, :]
            intensities = U * s @ Vt
    else:
        U, s, Vt = svd(matrix, full_matrices=False)

        if not isinstance(spectral_map, SpectralMap):
            raise ValueError("SVD spatial-ratio mode requires a SpectralMap, not a Spectrum.")
        spatial_signal_ratio = compute_spatial_signal_ratio(
            Vt, roi_threshold=roi_threshold, shape=spectral_map.map_shape
        )

        # lim = 700
        # threshold = np.std(spatial_signal_ratio[lim:]) * spatial_ratio_threshold_multiplier
        threshold = np.std(spatial_signal_ratio) * spatial_ratio_threshold_multiplier

        sv_keep = np.flatnonzero(spatial_signal_ratio > threshold)
        if sv_keep.size == 0:
            sv_keep = np.array([int(np.argmax(spatial_signal_ratio))])

        intensities = U[:, sv_keep] * s[sv_keep] @ Vt[sv_keep, :]

    intensities = intensities.T

    spectral_map.data[:] = intensities[:]
