"""Intensity normalization processing step and helper transforms."""

import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.integrate import simpson
from sklearn.preprocessing import minmax_scale, normalize, scale

from ramappy import const
from ramappy.core import SpectralMap, Spectrum
from ramappy.core.pipeline import (
    ParamsAcceptMask,
    ParamsAcceptRoIX,
    ParamsSeparateRegions,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)
from ramappy.utils import clean_roi, common_roi_overlap, find_nearest_x, select_x_indices


def norm_lp(s: np.ndarray, x: np.ndarray | None, axis: int, norm: str, **kwargs) -> np.ndarray:
    """Apply L1, L2, or max normalization."""
    return normalize(s, axis=axis, norm=norm)


def norm_scale(s: np.ndarray, x: np.ndarray | None, axis: int, **kwargs) -> np.ndarray:
    """Apply standard scaling (zero mean, unit variance)."""
    return scale(s, axis=axis)


def norm_minmax_scale(s: np.ndarray, x: np.ndarray | None, axis: int, **kwargs) -> np.ndarray:
    """Apply min-max scaling to [0, 1] range."""
    return minmax_scale(s, axis=axis)


def norm_frobenius(s: np.ndarray, x: np.ndarray | None, axis: int, **kwargs) -> np.ndarray:
    """Apply Frobenius norm normalization."""
    return s / np.linalg.norm(s, "fro")


def norm_wavenumber(s: np.ndarray, x: np.ndarray, axis: int, wn: float, **kwargs) -> np.ndarray:
    """Normalize by the intensity at a specific wavenumber."""
    wn_idx = find_nearest_x(x, wn)
    # Use keepdims=True for better broadcasting
    divisor = s[:, [wn_idx]] if axis == 1 else s[[wn_idx], :]
    return s / divisor


def norm_area(s: np.ndarray, x: np.ndarray, axis: int, **kwargs) -> np.ndarray:
    """Normalize by the area under the curve."""
    integrals = np.abs(simpson(s, x=x, axis=axis))
    # Reshape for proper broadcasting
    shape = [1, 1]
    shape[1 - axis] = -1
    return s / integrals.reshape(shape)


NORM_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "l1": norm_lp,
    "l2": norm_lp,
    "max": norm_lp,
    "scale": norm_scale,
    "minmax_scale": norm_minmax_scale,
    "frobenius": norm_frobenius,
    "spectral_position": norm_wavenumber,
    "area": norm_area,
}


NormName = Literal[
    "l1",
    "l2",
    "max",
    "scale",
    "minmax_scale",
    "frobenius",
    "spectral_position",
    "area",
]


def _compose_x_index(base_x_idx: np.ndarray | slice, roi: np.ndarray | slice) -> np.ndarray | slice:
    """Compose an x-indexer returned by `get_indices` with a ROI indexer relative to that x-axis.

    Parameters
    ----------
    base_x_idx
        Indexer produced by `select_x_indices(self.x, roi_x)`.
    roi
        Indexer relative to the *selected* x-axis.
    """
    if isinstance(base_x_idx, slice):
        if not isinstance(roi, slice):
            raise TypeError("ROI indices must be slices")

        base_start = 0 if base_x_idx.start is None else base_x_idx.start
        base_step = 1 if base_x_idx.step is None else base_x_idx.step
        roi_start = 0 if roi.start is None else roi.start
        # NOTE: `roi.stop` is interpreted in the sliced coordinate system.
        composed_stop = None if roi.stop is None else base_start + roi.stop * base_step

        composed_start = base_start + roi_start * base_step
        composed_step = base_step if roi.step is None else base_step * roi.step
        return slice(composed_start, composed_stop, composed_step)

    # base_x_idx is an array of indices.
    return base_x_idx[roi]


def _normalize_intensities(
    intensities: np.ndarray,
    x_axis: np.ndarray,
    *,
    norm: str = "l2",
    by_pixel: bool = True,
    roi_x: npt.ArrayLike | None = None,
    separate_regions: bool = True,
    wn: float | None = None,
) -> np.ndarray:
    """
    Normalize intensity array.

    Parameters
    ----------
    intensities : np.ndarray
        Input intensity array to normalize.
    x_axis : np.ndarray
        X-axis values (wavenumbers).
    norm : str, default="l2"
        Normalization method to apply.
    by_pixel : bool, default=True
        If True, normalize by pixel; otherwise by wavenumber.
    roi_x : array_like or None, default=None
        Spectral regions of interest.
    separate_regions : bool, default=True
        If True, normalize each ROI region separately.
    wn : float or None, default=None
        Reference wavenumber for wavenumber normalization.

    Returns
    -------
    np.ndarray
        Normalized intensity array.

    Raises
    ------
    ValueError
        If norm is invalid or required parameters are missing.
    """
    axis = const.Axis.SPECTRAL if by_pixel else const.Axis.PIXEL

    # Validate norm function
    norm_func = NORM_FUNCTIONS.get(norm)
    if norm_func is None:
        raise ValueError(f"Invalid norm: {norm}. Valid options: {list(NORM_FUNCTIONS.keys())}")

    # Validate parameters for specific norms
    if norm == "spectral_position" and wn is None:
        raise ValueError("spectral_position normalization requires 'wn' parameter")

    # Handle incompatible parameter combinations
    if norm == "frobenius" and (by_pixel or separate_regions):
        warnings.warn(
            f"'by_pixel' and 'separate_regions' are incompatible with '{norm}' normalization. "
            "These settings will be ignored.",
            UserWarning,
            stacklevel=3,
        )
        by_pixel = False
        separate_regions = False

    if norm == "spectral_position" and not by_pixel:
        warnings.warn(
            "'by_pixel=False' is incompatible with 'spectral_position' normalization. Setting to True.",
            UserWarning,
            stacklevel=3,
        )
        by_pixel = True

    if norm == "area" and not by_pixel:
        warnings.warn(
            "'by_pixel=False' is incompatible with 'area' normalization. Setting to True.", UserWarning, stacklevel=3
        )
        by_pixel = True

    # Process ROI
    if roi_x is not None:
        roi_x_arr = np.asarray(roi_x)
        roi_x_arr = clean_roi(roi_x_arr)
        x_extent = np.array([[float(np.min(x_axis)), float(np.max(x_axis))]], dtype=float)
        roi_x = common_roi_overlap(roi_x_arr, x_extent)
        separate_regions = separate_regions and (roi_x is not None)
    else:
        separate_regions = False

    # Apply normalization
    if separate_regions:
        x_indices = select_x_indices(x_axis, np.asarray(roi_x) if roi_x is not None else None, keep_regions=True)

        # Fast-path: ROI is the whole current x-axis -> no need to preserve untouched columns.
        if (
            len(x_indices) == 1
            and isinstance(x_indices[0], slice)
            and (x_indices[0].start in {None, 0})
            and (x_indices[0].stop in {None, x_axis.shape[0]})
            and (x_indices[0].step in {None, 1})
        ):
            return norm_func(intensities, x_axis, axis=axis, norm=norm, wn=wn)

        normalized_intensities = intensities.copy()

        for roi in x_indices:
            normalized_intensities[:, roi] = norm_func(intensities[:, roi], x_axis[roi], axis=axis, norm=norm, wn=wn)
    else:
        normalized_intensities = norm_func(intensities, x_axis, axis=axis, norm=norm, wn=wn)

    return normalized_intensities


class StepNormalizeParams(StepParams, ParamsAcceptMask, ParamsAcceptRoIX, ParamsSeparateRegions):
    """Parameters for normalization step."""

    norm: NormName = "l2"
    """Normalization method to apply."""

    by_pixel: bool = True
    """If True, normalize each pixel spectrum independently; otherwise, normalize across pixels for each wavenumber."""

    wn: float | None = None
    """Reference wavenumber for 'spectral_position' normalization. Required when norm='spectral_position', ignored otherwise."""


@pipeline_step(
    step_name="normalize",
    params_validator=SingleModelParamsValidator(StepNormalizeParams),
    friendly_name="Normalize",
    step_category=StepClass.PROCESSING,
    supports_preview=True,
)
def normalize_intensities(
    spectral_map: SpectralMap | Spectrum,
    *,
    norm: str = "l2",
    by_pixel: bool = True,
    mask: str | None = None,
    roi_x: npt.ArrayLike | None = None,
    separate_regions: bool = True,
    wn: float | None = None,
) -> None:
    """
    Normalize the spectra in place.

    Parameters
    ----------
    spectral_map : SpectralMap or Spectrum
        The hyperspectral image or spectrum to normalize.
    norm : {'l1', 'l2', 'max', 'scale', 'minmax_scale', 'frobenius', 'wavenumber', 'area'}, default='l2'
        Normalization method to apply:

        - 'l1': L1 norm (sum of absolute values = 1)
        - 'l2': L2 norm (Euclidean norm = 1)
        - 'max': Max norm (maximum absolute value = 1)
        - 'scale': Standard scaling (zero mean, unit variance)
        - 'minmax_scale': Min-max scaling to [0, 1] range
        - 'frobenius': Frobenius norm
        - 'wavenumber': Normalize by intensity at specific wavenumber
        - 'area': Normalize by area under the curve
    by_pixel : bool, default=True
        If True, normalize each pixel spectrum independently.
        If False, normalize across pixels for each wavenumber.
    mask : str or None, default=None
        Name of mask to restrict processing to specific spatial regions.
        If None, process the entire SpectralMap.
    roi_x : array_like or None, default=None
        Spectral regions of interest to restrict processing.
        Can be a single range [start, end] or multiple ranges [[start1, end1], [start2, end2]].
        If None, use the entire spectral axis.
    separate_regions : bool, default=True
        If True and multiple spectral regions are specified,
        normalize each region independently.
    wn : float or None, default=None
        Reference wavenumber for 'wavenumber' normalization.
        Required when norm='wavenumber', ignored otherwise.

    Raises
    ------
    ValueError
        If invalid normalization method is specified or required parameters are missing.

    Notes
    -----
    This function modifies the SpectralMap object in place. The original intensities
    are overwritten with normalized values.

    Some normalization methods have restrictions:
    - 'frobenius' and 'wavenumber' norms ignore `by_pixel` and `separate_regions` settings
    - 'area' norm requires `by_pixel=True`
    - 'wavenumber' norm requires the `wn` parameter

    Examples
    --------
    >>> # L2 normalization by pixel
    >>> normalize_intensities(spectral_map, norm='l2', by_pixel=True)

    >>> # Wavenumber normalization at 1000 cm⁻¹
    >>> normalize_intensities(spectral_map, norm='wavenumber', wn=1000)

    >>> # Min-max scaling in specific spectral regions
    >>> normalize_intensities(spectral_map, norm='minmax_scale', roi_x=[[500, 1800], [2800, 3200]])
    """
    idxs, x_idx, pixel_idx = spectral_map.get_indices(mask, roi_x)

    # Work on the selected x-axis only.
    x_axis = spectral_map.x[x_idx]

    # If separate_regions is requested, avoid copying the whole selected block.
    # Instead, normalize each ROI region and write back only those columns.
    if roi_x is not None:
        roi_x_arr = np.asarray(roi_x)
        roi_x_arr = clean_roi(roi_x_arr)
        x_extent = np.array([[float(np.min(x_axis)), float(np.max(x_axis))]], dtype=float)
        roi_x_clean = common_roi_overlap(roi_x_arr, x_extent)
        separate_regions = separate_regions and (roi_x_clean is not None)
    else:
        roi_x_clean = None
        separate_regions = False

    axis = const.Axis.SPECTRAL if by_pixel else const.Axis.PIXEL
    norm_func = NORM_FUNCTIONS.get(norm)
    if norm_func is None:
        raise ValueError(f"Invalid norm: {norm}. Valid options: {list(NORM_FUNCTIONS.keys())}")

    if norm == "spectral_position" and wn is None:
        raise ValueError("spectral_position normalization requires 'wn' parameter")

    if norm == "frobenius" and (by_pixel or separate_regions):
        warnings.warn(
            f"'by_pixel' and 'separate_regions' are incompatible with '{norm}' normalization. "
            "These settings will be ignored.",
            UserWarning,
            stacklevel=3,
        )
        by_pixel = False
        separate_regions = False
        axis = const.Axis.PIXEL

    if norm == "spectral_position" and not by_pixel:
        warnings.warn(
            "'by_pixel=False' is incompatible with 'spectral_position' normalization. Setting to True.",
            UserWarning,
            stacklevel=3,
        )
        by_pixel = True
        axis = const.Axis.SPECTRAL

    if norm == "area" and not by_pixel:
        warnings.warn(
            "'by_pixel=False' is incompatible with 'area' normalization. Setting to True.",
            UserWarning,
            stacklevel=3,
        )
        by_pixel = True
        axis = const.Axis.SPECTRAL

    if separate_regions:
        x_regions = select_x_indices(x_axis, roi_x_clean, keep_regions=True)
        for roi in x_regions:
            col_idx = _compose_x_index(x_idx, roi)
            pixel_is_scalar = (not isinstance(pixel_idx, slice)) and (getattr(pixel_idx, "ndim", 0) == 0)
            col_is_scalar = (not isinstance(col_idx, slice)) and (getattr(col_idx, "ndim", 0) == 0)

            if isinstance(pixel_idx, slice) or isinstance(col_idx, slice) or pixel_is_scalar or col_is_scalar:
                region_idxs: Any = (pixel_idx, col_idx)
            else:
                region_idxs = np.ix_(pixel_idx, col_idx)

            spectral_map.data[region_idxs] = norm_func(
                spectral_map.data[region_idxs],
                x_axis[roi],
                axis=axis,
                norm=norm,
                wn=wn,
            )
    else:
        spectral_map.data[idxs] = norm_func(
            spectral_map.data[idxs],
            x_axis,
            axis=axis,
            norm=norm,
            wn=wn,
        )
