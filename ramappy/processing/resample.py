"""Spectral-axis resampling processing step."""

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.interpolate import make_interp_spline

from ramappy import const
from ramappy.core import SpectralMap, Spectrum
from ramappy.core.pipeline import (
    ParamsAcceptRoIX,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)
from ramappy.utils import common_roi_overlap, merge_arrays, whittaker_smooth


class StepResampleParams(StepParams, ParamsAcceptRoIX):
    """Parameters for `resample`."""

    method: Literal["spline", "whittaker"] = "spline"
    """Resampling method (`"spline"` or `"whittaker"`)."""

    spline_kind: Literal["zero", "slinear", "quadratic", "cubic"] | None = "cubic"
    """Spline interpolation kind when ``method='spline'``."""

    spectrum_step: float | None = None
    """Step used to build an evenly spaced grid when ``x_grid`` is not provided."""

    # x_grid: npt.ArrayLike | None = None  # either this, or ref_spectrum_id
    ref_spectrum_id: str | None = None
    """Identifier of the reference spectrum to use for resampling. If provided, the x-axis of this spectrum will be used as the target grid."""


@pipeline_step(
    step_name="resample",
    params_validator=SingleModelParamsValidator(StepResampleParams),
    friendly_name="Resample",
    step_category=StepClass.PROCESSING,
    modifies_spectral_axis=True,
)
def resample(
    spectral_map: SpectralMap | Spectrum,
    *,
    method: Literal["spline", "whittaker"] = "spline",
    spline_kind: Literal["zero", "slinear", "quadratic", "cubic"] | None = "cubic",
    spectrum_step: float | None = None,
    roi_x: npt.ArrayLike | None = None,
    x_grid: npt.ArrayLike | None = None,
):
    """Resample spectra onto a new spectral grid.

    Parameters
    ----------
    spectral_map
        Map or spectrum container to be resampled in place.
    method
        Resampling method (`"spline"` or `"whittaker"`).
    spline_kind
        Spline interpolation kind when ``method='spline'``.
    spectrum_step
        Step used to build an evenly spaced grid when ``x_grid`` is not
        provided.
    roi_x
        Target ROI used to build ``x_grid`` when no explicit grid is given.
    x_grid
        Explicit per-region spectral grid.

    Raises
    ------
    ValueError
        If method is unknown, required inputs are missing, ROIs are
        incompatible, or interpolation fails.
    """
    if method not in {"spline", "whittaker"}:
        raise ValueError(f"Method {method} is not recognized.")

    check_roi = True
    roi_x_arr: np.ndarray | None = None
    if x_grid is None:
        if spectrum_step is None:
            raise ValueError("Missing spectrum step for resampling")
        if roi_x is None:
            roi_x_arr = np.asarray(spectral_map.roi_x)
            check_roi = False
        else:
            roi_x_arr = np.asarray(roi_x)
        x_grid = [np.arange(float(r[0]), float(r[1]) + spectrum_step / 2.0, spectrum_step) for r in roi_x_arr]
    else:
        # Check if x_grid is a single 1D grid (either a 1D array, or a list of numbers)
        # We can check this by testing if the first element is a scalar (i.e. not iterable)
        x_grid_arr = np.asarray(x_grid, dtype=object)
        first_elem = x_grid_arr[0] if x_grid_arr.size > 0 else None
        is_flat_grid = first_elem is None or not hasattr(first_elem, "__iter__") or isinstance(first_elem, (str, bytes))
        x_grid = [np.asarray(x_grid, dtype=float)] if is_flat_grid else [np.asarray(r, dtype=float) for r in x_grid_arr]
        roi_x_arr = np.array([[ri[0], ri[-1]] for ri in x_grid])

    if check_roi:
        assert roi_x_arr is not None
        roi_x_common = common_roi_overlap(roi_x_arr, spectral_map.roi_x)
        if len(roi_x_common) == 0:
            raise ValueError("No overlapping regions of interest")
        # Allow endpoints to overshoot the data boundary by up to one mean spectral step.
        # Small overruns are safe because the spline branch already clamps out-of-range
        # queries to edge values, so no unphysical extrapolation occurs.
        delta = np.diff(spectral_map.x).mean() if len(spectral_map.x) > 1 else 1.0
        if not len(roi_x_arr) == len(roi_x_common) or not np.allclose(
            roi_x_common.ravel(), roi_x_arr.ravel(), atol=delta
        ):
            raise ValueError("Regions of interest do not fully match")

    new_x = np.hstack(x_grid, dtype=spectral_map.x.dtype)
    intensities = np.empty((spectral_map.data.shape[const.Axis.PIXEL], len(new_x)), dtype=spectral_map.data.dtype)

    assert roi_x_arr is not None
    offset = 0
    for i, x_region in enumerate(x_grid):
        roi_x_cur = roi_x_arr[i]
        idxs, x_idx, _ = spectral_map.get_indices(mask=None, roi_x=roi_x_cur)
        x_cur = np.array(x_region, dtype=spectral_map.x.dtype)
        x_cur.sort()
        try:
            if method == "whittaker":
                x_merge, (old_roi_idx, grid_idx) = merge_arrays(spectral_map.x[x_idx], x_cur)

                w = np.zeros_like(x_merge, dtype=spectral_map.data.dtype)  # weights
                w[old_roi_idx] = 1.0  # set weight for observed values to 1

                intensities_new = np.zeros(
                    (spectral_map.data[idxs].shape[const.Axis.PIXEL], len(x_merge)),
                    dtype=spectral_map.data.dtype,
                )
                intensities_new[:, old_roi_idx] = spectral_map.data[idxs]

                intensities[:, offset : offset + len(x_cur)] = whittaker_smooth(
                    intensities_new, 0.1, order=4, w=w, x=x_merge
                )[:, grid_idx]

            elif method == "spline":
                spline_kind = spline_kind or "cubic"
                src_data = spectral_map.data[idxs]
                # Use edge values for any out-of-range points (e.g., tiny floating-point
                # boundary overruns after ROI validation). Extrapolating with a
                # polynomial outside the measured range can produce large unphysical
                # values; clamping to the edge spectrum is always conservative.
                fill_below = src_data[:, 0]
                fill_above = src_data[:, -1]
                x_src = spectral_map.x[x_idx]
                if spline_kind == "zero":
                    # Piecewise-constant: return the value of the left-hand neighbour.
                    # np.clip ensures out-of-range queries map to edge values.
                    idx = np.clip(
                        np.searchsorted(x_src, x_cur, side="right") - 1,
                        0,
                        len(x_src) - 1,
                    )
                    intensities[:, offset : offset + len(x_cur)] = src_data[:, idx]
                else:
                    k = {"slinear": 1, "quadratic": 2, "cubic": 3}[spline_kind]
                    spl = make_interp_spline(x_src, src_data, k=k, axis=int(const.Axis.SPECTRAL), check_finite=False)
                    result = spl(x_cur).astype(src_data.dtype, copy=False)
                    below = x_cur < x_src[0]
                    above = x_cur > x_src[-1]
                    if below.any():
                        result[:, below] = fill_below[:, np.newaxis]
                    if above.any():
                        result[:, above] = fill_above[:, np.newaxis]
                    intensities[:, offset : offset + len(x_cur)] = result
        except ValueError as e:
            raise ValueError("Could not resample spectrum over the provided range.") from e
        offset += len(x_cur)

    spectral_map.data = intensities
    spectral_map.x = new_x
    spectral_map.roi_x = roi_x_arr  # type: ignore[assignment]
