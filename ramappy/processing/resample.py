"""Spectral-axis resampling processing step."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import model_validator
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
from ramappy.utils import common_roi_overlap, merge_arrays, select_x_indices, whittaker_smooth


class StepResampleParams(StepParams, ParamsAcceptRoIX):
    """Parameters for `resample`."""

    method: Literal["spline", "whittaker"] = "spline"
    """Resampling method (`"spline"` or `"whittaker"`)."""

    spline_kind: Literal["zero", "slinear", "quadratic", "cubic"] | None = "cubic"
    """Spline interpolation kind when ``method='spline'``."""

    spectrum_step: float | None = None
    """Step used to build an evenly spaced grid. Mutually exclusive with ``ref_spectrum_id``."""

    ref_spectrum_id: str | None = None
    """Identifier of the reference spectrum to use for resampling. If provided, the x-axis of this spectrum will be used as the target grid. Mutually exclusive with ``spectrum_step``."""

    @model_validator(mode="after")
    def _check_mutually_exclusive_grid(self) -> StepResampleParams:
        provided = sum(x is not None for x in (self.spectrum_step, self.ref_spectrum_id))
        if provided != 1:
            raise ValueError("Exactly one of 'spectrum_step' or 'ref_spectrum_id' must be provided.")
        return self


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
    ref_spectrum_id: str | None = None,
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
        Step used to build an evenly spaced grid. Mutually exclusive with
        ``ref_spectrum_id`` and ``x_grid``.
    roi_x
        Target ROI used to restrict or build the target grid when ``spectrum_step`` or ``ref_spectrum_id`` is used.
    x_grid
        Explicit per-region spectral grid. Mutually exclusive with
        ``spectrum_step`` and ``ref_spectrum_id``.
    ref_spectrum_id
        Identifier of the reference spectrum to use for resampling. If provided,
        the x-axis of this spectrum will be used as the target grid. Mutually
        exclusive with ``spectrum_step`` and ``x_grid``.

    Raises
    ------
    ValueError
        If method is unknown, parameters are invalid or conflicting (e.g.
        multiple target grid parameters provided), reference spectrum is not
        found, ROIs are incompatible, or interpolation fails.
    """
    if method not in {"spline", "whittaker"}:
        raise ValueError(f"Method {method} is not recognized.")

    provided = sum(x is not None for x in (spectrum_step, ref_spectrum_id, x_grid))
    if provided != 1:
        raise ValueError("Exactly one of 'spectrum_step', 'ref_spectrum_id', or 'x_grid' must be provided.")

    if spectrum_step is not None:
        check_roi = roi_x is not None
        roi_x_arr = np.asarray(spectral_map.roi_x) if roi_x is None else np.asarray(roi_x)
        x_grid_list = [np.arange(float(r[0]), float(r[1]) + spectrum_step / 2.0, spectrum_step) for r in roi_x_arr]
    elif ref_spectrum_id is not None:
        check_roi = True
        if isinstance(spectral_map, SpectralMap):
            try:
                ref_spec, _ = spectral_map.get_reference_spectrum(ref_spectrum_id)
            except KeyError as e:
                raise ValueError(f"Reference spectrum '{ref_spectrum_id}' not found.") from e
        else:
            raise ValueError("ref_spectrum_id is provided but spectral_map is not a SpectralMap instance")

        target_roi = roi_x if roi_x is not None else getattr(ref_spec, "roi_x", None)
        roi_x_arr = np.array([[ref_spec.x[0], ref_spec.x[-1]]]) if target_roi is None else np.asarray(target_roi)
        x_indices = select_x_indices(ref_spec.x, roi_x_arr, keep_regions=True)
        if not isinstance(x_indices, list):
            x_indices = [x_indices]
        x_grid_list = [np.asarray(ref_spec.x[idx], dtype=float) for idx in x_indices]
    else:  # x_grid is not None
        check_roi = True
        x_grid_arr = np.asarray(x_grid, dtype=object)
        first_elem = x_grid_arr[0] if x_grid_arr.size > 0 else None
        is_flat_grid = first_elem is None or not hasattr(first_elem, "__iter__") or isinstance(first_elem, (str, bytes))
        x_grid_list = (
            [np.asarray(x_grid, dtype=float)] if is_flat_grid else [np.asarray(r, dtype=float) for r in x_grid_arr]
        )
        roi_x_arr = np.array([[ri[0], ri[-1]] for ri in x_grid_list])

    if check_roi:
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

    new_x = np.hstack(x_grid_list, dtype=spectral_map.x.dtype)
    intensities = np.empty((spectral_map.data.shape[const.Axis.PIXEL], len(new_x)), dtype=spectral_map.data.dtype)

    offset = 0
    for i, x_region in enumerate(x_grid_list):
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
