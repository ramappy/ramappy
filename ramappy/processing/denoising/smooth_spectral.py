"""Spectral denoising step for smoothers along the wavenumber axis."""

import warnings
from typing import Annotated, Literal

import numpy.typing as npt
from pydantic import ConfigDict, Field
from scipy.signal import savgol_filter

from ramappy import const
from ramappy.core import SpectralMap, Spectrum
from ramappy.core.pipeline import (
    ParamsAcceptMask,
    ParamsAcceptRoIX,
    ParamsSeparateRegions,
    StepClass,
    StepParams,
    UnionParamsValidator,
    pipeline_step,
)
from ramappy.utils import whittaker_smooth


class StepSmoothSpectrumParams(StepParams, ParamsAcceptMask, ParamsAcceptRoIX, ParamsSeparateRegions):
    """Base parameter model for `smooth_spectral`."""

    model_config = ConfigDict(extra="ignore")


class StepSmoothSpectrumParamsWhittaker(StepSmoothSpectrumParams):
    """Parameters for `smooth_spectral` with Whittaker smoother."""

    method: Literal["whittaker"] = "whittaker"
    """Use the Whittaker smoother."""

    lambda_: float | None = 10
    """The smoothing parameter for the Whittaker smoother. Higher values result in smoother spectra."""

    d: int | None = Field(ge=1)
    """The order of the difference operator for the Whittaker smoother. Must be a positive integer."""

    ignore_x_axis: bool | None = False
    """If True, ignore the x-axis values and assume an equispaced grid for the Whittaker smoother. If False, use the actual x-axis values for smoothing. The non-equispaced x-axis variant is slower."""


class StepSmoothSpectrumParamsSavGol(StepSmoothSpectrumParams):
    """Parameters for `smooth_spectral` with Savitzky-Golay filter."""

    method: Literal["savgol"] = "savgol"
    """Use the Savitzky-Golay filter."""

    polyorder: int | None = 2
    """The order of the polynomial to use for the Savitzky-Golay filter. Must be a non-negative integer less than `window_length`. Higher values result in a better fit to the data but may introduce artifacts."""

    window_length: int | None = 7
    """The length of the window to use for the Savitzky-Golay filter. Must be a positive odd integer. Larger values result in more smoothing but may also remove important features from the spectra."""


@pipeline_step(
    step_name="smooth_spectral",
    params_validator=UnionParamsValidator(
        Annotated[
            StepSmoothSpectrumParamsWhittaker | StepSmoothSpectrumParamsSavGol,
            Field(discriminator="method"),
        ]
    ),
    friendly_name="Smooth Spectral",
    step_category=StepClass.PROCESSING,
    supports_preview=True,
)
def smooth_spectral(
    spectral_map: SpectralMap | Spectrum,
    method: Literal["whittaker", "savgol"] = "whittaker",
    mask: str | None = None,
    roi_x: npt.ArrayLike | None = None,
    **kwargs,
):
    """Apply a non-linear digital filter to remove noise from the maps corresponding to each wavenumber.

    Parameters
    ----------
    method : {"whittaker", "savgol"}, default="whittaker"
        "whittaker" : Use the Whittaker smoother.
        "savgol" : Use the Savitzky-Golay filter.
    mask : str or None, default=None
        Restrict search to this masked region in the 2-D spectral_map.
        If None, use the whole spectral_map.
    roi_x : array_like or None, default=None
        Restrict search to these regions in the Raman shift.
        If None, use the whole spectral axis.
        Example: roi_x = [[600,1800],[2800,3030]] or roi_x = [600,3030]

    Other Parameters
    ----------------
    **kwargs : dict
        For `method == 'whittaker'`:
            `lambda_`: float, default=10
                The smoothing parameter
        For `method == 'savgol`:
            `window_length`: odd int, default=7
                The length of the window. N.B. should be an odd integer.
            `polyorder`: int, default=2
                The order of the polynomial to use
    """
    if not hasattr(spectral_map, "img_width") and mask is not None:
        mask = None
        warnings.warn("Mask is not supported for single-spectrum data, ignoring it", stacklevel=2)
    idxs, x_idx, _pixel_idx = spectral_map.get_indices(mask, roi_x)
    intensities = spectral_map.data[idxs]

    if method == "savgol":
        window_length = kwargs.get("window_length", 7)
        polyorder = kwargs.get("polyorder", 2)
        spectral_map.data[idxs] = savgol_filter(
            intensities, window_length=window_length, polyorder=polyorder, deriv=0, axis=const.Axis.SPECTRAL
        )
    elif method == "whittaker":
        lambda_ = kwargs.get("lambda_", 10)
        d = kwargs.get("d", 2)

        x = None if kwargs.get("ignore_x_axis", False) else spectral_map.x[x_idx]
        # if `ignore_x_axis` assume equispaced grid for the x axis

        spectral_map.data[idxs] = whittaker_smooth(intensities, lambda_, d, x=x)
