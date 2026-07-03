"""Spatial denoising step for per-wavenumber maps."""

from typing import Annotated, Literal

import numpy.typing as npt
from pydantic import ConfigDict, Field
from scipy.ndimage import gaussian_filter, median_filter  # , gaussian_filter1d

from ramappy.core import SpectralMap
from ramappy.core.pipeline import (
    ParamsAcceptRoIX,
    StepClass,
    StepParams,
    UnionParamsValidator,
    pipeline_step,
)


class StepSmoothMapParamsBase(StepParams, ParamsAcceptRoIX):
    """Base parameter model for `smooth_spatial`."""

    model_config = ConfigDict(extra="ignore")


class StepSmoothMapParamsMedian(StepSmoothMapParamsBase):
    """Parameters for `smooth_spatial` with median filter."""

    method: Literal["median"] = "median"
    """Use a median filter."""

    size: int = Field(default=3, ge=1)
    """Size of the median filter (must be an odd integer greater than 1)."""

    boundary_mode: Literal["reflect", "nearest"] = "reflect"
    """Boundary mode for the median filter. Options are 'reflect' or 'nearest'."""


class StepSmoothMapParamsGaussian(StepSmoothMapParamsBase):
    """Parameters for `smooth_spatial` with Gaussian filter."""

    method: Literal["gaussian"] = "gaussian"
    """Use a Gaussian filter."""

    sigma: float = Field(gt=0)
    """The scale of the Gaussian filter."""


@pipeline_step(
    step_name="smooth_spatial",
    params_validator=UnionParamsValidator(
        Annotated[
            StepSmoothMapParamsMedian | StepSmoothMapParamsGaussian,
            Field(discriminator="method"),
        ]
    ),
    friendly_name="Smooth Spatial",
    step_category=StepClass.PROCESSING,
)
def smooth_spatial(
    spectral_map: SpectralMap,
    *,
    roi_x: npt.ArrayLike | None = None,
    method: str = "median",
    size: int = 3,
    boundary_mode: Literal["reflect", "nearest"] = "reflect",
    sigma: float | None = None,
):
    """Apply a non-linear digital filter to remove noise from the maps corresponding to each wavenumber.

    Parameters
    ----------
    method : {"median", "gaussian"}, default="median"
        "median" : Use a median filter.
        "gaussian" : Use a Gaussian filter.
    size : int, default=3
        Used by `method == 'median'`. The size of the filter
    sigma : float or None, default=None
        Used by `method == 'gaussian'`. The scale of the filter

    Raises
    ------
    ValueError
        If the method is not recognized.
    """
    idxs, _, _ = spectral_map.get_indices(None, roi_x)

    # functions releases the GIL => use threads for parallel
    if method == "median":
        if size > 1:
            spectral_map.data[idxs] = spectral_map.apply_func(
                median_filter,
                data=spectral_map.data[idxs],
                by="map",
                flatten=True,
                parallel="threads",
                # actiual params for median_filter
                size=size,
                mode=boundary_mode,
            )
    elif method == "gaussian":
        spectral_map.data[idxs] = spectral_map.apply_func(
            gaussian_filter,
            data=spectral_map.data[idxs],
            by="map",
            flatten=True,
            parallel="threads",
            # actiual params for gaussian_filter
            sigma=sigma,
        )
    else:
        raise ValueError("Method not recognized")
