"""Spectral-axis cropping step."""

import numpy.typing as npt

from ramappy.core import SpectralMap, Spectrum
from ramappy.core.pipeline import (
    ParamsRequireRoIX,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)
from ramappy.utils import select_x_indices


class StepCropSpectralParams(StepParams, ParamsRequireRoIX):
    """Parameters for `crop_spectral`."""

    pass


@pipeline_step(
    step_name="crop_spectral",
    params_validator=SingleModelParamsValidator(StepCropSpectralParams),
    friendly_name="Crop spectral axis",
    step_category=StepClass.PROCESSING,
    modifies_spectral_axis=True,
)
def crop_spectral(spectral_map: SpectralMap | Spectrum, roi_x: npt.ArrayLike):
    """Truncate data to one or more spectral regions of interest.

    Parameters
    ----------
    spectral_map
        Map or spectrum container modified in place.
    roi_x
        Spectral region(s) as ``[start, end]`` or ``[[start1, end1], ...]``.

    Notes
    -----
    This operation is irreversible for the in-memory object.
    """
    spectral_map.roi_x = spectral_map.adapt_roi_x(roi_x)
    x_idx = select_x_indices(spectral_map.x, spectral_map.roi_x)
    spectral_map.x = spectral_map.x[x_idx]
    spectral_map.data = spectral_map.data[:, x_idx]
