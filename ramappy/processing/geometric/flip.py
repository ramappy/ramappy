"""Spatial flip transformation step."""

import numpy as np
import PIL.ImageOps

from ramappy import const
from ramappy.core import SpectralMap
from ramappy.core.pipeline import (
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)


class StepMapFlipParams(StepParams):
    """Parameters for `flip`."""

    axis_to_flip: const.SpatialAxis
    """Spatial axis to flip (`vertical` or `horizontal`)."""


@pipeline_step(
    step_name="flip",
    params_validator=SingleModelParamsValidator(StepMapFlipParams),
    friendly_name="Flip map",
    step_category=StepClass.PROCESSING,
    modifies_data=False,
    modifies_images=True,
)
def flip(spectral_map: SpectralMap, axis_to_flip: const.SpatialAxis = const.SpatialAxis.VERTICAL):
    """Flip map data, masks, and images across one spatial axis.

    Parameters
    ----------
    spectral_map
        Input map modified in place.
    axis_to_flip
        Spatial axis to flip (`vertical` or `horizontal`).
    """

    cube = spectral_map.cube
    cube = np.flip(cube, axis=axis_to_flip)
    spectral_map.data = cube.reshape(
        spectral_map.data.shape[const.Axis.PIXEL], spectral_map.data.shape[const.Axis.SPECTRAL]
    )

    for mask_id, mask in spectral_map.masks.items():
        img = mask.get_2Dmask()
        img = np.flip(img, axis=axis_to_flip)
        spectral_map.masks[mask_id].idxs = np.flatnonzero(img)

    for im_name, im in spectral_map.images.items():
        if hasattr(im, "data") and im.data is not None:
            spectral_map.images[im_name].data = np.flip(im.data, axis=axis_to_flip)

        pil_img = im._image
        if pil_img is not None:
            if axis_to_flip == const.SpatialAxis.VERTICAL:
                spectral_map.images[im_name]._image = PIL.ImageOps.flip(pil_img)
            elif axis_to_flip == const.SpatialAxis.HORIZONTAL:
                spectral_map.images[im_name]._image = PIL.ImageOps.mirror(pil_img)
