"""Spatial rotation transformation step."""

from typing import Literal

import numpy as np

from ramappy import const
from ramappy.core import SpectralMap
from ramappy.core.pipeline import (
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)


class StepMapRotationParams(StepParams):
    """Parameters for `rotate`."""

    degrees: Literal[-90, 90, 180, -180, 270, -270, 0, 360]
    """Rotation angle in multiples of 90 degrees."""


@pipeline_step(
    step_name="rotate",
    params_validator=SingleModelParamsValidator(StepMapRotationParams),
    friendly_name="Rotate",
    step_category=StepClass.PROCESSING,
    modifies_data=False,
    modifies_images=True,
)
def rotate(spectral_map: SpectralMap, degrees: Literal[-90, 90, 180, -180, 270, -270, 0, 360]) -> None:
    """Rotate map data, masks, and derived images.

    Parameters
    ----------
    spectral_map
        Input map modified in place.
    degrees
        Rotation angle in multiples of 90 degrees.
    """

    img_width, img_height = spectral_map.img_width, spectral_map.img_height
    cube = spectral_map.cube
    cube = np.rot90(cube, k=degrees // 90, axes=(0, 1))
    if degrees % 180 != 0:
        spectral_map.img_width, spectral_map.img_height = img_height, img_width
    spectral_map.data = cube.reshape(
        spectral_map.data.shape[const.Axis.PIXEL],
        spectral_map.data.shape[const.Axis.SPECTRAL],
    )
    spectral_map.spatial_grid = spectral_map.spatial_grid.rotate_90(degrees // 90)

    for mask_id, mask in spectral_map.masks.items():
        img = mask.get_2Dmask()
        img = np.rot90(img, k=degrees // 90)
        spectral_map.masks[mask_id].idxs = np.flatnonzero(img)
        if degrees % 180 != 0:
            spectral_map.masks[mask_id].set_mask_shape(*(mask.mask_shape[::-1]))

    for im_name, im in spectral_map.images.items():
        if hasattr(im, "data") and im.data is not None:
            spectral_map.images[im_name].data = np.rot90(im.data, k=degrees // 90)
        if hasattr(im, "_image") and im._image is not None:
            spectral_map.images[im_name]._image = im._image.rotate(degrees, expand=True)
