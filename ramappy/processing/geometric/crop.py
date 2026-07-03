import numpy as np

from ramappy import const
from ramappy.core import SpectralMap
from ramappy.core.pipeline import (
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)


class StepCropSpatialParams(StepParams):
    """Parameters for `crop_spatial`."""

    row_start: int
    """Starting row index for cropping (inclusive)."""

    row_end: int
    """Ending row index for cropping (exclusive)."""

    col_start: int
    """Starting column index for cropping (inclusive)."""

    col_end: int
    """Ending column index for cropping (exclusive)."""

    crop_images: bool = True
    """Whether to crop all the project images. If False, images will not be cropped."""


@pipeline_step(
    step_name="crop_spatial",
    params_validator=SingleModelParamsValidator(StepCropSpatialParams),
    friendly_name="Crop map",
    step_category=StepClass.PROCESSING,
)
def crop_spatial(
    spectral_map: SpectralMap,
    *,
    row_start: int | None = None,
    row_end: int | None = None,
    col_start: int | None = None,
    col_end: int | None = None,
    crop_images: bool = True,
):
    """Crop the spectral map to the provided rectangle.

    Parameters
    ----------
    row_start, row_end, col_start, col_end: int or None, default=None
        The coordinates of the SW (`row_start`, `col_start`) and NE (`row_end`, `col_end`) vertices of
        the rectangle, relative to the origin (0,0) at the SW corner.
        If `*_start` is None, it will be considered as `0`; if `*_end` is None,
        it will be filled with either `img_height` or `img_width`.

    Notes
    -----
    This operation is **irreversible** - the original map data is modified in-place.
    """
    row_start = row_start if row_start is not None else 0
    col_start = col_start if col_start is not None else 0
    row_end = row_end if row_end is not None else spectral_map.img_height
    col_end = col_end if col_end is not None else spectral_map.img_width

    img_height_new = row_end - row_start
    img_width_new = col_end - col_start

    for mask_id, mask in spectral_map.masks.items():
        y, x = mask.unravel_index()
        if len(y) == 0:
            continue
        x -= col_start
        y -= row_start
        keep_idxs = np.argwhere((x >= 0) * (x < img_width_new) * (y >= 0) * (y < img_height_new)).ravel()
        idxs = np.ravel_multi_index([y[keep_idxs], x[keep_idxs]], (img_height_new, img_width_new))
        spectral_map.masks[mask_id].idxs = idxs
        spectral_map.masks[mask_id].set_mask_shape(img_height_new, img_width_new)

    for im_name, im in spectral_map.images.items():
        if hasattr(im, "data") and im.data is not None:
            spectral_map.images[im_name].data = im.data[row_start:row_end, col_start:col_end]
        elif crop_images and im._image is not None:
            pil_img = im._image  # im._image is not None here
            pil_w, pil_h = pil_img.size  # PIL: (width, height)
            if (pil_h, pil_w) == spectral_map.map_shape:
                spectral_map.images[im_name]._image = im._image.crop(
                    (
                        col_start,
                        row_start,
                        col_end,
                        row_end,
                    )
                )
            else:
                res_x = pil_w / spectral_map.img_width  # horizontal (column) scaling
                res_y = pil_h / spectral_map.img_height  # vertical (row) scaling
                left = round(col_start * res_x)
                right = round(col_end * res_x)
                top = round(row_start * res_y)
                bottom = round(row_end * res_y)

                spectral_map.images[im_name]._image = im._image.crop((left, top, right, bottom))

    # crop aux_data (if any)
    for aux_name, aux_data in spectral_map.aux_data.items():
        cube = aux_data.reshape(spectral_map.img_height, spectral_map.img_width, aux_data.shape[const.Axis.SPECTRAL])
        cube = cube[row_start:row_end, col_start:col_end, :]
        spectral_map.aux_data[aux_name] = cube.reshape(img_width_new * img_height_new, spectral_map.x_size)

    cube = spectral_map.cube
    cube = cube[row_start:row_end, col_start:col_end, :]
    spectral_map.data = cube.reshape(img_width_new * img_height_new, spectral_map.x_size)

    spectral_map.img_height = img_height_new
    spectral_map.img_width = img_width_new

    origin_y, origin_x = spectral_map.spatial_grid.pixel_to_physical(row_start, col_start)
    spectral_map.spatial_grid.origin_y = origin_y
    spectral_map.spatial_grid.origin_x = origin_x

    # Mask '0' must always have one and only one pixel
    if len(spectral_map.masks["0"].idxs) == 0:
        spectral_map.masks["0"] = spectral_map.generate_random_selection()
