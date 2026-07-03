from __future__ import annotations

import numpy as np

from ramappy.core.images2d import Image2DRenderer
from ramappy.core.masks import Mask

from .common import decode_dict, to_jsonable
from .spectrum import read_spectrum, write_spectrum


def write_zarr_mask(zgroup, mask: Mask, *, compressors) -> None:
    if "idxs" in zgroup:
        del zgroup["idxs"]

    mask_2d = mask.get_2Dmask()
    cur_mask_mask = zgroup.create_array(
        shape=mask_2d.shape,
        dtype=mask_2d.dtype,
        name="mask",
        overwrite=True,
        compressors=compressors,
    )
    cur_mask_mask[:] = mask_2d

    mask_image = np.array(Image2DRenderer.render(mask.to_image()))
    cur_mask_img = zgroup.create_array(
        shape=mask_image.shape,
        dtype=mask_image.dtype,
        name="img",
        overwrite=True,
        compressors=compressors,
    )
    cur_mask_img[:] = mask_image
    cur_mask_img.attrs.update(
        {
            "color": mask.color,
            "visible": mask.visible,
        }
    )

    if mask.spectrum is not None:
        write_spectrum(zgroup, "spectrum", mask.spectrum, compressors=compressors)
    elif "spectrum" in zgroup:
        del zgroup["spectrum"]

    zgroup.attrs.update(
        {
            "name": mask.name,
            "editable": mask.editable,
            "spectrum_agg": mask.spectrum_agg,
            "single_mask": True,
            "source_step_id": mask.source_step_id,
            "visible": mask.visible,
            "color": mask.color,
            "metadata": to_jsonable(mask.mask_metadata.model_dump()),
        }
    )


def read_zarr_mask(zgroup) -> Mask:
    mask_arr = zgroup["mask"]
    idxs = np.argwhere(mask_arr[:].flatten()).flatten()

    spectrum = None
    if "spectrum" in zgroup:
        spectrum = read_spectrum(zgroup["spectrum"])

    return Mask(
        idxs=idxs,
        name=zgroup.attrs["name"],
        img_shape=mask_arr.shape,
        spectrum=spectrum,
        spectrum_agg=zgroup.attrs.get("spectrum_agg", "mean"),
        color=zgroup.attrs.get("color"),
        visible=zgroup.attrs.get("visible", True),
        editable=zgroup.attrs.get("editable", True),
        source_step_id=zgroup.attrs.get("source_step_id"),
        metadata=decode_dict(zgroup.attrs.get("metadata", {})),
    )
