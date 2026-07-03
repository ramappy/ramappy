from __future__ import annotations

import numpy as np
from PIL import Image

from ramappy.core.images2d import Image2D

from .common import decode_dict, to_jsonable
from .spectrum import read_spectrum, write_spectrum


def write_zarr_image(zgroup, image: Image2D, *, compressors) -> None:
    data = image.data
    if data is not None:
        if "img" in zgroup:
            del zgroup["img"]
        if np.ma.isMaskedArray(data):
            data_to_store = np.ma.filled(data, np.nan)
            zgroup.attrs["masked_data"] = True
        else:
            data_to_store = data
            zgroup.attrs["masked_data"] = False

        zgroup_data = zgroup.create_array(
            name="data",
            shape=data_to_store.shape,
            dtype=data_to_store.dtype,
            compressors=compressors,
            fill_value=np.nan,
            overwrite=True,
        )
        zgroup_data[:] = data_to_store
    else:
        if "data" in zgroup:
            del zgroup["data"]
        img = np.array(image._image)
        zgroup_img = zgroup.create_array(
            name="img",
            shape=img.shape,
            dtype=img.dtype,
            compressors=compressors,
            overwrite=True,
        )
        zgroup_img[:] = img

    if image.spectrum is not None:
        write_spectrum(zgroup, "spectrum", image.spectrum, compressors=compressors)
    elif "spectrum" in zgroup:
        del zgroup["spectrum"]

    zgroup.attrs.update(
        {
            "name": image.name,
            "visible": image.visible,
            "locked": image.locked,
            "single_image": True,
            "source_step_id": image.source_step_id,
            "viz_rules": image.viz_rules.model_dump() if image.viz_rules is not None else None,
            "data_rules": image.data_rules.model_dump() if image.data_rules is not None else None,
            "metadata": to_jsonable(image.image_metadata.model_dump()),
        }
    )


def read_zarr_image(zgroup) -> Image2D:
    data_arr = zgroup.get("data")
    img = None
    data = None

    if data_arr is not None:
        raw = data_arr[:]
        if zgroup.attrs.get("masked_data", False):
            data = np.ma.masked_array(raw, mask=np.isnan(raw), fill_value=np.nan)
        else:
            data = raw
    else:
        # If we don't have data, read the stored RGBA image.
        img_arr = zgroup.get("img")
        if img_arr is not None:
            img = Image.fromarray(img_arr[:])

    spectrum = None
    if "spectrum" in zgroup:
        spectrum = read_spectrum(zgroup["spectrum"])

    return Image2D(
        data=data,
        image=img,
        name=zgroup.attrs["name"],
        visible=zgroup.attrs["visible"],
        locked=zgroup.attrs["locked"],
        source_step_id=zgroup.attrs.get("source_step_id"),
        viz_rules=zgroup.attrs.get("viz_rules"),
        data_rules=zgroup.attrs.get("data_rules"),
        spectrum=spectrum,
        metadata=decode_dict(zgroup.attrs.get("metadata", {})),
    )
