from __future__ import annotations

import os
from typing import Any

import numpy as np
import zarr
from PIL import Image as PILImage
from zarr.storage import LocalStore, ZipStore

from ramappy.core import SpectralMap
from ramappy.core.images2d import Image2D, Image2DGroup
from ramappy.core.masks import Mask, MaskGroup
from ramappy.core.spectrum import Spectrum
from ramappy.io.core import IOParams, input_format

from .common import decode_dict, iter_members, parse_xunit, read_attr_array
from .images import read_zarr_image
from .masks import read_zarr_mask
from .spectrum import read_spectrum


class ZarrFormatInputParams(IOParams):
    """Parameters for reading Zarr files."""

    pass


def _align_legacy_layer_2d(data: np.ndarray, *, target_shape: tuple[int, int]) -> np.ndarray:
    """Align a legacy 2-D layer to the target map shape, transposing when needed."""

    if data.shape == target_shape:
        return data

    transposed = np.transpose(data)
    if transposed.shape == target_shape:
        return transposed

    return data


def _align_legacy_mask(mask: Mask, *, target_shape: tuple[int, int]) -> None:
    """Align a legacy v1 mask to the target map shape in-place."""

    aligned_mask = _align_legacy_layer_2d(mask.get_2Dmask(), target_shape=target_shape)
    mask.set_mask_shape(*aligned_mask.shape)
    mask.idxs = np.flatnonzero(np.asarray(aligned_mask, dtype=bool).ravel())


def _align_legacy_image(image: Image2D, *, target_shape: tuple[int, int]) -> None:
    """Align a legacy v1 image layer to the target map shape in-place."""

    if image.data is not None:
        image.data = _align_legacy_layer_2d(image.data, target_shape=target_shape)
        return

    if image._image is not None:
        raster = np.asarray(image._image)
        if raster.ndim >= 2 and (
            raster.shape[:2] != target_shape
            and np.transpose(raster, (1, 0, *range(2, raster.ndim))).shape[:2] == target_shape
        ):
            image._image = PILImage.fromarray(np.transpose(raster, (1, 0, *range(2, raster.ndim))))


@input_format(
    "zarr",
    friendly_name="Zarr (RamApp)",
    extensions={"zarr"},
    has_custom_params=False,
    format_params_model=ZarrFormatInputParams,
)
def read_zarr(filepath_or_buffer):
    if os.path.isdir(filepath_or_buffer):
        store = LocalStore(filepath_or_buffer)
    else:
        store = ZipStore(filepath_or_buffer, mode="r")

    try:
        root: Any = zarr.open(store=store, mode="r")

        images: dict[str, Image2D] = {}
        image_groups: dict[str, Image2DGroup] = {}
        for k, im in iter_members(root["images"]):
            if "group" in im.attrs:
                for i, g in iter_members(im):
                    images[i] = read_zarr_image(g)
                    images[i].parent_group = k
                image_groups[k] = Image2DGroup(
                    name=im.attrs.get("name", k),
                    visible=im.attrs.get("visible", True),
                    elem_ids=im.attrs.get("key_order", []),
                )
            else:
                images[k] = read_zarr_image(im)

        masks: dict[str, Mask] = {}
        mask_groups: dict[str, MaskGroup] = {}
        for k, m in iter_members(root["masks"]):
            if "group" in m.attrs:
                for i, g in iter_members(m):
                    masks[i] = read_zarr_mask(g)
                    masks[i].parent_group = k
                mask_groups[k] = MaskGroup(
                    name=m.attrs.get("name", k),
                    visible=m.attrs.get("visible", True),
                    color=m.attrs.get("cmap", "#ff0000"),
                    spectrum_agg=m.attrs.get("mask_spectrum_aggregation", "mean"),
                    elem_ids=m.attrs.get("key_order", []),
                )
            else:
                masks[k] = read_zarr_mask(m)

        spectra: dict[str, Spectrum] = {}
        if "spectra" in root:
            for k, sp_group in iter_members(root["spectra"]):
                spectra[k] = read_spectrum(sp_group)

        data = root["data/data"]
        format_version = int(root.attrs.get("format_version", 1))

        if data.ndim != 3:
            raise ValueError(f"Invalid Zarr data shape {data.shape}: expected 3-D (height, width, n_spectral)")

        if format_version >= 2:
            img_height, img_width, n_spectral = data.shape
        else:
            # legacy v1 stored as (width, height, spectral)
            img_width, img_height, n_spectral = data.shape
        cube = np.ascontiguousarray(data[:])

        if format_version < 2:
            cube = np.ascontiguousarray(np.transpose(cube, (1, 0, 2)))

        if root["data/x"].shape[0] != n_spectral:
            raise ValueError(
                "Invalid Zarr spectral axis length: "
                f"x has length {root['data/x'].shape[0]} while data has {n_spectral} spectral channels"
            )

        history = decode_dict(root.attrs.get("history", {}))

        metadata = decode_dict(root.attrs.get("metadata", "{}"))

        # Clean up transient backend/UI state accidentally saved in older versions
        exclude_keys = ["live_update"]
        if isinstance(metadata, dict):
            metadata = {k: v for k, v in metadata.items() if k not in exclude_keys}

        spatial_grid = decode_dict(root.attrs.get("spatial_grid", {}))

        spectral_map = SpectralMap(
            metadata=metadata,
            x=root["data/x"][:],
            x_axis_unit=parse_xunit(root["data"].attrs.get("x_axis_unit", root["data"].attrs.get("x_unit"))),
            data_unit=root["data"].attrs.get("data_unit", root["data"].attrs.get("y_unit")),
            roi_x=read_attr_array(root["data"], "roi_x"),
            data=cube.reshape(img_height * img_width, n_spectral),
            ignore_sort=True,
            name=root.attrs.get("name"),
            img_width=img_width,
            img_height=img_height,
            spatial_grid=spatial_grid,
            masks=masks,
            masks_group=mask_groups,
            images=images,
            images_group=image_groups,
            spectra=spectra,
            history=history,
        )

        if format_version < 2:
            target_shape = spectral_map.map_shape
            for mask in spectral_map.masks.values():
                _align_legacy_mask(mask, target_shape=target_shape)
            for image in spectral_map.images.values():
                _align_legacy_image(image, target_shape=target_shape)

        spectral_map.images.ordered_keys = root["images"].attrs.get("key_order", list(spectral_map.images.keys()))
        spectral_map.masks.ordered_keys = root["masks"].attrs.get("key_order", list(spectral_map.masks.keys()))
        if "groups_order" in root["images"].attrs:
            spectral_map.images_group.ordered_keys = root["images"].attrs["groups_order"]
        if "groups_order" in root["masks"].attrs:
            spectral_map.masks_group.ordered_keys = root["masks"].attrs["groups_order"]

        return spectral_map
    finally:
        store.close()


def read_zarr_metadata_only(filepath_or_buffer):
    if os.path.isdir(filepath_or_buffer):
        store = LocalStore(filepath_or_buffer)
    else:
        store = ZipStore(filepath_or_buffer, mode="r")
    try:
        root: Any = zarr.open(store=store, mode="r")

        data = root["data/data"]
        format_version = int(root.attrs.get("format_version", 1))
        if data.ndim != 3:
            raise ValueError(f"Invalid Zarr data shape {data.shape}: expected 3-D")

        if format_version >= 2:
            img_height, img_width, spectral_size = data.shape
        else:
            # legacy v1 stored as (width, height, spectral)
            img_width, img_height, spectral_size = data.shape

        metadata = {
            "name": decode_dict(root.attrs.get("name")),
            "created": decode_dict(root.attrs.get("created")),
            "img_width": img_width,
            "img_height": img_height,
            "spectral_size": spectral_size,
            "thumbnail": decode_dict(root.attrs.get("thumbnail")),
            "format_version": format_version,
        }

        return metadata
    finally:
        store.close()
