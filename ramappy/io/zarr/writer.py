from __future__ import annotations

import contextlib
import warnings
import zipfile
from types import MethodType

import numpy as np
import zarr
from zarr.storage import LocalStore, ZipStore

import ramappy
from ramappy.core import SpectralMap
from ramappy.io.core import IOParams, SpectrumType, output_format

from ._project import sync_zarr_collection_attrs, update_zarr_project_attrs
from .common import get_cube_chunks, get_default_compressors
from .images import write_zarr_image
from .masks import write_zarr_mask
from .spectrum import write_spectrum


class ZarrFormatOutputParams(IOParams):
    """Parameters for writing Zarr files."""

    as_zip: bool = False
    """Whether to write the Zarr file as a zip archive. If `True`, the output will be a single .zip file containing the Zarr dataset. If `False`, the output will be a directory containing the Zarr dataset."""


def _dedup_zip_central_directory(zf: zipfile.ZipFile) -> None:
    """
    Remove duplicate entries from a ZipFile's central directory.

    This addresses the issue where writing the same key multiple times to a ZipStore
    (e.g., zarr.json during array resize operations) creates duplicate entries in
    the central directory. See: https://github.com/zarr-developers/zarr-python/issues/3580

    This function keeps only the most recent (last) entry for each filename by
    scanning the filelist in reverse and removing earlier duplicates from the
    central directory before the zip file is closed.

    Args:
        zf: The ZipFile to deduplicate (must be open in write mode).
    """
    try:
        filelist = getattr(zf, "filelist", None)
        if not isinstance(filelist, list) or len(filelist) < 2:
            return

        # Keep only the most recent (last) entry per filename while preserving
        # original order among retained records.
        seen_filenames: set[str] = set()
        indices_to_keep: set[int] = set()
        for idx in range(len(filelist) - 1, -1, -1):
            filename = filelist[idx].filename
            if filename not in seen_filenames:
                seen_filenames.add(filename)
                indices_to_keep.add(idx)

        if len(indices_to_keep) == len(filelist):
            return

        zf.filelist[:] = [info for idx, info in enumerate(filelist) if idx in indices_to_keep]  # type: ignore

        # Keep ZipFile's name index aligned with the deduplicated filelist.
        # This mirrors zipfile's "last write wins" behavior.
        zf.NameToInfo = {info.filename: info for info in zf.filelist}
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            (
                "Could not deduplicate ZipFile central directory entries; "
                "archive remains readable but may contain duplicate metadata entries. "
                f"Reason: {exc!r}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )


def _enable_zip_overwrite_semantics(zf: zipfile.ZipFile) -> None:
    """
    Make duplicate key writes overwrite central-directory metadata in-place.

    Zip files are append-only, so replacing an existing key still appends bytes,
    but by removing the prior directory entry before each write we avoid duplicate
    names in the final central directory and suppress duplicate-name warnings.
    """

    original_writecheck = getattr(zf, "_writecheck", None)
    if original_writecheck is None:
        return

    def _writecheck_overwrite(self: zipfile.ZipFile, zinfo: zipfile.ZipInfo):
        existing = self.NameToInfo.pop(zinfo.filename, None)
        if existing is not None:
            with contextlib.suppress(ValueError):
                self.filelist.remove(existing)
        return original_writecheck(zinfo)

    zf._writecheck = MethodType(_writecheck_overwrite, zf)  # type: ignore


@output_format(
    "zarr",
    friendly_name="Zarr",
    extensions={"zarr"},
    format_params_model=ZarrFormatOutputParams,
    supported_types=SpectrumType.SPECTRAL_MAP,
)
def write_zarr(spectral_map: SpectralMap, *, out_file, as_zip: bool = False):
    """Write SpectralMap to a zarr file, RamApp-style."""

    compressors = get_default_compressors()

    store: ZipStore | LocalStore
    if as_zip:
        store = ZipStore(out_file, mode="w")
        if hasattr(store, "_sync_open"):
            store._sync_open()
        if hasattr(store, "_zf"):
            _enable_zip_overwrite_semantics(store._zf)
    else:
        store = LocalStore(out_file)

    try:
        root = zarr.group(store=store)

        data_group = update_zarr_project_attrs(root, spectral_map)
        cube = np.asarray(spectral_map.cube)
        shape_intensities = cube.shape
        data_group_intensities = data_group.create_array(
            name="data",
            shape=shape_intensities,
            chunks=get_cube_chunks(shape_intensities, cube.dtype),
            dtype=cube.dtype,
            compressors=compressors,
        )
        data_group_intensities[:] = cube

        data_group_x = data_group.create_array(
            name="x", shape=spectral_map.x.shape, dtype=spectral_map.x.dtype, compressors=compressors
        )
        data_group_x[:] = spectral_map.x

        data_group.attrs.update(
            {
                "roi_x": spectral_map.roi_x.tolist(),
                "x_axis_unit": list(spectral_map.x_axis_unit.to_tuple())
                if spectral_map.x_axis_unit is not None
                else None,
                "data_unit": list(spectral_map.data_unit.to_tuple()) if spectral_map.data_unit is not None else None,
            }
        )

        zarr_images_group, masks, spectra = sync_zarr_collection_attrs(root, spectral_map)

        for k, im in spectral_map.images.items():
            cur_pos = zarr_images_group
            if im.parent_group is not None:
                cur_pos = zarr_images_group.require_group(im.parent_group)

            cur_image = cur_pos.require_group(k)
            write_zarr_image(cur_image, im, compressors=compressors)

        for k, m in spectral_map.masks.items():
            cur_pos = masks
            if m.parent_group is not None:
                cur_pos = masks.require_group(m.parent_group)

            cur_mask = cur_pos.require_group(k)
            write_zarr_mask(cur_mask, m, compressors=compressors)

        for k, sp in spectral_map.spectra.items():
            write_spectrum(spectra, k, sp, compressors=compressors)

        if as_zip and hasattr(store, "_zf"):
            _dedup_zip_central_directory(store._zf)  # type: ignore
            store._zf.comment = f"Created by RamApp {ramappy.__version__}".encode()  # type: ignore
    finally:
        store.close()
