from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr

from ._project import sync_zarr_collection_attrs, update_zarr_project_attrs
from .common import get_cube_chunks, get_default_compressors, to_jsonable
from .images import write_zarr_image
from .masks import write_zarr_mask
from .spectrum import write_spectrum

if TYPE_CHECKING:
    from ramappy.core.images2d import Image2D
    from ramappy.core.masks import Mask
    from ramappy.core.spectral_map import SpectralMap
    from ramappy.core.spectrum import Spectrum
    from ramappy.pipeline.pipeline import Pipeline


class ZarrProjectStore:
    """Incremental Zarr engine for RamApp projects.
    Keeps a Zarr group open and allows writing individual entities.
    """

    def __init__(self, path: Path | str, mode: str = "a"):
        self.path = Path(path)
        self.compressors = get_default_compressors()
        self.root = zarr.open_group(str(self.path), mode=mode)  # type: ignore

        # Ensure base groups exist
        self.data_group = self.root.require_group("data")
        self.masks_group = self.root.require_group("masks")
        self.images_group = self.root.require_group("images")
        self.spectra_group = self.root.require_group("spectra")

    @classmethod
    def open(cls, path: Path | str) -> ZarrProjectStore:
        return cls(path, mode="a")

    @staticmethod
    def _delete_entity_from_collection(collection, entity_id: str) -> None:
        if entity_id in collection:
            del collection[entity_id]

        for key in list(collection.keys()):
            child = collection[key]
            if getattr(child, "attrs", {}).get("group") and entity_id in child:
                del child[entity_id]

    @staticmethod
    def _entity_target_group(collection, entity_id: str, parent_group: str | None):
        if parent_group is None:
            return collection.require_group(entity_id)

        return collection.require_group(parent_group).require_group(entity_id)

    def write_intensities(self, intensities: np.ndarray):
        """Write the main spectral cube as ``data/data`` in ``(height, width, n_spectral)`` order."""
        cube = np.asarray(intensities)
        if cube.ndim != 3:
            raise ValueError(f"Expected cube-shaped intensities with 3 dimensions, got shape {cube.shape}")

        zarr_data = self.data_group.create_array(
            "data",
            shape=cube.shape,
            chunks=get_cube_chunks(cube.shape, cube.dtype),
            dtype=cube.dtype,
            compressors=self.compressors,
            overwrite=True,
        )
        zarr_data[:] = cube

    def write_x(self, x: np.ndarray):
        """Write the spectral axis."""
        spectral_axis = np.asarray(x)
        zarr_x = self.data_group.create_array(
            "x",
            shape=spectral_axis.shape,
            dtype=spectral_axis.dtype,
            compressors=self.compressors,
            overwrite=True,
        )
        zarr_x[:] = spectral_axis

    def write_mask(self, mask_id: str, mask: Mask):
        """Write a mask to the store."""
        self._delete_entity_from_collection(self.masks_group, mask_id)
        m_grp = self._entity_target_group(self.masks_group, mask_id, mask.parent_group)
        write_zarr_mask(m_grp, mask, compressors=self.compressors)

    def delete_mask(self, mask_id: str):
        self._delete_entity_from_collection(self.masks_group, mask_id)

    def write_image(self, image_id: str, image: Image2D):
        """Write an image to the store."""
        self._delete_entity_from_collection(self.images_group, image_id)
        i_grp = self._entity_target_group(self.images_group, image_id, image.parent_group)
        write_zarr_image(i_grp, image, compressors=self.compressors)

    def delete_image(self, image_id: str):
        self._delete_entity_from_collection(self.images_group, image_id)

    def write_spectrum(self, spectrum_id: str, sp: Spectrum):
        """Write an external spectrum to the store."""
        write_spectrum(self.spectra_group, spectrum_id, sp, compressors=self.compressors)

    def delete_spectrum(self, spectrum_id: str):
        if spectrum_id in self.spectra_group:
            del self.spectra_group[spectrum_id]

    def flush_attrs(self, *, pipeline: Pipeline | None, name: str, data: SpectralMap):
        """Flush global project attributes."""
        update_zarr_project_attrs(self.root, data, name=name)
        self.root.attrs["pipeline"] = to_jsonable(
            {
                "input_format": pipeline.input_format,
                "input_params": pipeline.input_params,
                "steps": [s.model_dump() for s in pipeline.steps],
            }
            if pipeline
            else {}
        )
        sync_zarr_collection_attrs(self.root, data)

    def get_intensities_memmap(self) -> zarr.Array:
        return self.data_group["data"]  # type: ignore
