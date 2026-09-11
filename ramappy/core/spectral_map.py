"""Core spectral map dataset container.

Public API
----------
- :class:`SpectralMap <ramappy.core.spectral_map.SpectralMap>`: the single entry point for all spectral-map data.
- :meth:`SpectralMap.from_cube <ramappy.core.spectral_map.SpectralMap.from_cube>`: convenience constructor from a 3-D cube.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ramappy import const, units
from ramappy.core._spectral_map import _SpectralMapFunctionalMixin, _SpectralMapImagesMixin, _SpectralMapMasksMixin
from ramappy.core.collections import OrderedEntityMap
from ramappy.core.images2d import Image2D, Image2DGroup, SpatialGrid
from ramappy.core.masks import Mask, MaskGroup
from ramappy.core.pipeline import (
    HistoryLog,
    SpectralAxis,
    SpectralData,
)
from ramappy.core.spectrum import SpectralMapMetadata, Spectrum
from ramappy.utils import (
    aggregate,
    generate_key,
    generate_random_pixel,
    select_x_indices,
)


class SpectralMap(_SpectralMapFunctionalMixin, _SpectralMapImagesMixin, _SpectralMapMasksMixin, Spectrum):
    """Core hyperspectral dataset container.

    A :class:`SpectralMap <ramappy.core.spectral_map.SpectralMap>` wraps a 2-D flattened spectral data array
    ``(n_pixels, n_spectral)`` together with a spectral axis, spatial
    grid, masks, 2-D images, and a processing history.

    Tip:
        Prefer :meth:`SpectralMap.from_cube <ramappy.core.spectral_map.SpectralMap.from_cube>` when constructing from a 3-D array
        ``(height, width, n_spectral)``.

    Parameters
    ----------
    x : SpectralAxis
        Spectral axis (e.g., Raman shift in cm⁻¹). Sorted ascending.
    data : SpectralData
        Flat data array, shape ``(img_height * img_width, n_spectral)``.
    img_width : int
        Number of pixels along the horizontal axis.
    img_height : int
        Number of pixels along the vertical axis.
    spatial_grid : SpatialGrid | None
        Physical spatial grid (pixel size, origin, unit).
        If ``None``, defaults to a 1 px/px isotropic grid.
    x_axis_unit : type_spectral_quantityunit
        Unit of the spectral axis.
    data_unit : type_data_quantityunit
        Unit of the spectral data.
    metadata : SpectralMapMetadata | dict | None
        Dataset-level metadata. Unknown dict keys are stored in ``extra``.
    roi_x : array_like | None
        Spectral regions of interest, shape ``(n_regions, 2)``.
    name : str | None
        Human-readable label (falls back to ``metadata.original_filename``).
    aux_data : dict-like | None
        Auxiliary data arrays sharing the same shape as *data*.
    spectra : dict-like | None
        External reference spectra.
    masks : dict | None
        Initial masks keyed by string ID.
    masks_group : dict | None
        Initial mask groups.
    images : dict | None
        Initial 2-D images.
    images_group : dict | None
        Initial image groups.
    history : HistoryLog | dict | None
        Processing history.
    ignore_sort : bool
        Skip sorting the spectral axis.
    dtype : dtype-like
        dtype for spectral data storage (default ``float32``).

    """

    def __init__(
        self,
        *,
        x: SpectralAxis,
        data: SpectralData,
        img_width: int,
        img_height: int,
        roi_x: npt.ArrayLike | None = None,
        name: str | None = None,
        spatial_grid: SpatialGrid | None = None,
        x_axis_unit: units.type_spectral_quantityunit = units.DEFAULT_SPECTRAL_QUANTITY_UNIT,
        data_unit: units.type_data_quantityunit = units.DEFAULT_DATA_QUANTITY_UNIT,
        metadata: SpectralMapMetadata | dict[str, Any] | None = None,
        aux_data: OrderedEntityMap | dict[str, SpectralData] | None = None,
        spectra: OrderedEntityMap | dict[str, Spectrum] | None = None,
        masks: dict[str, Mask] | None = None,
        masks_group: dict[str, MaskGroup] | None = None,
        images: dict[str, Image2D] | None = None,
        images_group: dict[str, Image2DGroup] | None = None,
        history: HistoryLog | None = None,
        ignore_sort: bool = False,
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        # resolve metadata
        smap_metadata = SpectralMapMetadata.from_any(metadata)
        resolved_name = name or smap_metadata.original_filename or "Map"
        img_width = int(img_width)
        img_height = int(img_height)

        # delegate to Spectrum (and mixin) __init__
        super().__init__(
            x=x,
            data=data,
            x_axis_unit=x_axis_unit,
            data_unit=data_unit,
            roi_x=roi_x,
            name=resolved_name,
            metadata=smap_metadata.extra,  # keep extra dict on Spectrum.metadata
            aux_data=aux_data,
            history=history,
            ignore_sort=ignore_sort,
            dtype=dtype,
        )

        # store the typed metadata on SpectralMap itself
        self._smap_metadata: SpectralMapMetadata = smap_metadata

        # validate pixel count
        expected_pixels = img_width * img_height
        actual_pixels = np.asarray(self.data).shape[const.Axis.PIXEL]
        if actual_pixels != expected_pixels:
            raise ValueError(
                "Data pixel count does not match the provided image dimensions: "
                f"expected {expected_pixels}, got {actual_pixels}"
            )

        self.img_height: int = img_height
        self.img_width: int = img_width

        # spatial grid
        if spatial_grid is None:
            self.spatial_grid = SpatialGrid()
        elif isinstance(spatial_grid, SpatialGrid):
            self.spatial_grid = spatial_grid
        else:
            self.spatial_grid = SpatialGrid.model_validate(spatial_grid)

        # entity maps
        self.masks = OrderedEntityMap(masks)
        self.spectra = OrderedEntityMap(dict(spectra) if isinstance(spectra, OrderedEntityMap) else spectra)

        # key '0' is the undeletable default pixel selection (i.e., the cursor)
        if "0" not in self.masks or len(self.masks["0"].idxs) == 0:
            with self.masks.begin_transaction():
                self.masks["0"] = self.generate_random_selection()

        self.masks_group = OrderedEntityMap(masks_group)
        self.images = OrderedEntityMap(images)
        self.images_group = OrderedEntityMap(images_group)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.name}, "
            f"{self.img_width}x{self.img_height}x{self.n_spectral}, "
            f"ROI={self.roi_x!r}, "
            f"num_masks={len(self.masks)}, num_images={len(self.images)}, "
            # f"spatial_grid={self.spatial_grid!r}, "
            f"x_axis_unit='{self.x_axis_unit}', data_unit='{self.data_unit}')"
        )

    # class methods / alternate constructors

    @classmethod
    def from_cube(
        cls,
        cube: np.ndarray,
        spectral_axis: np.ndarray,
        *,
        spatial_grid: SpatialGrid | None = None,
        x_axis_unit: units.type_spectral_quantityunit = units.DEFAULT_SPECTRAL_QUANTITY_UNIT,
        data_unit: units.type_data_quantityunit = units.DEFAULT_DATA_QUANTITY_UNIT,
        metadata: SpectralMapMetadata | dict[str, Any] | None = None,
        name: str | None = None,
        x: np.ndarray | None = None,
        **kwargs,
    ) -> SpectralMap:
        """Construct a :class:`SpectralMap <ramappy.core.spectral_map.SpectralMap>` from a 3-D cube.

        Parameters
        ----------
        cube : np.ndarray, shape (height, width, n_spectral)
            Spectral cube in row-major order (height first).
        spectral_axis : np.ndarray, shape (n_spectral,)
            Spectral axis values.
        spatial_grid : SpatialGrid | None
            Physical spatial grid.
        x_axis_unit : type_spectral_quantityunit
            Spectral axis unit.
        data_unit : type_data_quantityunit
            Spectral data axis unit.
        metadata : SpectralMapMetadata | dict | None
            Dataset-level metadata.
        name : str | None
            Human-readable label.
        **kwargs
            Additional keyword arguments forwarded to :class:`SpectralMap <ramappy.core.spectral_map.SpectralMap>`.

        Returns
        -------
        SpectralMap
        """
        if x is not None:
            if spectral_axis is not None and not np.array_equal(np.asarray(spectral_axis), np.asarray(x)):
                raise ValueError("Provide either 'spectral_axis' or 'x'; if both are provided they must match.")
            spectral_axis = x

        if cube.ndim != 3:
            raise ValueError(f"Expected a 3-D cube (height, width, n_spectral), got shape {cube.shape}")
        spectral_axis_arr = np.asarray(spectral_axis)
        if spectral_axis_arr.ndim != 1:
            raise ValueError(f"Expected a 1-D spectral axis, got shape {spectral_axis_arr.shape}")

        height, width, n_spectral = cube.shape
        if spectral_axis_arr.size != n_spectral:
            raise ValueError(
                f"Spectral axis length does not match cube spectral dimension: {spectral_axis_arr.size} != {n_spectral}"
            )

        cube_arr = np.asarray(cube)
        if not np.isfinite(cube_arr).all():
            raise ValueError("Cube contains non-finite values.")

        return cls(
            x=spectral_axis_arr,
            data=cls._flatten_cube(cube_arr),
            img_height=height,
            img_width=width,
            spatial_grid=spatial_grid,
            x_axis_unit=x_axis_unit,
            data_unit=data_unit,
            metadata=metadata,
            name=name,
            **kwargs,
        )

    @staticmethod
    def _flatten_cube(cube: np.ndarray) -> np.ndarray:
        """Return a 2-D ``(n_pixels, n_spectral)`` view from a ``(h, w, n)`` cube."""
        height, width, n_spectral = cube.shape
        return cube.reshape(height * width, n_spectral)

    # dataset metadata

    @property
    def smap_metadata(self) -> SpectralMapMetadata:
        """Structured :class:`SpectralMapMetadata <ramappy.core.spectrum.SpectralMapMetadata>` for this dataset."""
        return self._smap_metadata

    @smap_metadata.setter
    def smap_metadata(self, value: SpectralMapMetadata | dict[str, Any] | None) -> None:
        self._smap_metadata = SpectralMapMetadata.from_any(value)

    # spatial properties

    @property
    def cube(self) -> np.ndarray:
        """Zero-copy 3-D view of data, shape ``(height, width, n_spectral)``.

        Uses `numpy.ndarray.reshape` which never copies data for
        C-contiguous arrays.

        Returns
        -------
        np.ndarray, shape (img_height, img_width, n_spectral)
        """
        return np.asarray(self.data).reshape(self.img_height, self.img_width, self.n_spectral)

    # dimension / axis properties

    @property
    def x_size(self) -> int:
        """Number of spectral channels (alias for `n_spectral`)."""
        return self.x.size

    @property
    def n_spectral(self) -> int:
        """Number of spectral channels."""
        return int(self.x.size)

    @property
    def n_pixels(self) -> int:
        """Total number of spatial pixels (``img_height * img_width``)."""
        return self.img_height * self.img_width

    @property
    def map_shape(self) -> tuple[int, int]:
        """Spatial shape ``(height, width)``."""
        return (self.img_height, self.img_width)

    @map_shape.setter
    def map_shape(self, shape: tuple[int, int]) -> None:
        img_height, img_width = shape
        if img_height * img_width == self.img_height * self.img_width:
            self.img_height, self.img_width = img_height, img_width
        else:
            from ramappy.processing.geometric.crop import crop_spatial

            warnings.warn(
                f"New shape {shape} has different number of pixels than current shape "
                f"{(self.img_height, self.img_width)}. Performing crop (top-left aligned).",
                stacklevel=2,
            )
            crop_spatial(self, row_end=img_height, col_end=img_width)

    def apply_func(
        self,
        f: Callable,
        *,
        data: SpectralData | None = None,
        preserve_input_dtype: bool = True,
        by: Literal["pixel", "wavenumber", "map"] = "pixel",
        flatten: bool = True,
        parallel: bool | Literal["threads", "processes"] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply a function to SpectralMap data with optional parallelization."""
        return super().apply_func(
            f,
            data=data,
            preserve_input_dtype=preserve_input_dtype,
            by=by,
            flatten=flatten,
            parallel=parallel,
            **kwargs,
        )

    # index helpers

    def get_indices(  # type: ignore
        self,
        mask: str | Mask | None = None,
        roi_x: npt.ArrayLike | None = None,
        ignore_empty_mask: bool = False,
    ) -> tuple[np.ndarray | slice, np.ndarray | slice, np.ndarray | slice]:
        """Return index tuples for slicing the spectral data array.

        Parameters
        ----------
        mask : str | Mask | None
            ID of a registered mask, a :class:`Mask <ramappy.core.masks.Mask>` object,
            or ``None`` for the full map.
        roi_x : array_like | None
            Spectral region(s) of interest.
        ignore_empty_mask : bool
            If ``True``, return an empty selection rather than raising for an
            empty mask.

        Returns
        -------
        tuple
            ``(idxs, x_idx, pixel_idx)`` ready for ``data[idxs]``.
        """
        if isinstance(mask, str):
            mask = self.masks.get(mask)

        if mask is None:
            pixel_idx = slice(None)
        else:
            if mask.is_empty() and not ignore_empty_mask:
                raise ValueError(f"Mask {mask} is empty")
            pixel_idx = mask.indices

        x_idx = select_x_indices(self.x, roi_x)  # type: ignore

        pixel_is_scalar = (not isinstance(pixel_idx, slice)) and (getattr(pixel_idx, "ndim", 0) == 0)
        x_is_scalar = (not isinstance(x_idx, slice)) and (getattr(x_idx, "ndim", 0) == 0)

        if isinstance(pixel_idx, slice) or isinstance(x_idx, slice) or pixel_is_scalar or x_is_scalar:
            idxs = (pixel_idx, x_idx)
        else:
            idxs = np.ix_(pixel_idx, x_idx)
        return idxs, x_idx, pixel_idx  # type: ignore

    # spectrum extraction

    def generate_random_selection(self) -> Mask:
        """Return a new :class:`Mask <ramappy.core.masks.Mask>` with one random pixel."""
        return self.new_mask([generate_random_pixel(self.img_width, self.img_height)], editable=False)

    def get_spectrum(
        self,
        agg: Literal["mean", "median", "mean95", "std", "max", "min", "p5", "p95", "p5p95", "p25", "p75", "p25p75"]
        | None = "mean",
        prefer_stored_property: bool = True,
        mask: str | Mask | None = None,
        roi_x: npt.ArrayLike | None = None,
        aux_data: str | None = None,
        ignore_empty_mask: bool = False,
    ) -> Spectrum:
        """Extract a :class:`Spectrum <ramappy.core.spectrum.Spectrum>` from the cube.

        Parameters
        ----------
        agg : str | None
            Aggregation method over selected pixels.
        prefer_stored_property : bool
            Return the pre-computed spectrum stored on the mask when available.
        mask : str | Mask | None
            Pixel selection. ``None`` is the full map.
        roi_x : array_like | None
            Spectral region of interest.
        aux_data : str | None
            Key of an `aux_data` entry to use instead of data.
        ignore_empty_mask : bool
            Return NaN spectrum instead of raising for an empty mask.

        Returns
        -------
        Spectrum
        """
        color = None
        name = None
        mask_id = mask
        if mask_id is not None:
            mask = self.masks[mask_id] if isinstance(mask_id, str) else mask_id
            name = mask.name if mask.name is not None else str(mask_id)
            color = mask.color if mask.color is not None else None

            if prefer_stored_property and mask.spectrum is not None:
                return mask.spectrum

        idxs, roi_idx, px_idx = self.get_indices(mask, roi_x, ignore_empty_mask=ignore_empty_mask)
        roi_x = self.adapt_roi_x(roi_x) if roi_x is not None else None

        if aux_data is not None:
            selected_data = self.aux_data.get(aux_data)[idxs]  # type: ignore
        elif ignore_empty_mask and (
            (isinstance(px_idx, np.ndarray) and len(px_idx) == 0) or (isinstance(px_idx, list) and len(px_idx) == 0)
        ):
            agg = None
            selected_data = np.full(
                len(roi_idx) if not isinstance(roi_idx, slice) else len(self.x),
                np.nan,
                dtype=np.asarray(self.data).dtype,
            )
        else:
            selected_data = np.asarray(self.data)[idxs]

        selected_data = aggregate(selected_data, agg=agg)

        return Spectrum(
            x=self.x[roi_idx],
            roi_x=roi_x,
            x_axis_unit=self.x_axis_unit,
            data_unit=self.data_unit,
            name=name,
            data=selected_data,
            ignore_sort=True,
            color=color,
        )

    def get_reference_spectrum(
        self,
        ref_id: str,
        agg_method: str | None = "mean",
        roi_x: npt.ArrayLike | None = None,
    ) -> tuple[Spectrum, bool]:
        """Look up a reference spectrum by ID.

        Searches, in order: image-attached spectra, masks, external spectra.

        Parameters
        ----------
        ref_id : str
            Identifier of the desired spectrum.
        agg_method : str | None
            Aggregation method applied if *ref_id* refers to a mask.
        roi_x : array_like | None
            Spectral ROI passed through to mask aggregation.

        Returns
        -------
        tuple[Spectrum, bool]
            ``(spectrum, is_external)`` where *is_external* is ``True`` when
            the spectrum came from `spectra`.
        """
        if ref_id in self.images and self.images[ref_id].spectrum is not None:
            spectrum = self.images[ref_id].spectrum
            assert spectrum is not None
            return spectrum, False

        if ref_id in self.masks:
            return (
                self.get_spectrum(
                    agg=agg_method,  # type: ignore
                    prefer_stored_property=(agg_method == "stored_property"),
                    mask=ref_id,
                    roi_x=roi_x,
                ),
                False,
            )
        if ref_id in self.spectra:
            return self.get_spectra(ref_id), True
        raise KeyError(f"Selected id {ref_id} not found")

    # external spectra management

    def add_spectra(self, spectra: Spectrum, key: str | None = None) -> str:
        """Add an external spectrum and return its key.

        Parameters
        ----------
        spectra : Spectrum
            The spectrum to add.
        key : str | None
            Storage key. Auto-generated when ``None``.

        Returns
        -------
        str
            The key under which *spectra* is stored.
        """
        key = key or generate_key()
        self.spectra[key] = spectra
        return key

    def get_spectra(self, spectrum_id: str) -> Spectrum:
        """Retrieve an external spectrum by ID.

        Raises
        ------
        KeyError
            If *spectrum_id* is not found.
        """
        if spectrum_id in self.spectra:
            return self.spectra[spectrum_id]
        raise KeyError(f"External spectrum {spectrum_id} not found")

    def delete_spectra(self, spectrum_id: str) -> None:
        """Delete an external spectrum by ID."""
        self.spectra.pop(spectrum_id)

    # math / in-place operations

    def math(
        self,
        operand: str | Spectrum | float,
        agg_operand: str = "mean",
        mask: str | None = None,
        op: Literal["sub", "add"] = "sub",
        roi_x: npt.ArrayLike | None = None,
    ) -> None:
        """In-place arithmetic between data and a reference.

        Parameters
        ----------
        operand : str | Spectrum | float
            Reference: mask ID, :class:`Spectrum <ramappy.core.spectrum.Spectrum>`, or scalar.
        agg_operand : str
            Aggregation method when *operand* is a mask or external spectrum.
        mask : str | None
            Pixel selection. ``None`` is the full map.
        op : Literal["sub", "add"]
            Operation to perform: subtract or add the reference.
        roi_x : array_like | None
            Spectral ROI.
        """
        idxs, _, _ = self.get_indices(mask, roi_x)

        if isinstance(operand, (Spectrum, str)):
            if isinstance(operand, Spectrum):
                spectrum = operand
                needs_alignment = True
            else:
                spectrum, needs_alignment = self.get_reference_spectrum(operand, agg_operand, roi_x)

            if needs_alignment:
                # allow_multiple_spectra=True: a full per-pixel reference map is a supported operand below
                spectrum = self.align_external_spectrum(spectrum, overlap="full", allow_multiple_spectra=True)
            ref_data: Any = spectrum.data
        elif np.ndim(operand) == 0:
            ref_data = operand
        else:
            raise ValueError(f"{operand} appears to be neither a mask nor a spectrum")

        if np.ndim(ref_data) > 1:
            if ref_data.shape[const.Axis.PIXEL] != np.prod(self.map_shape):  # type: ignore
                ref_data = aggregate(ref_data, agg=agg_operand)  # type: ignore
            else:
                # keep flat (n_pixels, n_spectral) to match self._data[idxs], not the (H, W, C) map shape
                ref_data = ref_data.reshape(-1, ref_data.shape[-1])  # type: ignore
        elif np.ndim(ref_data) == 1:
            ref_data = ref_data[np.newaxis, :]  # type: ignore

        data = self._data
        if op == "sub":
            data[idxs] -= ref_data
        elif op == "add":
            data[idxs] += ref_data

    # transaction support

    def begin_transaction(self) -> dict[str, Any]:
        """Begin a transaction on all entity maps simultaneously."""
        return {
            "masks": self.masks.begin_transaction(),
            "masks_group": self.masks_group.begin_transaction(),
            "images": self.images.begin_transaction(),
            "images_group": self.images_group.begin_transaction(),
            "spectra": self.spectra.begin_transaction(),
        }
