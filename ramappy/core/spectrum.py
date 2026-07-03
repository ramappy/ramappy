"""Core spectrum type for ramappy.

Provides:

- :class:`Spectrum <ramappy.core.spectrum.Spectrum>`: a single or multi-pixel spectrum with x-axis, spectral
  data, units, and optional processing history.
- :class:`SpectralMapMetadata <ramappy.core.spectrum.SpectralMapMetadata>`: structured metadata attached to a dataset.
- :class:`InstrumentMetadata <ramappy.core.spectrum.InstrumentMetadata>`: instrument / measurement condition metadata.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ramappy import const, units
from ramappy.core._spectral_map.functional import _apply_rows_standalone
from ramappy.core.collections import OrderedEntityMap
from ramappy.core.metadata import SpectralMetadata
from ramappy.core.pipeline import HistoryLog, ProcessingStepConfig, SpectralData
from ramappy.utils import (
    clean_roi,
    color_generator,
    common_roi_overlap,
    find_nearest_x_many,
    is_sorted,
    select_x_indices,
)
from ramappy.utils.dependencies import _require_pandas_dependencies

TSpectrum = TypeVar("TSpectrum", bound="Spectrum")


# Structured metadata models


class InstrumentMetadata(BaseModel):
    """Metadata describing the instrument and measurement conditions.

    All fields are optional. Unknown fields round-trip through
    `extra` (``model_config = extra="allow"``).
    """

    model_config = ConfigDict(extra="allow")

    laser_wavelength_nm: float | None = Field(default=None, description="Excitation laser wavelength (nm).")
    laser_power_mw: float | None = Field(default=None, description="Laser power at sample (mW).")
    acquisition_time_s: float | None = Field(default=None, description="Per-pixel acquisition time (s).")
    accumulations: int | None = Field(default=None, description="Number of spectral accumulations per pixel.")
    detector: str | None = Field(default=None, description="Detector model or description.")
    objective: str | None = Field(default=None, description='Microscope objective label (e.g., ``"100x/0.9 NA"``).')
    grating: str | None = Field(default=None, description="Spectrometer grating description.")
    instrument_model: str | None = Field(default=None, description="Instrument model name.")


class SpectralMapMetadata(BaseModel):
    """Structured metadata for a :class:`SpectralMap <ramappy.core.spectral_map.SpectralMap>`.

    Designed to be serialised to/from Zarr attributes as a JSON object.
    """

    model_config = ConfigDict(extra="forbid")

    original_filename: str | None = Field(default=None, description="Source file from which the data was loaded.")
    title: str | None = Field(default=None, description="Human-readable title for the dataset.")
    description: str | None = Field(default=None, description="Free-text description or notes.")
    instrument: InstrumentMetadata | None = Field(
        default=None, description="Instrument and measurement condition metadata."
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional metadata not covered by the schema above. Preserved across serialization without validation.",
    )

    @classmethod
    def from_any(cls, value: SpectralMapMetadata | dict[str, Any] | None) -> SpectralMapMetadata:
        """Coerce *value* to a :class:`SpectralMapMetadata <ramappy.core.spectrum.SpectralMapMetadata>` instance.

        - ``None`` -> empty instance.
        - ``dict`` -> validated instance (unknown keys go to ``extra``).
        - ``SpectralMapMetadata`` -> returned as-is.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            # Pull any unexpected keys into `extra`
            known = cls.model_fields.keys()
            extra: dict[str, Any] = {}
            clean: dict[str, Any] = {}
            for k, v in value.items():
                if k in known:
                    clean[k] = v
                else:
                    extra[k] = v
            if extra:
                clean.setdefault("extra", {}).update(extra)
            return cls.model_validate(clean)
        raise TypeError(f"Cannot convert {type(value)} to SpectralMapMetadata")


# Spectrum


class Spectrum:
    """A single or multi-pixel spectrum.

    Parameters
    ----------
    x : np.ndarray | list[float]
        Spectral axis (e.g., wavenumbers). Sorted ascending automatically.
    data : SpectralData
        Spectral data array, shape ``(n_pixels, n_spectral)`` or ``(n_spectral,)``.
        Converted to at-least-2D internally.
    x_axis_unit : type_spectral_quantityunit
        Unit of the spectral axis.
    data_unit : type_data_quantityunit
        Unit of the spectral data axis.
    roi_x : array_like | None
        Spectral regions of interest, shape ``(n_regions, 2)``.
    aux_data : dict-like | None
        Additional data arrays keyed by name (e.g., ``"old_data"``).
    name : str | None
        Human-readable label.
    metadata : dict | None
        Arbitrary metadata dict (legacy; prefer storing in the parent
        :class:`SpectralMap <ramappy.core.spectral_map.SpectralMap>`).
    history : HistoryLog | dict | None
        Processing history.
    ignore_sort : bool
        Skip sorting the spectral axis (use only if already sorted).
    color : str | None
        Display colour for plots.
    dtype : dtype-like
        dtype for spectral data storage (default ``float32``).
    """

    _data: np.ndarray  # backing store; always ndarray (set via data property setter)

    def __init__(
        self,
        x: np.ndarray | list[float],
        data: SpectralData,
        x_axis_unit: units.type_spectral_quantityunit = units.DEFAULT_SPECTRAL_QUANTITY_UNIT,
        data_unit: units.type_data_quantityunit = units.DEFAULT_DATA_QUANTITY_UNIT,
        roi_x: npt.ArrayLike | None = None,
        aux_data: OrderedEntityMap | dict[str, SpectralData] | None = None,
        name: str | None = None,
        metadata: SpectralMetadata | dict[str, Any] | None = None,
        history: HistoryLog | dict | None = None,
        ignore_sort: bool = False,
        color: str | None = None,
        dtype: npt.DTypeLike = np.float32,
    ) -> None:
        sorted_x_idxs = slice(None) if ignore_sort or is_sorted(x) else np.argsort(x)
        self.x = np.asarray(x)[sorted_x_idxs]
        self.data = np.asarray(np.atleast_2d(data))[:, sorted_x_idxs]

        requested_dtype = np.dtype(dtype)
        if self.data.dtype != requested_dtype:
            should_warn = not (requested_dtype == np.dtype(np.float32) and self.data.dtype.kind == "f")
            if should_warn:
                warnings.warn(f"Data type forced to {requested_dtype}", stacklevel=2)
            self.data = self.data.astype(requested_dtype, casting="unsafe", copy=False)

        self.roi_x = np.array([[self.x[0], self.x[-1]]])
        if roi_x is not None:
            self.roi_x = self.adapt_roi_x(roi_x)

        self.aux_data = OrderedEntityMap(dict(aux_data) if isinstance(aux_data, OrderedEntityMap) else aux_data)
        self.name = name
        self._spectral_metadata: SpectralMetadata = SpectralMetadata.from_any(copy.deepcopy(metadata))

        # Parse unit arguments (accept QuantityUnit, tuple, dict, or None).
        self.x_axis_unit = self._parse_unit(x_axis_unit)
        self.data_unit = self._parse_unit(data_unit)

        self.color: str | None = None
        self.set_color(color)

        self.history = self._parse_history(history)

    # internal helpers

    @staticmethod
    def _parse_unit(value: units.type_spectral_quantityunit) -> units.QuantityUnit | None:
        if value is None:
            return None
        if isinstance(value, units.QuantityUnit):
            return value
        if isinstance(value, dict):
            return units.QuantityUnit(**value)
        if isinstance(value, (list, tuple)):
            return units.QuantityUnit(*value)
        return value

    @staticmethod
    def _parse_history(history: HistoryLog | dict | None) -> HistoryLog:
        if history is None:
            return HistoryLog()
        if isinstance(history, HistoryLog):
            return history
        if isinstance(history, dict):
            try:
                return HistoryLog.model_validate(history)
            except ValidationError as exc:
                warnings.warn(f"Error parsing history: {exc}", stacklevel=2)
                log = HistoryLog()
                input_data = history.get("input")
                if isinstance(input_data, dict):
                    log.set_input_config(
                        input_format=input_data.get("format_name"),
                        input_config=input_data.get("format_params"),
                    )
                parsed_steps: list[ProcessingStepConfig | dict] = []
                for step in history.get("steps", []):
                    if not isinstance(step, dict):
                        continue
                    try:
                        parsed_steps.append(ProcessingStepConfig.model_validate(step))
                    except ValidationError:
                        parsed_steps.append(step)
                log.steps = parsed_steps
                return log
        return HistoryLog()

    # properties

    @property
    def data(self) -> np.ndarray:
        """Primary spectral data array."""
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        self._data = np.atleast_2d(value)

    @property
    def spectral_axis(self) -> np.ndarray:
        """Public alias for `x`."""
        return self.x

    @spectral_axis.setter
    def spectral_axis(self, value: npt.ArrayLike) -> None:
        self.x = np.asarray(value)

    @property
    def metadata(self) -> dict[str, Any]:
        """Additional metadata fields (`extra`) as a mutable dict."""
        return self._spectral_metadata.extra

    @metadata.setter
    def metadata(self, value: SpectralMetadata | dict[str, Any]) -> None:
        self._spectral_metadata = SpectralMetadata.from_any(value)

    @property
    def spectral_metadata(self) -> SpectralMetadata:
        """Structured metadata model for this spectrum."""
        return self._spectral_metadata

    @spectral_metadata.setter
    def spectral_metadata(self, value: SpectralMetadata | dict[str, Any] | None) -> None:
        self._spectral_metadata = SpectralMetadata.from_any(value)

    @property
    def n_pixels(self) -> int:
        """Number of pixels (spectra) in this object."""
        return int(self.data.shape[const.Axis.PIXEL])

    @property
    def n_spectral(self) -> int:
        """Number of spectral channels."""
        return int(self.data.shape[const.Axis.SPECTRAL])

    # core methods

    def apply_func(
        self,
        f: Callable,
        *,
        data: SpectralData | None = None,
        preserve_input_dtype: bool = True,
        by: str = "pixel",
        parallel: bool | str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply *f* to spectral data row-by-row (``by='pixel'``)."""
        if by != "pixel":
            raise ValueError("Only by='pixel' is supported for Spectrum.apply_func")
        if data is None:
            data = self.data
        data = np.asarray(data)
        output_dtype = data.dtype if preserve_input_dtype else None
        return _apply_rows_standalone(f, data, parallel=parallel, output_dtype=output_dtype, kwargs=kwargs)

    def get_indices(
        self,
        mask: str | None = None,
        roi_x: npt.ArrayLike | None = None,
        ignore_empty_mask: bool = False,
    ) -> tuple[tuple, np.ndarray | slice, slice]:
        """Return index tuples for spectral data access.

        Parameters
        ----------
        mask : str | None
            Ignored for :class:`Spectrum <ramappy.core.spectrum.Spectrum>` (masks apply to SpectralMap).
        roi_x : array_like | None
            Spectral ROI to restrict the spectral axis.

        Returns
        -------
        tuple
            ``(idxs, x_idx, pixel_idx)`` as used by
            :meth:`SpectralMap.get_indices <ramappy.core.spectral_map.SpectralMap.get_indices>`.
        """
        if mask is not None:
            warnings.warn("Mask is not supported for single-spectrum data: ignoring it", stacklevel=2)
        pixel_idx = slice(None)
        x_idx = select_x_indices(self.x, roi_x)
        return (pixel_idx, x_idx), x_idx, pixel_idx

    def adapt_roi_x(self, roi_x: npt.ArrayLike | None) -> np.ndarray:
        """Clip *roi_x* to the spectral range of this spectrum."""
        if roi_x is not None:
            roi_arr = common_roi_overlap(clean_roi(np.asarray(roi_x)), self.roi_x)
            endpoint_idxs = find_nearest_x_many(self.x, roi_arr.ravel())
            return self.x[endpoint_idxs].reshape(roi_arr.shape)
        return self.roi_x

    def align_external_spectrum(
        self,
        spectrum: TSpectrum,
        overlap: bool | Literal["full"] = True,
        common_spectral_axis: bool = False,
        ignore_calibration: bool = False,
        allow_multiple_spectra: bool = False,
    ) -> TSpectrum:
        """Align *spectrum* to this spectrum's spectral axis and ROI.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum to align.
        overlap : bool | 'full'
            ``True``: allow partial overlap. ``'full'``: ROIs must match
            exactly. ``False``: skip overlap check entirely.
        common_spectral_axis : bool
            If ``True``, require identical spectral axes.
        ignore_calibration : bool
            Skip unit comparison.
        allow_multiple_spectra : bool
            If ``False``, raise when *spectrum* contains more than one pixel.

        Returns
        -------
        Spectrum
            A (possibly sliced/resampled) copy of *spectrum*.
        """
        x_resampling_grid = None

        if (
            not ignore_calibration
            and self.x_axis_unit is not None
            and spectrum.x_axis_unit is not None
            and self.x_axis_unit.unit != spectrum.x_axis_unit.unit
        ):
            raise ValueError("Reference spectrum has a different spectral unit")

        if allow_multiple_spectra and np.squeeze(spectrum.data).ndim != 1:
            raise ValueError("Reference spectrum is composed of multiple spectra")

        if common_spectral_axis:
            if self.x_axis_unit != spectrum.x_axis_unit:
                if len(spectrum.x) == len(self.x):
                    warnings.warn(
                        "Reference spectrum has the same size, but different unit. Using it anyway…", stacklevel=2
                    )
                    return spectrum
                raise ValueError("Reference spectrum has a different spectral unit")
            if np.array_equal(spectrum.x, self.x):
                return spectrum
            raise ValueError("Reference spectrum has a different spectral axis")

        if overlap is not False:
            if spectrum.roi_x[-1][-1] < self.roi_x[0][0] or spectrum.roi_x[0][0] > self.roi_x[-1][-1]:
                raise ValueError("Reference spectrum is out of the spectral range of the SpectralMap data")
            roi_x_common = common_roi_overlap(spectrum.roi_x, self.roi_x)
            if len(roi_x_common) == 0:
                raise ValueError("No overlapping regions of interest")
            if overlap == "full" and not np.allclose(self.roi_x, roi_x_common, rtol=1e-04):
                raise ValueError("Regions of interest do not fully match")

            x_idx_hsi = select_x_indices(self.x, roi_x_common)
            x_idx_ref = select_x_indices(spectrum.x, roi_x_common)
            if not np.array_equal(spectrum.x[x_idx_ref], self.x[x_idx_hsi]):
                x_resampling_grid = select_x_indices(self.x, roi_x_common, keep_regions=True)
                x_resampling_grid = [self.x[x_idx] for x_idx in x_resampling_grid]
        else:
            return spectrum

        new_spectrum = copy.deepcopy(spectrum)

        if x_resampling_grid is not None:
            from ramappy.processing import resample

            resample(new_spectrum, x_grid=x_resampling_grid, bounds_error=False)
            return new_spectrum

        new_spectrum.data = new_spectrum.data[:, x_idx_ref]
        new_spectrum.x = spectrum.x[x_idx_ref]
        new_spectrum.roi_x = roi_x_common
        return new_spectrum

    def set_color(self, color: str | None) -> None:
        """Set the display colour, using a generated colour if *color* is ``None``."""
        if color is not None and mpl.colors.is_color_like(color):
            self.color = color
        else:
            self.color = color_generator()

    # export

    def to_polars(
        self,
        *,
        transpose: bool = False,
        metadata: dict[str, str | float] | None = None,
    ) -> pl.DataFrame:
        """Export to a `polars.DataFrame`.

        Parameters
        ----------
        transpose : bool
            If ``True``, transpose so that each row is a sample and each
            column is a wavenumber.
        metadata : dict | None
            Extra columns to embed in non-transposed mode (stored in column
            names or as extra literal columns when transposed).

        Returns
        -------
        pl.DataFrame
        """
        name = self.name if self.name is not None else "data"
        df = pl.DataFrame({"x": self.x})

        if self.data.shape[const.Axis.PIXEL] > 1:
            data_df = pl.DataFrame(self.data.T)
            if not transpose and metadata is not None:
                data_df.columns = [str({name: i} | metadata) for i in range(self.data.shape[const.Axis.PIXEL])]
            else:
                data_df.columns = [f"{name}_{i}" for i in range(self.data.shape[const.Axis.PIXEL])]
            df = pl.concat([df, data_df], how="horizontal")
        else:
            column_name = str({"sample": name} | metadata) if not transpose and metadata is not None else name
            df = df.with_columns(pl.lit(self.data.flatten()).alias(column_name))

        if transpose:
            df = df.with_columns(pl.col("x").cast(pl.String)).transpose(
                include_header=True,
                column_names="x",
                header_name="samples",
            )
            if metadata is not None:
                for k, v in metadata.items():
                    df = df.with_columns(pl.lit(str(v), dtype=pl.Categorical).alias(k))

        return df

    def to_pandas(self, *, transpose: bool = False, metadata: dict[str, str | float] | None = None):
        """Export to a :class:`pandas.DataFrame`.

        .. deprecated::
            Use :meth:`to_polars` instead. This method delegates to
            ``to_polars(...).to_pandas()`` and will be removed in a future
            version.
        """
        warnings.warn(
            "Spectrum.to_pandas() is deprecated; use to_polars() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _require_pandas_dependencies()
        return self.to_polars(transpose=transpose, metadata=metadata).to_pandas()
