"""Core IO abstractions and registries.

This module defines the public building blocks used to register input/output
file formats in `ramappy`:

- parameter models (:class:`IOParams <ramappy.io.core.IOParams>` and mixins)
- format descriptors (:class:`InputFormat <ramappy.io.core.InputFormat>`, :class:`OutputFormat <ramappy.io.core.OutputFormat>`)
- global registries and registration decorators

All readers/writers from `ramappy.io` use these abstractions.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Flag, auto
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from ramappy.core import SpectralMap, Spectrum
from ramappy.core.exceptions import PipelineError
from ramappy.core.metadata import metadata_as_extra as _metadata_as_extra
from ramappy.core.metadata import metadata_with_extra as _metadata_with_extra


class IOParams(BaseModel):
    """Base model for IO parameter schemas.

    Reader and writer format models should inherit from this class (directly or
    through mixins) so registry helpers can validate arguments consistently.
    """

    pass


class IOParamsAsHsi(BaseModel):
    """Mixin for readers that can return either map or spectrum objects."""

    as_hsi: bool = True
    """If `True`, return a :class:`SpectralMap`. If `False`, return a :class:`Spectrum`."""


class IOParamsUnits(BaseModel):
    """Mixin exposing optional axis-unit overrides for readers."""

    x_axis_unit: tuple[str, str] | None = None
    """Optional spectral axis unit as ``(quantity, unit)``."""

    data_unit: tuple[str, str] | None = None
    """Optional data/intensity axis unit as ``(quantity, unit)``."""


def validate_params(
    format_params_model: type[IOParams], params: dict[str, Any] | IOParams, *, legacy: bool = False
) -> IOParams:
    """Validate input/output format parameters against a Pydantic model.

    Parameters
    ----------
    format_params_model : type[IOParams]
        Model class used to validate parameters.
    params : dict[str, Any] | IOParams
        Parameter payload.
    legacy : bool, default=False
        If ``True``, missing fields in dictionary payloads are backfilled with
        model defaults before validation.

    Returns
    -------
    IOParams
        Validated parameter object.

    Raises
    ------
    PipelineError
        If validation fails.
    """
    if isinstance(params, format_params_model):
        return params

    try:
        if legacy and isinstance(params, dict):
            params = {
                k: params.get(k, format_params_model.model_fields[k].default) for k in format_params_model.model_fields
            }

        if isinstance(params, bytes):
            return format_params_model.model_validate_json(params)

        return format_params_model.model_validate(params)
    except ValidationError as e:
        raise PipelineError(e, input_data=params, model=format_params_model) from e


def metadata_as_extra(metadata: dict[str, Any] | None = None, /, **kwargs: Any) -> dict[str, Any]:
    """Wrap reader metadata so values are persisted under ``extra``.

    Parameters
    ----------
    metadata : dict[str, Any] | None
        Base metadata mapping.
    **kwargs : Any
        Extra key/value pairs merged into *metadata*.

    Returns
    -------
    dict[str, Any]
        Metadata payload compatible with ``SpectralMapMetadata`` where all
        imported fields are stored under ``extra``.
    """

    return _metadata_as_extra(metadata, **kwargs)


def metadata_with_extra(
    *,
    title: str | None = None,
    description: str | None = None,
    original_filename: str | None = None,
    instrument: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    vendor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build structured metadata with a single flexible ``extra`` container.

    Thin wrapper around :func:`ramappy.core.metadata.metadata_with_extra`.
    """

    return _metadata_with_extra(
        title=title,
        description=description,
        original_filename=original_filename,
        instrument=instrument,
        extra=extra,
        vendor=vendor,
    )


@dataclass
class InputFormat:
    """Descriptor for a registered input format.

    Instances are created by :class:`InputFormatRegistry <ramappy.io.core.InputFormatRegistry>` and expose the
    :meth:`read` method used by high-level IO helpers.
    """

    format_name: str
    format_params_model: type[IOParams]
    friendly_name: str
    extensions: set[str]
    has_custom_params: bool

    _read_func: Callable[..., SpectralMap | Spectrum]
    param_hints_from_file: Callable[[str], dict[str, Any]] | None = None
    sniffer: Callable[[str], bool] | None = None

    def read(
        self,
        filepath_or_buffer: Path | str,
        params: dict[str, Any] | IOParams | None = None,
        *,
        as_hsi: bool | None = None,
        legacy: bool = False,
    ) -> SpectralMap | Spectrum:
        """Read a file-like input using this format definition.

        Parameters
        ----------
        filepath_or_buffer : Path | str
            Input source path or file-like handle.
        params : dict[str, Any] | IOParams | None, default=None
            Reader parameters.
        as_hsi : bool | None, optional
            Optional runtime override for the ``as_hsi`` parameter.
        legacy : bool, default=False
            Enable legacy parameter normalization behavior.

        Returns
        -------
        SpectralMap | Spectrum
            Parsed object produced by the underlying reader.
        """

        params = validate_params(self.format_params_model, params or {}, legacy=legacy)

        if issubclass(self.format_params_model, IOParamsAsHsi) and as_hsi is not None:
            # override as_hsi if it's in the params
            params_any: Any = params
            params_any.as_hsi = as_hsi
        spectral_map = self._read_func(filepath_or_buffer, **params.model_dump())
        input_cfg = spectral_map.history.input
        has_input_format = isinstance(input_cfg, dict) and bool(input_cfg.get("format_name"))
        if not has_input_format:
            spectral_map.history.set_input_config(input_format=self.format_name, input_config=params)
        return spectral_map

    def __repr__(self):
        return f'<InputFormat "{self.friendly_name}" ({self.format_name})>'

    def __hash__(self):
        return hash(self.format_name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InputFormat):
            return NotImplemented
        return self.format_name == other.format_name

    def validate_params(
        self,
        params: dict[str, Any] | IOParams,
        *,
        legacy: bool = False,
    ) -> IOParams:
        """Validate parameters for this input format."""
        try:
            return validate_params(self.format_params_model, params, legacy=legacy)
        except ValidationError as e:
            raise PipelineError(e, input_data=params, model=self.format_params_model) from e


class InputFormatRegistry:
    """Global registry of all available input formats."""

    formats: ClassVar[dict[str, InputFormat]] = {}
    extension_map: ClassVar[dict[str, set[InputFormat]]] = {}
    formats_with_param_hints: ClassVar[set[str]] = set()

    @classmethod
    def register(
        cls,
        format_name: str,
        format_params_model: type[IOParams],
        *,
        friendly_name: str,
        extensions: set[str],
        read_func: Callable[[Path | str, Any], SpectralMap | Spectrum],
        has_custom_params: bool = True,
        param_hints_from_file: Callable[[str], dict[str, Any]] | None = None,
        sniffer: Callable[[str], bool] | None = None,
    ):
        """Register a new input format.

        Parameters
        ----------
        format_name : str
            Stable machine name (for example ``"zarr"``).
        format_params_model : type[IOParams]
            Parameter schema model.
        friendly_name : str
            Human-readable format label.
        extensions : set[str]
            File extensions supported by this format.
        read_func : Callable[[Path | str, Any], SpectralMap | Spectrum]
            Reader callable implementation.
        has_custom_params : bool, default=True
            Whether this format exposes configurable parameters.
        param_hints_from_file : Callable[[str], dict[str, Any]] | None, optional
            Optional callback to infer defaults from file content.
        sniffer : Callable[[str], bool] | None, optional
            Optional detector used when multiple formats share an extension.
        """

        cls.formats[format_name] = InputFormat(
            format_name=format_name,
            format_params_model=format_params_model,
            friendly_name=friendly_name,
            extensions=extensions,
            has_custom_params=has_custom_params,
            _read_func=read_func,
            param_hints_from_file=param_hints_from_file,
            sniffer=sniffer,
        )
        for extension in extensions:
            cls.extension_map.setdefault(extension, set()).add(cls.formats[format_name])

        if param_hints_from_file is not None:
            cls.formats_with_param_hints.add(format_name)

    @classmethod
    def from_extension(cls, extension: str) -> set[InputFormat] | None:
        return cls.extension_map.get(extension)

    @classmethod
    def get_format(cls, format_name: str) -> InputFormat | None:
        return cls.formats.get(format_name)

    @classmethod
    def list_formats(cls) -> list[str]:
        return list(cls.formats.keys())


def input_format(
    format_name: str,
    *,
    friendly_name: str,
    extensions: set[str],
    format_params_model: type[IOParams],
    has_custom_params: bool = True,
    param_hints_from_file: Callable[[Path | str], dict[str, Any]] | None = None,
    sniffer: Callable[[str], bool] | None = None,
):
    """Decorator registering a function as an input format reader.

    Returns
    -------
    Callable
        Decorator that registers the wrapped function and returns it unchanged.
    """

    def decorator(read_func: Callable[[Path | str, Any], SpectralMap | Spectrum]):
        InputFormatRegistry.register(
            format_name=format_name,
            format_params_model=format_params_model,
            friendly_name=friendly_name,
            extensions=extensions,
            has_custom_params=has_custom_params,
            read_func=read_func,
            param_hints_from_file=param_hints_from_file,
            sniffer=sniffer,
        )
        return read_func

    return decorator


class SpectrumType(Flag):
    """Bitwise flags describing which spectral object types are supported."""

    SPECTRAL_MAP = auto()
    SPECTRUM = auto()


@dataclass
class OutputFormat:
    """Descriptor for a registered output format."""

    format_name: str
    format_params_model: type[IOParams]
    friendly_name: str
    extensions: set[str]
    has_custom_params: bool
    _write_func: Callable[..., Any]
    supported_types: SpectrumType = SpectrumType.SPECTRAL_MAP | SpectrumType.SPECTRUM

    def write(
        self,
        spectral_map: SpectralMap | Spectrum,
        out_file: str | os.PathLike | BytesIO | None,
        params: dict[str, Any] | IOParams | None = None,
    ) -> Any:
        """Write a spectrum/map object using this output format.

        Parameters
        ----------
        spectral_map : SpectralMap | Spectrum
            Object to serialize.
        out_file : str
            Destination path.
        params : dict[str, Any] | IOParams | None, default=None
            Writer parameters.

        Returns
        -------
        Any
            Writer-specific return value (often ``None``).
        """

        params = validate_params(self.format_params_model, params or {})
        # some might return something other than None
        return self._write_func(spectral_map, out_file=out_file, **params.model_dump())

    def __repr__(self):
        return f'<OutputFormat "{self.friendly_name}" ({self.format_name})>'

    def __hash__(self):
        return hash((self.format_name, self.supported_types))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OutputFormat):
            return NotImplemented
        return (self.format_name, self.supported_types) == (other.format_name, other.supported_types)

    def validate_params(
        self,
        params: dict[str, Any] | IOParams,
    ) -> IOParams:
        """Validate parameters for this input format."""
        try:
            return validate_params(self.format_params_model, params)
        except ValidationError as e:
            raise PipelineError(e, input_data=params, model=self.format_params_model) from e


class OutputFormatRegistry:
    """Global registry of all available output formats."""

    formats: ClassVar[dict[str, OutputFormat]] = {}
    formats_by_name: ClassVar[dict[str, list[OutputFormat]]] = {}
    extension_map: ClassVar[dict[str, set[OutputFormat]]] = {}

    @classmethod
    def register(
        cls,
        format_name: str,
        format_params_model: type[IOParams],
        *,
        friendly_name: str,
        extensions: set[str],
        write_func: Callable[[str, Any], SpectralMap | Spectrum],
        has_custom_params: bool = True,
        supported_types: SpectrumType = SpectrumType.SPECTRAL_MAP | SpectrumType.SPECTRUM,
    ):
        output_format = OutputFormat(
            format_name=format_name,
            format_params_model=format_params_model,
            friendly_name=friendly_name,
            extensions=extensions,
            has_custom_params=has_custom_params,
            _write_func=write_func,
            supported_types=supported_types,
        )

        variants = cls.formats_by_name.setdefault(format_name, [])
        variants.append(output_format)

        # Backward-compatible "primary" entry for code paths that still access
        # `OutputFormatRegistry.formats[name]` directly
        cls.formats[format_name] = cls._select_default_variant(variants)

        for extension in extensions:
            cls.extension_map.setdefault(extension, set()).add(output_format)

    @staticmethod
    def _select_default_variant(variants: list[OutputFormat]) -> OutputFormat:
        # Prefer variants that support SpectralMap exports (the most common
        # `export_spectra` call path), then keep first registration order.
        map_capable = [fmt for fmt in variants if SpectrumType.SPECTRAL_MAP in fmt.supported_types]
        if map_capable:
            return map_capable[0]
        return variants[0]

    @classmethod
    def from_extension(cls, extension: str) -> set[OutputFormat] | None:
        return cls.extension_map.get(extension)

    @classmethod
    def get_format(cls, format_name: str, *, supported_type: SpectrumType | None = None) -> OutputFormat | None:
        if supported_type is None:
            return cls.formats.get(format_name)

        for writer in cls.formats_by_name.get(format_name, []):
            if supported_type in writer.supported_types:
                return writer
        return None

    @classmethod
    def get_formats(cls, format_name: str) -> list[OutputFormat]:
        return list(cls.formats_by_name.get(format_name, []))

    @classmethod
    def get_format_for_data(cls, format_name: str, data: SpectralMap | Spectrum) -> OutputFormat | None:
        required_type = SpectrumType.SPECTRAL_MAP if isinstance(data, SpectralMap) else SpectrumType.SPECTRUM
        return cls.get_format(format_name, supported_type=required_type)

    @classmethod
    def list_formats(cls) -> list[str]:
        return list(cls.formats.keys())


def output_format(
    format_name: str,
    *,
    friendly_name: str,
    extensions: set[str],
    format_params_model: type[IOParams],
    has_custom_params: bool = True,
    supported_types: SpectrumType = SpectrumType.SPECTRAL_MAP | SpectrumType.SPECTRUM,
):
    """Decorator registering a function as an output format writer.

    Returns
    -------
    Callable
        Decorator that registers the wrapped function and returns it unchanged.
    """

    def decorator(write_func: Callable[[str, Any], SpectralMap | Spectrum]):
        OutputFormatRegistry.register(
            format_name=format_name,
            format_params_model=format_params_model,
            friendly_name=friendly_name,
            extensions=extensions,
            has_custom_params=has_custom_params,
            write_func=write_func,
            supported_types=supported_types,
        )
        return write_func

    return decorator
