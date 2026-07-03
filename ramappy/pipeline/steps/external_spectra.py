from __future__ import annotations

import copy
from pathlib import Path

from pydantic import Field, FilePath
from pydantic_extra_types.color import Color

from ramappy.core.images2d.rules import ColorType
from ramappy.core.pipeline import ParamsSetResultId, SingleModelParamsValidator, StepClass, StepParams, pipeline_step
from ramappy.core.spectral_map import SpectralMap
from ramappy.io.core import InputFormatRegistry


class InputConfigParams(StepParams, ParamsSetResultId):
    """Parameters for `add_spectrum`."""

    file_path: FilePath | None = None
    """Path to the input file containing the spectrum data."""

    format_name: str | None = None
    """Name of the input format (must be registered in :class:`InputFormatRegistry <ramappy.io.core.InputFormatRegistry>`)."""

    format_params: dict  # will be parsed later
    """Format-specific reader parameters (validated by the format reader)."""

    as_hsi: bool = False
    """Whether to treat the input file as a hyperspectral image (HSI)."""


@pipeline_step(
    step_name="add_spectrum",
    params_validator=SingleModelParamsValidator(InputConfigParams),
    friendly_name="Add External Spectrum",
    step_category=StepClass.UTILITY,
)
def add_spectrum(
    spectral_map: SpectralMap,
    *,
    file_path: Path | None,
    format_name: str,
    res_id: str,
    as_hsi: bool | None = None,
    format_params: dict,
):
    """Import an external spectrum file and attach it to the map.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    file_path : Path or None
        Path to the input spectrum file. Required.
    format_name : str
        Name of the registered input format reader.
    res_id : str
        Key under which the imported spectrum is stored.
    as_hsi : bool or None, optional
        If ``True``, treat the file as a hyperspectral image.
    format_params : dict
        Format-specific reader parameters.

    Raises
    ------
    ValueError
        If *file_path* is ``None`` or *format_name* is not registered.
    """
    if file_path is None:
        raise ValueError("file_path is required")
    reader = InputFormatRegistry.get_format(format_name)
    if reader is None:
        raise ValueError(
            f"Format {format_name} not supported. Available formats: {list(InputFormatRegistry.formats.keys())}"
        )
    spectra = reader.read(file_path, format_params, as_hsi=as_hsi)
    spectral_map.add_spectra(spectra, key=res_id)


class StepDeleteExternalSpectrumParams(StepParams):
    """Parameters for `delete_external_spectrum`."""

    sp_id: str = Field(alias="id")
    """Identifier of the spectrum to delete from the map."""


@pipeline_step(
    step_name="delete_external_spectrum",
    params_validator=SingleModelParamsValidator(StepDeleteExternalSpectrumParams),
    friendly_name="Delete External Spectrum",
    step_category=StepClass.UTILITY,
)
def delete_external_spectrum(spectral_map: SpectralMap, sp_id: str):
    """Remove an external spectrum from the spectral map.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    sp_id : str
        Identifier of the spectrum to remove.
    """
    spectral_map.delete_spectra(sp_id)


class StepColorExternalSpectrumParams(StepParams):
    """Parameters for `color_external_spectrum`."""

    sp_id: str = Field(alias="id")
    """Identifier of the spectrum to color in the map."""

    color: ColorType | None = None
    """Color to assign to the spectrum (can be a hex string or a Color object)."""


@pipeline_step(
    step_name="color_external_spectrum",
    params_validator=SingleModelParamsValidator(StepColorExternalSpectrumParams),
    friendly_name="Set Color for External Spectrum",
    step_category=StepClass.UTILITY,
)
def color_external_spectrum(spectral_map: SpectralMap, sp_id: str, color: ColorType | None):
    """Assign a colour to an external spectrum in the spectral map.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    sp_id : str
        Identifier of the spectrum to colour.
    color : ColorType or None
        New colour. If ``None``, the function returns without making changes.
    """
    if color is None:
        return
    color_hex = color.as_hex(format="long") if isinstance(color, Color) else color
    # Workaround: EntityTransaction detects changes via object identity (is not).
    # set_color() mutates in place, so without copy.copy the transaction would miss
    # the change and the updated color would not be propagated.
    spectrum = copy.copy(spectral_map.get_spectra(sp_id))
    spectrum.set_color(color_hex)
    spectral_map.spectra[sp_id] = spectrum


__all__ = [
    "InputConfigParams",
    "StepColorExternalSpectrumParams",
    "StepDeleteExternalSpectrumParams",
    "add_spectrum",
    "color_external_spectrum",
    "delete_external_spectrum",
]
