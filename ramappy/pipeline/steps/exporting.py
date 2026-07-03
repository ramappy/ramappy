from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import polars as pl
from pydantic import Field, model_validator

from ramappy.core import SpectralMap, Spectrum
from ramappy.core.pipeline import (
    ParamsAcceptMask,
    ParamsOutputFile,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)
from ramappy.io import export_grouped_spectra, export_spectra, export_spectrum
from ramappy.io.core import IOParams, OutputFormatRegistry, SpectrumType


class StepExportSpectrumParams(StepParams, ParamsAcceptMask, ParamsOutputFile):
    """Parameters for `export_spectrum`."""

    agg: Literal["mean", "median", "mean95", "std", "max", "min"] | None = "mean"
    """Aggregation method for the exported spectrum (ignored if exporting a single-pixel mask)."""

    format_name: str = Field("csv", alias="format")
    """Output format name (must be registered in :class:`OutputFormatRegistry <ramappy.io.core.OutputFormatRegistry>`)."""

    add_mask_metadata: bool = False
    """If ``True``, add the number of pixels in the mask to the exported spectrum metadata."""

    ignore_empty_mask: bool = False
    """If ``True``, export an empty spectrum if the mask has no pixels (otherwise raise an error)."""

    as_dataframe: bool | Literal["pandas", "polars"] = False
    """If ``True``, return the exported spectrum as a dataframe instead of writing to a file. If ``"pandas"`` or ``"polars"``, return a dataframe of the specified type."""

    additional_info: dict[str, str | float] | None = None
    """Additional metadata to add to the exported spectrum."""

    transpose: bool = False
    """If ``True``, transpose the exported spectrum dataframe (only applies if ``as_dataframe`` is ``True``)."""


@pipeline_step(
    step_name="export_spectrum",
    params_validator=SingleModelParamsValidator(StepExportSpectrumParams),
    friendly_name="Export Spectrum",
    step_category=StepClass.UTILITY,
)
def _export_spectrum(
    spectral_map: SpectralMap,
    *,
    mask: str,
    agg: Literal["mean", "median", "mean95", "std", "max", "min"] | None = None,
    format_name: str,
    add_mask_metadata: bool,
    ignore_empty_mask: bool,
    additional_info: dict | None = None,
    as_dataframe: bool | Literal["pandas", "polars"] = False,
    out_file: str | None = None,
    transpose: bool = False,
) -> pl.DataFrame | None:
    metadata = additional_info or {}
    sp: Spectrum | None = None
    if mask is None or mask in spectral_map.masks:
        m = spectral_map.get_mask(mask)
        sp = spectral_map.get_spectrum(
            mask=mask,
            agg=agg,
            ignore_empty_mask=ignore_empty_mask,
            prefer_stored_property=m.spectrum_agg == "centroid",
        )
        if add_mask_metadata:
            metadata["num_pixels"] = len(m.idxs)
    elif mask in spectral_map.images:
        img = spectral_map.get_image(mask)
        sp = img.spectrum
        if sp is None:
            raise ValueError(f"Image {mask} does not have a spectrum.")
    else:
        raise ValueError(f"Unable to export {mask} spectrum (wrong mask?)")

    out = export_spectrum(
        sp,
        format_name,
        as_dataframe=as_dataframe,
        out_file=out_file,
        metadata=metadata,
        transpose=transpose,
    )
    if as_dataframe:
        return out
    return None


class StepExportListSpectraImageGroupParams(StepParams, ParamsOutputFile):
    """Parameters for `export_spectra_from_image_group`."""

    group_id: str
    """Image group identifier to export spectra from."""


@pipeline_step(
    step_name="export_spectra_from_image_group",
    params_validator=SingleModelParamsValidator(StepExportListSpectraImageGroupParams),
    friendly_name="Export Spectra from Image Group",
    step_category=StepClass.UTILITY,
)
def _export_spectra_from_image_group(
    spectral_map: SpectralMap,
    group_id: str,
    out_file: Path | str | None = None,
):
    out_file = out_file or f"{spectral_map.name}_{group_id}_spectra.csv"
    if group_id not in spectral_map.images_group:
        raise KeyError(f"Group {group_id} not found in images group.")
    sp = []
    for img_key in reversed(spectral_map.images_group[group_id].elem_ids or []):
        img = spectral_map.images[img_key]
        s = img.spectrum
        if s is None:
            raise ValueError(f"Image {img_key} does not have a spectrum.")
        sp.append(s.to_polars())
    pl.concat(sp, how="align").write_csv(out_file)


class StepExportListSpectraMaskGroupParams(StepParams, ParamsOutputFile):
    """Parameters for `export_spectra_from_mask_group`."""

    group_id: str
    """Mask group identifier to export spectra from."""


@pipeline_step(
    step_name="export_spectra_from_mask_group",
    params_validator=SingleModelParamsValidator(StepExportListSpectraMaskGroupParams),
    friendly_name="Export Spectra from Mask Group",
    step_category=StepClass.UTILITY,
)
def _export_spectra_from_mask_group(
    spectral_map: SpectralMap,
    group_id: str,
    out_file: Path | str | None = None,
):
    out_file = out_file or f"{spectral_map.name}_{group_id}_spectra.csv"
    if group_id not in spectral_map.masks_group:
        raise KeyError(f"Group {group_id} not found in masks group.")
    sp = []
    for mask_key in reversed(spectral_map.masks_group[group_id].elem_ids or []):
        mask = spectral_map.masks[mask_key]
        s = spectral_map.get_spectrum(
            mask=mask,
            agg="mean",
            prefer_stored_property=mask.spectrum_agg == "centroid",
        )
        if s is None:
            raise ValueError(f"Mask {mask_key} does not have a spectrum.")
        if s.name is None:
            s.name = mask_key
        sp.append(s.to_polars())
    pl.concat(sp, how="align").write_csv(out_file)


class StepExportSpectraParams(StepParams, ParamsOutputFile):
    """Parameters for `export_spectra`."""

    format_name: str
    """Output format name (must be registered in :class:`OutputFormatRegistry <ramappy.io.core.OutputFormatRegistry>`)."""

    format_params: dict | None = None
    """Format-specific writer parameters (validated by the format writer)."""

    @model_validator(mode="after")
    def check_format_params(self) -> Self:
        writer = OutputFormatRegistry.get_format(self.format_name, supported_type=SpectrumType.SPECTRAL_MAP)
        if writer is None:
            raise ValueError(f"Format {self.format_name} not supported.") from None
        validated_params = writer.validate_params(self.format_params or {})
        return self.model_copy(update={"format_params": validated_params})


@pipeline_step(
    step_name="export_spectra",
    params_validator=SingleModelParamsValidator(StepExportSpectraParams),
    friendly_name="Export Spectra",
    step_category=StepClass.UTILITY,
)
def _export_spectra(
    spectral_map: SpectralMap, format_name: str, out_file: str | None, format_params: IOParams | dict | None
):
    export_spectra(spectral_map, out_file=out_file, format_name=format_name, format_params=format_params)


class StepExportGroupedSpectraParams(StepParams):
    """Parameters for `export_grouped_spectra`."""

    group_key: str
    """Group identifier to export spectra from (must be a mask or image group)."""

    format_name: str = Field(default="csv", alias="format")
    """Output format name (must be registered in :class:`OutputFormatRegistry <ramappy.io.core.OutputFormatRegistry>`)."""

    out_path: str
    """Output folder path to write the exported spectra files."""


@pipeline_step(
    step_name="export_grouped_spectra",
    params_validator=SingleModelParamsValidator(StepExportGroupedSpectraParams),
    friendly_name="Export Grouped Spectra",
    step_category=StepClass.UTILITY,
)
def _export_grouped_spectra(spectral_map: SpectralMap, group_key: str, format_name: str, out_path: str):
    sp = []
    if group_key in spectral_map.images_group:
        for img_key in spectral_map.get_images_group(group_key).elem_ids or []:
            img = spectral_map.images[img_key]
            s = img.spectrum
            if s is not None:
                sp.append(s)
    elif group_key in spectral_map.masks_group:
        for mask_key in spectral_map.get_masks_group(group_key).elem_ids or []:
            m = spectral_map.get_mask(mask_key)
            s = spectral_map.get_spectrum(mask=mask_key, prefer_stored_property=m.spectrum_agg == "centroid")
            s.name = m.name
            sp.append(s)
    else:
        raise KeyError(f"Unable to export spectra from group {group_key}. Key not found.")

    export_grouped_spectra(sp, out_path, group_key, format=format_name)


__all__ = [
    "StepExportGroupedSpectraParams",
    "StepExportListSpectraImageGroupParams",
    "StepExportListSpectraMaskGroupParams",
    "StepExportSpectraParams",
    "StepExportSpectrumParams",
    "_export_grouped_spectra",
    "_export_spectra",
    "_export_spectra_from_image_group",
    "_export_spectra_from_mask_group",
    "_export_spectrum",
]
