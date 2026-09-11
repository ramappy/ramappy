"""Common IO convenience functions.

This module provides the primary entrypoints for reading and exporting
spectral data in `ramappy`. It delegates the actual work to the
format-specific readers and writers registered in [ramappy.io.core][].
"""

from __future__ import annotations

import logging
import os
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl

from ramappy.core import SpectralMap, Spectrum
from ramappy.io.core import (
    InputFormat,
    InputFormatRegistry,
    IOParams,
    OutputFormatRegistry,
    SpectrumType,
)
from ramappy.io.csv import write_csv_spectrum

logger = logging.getLogger(__name__)


def guess_file_type(filename: str) -> InputFormat:
    """Guess the most appropriate registered input format for a file path.

    Parameters
    ----------
    filename : str
        Input file path.

    Returns
    -------
    InputFormat
        Selected input format descriptor.

    Raises
    ------
    ValueError
        If no format supports the file extension.
    """

    ext = Path(filename).suffix[1:].lower()
    input_formats = list(InputFormatRegistry.extension_map.get(ext, []))
    if len(input_formats) == 0:
        raise ValueError(f"Extension {ext} not supported.")

    # Make selection deterministic.
    input_formats = sorted(input_formats, key=lambda f: f.format_name)

    if len(input_formats) > 1:
        # First try all sniffers.
        for fmt in input_formats:
            if fmt.sniffer is not None:
                try:
                    if fmt.sniffer(filename):
                        return fmt
                except Exception:
                    logger.debug("Input format sniffer failed for %s", fmt.format_name, exc_info=True)

        # Fallback if no sniffer matches:
        # prefer the generic format (without sniffer), e.g., `hdf5` over a specific flavour.
        generic_formats = [fmt for fmt in input_formats if fmt.sniffer is None]
        if len(generic_formats) > 0:
            return generic_formats[0]

        # If every candidate has a sniffer, pick the first deterministic entry.
        return input_formats[0]

    return input_formats[0]


def read_file(
    filepath_or_buffer: str,
    format: str = "auto",
    as_hsi: bool = True,
    **kwargs,
):
    """Import the map from file.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        The name of the file to read (any valid string path is acceptable) or an object with a read()
        method, such as a file handle (e.g., via builtin open function) or StringIO.
    format : str, default='auto'
        The format of the file to read.
    **kwargs : dict, optional
        Extra arguments to `format`: refer to each format documentation for a list of all possible
        arguments.

    Returns
    -------
    SpectralMap | Spectrum

    Raises
    ------
    ValueError
        If the format is not recognized.
    """
    if format == "auto":  # and isinstance(filepath_or_buffer, str)
        format_obj = guess_file_type(filepath_or_buffer)
        if format_obj is None:
            raise ValueError(f"Could not determine the input format for {filepath_or_buffer!r}.")
        format_name = format_obj.format_name
    else:
        format_name = format

    try:
        reader = InputFormatRegistry.formats[format_name]
    except KeyError as err:
        raise ValueError(f"Format {format_name} not supported.") from err

    spectral_map = reader.read(filepath_or_buffer, kwargs, as_hsi=as_hsi)

    if not as_hsi:
        return spectral_map  # in this case spectral_map is a Spectrum

    return spectral_map


def export_spectra(
    spectral_map: SpectralMap,
    *,
    out_file: BytesIO | os.PathLike | str | None = None,
    format_name: str | None = None,
    format_params: dict[str, Any] | IOParams | None,
) -> pl.DataFrame | BytesIO | os.PathLike:
    """Export data to a registered output format.

    Parameters
    ----------
    spectral_map : SpectralMap
        The dataset to export.
    out_file : BytesIO | os.PathLike | str | None, optional
        Destination file path or buffer. When ``None`` the writer may return
        an in-memory object (format-dependent).
    format_name : str | None, optional
        Registered output format name (e.g., "zarr", "matlab").
    format_params : dict | IOParams | None
        Format-specific writer parameters.

    Returns
    -------
    pl.DataFrame | BytesIO | os.PathLike
        Writer-specific return value.

    Raises
    ------
    ValueError
        If *format_name* is not a registered output format, or if the format
        does not support the supplied data type.
    """
    if format_params is None:
        format_params = {}

    if format_name is None:
        raise ValueError("format_name is required.")
    writer = OutputFormatRegistry.get_format_for_data(format_name, spectral_map)
    if writer is None:
        raise ValueError(f"Format {format_name} not supported.")

    format_params = writer.validate_params(format_params)

    if isinstance(spectral_map, SpectralMap) and SpectrumType.SPECTRAL_MAP not in writer.supported_types:
        raise ValueError(f"{writer.friendly_name} ({format_name}) does not support writing HSIs")
    if not hasattr(spectral_map, "images") and SpectrumType.SPECTRUM not in writer.supported_types:
        raise ValueError(f"{writer.friendly_name} ({format_name}) does not support writing Spectra")

    return writer.write(spectral_map, out_file=out_file, params=format_params)


def export_spectrum(sp: Spectrum, format: str = "csv", **kwargs):
    """Export a single :class:`Spectrum <ramappy.core.spectrum.Spectrum>`.

    Parameters
    ----------
    sp : Spectrum
        Spectrum to export.
    format : str, default="csv"
        Output format name.
    **kwargs
        Format-specific writer arguments.

    Returns
    -------
    Any
        Writer-specific return value.
    """

    if format == "csv":
        return write_csv_spectrum(sp, **kwargs)
    raise ValueError(f"Format {format} not supported.")


def export_grouped_spectra(sp: list[Spectrum], out_path: str, filename: str, format: str = "csv"):
    """Export in a zipped folder all spectra of a group of masks/images.

    Parameters
    ----------
    sp: list[Spectrum]
        List containing all the spectra to export
    format : str or None, default='csv'
        Currently only csv is supported

    Returns
    -------
    None
    """

    dir_path = os.path.join(out_path, filename)
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

    if os.path.exists(dir_path + ".zip"):
        os.remove(dir_path + ".zip")

    os.mkdir(dir_path)

    for s in sp:
        export_spectrum(s, format, out_file=os.path.join(dir_path, (s.name or "spectrum") + ".csv"))

    shutil.make_archive(dir_path, "zip", dir_path)
