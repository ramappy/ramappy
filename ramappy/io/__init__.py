"""Public IO entrypoints for reading and exporting spectral data.

This module exposes high-level convenience functions around the input/output
registries defined in [ramappy.io.core][].
"""

from __future__ import annotations

from ..core.plugin_factory import discover_and_import_plugins
from .common import (
    export_grouped_spectra,
    export_spectra,
    export_spectrum,
    guess_file_type,
    read_file,
)
from .core import InputFormat, InputFormatRegistry, IOParams, OutputFormatRegistry
from .csv import read_csv, write_csv_spectrum
from .custom import read_own_custom, write_own_custom
from .hdf5_nexus import read_hdf5_nexus
from .horiba import read_horiba5
from .matlab import read_matlab, read_matlab_witec, write_matlab
from .pickle import read_pickle, write_pickle
from .renishaw import read_renishaw_wdf
from .witec import read_witec
from .zarr import read_zarr, write_zarr

discover_and_import_plugins("io")


__all__ = [
    "IOParams",
    # Registry access
    "InputFormat",
    "InputFormatRegistry",
    "OutputFormatRegistry",
    "export_grouped_spectra",
    "export_spectra",
    "export_spectrum",
    # High-level read/write helpers
    "guess_file_type",
    # Format-specific readers
    "read_csv",
    "read_file",
    "read_hdf5_nexus",
    "read_horiba5",
    "read_matlab",
    "read_matlab_witec",
    "read_own_custom",
    "read_pickle",
    "read_renishaw_wdf",
    "read_witec",
    "read_zarr",
    # Format-specific writers
    "write_csv_spectrum",
    "write_matlab",
    "write_own_custom",
    "write_pickle",
    "write_zarr",
]
