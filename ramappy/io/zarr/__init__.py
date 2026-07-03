"""IO for Zarr files (RamApp-specific hierarchy)."""

from __future__ import annotations

import zarr

# Note: ZipStore support is not formally guaranteed by zarrs; it works today
# because zarr-python falls back to the BatchedCodecPipeline for unsupported
# stores, but this behaviour may change in future zarrs releases.
import zarrs  # noqa: F401, registers ZarrsCodecPipeline in zarr's registry

from .reader import ZarrFormatInputParams, read_zarr, read_zarr_metadata_only
from .writer import ZarrFormatOutputParams, write_zarr

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

__all__ = [
    "ZarrFormatInputParams",
    "ZarrFormatOutputParams",
    "read_zarr",
    "read_zarr_metadata_only",
    "write_zarr",
]
