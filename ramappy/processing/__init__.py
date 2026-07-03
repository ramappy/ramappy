"""Processing step namespace.

This package exposes spectral and spatial processing operations used by
`ramappy` pipelines (baseline correction, normalization, resampling,
denoising, and geometric transforms).

Notes
-----
Public symbols are re-exported from submodules for convenient imports.
"""

from .baseline import correct_baseline
from .crop_spectral import crop_spectral
from .denoising import despike, smooth_spatial, smooth_spectral, smooth_svd
from .geometric import crop_spatial, flip, rotate
from .normalize import normalize_intensities
from .resample import resample

__all__ = [
    "correct_baseline",
    "crop_spatial",
    "crop_spectral",
    "despike",
    "flip",
    "normalize_intensities",
    "resample",
    "rotate",
    "smooth_spatial",
    "smooth_spectral",
    "smooth_svd",
]
