"""Denoising processing steps.

Includes spike removal and spatial/spectral smoothing methods.
"""

from .despike import despike
from .smooth_spatial import smooth_spatial
from .smooth_spectral import smooth_spectral
from .svd import smooth_svd

__all__ = [
    "despike",
    "smooth_spatial",
    "smooth_spectral",
    "smooth_svd",
]
