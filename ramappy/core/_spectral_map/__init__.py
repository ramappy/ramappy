"""Internal implementation details for :class:`SpectralMap <ramappy.core.spectral_map.SpectralMap>`.

The public API remains in `ramappy.core.spectral_map`.
"""

from .functional import _SpectralMapFunctionalMixin
from .images import _SpectralMapImagesMixin
from .masks import _SpectralMapMasksMixin

__all__ = [
    "_SpectralMapFunctionalMixin",
    "_SpectralMapImagesMixin",
    "_SpectralMapMasksMixin",
]
