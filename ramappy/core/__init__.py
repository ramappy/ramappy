"""Core public abstractions for ramappy.

Exports
-------
SpectralMap
    Main hyperspectral map container.
Spectrum
    One- or multi-pixel spectral container.
OrderedEntityMap
    Ordered mapping used for masks, images and spectra.
EntityTransaction
    Transaction helper for change tracking.
ProcessingStepConfig
    Serialized pipeline step configuration model.
"""

from .collections import EntityTransaction, OrderedEntityMap
from .metadata import ImageMetadata, MaskMetadata, SpectralMetadata
from .pipeline import ProcessingStepConfig
from .spectral_map import SpectralMap
from .spectrum import Spectrum

__all__ = [
    "EntityTransaction",
    "ImageMetadata",
    "MaskMetadata",
    "OrderedEntityMap",
    "ProcessingStepConfig",
    "SpectralMap",
    "SpectralMetadata",
    "Spectrum",
]
