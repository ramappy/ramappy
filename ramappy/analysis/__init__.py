"""Analysis step namespace.

This package exposes analysis operations used by `ramappy` pipelines.
Public symbols are re-exported from root modules for convenient imports.
"""

from .cluster import StepClusterParams, cluster
from .mcr import StepMCRParams, mcr
from .nfindr import StepNFINDRParams, nfindr
from .pca import StepPrincipalComponentsParams, principal_components
from .spectral_fitting import StepSpectralFittingParams, spectral_fitting
from .substrate import StepCellSegmentationParams, substrate_extraction

__all__ = [
    "StepCellSegmentationParams",
    "StepClusterParams",
    "StepMCRParams",
    "StepNFINDRParams",
    "StepPrincipalComponentsParams",
    "StepSpectralFittingParams",
    "cluster",
    "mcr",
    "nfindr",
    "principal_components",
    "spectral_fitting",
    "substrate_extraction",
]
