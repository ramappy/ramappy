"""2D image helpers for ramappy.

Public API:

- :class:`Image2D <ramappy.core.images2d.Image2D>`: 2-D image data container.
- :class:`Image2DGroup <ramappy.core.images2d.Image2DGroup>`: ordered collection of related images.
- :class:`Image2DRenderer <ramappy.core.images2d.Image2DRenderer>`: renders float data to PIL images.
- :class:`DataRules <ramappy.core.images2d.rules.DataRules>` / :class:`VizRules <ramappy.core.images2d.rules.VizRules>`: Pydantic models for image rules.
- :class:`SpatialGrid <ramappy.core.images2d.SpatialGrid>`: physical spatial grid for coordinate transforms.
"""

from __future__ import annotations

from .group import Image2DGroup
from .image import Image2D
from .renderer import Image2DRenderer
from .rules import COLORMAPS_NAMES, ColorType, DataRules, VizRules
from .spatial import SpatialGrid

__all__ = [
    "COLORMAPS_NAMES",
    "ColorType",
    "DataRules",
    "Image2D",
    "Image2DGroup",
    "Image2DRenderer",
    "SpatialGrid",
    "VizRules",
]
