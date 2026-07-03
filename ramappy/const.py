"""Application constants for ramappy.

Exported names
--------------
- :class:`Axis <ramappy.const.Axis>`: index into the ``(n_pixels, n_spectral)`` data array.
- :class:`SpatialAxis <ramappy.const.SpatialAxis>`: index into a ``(height, width)`` spatial array.
- :class:`BlendModes <ramappy.const.BlendModes>`: compositing blend modes for image overlays.
- :data:`DEFAULT_COLORS <ramappy.const.DEFAULT_COLORS>`: ordered default color palette for masks/spectra.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

# Array-index helpers


class Axis(IntEnum):
    """Integer indices into the 2-D ``(n_pixels, n_spectral)`` data array.

    Examples
    --------
    >>> data.shape[Axis.PIXEL]     # number of pixels
    >>> data.shape[Axis.SPECTRAL]  # number of spectral channels
    """

    PIXEL = 0
    SPECTRAL = 1


class SpatialAxis(IntEnum):
    """Integer indices into a 2-D spatial ``(height, width)`` array."""

    VERTICAL = 0
    HORIZONTAL = 1


# Image compositing


class BlendModes(StrEnum):
    """Supported compositing blend modes for image overlays."""

    screen = "screen"
    add = "add"
    multiply = "multiply"


# Default color palette

DEFAULT_COLORS: list[str] = [
    "#EF4444",
    "#10B981",
    "#3B82F6",
    "#F59E0B",
    "#D946EF",
    "#06B6D4",
]
