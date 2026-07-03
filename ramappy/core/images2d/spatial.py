"""Physical spatial grid for hyperspectral maps."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ramappy import units


class SpatialGrid(BaseModel):
    """Physical spatial grid describing the measurement geometry of a map.

    Stores the pixel size and origin in physical coordinates, allowing
    round-trip conversion between pixel indices ``(row, col)`` and physical
    coordinates ``(y, x)``.

    Examples
    --------
    Create a grid from a known pixel size in micrometers:

    >>> grid = SpatialGrid.from_pixel_size(0.5, spatial_unit=units.Unit.MICROMETER)
    >>> grid.pixel_to_physical(0, 0)
    (0.0, 0.0)
    >>> grid.pixel_to_physical(1, 2)
    (0.5, 1.0)

    Create from explicit stage-coordinate arrays (e.g., from a WiRE file):

    >>> grid = SpatialGrid.from_grid_coords(x_coords, y_coords, spatial_unit="μm")
    """

    model_config = ConfigDict(validate_assignment=True)

    pixel_size_y: float = Field(default=1.0, gt=0, description="Row pixel size in spatial_unit.")
    pixel_size_x: float = Field(default=1.0, gt=0, description="Column pixel size in spatial_unit.")
    origin_y: float = Field(default=0.0, description="Physical Y of the top-left pixel centre.")
    origin_x: float = Field(default=0.0, description="Physical X of the top-left pixel centre.")
    spatial_unit: str = Field(default=units.Unit.PIXEL, description="Physical unit for pixel sizes and origin.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_unit(cls, data: object) -> object:
        """Normalise legacy unit strings (e.g., '1/cm' to 'cm⁻¹')."""
        if isinstance(data, dict) and "spatial_unit" in data:
            raw = data["spatial_unit"]
            if isinstance(raw, str):
                data = {**data, "spatial_unit": str(units.Unit(raw))}
        return data

    # Convenience constructors

    @classmethod
    def from_pixel_size(
        cls,
        pixel_size: float,
        *,
        pixel_size_x: float | None = None,
        spatial_unit: str = units.Unit.PIXEL,
        origin_y: float = 0.0,
        origin_x: float = 0.0,
    ) -> SpatialGrid:
        """Create a grid from a single (square) or anisotropic pixel size.

        Parameters
        ----------
        pixel_size : float
            Row (Y) pixel size. Used for both axes when *pixel_size_x* is
            ``None``.
        pixel_size_x : float | None
            Column (X) pixel size. If ``None``, defaults to *pixel_size*
            (square pixels).
        spatial_unit : str
            Physical unit string (default: pixel).
        origin_y, origin_x : float
            Physical coordinates of the top-left pixel (default: 0, 0).

        Returns
        -------
        SpatialGrid
        """
        return cls(
            pixel_size_y=pixel_size,
            pixel_size_x=pixel_size_x if pixel_size_x is not None else pixel_size,
            spatial_unit=spatial_unit,
            origin_y=origin_y,
            origin_x=origin_x,
        )

    @classmethod
    def from_grid_coords(
        cls,
        x_coords: npt.ArrayLike,
        y_coords: npt.ArrayLike,
        spatial_unit: str,
    ) -> SpatialGrid:
        """Create a grid from explicit stage-coordinate arrays.

        Pixel sizes are inferred from the spacing between successive unique
        coordinate values. Works for regular grids (the common case for
        raster scans).

        Parameters
        ----------
        x_coords : array_like, shape (n_pixels,)
            Flat array of X (column) physical coordinates, one per pixel.
        y_coords : array_like, shape (n_pixels,)
            Flat array of Y (row) physical coordinates, one per pixel.
        spatial_unit : str
            Physical unit of the coordinate values.

        Returns
        -------
        SpatialGrid
        """
        x = np.asarray(x_coords, dtype=float)
        y = np.asarray(y_coords, dtype=float)

        origin_x = float(x.min())
        origin_y = float(y.min())

        unique_x = np.unique(x)
        unique_y = np.unique(y)

        pixel_size_x = float(np.median(np.diff(np.sort(unique_x)))) if len(unique_x) > 1 else 1.0

        pixel_size_y = float(np.median(np.diff(np.sort(unique_y)))) if len(unique_y) > 1 else 1.0

        # Pixel sizes must be positive.
        pixel_size_x = abs(pixel_size_x) if pixel_size_x != 0.0 else 1.0
        pixel_size_y = abs(pixel_size_y) if pixel_size_y != 0.0 else 1.0

        return cls(
            pixel_size_y=pixel_size_y,
            pixel_size_x=pixel_size_x,
            origin_y=origin_y,
            origin_x=origin_x,
            spatial_unit=spatial_unit,
        )

        # Coordinate transforms

    def pixel_to_physical(self, row: int | float, col: int | float) -> tuple[float, float]:
        """Convert pixel ``(row, col)`` to physical ``(y, x)`` coordinates.

        Parameters
        ----------
        row : int | float
            Row index (0-based, top-to-bottom).
        col : int | float
            Column index (0-based, left-to-right).

        Returns
        -------
        tuple[float, float]
            ``(y, x)`` in `spatial_unit`.
        """
        y = self.origin_y + row * self.pixel_size_y
        x = self.origin_x + col * self.pixel_size_x
        return float(y), float(x)

    def physical_to_pixel(self, y: float, x: float) -> tuple[int, int]:
        """Convert physical ``(y, x)`` to the nearest pixel ``(row, col)``.

        Parameters
        ----------
        y : float
            Physical Y coordinate in `spatial_unit`.
        x : float
            Physical X coordinate in `spatial_unit`.

        Returns
        -------
        tuple[int, int]
            ``(row, col)`` pixel indices (may be out-of-bounds).
        """
        row = round((y - self.origin_y) / self.pixel_size_y)
        col = round((x - self.origin_x) / self.pixel_size_x)
        return row, col

    def rotate_90(self, n: int = 1) -> SpatialGrid:
        """Return a new :class:`SpatialGrid <ramappy.core.images2d.SpatialGrid>` rotated by *n* * 90 degrees.

        Rotation swaps pixel sizes and negates/shifts origin as appropriate.
        For simplicity this implementation only swaps x/y pixel sizes,
        sufficient for square-shaped maps.

        Parameters
        ----------
        n : int
            Number of 90° clockwise rotations (default 1).

        Returns
        -------
        SpatialGrid
        """
        if n % 2 == 0:
            return self.model_copy()
        return self.model_copy(
            update={
                "pixel_size_y": self.pixel_size_x,
                "pixel_size_x": self.pixel_size_y,
                "origin_y": self.origin_x,
                "origin_x": self.origin_y,
            }
        )

    # Serialization helpers

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON/Zarr attribute storage."""
        return self.model_dump()
