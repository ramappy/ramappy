"""Physical units for ramappy.

This module provides the unit vocabulary shared across the whole library:

- :class:`ramappy.units.Unit`: string enum of all supported physical units.
- :class:`ramappy.units.Quantity`: string enum of measurable physical quantities.
- :class:`ramappy.units.QuantityUnit`: a ``(quantity, unit)`` pair used to annotate spectral and data axes.
- Pre-built defaults (:attr:`DEFAULT_SPECTRAL_QUANTITY_UNIT <ramappy.units.DEFAULT_SPECTRAL_QUANTITY_UNIT>`, :attr:`DEFAULT_DATA_QUANTITY_UNIT <ramappy.units.DEFAULT_DATA_QUANTITY_UNIT>`).
"""

from __future__ import annotations

import warnings
from enum import StrEnum

# Physical unit enum


class Unit(StrEnum):
    """Supported physical units.

    Values are Unicode strings that render as conventional notation,
    e.g., ``Unit.CM_1 == 'cm⁻¹'``.
    """

    # Spatial / length
    MICROMETER = "μm"
    NANOMETER = "nm"
    MILLIMETER = "mm"
    PIXEL = "px"

    # Spectral / energy
    CM_1 = "cm⁻¹"
    ELECTRONVOLT = "eV"
    MILLI_ELECTRONVOLT = "meV"
    JOULE = "J"

    # Intensity / data
    ARBITRARY_UNITS = "a.u."
    COUNTS = "counts"
    COUNTS_PER_SECOND = "counts/s"
    PERCENT = "%"

    # Time
    SECOND = "s"

    # Angular
    DEGREE = "°"
    RADIANS = "rad"

    # Frequency / temperature / other
    HERTZ = "Hz"
    KELVIN = "K"

    UNDEFINED = ""

    @classmethod
    def _missing_(cls, value: object) -> Unit:
        """Flexibly match legacy / alternate spellings."""
        if isinstance(value, str):
            normalized = value.lower()
            if normalized in {"1/cm", "cm-1", "cm^-1"}:
                return cls.CM_1
            if normalized in {"um", "micrometer", "micron"}:
                return cls.MICROMETER
            if normalized in {"nm", "nanometer"}:
                return cls.NANOMETER
            # Legacy ASCII spelling for arbitrary units.
            if normalized in {"au", "a.u"}:
                return cls.ARBITRARY_UNITS
            for member in cls:
                if member.value == value:
                    return member
        warnings.warn(f"Unknown unit {value!r}; falling back to UNDEFINED", stacklevel=3)
        return cls.UNDEFINED


# Physical quantity enum


class Quantity(StrEnum):
    """Measurable physical quantities.

    Used together with :class:`ramappy.units.Unit` to form a :class:`ramappy.units.QuantityUnit` pair.
    """

    # --- Spectral axis ---
    WAVENUMBER = "wavenumber"
    RAMANSHIFT = "Raman shift"
    WAVELENGTH = "wavelength"
    ENERGY = "energy"
    FREQUENCY = "frequency"
    UNCALIBRATED = "uncalibrated"

    # --- Data / intensity axis ---
    INTENSITY = "intensity"
    RELATIVE_INTENSITY = "relative intensity"
    ABSORBANCE = "absorbance"
    REFLECTANCE = "reflectance"
    TRANSMITTANCE = "transmittance"

    # --- Spatial ---
    LENGTH = "length"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"

    # --- Other ---
    TEMPERATURE = "temperature"
    TIME = "time"

    UNDEFINED = ""

    @classmethod
    def _missing_(cls, value: object) -> Quantity:
        if isinstance(value, str):
            normalized = value.lower()
            for member in cls:
                if member.value == normalized:
                    return member
        return cls.UNDEFINED


class QuantityUnit:
    """A ``(quantity, unit)`` pair describing a physical axis.

    Parameters
    ----------
    quantity : Quantity | str
        The measurable quantity (e.g., ``Quantity.WAVENUMBER``).
    unit : Unit | str
        The corresponding physical unit (e.g., ``Unit.CM_1``).

    Examples
    --------
    >>> qu = QuantityUnit(Quantity.WAVENUMBER, Unit.CM_1)
    >>> str(qu)
    'wavenumber [cm⁻¹]'
    >>> qu == QuantityUnit("wavenumber", "cm⁻¹")
    True
    """

    __slots__ = ("quantity", "unit")

    def __init__(self, quantity: Quantity | str, unit: Unit | str) -> None:
        self.quantity = Quantity(quantity) if isinstance(quantity, str) else quantity
        self.unit = Unit(unit) if isinstance(unit, str) else unit

    # human-readable representation

    def __repr__(self) -> str:
        return f"QuantityUnit({self.quantity!r}, {self.unit!r})"

    def __str__(self) -> str:
        return f"{self.quantity} [{self.unit}]"

    # equality & hashing (needed for use in sets/dicts and comparisons)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, QuantityUnit):
            return self.quantity == other.quantity and self.unit == other.unit
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.quantity, self.unit))

    # serialization helpers

    def to_tuple(self) -> tuple[str, str]:
        """Return ``(quantity_str, unit_str)``, used for Zarr serialization."""
        return (str(self.quantity), str(self.unit))

    def to_dict(self) -> dict[str, str]:
        """Return ``{"quantity": ..., "unit": ...}``."""
        return {"quantity": str(self.quantity), "unit": str(self.unit)}


# Predefined defaults

DEFAULT_SPECTRAL_QUANTITY_UNIT: QuantityUnit = QuantityUnit(Quantity.WAVENUMBER, Unit.CM_1)
"""Default spectral-axis unit: Raman shift in cm⁻¹."""

DEFAULT_DATA_QUANTITY_UNIT: QuantityUnit = QuantityUnit(Quantity.INTENSITY, Unit.ARBITRARY_UNITS)
"""Default data (intensity) axis unit: intensity in a.u."""


# Type aliases for quantity-unit parameters

type_spectral_quantityunit = tuple[Quantity, Unit] | QuantityUnit | None
"""Type accepted for spectral-axis unit parameters."""

type_intensity_quantityunit = tuple[Quantity, Unit] | QuantityUnit | None
"""Type accepted for data/intensity-axis unit parameters."""

# Canonical alias for spectral-map data values.
type_data_quantityunit = type_intensity_quantityunit
