from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import AliasChoices, Field

from ramappy.core.pipeline import ParamsAcceptRoIX, SingleModelParamsValidator, StepClass, StepParams, pipeline_step
from ramappy.core.spectral_map import SpectralMap
from ramappy.core.spectrum import Spectrum


class StepSubtractSubstrateParams(StepParams):
    """Parameters for `subtract_substrate`."""

    aggregation: Literal["mean95", "mean", "median", "min", "max"] = Field(default="mean", alias="how")
    """Aggregation method to compute the substrate spectrum from the map (default: ``"mean"``)."""


@pipeline_step(
    step_name="subtract_substrate",
    params_validator=SingleModelParamsValidator(StepSubtractSubstrateParams),
    friendly_name="Subtract Substrate",
    step_category=StepClass.UTILITY,
)
def subtract_substrate(
    spectral_map: SpectralMap, aggregation: Literal["mean95", "mean", "median", "min", "max"]
) -> None:
    """Subtract the aggregated substrate spectrum from every pixel.

    The substrate spectrum is computed by aggregating all pixel spectra in the
    map using the specified method, then subtracted in place.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    aggregation : {'mean95', 'mean', 'median', 'min', 'max'}
        Aggregation method used to compute the substrate spectrum.
    """
    spectral_map.math(operand="substrate", agg_operand=aggregation, op="sub")


class StepSubtractSpectraParams(StepParams):
    """Parameters for `subtract_spectra`."""

    ref_spectra: str
    """Identifier of the reference spectrum to subtract."""


@pipeline_step(
    step_name="subtract_spectra",
    params_validator=SingleModelParamsValidator(StepSubtractSpectraParams),
    friendly_name="Subtract Spectra",
    step_category=StepClass.UTILITY,
)
def subtract_spectra(spectral_map: SpectralMap, *, ref_spectra: str):
    """Subtract an external reference spectrum from every pixel in the map.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    ref_spectra : str
        Identifier of the external spectrum to subtract.
    """
    spectral_map.math(operand=ref_spectra, op="sub")


class StepMathOperationParams(StepParams, ParamsAcceptRoIX):
    """Parameters for `math_operation`."""

    B: str | float = 0
    """Identifier of the second operand (mask or spectrum) or a numeric value. If a numeric value is provided, it is treated as a constant to subtract from each pixel."""

    A: str | None = None
    """Identifier of the first operand (mask or spectrum) (if ``None``, the operation is applied to the whole map)."""

    aggregate_B: str = "mean"
    """Aggregation method to apply to the second operand (if it is a mask) before performing the operation."""

    operation: Literal["sub", "add"] = "sub"
    """Operation to perform on the operands (`"sub"` for subtraction, `"add"` for addition)."""


@pipeline_step(
    step_name="math_operation",
    params_validator=SingleModelParamsValidator(StepMathOperationParams),
    friendly_name="Math Operation",
    step_category=StepClass.UTILITY,
)
def math_operation(
    spectral_map: SpectralMap,
    *,
    B: str | float,
    A: str,
    aggregate_B: str,
    operation: Literal["sub", "add"],
    roi_x: list,
):
    """Apply an arithmetic operation between pixel spectra and an operand.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    B : str or float
        Identifier of the second operand (mask or spectrum) or a numeric constant.
    A : str
        Identifier of the first operand (mask). If ``None``, the full map is used.
    aggregate_B : str
        Aggregation method for *B* when it refers to a mask.
    operation : {'sub', 'add'}
        Arithmetic operation to perform.
    roi_x : list
        Spectral region of interest to restrict the operation to.
    """
    spectral_map.math(operand=B, agg_operand=aggregate_B, mask=A, op=operation, roi_x=roi_x)


class StepSubtractScaledSpectrumParams(StepParams):
    """Subtract a spectrum from each pixel using a per-pixel coefficient map.

    This step computes:
        :math:`corrected(x, y, lambda) = cube(x, y, lambda) - coeff(x, y) * spectrum(lambda)`

    Notes
    -----
    - Raw abundance is used by default (no normalization).
    - Spectrum ids can refer to: external spectra, mask spectra, or image-attached spectra.
      When ids clash between masks and images, image-attached spectra are preferred.
    """

    abundance_image_id: str = Field(
        validation_alias=AliasChoices("abundance_image_id", "image_id", "abundanceImageId", "imageId"),
        description="Identifier of the image providing per-pixel coefficients.",
    )
    spectrum_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("spectrum_id", "ref_spectra", "ref_spectrum", "spectrumId"),
        description="Identifier of the reference spectrum to subtract. If ``None``, the spectrum attached to the abundance image is used.",
    )
    scale_abundance: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "scale_abundance",
            "scaleAbundance",
            "normalize_abundance",
            "normalizeAbundance",
            "scale",
        ),
        description="Whether to linearly scale the abundance map to [0, 1] before applying it.",
    )
    invert_abundance: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "invert_abundance",
            "invertAbundance",
        ),
        description="Whether to take the opposite of the abundance map before applying it, effectively inverting the scaling.",
    )


def _get_coeff_map(
    spectral_map: SpectralMap, *, abundance_image_id: str, scale_to_unit: bool, invert_abundance: bool
) -> np.ndarray:
    if abundance_image_id not in spectral_map.images:
        raise ValueError(f"Image '{abundance_image_id}' not found")

    img = spectral_map.images[abundance_image_id]
    if img.data is None:
        raise ValueError(f"Image '{abundance_image_id}' has no numeric data")

    coeff = np.asarray(img.data)
    if coeff.ndim != 2:
        raise ValueError("Coefficient image must be a 2-D array")
    if coeff.shape != spectral_map.map_shape:
        raise ValueError(
            f"Coefficient image shape does not match cube spatial shape: expected {spectral_map.map_shape}, got {coeff.shape}"
        )
    if scale_to_unit:
        coeff = coeff.astype(np.float64, copy=False)
        coeff_min = np.nanmin(coeff)
        coeff_max = np.nanmax(coeff)
        denom = coeff_max - coeff_min
        coeff = np.zeros_like(coeff) if denom == 0 else (coeff - coeff_min) / denom
    if invert_abundance:
        coeff = -coeff
    return coeff


def _resolve_spectrum_to_subtract(
    spectral_map: SpectralMap,
    *,
    abundance_image_id: str,
    spectrum_id: str | None,
) -> Spectrum:
    img = spectral_map.images[abundance_image_id]

    if spectrum_id is None:
        if img.spectrum is None:
            raise ValueError(
                "No spectrum_id provided and the selected image has no attached spectrum. "
                "Select a spectrum explicitly or choose an image with an associated spectrum."
            )
        return img.spectrum

    spectrum, needs_alignment = spectral_map.get_reference_spectrum(spectrum_id, agg_method="mean", roi_x=None)
    if needs_alignment:
        spectrum = spectral_map.align_external_spectrum(spectrum, overlap="full")
    return spectrum


def _spectrum_to_vector(spectrum: Spectrum, *, expected_len: int) -> np.ndarray:
    y = np.asarray(spectrum.data)
    if y.ndim == 2:
        # Most Spectrum objects store spectral data as (1, bands).
        y = y[0]
    if y.ndim != 1:
        raise ValueError("Expected a 1-D spectrum data vector")
    if y.shape[0] != expected_len:
        raise ValueError(f"Spectrum length mismatch: expected {expected_len}, got {y.shape[0]}")
    return y


@pipeline_step(
    step_name="subtract_scaled_spectrum",
    params_validator=SingleModelParamsValidator(StepSubtractScaledSpectrumParams),
    friendly_name="Subtract Scaled Spectrum",
    step_category=StepClass.PROCESSING,
)
def subtract_scaled_spectrum(
    spectral_map: SpectralMap,
    *,
    abundance_image_id: str,
    spectrum_id: str | None,
    scale_abundance: bool = True,
    invert_abundance: bool = False,
) -> None:
    """Subtract a scaled reference spectrum from each pixel.

    The correction applied is:

    .. math::

       \\text{corrected}(x, y, \\lambda) =
       \\text{cube}(x, y, \\lambda) - \\text{coeff}(x, y) \\cdot \\text{spectrum}(\\lambda)

    where ``coeff`` comes from the abundance image and ``spectrum`` is the
    reference spectrum (from the image or an explicit identifier).

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    abundance_image_id : str
        Identifier of the image providing per-pixel coefficients.
    spectrum_id : str or None
        Identifier of the reference spectrum to subtract. If ``None``, the
        spectrum attached to the abundance image is used.
    scale_abundance : bool, optional
        If ``True``, linearly scale the abundance map to ``[0, 1]``.
    invert_abundance : bool, optional
        If ``True``, negate the abundance values before applying them.
    """
    coeff = _get_coeff_map(
        spectral_map,
        abundance_image_id=abundance_image_id,
        scale_to_unit=scale_abundance,
        invert_abundance=invert_abundance,
    )
    spectrum = _resolve_spectrum_to_subtract(
        spectral_map, abundance_image_id=abundance_image_id, spectrum_id=spectrum_id
    )
    y = _spectrum_to_vector(spectrum, expected_len=spectral_map.x_size)

    cube = spectral_map.cube
    coeff = coeff.astype(cube.dtype, copy=False)
    y = y.astype(cube.dtype, copy=False)

    # Broadcast: (H, W, 1) * (1, 1, B) => (H, W, B)
    correction = coeff[..., np.newaxis] * y[np.newaxis, np.newaxis, :]

    corrected = cube - correction
    # `corrected` is a cube with shape (H, W, bands). Flatten back to
    # the internal data shape (pixels, bands).
    spectral_map.data = corrected.reshape(-1, corrected.shape[-1])


__all__ = [
    "StepMathOperationParams",
    "StepSubtractScaledSpectrumParams",
    "StepSubtractSpectraParams",
    "StepSubtractSubstrateParams",
    "math_operation",
    "subtract_scaled_spectrum",
    "subtract_spectra",
    "subtract_substrate",
]
