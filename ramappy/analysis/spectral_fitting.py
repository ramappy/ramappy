"""Reference-spectrum fitting analysis step."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic_extra_types.color import Color
from sklearn.preprocessing import minmax_scale, normalize, scale

from ramappy.core import SpectralMap, Spectrum
from ramappy.core.images2d import Image2D, VizRules
from ramappy.core.pipeline import (
    ParamsAcceptMask,
    ParamsAcceptRoIX,
    ParamsSetResultId,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)
from ramappy.utils import (
    generate_key,
    pearson_correlation_coefficient,
    r2_score,
    spectral_angle_mapper,
    spectral_information_divergence,
)


class StepSpectralFittingParams(StepParams, ParamsAcceptMask, ParamsAcceptRoIX, ParamsSetResultId):
    """Parameters for `spectral_fitting`."""

    ref_spectra: str
    """Reference spectrum id."""

    method: Literal["SAM", "SCM", "SID", "pearson", "r2_score"] = "SAM"
    """Similarity/divergence metric (`"SAM"`, `"SCM"`, `"SID"`, `"pearson"`, `"r2_score"`)."""

    norm: Literal["l1", "l2", "max", "scale", "minmax_scale", "frobenius"] | None = None
    """Optional normalization method applied to both the reference and pixel spectra before scoring."""


@pipeline_step(
    step_name="spectral_fitting",
    params_validator=SingleModelParamsValidator(StepSpectralFittingParams),
    friendly_name="Spectral Fitting",
    step_category=StepClass.ANALYSIS,
)
def spectral_fitting(
    spectral_map: SpectralMap | Spectrum,
    *,
    ref_spectra: str | Spectrum,
    method: Literal["SAM", "SCM", "SID", "pearson", "r2_score"] = "SAM",
    mask: str | None = None,
    res_id: str | None = None,
    **kwargs,
):
    """Score each pixel spectrum against a reference spectrum.

    Parameters
    ----------
    spectral_map
        Input map or spectrum collection.
    ref_spectra
        Reference spectrum id or explicit :class:`Spectrum <ramappy.core.Spectrum>`.
    method
        Similarity/divergence metric (`"SAM"`, `"SCM"`, `"SID"`, `"pearson"`, `"r2_score"`).
    mask
        Optional spatial mask identifier.
    res_id
        Optional output image key.
    **kwargs
        Additional options, currently including ``norm`` for pre-normalization.

    Raises
    ------
    ValueError
        If an unsupported normalization option is provided.
    """
    if not isinstance(spectral_map, SpectralMap):
        raise TypeError("spectral_map must be a SpectralMap instance")

    if isinstance(ref_spectra, str):
        ref_spectra, is_spectrum = spectral_map.get_reference_spectrum(ref_spectra)
    else:
        is_spectrum = True

    if is_spectrum:
        ref_spectra = spectral_map.align_external_spectrum(ref_spectra, overlap=True)
    common_roi_x = ref_spectra.roi_x

    idxs, _x_idx, pixel_idx = spectral_map.get_indices(mask, common_roi_x)
    intensities = np.asarray(spectral_map.data)[idxs]

    ref_intensities = ref_spectra.data

    norm = kwargs.get("norm")
    if norm is not None:
        if norm in {"l1", "l2", "max"}:
            intensities = normalize(intensities, axis=1, norm=norm)
            ref_intensities = normalize(ref_intensities, axis=1, norm=norm)
        elif norm == "scale":
            intensities = scale(intensities, axis=1)
            ref_intensities = scale(ref_intensities, axis=1)
        elif norm == "minmax_scale":
            intensities = minmax_scale(intensities, axis=1)
            ref_intensities = minmax_scale(ref_intensities, axis=1)
        elif norm == "frobenius":
            intensities /= np.linalg.norm(intensities, "fro")
            ref_intensities /= np.linalg.norm(ref_intensities, "fro")
        else:
            raise ValueError("No valid norm selected.")

    invert_cmap = True
    method_name: str = method
    if method in {"SAM", "SCM"}:
        scores = spectral_angle_mapper(intensities, ref_intensities, centre=(method == "SCM"))
    elif method == "SID":
        scores = spectral_information_divergence(intensities, ref_intensities)
    elif method == "pearson":
        scores = pearson_correlation_coefficient(intensities, ref_intensities)
        invert_cmap = False
        method_name = "Pearson correlation"
    elif method == "r2_score":
        scores = r2_score(intensities, ref_intensities)
        invert_cmap = False
        method_name = "R²"
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if not isinstance(pixel_idx, slice):
        img = np.empty(spectral_map.map_shape, dtype=scores.dtype)
        img.ravel()[pixel_idx] = scores.flatten()
        mask_obj = spectral_map.get_mask(mask) if isinstance(mask, str) else None
        if mask_obj is not None:
            mask_2d = mask_obj.get_2Dmask(invert=True)
        else:
            mask_2d = np.zeros(spectral_map.map_shape, dtype=bool)
        img = np.ma.masked_array(img, mask=mask_2d, fill_value=np.nan)
    else:
        img = scores.reshape(spectral_map.map_shape)

    key = res_id or generate_key()
    from ramappy.core.images2d.rules import COLORMAPS_NAMES  # local import to avoid circularity

    raw_color = ref_spectra.color
    fallback_cmap: Any = getattr(COLORMAPS_NAMES, "turbo", None)
    if raw_color is not None:
        try:
            cmap: Any = COLORMAPS_NAMES[raw_color]  # type: ignore[index]
        except KeyError:
            try:
                cmap = Color(raw_color)
            except Exception:
                cmap = fallback_cmap
    else:
        cmap = fallback_cmap
    spectral_map.images[key] = Image2D(
        data=img,
        name=f"{ref_spectra.name} ({method_name})",
        viz_rules=VizRules(cmap=cmap, color2=Color("black"), invert=invert_cmap),
        data_rules=None,
        locked=True,
    )
