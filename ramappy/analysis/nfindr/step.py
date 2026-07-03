"""N-FINDR endmember extraction analysis step."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field
from sklearn.decomposition import PCA

from ramappy.analysis.common import MAX_IMG_TO_SHOW
from ramappy.analysis.nfindr.nfindr import NFINDR
from ramappy.core import SpectralMap, Spectrum
from ramappy.core.images2d import Image2D
from ramappy.core.masks import Mask
from ramappy.core.pipeline import (
    ParamsAcceptMask,
    ParamsAcceptRoIX,
    ParamsSetResultId,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)
from ramappy.utils import color_generator, generate_key


class StepNFINDRParams(StepParams, ParamsAcceptMask, ParamsAcceptRoIX, ParamsSetResultId):
    """Parameters for the `nfindr` step."""

    n_endmembers: int = Field(..., gt=1)
    """Number of endmembers to identify."""

    transform_method: Literal["nnls", "barycentric"] = "nnls"
    """Coefficient estimation method used after simplex discovery."""

    normalize_concentrations: bool = False
    """If ``True``, normalize abundance coefficients per pixel."""

    gen_endmember_masks: bool = False
    """If ``True``, create one-pixel masks at endmember coordinates."""

    work_dtype: Literal["float32", "float64"] | None = None
    """Floating-point precision for the simplex search (``None`` preserves the dtype of the PCA-reduced data, ``"float64"`` forces double precision)."""


@pipeline_step(
    step_name="nfindr",
    params_validator=SingleModelParamsValidator(StepNFINDRParams),
    friendly_name="N-FINDR",
    step_category=StepClass.ANALYSIS,
)
def nfindr(
    spectral_map: SpectralMap,
    *,
    n_endmembers: int,
    transform_method: Literal["nnls", "barycentric"] = "nnls",
    normalize_concentrations: bool = False,
    gen_endmember_masks: bool = False,
    mask: str | Mask | None = None,
    roi_x: npt.ArrayLike | None = None,
    res_id: str | None = None,
    work_dtype: Literal["float32", "float64"] | None = None,
):
    """Extract endmembers with N-FINDR and generate abundance images.

    Parameters
    ----------
    spectral_map
        Input map to analyze.
    n_endmembers
        Number of endmembers to identify.
    transform_method
        Coefficient estimation method used after simplex discovery.
    normalize_concentrations
        If ``True``, normalize abundance coefficients per pixel.
    gen_endmember_masks
        If ``True``, create one-pixel masks at endmember coordinates.
    mask
        Optional spatial mask limiting processed pixels.
    roi_x
        Optional spectral ROI for preprocessing.
    res_id
        Optional explicit result identifier.
    work_dtype
        Floating-point precision for the simplex search (``None`` preserves
        the dtype of the PCA-reduced data, ``"float64"`` forces double
        precision).
    """

    idxs, x_idx, pixel_idx = spectral_map.get_indices(mask, roi_x)
    roi_x = spectral_map.adapt_roi_x(roi_x) if roi_x is not None else spectral_map.roi_x
    intensities = np.asarray(spectral_map.data)[idxs]

    pca = PCA(n_components=n_endmembers - 1, random_state=0)
    intensities_reduced = pca.fit_transform(intensities)

    nf = NFINDR(n_endmembers=n_endmembers, random_state=0, work_dtype=work_dtype)
    nfindr_coefs = nf.fit_transform(intensities_reduced, transform_method)

    if normalize_concentrations:
        nfindr_coefs /= nfindr_coefs.sum(axis=1, keepdims=True)

    # Extract pure spectra (the endmembers) from the original intensities.
    endmembers_spectra = np.asarray(intensities)[nf.endmember_indices_, ...].T

    group_key = res_id or generate_key()
    img_ids: list[str] = []

    masked_img_template = None
    if not isinstance(pixel_idx, slice):
        masked_img_template = np.empty(spectral_map.map_shape, dtype=nfindr_coefs.dtype)
        mask_obj = spectral_map.get_mask(mask) if isinstance(mask, str) else None
        if mask_obj is not None:
            mask_2d = mask_obj.get_2Dmask(invert=True)
        else:
            mask_2d = np.zeros(spectral_map.map_shape, dtype=bool)
        masked_img_template = np.ma.masked_array(
            masked_img_template,
            mask=mask_2d,
            fill_value=np.nan,
        )

    data_unit = spectral_map.data_unit

    for i in reversed(range(n_endmembers)):
        if masked_img_template is not None:
            img = masked_img_template.copy()
            img.ravel()[pixel_idx] = nfindr_coefs[:, i]
        else:
            img = nfindr_coefs[:, i].reshape(spectral_map.map_shape)

        img_key = f"{group_key}_{i}"
        spectral_map.images[img_key] = Image2D(
            data=img,
            data_rules=None,
            viz_rules={"cmap": color_generator(i), "color2": "k"},
            name=f"Endmember {i + 1}",
            spectrum=Spectrum(
                data=endmembers_spectra[:, i],
                x=spectral_map.x[x_idx],
                roi_x=roi_x,
                x_axis_unit=spectral_map.x_axis_unit,
                data_unit=data_unit,
                ignore_sort=True,
                name=f"Endmember {i + 1}",
            ),
            parent_group=group_key,
            locked=True,
            visible=i < MAX_IMG_TO_SHOW,
        )
        img_ids.append(img_key)

    spectral_map.images_group[group_key] = spectral_map.new_images_group(img_ids, name="N-FINDR")

    if gen_endmember_masks:
        masks_group_key = f"{group_key}_mask"
        masks_ids: list[str] = []
        for i in reversed(range(n_endmembers)):
            img_viz = spectral_map.images[img_ids[i]].viz_rules
            mask_key = f"{masks_group_key}_{i}"
            spectral_map.masks[mask_key] = spectral_map.new_mask(
                idxs=[nf.endmember_indices_[i]],
                name=f"Endmember {i + 1} mask",
                color=(img_viz.cmap if img_viz is not None else None),
                parent_group=masks_group_key,
                editable=False,
                visible=i < MAX_IMG_TO_SHOW,
            )
            masks_ids.append(mask_key)
        spectral_map.masks_group[masks_group_key] = spectral_map.new_masks_group(
            masks_ids,
            spectrum_agg=None,
            name="N-FINDR endmembers",
        )
