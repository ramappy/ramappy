"""Principal component analysis step for spectral maps."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import field_validator
from sklearn.decomposition import PCA

from ramappy import const
from ramappy.analysis.common import MAX_IMG_TO_SHOW, validate_pca_n_components
from ramappy.core import SpectralMap, Spectrum
from ramappy.core.images2d import Image2D
from ramappy.core.pipeline import (
    ParamsAcceptMask,
    ParamsAcceptRoIX,
    ParamsSetResultId,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)
from ramappy.processing.normalize import StepNormalizeParams, _normalize_intensities
from ramappy.utils import color_generator, generate_key


class StepPrincipalComponentsParams(StepParams, ParamsAcceptMask, ParamsAcceptRoIX, ParamsSetResultId):
    """Parameters for `principal_components`."""

    pca_n_components: int | float | None = 0
    """Number of components or explained variance ratio accepted by `sklearn.decomposition.PCA`."""

    compute_residuals: bool = False
    """If ``True``, store a residual image after reconstruction."""

    normalize_kwargs: StepNormalizeParams | None = None
    """Optional normalization arguments applied before PCA."""

    @field_validator("pca_n_components", mode="before")
    @classmethod
    def _validate_pca_n_components(cls, value: int | float | None) -> int | float | None:
        return validate_pca_n_components(value)


@pipeline_step(
    step_name="pca",
    params_validator=SingleModelParamsValidator(StepPrincipalComponentsParams),
    friendly_name="PCA",
    step_category=StepClass.ANALYSIS,
)
def principal_components(
    spectral_map: SpectralMap,
    *,
    mask: str | None = None,
    roi_x: npt.ArrayLike | None = None,
    pca_n_components: float | None = None,
    compute_residuals: bool = False,
    normalize_kwargs: dict | None = None,
    res_id: str | None = None,
) -> None:
    """Perform PCA and persist scores/loadings in the map outputs.

    Parameters
    ----------
    spectral_map
        Input map containing spectra to decompose.
    mask
        Optional mask identifier restricting processed pixels.
    roi_x
        Optional spectral ROI.
    pca_n_components
        Number of components or explained variance ratio accepted by
        `sklearn.decomposition.PCA`.
    compute_residuals
        If ``True``, store a residual image after reconstruction.
    normalize_kwargs
        Optional normalization arguments applied before PCA.
    res_id
        Optional explicit output group id.

    Raises
    ------
    ValueError
        If ``pca_n_components`` is missing or not strictly positive.
    """

    if pca_n_components is None or pca_n_components <= 0:
        raise ValueError("pca_n_components must be a positive number")

    idxs, x_idx, pixel_idx = spectral_map.get_indices(mask, roi_x)
    roi_x = spectral_map.adapt_roi_x(roi_x if roi_x is not None else spectral_map.roi_x)

    intensities = np.asarray(spectral_map.data)[idxs]
    x_axis = spectral_map.x[x_idx]

    if normalize_kwargs is not None:
        if isinstance(normalize_kwargs, StepNormalizeParams):
            normalize_kwargs = normalize_kwargs.model_dump(exclude_none=True)
        intensities = _normalize_intensities(intensities, x_axis, roi_x=roi_x, **normalize_kwargs)

    pca = PCA(n_components=pca_n_components, random_state=0)
    scores = pca.fit_transform(intensities)

    elem_ids: list[str] = []
    group_key = res_id or generate_key()

    masked_img_template = None
    if not isinstance(pixel_idx, slice):
        masked_img_template = np.empty(spectral_map.map_shape, dtype=scores.dtype)
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

    if compute_residuals:
        reconstruct = pca.inverse_transform(scores)
        residuals = intensities - reconstruct
        residuals = np.mean(residuals, axis=const.Axis.SPECTRAL)

        if masked_img_template is not None:
            img = masked_img_template.copy()
            img.ravel()[pixel_idx] = residuals
        else:
            img = residuals.reshape(spectral_map.map_shape)

        img_key = f"{group_key}_residuals"
        spectral_map.images[img_key] = Image2D(
            data=img,
            data_rules=None,
            viz_rules=None,
            name="Residuals",
            locked=True,
            parent_group=group_key,
            visible=False,
        )
        elem_ids.append(img_key)

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    data_unit = spectral_map.data_unit

    for i in range(pca.n_components_ - 1, -1, -1):
        if masked_img_template is not None:
            img = masked_img_template.copy()
            img.ravel()[pixel_idx] = scores[:, i]
        else:
            img = scores[:, i].reshape(spectral_map.map_shape)

        img_key = f"{group_key}_{i}"
        spectral_map.images[img_key] = Image2D(
            data=img,
            data_rules=None,
            viz_rules={"cmap": color_generator(i), "color2": "k"},
            name=f"PC{i + 1}",
            locked=True,
            visible=i < MAX_IMG_TO_SHOW,
            parent_group=group_key,
            spectrum=Spectrum(
                data=loadings[:, i],
                x=spectral_map.x[x_idx],
                roi_x=roi_x,
                x_axis_unit=spectral_map.x_axis_unit,
                data_unit=data_unit,
                ignore_sort=True,
                name=f"PC{i + 1} loading",
                metadata={"explained_variance": pca.explained_variance_ratio_[i]},
            ),
        )
        elem_ids.append(img_key)

    spectral_map.images_group[group_key] = spectral_map.new_images_group(elem_ids, name="PCA")
