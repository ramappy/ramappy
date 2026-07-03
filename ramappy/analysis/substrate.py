"""Substrate/foreground segmentation analysis step."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import field_validator
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA

from ramappy import const
from ramappy.core import SpectralMap
from ramappy.core.pipeline import (
    ParamsAcceptMask,
    ParamsAcceptRoIX,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)
from ramappy.processing.normalize import StepNormalizeParams, _normalize_intensities

from .common import _make_hierarchical_clustering, validate_pca_n_components


def _group_cluster_local_indices(labels: npt.ArrayLike, *, n_clusters: int) -> list[np.ndarray]:
    """Return cluster-member local indices grouped once, preserving original order."""

    labels_arr = np.asarray(labels, dtype=np.intp)
    order = np.argsort(labels_arr, kind="stable")
    counts = np.bincount(labels_arr, minlength=n_clusters)
    return np.split(order, np.cumsum(counts)[:-1])


class StepCellSegmentationParams(StepParams, ParamsAcceptMask, ParamsAcceptRoIX):
    """Parameters for `substrate_extraction`."""

    n_clusters: int = 2
    """Number of clusters used for segmentation."""

    method: Literal["kmeans", "minibatchkmeans", "hierarchical"] = "kmeans"
    """Clustering method (`"kmeans"`, `"minibatchkmeans"`, `"hierarchical"`)."""

    pca_n_components: int | float | None = 0
    """Number of PCA components to retain. Values <= 0 disable PCA."""

    morphology_clean: bool = True
    """Whether to run morphological cleanup on the foreground mask."""

    morphology_minsize: int = 64
    """Minimum connected-component size for cleanup."""

    erosion: int = 0
    """Size of theerosion disk for cleanup."""

    normalize_kwargs: StepNormalizeParams | None = None
    """Optional normalization settings used before clustering."""

    @field_validator("pca_n_components", mode="before")
    @classmethod
    def _validate_pca_n_components(cls, value: int | float | None) -> int | float | None:
        return validate_pca_n_components(value)


@pipeline_step(
    step_name="identify_substrate",
    params_validator=SingleModelParamsValidator(StepCellSegmentationParams),
    friendly_name="Identify Substrate",
    step_category=StepClass.ANALYSIS,
)
def substrate_extraction(
    spectral_map: SpectralMap,
    *,
    n_clusters: int = 4,
    method: Literal["kmeans", "minibatchkmeans", "hierarchical"] = "kmeans",
    mask: str | None = None,
    roi_x: npt.ArrayLike | None = None,
    pca_n_components: float = 0,
    morphology_clean: bool = True,
    morphology_minsize: int = 64,
    erosion: int = 0,
    normalize_kwargs: dict[str, Any] | None = None,
) -> None:
    """Identify substrate and foreground masks through clustering.

    Parameters
    ----------
    spectral_map
        Input map to segment.
    n_clusters
        Number of clusters used for segmentation.
    method
        Clustering method (`"kmeans"`, `"minibatchkmeans"`, `"hierarchical"`).
    mask
        Optional spatial mask restricting candidate pixels.
    roi_x
        Optional spectral ROI.
    pca_n_components
        Optional PCA dimensionality reduction prior to clustering.
    morphology_clean
        Whether to run morphological cleanup on the foreground mask.
    morphology_minsize
        Minimum connected-component size for cleanup.
    erosion
        Number of erosion iterations during cleanup.
    normalize_kwargs
        Optional normalization settings used before clustering.

    Raises
    ------
    ValueError
        If an unsupported clustering method is requested.
    """

    idxs, x_idx, pixel_idx = spectral_map.get_indices(mask, roi_x)
    roi_x = spectral_map.adapt_roi_x(roi_x)
    raw_intensities = spectral_map.data[idxs]
    intensities = raw_intensities
    x_axis = spectral_map.x[x_idx]

    if normalize_kwargs is not None:
        if isinstance(normalize_kwargs, StepNormalizeParams):
            normalize_kwargs = normalize_kwargs.model_dump(exclude_none=True)
        intensities = _normalize_intensities(intensities, x_axis, roi_x=roi_x, **normalize_kwargs)

    if pca_n_components > 0:
        intensities = PCA(n_components=pca_n_components, random_state=0).fit_transform(intensities)

    clusters_mean = None
    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(intensities)
        clusters_mean = kmeans.cluster_centers_
    elif method == "minibatchkmeans":
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(intensities)
        clusters_mean = kmeans.cluster_centers_
    elif method == "hierarchical":
        hierarchical = _make_hierarchical_clustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(intensities)
    else:
        raise ValueError("No method specified")

    if clusters_mean is None or pca_n_components > 0:
        grouped_local_indices = _group_cluster_local_indices(labels, n_clusters=n_clusters)
        clusters_mean = np.array(
            [np.mean(raw_intensities[group], axis=const.Axis.PIXEL) for group in grouped_local_indices]
        )

    # We assume the substrate is the cluster with the lowest mean signal.
    bg = clusters_mean.mean(axis=1).argmin()

    is_background = np.asarray(labels) == bg
    local_background = np.flatnonzero(is_background)
    local_foreground = np.flatnonzero(~is_background)

    if isinstance(pixel_idx, slice):
        idxs_backgr = local_background
        idxs_foregr = local_foreground
    else:
        idxs_backgr = pixel_idx[local_background]
        idxs_foregr = pixel_idx[local_foreground]

    spectral_map.masks["substrate"] = spectral_map.new_mask(
        idxs=idxs_backgr,
        name="Substrate",
        editable=False,
        parent_group="substrate_id",
    )
    spectral_map.masks["foreground"] = spectral_map.new_mask(
        idxs=idxs_foregr,
        name="Foreground",
        editable=False,
        parent_group="substrate_id",
    )

    spectral_map.masks_group["substrate_id"] = spectral_map.new_masks_group(
        ["substrate", "foreground"],
        name="Substrate identification",
        color="coolwarm",
    )

    if morphology_clean:
        spectral_map.masks["foreground"].refine(morphology_minsize=morphology_minsize, erosion=erosion)
