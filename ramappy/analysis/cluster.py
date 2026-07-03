"""Clustering analysis step.

This module exposes a pipeline step that clusters spectra into :math:`k` groups and
stores each cluster as a derived mask, optionally with centroid spectra.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import field_validator
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA

from ramappy.core import SpectralMap, Spectrum
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
from ramappy.utils import generate_key

from .common import _make_hierarchical_clustering, validate_pca_n_components


def _group_cluster_indices(
    labels: npt.ArrayLike,
    *,
    n_clusters: int,
    pixel_idx: np.ndarray | slice,
) -> list[np.ndarray]:
    """Return cluster-member indices grouped once, preserving original order."""

    labels_arr = np.asarray(labels, dtype=np.intp)
    order = np.argsort(labels_arr, kind="stable")
    counts = np.bincount(labels_arr, minlength=n_clusters)
    grouped_local_indices = np.split(order, np.cumsum(counts)[:-1])

    if isinstance(pixel_idx, slice):
        return grouped_local_indices

    pixel_idx_arr = np.asarray(pixel_idx)
    return [pixel_idx_arr[group] for group in grouped_local_indices]


class StepClusterParams(StepParams, ParamsAcceptMask, ParamsAcceptRoIX, ParamsSetResultId):
    """Parameters for `cluster`."""

    method: Literal["kmeans", "minibatchkmeans", "hierarchical"] = "kmeans"
    """Clustering backend (`"kmeans"`, `"minibatchkmeans"`, or `"hierarchical"`)."""

    n_clusters: int = 4
    """Number of clusters to compute."""

    pca_n_components: int | float | None = 0
    """Optional PCA dimensionality reduction before clustering. Values ``<= 0`` disable PCA."""

    normalize_kwargs: StepNormalizeParams | None = None
    """Optional normalization settings applied before PCA/clustering."""

    @field_validator("pca_n_components", mode="before")
    @classmethod
    def _validate_pca_n_components(cls, value: int | float | None) -> int | float | None:
        return validate_pca_n_components(value)


@pipeline_step(
    step_name="cluster",
    params_validator=SingleModelParamsValidator(StepClusterParams),
    friendly_name="Cluster",
    step_category=StepClass.ANALYSIS,
)
def cluster(
    spectral_map: SpectralMap,
    *,
    n_clusters: int = 4,
    method: Literal["kmeans", "minibatchkmeans", "hierarchical"] = "kmeans",
    mask: str | None = None,
    roi_x: npt.ArrayLike | None = None,
    pca_n_components: float | None = None,
    normalize_kwargs: dict[str, Any] | None = None,
    res_id: str | None = None,
) -> None:
    """Cluster spectra into mask groups.

    Parameters
    ----------
    spectral_map
        Input map containing spectra to cluster.
    n_clusters
        Number of clusters to compute.
    method
        Clustering backend (`"kmeans"`, `"minibatchkmeans"`, or
        `"hierarchical"`).
    mask
        Optional spatial mask identifier restricting participating pixels.
    roi_x
        Optional spectral ROI used before feature extraction.
    pca_n_components
        Optional PCA dimensionality reduction before clustering. Values
        ``<= 0`` disable PCA.
    normalize_kwargs
        Optional normalization settings applied before PCA/clustering.
    res_id
        Optional explicit group identifier for generated outputs.

    Raises
    ------
    ValueError
        If the requested clustering method is not supported.
    """

    idxs, x_idx, pixel_idx = spectral_map.get_indices(mask, roi_x)
    roi_x = spectral_map.adapt_roi_x(roi_x)

    intensities = spectral_map.data[idxs]
    x_axis = spectral_map.x[x_idx]

    if normalize_kwargs is not None:
        if isinstance(normalize_kwargs, StepNormalizeParams):
            normalize_kwargs = normalize_kwargs.model_dump(exclude_none=True)
        intensities = _normalize_intensities(intensities, x_axis, roi_x=roi_x, **normalize_kwargs)

    if pca_n_components is not None and pca_n_components > 0:
        pca = PCA(n_components=pca_n_components, random_state=0)
        intensities = pca.fit_transform(intensities)

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

    if pca_n_components is not None and pca_n_components > 0:
        clusters_mean = None

    data_unit = spectral_map.data_unit
    grouped_cluster_indices = _group_cluster_indices(labels, n_clusters=n_clusters, pixel_idx=pixel_idx)

    group_key = res_id or generate_key()
    masks_list: list[str] = []
    for i in range(n_clusters - 1, -1, -1):
        cluster_idxs = grouped_cluster_indices[i]
        mask_key = f"{group_key}_{i}"

        spectrum = None
        if clusters_mean is not None:
            spectrum = Spectrum(
                data=clusters_mean[i, :],
                x=spectral_map.x[x_idx],
                roi_x=roi_x,
                x_axis_unit=spectral_map.x_axis_unit,
                data_unit=data_unit,
                ignore_sort=True,
            )

        spectral_map.masks[mask_key] = spectral_map.new_mask(
            idxs=cluster_idxs,
            spectrum=spectrum,
            name=f"Cluster {i + 1}",
            editable=False,
            parent_group=group_key,
        )
        masks_list.append(mask_key)

    spectral_map.masks_group[group_key] = spectral_map.new_masks_group(masks_list, name=f"{method} clustering")
    if clusters_mean is not None:
        spectral_map.change_spectrum_agg(group_key, "centroid")
