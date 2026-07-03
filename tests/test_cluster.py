from __future__ import annotations

from importlib import import_module

import numpy as np

from ramappy.core.spectral_map import SpectralMap

cluster_mod = import_module("ramappy.analysis.cluster")
common_mod = import_module("ramappy.analysis.common")


def test_make_hierarchical_clustering_prefers_metric_when_supported(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAgglomerativeClustering:
        def __init__(self, *, n_clusters, linkage, metric):
            captured.update({"n_clusters": n_clusters, "linkage": linkage, "metric": metric})

    monkeypatch.setattr(common_mod, "AgglomerativeClustering", FakeAgglomerativeClustering)

    estimator = cluster_mod._make_hierarchical_clustering(n_clusters=3)

    assert isinstance(estimator, FakeAgglomerativeClustering)
    assert captured == {"n_clusters": 3, "linkage": "ward", "metric": "euclidean"}


def test_make_hierarchical_clustering_falls_back_to_affinity(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAgglomerativeClustering:
        def __init__(self, *, n_clusters, linkage, affinity):
            captured.update({"n_clusters": n_clusters, "linkage": linkage, "affinity": affinity})

    monkeypatch.setattr(common_mod, "AgglomerativeClustering", FakeAgglomerativeClustering)

    estimator = cluster_mod._make_hierarchical_clustering(n_clusters=2)

    assert isinstance(estimator, FakeAgglomerativeClustering)
    assert captured == {"n_clusters": 2, "linkage": "ward", "affinity": "euclidean"}


def test_group_cluster_indices_preserves_original_order_and_maps_masked_pixels():
    labels = np.array([1, 0, 1, 2, 0, 2], dtype=int)
    pixel_idx = np.array([10, 20, 30, 40, 50, 60], dtype=int)

    grouped = cluster_mod._group_cluster_indices(labels, n_clusters=3, pixel_idx=pixel_idx)

    assert [group.tolist() for group in grouped] == [[20, 50], [10, 30], [40, 60]]


def test_kmeans_cluster_creates_mask_group_with_centroid_spectra():
    x = np.array([100.0, 200.0, 300.0, 400.0])
    intensities = np.array(
        [
            [1.0, 1.5, 2.0, 2.5],
            [1.1, 1.4, 2.1, 2.4],
            [9.5, 9.0, 8.5, 8.0],
            [9.6, 9.1, 8.4, 8.1],
        ]
    )

    spectral_map = SpectralMap(x=x, data=intensities, img_width=2, img_height=2, ignore_sort=True)

    cluster_mod.cluster(spectral_map, n_clusters=2, method="kmeans", res_id="kmeans_group")

    group = spectral_map.masks_group["kmeans_group"]
    assert group.name == "kmeans clustering"
    assert group.spectrum_agg == "centroid"

    cluster_members = {frozenset(spectral_map.masks[mask_id].idxs.tolist()) for mask_id in group.elem_ids}
    assert cluster_members == {frozenset({0, 1}), frozenset({2, 3})}

    for mask_id in group.elem_ids:
        mask = spectral_map.masks[mask_id]
        assert mask.parent_group == "kmeans_group"
        assert mask.editable is False
        assert mask.spectrum is not None
        assert mask.spectrum_agg == "centroid"
        np.testing.assert_allclose(mask.spectrum.x, x)
        np.testing.assert_allclose(mask.spectrum.data.reshape(-1), intensities[mask.idxs].mean(axis=0))


def test_cluster_with_mask_and_pca_maps_cluster_pixels_back_to_original_indices():
    x = np.array([150.0, 250.0, 350.0, 450.0])
    intensities = np.array(
        [
            [1.0, 1.2, 1.4, 1.6],
            [50.0, 50.0, 50.0, 50.0],
            [1.1, 1.3, 1.5, 1.7],
            [60.0, 60.0, 60.0, 60.0],
            [8.0, 8.2, 8.4, 8.6],
            [8.1, 8.3, 8.5, 8.7],
        ]
    )

    spectral_map = SpectralMap(x=x, data=intensities, img_width=3, img_height=2, ignore_sort=True)
    selected_idxs = np.array([0, 2, 4, 5], dtype=int)
    spectral_map.masks["selected"] = spectral_map.new_mask(idxs=selected_idxs)

    cluster_mod.cluster(
        spectral_map,
        n_clusters=2,
        method="kmeans",
        mask="selected",
        pca_n_components=1,
        res_id="masked_pca_cluster",
    )

    group = spectral_map.masks_group["masked_pca_cluster"]
    assert group.name == "kmeans clustering"
    assert group.spectrum_agg == "mean"

    combined_idxs = np.sort(
        np.concatenate([np.asarray(spectral_map.masks[mask_id].idxs) for mask_id in group.elem_ids], dtype=int)
    )
    np.testing.assert_array_equal(combined_idxs, np.sort(selected_idxs))

    for mask_id in group.elem_ids:
        mask = spectral_map.masks[mask_id]
        assert mask.spectrum is None
        assert mask.spectrum_agg == "mean"
        assert set(mask.idxs).issubset(set(selected_idxs.tolist()))
