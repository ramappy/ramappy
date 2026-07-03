from __future__ import annotations

from importlib import import_module

import numpy as np

from ramappy.core.spectral_map import SpectralMap

substrate_mod = import_module("ramappy.analysis.substrate")
common_mod = import_module("ramappy.analysis.common")


def test_make_hierarchical_clustering_prefers_metric_when_supported(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAgglomerativeClustering:
        def __init__(self, *, n_clusters, linkage, metric):
            captured.update({"n_clusters": n_clusters, "linkage": linkage, "metric": metric})

    monkeypatch.setattr(common_mod, "AgglomerativeClustering", FakeAgglomerativeClustering)

    estimator = substrate_mod._make_hierarchical_clustering(n_clusters=2)

    assert isinstance(estimator, FakeAgglomerativeClustering)
    assert captured == {"n_clusters": 2, "linkage": "ward", "metric": "euclidean"}


def test_make_hierarchical_clustering_falls_back_to_affinity(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAgglomerativeClustering:
        def __init__(self, *, n_clusters, linkage, affinity):
            captured.update({"n_clusters": n_clusters, "linkage": linkage, "affinity": affinity})

    monkeypatch.setattr(common_mod, "AgglomerativeClustering", FakeAgglomerativeClustering)

    estimator = substrate_mod._make_hierarchical_clustering(n_clusters=3)

    assert isinstance(estimator, FakeAgglomerativeClustering)
    assert captured == {"n_clusters": 3, "linkage": "ward", "affinity": "euclidean"}


def test_substrate_with_mask_and_pca_maps_masks_back_to_original_indices_in_order():
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

    substrate_mod.substrate_extraction(
        spectral_map,
        n_clusters=2,
        method="kmeans",
        mask="selected",
        pca_n_components=1,
        morphology_clean=False,
    )

    np.testing.assert_array_equal(spectral_map.masks["substrate"].idxs, np.array([0, 2]))
    np.testing.assert_array_equal(spectral_map.masks["foreground"].idxs, np.array([4, 5]))

    group = spectral_map.masks_group["substrate_id"]
    assert group.elem_ids == ["substrate", "foreground"]
    assert group.name == "Substrate identification"
