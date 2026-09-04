"""Tests for the N-FINDR endmember extraction algorithm."""

from __future__ import annotations

import numpy as np

from ramappy.analysis.nfindr.nfindr import NFINDR
from ramappy.analysis.nfindr.step import nfindr
from ramappy.core import SpectralMap
from ramappy.core.masks import MaskGroup


def _make_simplex_data(n_endmembers: int = 4, n_extra: int = 100, seed: int = 0) -> np.ndarray:
    """Unit-simplex corners in R^(n_endmembers-1) plus random interior points.

    The first *n_endmembers* rows are the ground-truth endmembers.
    """
    rng = np.random.default_rng(seed)
    n_features = n_endmembers - 1
    vertices = np.eye(n_features, dtype=np.float64)
    vertices = np.vstack([vertices, np.zeros(n_features)])
    alphas = rng.dirichlet(np.ones(n_endmembers), size=n_extra)
    interior = alphas @ vertices
    return np.vstack([vertices, interior])


def test_nfindr_converges_and_finds_endmembers_float64():
    X = _make_simplex_data(n_endmembers=4)
    nf = NFINDR(n_endmembers=4, random_state=0)
    nf.fit(X)

    assert nf.converged_, "algorithm should converge on this simple dataset"
    assert nf.n_iters_ >= 1
    assert set(nf.endmember_indices_) == {0, 1, 2, 3}


def test_nfindr_converges_float32():
    """float32 inputs must converge: adaptive threshold is above the float32 noise floor."""
    X = _make_simplex_data(n_endmembers=4).astype(np.float32)
    nf = NFINDR(n_endmembers=4, random_state=0)
    nf.fit(X)

    assert nf.converged_, "float32 input must converge with dtype-adaptive threshold"
    assert set(nf.endmember_indices_) == {0, 1, 2, 3}


def test_nfindr_work_dtype_float64_upcasts_float32():
    """work_dtype='float64' upcasts float32 input and still recovers the correct endmembers."""
    X = _make_simplex_data(n_endmembers=4).astype(np.float32)
    nf = NFINDR(n_endmembers=4, random_state=0, work_dtype="float64")
    nf.fit(X)

    assert nf.converged_
    assert set(nf.endmember_indices_) == {0, 1, 2, 3}


def test_nfindr_barycentric_coordinates_sum_to_one():
    """Barycentric coordinates of every point in the dataset must sum to 1."""
    X = _make_simplex_data(n_endmembers=4)
    nf = NFINDR(n_endmembers=4, random_state=0)
    nf.fit(X)
    coefs = nf.transform(X, method="barycentric")
    np.testing.assert_allclose(coefs.sum(axis=1), 1.0, atol=1e-6)


def test_nfindr_endmember_masks_follow_matching_colors(monkeypatch):
    """Endmember masks must inherit the color of their matching endmember image before group recoloring."""
    X = _make_simplex_data(n_endmembers=4, n_extra=36)
    hsi = SpectralMap(x=np.linspace(100, 1800, X.shape[1]), data=X, img_width=8, img_height=5, ignore_sort=True)

    def no_recolor(self, elem_ids=None, **kwargs):
        return MaskGroup(elem_ids=elem_ids, **kwargs)

    monkeypatch.setattr(SpectralMap, "new_masks_group", no_recolor)

    nfindr(hsi, n_endmembers=4, gen_endmember_masks=True)

    group_key = next(iter(hsi.images_group.keys()))
    for i in range(4):
        img_id = f"{group_key}_{i}"
        mask_id = f"{group_key}_mask_{i}"
        assert img_id in hsi.images
        assert mask_id in hsi.masks
        assert hsi.masks[mask_id].color == str(hsi.images[img_id].viz_rules.cmap)
