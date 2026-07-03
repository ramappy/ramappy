from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import normalize as sk_normalize

from ramappy import const
from ramappy.core.spectral_map import SpectralMap
from ramappy.processing.normalize import normalize_intensities
from ramappy.utils import clean_roi, common_roi_overlap, select_x_indices


def _expected_l2_by_pixel_separate_regions(
    *,
    intensities: np.ndarray,
    x: np.ndarray,
    mask_pixels: np.ndarray,
    roi_x: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    x_idx = select_x_indices(x, roi_x)
    if isinstance(x_idx, slice):
        raise AssertionError("Expected a non-contiguous ROI to produce an index array")

    x_axis = x[x_idx]
    roi_x_arr = clean_roi(np.asarray(roi_x))
    x_extent = np.array([[float(np.min(x_axis)), float(np.max(x_axis))]], dtype=float)
    roi_x_clean = common_roi_overlap(roi_x_arr, x_extent)
    regions = select_x_indices(x_axis, roi_x_clean, keep_regions=True)

    expected_block = intensities[np.ix_(mask_pixels, x_idx)].copy()
    for region in regions:
        expected_block[:, region] = sk_normalize(expected_block[:, region], axis=const.Axis.SPECTRAL, norm="l2")
    return x_idx, expected_block


def test_normalize_separate_regions_with_mask_and_multiregion_roi():
    rng = np.random.default_rng(0)
    x = np.arange(10, dtype=float)

    img_width, img_height = 2, 3
    n_pix = img_width * img_height
    intensities0 = rng.normal(size=(n_pix, x.size)).astype(np.float64)
    hsi = SpectralMap(x=x, data=intensities0.copy(), img_width=img_width, img_height=img_height)

    # Non-slice mask (forces advanced indexing on pixel axis)
    mask_pixels = np.array([0, 2, 5], dtype=int)
    hsi.masks["m"] = hsi.new_mask(idxs=mask_pixels)

    roi_x = [[2.0, 4.0], [7.0, 8.0]]
    x_idx, expected_block = _expected_l2_by_pixel_separate_regions(
        intensities=intensities0,
        x=x,
        mask_pixels=mask_pixels,
        roi_x=roi_x,
    )

    normalize_intensities(hsi, norm="l2", by_pixel=True, mask="m", roi_x=roi_x, separate_regions=True)

    out = np.asarray(hsi.data)
    expected_block = expected_block.astype(out.dtype, copy=False)
    np.testing.assert_allclose(out[np.ix_(mask_pixels, x_idx)], expected_block, rtol=1e-6, atol=1e-6)

    other_pixels = np.setdiff1d(np.arange(n_pix), mask_pixels)
    np.testing.assert_allclose(
        out[np.ix_(other_pixels, x_idx)],
        intensities0[np.ix_(other_pixels, x_idx)].astype(out.dtype, copy=False),
        rtol=0,
        atol=0,
    )

    outside_cols = np.setdiff1d(np.arange(x.size), x_idx)
    np.testing.assert_allclose(
        out[:, outside_cols],
        intensities0[:, outside_cols].astype(out.dtype, copy=False),
        rtol=0,
        atol=0,
    )


def test_normalize_full_axis_roi_fast_path_matches_plain_normalization():
    rng = np.random.default_rng(1)
    x = np.linspace(100.0, 1000.0, 12)
    intensities0 = rng.normal(size=(6, x.size)).astype(np.float64)

    hsi_plain = SpectralMap(x=x, data=intensities0.copy(), img_width=2, img_height=3)
    hsi_roi = SpectralMap(x=x, data=intensities0.copy(), img_width=2, img_height=3)

    normalize_intensities(hsi_plain, norm="l2", by_pixel=True, roi_x=None, separate_regions=False)
    normalize_intensities(
        hsi_roi,
        norm="l2",
        by_pixel=True,
        roi_x=[[float(x.min()), float(x.max())]],
        separate_regions=True,
    )

    np.testing.assert_allclose(hsi_roi.data, hsi_plain.data, rtol=1e-6, atol=1e-6)


def test_area_normalization_warns_and_matches_by_pixel_true():
    rng = np.random.default_rng(2)
    x = np.linspace(400.0, 1800.0, 16)
    intensities0 = np.abs(rng.normal(size=(4, x.size))).astype(np.float64) + 0.5

    hsi_warn = SpectralMap(x=x, data=intensities0.copy(), img_width=2, img_height=2)
    hsi_reference = SpectralMap(x=x, data=intensities0.copy(), img_width=2, img_height=2)

    with pytest.warns(UserWarning, match="by_pixel=False"):
        normalize_intensities(hsi_warn, norm="area", by_pixel=False)

    normalize_intensities(hsi_reference, norm="area", by_pixel=True)

    np.testing.assert_allclose(hsi_warn.data, hsi_reference.data, rtol=1e-6, atol=1e-6)
