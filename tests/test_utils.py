"""Tests for ramappy.utils."""

import numpy as np
import pytest

from ramappy.utils import (
    atleast_2d,
    blend_images,
    color_generator,
    common_roi_overlap,
    create_cmap,
    find_nearest_x,
    fwhm,
    generate_key,
    get_x_regions,
    is_sorted,
    merge_arrays,
    minmax,
    pearson_correlation_coefficient,
    r2_score,
    rubberband,
    spectral_angle_mapper,
    whittaker_smooth,
)


class TestArrayUtils:
    def test_is_sorted(self):
        assert is_sorted(np.array([1, 2, 3]))
        assert not is_sorted(np.array([1, 3, 2]))
        with pytest.raises(ValueError, match="X-axis has repeated values"):
            is_sorted(np.array([1, 2, 2]))

    def test_minmax(self):
        arr = np.array([1, 5, 2, 8, 3])
        assert minmax(arr) == (1, 8)

        masked_arr = np.ma.masked_array([1, 5, 2, 8, 3], mask=[0, 0, 1, 0, 0])
        assert minmax(masked_arr) == (1, 8)

    def test_merge_arrays(self):
        x = np.array([1, 3, 5])
        x_grid = np.array([2, 4, 6])
        merged, (old_idx, grid_idx) = merge_arrays(x, x_grid)
        np.testing.assert_array_equal(merged, np.array([1, 2, 3, 4, 5, 6]))
        np.testing.assert_array_equal(old_idx, np.array([0, 2, 4]))
        np.testing.assert_array_equal(grid_idx, np.array([1, 3, 5]))

    def test_atleast_2d(self):
        arr = np.array([1, 2, 3])
        assert atleast_2d(arr, axis=0).shape == (1, 3)
        assert atleast_2d(arr, axis=1).shape == (3, 1)
        assert atleast_2d(np.array(1)).shape == (1, 1)


class TestSmoothingUtils:
    def test_whittaker_smooth(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)
        smoothed = whittaker_smooth(y[np.newaxis, :], lambda_=100)
        assert smoothed.shape == (1, 100)
        assert np.std(smoothed) < np.std(y)

    def test_rubberband(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + x  # Rising baseline
        baseline = rubberband(y, x)
        assert baseline.shape == y.shape
        assert np.all(baseline <= y + 1e-6)  # Allow small float error

    def test_rubberband_fast_mode(self):
        x = np.linspace(0, 10, 2000)
        y = np.sin(x) + x

        baseline = rubberband(y, x, max_points=128)
        assert baseline.shape == y.shape
        assert np.all(baseline <= y + 1e-6)


class TestROIUtils:
    def test_common_roi_overlap(self):
        roi_a = np.array([[10, 20], [30, 40]])
        roi_b = np.array([[15, 25], [35, 45]])
        overlap = common_roi_overlap(roi_a, roi_b)
        np.testing.assert_array_equal(overlap, np.array([[15, 20], [35, 40]]))

    def test_find_nearest_x(self):
        x = np.array([10, 20, 30])
        assert find_nearest_x(x, 12) == 0
        assert find_nearest_x(x, 18) == 1
        assert find_nearest_x(x, 28) == 2

    def test_get_x_regions(self):
        wn = np.array([1, 2, 3, 10, 11, 12])
        regions = get_x_regions(wn, threshold=1)
        # Expecting [[1, 3], [10, 12]]
        np.testing.assert_array_equal(regions, np.array([[1, 3], [10, 12]]))


class TestMetricsUtils:
    def test_spectral_angle_mapper(self):
        s1 = np.array([[1, 0, 0]])
        s2 = np.array([[0, 1, 0]])
        # Orthogonal, angle should be pi/2
        sam = spectral_angle_mapper(s1, s2)
        np.testing.assert_allclose(sam, np.pi / 2)

        s3 = np.array([[1, 0, 0]])
        sam_same = spectral_angle_mapper(s1, s3)
        np.testing.assert_allclose(sam_same, 0, atol=1e-6)

    def test_spectral_angle_mapper_zero_norm(self):
        # Zero-norm spectrum must not produce NaN/inf (guarded by epsilon)
        s_zero = np.array([[0.0, 0.0, 0.0]])
        s_ref = np.array([[1.0, 0.0, 0.0]])
        sam = spectral_angle_mapper(s_zero, s_ref)
        assert np.all(np.isfinite(sam))

    def test_pearson_correlation(self):
        s1 = np.array([[1, 2, 3]])
        s2 = np.array([[2, 4, 6]])
        corr = pearson_correlation_coefficient(s1, s2)
        np.testing.assert_allclose(corr, 1.0)

    def test_r2_score_perfect(self):
        # Perfect match → R² = 1 for all pixels
        s1 = np.array([[1, 2, 3]])
        s2 = np.array([[1, 2, 3]])
        r2 = r2_score(s1, s2)
        np.testing.assert_allclose(r2, 1.0)

    def test_r2_score_shape(self):
        # r2_score returns (N, M) for (N, W) × (M, W) inputs
        intensities = np.random.rand(5, 10)
        references = np.random.rand(3, 10)
        r2 = r2_score(intensities, references)
        assert r2.shape == (5, 3)

    def test_r2_score_known_value(self):
        # Reference: [1, 2, 3], mean=2, SS_tot=2
        # Prediction shifted by constant +1: [2, 3, 4]
        # SS_res = (1-2)^2 + (2-3)^2 + (3-4)^2 = 3
        # R² = 1 - 3/2 = -0.5
        ref = np.array([[1.0, 2.0, 3.0]])
        pred = np.array([[2.0, 3.0, 4.0]])
        r2 = r2_score(pred, ref)
        np.testing.assert_allclose(r2, -0.5, rtol=1e-6)

    def test_r2_score_no_centering_of_predictions(self):
        # R² must NOT centre the pixel spectra - a constant positive shift must
        # lower R², not preserve it.
        ref = np.array([[0.0, 1.0, 2.0]])
        pred_exact = np.array([[0.0, 1.0, 2.0]])
        pred_shifted = np.array([[1.0, 2.0, 3.0]])
        r2_exact = r2_score(pred_exact, ref).item()
        r2_shifted = r2_score(pred_shifted, ref).item()
        assert r2_exact > r2_shifted, "Centering predictions would incorrectly equalise the two scores"

    def test_sid_normalization(self):
        from ramappy.utils import spectral_information_divergence

        # Identical spectra should give SID = 0
        s = np.array([[1.0, 2.0, 3.0]])
        sid = spectral_information_divergence(s, s)
        np.testing.assert_allclose(sid, 0.0, atol=1e-10)

        # SID must be non-negative (symmetric KL divergence property)
        rng = np.random.default_rng(42)
        a = np.abs(rng.standard_normal((10, 20))) + 0.1
        b = np.abs(rng.standard_normal((1, 20))) + 0.1
        sid_vals = spectral_information_divergence(a, b)
        assert np.all(sid_vals >= 0)

    def test_fwhm(self):
        x = np.linspace(-10, 10, 100)
        y = np.exp(-(x**2) / (2 * 2**2))  # Gaussian with sigma=2
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma approx 2.355 * 2 = 4.71
        width = fwhm(y)
        # width is in indices, we need to convert to x units if we want exact check
        # but here we just check it returns a positive number
        assert width > 0


class TestImageUtils:
    def test_create_cmap(self):
        cmap = create_cmap("red", "blue")
        assert cmap.N == 256

    def test_blend_images(self):
        img1 = np.zeros((10, 10, 4))
        img2 = np.ones((10, 10, 4))
        blended = blend_images([img1, img2], mode="plus")
        assert blended is not None


class TestMiscUtils:
    def test_generate_key(self):
        key = generate_key()
        assert isinstance(key, str)
        assert len(key) > 0

    def test_color_generator(self):
        color = color_generator(0)
        assert color.startswith("#")
        assert len(color) == 7
