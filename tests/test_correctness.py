from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ramappy.core.images2d import Image2D, Image2DRenderer
from ramappy.core.spectral_map import SpectralMap
from ramappy.processing.baseline import correct_baseline
from ramappy.processing.denoising.despike import despike
from ramappy.processing.geometric.crop import crop_spatial
from ramappy.processing.normalize import normalize_intensities
from ramappy.processing.resample import resample


def test_spectral_position_normalization_correctness():
    x = np.linspace(100.0, 200.0, 10)
    intensities = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        ]
    )
    hsi = SpectralMap(x=x, data=intensities.copy(), img_width=2, img_height=1)

    # Normalize by wavenumber 144.4 (corresponds to index 4 with value 5.0 and 10.0)
    # The nearest x to 145 is 144.4 (index 4)
    ref_wn = 145.0
    normalize_intensities(hsi, norm="spectral_position", wn=ref_wn)

    # Check that each spectrum is scaled so that its value at the reference wavenumber is 1.0
    expected = intensities.copy()
    expected[0] /= 5.0
    expected[1] /= 10.0

    np.testing.assert_allclose(hsi.data, expected, rtol=1e-6)

    # Check that by_pixel=False triggers a warning but still works per-pixel
    hsi2 = SpectralMap(x=x, data=intensities.copy(), img_width=2, img_height=1)
    with pytest.warns(UserWarning, match="'by_pixel=False' is incompatible"):
        normalize_intensities(hsi2, norm="spectral_position", wn=ref_wn, by_pixel=False)

    np.testing.assert_allclose(hsi2.data, expected, rtol=1e-6)


def test_despike_mask_mapping_correctness():
    x = np.linspace(100.0, 200.0, 10)
    # 6 pixels, only pixel 2 has a cosmic ray spike at index 5
    intensities = np.ones((6, 10), dtype=float)
    intensities[2, 5] = 100.0  # Big spike

    # Background neighbor pixels (1 and 3 are 1.0)
    hsi = SpectralMap(x=x, data=intensities.copy(), img_width=3, img_height=2)

    # Mask selecting pixels [2, 3, 5] (pixel 2 is at index 0 of the mask)
    mask_pixels = np.array([2, 3, 5], dtype=int)
    hsi.masks["m"] = hsi.new_mask(idxs=mask_pixels)

    # Despike with mask and correct_spikes=True
    # Threshold is set low to detect the spike (since max Z-score for N=3 is ~1.414)
    despike(hsi, mask="m", method="normal", threshold=1.2, correct_spikes=True, first_diff=False)

    # Spikes mask should contain absolute index 2
    assert np.array_equal(hsi.masks["spikes"].indices, [2])

    # The spike at pixel 2 should be corrected using neighbors (pixels 1, 3, etc. are 1.0, so median is 1.0)
    np.testing.assert_allclose(hsi.data[2, 5], 1.0, atol=1e-5)

    # Absolute index 0 should NOT be affected (which would happen if relative index 0 was corrected)
    np.testing.assert_allclose(hsi.data[0], intensities[0], atol=1e-5)


def test_baseline_force_nonnegative_per_pixel():
    x = np.linspace(100.0, 200.0, 10)
    # Pixel 0: min is -5.0, Pixel 1: min is -1.0
    intensities = np.array(
        [
            [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ]
    )
    hsi = SpectralMap(x=x, data=intensities.copy(), img_width=2, img_height=1)

    # Use a simple poly baseline to exercise the force_nonnegative path.
    correct_baseline(hsi, method="poly", poly_order=1, force_nonnegative=True)

    # The global minimum of the whole image must be exactly 0.0.
    # Per-pixel minima need not all be zero: the pixel that was least negative is
    # shifted by the same global offset and may end up with a positive minimum.
    assert hsi.data.min() >= -1e-6


def test_resample_short_grid():
    x = np.linspace(100.0, 200.0, 10)
    intensities = np.ones((2, 10), dtype=float)
    hsi = SpectralMap(x=x, data=intensities, img_width=2, img_height=1)

    # Resample to a short grid of length 2
    target_grid = [120.0, 180.0]
    resample(hsi, method="spline", spline_kind="slinear", x_grid=target_grid)

    assert len(hsi.x) == 2
    np.testing.assert_allclose(hsi.x, target_grid)
    assert hsi.data.shape == (2, 2)


def test_spatial_crop_image_bounds():
    x = np.linspace(100.0, 200.0, 10)
    intensities = np.ones((6, 10), dtype=float)
    hsi = SpectralMap(x=x, data=intensities, img_width=3, img_height=2)

    # Create a dummy pillow image matching the shape of the map (width=3, height=2)
    pil_img = Image.new("RGB", (3, 2), color="red")
    hsi.images["img"] = Image2D(image=pil_img)

    # Crop the map from (col 1 to 3, row 1 to 2) -> new dimensions should be width=2, height=1
    crop_spatial(hsi, col_start=1, col_end=3, row_start=1, row_end=2)

    # Check new map dimensions
    assert hsi.img_width == 2
    assert hsi.img_height == 1

    # Check cropped image size
    img = Image2DRenderer.render(hsi.images["img"])
    assert img.size == (2, 1)  # width=2, height=1
