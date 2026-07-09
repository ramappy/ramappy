import numpy as np

from ramappy.core.spectral_map import SpectralMap
from ramappy.processing.denoising.svd import compute_spatial_signal_ratio, smooth_svd


def test_compute_spatial_signal_ratio_does_not_mutate_vt():
    rng = np.random.default_rng(0)
    n_rows, n_cols = 6, 7
    Vt = rng.normal(size=(10, n_rows * n_cols))
    Vt_before = Vt.copy()

    compute_spatial_signal_ratio(Vt, shape=(n_rows, n_cols), roi_threshold=0.5)

    np.testing.assert_array_equal(Vt, Vt_before)


def test_smooth_svd_spatial_ratio_matches_manual_reconstruction():
    rng = np.random.default_rng(0)
    n_rows, n_cols = 8, 9
    n_wn = 40
    x = np.linspace(0.0, 100.0, n_wn, dtype=np.float32)

    yy, xx = np.mgrid[0:n_rows, 0:n_cols]
    spatial_signal = np.exp(-((yy - n_rows / 2) ** 2 + (xx - n_cols / 2) ** 2) / (2 * 2.0**2))
    spectral_shape = np.exp(-((x - 50.0) ** 2) / (2 * 10.0**2))

    clean = spatial_signal.reshape(-1, 1) * spectral_shape.reshape(1, -1)
    noisy = (clean + rng.normal(0, 0.05, size=clean.shape)).astype(np.float32)

    manual = SpectralMap(x=x, data=noisy.copy(), img_width=n_cols, img_height=n_rows, ignore_sort=True)
    smooth_svd(manual, method="manual", num_sv=1)

    spatial_ratio = SpectralMap(x=x, data=noisy.copy(), img_width=n_cols, img_height=n_rows, ignore_sort=True)
    smooth_svd(spatial_ratio, method="spatial_ratio", roi_threshold=0.5, spatial_ratio_threshold_multiplier=3.5)

    # Both select the same single dominant singular value on this data; a
    # corrupted Vt would make spatial_ratio diverge sharply from the manual
    # reconstruction instead of matching it closely.
    np.testing.assert_allclose(spatial_ratio.data, manual.data, atol=1e-4)
