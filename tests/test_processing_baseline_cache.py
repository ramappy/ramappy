from __future__ import annotations

import numpy as np
import pytest

import ramappy.processing.baseline as baseline_mod
from ramappy.core.spectral_map import SpectralMap
from ramappy.processing.baseline import correct_baseline


def test_pybaselines_baseline_cache_reuses_fitter_for_same_x_view():
    baseline_mod._PYBASELINES_BASELINE_CACHE.clear()

    x = np.linspace(100.0, 200.0, 50)
    img_width, img_height = 2, 2
    n_pix = img_width * img_height

    # Simple sloped spectra
    intensities = (np.arange(n_pix, dtype=float)[:, None] + 1.0) * (0.01 * x[None, :] + 2.0)
    hsi = SpectralMap(x=x, data=intensities.copy(), img_width=img_width, img_height=img_height)

    roi_x = [120.0, 160.0]
    _idxs, x_idx, _px_idx = hsi.get_indices(mask=None, roi_x=roi_x)

    key = baseline_mod._x_cache_key(hsi.x[x_idx])

    # First call should populate cache
    correct_baseline(hsi, method="poly", roi_x=roi_x, poly_order=2)
    assert key in baseline_mod._PYBASELINES_BASELINE_CACHE
    fitter1 = baseline_mod._PYBASELINES_BASELINE_CACHE[key]
    assert len(baseline_mod._PYBASELINES_BASELINE_CACHE) == 1

    # Second call with same ROI should reuse same object
    correct_baseline(hsi, method="poly", roi_x=roi_x, poly_order=2)
    fitter2 = baseline_mod._PYBASELINES_BASELINE_CACHE[key]

    assert fitter1 is fitter2
    assert len(baseline_mod._PYBASELINES_BASELINE_CACHE) == 1


def test_correct_baseline_force_nonnegative_does_not_print_debug_output(capsys: pytest.CaptureFixture[str]):
    x = np.linspace(100.0, 200.0, 20)
    intensities = np.vstack(
        [
            0.02 * x + 1.0,
            0.015 * x + 0.5,
            0.01 * x + 0.75,
            0.03 * x + 0.25,
        ]
    )
    hsi = SpectralMap(x=x, data=intensities, img_width=2, img_height=2)

    correct_baseline(hsi, method="poly", poly_order=1, force_nonnegative=True)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
