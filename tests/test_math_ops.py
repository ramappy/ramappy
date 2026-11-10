import numpy as np
import pytest

from ramappy.core.pipeline import PipelineStepRegistry
from ramappy.core.spectral_map import SpectralMap
from ramappy.core.spectrum import Spectrum


def _make_map() -> SpectralMap:
    x = np.linspace(400.0, 1800.0, 40)
    data = np.random.default_rng(0).random((9, 40)) + 1.0
    spectral_map = SpectralMap(x=x, data=data, img_width=3, img_height=3, ignore_sort=True)
    spectral_map.add_spectra(Spectrum(x=x, data=np.full((1, 40), 0.5)), key="ref")
    # normally created by the substrate identification step
    spectral_map.masks["substrate"] = spectral_map.new_mask(idxs=np.array([0, 1, 2]), name="Substrate")
    return spectral_map


@pytest.mark.parametrize(
    ("step_name", "params"),
    [
        ("subtract_substrate", {"how": "mean"}),
        ("subtract_spectra", {"ref_spectra": "ref"}),
        ("math_operation", {"B": 1.0}),
    ],
)
def test_math_steps_report_modified_data(step_name, params):
    spectral_map = _make_map()
    before = spectral_map.data.copy()

    # options that request change tracking and a previous-data snapshot
    PipelineStepRegistry.get_step(step_name).run(spectral_map, params, track_changes=True, save_old_data=True)

    assert not np.allclose(spectral_map.data, before)
    assert spectral_map.history.get_last_changes().modified_data is True
    np.testing.assert_array_equal(spectral_map.aux_data["data_previous"], before)
