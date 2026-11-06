import numpy as np

from ramappy.core.spectral_map import SpectralMap
from ramappy.pipeline.steps.images import edit_image


def _map_with_projection_image(*, locked: bool) -> SpectralMap:
    x = np.linspace(400.0, 1800.0, 50, dtype=np.float32)
    data = np.random.default_rng(0).random((9, 50)).astype(np.float32)
    spectral_map = SpectralMap(x=x, data=data, img_width=3, img_height=3, ignore_sort=True)
    img = spectral_map.intensity_map(
        data_rules={"roi_x_A": (500.0, 900.0)},
        viz_rules={"cmap": "viridis", "invert": True},
        name="old",
        locked=locked,
        visible=False,
    )
    img.parent_group = "grp"
    spectral_map.images["img"] = img
    return spectral_map


def _edit(spectral_map: SpectralMap, **kwargs) -> None:
    edit_image(spectral_map, image_id="img", keys_order=[], group_id="", group_order=[], **kwargs)


def test_edit_image_unlock_and_rename_together():
    spectral_map = _map_with_projection_image(locked=True)

    _edit(spectral_map, locked=False, name="new", visible=True)

    img = spectral_map.images["img"]
    assert img.locked is False
    assert img.name == "new"
    assert img.visible is True
    assert img.parent_group == "grp"


def test_edit_image_new_data_rules_keep_display_attributes():
    spectral_map = _map_with_projection_image(locked=False)

    _edit(spectral_map, data_rules={"roi_x_A": (1000.0, 1500.0)}, viz_rules={"invert": False})

    img = spectral_map.images["img"]
    assert img.data_rules.roi_x_A == (1000.0, 1500.0)
    assert img.name == "old"
    assert img.visible is False
    assert img.parent_group == "grp"
    assert img.viz_rules.cmap == "viridis"  # kept from the existing rules
    assert img.viz_rules.invert is False  # partial update applied
