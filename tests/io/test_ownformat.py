from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from ramappy.core.images2d import Image2D, Image2DGroup
from ramappy.core.masks import MaskGroup
from ramappy.core.spectral_map import SpectralMap
from ramappy.io.custom import read_own_custom, write_own_custom


@pytest.fixture
def small_hsi() -> SpectralMap:
    x = np.array([100.0, 110.0, 120.0, 130.0])
    img_width = 3
    img_height = 2

    # 6 pixels, 4 spectral points
    intensities = np.arange(img_width * img_height * x.size, dtype=float).reshape(img_width * img_height, x.size)
    spectral_map = SpectralMap(x=x, data=intensities, img_width=img_width, img_height=img_height)

    # Add a mask (with a group)
    m = spectral_map.new_mask(idxs=[0, 2, 5], name="My mask", parent_group="mg")
    spectral_map.masks["m1"] = m
    spectral_map.masks_group["mg"] = MaskGroup(elem_ids=["m1"], name="Mask group")

    # Add an image (with a group)
    img = Image2D(data=np.arange(img_width * img_height, dtype=float).reshape(img_height, img_width), name="My image")
    img.parent_group = "ig"
    spectral_map.images["i1"] = img
    spectral_map.images_group["ig"] = Image2DGroup(elem_ids=["i1"], name="Image group")

    return spectral_map


def test_ownformat_roundtrip_dataframe_explicit_coords(small_hsi: SpectralMap):
    df = write_own_custom(small_hsi, format="pandas", add_masks_images=True, explicit_coordinates=True)

    # Read back from a pandas dataframe
    hsi2 = read_own_custom(df, as_hsi=True)

    assert hsi2.map_shape == small_hsi.map_shape
    np.testing.assert_allclose(hsi2.x, small_hsi.x)
    np.testing.assert_allclose(hsi2.data, small_hsi.data)

    # Masks/images roundtrip
    assert "m1" in hsi2.masks
    np.testing.assert_array_equal(hsi2.masks["m1"].idxs, small_hsi.masks["m1"].idxs)
    assert "i1" in hsi2.images
    np.testing.assert_allclose(hsi2.images["i1"].data, small_hsi.images["i1"].data)


def test_ownformat_roundtrip_dataframe_pixel_index_and_shuffle(small_hsi: SpectralMap):
    df = write_own_custom(small_hsi, format="pandas", add_masks_images=False, explicit_coordinates=False)

    # Shuffle the rows: pixel_index should allow reconstruction.
    df_shuffled = df.sample(frac=1.0, random_state=0)

    hsi2 = read_own_custom(df_shuffled, img_width=small_hsi.img_width, img_height=small_hsi.img_height, as_hsi=True)

    np.testing.assert_allclose(hsi2.data, small_hsi.data)


def test_ownformat_roundtrip_feather_bytesio(small_hsi: SpectralMap):
    buf = BytesIO()
    out = write_own_custom(small_hsi, format="feather", out_file=buf, add_masks_images=False, explicit_coordinates=True)

    assert isinstance(out, BytesIO)
    out.seek(0)

    hsi2 = read_own_custom(out, img_width=small_hsi.img_width, img_height=small_hsi.img_height, as_hsi=True)
    np.testing.assert_allclose(hsi2.data, small_hsi.data)


def test_ownformat_subset_export_fills_missing_with_nan(small_hsi: SpectralMap):
    # Export only pixels included in mask m1
    df = write_own_custom(small_hsi, format="pandas", mask="m1", add_masks_images=False, explicit_coordinates=False)

    hsi2 = read_own_custom(df, img_width=small_hsi.img_width, img_height=small_hsi.img_height, as_hsi=True)

    # Included pixels match
    idxs = small_hsi.masks["m1"].idxs
    np.testing.assert_allclose(hsi2.data[idxs], small_hsi.data[idxs])

    # Excluded pixels become NaN
    excluded = np.setdiff1d(np.arange(small_hsi.img_width * small_hsi.img_height), small_hsi.masks["m1"].idxs)
    assert np.isnan(hsi2.data[excluded]).all()
