import numpy as np
import pytest

from ramappy import units
from ramappy.core.images2d import SpatialGrid
from ramappy.core.spectral_map import SpectralMap


@pytest.fixture
def sample_hsi():
    """Create a sample SpectralMap object for testing."""
    x = np.linspace(100, 200, 10)
    img_width = 5
    img_height = 5
    # Intensities: 25 pixels, 10 spectral points; pixel i has a flat spectrum of value (i+1).
    # Shape: (pixels, wavenumbers) -> (25, 10)
    intensities = np.zeros((25, 10))
    for i in range(25):
        intensities[i, :] = (i + 1) * np.ones_like(x)

    return SpectralMap(x=x, data=intensities, img_width=img_width, img_height=img_height)


def test_hsi_initialization(sample_hsi):
    """Test SpectralMap initialization."""
    assert sample_hsi.img_width == 5
    assert sample_hsi.img_height == 5
    assert sample_hsi.x.shape == (10,)
    assert sample_hsi.data.shape == (25, 10)
    assert sample_hsi.x_size == 10
    assert sample_hsi.map_shape == (5, 5)


def test_get_spectrum_mean(sample_hsi):
    """Test getting mean spectrum."""
    spectrum = sample_hsi.get_spectrum(agg="mean")
    # Mean of 1..25 is 13
    assert np.allclose(spectrum.data, 13.0)
    assert spectrum.x.shape == (10,)


def test_get_spectrum_with_mask(sample_hsi):
    """Test getting spectrum with a mask."""
    # Create a mask for the first pixel (index 0, value 1)
    mask = sample_hsi.new_mask(idxs=[0])
    sample_hsi.masks["test_mask"] = mask

    spectrum = sample_hsi.get_spectrum(mask="test_mask", agg="mean")
    assert np.allclose(spectrum.data, 1.0)


def test_get_indices_full(sample_hsi):
    """Test get_indices without mask."""
    _idxs, x_idx, px_idx = sample_hsi.get_indices()
    assert isinstance(px_idx, slice)
    assert isinstance(x_idx, slice)


def test_get_indices_mask(sample_hsi):
    """Test get_indices with mask."""
    mask = sample_hsi.new_mask(idxs=[0, 1, 2])
    _idxs, _x_idx, px_idx = sample_hsi.get_indices(mask=mask)
    assert len(px_idx) == 3
    assert np.array_equal(px_idx, [0, 1, 2])


def test_apply_func_pixel(sample_hsi):
    """Test apply_func by pixel."""

    # Function to double the values
    def double(x):
        return x * 2

    res = sample_hsi.apply_func(double, by="pixel")
    assert np.array_equal(res, sample_hsi.data * 2)


def test_math_sub_spectrum(sample_hsi):
    """Test subtraction of a spectrum."""
    # Subtract the mean spectrum from the dataset
    mean_spec = sample_hsi.get_spectrum(agg="mean")

    # We need to copy because math modifies in place
    import copy

    hsi_copy = copy.deepcopy(sample_hsi)

    hsi_copy.math(operand=mean_spec, op="sub")

    # Check a pixel. Pixel 0 was 1.0, mean was 13.0. Result should be -12.0
    # Pixel 24 was 25.0, mean was 13.0. Result should be 12.0
    # intensities is (pixels, wavenumbers) -> (25, 10)
    assert np.allclose(hsi_copy.data[0, :], -12.0)
    assert np.allclose(hsi_copy.data[24, :], 12.0)


def test_mask_operation(sample_hsi):
    """Test mask operations."""
    m1 = sample_hsi.new_mask(idxs=[0, 1, 2])
    m2 = sample_hsi.new_mask(idxs=[2, 3, 4])

    sample_hsi.masks["m1"] = m1
    sample_hsi.masks["m2"] = m2

    # Union
    m_union = sample_hsi.mask_operation(mask_a="m1", mask_b="m2", op="union")
    assert len(m_union.idxs) == 5  # 0, 1, 2, 3, 4

    # Intersection
    m_inter = sample_hsi.mask_operation(mask_a="m1", mask_b="m2", op="intersection")
    assert len(m_inter.idxs) == 1
    assert m_inter.idxs[0] == 2

    # Diff (m1 - m2) -> {0, 1}
    m_diff = sample_hsi.mask_operation(mask_a="m1", mask_b="m2", op="diff")
    assert len(m_diff.idxs) == 2
    assert np.all(np.isin(m_diff.idxs, [0, 1]))


def test_map_shape_reshape(sample_hsi):
    """Test setting map_shape with same number of pixels (reshape)."""
    # Original is 5x5 = 25 pixels
    # Reshape to 1x25
    sample_hsi.map_shape = (1, 25)
    assert sample_hsi.img_height == 1
    assert sample_hsi.img_width == 25
    assert sample_hsi.data.shape == (25, 10)  # Data shouldn't change shape


def test_map_shape_crop(sample_hsi):
    """Test setting map_shape with different number of pixels (crop)."""
    # Original is 5x5 = 25 pixels
    # Set to 3x3 = 9 pixels. Should crop top-left.

    with pytest.warns(UserWarning, match="Performing crop"):
        sample_hsi.map_shape = (3, 3)

    assert sample_hsi.img_height == 3
    assert sample_hsi.img_width == 3
    # Intensities should be cropped to 9 pixels
    assert sample_hsi.data.shape == (9, 10)

    # Verify content: original was 5x5.
    # Pixel (r, c) index in flattened array was r*5 + c
    # New image is 3x3.
    # Kept pixels should be (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
    # Original indices: 0, 1, 2, 5, 6, 7, 10, 11, 12
    # Check first pixel (0,0) -> index 0. Value should be same as original index 0.
    # Check last pixel (2,2) -> index 8 in new array. Was index 12 in old array.

    # Re-create original intensities logic to verify
    # In fixture: intensities[i, :] = (i + 1) * ones
    # So pixel 0 has value 1.0
    # Pixel 12 has value 13.0

    assert np.allclose(sample_hsi.data[0, :], 1.0)
    assert np.allclose(sample_hsi.data[8, :], 13.0)


def test_from_cube_builds_flattened_data_and_validates_axis_length():
    cube = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    spectral_axis = np.linspace(100, 200, 5)

    spectral_map = SpectralMap.from_cube(cube, spectral_axis=spectral_axis)

    assert spectral_map.data.shape == (12, 5)
    assert spectral_map.map_shape == (3, 4)
    np.testing.assert_allclose(spectral_map.x, spectral_axis)

    with pytest.raises(ValueError, match="Spectral axis length"):
        SpectralMap.from_cube(cube, spectral_axis=np.linspace(100, 200, 4))


def test_cube_property_is_a_view_not_copy(sample_hsi):
    cube = sample_hsi.cube
    assert cube.shape == (sample_hsi.img_height, sample_hsi.img_width, sample_hsi.n_spectral)
    assert np.shares_memory(cube, sample_hsi.data)

    cube[0, 0, 0] = 999.0
    assert sample_hsi.data[0, 0] == pytest.approx(999.0)


def test_spatial_grid_coordinate_roundtrip():
    grid = SpatialGrid(
        pixel_size_y=1.5,
        pixel_size_x=2.0,
        origin_y=10.0,
        origin_x=20.0,
        spatial_unit=units.Unit.MICROMETER,
    )
    y, x = grid.pixel_to_physical(2, 3)
    assert y == pytest.approx(13.0)
    assert x == pytest.approx(26.0)

    row, col = grid.physical_to_pixel(y, x)
    assert row == 2
    assert col == 3
