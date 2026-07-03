import h5py
import numpy as np

from ramappy.io.hdf5_nexus import read_hdf5_nexus


def test_nexus_reader_supports_nested_metadata_and_pixel_sizes(tmp_path):
    file_path = tmp_path / "nexus_nested.h5"

    with h5py.File(file_path, "w") as h5file:
        entry_data = h5file.create_group("ENTRY/data")
        entry_data.create_dataset("intensity", data=np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
        entry_data.create_dataset("raman_shift", data=np.linspace(100.0, 400.0, 4, dtype=np.float32))
        entry_data.create_dataset("x", data=np.array([0.0, 2.0], dtype=np.float32))
        entry_data.create_dataset("y", data=np.array([0.0, 1.0, 2.0], dtype=np.float32))

        project = h5file.create_group("PROJECT")
        project.create_dataset("project_id", data="proj-1")
        project.create_dataset("author_id", data="author-1")

        sample = h5file.create_group("SAMPLE")
        sample.create_dataset("sample_id", data="sample-1")

        entry = h5file["ENTRY"]
        instrument = entry.create_group("instrument")
        laser = instrument.create_group("laser")
        laser.create_dataset("wavelength_nm", data=np.array(532, dtype=np.int32))
        laser.attrs["model"] = np.bytes_(b"Laser-X")

    hsi = read_hdf5_nexus(str(file_path), as_hsi=True)

    acq = hsi.metadata["Acquisition"]
    assert "instrument" in acq
    assert acq["instrument"]["laser"]["wavelength_nm"] == 532
    assert acq["instrument"]["laser"]["model"] == "Laser-X"

    assert hsi.metadata["Acquisition"]["instrument"]["laser"]["wavelength_nm"] == 532
    assert hsi.metadata["Sample"]["sample_id"] == "sample-1"

    assert hsi.spatial_grid.pixel_size_x == 2.0
    assert hsi.spatial_grid.pixel_size_y == 1.0
    assert hsi.spatial_grid.spatial_unit
    assert "sample_metadata" not in hsi.metadata
    assert "Acquisition_metadata" not in hsi.metadata

    # Legacy NeXus cubes are (x, y, spectral): reader must normalize to
    # SpectralMap convention (height, width, spectral) = (y, x, spectral).
    assert hsi.map_shape == (3, 2)
    expected_cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4).transpose(1, 0, 2)
    np.testing.assert_array_equal(hsi.cube, expected_cube)


def test_nexus_reader_honors_explicit_axes_attribute(tmp_path):
    file_path = tmp_path / "nexus_axes.h5"

    with h5py.File(file_path, "w") as h5file:
        entry_data = h5file.create_group("ENTRY/data")
        entry_data.attrs["axes"] = np.array([b"y", b"x", b"ram"], dtype="S")

        cube = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
        entry_data.create_dataset("intensity", data=cube)
        entry_data.create_dataset("raman_shift", data=np.linspace(100.0, 400.0, 4, dtype=np.float32))
        entry_data.create_dataset("x", data=np.array([10.0, 20.0], dtype=np.float32))
        entry_data.create_dataset("y", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))

        project = h5file.create_group("PROJECT")
        project.create_dataset("project_id", data="proj-1")
        project.create_dataset("author_id", data="author-1")

        h5file.create_group("SAMPLE")

    hsi = read_hdf5_nexus(str(file_path), as_hsi=True)

    assert hsi.map_shape == (3, 2)
    np.testing.assert_array_equal(hsi.cube, cube)
