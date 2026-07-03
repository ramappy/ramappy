import h5py
import numpy as np
import zarr

from ramappy.io.hdf5_nexus import read_hdf5_nexus
from ramappy.io.zarr.writer import write_zarr


def test_nexus_read_then_write_zarr_preserves_unit_shape(tmp_path):
    h5_path = tmp_path / "nexus_export.h5"
    zarr_path = tmp_path / "out.zarr.zip"

    with h5py.File(h5_path, "w") as h5file:
        entry_data = h5file.create_group("ENTRY/data")
        entry_data.create_dataset("intensity", data=np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
        entry_data.create_dataset("raman_shift", data=np.linspace(100.0, 400.0, 4, dtype=np.float32))
        entry_data.create_dataset("x", data=np.array([0.0, 1.0], dtype=np.float32))
        entry_data.create_dataset("y", data=np.array([0.0, 1.0, 2.0], dtype=np.float32))

        project = h5file.create_group("PROJECT")
        project.create_dataset("project_id", data="proj-1")
        project.create_dataset("author_id", data="author-1")

        h5file.create_group("SAMPLE")

    hsi = read_hdf5_nexus(str(h5_path), as_hsi=True)
    write_zarr(hsi, out_file=str(zarr_path), as_zip=True)

    with zarr.storage.ZipStore(str(zarr_path), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        x_unit = root["data"].attrs["x_axis_unit"]

    assert isinstance(x_unit, (list, tuple))
    assert len(x_unit) == 2
