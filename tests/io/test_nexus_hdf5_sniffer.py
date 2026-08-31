import h5py
import numpy as np
import pytest

from ramappy.io import guess_file_type
from ramappy.io.core import InputFormatRegistry
from ramappy.io.matlab import sniff_nexus


def test_sniff_nexus_matches_required_nested_paths(tmp_path):
    file_path = tmp_path / "nexus.h5"
    with h5py.File(file_path, "w") as h5file:
        entry_data = h5file.create_group("ENTRY/data")
        entry_data.create_dataset("intensity", data=np.zeros((2, 3, 4), dtype=np.float32))
        entry_data.create_dataset("raman_shift", data=np.arange(4, dtype=np.float32))

        project = h5file.create_group("PROJECT")
        project.create_dataset("project_id", data="proj-1")
        project.create_dataset("author_id", data="author-1")

    assert sniff_nexus(str(file_path)) is True
    assert guess_file_type(str(file_path)).format_name == "hdf5_nexus"


def test_guess_file_type_falls_back_to_generic_hdf5_when_not_nexus(tmp_path):
    file_path = tmp_path / "generic.h5"
    with h5py.File(file_path, "w") as h5file:
        h5file.create_dataset("data", data=np.zeros((2, 3, 4), dtype=np.float32))
        h5file.create_dataset("x", data=np.arange(4, dtype=np.float32))

    assert sniff_nexus(str(file_path)) is False
    assert guess_file_type(str(file_path)).format_name == "hdf5"


def test_guess_file_type_logs_sniffer_errors_and_keeps_fallback(tmp_path, caplog, monkeypatch):
    file_path = tmp_path / "generic.h5"
    with h5py.File(file_path, "w") as h5file:
        h5file.create_dataset("data", data=np.zeros((2, 3, 4), dtype=np.float32))
        h5file.create_dataset("x", data=np.arange(4, dtype=np.float32))

    nexus_format = next(fmt for fmt in InputFormatRegistry.extension_map["h5"] if fmt.format_name == "hdf5_nexus")

    def failing_sniffer(_filename):
        raise RuntimeError("broken test sniffer")

    monkeypatch.setattr(nexus_format, "sniffer", failing_sniffer)
    with caplog.at_level("DEBUG", logger="ramappy.io.common"):
        selected = guess_file_type(str(file_path))

    assert selected.format_name == "hdf5"
    assert "Input format sniffer failed for hdf5_nexus" in caplog.text
