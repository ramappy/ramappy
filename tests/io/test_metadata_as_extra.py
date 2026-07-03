from ramappy.core.metadata import metadata_as_extra, metadata_with_extra


def test_metadata_as_extra_wraps_all_fields_under_extra():
    payload = metadata_as_extra({"source": "file.h5"}, project_id="p1", nullable=None)

    assert "extra" in payload
    assert payload["extra"] == {
        "source": "file.h5",
        "project_id": "p1",
    }


def test_metadata_as_extra_normalizes_input_filename_to_original_filename():
    payload = metadata_as_extra(input_filename="/tmp/acq/map_01.ngs", role="import")

    assert payload["original_filename"] == "map_01.ngs"
    assert payload["extra"]["role"] == "import"
    assert "input_filename" not in payload["extra"]


def test_metadata_with_extra_builds_standard_and_grouped_extra_payload():
    payload = metadata_with_extra(
        title="my-map",
        instrument={"laser_wavelength_nm": 532.0},
        extra={
            "project": {"id": "p1"},
            "pixel_physical_size_x": 1.5,
        },
        vendor={"format": "witec", "nullable": None},
    )

    assert payload["title"] == "my-map"
    assert payload["instrument"]["laser_wavelength_nm"] == 532.0
    assert payload["extra"]["vendor"] == {"format": "witec"}
    assert payload["extra"]["project"]["id"] == "p1"
    assert payload["extra"]["pixel_physical_size_x"] == 1.5
