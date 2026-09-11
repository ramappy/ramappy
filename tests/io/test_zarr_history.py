from __future__ import annotations

import base64
import sys
import warnings
import zipfile
from io import BytesIO
from types import ModuleType

import numpy as np
import pytest
import zarr
from PIL import Image

# Optional dependency imported by ramappy.io package init; not needed in this test.
try:  # pragma: no cover - environment dependent
    import h5py as _h5py
except ModuleNotFoundError:  # pragma: no cover - fallback only when unavailable
    sys.modules.setdefault("h5py", ModuleType("h5py"))

renishaw_wire = ModuleType("renishawWiRE")
renishaw_wire.WDFReader = object
sys.modules.setdefault("renishawWiRE", renishaw_wire)

renishaw_wire_types = ModuleType("renishawWiRE.types")


class _UnitType:
    Arbitrary = "Arbitrary"
    RamanShift = "RamanShift"
    Wavenumber = "Wavenumber"
    Nanometre = "Nanometre"
    ElectronVolt = "ElectronVolt"
    Micron = "Micron"
    Counts = "Counts"
    Pixels = "Pixels"
    Intensity = "Intensity"
    RelativeIntensity = "RelativeIntensity"


renishaw_wire_types.UnitType = _UnitType
sys.modules.setdefault("renishawWiRE.types", renishaw_wire_types)

from ramappy.core.images2d import Image2D
from ramappy.core.images2d.group import Image2DGroup
from ramappy.core.masks import Mask, MaskGroup
from ramappy.core.spectral_map import SpectralMap
from ramappy.core.spectrum import Spectrum
from ramappy.io.core import InputFormatRegistry
from ramappy.io.zarr import reader as _zarr_reader
from ramappy.io.zarr.common import get_default_compressors
from ramappy.io.zarr.images import read_zarr_image, write_zarr_image
from ramappy.io.zarr.masks import write_zarr_mask
from ramappy.io.zarr.reader import read_zarr
from ramappy.io.zarr.store import ZarrProjectStore
from ramappy.io.zarr.writer import _dedup_zip_central_directory, write_zarr


def _patch_find_nearest_x_many(monkeypatch) -> None:
    from ramappy.core import spectrum as _spectrum_module

    def _stable_find_nearest_x_many(x: np.ndarray, roi_x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        values = np.asarray(roi_x)
        idx = np.searchsorted(x, values, side="left")
        idx = np.clip(idx, 0, x.size - 1)
        prev = np.clip(idx - 1, 0, x.size - 1)
        use_prev = np.abs(x[prev] - values) <= np.abs(x[idx] - values)
        return np.where(use_prev, prev, idx).astype(np.int64)

    monkeypatch.setattr(_spectrum_module, "find_nearest_x_many", _stable_find_nearest_x_many)


def _make_small_hsi() -> SpectralMap:
    x = np.array([100.0, 110.0, 120.0], dtype=np.float32)
    img_width, img_height = 2, 2
    intensities = np.arange(img_width * img_height * x.size, dtype=np.float32).reshape(img_width * img_height, x.size)
    return SpectralMap(x=x, data=intensities, img_width=img_width, img_height=img_height)


def test_zarr_roundtrip_preserves_history_input_and_unknown_steps(tmp_path):
    hsi = _make_small_hsi()

    hsi.history.set_input_config(input_format="renishaw_wdf", input_config={"filename": "raw.wdf", "laser_nm": 785})
    hsi.history.steps = [
        {
            "name": "legacy_unknown_step",
            "params": {
                "res_id": "abc123",
                "pca_n_components": 4,
                "compute_residuals": False,
            },
            "changes": {"modified_data": True},
        }
    ]

    out = tmp_path / "history_roundtrip.zarr"
    write_zarr(hsi, out_file=out, as_zip=False)

    loaded = read_zarr(out)

    assert loaded.history.input is not None
    assert loaded.history.input["format_name"] == "renishaw_wdf"
    assert loaded.history.input["format_params"]["filename"] == "raw.wdf"

    assert len(loaded.history.steps) == 1
    assert isinstance(loaded.history.steps[0], dict)
    assert loaded.history.steps[0]["name"] == "legacy_unknown_step"
    assert loaded.history.steps[0]["params"]["pca_n_components"] == 4


def test_read_file_sets_zarr_input_when_history_input_missing(tmp_path):
    hsi = _make_small_hsi()

    out = tmp_path / "no_input_history.zarr"
    write_zarr(hsi, out_file=out, as_zip=False)

    reader = InputFormatRegistry.get_format("zarr")
    assert reader is not None
    loaded = reader.read(out)

    assert loaded.history.input is not None
    assert loaded.history.input["format_name"] == "zarr"

@pytest.mark.parametrize("reader_name", ["read_zarr", "read_zarr_metadata_only"])
def test_zarr_reader_closes_store_when_open_fails(monkeypatch, reader_name):
    closed = False

    class FakeStore:
        def close(self):
            nonlocal closed
            closed = True

    monkeypatch.setattr(_zarr_reader, "ZipStore", lambda *_args, **_kwargs: FakeStore())

    def fail_open(*_args, **_kwargs):
        raise ValueError("invalid zarr")

    monkeypatch.setattr(_zarr_reader.zarr, "open", fail_open)

    with pytest.raises(ValueError, match="invalid zarr"):
        getattr(_zarr_reader, reader_name)("not-a-directory.zarr.zip")

    assert closed


def test_zarr_v2_roundtrip_preserves_cube_and_spatial_grid(tmp_path):
    cube = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    spectral_axis = np.linspace(50.0, 150.0, 5, dtype=np.float32)

    spectral_map = SpectralMap.from_cube(
        cube,
        spectral_axis=spectral_axis,
        spatial_grid={
            "pixel_size_y": 1.2,
            "pixel_size_x": 2.4,
            "origin_y": 10.0,
            "origin_x": 20.0,
            "spatial_unit": "μm",
        },
    )

    out = tmp_path / "roundtrip_v2.zarr.zip"
    write_zarr(spectral_map, out_file=out, as_zip=True)

    with zarr.storage.ZipStore(str(out), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        assert root.attrs["format_version"] == 2
        assert root["data/data"].shape == (3, 4, 5)
        assert isinstance(root.attrs["metadata"], dict)
        assert isinstance(root.attrs["history"], dict)
        assert isinstance(root.attrs["spatial_grid"], dict)

    loaded = read_zarr(out)
    np.testing.assert_array_equal(loaded.cube, cube)
    np.testing.assert_array_equal(loaded.x, spectral_axis)
    assert loaded.spatial_grid.pixel_size_y == 1.2
    assert loaded.spatial_grid.pixel_size_x == 2.4
    assert loaded.spatial_grid.origin_y == 10.0
    assert loaded.spatial_grid.origin_x == 20.0


def test_write_zarr_strips_transient_metadata_keys_from_structured_metadata(tmp_path):
    spectral_map = _make_small_hsi()
    spectral_map.smap_metadata = {
        "title": "Persistent title",
        "project_id": "top-level-project",
        "last_access": "yesterday",
        "extra": {
            "keep_me": "still here",
            "live_update": True,
            "project_id": "nested-project",
        },
    }

    out = tmp_path / "sanitized_metadata.zarr"
    write_zarr(spectral_map, out_file=out, as_zip=False)

    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="r")
    metadata = root.attrs["metadata"]

    assert metadata["title"] == "Persistent title"
    assert "project_id" not in metadata
    assert "last_access" not in metadata
    assert metadata["extra"] == {"keep_me": "still here"}


def test_write_zarr_zip_has_no_duplicate_entries(tmp_path):
    cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    spectral_axis = np.linspace(100.0, 400.0, 4, dtype=np.float32)
    spectral_map = SpectralMap.from_cube(cube, spectral_axis=spectral_axis)

    out = tmp_path / "no_duplicate_entries.zarr.zip"
    write_zarr(spectral_map, out_file=out, as_zip=True)

    with zipfile.ZipFile(out, mode="r") as zf:
        names = [info.filename for info in zf.infolist()]

    assert len(names) == len(set(names))


def test_zarr_thumbnail_is_capped_to_256px(tmp_path):
    cube = np.zeros((700, 520, 8), dtype=np.float32)
    spectral_axis = np.linspace(100.0, 200.0, 8, dtype=np.float32)
    spectral_map = SpectralMap.from_cube(cube, spectral_axis=spectral_axis)

    out = tmp_path / "thumb_size.zarr"
    write_zarr(spectral_map, out_file=out, as_zip=False)

    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="r")
    thumb_b64 = root.attrs["thumbnail"]

    decoded = base64.b64decode(thumb_b64)
    with Image.open(BytesIO(decoded)) as image:
        assert image.width <= 256
        assert image.height <= 256


def test_write_zarr_image_spectrum_allows_missing_data_unit():
    root = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")

    spectrum = Spectrum(
        x=np.array([100.0, 110.0], dtype=np.float32),
        data=np.array([1.0, 2.0], dtype=np.float32),
        data_unit=None,
    )
    image = Image2D(data=np.ones((2, 2), dtype=np.float32), name="legacy", spectrum=spectrum)

    image_group = root.create_group("img")
    write_zarr_image(image_group, image, compressors=get_default_compressors())

    assert image_group["spectrum"].attrs.get("data_unit") is None

    loaded_image = read_zarr_image(image_group)
    assert loaded_image.spectrum is not None
    assert loaded_image.spectrum.data_unit is None


def test_dedup_zip_central_directory_keeps_last_entry(tmp_path):
    out = tmp_path / "dup_entries.zip"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate name", category=UserWarning)
        with zipfile.ZipFile(out, mode="w") as zf:
            zf.writestr("zarr.json", "first")
            zf.writestr("other.json", "stable")
            zf.writestr("zarr.json", "second")

            # Deduplicate before close; central directory should only keep latest entry.
            _dedup_zip_central_directory(zf)

    with zipfile.ZipFile(out, mode="r") as zf:
        names = [info.filename for info in zf.infolist()]
        assert names.count("zarr.json") == 1
        assert zf.read("zarr.json") == b"second"


def test_read_zarr_v1_always_transposes_regardless_of_layer_orientation(tmp_path, monkeypatch):
    """v1 data is always transposed: layers in as-is orientation are re-aligned."""
    _patch_find_nearest_x_many(monkeypatch)

    # v1 convention stored data as (width, height, n_spectral).
    # Simulate by writing a v2 file then downgrading format_version.
    cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # height=2, width=3
    spectral_axis = np.linspace(100.0, 400.0, 4, dtype=np.float32)
    spectral_map = SpectralMap.from_cube(cube, spectral_axis=spectral_axis)

    out = tmp_path / "legacy_v1_always_transpose.zarr"
    write_zarr(spectral_map, out_file=out, as_zip=False)

    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="a")
    root.attrs["format_version"] = 1

    # The auto-generated mask is in (2, 3) format. Even though it matches as-is,
    # v1 files are always transposed and layers are re-aligned to the result.
    loaded = read_zarr(out)
    expected_cube = np.transpose(cube, (1, 0, 2))
    assert loaded.map_shape == (3, 2)
    np.testing.assert_array_equal(loaded.cube, expected_cube)


def test_read_zarr_v1_transposes_masks_when_images_drive_orientation(tmp_path, monkeypatch):
    _patch_find_nearest_x_many(monkeypatch)

    cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    spectral_axis = np.linspace(100.0, 400.0, 4, dtype=np.float32)
    spectral_map = SpectralMap.from_cube(cube, spectral_axis=spectral_axis)

    out = tmp_path / "legacy_v1_image_driven_transpose.zarr"
    write_zarr(spectral_map, out_file=out, as_zip=False)

    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="a")
    root.attrs["format_version"] = 1

    original_mask = root["masks/0/mask"][:]

    images_group = root["images"]
    image_ids: list[str] = []
    for i in range(3):
        image_id = f"legacy_locked_{i}"
        image_ids.append(image_id)
        img_group = images_group.require_group(image_id)
        write_zarr_image(
            img_group,
            Image2D(data=np.ones((3, 2), dtype=np.float32), name=image_id, locked=True),
            compressors=get_default_compressors(),
        )
    images_group.attrs["key_order"] = image_ids

    loaded = read_zarr(out)

    transposed_cube = np.transpose(cube, (1, 0, 2))
    assert loaded.map_shape == (3, 2)
    np.testing.assert_array_equal(loaded.cube, transposed_cube)

    expected_mask = np.transpose(original_mask, (1, 0))
    np.testing.assert_array_equal(loaded.masks["0"].get_2Dmask(), expected_mask)

    selected_idxs = np.flatnonzero(expected_mask.ravel())
    expected_mean = transposed_cube.reshape(-1, transposed_cube.shape[-1])[selected_idxs].mean(axis=0)
    np.testing.assert_allclose(np.asarray(loaded.get_spectrum(mask="0").data).ravel(), expected_mean)


def test_read_zarr_v1_aligns_layers_in_as_is_orientation_after_transpose(tmp_path, monkeypatch):
    """Locked images and masks stored in old (width, height) convention are re-aligned."""
    _patch_find_nearest_x_many(monkeypatch)

    cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # height=2, width=3
    spectral_axis = np.linspace(100.0, 400.0, 4, dtype=np.float32)
    spectral_map = SpectralMap.from_cube(cube, spectral_axis=spectral_axis)

    out = tmp_path / "legacy_v1_layer_align.zarr"
    write_zarr(spectral_map, out_file=out, as_zip=False)

    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="a")
    root.attrs["format_version"] = 1

    # One locked image in old (width=3, height=2) convention.
    write_zarr_image(
        root["images"].require_group("legacy_locked_ref"),
        Image2D(data=np.ones((3, 2), dtype=np.float32), name="legacy_locked_ref", locked=True),
        compressors=get_default_compressors(),
    )
    root["images"].attrs["key_order"] = ["legacy_locked_ref"]

    # Extra masks in as-is (2, 3) = (height, width) v2 convention.
    base_mask = root["masks/0/mask"][:]
    extra_mask_ids: list[str] = []
    for i in range(4):
        mask_id = f"extra_mask_{i}"
        extra_mask_ids.append(mask_id)
        write_zarr_mask(
            root["masks"].require_group(mask_id),
            Mask(
                idxs=np.flatnonzero(base_mask.ravel()),
                name=mask_id,
                img_shape=base_mask.shape,
                color="#00ffff",
            ),
            compressors=get_default_compressors(),
        )
    root["masks"].attrs["key_order"] = ["0", *extra_mask_ids]

    loaded = read_zarr(out)

    expected_cube = np.transpose(cube, (1, 0, 2))
    assert loaded.map_shape == (3, 2)
    np.testing.assert_array_equal(loaded.cube, expected_cube)

    # As-is masks are auto-aligned to the transposed target shape.
    expected_mask = np.transpose(base_mask, (1, 0))
    np.testing.assert_array_equal(loaded.masks[extra_mask_ids[0]].get_2Dmask(), expected_mask)


def test_read_zarr_v1_aligns_locked_image_data_when_cube_is_transposed(tmp_path, monkeypatch):
    """Legacy locked image data in (W, H) orientation gets transposed to (H, W)."""
    _patch_find_nearest_x_many(monkeypatch)

    cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # height=2, width=3
    spectral_axis = np.linspace(100.0, 400.0, 4, dtype=np.float32)
    spectral_map = SpectralMap.from_cube(cube, spectral_axis=spectral_axis)

    out = tmp_path / "legacy_v1_image_align.zarr"
    write_zarr(spectral_map, out_file=out, as_zip=False)

    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="a")
    root.attrs["format_version"] = 1

    # Locked image in old (width=3, height=2) convention should be transposed.
    old_data = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    write_zarr_image(
        root["images"].require_group("legacy_intensity"),
        Image2D(data=old_data, name="Intensity", locked=True),
        compressors=get_default_compressors(),
    )
    root["images"].attrs["key_order"] = ["legacy_intensity"]

    loaded = read_zarr(out)

    expected_cube = np.transpose(cube, (1, 0, 2))
    assert loaded.map_shape == (3, 2)
    np.testing.assert_array_equal(loaded.cube, expected_cube)

    # Locked image should be transposed to match cube orientation.
    loaded_img = loaded.images["legacy_intensity"]
    assert loaded_img.data is not None
    assert loaded_img.data.shape == (3, 2)
    np.testing.assert_array_equal(loaded_img.data, old_data.T)


def test_read_zarr_v1_with_single_legacy_intensity_image_in_width_height_format(tmp_path, monkeypatch):
    """Regression test: old v1 files stored data as (width, height, n) and intensity
    images in (width, height) format.  With a single locked image voting for as_is,
    the old heuristic incorrectly skipped the transpose.  The fix always transposes
    so that img_height / img_width and the cube orientation are correct.
    """
    _patch_find_nearest_x_many(monkeypatch)

    # True map: height=3, width=5.
    cube = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)
    spectral_axis = np.linspace(100.0, 400.0, 4, dtype=np.float32)
    spectral_map = SpectralMap.from_cube(cube, spectral_axis=spectral_axis)

    out = tmp_path / "legacy_v1_intensity_bug.zarr"
    write_zarr(spectral_map, out_file=out, as_zip=False)

    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="a")
    root.attrs["format_version"] = 1

    # Overwrite data with the old v1 (width=5, height=3, n) layout.
    data_arr = root["data/data"][:]  # (3, 5, 4) in v2
    v1_data = np.transpose(data_arr, (1, 0, 2))  # (5, 3, 4) - v1 (W, H, n)
    del root["data/data"]
    root["data"].create_array("data", shape=v1_data.shape, dtype=v1_data.dtype)[:] = v1_data

    # Locked intensity image also in old (width=5, height=3) format.
    img_data = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    write_zarr_image(
        root["images"].require_group("intensity"),
        Image2D(data=img_data, name="Intensity", locked=True),
        compressors=get_default_compressors(),
    )
    root["images"].attrs["key_order"] = ["intensity"]

    # Remove the auto-created mask so the only hint is the locked image.
    del root["masks/0"]

    loaded = read_zarr(out)

    assert loaded.map_shape == (3, 5), f"Expected (3, 5), got {loaded.map_shape}"
    np.testing.assert_array_equal(loaded.cube, cube)

    # Intensity image must be transposed to (height=3, width=5).
    loaded_img = loaded.images["intensity"]
    assert loaded_img.data is not None
    assert loaded_img.data.shape == (3, 5), f"Expected (3, 5), got {loaded_img.data.shape}"
    np.testing.assert_array_equal(loaded_img.data, img_data.T)

    # New 2-D images generated from the cube must have shape (height, width) = (3, 5).
    new_image_band = loaded.cube[..., 0]
    assert new_image_band.shape == (3, 5), f"New image band shape wrong: {new_image_band.shape}"


def test_zarr_project_store_roundtrip_reuses_canonical_serialization(tmp_path):
    spectral_map = _make_small_hsi()
    spectral_map.name = "incremental-project"
    spectral_map.smap_metadata = {
        "title": "Incremental",
        "project_id": "transient-project",
        "extra": {"keep_me": 1, "live_update": True},
    }

    image = Image2D(data=np.arange(4, dtype=np.float32).reshape(2, 2), name="band-1")
    image.parent_group = "derived-images"
    mask = Mask(
        idxs=np.array([0, 3]),
        img_shape=spectral_map.map_shape,
        name="roi-1",
        color="#00ff00",
        visible=False,
    )
    mask.parent_group = "regions"
    spectrum = Spectrum(
        x=spectral_map.x,
        data=np.array([5.0, 6.0, 7.0], dtype=np.float32),
        name="external",
        color="#abcdef",
        ignore_sort=True,
    )

    spectral_map.images["band-1"] = image
    spectral_map.images_group["derived-images"] = Image2DGroup(elem_ids=["band-1"], name="Derived", visible=False)
    spectral_map.images.ordered_keys = ["band-1"]
    spectral_map.images_group.ordered_keys = ["derived-images"]

    spectral_map.masks["roi-1"] = mask
    spectral_map.masks_group["regions"] = MaskGroup(elem_ids=["roi-1"], name="Regions", visible=False)
    spectral_map.masks.ordered_keys = list(spectral_map.masks.keys())
    spectral_map.masks_group.ordered_keys = ["regions"]

    spectral_map.spectra["external"] = spectrum

    out = tmp_path / "incremental_store_roundtrip.zarr"
    store = ZarrProjectStore.open(out)
    store.write_intensities(spectral_map.cube)
    store.write_x(spectral_map.x)
    store.write_image("band-1", image)
    store.write_mask("roi-1", mask)
    store.write_spectrum("external", spectrum)
    store.flush_attrs(pipeline=None, name=spectral_map.name, data=spectral_map)

    loaded = read_zarr(out)

    np.testing.assert_array_equal(loaded.cube, spectral_map.cube)
    np.testing.assert_array_equal(loaded.images["band-1"].data, image.data)
    np.testing.assert_array_equal(loaded.masks["roi-1"].get_2Dmask(), mask.get_2Dmask())
    np.testing.assert_array_equal(loaded.spectra["external"].data, spectrum.data)
    assert loaded.images["band-1"].parent_group == "derived-images"
    assert loaded.masks["roi-1"].parent_group == "regions"
    assert loaded.images_group["derived-images"].name == "Derived"
    assert loaded.masks_group["regions"].name == "Regions"

    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="r")
    metadata = root.attrs["metadata"]
    assert metadata["title"] == "Incremental"
    assert metadata["extra"] == {"keep_me": 1}
    assert "project_id" not in metadata
    assert root["spectra/external/data"].shape[-1] == 3


def test_zarr_project_store_deletes_external_spectra(tmp_path):
    spectral_map = _make_small_hsi()
    spectral_map.spectra["external"] = Spectrum(
        x=spectral_map.x,
        data=np.array([5.0, 6.0, 7.0], dtype=np.float32),
        name="external",
        ignore_sort=True,
    )

    out = tmp_path / "incremental_store_delete_spectrum.zarr"
    write_zarr(spectral_map, out_file=out, as_zip=False)

    store = ZarrProjectStore.open(out)
    store.delete_spectrum("external")
    store.flush_attrs(pipeline=None, name=spectral_map.name, data=spectral_map)

    loaded = read_zarr(out)
    assert "external" not in loaded.spectra
