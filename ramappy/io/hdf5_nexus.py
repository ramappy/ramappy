from pathlib import Path

import h5py
import numpy as np
import PIL.Image

from ramappy import units
from ramappy.core import SpectralMap
from ramappy.core.images2d import Image2D, SpatialGrid
from ramappy.io.core import IOParams, IOParamsAsHsi, input_format, metadata_with_extra
from ramappy.io.matlab import sniff_nexus
from ramappy.units import Unit


class HDF5NeXusParams(IOParams, IOParamsAsHsi):
    """Parameters for reading HDF5 (NeXus) files."""

    name: str | None = None
    """Optional name for the imported spectral map. If not provided, a name will be inferred from the file or metadata."""


@input_format(
    "hdf5_nexus",
    friendly_name="HDF5 (NeXus)",
    extensions={"h5", "hdf5"},
    sniffer=sniff_nexus,
    has_custom_params=False,
    format_params_model=HDF5NeXusParams,
)
def read_hdf5_nexus(filepath_or_buffer, *, as_hsi: bool = True, name: str | None = None):
    if not as_hsi:
        raise ValueError("HDF5 (NeXus) only supports spectral maps (as_hsi=True)")

    with h5py.File(filepath_or_buffer, "r") as f:
        data_group = f.get("ENTRY/data", {})

        def _decode_scalar(value):
            if isinstance(value, np.generic):
                value = value.item()  # np.bytes_ -> bytes, etc.
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value

        def _decode_axes_attr(value) -> list[str] | None:
            if value is None:
                return None

            if isinstance(value, (list, tuple, np.ndarray)):
                items = [_decode_scalar(v) for v in np.asarray(value).ravel().tolist()]
            else:
                text = str(_decode_scalar(value)).strip()
                if not text:
                    return None
                if ":" in text:
                    items = text.split(":")
                elif "," in text:
                    items = text.split(",")
                else:
                    items = text.split()

            axes = [s for item in items if (s := str(item).strip()) and s != "."]
            return axes or None

        def _canonical_axis_name(axis_name: str | None) -> str | None:
            if axis_name is None:
                return None

            token = str(axis_name).strip().lower()
            if not token:
                return None

            if "/" in token:
                token = token.rsplit("/", 1)[-1]

            aliases = {
                "x": "X",
                "y": "Y",
                "s": "S",
                "spec": "S",
                "spectral": "S",
                "spectrum": "S",
                "ram": "S",
                "raman": "S",
                "raman_shift": "S",
                "wavenumber": "S",
                "wavelength": "S",
                "energy": "S",
            }
            return aliases.get(token)

        def _coord_length(coords) -> int | None:
            if coords is None:
                return None
            arr = np.asarray(coords)
            if arr.ndim == 1:
                return int(arr.size)
            return None

        # HS cube
        if "intensity" not in data_group:
            raise ValueError("Expected ENTRY/data/intensity dataset")
        intensity = np.asarray(data_group["intensity"])

        # spectral axis
        if "raman_shift" not in data_group:
            raise ValueError("Expected ENTRY/data/raman_shift dataset")
        x_axis = np.asarray(data_group["raman_shift"]).squeeze()

        hx = data_group.get("x")
        hy = data_group.get("y")

        axes_attr = _decode_axes_attr(data_group.attrs.get("axes"))
        if axes_attr is None:
            axes_attr = _decode_axes_attr(data_group["intensity"].attrs.get("axes"))

        # NeXus HDF5 supports map data only
        if intensity.ndim != 3:
            raise ValueError(f"Expected ENTRY/data/intensity as a 3D map, got shape {intensity.shape}")

        dim_permutation = None
        if axes_attr is not None and len(axes_attr) == intensity.ndim:
            canonical_axes = [_canonical_axis_name(axis) for axis in axes_attr]
            # sorted() needs str, filter out any None before comparing
            canonical_str = [a for a in canonical_axes if a is not None]
            if sorted(canonical_str) == ["S", "X", "Y"] and None not in canonical_axes:
                dim_permutation = tuple(canonical_axes.index(axis) for axis in ("Y", "X", "S"))

        if dim_permutation is None:
            hx_len = _coord_length(hx)
            hy_len = _coord_length(hy)
            if hx_len == intensity.shape[0] and hy_len == intensity.shape[1]:
                # Legacy NeXus export order is (x, y, spectral); normalize to
                # SpectralMap convention (height=y, width=x, spectral).
                dim_permutation = (1, 0, 2)
            elif hy_len == intensity.shape[0] and hx_len == intensity.shape[1]:
                dim_permutation = (0, 1, 2)
            else:
                # Conservative fallback for historical exports without explicit
                # axis metadata.
                dim_permutation = (1, 0, 2)

        if dim_permutation != (0, 1, 2):
            intensity = np.transpose(intensity, dim_permutation)

        if intensity.shape[2] != len(x_axis):
            raise ValueError(
                f"Spectral axis length {len(x_axis)} does not match intensity cube shape {intensity.shape}"
            )

        # Get grids
        hx_arr = np.asarray(hx) if hx is not None else np.arange(intensity.shape[1])
        hy_arr = np.asarray(hy) if hy is not None else np.arange(intensity.shape[0])

        def _dataset_to_value(dataset: h5py.Dataset):
            arr = np.asarray(dataset)
            if arr.shape == () or arr.size == 1:
                return _decode_scalar(arr.item() if hasattr(arr, "item") else arr)

            # Keep metadata compact: include only small datasets.
            if arr.size <= 128:
                if arr.dtype.kind in {"S", "U"}:
                    return [_decode_scalar(v) for v in arr.ravel().tolist()]
                return arr.tolist()

            return None

        def get_group_dict(group, *, skip_keys: set[str] | None = None):
            if group is None:
                return {}
            skip_keys = skip_keys or set()
            res = {}
            for k, v in group.attrs.items():
                if k == "NX_class":
                    # skip NX_class (we are shamelessly ignoring NeXus ontology for now, just raw metadata extraction)
                    continue
                res[k] = _decode_scalar(v)
            for k, v in group.items():
                if k in skip_keys:
                    continue
                if isinstance(v, h5py.Group):
                    nested = get_group_dict(v)
                    if nested:
                        res[k] = nested
                    continue
                if isinstance(v, h5py.Dataset):
                    parsed = _dataset_to_value(v)
                    if parsed is not None:
                        res[k] = parsed
            return res

        proj_meta = get_group_dict(f.get("PROJECT"))
        samp_meta = get_group_dict(f.get("SAMPLE"))
        acq_meta = get_group_dict(f.get("ENTRY"), skip_keys={"data", "auxiliary"})

        images = {}

        def _estimate_spacing(coords: np.ndarray | None) -> float | None:
            if coords is None:
                return None
            arr = np.asarray(coords, dtype=float)
            if arr.size <= 1:
                return None

            diffs = []
            if arr.ndim == 1:
                diffs.append(np.diff(arr))
            else:
                if arr.shape[-1] > 1:
                    diffs.append(np.diff(arr, axis=-1).ravel())
                if arr.shape[0] > 1:
                    diffs.append(np.diff(arr, axis=0).ravel())

            if not diffs:
                return None

            merged = np.concatenate([d.ravel() for d in diffs])
            merged = np.abs(merged[np.isfinite(merged)])
            merged = merged[merged > 0]
            if merged.size == 0:
                return None
            return float(np.median(merged))

        px_size_x = _estimate_spacing(hx_arr if hx is not None else None)
        px_size_y = _estimate_spacing(hy_arr if hy is not None else None)

        # white light image - fetch once, derive pixel size and import image
        wl_group = f.get("ENTRY/auxiliary/white_light")
        if wl_group and "image" in wl_group:
            wl_img_ds = wl_group["image"]
            dx = wl_img_ds.attrs.get("dx_um_per_px")
            dy = wl_img_ds.attrs.get("dy_um_per_px")
            if dx is not None and dy is not None:
                px_size_x = float(_decode_scalar(dx))
                px_size_y = float(_decode_scalar(dy))

        if px_size_x is None and px_size_y is None:
            px_size_x = px_size_y = 1.0
        elif px_size_x is None:
            px_size_x = px_size_y if px_size_y is not None else 1.0
        elif px_size_y is None:
            px_size_y = px_size_x

        spatial_grid = SpatialGrid(
            pixel_size_y=float(px_size_y),  # type: ignore[arg-type]
            pixel_size_x=float(px_size_x),  # type: ignore[arg-type]
            spatial_unit=Unit.MICROMETER,
        )

        # reuse the already-fetched wl_group for the actual image import
        if wl_group and "image" in wl_group:
            wl_img = np.asarray(wl_group["image"])
            # try to crop to the grid
            wx = np.asarray(wl_group.get("x")) if "x" in wl_group else None
            wy = np.asarray(wl_group.get("y")) if "y" in wl_group else None

            # Very basic assumption to crop or at least align
            if wx is not None and wy is not None and hx is not None and hy is not None:
                try:
                    # Filter grid points covering HSI
                    # Just taking coordinates as 1D arrays for min/max
                    hsi_x_min, hsi_x_max = np.min(hx_arr), np.max(hx_arr)
                    hsi_y_min, hsi_y_max = np.min(hy_arr), np.max(hy_arr)

                    if wx.ndim == 1 and wy.ndim == 1:
                        x_mask = (wx >= hsi_x_min) & (wx <= hsi_x_max)
                        y_mask = (wy >= hsi_y_min) & (wy <= hsi_y_max)
                        wl_img_cropped = wl_img[y_mask][:, x_mask]
                    else:
                        wl_img_cropped = wl_img
                except Exception:
                    wl_img_cropped = wl_img
            else:
                wl_img_cropped = wl_img

            wl_image = PIL.Image.fromarray(wl_img_cropped) if isinstance(wl_img_cropped, np.ndarray) else wl_img_cropped
            images["white_light"] = Image2D(
                image=wl_image,
                locked=True,
                name="White Light",
                visible=False,
            )

        def _as_nonempty_string(value) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text if text else None

        entry_title = _as_nonempty_string(acq_meta.get("title"))
        project_title = _as_nonempty_string(proj_meta.get("title") or proj_meta.get("name"))
        sample_title = _as_nonempty_string(samp_meta.get("name"))
        source_stem = Path(filepath_or_buffer).stem if isinstance(filepath_or_buffer, (str, Path)) else None

        inferred_name = (
            _as_nonempty_string(name)
            or entry_title
            or project_title
            or sample_title
            or _as_nonempty_string(source_stem)
            or "Imported HDF5 Cube"
        )

        return SpectralMap(
            x=x_axis,
            data=intensity.reshape(-1, len(x_axis)),
            img_width=intensity.shape[1],
            img_height=intensity.shape[0],
            name=inferred_name,
            metadata=metadata_with_extra(
                title=inferred_name,
                original_filename=Path(filepath_or_buffer).name
                if isinstance(filepath_or_buffer, (str, Path))
                else None,
                extra={
                    "Project": proj_meta,
                    "Sample": samp_meta,
                    "Acquisition": acq_meta,
                },
            ),
            images=images,
            x_axis_unit=units.QuantityUnit(units.Quantity.WAVENUMBER, Unit.CM_1),
            spatial_grid=spatial_grid,
        )
