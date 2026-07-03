"""Reader for the "ownformat" table layout."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from ramappy import units
from ramappy.core import SpectralMap, Spectrum
from ramappy.core.images2d import Image2D, Image2DGroup
from ramappy.core.masks import Mask, MaskGroup

from .columns import parse_item_column, split_spectral_and_metadata_columns

MetadataFrame = pl.DataFrame


def _series_to_numpy(values: Any) -> np.ndarray:
    """Convert a Series-like object to a NumPy array."""
    to_numpy = getattr(values, "to_numpy", None)
    if callable(to_numpy):
        return to_numpy()
    return np.asarray(values)


def _is_file_like(obj: Any) -> bool:
    return hasattr(obj, "read") and callable(obj.read)


def _read_df_from_path(path: Path) -> pl.DataFrame:
    readers_by_ext: Mapping[str, Any] = {
        ".csv": pl.read_csv,
        ".feather": pl.read_ipc,
        ".parquet": pl.read_parquet,
    }
    reader = readers_by_ext.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Ownformat: unsupported file extension {path.suffix!r}")
    return reader(str(path))


def _read_df_from_file_like(buf: Any) -> pl.DataFrame:
    # We don't know the format, try all supported readers.
    for reader in (pl.read_ipc, pl.read_parquet, pl.read_csv):
        try:
            if hasattr(buf, "seek"):
                buf.seek(0)
            return reader(buf)
        except Exception:
            continue

    # As a fallback, read bytes and probe again (handles non-seekable streams).
    if hasattr(buf, "seek"):
        buf.seek(0)
    raw = buf.read()
    tmp = BytesIO(raw)
    for reader in (pl.read_ipc, pl.read_parquet, pl.read_csv):
        try:
            tmp.seek(0)
            return reader(tmp)
        except Exception:
            continue
    raise ValueError("Ownformat: could not read file-like object")


def _read_df(filepath_or_buffer: Any) -> pl.DataFrame:
    """Read a tabular file into a Polars DataFrame."""
    if isinstance(filepath_or_buffer, pl.DataFrame):
        return filepath_or_buffer
    if type(filepath_or_buffer).__module__.split(".")[0] == "pandas":
        return pl.from_pandas(filepath_or_buffer)
    if isinstance(filepath_or_buffer, (str, Path)):
        return _read_df_from_path(Path(filepath_or_buffer))
    if _is_file_like(filepath_or_buffer):
        return _read_df_from_file_like(filepath_or_buffer)
    raise ValueError(f"Ownformat: unsupported input type {type(filepath_or_buffer)!r}")


def _ensure_int_like(a: npt.ArrayLike, *, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.size == 0:
        return arr.astype(np.intp)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.intp)

    # float columns are common in CSVs; accept integer-like values.
    rounded = np.round(arr)
    if not np.allclose(arr, rounded):
        raise ValueError(f"Ownformat: {name} must be integer-like")
    return rounded.astype(np.intp)


def _infer_shape_from_size(img_size: int, img_width: int | None, img_height: int | None) -> tuple[int, int]:
    if img_width is None and img_height is None:
        # assume square only if exact
        root = round(np.sqrt(img_size))
        if root * root != img_size:
            raise ValueError(
                "Ownformat: img_width/img_height not provided and image is not a perfect square; cannot infer shape"
            )
        return int(root), int(root)

    if img_width is None:
        if img_height is None or img_height <= 0:
            raise ValueError("Ownformat: invalid img_height")
        if img_size % img_height != 0:
            raise ValueError("Ownformat: img_size is not divisible by img_height; cannot infer img_width")
        return img_size // img_height, img_height

    if img_height is None:
        if img_width <= 0:
            raise ValueError("Ownformat: invalid img_width")
        if img_size % img_width != 0:
            raise ValueError("Ownformat: img_size is not divisible by img_width; cannot infer img_height")
        return img_width, img_size // img_width

    if img_width * img_height != img_size:
        raise ValueError("Ownformat: provided img_width/img_height do not match data size")

    return img_width, img_height


def _place_rows_into_full_array(
    data_rows: np.ndarray,
    pixel_index: np.ndarray,
    img_size: int,
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Place rows into a full (img_size, n_spectral) array using pixel_index mapping."""
    n_rows, n_spec = data_rows.shape

    # If any pixels are missing, use float so we can represent missing values.
    if n_rows != img_size and not np.issubdtype(data_rows.dtype, np.floating):
        data_rows = data_rows.astype(float)

    full = np.full((img_size, n_spec), fill_value, dtype=data_rows.dtype)

    if pixel_index.size != n_rows:
        raise ValueError("Ownformat: pixel_index length does not match number of rows")

    # Guard against out-of-range indices
    if pixel_index.size and (pixel_index.min() < 0 or pixel_index.max() >= img_size):
        raise ValueError("Ownformat: pixel_index out of bounds")

    full[pixel_index] = data_rows
    return full


def _metadata_to_full_vector(
    values: np.ndarray,
    row_pixel_index: np.ndarray | None,
    img_size: int,
    *,
    default: Any,
) -> np.ndarray:
    if row_pixel_index is None:
        if values.shape[0] != img_size:
            raise ValueError("Ownformat: metadata length does not match inferred image size")
        return values

    full = np.full((img_size,), default, dtype=values.dtype)
    full[row_pixel_index] = values
    return full


def _build_geometry(
    *,
    metadata: MetadataFrame,
    img_width: int | None,
    img_height: int | None,
    reorder_using_xy_pos: bool,
) -> tuple[np.ndarray | None, int, int, int]:
    """Return (row_pixel_index, img_width, img_height, img_size)."""
    row_pixel_index: np.ndarray | None = None
    if "pixel_index" in metadata.columns:
        row_pixel_index = _ensure_int_like(_series_to_numpy(metadata["pixel_index"]), name="pixel_index")
        img_size = int(row_pixel_index.max()) + 1 if row_pixel_index.size else 0
        img_width, img_height = _infer_shape_from_size(img_size, img_width, img_height)
        return row_pixel_index, img_width, img_height, img_size

    row_pixel_index, img_width, img_height = _coords_to_row_pixel_index(
        metadata,
        img_width=img_width,
        img_height=img_height,
        reorder_using_xy_pos=reorder_using_xy_pos,
    )
    img_size = int(img_width * img_height)
    return row_pixel_index, img_width, img_height, img_size


def _get_or_create_group(
    *,
    group_id: str,
    group_name: str | None,
    groups: dict[str, Any],
    factory: type[Image2DGroup] | type[MaskGroup],
) -> Image2DGroup | MaskGroup:
    group = groups.get(group_id)
    if group is None:
        group = factory(name=group_name or "Group", elem_ids=[])
        groups[group_id] = group
    return group


def _extract_table(filepath_or_buffer: Any) -> tuple[np.ndarray, np.ndarray, MetadataFrame]:
    df = _read_df(filepath_or_buffer)
    spectral_cols, wn, meta_cols = split_spectral_and_metadata_columns(df.columns)
    data = df.select(spectral_cols).to_numpy()
    metadata: MetadataFrame = df.select(meta_cols) if meta_cols else pl.DataFrame()
    return wn, data, metadata


def _parse_masks_images(
    *,
    metadata: MetadataFrame,
    row_pixel_index: np.ndarray | None,
    img_size: int,
    img_width: int,
    img_height: int,
) -> tuple[
    dict[str, Image2D],
    dict[str, Mask],
    dict[str, Image2DGroup],
    dict[str, MaskGroup],
]:
    image_groups: dict[str, Image2DGroup] = {}
    mask_groups: dict[str, MaskGroup] = {}
    images: dict[str, Image2D] = {}
    masks: dict[str, Mask] = {}

    for col in metadata.columns:
        parsed = parse_item_column(col)
        if parsed is None:
            continue

        col_values = _series_to_numpy(metadata[col])
        default = False if parsed.kind == "mask" else np.nan
        col_full = _metadata_to_full_vector(col_values, row_pixel_index, img_size, default=default)

        group: Image2DGroup | MaskGroup | None = None
        if parsed.group_id is not None:
            if parsed.kind == "image":
                group = _get_or_create_group(
                    group_id=parsed.group_id,
                    group_name=parsed.group_name,
                    groups=image_groups,
                    factory=Image2DGroup,
                )
            else:
                group = _get_or_create_group(
                    group_id=parsed.group_id,
                    group_name=parsed.group_name,
                    groups=mask_groups,
                    factory=MaskGroup,
                )

        if parsed.kind == "image":
            images[parsed.item_id] = Image2D(
                data=np.asarray(col_full).reshape(img_height, img_width),
                parent_group=parsed.group_id,
                name=parsed.item_name,
            )
        else:
            mask_bool = np.asarray(col_full).astype(bool)
            masks[parsed.item_id] = Mask(
                idxs=np.flatnonzero(mask_bool),
                parent_group=parsed.group_id,
                name=parsed.item_name,
                img_height=img_height,
                img_width=img_width,
            )

        if group is not None:
            if group.elem_ids is None:
                group.elem_ids = []
            group.elem_ids.append(parsed.item_id)

    return images, masks, image_groups, mask_groups


def _coords_to_row_pixel_index(
    metadata: MetadataFrame,
    *,
    img_width: int | None,
    img_height: int | None,
    reorder_using_xy_pos: bool,
) -> tuple[np.ndarray | None, int, int]:
    x_col = "x" if "x" in metadata.columns else "x_pos" if "x_pos" in metadata.columns else None
    y_col = "y" if "y" in metadata.columns else "y_pos" if "y_pos" in metadata.columns else None
    if x_col is None or y_col is None:
        raise ValueError("Ownformat: missing coordinate columns (x/x_pos and y/y_pos) or pixel_index")

    x_pos = _ensure_int_like(_series_to_numpy(metadata[x_col]), name=x_col)
    y_pos = _ensure_int_like(_series_to_numpy(metadata[y_col]), name=y_col)

    if img_width is None:
        img_width = int(np.unique(x_pos).size)
    if img_height is None:
        img_height = int(np.unique(y_pos).size)

    if img_width <= 0 or img_height <= 0:
        raise ValueError("Ownformat: invalid inferred image shape")

    if not reorder_using_xy_pos:
        # No reordering requested: assume the table is already in raster order.
        # Returning None enables a fast-path later (avoid allocating a full array).
        # Keep the previous semantics: if the file is not a full raster, we'll error downstream.
        return None, img_width, img_height

    in_bounds = (
        x_pos.size
        and (x_pos.min() >= 0)
        and (y_pos.min() >= 0)
        and (x_pos.max() < img_width)
        and (y_pos.max() < img_height)
    )
    if in_bounds:
        return (y_pos * img_width + x_pos).astype(np.intp), img_width, img_height

    x_grid = np.sort(np.unique(x_pos))
    y_grid = np.sort(np.unique(y_pos))
    if x_grid.size != img_width or y_grid.size != img_height:
        raise ValueError("Ownformat: coordinate grid does not match provided/inferred shape")
    x_idx = np.searchsorted(x_grid, x_pos)
    y_idx = np.searchsorted(y_grid, y_pos)
    return (y_idx * img_width + x_idx).astype(np.intp), img_width, img_height


def read_ownformat(
    filepath_or_buffer: Any,
    *,
    img_width: int | None = None,
    img_height: int | None = None,
    reorder_using_xy_pos: bool = True,
    dtype: npt.DTypeLike | None = None,
    x_axis_unit: units.type_spectral_quantityunit = None,
    data_unit: units.type_data_quantityunit = None,
    as_hsi: bool = True,
) -> SpectralMap | Spectrum:
    """Read the ownformat export format from a table-like input."""

    wn, data, metadata = _extract_table(filepath_or_buffer)

    if dtype is not None:
        data = data.astype(dtype)

    name = (
        filepath_or_buffer.name
        if isinstance(filepath_or_buffer, Path)
        else Path(filepath_or_buffer).name
        if isinstance(filepath_or_buffer, (str, Path))
        else None
    )

    if not as_hsi:
        return Spectrum(x=wn, data=data, x_axis_unit=x_axis_unit, data_unit=data_unit, name=name)

    row_pixel_index, img_width, img_height, img_size = _build_geometry(
        metadata=metadata,
        img_width=img_width,
        img_height=img_height,
        reorder_using_xy_pos=reorder_using_xy_pos,
    )

    # Fast-path: full raster order, no missing pixels => avoid allocating/scattering.
    if row_pixel_index is None:
        if data.shape[0] != img_size:
            raise ValueError("Ownformat: table is not a full raster; provide pixel_index or set reorder_using_xy_pos")
        data_full = data
    else:
        data_full = _place_rows_into_full_array(data, row_pixel_index, img_size)
    images, masks, image_groups, mask_groups = _parse_masks_images(
        metadata=metadata,
        row_pixel_index=row_pixel_index,
        img_size=img_size,
        img_width=img_width,
        img_height=img_height,
    )

    return SpectralMap(
        x=wn,
        data=data_full,
        img_width=img_width,
        img_height=img_height,
        x_axis_unit=x_axis_unit,
        data_unit=data_unit,
        masks=masks,
        masks_group=mask_groups,
        images=images,
        images_group=image_groups,
        metadata={"original_filename": name},
    )
