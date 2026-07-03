"""Writer for the "ownformat" table layout."""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from ramappy.core import SpectralMap
from ramappy.core.masks import Mask
from ramappy.utils.dependencies import _require_pandas_dependencies


def _normalize_out_file(out_file: BytesIO | os.PathLike | str | None) -> BytesIO | os.PathLike | str | None:
    if out_file is None:
        return None
    if isinstance(out_file, (BytesIO, str, os.PathLike)):
        return out_file
    raise ValueError("`out_file` is not valid")


def _pixel_indices_from_selection(pixel_idx: np.ndarray | slice) -> np.ndarray:
    if isinstance(pixel_idx, slice):
        # caller should override with full range
        return np.empty((0,), dtype=np.intp)
    arr = np.asarray(pixel_idx)
    if arr.ndim == 0:
        return np.asarray([int(arr)], dtype=np.intp)
    return arr.astype(np.intp)


def _polars_sink(out_file: BytesIO | os.PathLike | str) -> Any:
    """Return a value suitable for Polars writers (path or binary IO)."""
    if isinstance(out_file, BytesIO):
        return out_file
    # PathLike is accepted by Polars at runtime, but type checkers are picky.
    return str(out_file)


def _add_coordinates(
    df: pl.DataFrame,
    *,
    selected_pixels: np.ndarray,
    spectral_map: SpectralMap,
    explicit_coordinates: bool,
) -> pl.DataFrame:
    if explicit_coordinates:
        y, x = np.unravel_index(selected_pixels, (spectral_map.img_height, spectral_map.img_width))
        return df.with_columns(
            pl.Series("x_pos", x.astype(np.intp)),
            pl.Series("y_pos", y.astype(np.intp)),
        )
    return df.with_columns(pl.Series("pixel_index", selected_pixels.astype(np.intp)))


def _add_masks_images(df: pl.DataFrame, *, spectral_map: SpectralMap, selected_pixels: np.ndarray) -> pl.DataFrame:
    total_pixels = int(spectral_map.img_height * spectral_map.img_width)

    for mask_id, m in spectral_map.masks.items():
        group_suffix = ""
        if m.parent_group is not None and m.parent_group in spectral_map.masks_group:
            g = spectral_map.masks_group[m.parent_group]
            group_suffix = f"(group[{m.parent_group}]: {g.name})"
        col_name = f"mask[{mask_id}]: {m.name} {group_suffix}".rstrip()

        rolled = np.zeros(total_pixels, dtype=bool)
        rolled[m.idxs] = True
        df = df.with_columns(pl.Series(col_name, rolled[selected_pixels]))

    for image_id, image in spectral_map.images.items():
        if image.data is None:
            continue
        group_suffix = ""
        if image.parent_group is not None and image.parent_group in spectral_map.images_group:
            img_g = spectral_map.images_group[image.parent_group]
            group_suffix = f"(group[{image.parent_group}]: {img_g.name})"
        col_name = f"image[{image_id}]: {image.name} {group_suffix}".rstrip()

        flat = np.asarray(image.data).reshape(-1)
        df = df.with_columns(pl.Series(col_name, flat[selected_pixels]))

    return df


def _write_table(
    df: pl.DataFrame,
    *,
    format: str,
    out_file: BytesIO | os.PathLike | str,
) -> BytesIO | os.PathLike | str:
    sink = _polars_sink(out_file)
    if format == "feather":
        df.write_ipc(sink, compression="zstd")
    elif format == "parquet":
        df.write_parquet(sink, compression="zstd", compression_level=10, statistics=False)
    elif format == "csv":
        df.write_csv(sink)
    else:
        raise ValueError(f'Format "{format}" not supported')
    return out_file


def write_ownformat(
    spectral_map: SpectralMap,
    *,
    roi_x: npt.ArrayLike | None = None,
    format: Literal["csv", "feather", "parquet", "pandas"] | None = None,
    mask: Mask | str | None = None,
    out_file: BytesIO | os.PathLike | str | None = None,
    additional_info: dict[str, Any] | None = None,
    add_masks_images: bool = True,
    explicit_coordinates: bool = True,
) -> pl.DataFrame | BytesIO | os.PathLike | str:
    """Export SpectralMap spectral data to a Polars DataFrame or to csv/feather/parquet/pandas."""

    out_file = _normalize_out_file(out_file)

    # Normalize mask parameter
    mask_obj = spectral_map.get_mask(mask if isinstance(mask, str) else None) if not isinstance(mask, Mask) else mask
    if mask_obj is None:
        mask_obj = spectral_map.new_mask()

    idxs_tuple, x_idx, pixel_idx = spectral_map.get_indices(mask_obj, roi_x)
    intensities = spectral_map.data[idxs_tuple]

    total_pixels = int(spectral_map.img_height * spectral_map.img_width)
    selected_pixels = (
        np.arange(total_pixels, dtype=np.intp)
        if isinstance(pixel_idx, slice)
        else _pixel_indices_from_selection(pixel_idx)
    )

    x_vals = np.asarray(spectral_map.x[x_idx])
    spectral_col_names = [str(v) for v in x_vals]

    df = pl.DataFrame(intensities)
    df.columns = spectral_col_names

    df = _add_coordinates(
        df, selected_pixels=selected_pixels, spectral_map=spectral_map, explicit_coordinates=explicit_coordinates
    )

    if add_masks_images:
        df = _add_masks_images(df, spectral_map=spectral_map, selected_pixels=selected_pixels)

    if additional_info:
        for k, v in additional_info.items():
            df = df.with_columns(pl.lit(v).alias(k))

    if format is None:
        return df

    if format == "pandas":
        _require_pandas_dependencies()
        return df.to_pandas()

    # Write to a buffer/path
    if out_file is None:
        out_file = BytesIO()

    return _write_table(df, format=format, out_file=out_file)
