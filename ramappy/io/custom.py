import os
from io import BytesIO
from typing import Any, Literal, cast

import numpy.typing as npt
import polars as pl

from ramappy import units
from ramappy.core import SpectralMap
from ramappy.core.masks import Mask
from ramappy.core.pipeline import ParamsAcceptMask, ParamsAcceptRoIX
from ramappy.io.core import IOParams, IOParamsAsHsi, IOParamsUnits, input_format, output_format
from ramappy.utils.dependencies import _require_pandas_dependencies

from .ownformat import read_ownformat, write_ownformat


class OwnFormatInputParams(IOParams, IOParamsAsHsi, IOParamsUnits):
    """Parameters for reading our flavour of CSV/Feather/Parquet files."""

    img_width: int | None = None
    """Width of the image in pixels (optional)."""

    img_height: int | None = None
    """Height of the image in pixels (optional)."""

    reorder_using_xy_pos: bool = True
    """If `True`, reorder pixels in the images using x, y columns as pixel position (infer index)."""


@input_format(
    "ownformat",
    friendly_name="CSV/Parquet/Feather",
    extensions={"csv", "feather", "parquet"},
    format_params_model=OwnFormatInputParams,
)
def read_own_custom(
    filepath_or_buffer,
    *,
    img_width: int | None = None,
    img_height: int | None = None,
    reorder_using_xy_pos: bool = True,
    dtype: npt.DTypeLike | None = None,
    x_axis_unit: units.type_spectral_quantityunit = None,
    data_unit: units.type_data_quantityunit = None,
    as_hsi: bool = True,
):
    """Read our export format (in a CSV/feather/parquet file).

    The header contains the values of the wavenumbers (as strings) and the position of the
    pixel (either using x and y position or the absolute index).
    Rows contain the spectral data for each pixel, ravelled in a row-major fashion.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        The name of the file to read (any valid string path is acceptable) or an object with a read()
        method, such as a file handle (e.g., via builtin open function) or StringIO.
    img_width, img_height : int or None, optional
        The width and height of the image in pixels.
    reorder_using_xy_pos :  bool, default=True
        Reorder pixels in the images using x, y columns as pixel position (infer index)
    dtype : dtype, optional
        Force Numpy dtype

    """
    return read_ownformat(
        filepath_or_buffer,
        img_width=img_width,
        img_height=img_height,
        reorder_using_xy_pos=reorder_using_xy_pos,
        dtype=dtype,
        x_axis_unit=x_axis_unit,
        data_unit=data_unit,
        as_hsi=as_hsi,
    )


class OwnFormatOutputParams(IOParams, ParamsAcceptRoIX, ParamsAcceptMask):
    """Parameters for writing our flavour of CSV/Feather/Parquet files."""

    format: Literal["csv", "feather", "parquet", "polars", "pandas"] | None = None
    """Format to use as export. If `None` or ``'polars'``, return a `polars.DataFrame`. If ``'pandas'``, return a `pandas.DataFrame`."""

    additional_info: dict[str, Any] | None = None
    """Add new columns to the dataset, containing e.g., filename, source info, etc. (useful when appending multiple dataframes)."""

    add_masks_images: bool = True
    """If `True`, add the masks as images to the exported dataset (as additional columns)."""

    explicit_coordinates: bool = True
    """If `False` write the position of the pixel to columns `x_pos` and `y_pos` (containing, respectively, the x and y coordinates of the pixel). If `True` write the position of the pixel as an index relative to the data ravelled in a row-major (C-style) fashion."""


class OwnFormatFeatherOutputParams(OwnFormatOutputParams):
    """Parameters for writing our flavour of Feather files."""

    format: Literal["feather"] = "feather"


class OwnFormatParquetOutputParams(OwnFormatOutputParams):
    """Parameters for writing our flavour of Parquet files."""

    format: Literal["parquet"] = "parquet"


class OwnFormatCSVOutputParams(OwnFormatOutputParams):
    """Parameters for writing our flavour of CSV files."""

    format: Literal["csv"] = "csv"


@output_format(
    "ownformat",
    friendly_name="DataFrame or Feather/Parquet/CSV",
    extensions={"csv", "feather", "parquet"},
    format_params_model=OwnFormatOutputParams,
)
@output_format(
    "feather",
    friendly_name="Feather",
    extensions={"feather"},
    format_params_model=OwnFormatFeatherOutputParams,
)
@output_format(
    "parquet",
    friendly_name="Parquet",
    extensions={"parquet"},
    format_params_model=OwnFormatParquetOutputParams,
)
@output_format(
    "csv",
    friendly_name="CSV",
    extensions={"csv"},
    format_params_model=OwnFormatCSVOutputParams,
)
def write_own_custom(
    spectral_map: SpectralMap,
    *,
    roi_x: npt.ArrayLike | None = None,
    format: Literal["csv", "feather", "parquet", "polars", "pandas"] | None = None,
    mask: Mask | str | None = None,
    out_file: BytesIO | os.PathLike | str | None = None,
    additional_info: dict[str, Any] | None = None,
    add_masks_images: bool = True,
    explicit_coordinates: bool = True,
) -> pl.DataFrame | Any | BytesIO | os.PathLike:
    """Export spectral data to csv / feather.

    Parameters
    ----------
    format : {'feather', 'parquet', 'csv', 'polars', 'pandas'} or None, default=None
        Format to use as export. If `None` or ``'polars'``, return a
        `polars.DataFrame`. If ``'pandas'``, return a
        `pandas.DataFrame`.
    additional_info : dict or None, default=None
        Add new columns to the dataset, containing e.g., filename, source info, etc.
        (useful when appending multiple dataframes)
    explicit_coordinates : bool, default=False
        If `False` write the position of the pixel to columns `x_pos` and `y_pos` (containing,
        respectively, the x and y coordinates of the pixel)
        If `True` write the position of the pixel as an index relative to the data ravelled in
        a row-major (C-style) fashion.

    Raises
    ------
    ValueError
        If no valid norm has been selected or is missing some required parameters.
    """
    as_pandas = format == "pandas"
    df_format_raw = None if format in {None, "polars", "pandas"} else format
    df_format = cast("Literal['csv', 'feather', 'parquet', 'pandas'] | None", df_format_raw)
    result = write_ownformat(
        spectral_map,
        roi_x=roi_x,
        format=df_format,
        mask=mask,
        out_file=out_file,
        additional_info=additional_info,
        add_masks_images=add_masks_images,
        explicit_coordinates=explicit_coordinates,
    )
    if as_pandas and isinstance(result, pl.DataFrame):
        _require_pandas_dependencies()
        return result.to_pandas()
    return result
