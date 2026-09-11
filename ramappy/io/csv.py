import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from ramappy import units
from ramappy.core import SpectralMap, Spectrum
from ramappy.io.core import (
    IOParams,
    IOParamsAsHsi,
    IOParamsUnits,
    SpectrumType,
    input_format,
    metadata_as_extra,
    output_format,
)
from ramappy.utils.dependencies import _require_pandas_dependencies

NPDTYPE_TO_PLDTYPE = {
    np.float32: pl.Float32,
    np.float64: pl.Float64,
    np.int32: pl.Int32,
    np.int64: pl.Int64,
    np.uint32: pl.UInt32,
    np.uint64: pl.UInt64,
    np.int16: pl.Int16,
    np.uint16: pl.UInt16,
    np.int8: pl.Int8,
    np.uint8: pl.UInt8,
}


class CsvInputParams(IOParams, IOParamsAsHsi, IOParamsUnits):
    """Parameters for reading CSV files."""

    img_width: int | None = None
    """Width of the spatial grid (number of pixels in x-direction). If not specified and the dataset is a map, it assumes a square geometry."""

    img_height: int | None = None
    """Height of the spatial grid (number of pixels in y-direction). If not specified and the dataset is a map, it assumes a square geometry."""

    has_x_axis: bool = True
    """If `True`, reads the first column (or first row in long format) as the spectral calibration axis (e.g., wavenumbers)."""

    invert_x_axis: bool = False
    """If `True`, reverses the order of the spectral axis (if `has_x_axis` is `False`)."""

    delimiter: Literal[r"\t", r"\s+", ",", ";"] | None = None
    """Field delimiter. Automatically parsed if `None`."""

    csv_format: Literal["wide", "long", "auto"] = "wide"
    """Format of the CSV file. 'wide' means columns represent pixels and rows represent wavenumbers, while 'long' means rows represent pixels and columns represent wavenumbers. 'auto' attempts to infer the format based on the data shape."""

    header: int | Literal["infer"] | None = None
    """Row index to use as column names. If `None`, no header is assumed. If `0` or `'infer'`, the first row is used as the header."""

    scan_pattern: Literal["bidirectional", "raster"] = "bidirectional"
    """Unraveling pattern used during acquisition. 'bidirectional' means that every other row is reversed, while 'raster' means that all rows are in the same order."""

    name: str | None = None
    """Custom name for the spectrum or map. If not provided, the filename will be used as the name."""


@input_format(
    "csv", friendly_name="CSV", extensions={"csv", "tsv", "txt", "dat", "asc"}, format_params_model=CsvInputParams
)
def read_csv(
    filepath_or_buffer,
    *,
    img_width: int | None = None,
    img_height: int | None = None,
    has_x_axis: bool = True,
    invert_x_axis: bool = False,
    delimiter: str | None = None,
    csv_format: Literal["wide", "long", "auto"] = "wide",
    header: int | Literal["infer"] | None = None,
    scan_pattern: Literal["bidirectional", "raster"] = "bidirectional",
    dtype: npt.DTypeLike = np.float32,
    name: str | None = None,
    x_axis_unit: units.type_spectral_quantityunit = None,
    data_unit: units.type_data_quantityunit = None,
    as_hsi: bool = True,
) -> Spectrum | SpectralMap:
    """Read custom CSV file.

    CSV file, where the first column represents the calibration axis with the measured wavenumbers;
    the remaining img_width * img_height columns store the measured intensities for the pixels, each
    row corresponding to a measured wn (single measurement).

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        The name of the file to read (any valid string path is acceptable) or an object with a read()
        method, such as a file handle (e.g., via builtin open function) or StringIO.
    img_width, img_height : int
        The width and height of the image in pixels.
    delimiter: str
        The delimiter to use while parsing the CSV.
        Example: '\\s+', r'\t'
    csv_format: {'auto', 'wide', 'long'}, default='wide'
        TODO: The format in which the data are written...
    scan_pattern : {'bidirectional', 'raster'}, default='bidirectional'
        The pattern used during acquisition (used to unravel data).
    dtype : dtype, default = np.float32
        The dtype of the output data.

    Raises
    ------
    ValueError
        If img_width and img_height are not provided and the map is not square.

    Returns
    -------
    SpectralMap or Spectrum
    """
    if delimiter == r"\s+":
        # polars does not support regex separators; numpy handles multi-whitespace natively
        skip = (header + 1) if isinstance(header, int) else 0
        data = np.genfromtxt(filepath_or_buffer, dtype=dtype, skip_header=skip)
    else:
        # header=None means "no header row"; header=0 or "infer" means first row is header
        pl_has_header = header in {0, "infer"}
        pl_skip_rows = header if isinstance(header, int) and header > 0 else 0
        pl_kwargs: dict[str, Any] = {"has_header": pl_has_header, "skip_rows": pl_skip_rows}
        if delimiter is not None and len(delimiter) == 1:
            pl_kwargs["separator"] = delimiter
        data = pl.read_csv(filepath_or_buffer, **pl_kwargs).to_numpy().astype(dtype)

    if csv_format == "auto":
        if img_width is None or img_height is None:
            # assume it's a square map
            d_shape = data.shape[1] - (1 if has_x_axis else 0)
            d_candidate = int(np.sqrt(d_shape))
            if d_candidate**2 == d_shape:
                csv_format = "wide"
            else:
                d_shape = data.shape[0] - (1 if has_x_axis else 0)
                d_candidate = int(np.sqrt(d_shape))
                if d_candidate**2 == d_shape:
                    csv_format = "long"
                else:
                    raise ValueError(
                        "Map does not appear to be square, please provide map width and height and/or set the appropriate header."
                    )
            img_height = d_candidate
            img_width = d_candidate
        elif img_width * img_height + (1 if has_x_axis else 0) == data.shape[1]:
            csv_format = "wide"
            if img_width * img_height + (1 if has_x_axis else 0) == data.shape[0]:
                warnings.warn("Ambiguous data size: could not univocally detect format type", stacklevel=2)
        else:
            csv_format = "long"
    elif img_width is None or img_height is None:
        # assume it's a square map
        axis = 1 if csv_format == "wide" else 0
        d_shape = data.shape[axis] - (1 if has_x_axis else 0)
        d_candidate = int(np.sqrt(d_shape))
        if d_candidate**2 == d_shape:
            img_height = d_candidate
            img_width = d_candidate
        else:
            raise ValueError(
                "Map does not appear to be square, please provide map width and height and/or set the appropriate header."
            )

    if csv_format == "wide":
        # wide format: columns are pixels, rows are wavenumbers

        if has_x_axis:
            # first col is the x-axis
            x = data[:, 0]
            # each row is a img_width x img_height pixels
            intensities = data[:, 1:].T
        else:
            intensities = data.T
            x = np.arange(data.shape[0])
            if invert_x_axis:
                x = np.flip(x)
    elif csv_format == "long":
        # long format: columns are wavenumbers, rows are pixels

        if has_x_axis:
            # first row is the x-axis
            x = data[0, :]
            # each column is a img_width x img_height pixels
            intensities = data[1:, :]
        else:
            intensities = data
            x = np.arange(data.shape[1])
            if invert_x_axis:
                x = np.flip(x)
    else:
        raise ValueError("Format not recognized")

    if not as_hsi:
        return Spectrum(
            x=x,
            data=intensities,
            x_axis_unit=x_axis_unit,
            data_unit=data_unit,
            name=name if name is not None else filepath_or_buffer,
        )

    img = intensities.reshape(img_height, img_width, -1)
    if scan_pattern == "bidirectional":
        # reshape from snake-like format to C-format
        img[1::2, :, :] = img[1::2, ::-1, :]

    # fix coordinate system (image vs matrix)
    img = np.rot90(img, k=1, axes=(0, 1))
    img = np.flipud(img)

    intensities = img.reshape(img_height * img_width, -1)

    return SpectralMap(
        x=x,
        data=intensities,
        img_width=img_width,
        img_height=img_height,
        x_axis_unit=x_axis_unit,
        data_unit=data_unit,
        name=name,
        metadata=metadata_as_extra(
            original_filename=Path(filepath_or_buffer).name if isinstance(filepath_or_buffer, (str, Path)) else None
        ),
    )


class SingleSpectrumCSVOutputParams(IOParams):
    as_dataframe: bool | Literal["pandas", "polars"] = False
    """If `True`, returns the data as a dataframe instead of writing to a file. If `'pandas'` or `'polars'`, returns a dataframe of the specified type."""

    metadata: dict[str, Any] | None = None
    """Dictionary of metadata to include in the output file. If `None`, no metadata is included."""

    transpose: bool = False
    """If `True`, transposes the output data so that the x-axis is in rows and the y-axis is in columns."""


@output_format(
    format_name="csv",
    friendly_name="CSV",
    extensions={"txt", "csv"},
    format_params_model=IOParams,
    supported_types=SpectrumType.SPECTRUM,
)
def write_csv_spectrum(
    sp: Spectrum,
    *,
    as_dataframe: bool | Literal["pandas", "polars"] = False,
    metadata: dict[str, str | float] | None = None,
    out_file: Path | str | None = None,
    transpose: bool = False,
) -> BytesIO | pl.DataFrame | Any | None:  # Any is for Pandas DataFrame, which is not imported here
    as_dataframe = as_dataframe is not False or out_file is None
    out: BytesIO | None = out_file if isinstance(out_file, BytesIO) else (BytesIO() if out_file is None else None)
    out_sink: Any = out if out is not None else out_file
    df = sp.to_polars(metadata=metadata, transpose=transpose)
    df.write_csv(out_sink)

    if as_dataframe:
        if as_dataframe == "pandas":
            _require_pandas_dependencies()
            return df.to_pandas()
        return df

    return out
