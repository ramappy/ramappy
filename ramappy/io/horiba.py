import warnings
from pathlib import Path

import numpy as np

from ramappy.core import SpectralMap, Spectrum
from ramappy.core.images2d import SpatialGrid
from ramappy.io.core import IOParams, IOParamsAsHsi, input_format, metadata_with_extra
from ramappy.units import Quantity, QuantityUnit, Unit

UNIT_MAP = {
    "Cnt": QuantityUnit(Quantity.INTENSITY, Unit.COUNTS),
    "Cnt/sec": QuantityUnit(Quantity.INTENSITY, Unit.COUNTS_PER_SECOND),
    "1/cm": QuantityUnit(Quantity.WAVENUMBER, Unit.CM_1),
    "um": QuantityUnit(Quantity.LENGTH, Unit.MICROMETER),
    "µm": QuantityUnit(Quantity.LENGTH, Unit.MICROMETER),
    "μm": QuantityUnit(Quantity.LENGTH, Unit.MICROMETER),
}


def _to_serializable(value):
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (np.floating, float)):
        return bool(np.isnan(value))
    return False


def _table_row_to_value(row: dict[str, object]):
    cleaned = {}
    for column, cell in row.items():
        if _is_missing(cell):
            continue
        cleaned[str(column)] = _to_serializable(cell)

    if not cleaned:
        return None

    if len(cleaned) == 1:
        return next(iter(cleaned.values()))

    lowered = {k.lower(): v for k, v in cleaned.items()}
    if set(lowered.keys()) == {"value", "unit"}:
        return {"value": lowered["value"], "unit": lowered["unit"]}

    return cleaned


def _parse_horiba_tables(tables: dict[str, dict[str, dict[str, object]]] | None):
    if not tables:
        return None

    parsed: dict[str, dict[str, object]] = {}
    for table_name, table in tables.items():
        if not isinstance(table, dict):
            parsed[str(table_name)] = _to_serializable(table)
            continue

        parsed[str(table_name)] = {
            str(index): _table_row_to_value(row if isinstance(row, dict) else {"value": row})
            for index, row in table.items()
        }

    return parsed


def _get_spatial_unit(axis_units: dict[str, str]) -> str | None:
    for axis_name in ("X", "Y"):
        axis_unit = axis_units.get(axis_name)
        if axis_unit is None:
            continue
        quantity_unit = UNIT_MAP.get(axis_unit)
        if quantity_unit is not None:
            return quantity_unit.unit
    return None


def _build_spatial_grid(ngc_data) -> SpatialGrid | None:
    x_axis = getattr(ngc_data, "X", None)
    y_axis = getattr(ngc_data, "Y", None)
    spatial_unit = _get_spatial_unit(getattr(ngc_data, "axes_units", {}))

    if spatial_unit is None or x_axis is None or y_axis is None:
        return None

    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    if x_axis.ndim != 1 or y_axis.ndim != 1 or x_axis.size == 0 or y_axis.size == 0:
        return None

    pixel_size_x = float(np.median(np.abs(np.diff(x_axis)))) if x_axis.size > 1 else 1.0
    pixel_size_y = float(np.median(np.abs(np.diff(y_axis)))) if y_axis.size > 1 else 1.0

    if pixel_size_x == 0.0:
        pixel_size_x = 1.0
    if pixel_size_y == 0.0:
        pixel_size_y = 1.0

    return SpatialGrid(
        pixel_size_y=pixel_size_y,
        pixel_size_x=pixel_size_x,
        origin_y=float(np.min(y_axis)),
        origin_x=float(np.min(x_axis)),
        spatial_unit=spatial_unit,
    )


def _build_horiba_metadata(ngc_data, filepath_or_buffer, *, read_tables: bool) -> dict:
    parsed_tables = _parse_horiba_tables(getattr(ngc_data, "tables", None))

    original_filename = None
    if isinstance(filepath_or_buffer, (str, Path)):
        original_filename = Path(filepath_or_buffer).name

    extra = {}
    if parsed_tables:
        extra.update(parsed_tables)

    return metadata_with_extra(
        description=getattr(ngc_data, "comment", None),
        original_filename=original_filename,
        extra=extra,
    )


class Horiba5:
    """Horiba(TM) LabSpec 5 NGC/NGS reader."""

    def __init__(self, filepath_or_buffer, *, read_tables: bool = True):
        """
        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            The name of the file to read (any valid string path is acceptable) or an object with a read()
            method, such as a file handle (e.g., via builtin open function) or StringIO.
        """
        self._fname = filepath_or_buffer

        self.read_tables = read_tables

        self.X: np.ndarray | None = None
        self.Y: np.ndarray | None = None
        self.Intens: np.ndarray | None = None
        self.axes_names: list[str] = []

        self.tables: dict[str, object] = {}
        with open(filepath_or_buffer, "rb") as self._fid:
            warnings.warn("LabSpec 5 format support is still experimental", stacklevel=2)
            self.read()

    def _read_numpydata(self, ntype="<i", size=1):
        return np.fromfile(self._fid, ntype, size)

    def _read_int(self, ntype="<i", size=1):  # np.uint32
        return self._read_numpydata(ntype=ntype, size=size)[0]

    def _read_float(self, ntype="<f4", size=1):  # np.float32
        return self._read_numpydata(ntype=ntype, size=size)[0]

    def _read_bytes(self, size):
        return self._fid.read(size)

    def _read_delimited_field(self):
        """Read a field (first byte is the field length)"""
        # size = self._read_int(ntype='<b')  # np.int8
        size = ord(self._fid.read(1))
        return self._read_bytes(size)

    def _read_string(self):
        return self._read_delimited_field().decode("iso-8859-1")

    def _read_block_string(self):
        """Assuming block of strings"""
        block_size = self._read_int("<h")
        return [self._read_string() for _ in range(block_size)]

    def _read_block_array(self, ntype="<i"):
        """Assuming block of strings"""
        array_size = self._read_int("<h")
        return self._read_numpydata(size=array_size, ntype=ntype)

    def _read_table(self):
        def read_string_none(duplicated_null=True):
            """An empty value is represented as b'\x00\x00'."""
            if self._read_bytes(1) == b"\x00":  # empty row
                if duplicated_null:
                    self._read_bytes(1)  # should be b'\x00'
                return None
            self._fid.seek(-1, 1)
            return self._read_string()

        def read_string_block(duplicated_null=True):
            num_strings_to_read = self._read_int("<h")
            return num_strings_to_read, [read_string_none(duplicated_null) for _ in range(num_strings_to_read)]

        table_name = (self._read_string(), self._read_string())  # table name (repeated twice?)

        n = self._read_int("<i")  # next table starts after n bytes from here

        if table_name[0] == "Hist":
            # ignore history
            self._fid.seek(n, 1)
            return None, None

        _num_rows, rows_name = read_string_block()
        num_columns, column_names = read_string_block()

        num_columns = self._read_int("<h")
        col_type = [self._read_int("<i") for _ in range(num_columns)]

        table = {str(row_name): {} for row_name in rows_name}

        for i in range(num_columns):
            _, data = read_string_block(duplicated_null=col_type[i] > 0)
            column_name = str(column_names[i])
            for row_name, cell in zip(rows_name, data, strict=False):
                if row_name is None:
                    continue
                row = table.setdefault(str(row_name), {})
                row[column_name] = cell

        return table_name[0], table

    def _read_axes_units(self):
        # axes name, axes units
        return self._read_block_string(), self._read_block_string()

    @property
    def img_shape(self):
        if self.ndim == 2:
            return (1, self.Intens.shape[0])
        return self.Intens.shape[:2]

    def read(self):
        header = self._fid.readline()
        if header != b"NGSNextGen\x01\x00\x00\x00\x01\x00\x00\x00\n":
            raise ValueError("Unexpected header!")

        if self._read_bytes(10) != b"DataMatrix":
            warnings.warn("Unexpected header, parsing of file may not be valid", stacklevel=2)

        instrument_type = self._read_delimited_field()
        if instrument_type not in {b"SpIm", b"Probe"}:  # , b'Map'):
            raise ValueError(f"Instrument type {instrument_type} is not supported")

        self.comment = self._read_string()  # the name

        self._read_int()
        self._read_numpydata(size=3)  # three numbers: [0, 8, -1]
        data_size = self._read_int()
        self.ndim = self._read_int("<h")  # int16
        shape = self._read_numpydata(size=self.ndim)

        skip = self._fid.read(2)
        if skip == b"\xff\xff":
            _ = self._read_numpydata(size=4, ntype=np.uint8)  # [0, 0, 1, 2]

        self.Intens = self._read_numpydata(size=data_size, ntype="<f4")
        self.Intens.shape = shape

        # read axes name & units (e.g., ('Spectr', '1/cm'), ('Intens', 'Cnt'), ('Y', 'µm'), ('X', 'µm'))
        self.axes_names, axes_units = self._read_axes_units()
        self.axes_units = dict(zip(self.axes_names, axes_units, strict=False))

        _ = self._read_block_string()  # axes... names?

        # ordered indices (wrt axes_names)
        field_indices = self._read_block_array(ntype="<i")
        # endpoints of axes
        self._read_block_array(ntype="<f4")

        # unknown values (probably relative to the axis stored content), if nonzero seems to mean we have data
        range_types = self._read_block_array(ntype="<i")

        for i, range_type in enumerate(range_types):
            if field_indices[i] < self.ndim and range_type != 0:
                axissize = self._read_int("<h") // 4
                arr = self._read_numpydata(size=axissize, ntype="<f4")
                setattr(self, self.axes_names[i], arr)

        if self.read_tables:
            try:
                # read tables
                num_tables = self._read_int("<i")
                for _ in range(num_tables):
                    if self._read_string() == "Table":
                        table_name, table = self._read_table()
                        self.tables[table_name] = table
                        # skip padding
                        while self._read_bytes(1) == b"\x00":
                            pass
                        self._fid.seek(-1, 1)
            except Exception as e:
                warnings.warn(f"Error reading tables: {e}", stacklevel=2)
                self.tables = {}


class Horiba5Params(IOParams, IOParamsAsHsi):
    """Parameters for reading Horiba LabSpec 5 NGC/NGS files."""

    read_tables: bool = True
    """Whether to read table data from the file (if any). Setting this to False may speed up the reading of files with large tables that are not needed."""


@input_format(
    "horiba5",
    friendly_name="Horiba LabSpec 5",
    extensions={"ngc", "ngs"},
    has_custom_params=False,
    format_params_model=Horiba5Params,
)
def read_horiba5(filepath_or_buffer, *, as_hsi: bool = True, read_tables: bool = True):
    """Horiba(TM) LabSpec 5 NGC/NGS reader.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        The name of the file to read (any valid string path is acceptable) or an object with a read()
        method, such as a file handle (e.g., via builtin open function) or StringIO.
    as_hsi : boolean, default = True
        Whether to return an SpectralMap object or a Spectrum
    read_tables : boolean, default = True
        Whether to read table data from the file (if any). Setting this to False may speed up the reading of files with large tables that are not needed.
    """

    ngc_data = Horiba5(filepath_or_buffer, read_tables=read_tables)

    x_axis_unit = UNIT_MAP.get(ngc_data.axes_units["Spectr"])
    data_unit = UNIT_MAP.get(ngc_data.axes_units["Intens"])
    spatial_grid = _build_spatial_grid(ngc_data)
    metadata = _build_horiba_metadata(ngc_data, filepath_or_buffer, read_tables=read_tables)

    # The spectral axis name is the first element of axes_names (e.g., "Spectr").
    # It is set dynamically via setattr(self, axes_names[i], arr) during read().
    spectral_axis_name = ngc_data.axes_names[0] if ngc_data.axes_names else None
    x_axis = getattr(ngc_data, spectral_axis_name, None) if spectral_axis_name else None
    intens = ngc_data.Intens
    if x_axis is None or intens is None:
        raise ValueError("Could not read spectral axis or intensity data from Horiba file")

    if as_hsi:
        return SpectralMap(
            x=x_axis,
            x_axis_unit=x_axis_unit,
            data_unit=data_unit,
            data=intens.reshape(np.prod(ngc_data.img_shape), -1),
            img_width=ngc_data.img_shape[1],
            img_height=ngc_data.img_shape[0],
            spatial_grid=spatial_grid,
            name=ngc_data.comment,
            metadata=metadata,
        )

    return Spectrum(
        x=x_axis,
        x_axis_unit=x_axis_unit,
        data_unit=data_unit,
        data=intens,
        name=ngc_data.comment,
        metadata=metadata,
    )
