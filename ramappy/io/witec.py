import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial

from ramappy.core.images2d import SpatialGrid
from ramappy.core.spectral_map import SpectralMap
from ramappy.core.spectrum import Spectrum
from ramappy.io.core import IOParams, IOParamsAsHsi, input_format, metadata_with_extra
from ramappy.units import Quantity, QuantityUnit, Unit

__all__ = ["WITecFormatParams", "read_witec"]


# map node type to dtype
NODE_DATA_TYPE = {
    2: np.float64,
    3: np.float32,
    4: np.int64,
    5: np.int32,
}


# map data for `node_type==7` (according to `DataType` field)
# bogus `None` in pos 0 since actual mapping starts from 1
DATA_TYPE_MAPPING = [
    None,
    np.int64,
    np.int32,
    np.int16,
    np.int8,
    np.uint32,
    np.uint16,
    np.uint8,
    bool,
    np.float32,
    np.float64,
]


UNIT_MAP = {
    "CCD cts": QuantityUnit(Quantity.INTENSITY, Unit.COUNTS),
    "1/cm": QuantityUnit(Quantity.WAVENUMBER, Unit.CM_1),
    "nm": QuantityUnit(Quantity.WAVELENGTH, Unit.NANOMETER),
    "um": QuantityUnit(Quantity.LENGTH, Unit.MICROMETER),
    "µm": QuantityUnit(Quantity.LENGTH, Unit.MICROMETER),
    "meV": QuantityUnit(Quantity.ENERGY, Unit.MILLI_ELECTRONVOLT),
    "None": QuantityUnit(Quantity.UNCALIBRATED, Unit.UNDEFINED),
}


def nm_to_cmminus1(x: float, laser_wavelength: float) -> float:
    return 1e7 * (1 / laser_wavelength - 1 / x)


def cmminus1_to_nm(x: float, laser_wavelength: float) -> float:
    return 1 / (1 / laser_wavelength - x / 1e7)


def _build_witec_metadata(wit, filepath_or_buffer) -> dict:
    getattr(wit, "spatial_grid", None)
    original_filename = str(filepath_or_buffer) if isinstance(filepath_or_buffer, (str, Path)) else None
    if isinstance(filepath_or_buffer, (str, Path)):
        original_filename = Path(filepath_or_buffer).name

    extra = {
        "witec_root_type": getattr(wit, "root_name", None),
        "witec_spatial_transform": getattr(wit, "spatial_transform", None),
    }

    instrument = None
    laser_wavelength = getattr(wit, "laser_wavelength", None)
    if laser_wavelength is not None:
        instrument = {"laser_wavelength_nm": float(laser_wavelength)}

    return metadata_with_extra(
        title=getattr(wit, "name", None),
        original_filename=original_filename,
        instrument=instrument,
        extra=extra,
        vendor=None,
    )


class WITec:
    """WITec *.wip/*.wid reader."""

    def __init__(self, filepath_or_buffer):
        """
        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            The name of the file to read (any valid string path is acceptable) or an object with a read()
            method, such as a file handle (e.g., via builtin open function) or StringIO.
        """
        self._fname = filepath_or_buffer

        self.data = None
        self.x = None
        self.name = None
        self.x_unit = "None"
        self.y_unit = "None"
        self.laser_wavelength = None
        self.spatial_grid = None
        self.spatial_unit = None
        self.spatial_transform = None
        self.data_classes = []
        self.root_name = None

        # self.tables = {}
        with open(filepath_or_buffer, "rb") as self._fid:
            warnings.warn("WITec *.wip format support is still experimental", stacklevel=2)

            header = self._read_bytes(8)
            if header not in {b"WIT_PR06", b"WIT_DA06"}:
                raise ValueError("Unsupported WITec file")

            # .wip (WIT_PR06) -> 'WITec Project'
            # .wid (WIT_DA06) -> 'WITec Data'

            self.raw_data = self._read_nodes(data={})

            self.parse_data()

    def _read_numpydata(self, ntype="<i", size=1, data_size=None):
        if data_size is not None:
            size = int(data_size / np.dtype(ntype).itemsize)
        return np.fromfile(self._fid, ntype, size)

    def _read_int(self, ntype="<i", size=1):  # np.uint32
        return self._read_numpydata(ntype=ntype, size=size)[0]

    def _read_float(self, ntype="<f4", size=1):  # np.float32
        return self._read_numpydata(ntype=ntype, size=size)[0]

    def _read_bytes(self, size):
        return self._fid.read(size)

    def _read_delimited_field(self):
        """Read a field (first byte is the field length)"""
        size = self._read_int()
        return self._read_bytes(size)

    def _read_string(self):
        return self._read_delimited_field().decode("iso-8859-1")

    def _read_node(self, pos: int | None = None, data=None):
        if pos is not None:
            self._fid.seek(pos)
        if data is None:
            data = {}
        node_name = self._read_string()
        node_type = self._read_int()
        node_start = self._read_int("<q")
        node_end = self._read_int("<q")

        data_length = node_end - node_start  # in bytes

        if node_type < 0 or node_type > 9 or node_type == 1:
            raise ValueError(f"Unknown type {node_type} for node")

        if node_type == 0:  # tree
            value = self._read_nodes(pos=node_start, end=node_end, data={})
            # FIXME: handle bogus container nodes (they are actually string). only (?) happens for SystemInformation
            # if isinstance(value, dict) and len(value) == 1:
            #     nkey = list(value.keys())[0]
            #     if isinstance(value[nkey], dict) and len(value[nkey]) == 0:
            #         print('EMPTY NODE', node_name)
            data[node_name] = value
        elif node_type == 7:  # data, need `DataType` for correct dtype, defer reading
            value = {"start_pos": self._fid.tell(), "data_length": data_length}
        elif node_type == 8:  # bool
            value = self._read_bytes(1) == b"\x01"
        elif node_type == 9:  # string
            value = self._read_string()
        else:
            if node_type == 6:  # either uint32 (data_length % 4 == 0) or uint16 (else)
                data_type = np.uint32 if data_length % 4 == 0 else np.uint16
            else:
                data_type = NODE_DATA_TYPE.get(node_type, np.float32)  # type: ignore[assignment]

            value = self._read_numpydata(ntype=data_type, data_size=data_length)
            if len(value) == 1:
                value = value[0]

        return {node_name: value}, node_end

    def _read_nodes(self, pos=None, end=None, data=None):
        if data is None:
            data = {}
        while pos is None or pos < end:
            value, node_end = self._read_node(pos)
            data.update(value)

            pos = node_end
            if end is None:
                break

        return data

    @staticmethod
    def _array_to_list(value):
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()

    def _parse_spatial_transform(self, data_block):
        transform = data_block.get("TDTransformation", {})
        viewport = data_block.get("TDSpaceTransformation", {}).get("ViewPort3D", {})

        self.spatial_unit = transform.get("StandardUnit")

        scale = viewport.get("Scale")
        scale_arr = np.asarray(scale, dtype=float).reshape(-1) if scale is not None else None
        model_origin = viewport.get("ModelOrigin")
        world_origin = viewport.get("WorldOrigin")

        self.spatial_transform = {
            "scale": self._array_to_list(scale_arr),
            "model_origin": self._array_to_list(model_origin),
            "world_origin": self._array_to_list(world_origin),
        }

        if scale_arr is None or scale_arr.size < 4:
            return

        if scale_arr.size >= 9:
            matrix = scale_arr[:9].reshape(3, 3)
            pixel_size_x = float(np.linalg.norm(matrix[:2, 0]))
            pixel_size_y = float(np.linalg.norm(matrix[:2, 1]))
        else:
            pixel_size_x = float(abs(scale_arr[0]))
            pixel_size_y = float(abs(scale_arr[1]))

        if pixel_size_x <= 0 or pixel_size_y <= 0:
            return

        spatial_quantity_unit = UNIT_MAP.get(self.spatial_unit)
        spatial_unit = spatial_quantity_unit.unit if spatial_quantity_unit is not None else Unit.PIXEL

        self.spatial_grid = SpatialGrid(
            pixel_size_y=pixel_size_y,
            pixel_size_x=pixel_size_x,
            spatial_unit=spatial_unit,
        )

    def build_metadata(self, filepath_or_buffer) -> dict:
        return _build_witec_metadata(self, filepath_or_buffer)

    def parse_data(self):
        root = next(iter(self.raw_data.keys()))  # either 'WITec Project' or 'WITec Data'
        self.root_name = root
        root_data = self.raw_data[root]["Data"]

        data_blocks = [(n, root_data[f"DataClassName {n}"]) for n in range(root_data["NumberOfData"])]
        self.data_classes = [block_type for _, block_type in data_blocks]
        # read *Interpretation first, then TDGraph, then the rest
        data_blocks.sort(key=lambda x: 0 if "Interpretation" in x[1] else 1 if x[1] == "TDGraph" else x[0])

        num_maps = 0

        # parse data
        for n, block_type in data_blocks:
            data_block = root_data[f"Data {n}"]

            if block_type == "TDGraph":
                if num_maps > 0:
                    raise NotImplementedError("The project containts multiple maps (currently not supported)")
                num_maps += 1
                # the actual y data
                self.name = data_block["TData"]["Caption"]
                self.size_x = data_block["TDGraph"]["SizeX"]
                self.size_y = data_block["TDGraph"]["SizeY"]
                self.spectral_size = data_block["TDGraph"]["SizeGraph"]

                inverted = data_block["TDGraph"].get("DataFieldInverted", False)

                data = data_block["TDGraph"]["GraphData"]["Data"]  # dict with info to read actual data
                dtype = data_block["TDGraph"]["GraphData"]["DataType"]
                if dtype <= 0 or dtype > 10:
                    raise ValueError(f"Unkown data type {dtype}")
                dtype = DATA_TYPE_MAPPING[dtype]

                self._fid.seek(data["start_pos"])
                self.data = self._read_numpydata(data_size=data["data_length"], ntype=dtype)

                self.data = self.data.reshape(self.size_x, self.size_y, self.spectral_size)
                self.x = np.arange(self.spectral_size)

                if inverted:
                    self.data = self.data.transpose([1, 0, 2])
            elif block_type == "TDZInterpretation":
                # data units
                self.y_unit = data_block["TDZInterpretation"]["UnitName"]
            elif block_type == "TDSpectralInterpretation":
                # spectral axis units
                self.laser_wavelength = data_block["TDSpectralInterpretation"]["ExcitationWaveLength"]
            elif block_type == "TDSpaceTransformation":
                self._parse_spatial_transform(data_block)
            elif block_type == "TDSpectralTransformation":
                self.x_unit = data_block["TDTransformation"]["StandardUnit"]
                is_calibrated = data_block["TDTransformation"]["IsCalibrated"]
                t = data_block["TDSpectralTransformation"]

                if not is_calibrated:
                    continue

                calib_type = t["SpectralTransformationType"]
                if calib_type == 0:
                    poly = Polynomial(t["Polynom"])  # quadratic polynomial
                    x = np.arange(self.spectral_size)
                    self.x = poly(x)
                elif calib_type == 1:
                    # [1] Gwyddion-website: http://gwyddion.net/module-list-nocss.en.php#wipfile
                    # [2] Theoretical background: http://www.horiba.com/us/en/scientific/products/optics-tutorial/wavelength-pixel-position/
                    nC = t["nC"]  # num of pixels at lambdaC
                    lambdaC = t["LambdaC"]  # centre lambda (wavelenth, nmn)
                    gamma = t["Gamma"]  # angle between incident and diffracted light
                    delta = t["Delta"]  # inclination angle of the CCD (WITec Project sign convention)
                    m = t["m"]  # grating diffraction order (1)
                    d = t["d"]  # grating groove density
                    px_width = t["x"]  # pixel width
                    f = t["f"]  # focal length

                    x = np.arange(self.spectral_size)

                    alpha = np.arcsin(lambdaC * m / d / (2 * np.cos(gamma / 2))) - gamma / 2  # angle of incidence
                    L_H = f * np.cos(delta)  # (ortho) distance from grating/focusing mirror to focal plane
                    HB_lambdaC = f * np.sin(
                        delta
                    )  # distance from the intercept of the normal to the focal plane to the wavelength lambdaC
                    HB_lambdaN = (
                        px_width * (nC - x) - HB_lambdaC
                    )  # distance from the intercept of the normal to the focal plane to the wavelength lambdaN
                    beta_lambdaC = gamma + alpha  # angle of diffraction at centre wavelength
                    beta_H = beta_lambdaC - delta  # angle from LH to the normal to the grating
                    beta_lambdaN = beta_H - np.arctan2(HB_lambdaN, L_H)  #  angle of diffraction at wavelength n

                    self.x = d / m * (np.sin(alpha) + np.sin(beta_lambdaN))
                elif calib_type == 2:
                    poly = Polynomial(t["FreePolynom"])

                    x_start = t["FreePolynomStartBin"]
                    x_end = t["FreePolynomStopBin"]

                    x = np.linspace(x_start, x_end, self.spectral_size, endpoint=False)
                    self.x = poly(x)
                else:
                    raise ValueError(f"Unkonw spectral axis calibration type {calib_type}")

        if self.x is not None and self.x_unit == "nm" and self.laser_wavelength is not None:
            self.x = nm_to_cmminus1(self.x, self.laser_wavelength)
            self.x_unit = "1/cm"

        if self.data is None:
            raise ValueError("No data found in WITec file")


class WITecFormatParams(IOParams, IOParamsAsHsi):
    """Parameters for reading WITec(TM) Project *.wid/*.wip files."""

    pass


@input_format(
    "witec",
    friendly_name="WITec",
    extensions={"wip", "wid"},
    has_custom_params=False,
    format_params_model=WITecFormatParams,
)
def read_witec(filepath_or_buffer, as_hsi: bool = True):
    """WITec(TM) Project *.wid/*.wip reader.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        The name of the file to read (any valid string path is acceptable) or an object with a read()
        method, such as a file handle (e.g., via builtin open function) or StringIO.
    """
    wit: Any = WITec(filepath_or_buffer)

    intensities = wit.data.reshape(np.prod(wit.data.shape[:2]), -1)
    x_axis_unit = UNIT_MAP.get(wit.x_unit)
    data_unit = UNIT_MAP.get(wit.y_unit)
    metadata = _build_witec_metadata(wit, filepath_or_buffer)
    spatial_grid = wit.spatial_grid

    if not as_hsi:
        return Spectrum(
            x=wit.x,
            data=intensities,
            x_axis_unit=x_axis_unit,
            data_unit=data_unit,
            name=wit.name,
            metadata=metadata,
        )

    return SpectralMap(
        x=wit.x,
        x_axis_unit=x_axis_unit,
        data_unit=data_unit,
        data=intensities,
        img_width=wit.data.shape[1],
        img_height=wit.data.shape[0],
        spatial_grid=spatial_grid,
        metadata=metadata,
        name=wit.name,
    )
