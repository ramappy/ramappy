from pathlib import Path
from typing import Any, Literal

import numpy as np
import PIL.Image
from renishawWiRE import WDFReader
from renishawWiRE.types import UnitType

from ramappy.core import SpectralMap, Spectrum
from ramappy.core.images2d import Image2D, SpatialGrid
from ramappy.io.core import IOParams, IOParamsAsHsi, input_format, metadata_with_extra
from ramappy.units import Quantity, QuantityUnit, Unit

UNIT_MAP = {
    UnitType.Arbitrary: Unit.ARBITRARY_UNITS,
    UnitType.RamanShift: Unit.CM_1,
    UnitType.Wavenumber: Unit.NANOMETER,  # from code: `Wavenumber = 2  # nm` (?)
    UnitType.Nanometre: Unit.NANOMETER,
    UnitType.ElectronVolt: Unit.ELECTRONVOLT,
    UnitType.Micron: Unit.MICROMETER,
    UnitType.Counts: Unit.COUNTS,
    UnitType.Pixels: Unit.PIXEL,
    # UnitType.Intensity: const.DataUnit.INTENSITY_AU,
    # UnitType.RelativeIntensity: const.DataUnit.RELATIVE_INTENSITY_AU,
}


def assign_unit(unit: UnitType, var: Literal["x", "data", "spatial"]):
    if unit == UnitType.ElectronVolt:
        return QuantityUnit(Quantity.ENERGY, Unit.ELECTRONVOLT)

    if var == "data":
        if unit == UnitType.Intensity:
            return QuantityUnit(Quantity.INTENSITY, Unit.ARBITRARY_UNITS)
        if unit == UnitType.RelativeIntensity:
            return QuantityUnit(Quantity.RELATIVE_INTENSITY, Unit.ARBITRARY_UNITS)
        if unit == UnitType.Counts:
            return QuantityUnit(Quantity.INTENSITY, Unit.COUNTS)

    if var == "x":
        if unit == UnitType.RamanShift:
            return QuantityUnit(Quantity.RAMANSHIFT, Unit.CM_1)
        if unit == UnitType.Nanometre:
            return QuantityUnit(Quantity.WAVELENGTH, Unit.NANOMETER)
        if unit == UnitType.Pixels:
            return QuantityUnit(Quantity.UNCALIBRATED, Unit.PIXEL)

    if var == "spatial":
        if unit == UnitType.Micron:
            return QuantityUnit(Quantity.LENGTH, Unit.MICROMETER)
        if unit == UnitType.Pixels:
            return QuantityUnit(Quantity.LENGTH, Unit.PIXEL)

    return QuantityUnit(Quantity.UNDEFINED, UNIT_MAP.get(unit, Unit.UNDEFINED))


def _build_renishaw_metadata(reader, filepath_or_buffer):
    """Extract Renishaw-specific metadata from WDFReader."""
    original_filename = None
    if isinstance(filepath_or_buffer, (str, Path)):
        original_filename = Path(filepath_or_buffer).name

    extra = {
        "WDF Info": {
            "Application": f"Renishaw {reader.application_name} ({'.'.join([str(s) for s in reader.application_version])})",
            "Username": reader.username,
            "Measurement Type": reader.measurement_type,
            "Scan Type": reader.scan_type,
            "Laser Wavenumber": f"{reader.laser_length} nm",
            "Accumulation Count": reader.accumulation_count,
            "Capacity": reader.capacity,
        }
    }

    return metadata_with_extra(
        title=reader.title,
        original_filename=original_filename,
        extra=extra,
    )


def _calc_crop_box(reader, ph, pw):
    """Get image crop box"""

    w_, h_ = reader.img_dimensions
    x0_, y0_ = reader.img_origins

    x_unique = np.unique(reader.xpos)
    x_pad = float(np.abs(np.diff(x_unique)).mean()) if len(x_unique) > 1 else float(reader.map_info.get("x_pad", 1))
    y_unique = np.unique(reader.ypos)
    y_pad = float(np.abs(np.diff(y_unique)).mean()) if len(y_unique) > 1 else float(reader.map_info.get("y_pad", 1))

    map_xl = reader.xpos.min() - x_pad / 2
    map_xr = reader.xpos.max() + x_pad / 2
    map_yt = reader.ypos.min() - y_pad / 2
    map_yb = reader.ypos.max() + y_pad / 2

    left = int(pw * (map_xl - x0_) / w_)
    right = int(pw * (map_xr - x0_) / w_)
    top = int(ph * (map_yt - y0_) / h_)
    bottom = int(ph * (map_yb - y0_) / h_)

    return (left, top, right, bottom)


class RenishawWDFInputParams(IOParams, IOParamsAsHsi):
    """Parameters for reading Renishaw WDF files."""

    pass


@input_format(
    "renishaw_wdf",
    friendly_name="Renishaw WDF",
    extensions={
        "wdf",
    },
    has_custom_params=False,
    format_params_model=RenishawWDFInputParams,
)
def read_renishaw_wdf(filepath_or_buffer, *, as_hsi: bool = True):
    """Read Renishaw(TM) WDF grid data.

    Parameters
    ----------
    filepath_or_buffer : str, path object or file-like object
        The name of the file to read (any valid string path is acceptable) or an object with a read()
        method, such as a file handle (e.g., via builtin open function) or StringIO.
    """
    reader: Any = WDFReader(filepath_or_buffer)

    if as_hsi and reader.measurement_type == 3:
        # WDFReader does not currently handle different scan patterns (just assumes everything is `raster`)!
        xpos = ((reader.xpos - reader.xpos[0]) / reader.map_info["x_pad"]).round().astype(int)
        ypos = ((reader.ypos - reader.ypos[0]) / reader.map_info["y_pad"]).round().astype(int)
        img_width, img_height = reader.map_shape

        if reader.is_completed:
            reader.spectra = reader.spectra[ypos, xpos, :]
            mask_idxs = None
        else:
            spectra = np.zeros((*reader.map_shape, len(reader.xdata)), dtype=reader.spectra.dtype)
            spectra[xpos, ypos, :] = reader.spectra
            reader.spectra = spectra.reshape(-1, len(reader.xdata))

            mask_idxs = np.ravel_multi_index((xpos, ypos), reader.map_shape)

            # why do I need this??
            reader.map_shape = (reader.map_shape[1], reader.map_shape[0])
            img_width, img_height = img_height, img_width

        images = None
        if reader.img is not None:
            img = PIL.Image.open(reader.img)

            images = {
                "white_light": Image2D(
                    image=img.crop(box=_calc_crop_box(reader, img.height, img.width)),
                    locked=True,
                    name="White Light",
                    visible=False,
                )
            }

        spatial_unit = assign_unit(reader.map_info["x_unit"], var="spatial").unit

        x_unique = np.unique(reader.xpos)
        x_pad = float(np.abs(np.diff(x_unique)).mean()) if len(x_unique) > 1 else float(reader.map_info.get("x_pad", 1))

        y_unique = np.unique(reader.ypos)
        y_pad = float(np.abs(np.diff(y_unique)).mean()) if len(y_unique) > 1 else float(reader.map_info.get("y_pad", 1))

        # WDFReader maps Y as first dim (rows) and X as second (cols) or vice-versa?
        # Above we have reader.spectra = spectra.reshape(-1, len(reader.xdata))
        # and reader.map_shape = (reader.map_shape[1], reader.map_shape[0])
        # By convention, pixel_size should match (row_size, col_size) -> (y_pad, x_pad)
        pixel_size = (y_pad, x_pad)

        spatial_grid = SpatialGrid(
            pixel_size_y=float(pixel_size[0]),
            pixel_size_x=float(pixel_size[1]),
            spatial_unit=spatial_unit,
        )

        spectral_map = SpectralMap(
            x=reader.xdata,
            data=reader.spectra,
            img_width=img_width,
            img_height=img_height,
            images=images,
            x_axis_unit=assign_unit(reader.xlist_unit, var="x"),
            data_unit=assign_unit(reader.spectral_unit, var="data"),
            spatial_grid=spatial_grid,
            name=reader.title,
            metadata=_build_renishaw_metadata(reader, filepath_or_buffer),
        )

        if mask_idxs is not None:
            spectral_map.masks["measured"] = spectral_map.new_mask(
                idxs=mask_idxs,
                name="Measurement",
                editable=False,
                visible=False,
            )

        return spectral_map
    # elif reader.measurement_type == 1 or reader.measurement_type == 2:
    return Spectrum(
        x=reader.xdata,
        data=reader.spectra,
        name=reader.title,
        x_axis_unit=assign_unit(reader.xlist_unit, var="x"),
        data_unit=assign_unit(reader.spectral_unit, var="data"),
        metadata=_build_renishaw_metadata(reader, filepath_or_buffer),
    )
