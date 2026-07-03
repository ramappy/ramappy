import uuid
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal

import h5py  # or hdf5storage / mat73, for MATLAB v7.3
import numpy as np
import scipy.io
from annotated_types import MaxLen, MinLen
from scipy.io.matlab._miobase import _get_matfile_version

from ramappy import units
from ramappy.core import SpectralMap, Spectrum
from ramappy.core.images2d import Image2D
from ramappy.io.core import IOParams, IOParamsAsHsi, IOParamsUnits, input_format, metadata_as_extra, output_format

MATLAB_v73 = 2


def get_matfile_version(filepath_or_buffer):
    with open(filepath_or_buffer, "rb") as f:
        try:
            mat_version, _ = _get_matfile_version(f)
            is_matlab = True
        except ValueError:
            mat_version = MATLAB_v73
            is_matlab = False
        return mat_version, is_matlab


def h5_get_var_info(mat_file):
    variables = []

    def visit_item(k, v):
        if isinstance(v, h5py.Dataset):
            variables.append((k, v.shape, v.dtype))

    with h5py.File(mat_file) as h5file:
        h5file.visititems(visit_item)
    return variables


def get_variables_names(filepath_or_buffer):
    mat_version, _ = get_matfile_version(filepath_or_buffer)
    # list of [(name, size, type)]
    variables = (
        scipy.io.whosmat(filepath_or_buffer) if mat_version < MATLAB_v73 else h5_get_var_info(filepath_or_buffer)
    )
    return {
        v[0]: {"size": v[1], "ndim": np.sum(np.asarray(v[1]) != 1)}
        for v in variables
        if v[0] not in {"__globals__", "__header__", "__version__"}
    }


def sniff_witec(filepath: str) -> bool:
    if not str(filepath).lower().endswith(".mat"):
        return False
    try:
        mat_version, _ = get_matfile_version(filepath)
        if mat_version < MATLAB_v73:
            mat = scipy.io.loadmat(filepath, simplify_cells=True)
            keys = [k for k in mat if k not in {"__globals__", "__header__", "__version__"}]
            if len(keys) != 1:
                return False
            inner = mat[keys[0]]
            return isinstance(inner, dict) and "imagesize" in inner and "data" in inner and "axisscale" in inner
        else:
            with h5py.File(filepath, "r") as f:
                keys = list(f.keys())
                if len(keys) != 1:
                    return False
                inner = f[keys[0]]
                return "imagesize" in inner and "data" in inner and "axisscale" in inner
    except Exception:
        return False


def sniff_nexus(filepath: str) -> bool:
    if not str(filepath).lower().endswith((".h5", ".hdf5")):
        return False
    try:
        with h5py.File(filepath, "r") as f:
            required_paths = (
                "ENTRY/data/intensity",
                "ENTRY/data/raman_shift",
                "PROJECT/project_id",
                "PROJECT/author_id",
            )
            return all(path in f for path in required_paths)
    except Exception:
        return False


class MATLABParams(IOParams, IOParamsAsHsi, IOParamsUnits):
    """Parameters for reading MATLAB/Octave mat files."""

    x_label: str | None = None
    """Name of the variable containing the x-axis (wavenumbers/wavelengths). If not provided, a dummy x-axis will be created."""

    data_label: str = "data"
    """Name of the variable containing the spectral data (3-D cube)."""

    invert_x_axis: bool = False
    """If `True`, invert the x-axis (useful for some legacy data)."""

    axes_order: Annotated[
        tuple[Literal["X", "Y", "S"], ...],
        MinLen(1),
        MaxLen(3),
    ] = ("X", "Y", "S")
    """Order of the axes in the data cube. 'X' and 'Y' are spatial dimensions, 'S' is the spectral dimension."""

    plain_h5_file: bool = False
    """If `True`, treat the file as a plain HDF5 file (not a MATLAB mat file)."""

    name: str | None = None
    """Optional name for the imported spectral map. If not provided, a name will be inferred from the file or metadata."""


class HDF5Params(MATLABParams):
    """Parameters for reading HDF5 files."""

    plain_h5_file: bool = True


class MATLABWITecParams(IOParams, IOParamsAsHsi):
    """Parameters for reading WITec's MATLAB mat files."""

    name: str | None = None
    """Optional name for the imported spectral map. If not provided, a name will be inferred from the file or metadata."""

    auto_import_images: bool = False
    """If `True`, automatically import 2-D images from the file (if any)."""


def _extract_images2d(mat_dict_or_h5, width, height, exclude_keys):
    images2d = {}

    def check_and_add(item_name, data):
        if not isinstance(data, (np.ndarray, h5py.Dataset)):
            return
        if item_name in exclude_keys:
            return

        shape = data.shape
        w_h = (width, height)
        valid = False
        fmt_data = data

        if len(shape) >= 2:
            if shape[:2] == w_h or shape[:2] == (height, width):
                if len(shape) == 2 or (len(shape) == 3 and shape[2] in {1, 3}):
                    valid = True
            elif shape[-2:] == w_h or shape[-2:] == (height, width):
                if len(shape) == 2:
                    valid = True
                elif len(shape) == 3 and shape[0] in {1, 3}:
                    valid = True
                    if isinstance(data, h5py.Dataset):
                        fmt_data = data[:]
                    fmt_data = np.transpose(fmt_data, (1, 2, 0))

        if valid:
            if isinstance(fmt_data, h5py.Dataset):
                fmt_data = fmt_data[:]
            images2d[uuid.uuid4().hex] = Image2D(data=np.asarray(fmt_data), name=item_name)

    if isinstance(mat_dict_or_h5, dict):
        for k, v in mat_dict_or_h5.items():
            if not k.startswith("__"):
                check_and_add(k, v)
    else:

        def visit_add(k, v):
            if isinstance(v, h5py.Dataset):
                item_name = k.split("/")[-1]
                check_and_add(item_name, v)

        mat_dict_or_h5.visititems(visit_add)

    return images2d if images2d else None


@input_format(
    "matlab",
    friendly_name="MATLAB/Octave",
    extensions={
        "mat",
    },
    format_params_model=MATLABParams,
    param_hints_from_file=get_variables_names,
)
@input_format(
    "hdf5",
    friendly_name="HDF5",
    extensions={"h5", "hdf5"},
    format_params_model=HDF5Params,
    param_hints_from_file=get_variables_names,
)
def read_matlab(
    filepath_or_buffer,
    *,
    x_label: str | None = None,
    invert_x_axis: bool = False,
    data_label: str = "data",
    as_hsi: bool = True,
    axes_order: Annotated[
        tuple[Literal["X", "Y", "S"], ...],
        MinLen(1),
        MaxLen(3),
    ] = ("X", "Y", "S"),
    plain_h5_file: bool = False,
    name: str | None = None,
    auto_import_images: bool = False,
    x_axis_unit: units.type_spectral_quantityunit = None,
    data_unit: units.type_data_quantityunit = None,
):
    """Read MATLAB/Octave mat files"""
    axes_order_ref = ("X", "Y", "S")

    if plain_h5_file:
        mat_version, is_matlab = None, False
    else:
        mat_version, is_matlab = get_matfile_version(filepath_or_buffer)

    if is_matlab and mat_version is not None and mat_version < MATLAB_v73:
        # simplify_cells=True
        # will auto skip any missing variable name
        mat = scipy.io.loadmat(filepath_or_buffer, variable_names=[x_label, data_label])
        h5_file = None
    else:
        h5_file = h5py.File(filepath_or_buffer)
        mat = h5_file

    try:
        if data_label not in mat:
            raise KeyError(
                f"Mat file has no `{data_label}` label (can't retrieve y data). Available labels: {list(mat.keys())}"
            )
        # Force array (needed if mat is HDF5).
        # IMPORTANT: do not blindly squeeze() before checking ndim when reading as an SpectralMap,
        # otherwise singleton spatial dimensions (e.g., 1x1xN) collapse to 1-D/2-D and get rejected.
        raw_y = np.asarray(mat[data_label])
        if as_hsi:
            # Prefer preserving true 3-D cubes even when some dims are 1.
            if raw_y.ndim == 3:
                spectral_map = raw_y
            else:
                spectral_map = np.squeeze(raw_y)
                # Special-case fully squeezed spectra vectors / 1xN / Nx1: interpret as 1x1xN.
                if spectral_map.ndim == 1:
                    spectral_map = spectral_map[None, None, :]
                elif spectral_map.ndim == 2 and 1 in spectral_map.shape:
                    spectral_map = spectral_map.reshape(-1)[None, None, :]
        else:
            spectral_map = np.atleast_2d(np.squeeze(raw_y))

        if as_hsi and spectral_map.ndim != 3:
            raise ValueError(f"Expecting a 3-D cube for `{data_label}`")
        if spectral_map.ndim > 3:
            raise ValueError(f"Expecting a variable with at most 3 dimensions for `{data_label}`")

        if spectral_map.ndim == 2:
            # axes_order should be either ('S'), ('X', 'S') or ('S', 'X') (w/ X being the index of the spectra)
            if axes_order == ["S"]:
                axes_order_ref = list(axes_order)  # type: ignore[assignment]
            elif len(axes_order) != 2:
                raise ValueError("Axes order mismatch: expecting either `('X', 'S')` or `('S', 'X')`")
            else:
                axes_order_ref = ["X", "S"]  # type: ignore[assignment]

        axes_order_arr: Any = np.asarray(axes_order)
        if not np.array_equal(axes_order_ref, axes_order_arr):
            dim_permutation = np.array([np.argwhere(axes_order_arr == c) for c in axes_order_ref]).flatten()
            spectral_map = spectral_map.transpose(dim_permutation)

        if x_label is None or len(x_label) == 0:
            warnings.warn("No x-axis (possibly uncalibrated data), creating dummy", stacklevel=2)
            x_data = np.arange(spectral_map.shape[-1])
            if invert_x_axis:
                x_data = np.flip(x_data)
        else:
            if x_label not in mat:
                raise KeyError(
                    f"Mat file has no `{x_label}` label (can't retrieve the x axis). Available labels: {list(mat.keys())}"
                )
            x_data = np.squeeze(mat[x_label])

        if x_data.ndim != 1:
            raise ValueError(f"`{x_label}` should be 1-D")

        if spectral_map.shape[-1] != len(x_data):
            raise ValueError(f"Mismatching spectral dimensions between `{x_label}` and `{data_label}`")

        if not as_hsi:
            if spectral_map.ndim == 3:
                spectral_map = spectral_map.reshape(-1, spectral_map.shape[2])
            return Spectrum(
                x=x_data,
                data=spectral_map,
                name=name
                if name is not None
                else (
                    Path(filepath_or_buffer).stem
                    if isinstance(filepath_or_buffer, (str, Path))
                    else str(filepath_or_buffer)
                ),
                x_axis_unit=x_axis_unit,
                data_unit=data_unit,
            )

        intensities = spectral_map.reshape(-1, spectral_map.shape[2])

        images2d = None
        if auto_import_images:
            images2d = _extract_images2d(mat, spectral_map.shape[1], spectral_map.shape[0], (x_label, data_label))

        return SpectralMap(
            x=x_data,
            data=intensities,
            img_width=spectral_map.shape[1],
            img_height=spectral_map.shape[0],
            x_axis_unit=x_axis_unit,
            data_unit=data_unit,
            name=name,
            metadata=metadata_as_extra(
                original_filename=Path(filepath_or_buffer).name if isinstance(filepath_or_buffer, (str, Path)) else None
            ),
            images=images2d,
        )
    finally:
        if h5_file is not None:
            h5_file.close()


@input_format(
    "matlab_witec",
    friendly_name="MATLAB/Octave (WITec export)",
    extensions={
        "mat",
    },
    sniffer=sniff_witec,
    has_custom_params=False,
    format_params_model=MATLABWITecParams,
)
def read_matlab_witec(
    filepath_or_buffer,
    as_hsi: bool = True,
    name: str | None = None,
    auto_import_images: bool = False,
):
    """Read Witec's MATLAB mat files"""
    mat = scipy.io.loadmat(filepath_or_buffer, simplify_cells=True)
    keys = [k for k in mat if k not in {"__globals__", "__header__", "__version__"}]
    if len(keys) != 1:
        raise ValueError("Wrong format")

    mat = mat[keys[0]]

    img_shape = mat["imagesize"]
    intensities = mat["data"]
    x_data = mat["axisscale"][1][0]
    x_axis_unit = mat["axisscale"][1][1]

    if x_axis_unit == "nm":
        x_axis_unit = units.QuantityUnit("wavelength", "nm")
    else:
        x_axis_unit = units.QuantityUnit("wavenumber", x_axis_unit)

    if not as_hsi:
        return Spectrum(
            x=x_data,
            data=intensities,
            name=keys[0],
            x_axis_unit=x_axis_unit,
        )

    images2d = None
    if auto_import_images:
        images2d = _extract_images2d(mat, img_shape[1], img_shape[0], ("data", "axisscale", "imagesize"))

    return SpectralMap(
        x=x_data,
        data=intensities,
        img_width=img_shape[1],
        img_height=img_shape[0],
        x_axis_unit=x_axis_unit,
        name=keys[0],
        images=images2d,
    )


class MATLABOutputParams(IOParams):
    """Parameters for writing MATLAB/Octave mat files."""

    include_masks: bool = True
    """If `True`, include 2-D masks in the output file."""

    include_images_data: bool = True
    """If `True`, include 2-D images in the output file."""


@output_format(
    "matlab",
    friendly_name="MATLAB/Octave",
    extensions={
        "mat",
    },
    format_params_model=MATLABOutputParams,
    has_custom_params=False,
)
def write_matlab(
    spectral_map: SpectralMap | Spectrum,
    out_file,
    *,
    include_masks: bool = True,
    include_images_data: bool = True,
):
    spectral_map_any: Any = spectral_map
    data_dict = {
        "data": spectral_map_any.cube if hasattr(spectral_map, "images") else spectral_map.data,
        "x": spectral_map.x,
    }
    if hasattr(spectral_map, "images"):
        if include_images_data:
            data_dict["images"] = {k: img.data for k, img in spectral_map_any.images.items() if hasattr(img, "data")}
        if include_masks:
            data_dict["masks"] = {k: mask.get_2Dmask() for k, mask in spectral_map_any.masks.items()}
    scipy.io.savemat(out_file, mdict=data_dict)


class HDF5OutputParams(MATLABOutputParams):
    pass


@output_format(
    "hdf5",
    friendly_name="HDF5",
    extensions={
        "h5",
        "hdf5",
    },
    format_params_model=HDF5OutputParams,
    has_custom_params=False,
)
def write_hdf5(
    spectral_map: SpectralMap | Spectrum,
    out_file,
    *,
    include_masks: bool = True,
    include_images_data: bool = True,
):
    spectral_map_any: Any = spectral_map
    data = spectral_map_any.cube if hasattr(spectral_map, "images") else spectral_map.data
    with h5py.File(out_file, "w") as h5file:
        h5file.create_dataset("data", data=np.asarray(data))
        h5file.create_dataset("x", data=np.asarray(spectral_map.x))
        if hasattr(spectral_map, "images"):
            if include_images_data:
                images_group = h5file.create_group("images")
                for key, img in spectral_map_any.images.items():
                    if hasattr(img, "data") and img.data is not None:
                        images_group.create_dataset(str(key), data=np.asarray(img.data))
            if include_masks:
                masks_group = h5file.create_group("masks")
                for key, mask in spectral_map_any.masks.items():
                    masks_group.create_dataset(str(key), data=np.asarray(mask.get_2Dmask()))
