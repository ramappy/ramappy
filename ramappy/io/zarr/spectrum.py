from __future__ import annotations

import numpy as np

from ramappy.core import Spectrum

from .common import decode_dict, parse_xunit, read_attr_array, to_jsonable


def _serialize_unit(unit):
    if hasattr(unit, "to_tuple"):
        return unit.to_tuple()
    return None


def write_spectrum(root, key: str, spectrum: Spectrum, *, compressors) -> None:
    spc = root.require_group(key)
    if "y" in spc:
        del spc["y"]
    spectral_metadata = spectrum.spectral_metadata.model_dump() if hasattr(spectrum, "spectral_metadata") else {}
    spc.attrs.update(
        {
            "roi_x": spectrum.roi_x.tolist(),
            "x_axis_unit": _serialize_unit(spectrum.x_axis_unit),
            "data_unit": _serialize_unit(spectrum.data_unit),
            "name": spectrum.name,
            "color": spectrum.color,
            "metadata": to_jsonable(spectral_metadata),
        }
    )

    spc_x = spc.create_array(
        "x", shape=spectrum.x.shape, dtype=spectrum.x.dtype, compressors=compressors, overwrite=True
    )
    spc_x[:] = spectrum.x

    spc_data = spc.create_array(
        shape=spectrum.data.shape,
        dtype=spectrum.data.dtype,
        name="data",
        compressors=compressors,
        overwrite=True,
    )
    spc_data[:] = spectrum.data

    if "intensities" in spc:
        del spc["intensities"]


def read_spectrum(spc) -> Spectrum:
    return Spectrum(
        x=spc["x"][:],
        x_axis_unit=parse_xunit(spc.attrs.get("x_axis_unit", spc.attrs.get("x_unit"))),
        data_unit=decode_dict(spc.attrs.get("data_unit", spc.attrs.get("y_unit"))),
        roi_x=read_attr_array(spc, "roi_x"),
        data=np.array(spc["data" if "data" in spc else "intensities"]),
        ignore_sort=True,
        metadata=decode_dict(spc.attrs.get("metadata", "{}")),
        name=decode_dict(spc.attrs.get("name")),
        color=decode_dict(spc.attrs.get("color")),
    )
