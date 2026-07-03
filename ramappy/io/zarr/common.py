from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import zarr
import zarr.codecs

from ramappy import units

# Ignore ZipStore duplicate metadata-entry warnings from zipfile when groups are
# updated incrementally (safe for our write pattern).
warnings.filterwarnings("ignore", message=r"Duplicate name: '.*zarr\.json'")


# Target uncompressed bytes per chunk for the spectral cube.
_CUBE_CHUNK_TARGET_BYTES = 4 * 1024 * 1024  # 4 MB


def get_default_compressors():
    """Return the default compressor configuration."""
    return zarr.codecs.BloscCodec(cname="lz4", clevel=5, shuffle=zarr.codecs.BloscShuffle.bitshuffle)


def get_cube_chunks(shape: tuple[int, ...], dtype: np.dtype) -> tuple[int, ...]:
    """Compute chunk shape for a 3-D spectral cube ``(H, W, S)``.

    Keeps full spatial planes intact and splits along the spectral axis so
    that each chunk is approximately ``_CUBE_CHUNK_TARGET_BYTES`` bytes
    uncompressed. Falls back to the full shape when the cube is smaller
    than the target.
    """
    if len(shape) != 3:
        return shape
    h, w, s = shape
    itemsize = np.dtype(dtype).itemsize
    spatial_bytes = h * w * itemsize
    if spatial_bytes == 0:
        return shape
    cs = max(1, min(s, _CUBE_CHUNK_TARGET_BYTES // spatial_bytes))
    return (h, w, cs)


def encode_dict(payload: Any) -> str:
    """Encode dict-like payloads to JSON, handling ndarray/unit types."""

    class NDArrayEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, units.QuantityUnit):
                return o.to_tuple()
            if type(o).__name__ in ("PydanticObjectId", "ObjectId"):
                return str(o)
            if hasattr(o, "model_dump"):
                return o.model_dump()  # type: ignore
            try:
                return super().default(o)
            except Exception:
                return str(o)

    return json.dumps(payload, cls=NDArrayEncoder)


def to_jsonable(payload: Any) -> Any:
    """Return a JSON-compatible Python object (dict/list/scalar), not a JSON string."""

    return json.loads(encode_dict(payload))


def decode_dict(payload: Any) -> Any:
    def _decode_bytes(value: Any) -> Any:
        if isinstance(value, np.bytes_):
            value = bytes(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, dict):
            return {_decode_bytes(k): _decode_bytes(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_decode_bytes(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_decode_bytes(v) for v in value)
        return value

    if isinstance(payload, np.bytes_):
        payload = bytes(payload)
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return payload

    return _decode_bytes(payload)


def parse_xunit(x_unit: Any):
    """Parse legacy and current x_unit representations."""

    x_quantity = None
    if isinstance(x_unit, (tuple, list)):
        if not x_unit[1]:
            # legacy
            x_quantity = units.Quantity.WAVENUMBER
            x_unit = units.Unit.CM_1
        else:
            return x_unit

    if x_quantity is None:
        x_quantity = units.Quantity.WAVENUMBER if x_unit == "1/cm" else units.Quantity.UNCALIBRATED
    x_unit = units.Unit(x_unit)
    return units.QuantityUnit(x_quantity, x_unit)


def read_attr_array(root, key: str):
    v = root.attrs.get(key)
    if v is None:
        return v
    return np.array(v)


def iter_members(group) -> Iterator[tuple[str, Any]]:
    """Compatibility wrapper for `Group.members()` (Zarr v2/v3 differences)."""

    # The existing code used `.members()`; keep that behaviour.
    return group.members()
