from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _BaseStructuredMetadata(BaseModel):
    """Base class for structured metadata with explicit typed fields + extensible extras."""

    model_config = ConfigDict(extra="forbid")

    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_any(cls, value: BaseModel | dict[str, Any] | None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            known = cls.model_fields.keys()
            clean: dict[str, Any] = {}
            extra: dict[str, Any] = {}
            for key, val in value.items():
                if key in known:
                    clean[key] = val
                else:
                    extra[key] = val
            if extra:
                clean.setdefault("extra", {}).update(extra)
            return cls.model_validate(clean)
        if isinstance(value, BaseModel):
            return cls.from_any(value.model_dump())
        raise TypeError(f"Cannot convert {type(value)} to {cls.__name__}")


def metadata_as_extra(metadata: dict[str, Any] | None = None, /, **kwargs: Any) -> dict[str, Any]:
    """Wrap arbitrary metadata values under ``extra`` and normalize filename fields.

    Canonical filename field is top-level ``original_filename``.
    Legacy ``input_filename`` is accepted but normalized to ``original_filename``.
    """

    merged = dict(metadata or {})
    merged.update(kwargs)

    original_filename = merged.pop("original_filename", None)
    legacy_input_filename = merged.pop("input_filename", None)
    filename = original_filename if original_filename is not None else legacy_input_filename

    if isinstance(filename, (str, Path)):
        filename = Path(filename).name

    cleaned = {k: v for k, v in merged.items() if v is not None}
    payload: dict[str, Any] = {"extra": cleaned}
    if filename is not None:
        payload["original_filename"] = filename
    return payload


def metadata_with_extra(
    *,
    title: str | None = None,
    description: str | None = None,
    original_filename: str | None = None,
    instrument: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    vendor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build structured metadata with a single flexible ``extra`` container.

    Parameters
    ----------
    title, description, original_filename
        Standard top-level metadata fields.
    instrument
        Standard instrument metadata fields.
    extra
        Arbitrary structured metadata stored directly under ``extra``.
    vendor
        Importer/vendor-specific metadata stored under ``extra.vendor``.

    Returns
    -------
    dict[str, Any]
        Payload compatible with ``SpectralMapMetadata``.
    """

    def _clean_dict(value: dict[str, Any] | None) -> dict[str, Any]:
        if value is None:
            return {}
        return {k: v for k, v in value.items() if v is not None}

    payload: dict[str, Any] = {}
    if title is not None:
        payload["title"] = title
    if description is not None:
        payload["description"] = description
    if original_filename is not None:
        payload["original_filename"] = original_filename

    clean_instrument = _clean_dict(instrument)
    if clean_instrument:
        payload["instrument"] = clean_instrument

    merged_extra = _clean_dict(extra)
    clean_vendor = _clean_dict(vendor)

    if clean_vendor:
        merged_extra["vendor"] = clean_vendor

    payload["extra"] = merged_extra
    return payload


class SpectralMetadata(_BaseStructuredMetadata):
    """Metadata for spectral entities (`Spectrum` and derived data)."""

    original_filename: str | None = None
    title: str | None = None
    description: str | None = None


class MaskMetadata(_BaseStructuredMetadata):
    """Metadata for spatial masks."""

    role: str | None = None
    description: str | None = None


class ImageMetadata(_BaseStructuredMetadata):
    """Metadata for image layers."""

    modality: str | None = None
    description: str | None = None
