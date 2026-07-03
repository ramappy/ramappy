from __future__ import annotations

from typing import Any

import ramappy
from ramappy.core import SpectralMap

from .common import to_jsonable

ZARR_THUMBNAIL_MAX_SIZE = (256, 256)
TRANSIENT_METADATA_KEYS = {"live_update", "mean_spectrum", "last_access", "project_id"}


def sanitize_metadata_payload(metadata_payload: object) -> object:
    """Remove transient RamApp runtime keys from persisted metadata payloads."""

    if not isinstance(metadata_payload, dict):
        return metadata_payload

    sanitized = {k: v for k, v in metadata_payload.items() if k not in TRANSIENT_METADATA_KEYS}

    extra = sanitized.get("extra")
    if isinstance(extra, dict):
        sanitized["extra"] = {k: v for k, v in extra.items() if k not in TRANSIENT_METADATA_KEYS}

    return sanitized


def update_zarr_project_attrs(root, spectral_map: SpectralMap, *, name: str | None = None):
    """Synchronize root and data attrs shared by full and incremental writers."""

    thumbnail_obj = spectral_map.get_composite_image(
        format="png",
        base64_encode=True,
        max_size=ZARR_THUMBNAIL_MAX_SIZE,
    )
    if isinstance(thumbnail_obj, bytes):
        thumbnail = thumbnail_obj.decode("ascii")
    elif isinstance(thumbnail_obj, str):
        thumbnail = thumbnail_obj
    else:
        thumbnail = str(thumbnail_obj)

    metadata = getattr(spectral_map, "smap_metadata", None)
    if metadata is not None and hasattr(metadata, "model_dump"):
        metadata_payload = metadata.model_dump()
    else:
        metadata_payload = (getattr(spectral_map, "metadata", None) or {}).copy()

    metadata_payload = sanitize_metadata_payload(metadata_payload)

    root.attrs.update(
        {
            "name": name if name is not None else spectral_map.name,
            "version": ramappy.__version__,
            "format_version": 2,
            "thumbnail": thumbnail,
            "metadata": to_jsonable(metadata_payload),
            "spatial_grid": to_jsonable(spectral_map.spatial_grid.to_dict()),
            "history": to_jsonable(spectral_map.history.gen_pipeline()),
        }
    )

    data_group = root.require_group("data")
    data_group.attrs.update(
        {
            "roi_x": spectral_map.roi_x.tolist(),
            "x_axis_unit": list(spectral_map.x_axis_unit.to_tuple()) if spectral_map.x_axis_unit is not None else None,
            "data_unit": list(spectral_map.data_unit.to_tuple()) if spectral_map.data_unit is not None else None,
        }
    )

    return data_group


def sync_zarr_collection_attrs(root, spectral_map: SpectralMap):
    """Synchronize collection/group metadata without rewriting entity payloads."""

    images_group = root.require_group("images")
    images_group.attrs["key_order"] = list(spectral_map.images.ordered_keys)
    images_group.attrs["groups_order"] = list(spectral_map.images_group.ordered_keys)
    for key in list(images_group.keys()):
        child = images_group[key]
        if getattr(child, "attrs", {}).get("group") and key not in spectral_map.images_group:
            del images_group[key]
    for group_id, group in spectral_map.images_group.items():
        current_group = images_group.require_group(group_id)
        current_group.attrs.update(
            {
                "name": group.name,
                "visible": group.visible,
                "group": True,
                "key_order": group.elem_ids,
            }
        )

    masks_group = root.require_group("masks")
    masks_group.attrs["key_order"] = list(spectral_map.masks.ordered_keys)
    masks_group.attrs["groups_order"] = list(spectral_map.masks_group.ordered_keys)
    for key in list(masks_group.keys()):
        child = masks_group[key]
        if getattr(child, "attrs", {}).get("group") and key not in spectral_map.masks_group:
            del masks_group[key]
    for group_id, mask_group in spectral_map.masks_group.items():
        grp_any: Any = mask_group  # MaskGroup has .color and .spectrum_agg; widen for type-checker
        current_group = masks_group.require_group(group_id)
        current_group.attrs.update(
            {
                "name": grp_any.name,
                "visible": grp_any.visible,
                "cmap": grp_any.color,
                "mask_spectrum_aggregation": grp_any.spectrum_agg,
                "group": True,
                "key_order": grp_any.elem_ids,
            }
        )

    spectra_group = root.require_group("spectra")
    return images_group, masks_group, spectra_group
