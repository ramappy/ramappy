from __future__ import annotations

import numpy as np
from PIL import Image

from ramappy import const
from ramappy.core.images2d import DataRules, Image2D, Image2DRenderer, VizRules
from ramappy.core.pipeline import (
    ParamsOutputFile,
    ParamsSetResultId,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)
from ramappy.core.spectral_map import SpectralMap
from ramappy.utils import generate_key, minmax


class IntensityImageParams(StepParams, ParamsSetResultId):
    """Parameters for `intensity_image`."""

    data_rules: DataRules | None = None
    """Data rules for the intensity image (if ``None``, the default intensity map is used)."""

    viz_rules: VizRules | None = None
    """Visualization rules for the intensity image (if ``None``, the default visualization is used)."""

    name: str = "Spectral Projection Image"
    """Name of the intensity image."""

    visible: bool = True
    """If ``True``, the intensity image is visible in the map outputs."""


@pipeline_step(
    step_name="spectral_projection_image",
    params_validator=SingleModelParamsValidator(IntensityImageParams),
    friendly_name="Spectral Projection Image",
    step_category=StepClass.VISUALIZATION,
)
def intensity_image(
    spectral_map: SpectralMap,
    *,
    res_id: str,
    data_rules: DataRules | None = None,
    viz_rules: VizRules | None = None,
    name: str = "Spectral Projection Image",
    visible: bool = True,
):
    """Generate a spectral projection (intensity map) image and store it in the map.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to compute the intensity image from.
    res_id : str
        Key under which the resulting :class:`~ramappy.core.images2d.Image2D` is stored.
    data_rules : DataRules or None, optional
        Data extraction rules. If ``None``, the map default is used.
    viz_rules : VizRules or None, optional
        Visualization rules. If ``None``, the map default is used.
    name : str, optional
        Display name for the generated image.
    visible : bool, optional
        Whether the image is visible in map outputs.
    """
    spectral_map.images[res_id] = spectral_map.intensity_map(
        data_rules=data_rules, viz_rules=viz_rules, name=name, locked=False, visible=visible
    )


class StepEditImageParams(StepParams):
    """Parameters for `edit_image`."""

    image_id: str | None = None  # TODO: use standard input id?
    """Identifier of the image to edit (if ``None``, a new image is created)."""

    keys_order: list[str] | None = None
    """New order of image keys in the map outputs (if ``None``, the order is not changed)."""

    group_id: str | None = None
    """Identifier of the image group to edit (if ``None``, the group is not changed)."""

    group_order: list[str] | None = None
    """New order of image group keys in the map outputs (if ``None``, the order is not changed)."""

    data_rules: DataRules | None = None
    """New data rules for the image (if ``None``, the data rules are not changed)."""

    viz_rules: dict | None = None  # we want to allow partial updates
    """New visualization rules for the image (if ``None``, the visualization rules are not changed)."""

    locked: bool | None = None
    """If ``True``, the image is locked and cannot be modified (if ``None``, the locked state is not changed)."""

    name: str | None = None
    """New name for the image or image group (if ``None``, the name is not changed)."""

    visible: bool | None = None
    """If ``True``, the image or image group is visible in the map outputs (if ``None``, the visibility is not changed)."""


@pipeline_step(
    step_name="edit_image",
    params_validator=SingleModelParamsValidator(StepEditImageParams),
    friendly_name="Edit Image",
    step_category=StepClass.VISUALIZATION,
)
def edit_image(
    spectral_map: SpectralMap,
    *,
    image_id: str,
    keys_order: list[str],
    group_id: str,
    group_order: list[str],
    data_rules: DataRules | None = None,
    viz_rules: dict | None = None,
    locked: bool | None = None,
    name: str | None = None,
    visible: bool | None = None,
):
    """Edit properties of an existing image, image group, or their ordering.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    image_id : str
        Identifier of the image or group to edit.
    keys_order : list of str
        New ordering of image keys when reordering.
    group_id : str
        Identifier of the image group to reorder.
    group_order : list of str
        New ordering of images within the group.
    data_rules : DataRules or None, optional
        Replacement data rules for the image.
    viz_rules : dict or None, optional
        Partial visualization rule updates for the image.
    locked : bool or None, optional
        New locked state for the image.
    name : str or None, optional
        New display name for the image or image group.
    visible : bool or None, optional
        New visibility state for the image or image group.
    """
    key = image_id
    group_key = group_id
    new_order = group_order

    if key in spectral_map.images:
        # modify/delete already existing image (if possible)
        if data_rules is not None and spectral_map.images[key].data_rules is not None:
            spectral_map.images[key] = spectral_map.intensity_map(data_rules=data_rules, viz_rules=viz_rules)
        elif viz_rules is not None:
            if spectral_map.images[key].viz_rules is not None:
                spectral_map.images[key].viz_rules.update(viz_rules)  # type: ignore
            else:
                spectral_map.images[key].viz_rules = viz_rules
        elif locked is not None:
            if spectral_map.images[key].data_rules is not None:
                spectral_map.images[key].locked = locked
                spectral_map.update_image(key)
        elif name is not None:
            spectral_map.images[key].name = name
            if spectral_map.images[key].spectrum is not None:
                spectral_map.images[key].spectrum.name = spectral_map.images[key].name  # type: ignore
        elif visible is not None:
            spectral_map.images[key].visible = visible
        else:
            spectral_map.delete_image(key)

    elif key in spectral_map.images_group:
        if name is not None:
            spectral_map.images_group[key].name = name
        elif visible is not None:
            spectral_map.images_group[key].visible = visible
        else:
            spectral_map.delete_images_group(key)
    elif keys_order is not None:
        spectral_map.images.set_order(keys_order)
        if group_key in spectral_map.images_group and set(new_order) == set(
            spectral_map.images_group[group_key].elem_ids or []
        ):
            spectral_map.images_group[group_key].elem_ids = new_order
    elif visible is not None:
        if not visible:
            for _k, image in spectral_map.images.items():
                image.visible = False
            for _k, group in spectral_map.images_group.items():
                group.visible = False
    else:
        if key is None:
            key = generate_key()
        spectral_map.images[key] = spectral_map.intensity_map(
            data_rules=data_rules, viz_rules=viz_rules, name=name, locked=locked, visible=visible
        )


class StepDeleteImageParams(StepParams):
    """Parameters for `delete_image`."""

    image_id: str
    """Identifier of the image to delete from the map."""


@pipeline_step(
    step_name="delete_image",
    params_validator=SingleModelParamsValidator(StepDeleteImageParams),
    friendly_name="Delete Image",
    step_category=StepClass.UTILITY,
)
def delete_image(spectral_map: SpectralMap, image_id: str):
    """Delete an image from the spectral map by its identifier.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    image_id : str
        Identifier of the image to remove.
    """
    spectral_map.delete_image(image_id)


class StepDeleteImagesGroupParams(StepParams):
    """Parameters for `delete_images_group`."""

    group_id: str
    """Identifier of the image group to delete from the map."""


@pipeline_step(
    step_name="delete_images_group",
    params_validator=SingleModelParamsValidator(StepDeleteImagesGroupParams),
    friendly_name="Delete Image Group",
    step_category=StepClass.UTILITY,
)
def delete_images_group(spectral_map: SpectralMap, group_id: str):
    """Delete an image group from the spectral map by its identifier.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    group_id : str
        Identifier of the image group to remove.
    """
    spectral_map.delete_images_group(group_id)


class StepExportImageParams(StepParams, ParamsOutputFile):
    """Parameters for `export_image`."""

    image_id: str | None = None
    """Identifier of the image to export."""


@pipeline_step(
    step_name="export_image",
    params_validator=SingleModelParamsValidator(StepExportImageParams),
    friendly_name="Export Image",
    step_category=StepClass.UTILITY,
)
def export_image(spectral_map: SpectralMap, image_id: str, out_file: str):
    """Render and export a single image to a PNG file.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map containing the image to export.
    image_id : str
        Identifier of the image to export.
    out_file : str
        Output file path. If empty, defaults to ``<map_name>_<image_id>.png``.
    """
    out_file = out_file or f"{spectral_map.name}_{image_id}.png"
    Image2DRenderer.render(spectral_map.images[image_id]).save(out_file)


class StepExportCompositeImageParams(StepParams, ParamsOutputFile):
    """Parameters for `export_composite_image`."""

    images: list[str] | str | None = None
    """List of image identifiers to include in the composite image (if ``None``, all images are included)."""

    images_groups: list[str] | str | None = None
    """List of image group identifiers to include in the composite image (if ``None``, all image groups are included)."""

    masks: list[str] | str | None = None
    """List of mask identifiers to include in the composite image (if ``None``, all masks are included)."""

    masks_groups: list[str] | str | None = None
    """List of mask group identifiers to include in the composite image (if ``None``, all mask groups are included)."""

    exclude_masks: list[str] | str | bool = False
    """List of mask identifiers to exclude from the composite image (if ``False``, no masks are excluded)."""

    background_color: str = "#ffffff00"
    """Background color for the composite image (in hex format, e.g., ``"#ffffff00"`` for transparent white)."""

    blend_mode: const.BlendModes = const.BlendModes.screen
    """Blend mode for combining images in the composite image (default is ``"screen"``)."""

    max_size: tuple[int, int] | None = None
    """Maximum size (width, height) for the composite image (if ``None``, no resizing is applied)."""

    only_visible: bool = False
    """If ``True``, only visible images and masks are included in the composite image."""


@pipeline_step(
    step_name="export_composite_image",
    params_validator=SingleModelParamsValidator(StepExportCompositeImageParams),
    friendly_name="Export Composite Image",
    step_category=StepClass.UTILITY,
)
def export_composite_image(
    spectral_map: SpectralMap,
    out_file: str,
    images: list[str] | str | None = None,
    images_groups: list[str] | str | None = None,
    masks: list[str] | str | None = None,
    masks_groups: list[str] | str | None = None,
    exclude_masks: list[str] | str | bool = False,
    background_color: str | None = None,
    blend_mode: const.BlendModes = const.BlendModes.screen,
    max_size: tuple[int, int] | None = None,
    only_visible: bool = False,
):
    """Composite selected images and masks and save to a single PNG file.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to composite from.
    out_file : str
        Output file path. If empty, defaults to ``<map_name>.png``.
    images : list of str or str or None, optional
        Image identifiers to include. ``None`` includes all.
    images_groups : list of str or str or None, optional
        Image group identifiers to include. ``None`` includes all.
    masks : list of str or str or None, optional
        Mask identifiers to include. ``None`` includes all.
    masks_groups : list of str or str or None, optional
        Mask group identifiers to include. ``None`` includes all.
    exclude_masks : list of str or str or bool, optional
        Mask identifiers to exclude from compositing.
    background_color : str or None, optional
        Background fill colour in hex format. Defaults to transparent white.
    blend_mode : BlendModes, optional
        Blending mode for combining layers.
    max_size : tuple of int or None, optional
        Maximum ``(width, height)`` to resize to. ``None`` keeps original size.
    only_visible : bool, optional
        When ``True``, only visible images and masks are included.
    """
    out_file = out_file or f"{spectral_map.name}.png"
    if background_color is None:
        background_color = "#ffffff00"
    spectral_map.get_composite_image(
        images=images,
        images_groups=images_groups,
        masks=masks,
        masks_groups=masks_groups,
        exclude_masks=exclude_masks,
        background_color=background_color,
        blend_mode=blend_mode,
        format=None,
        max_size=max_size,
    ).save(out_file)  # type: ignore


class RGBImageParams(StepParams, ParamsSetResultId):
    """Parameters for `rgb_image`."""

    r_id: str | None = None
    """Identifier of the image to use for the red channel (if ``None``, the red channel is black)."""

    g_id: str | None = None
    """Identifier of the image to use for the green channel (if ``None``, the green channel is black)."""

    b_id: str | None = None
    """Identifier of the image to use for the blue channel (if ``None``, the blue channel is black)."""

    name: str = "RGB Composite"
    """Name of the RGB composite image."""

    visible: bool = True
    """If ``True``, the RGB composite image is visible in the map outputs."""


@pipeline_step(
    step_name="rgb_image",
    params_validator=SingleModelParamsValidator(RGBImageParams),
    friendly_name="RGB Composite Image",
    step_category=StepClass.VISUALIZATION,
)
def rgb_image(
    spectral_map: SpectralMap,
    *,
    res_id: str,
    r_id: str | None = None,
    g_id: str | None = None,
    b_id: str | None = None,
    name: str = "RGB Composite",
    visible: bool = True,
):
    """Compose an RGB false-colour image from three grayscale images and store it.

    Each channel is independently min-max normalised to ``[0, 255]``. If an
    image identifier is ``None`` or not found, that channel is set to zero
    (black).

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map providing source images.
    res_id : str
        Key under which the resulting :class:`~ramappy.core.images2d.Image2D` is stored.
    r_id : str or None, optional
        Identifier of the image to use for the red channel.
    g_id : str or None, optional
        Identifier of the image to use for the green channel.
    b_id : str or None, optional
        Identifier of the image to use for the blue channel.
    name : str, optional
        Display name for the composite image.
    visible : bool, optional
        Whether the image is visible in map outputs.
    """
    rgb = np.zeros((*spectral_map.map_shape, 3), dtype=np.uint8)

    def _normalize(img_id):
        if not img_id or img_id not in spectral_map.images:
            return None
        img = spectral_map.images[img_id]
        if img.data is not None:
            data = img.data
            if hasattr(data, "filled"):
                data = data.filled(np.nan)  # type: ignore
            valid = ~np.isnan(data)
            if not np.any(valid):
                return np.zeros(spectral_map.map_shape, dtype=np.uint8)
            d_min, d_max = minmax(data[valid])
            if d_max == d_min:
                return np.zeros(spectral_map.map_shape, dtype=np.uint8)
            norm = (data - d_min) / (d_max - d_min)
            norm[~valid] = 0
            return (np.clip(norm, 0, 1) * 255).astype(np.uint8)
        elif img._image is not None:
            # Use raster image as fallback
            raster = img._image.convert("L")
            if raster.size != spectral_map.map_shape[::-1]:
                raster = raster.resize(spectral_map.map_shape[::-1], Image.Resampling.LANCZOS)
            return np.array(raster)
        return None

    r_data = _normalize(r_id)
    if r_data is not None:
        rgb[..., 0] = r_data

    g_data = _normalize(g_id)
    if g_data is not None:
        rgb[..., 1] = g_data

    b_data = _normalize(b_id)
    if b_data is not None:
        rgb[..., 2] = b_data

    pil_img = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    spectral_map.images[res_id] = Image2D(image=pil_img, name=name, locked=True, visible=visible)


__all__ = [
    "IntensityImageParams",
    "RGBImageParams",
    "StepDeleteImageParams",
    "StepDeleteImagesGroupParams",
    "StepEditImageParams",
    "StepExportCompositeImageParams",
    "StepExportImageParams",
    "delete_image",
    "delete_images_group",
    "edit_image",
    "export_composite_image",
    "export_image",
    "intensity_image",
    "rgb_image",
]
