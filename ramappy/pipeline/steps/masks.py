from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image
from pydantic_extra_types.color import Color

from ramappy.core.images2d.rules import ColorType
from ramappy.core.masks import Mask
from ramappy.core.pipeline import (
    ParamsAcceptRoIX,
    ParamsOutputFile,
    ParamsSetResultId,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)
from ramappy.core.spectral_map import SpectralMap
from ramappy.utils import color_generator


class StepDeleteMaskParams(StepParams):
    """Parameters for `delete_mask`."""

    mask_id: str
    """Identifier of the mask to delete from the map."""


@pipeline_step(
    step_name="delete_mask",
    params_validator=SingleModelParamsValidator(StepDeleteMaskParams),
    friendly_name="Delete Mask",
    step_category=StepClass.UTILITY,
)
def delete_mask(spectral_map: SpectralMap, mask_id: str):
    """Delete a mask from the spectral map by its identifier.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    mask_id : str
        Identifier of the mask to remove.
    """
    spectral_map.delete_mask(mask_id)


class StepDeleteMasksGroupParams(StepParams):
    """Parameters for `delete_masks_group`."""

    group_id: str
    """Identifier of the mask group to delete from the map."""


@pipeline_step(
    step_name="delete_masks_group",
    params_validator=SingleModelParamsValidator(StepDeleteMasksGroupParams),
    friendly_name="Delete Mask Group",
    step_category=StepClass.UTILITY,
)
def delete_masks_group(spectral_map: SpectralMap, group_id: str):
    """Delete a mask group from the spectral map by its identifier.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    group_id : str
        Identifier of the mask group to remove.
    """
    spectral_map.delete_masks_group(group_id)


class StepNewMaskFromIndicesParams(StepParams, ParamsSetResultId):
    """Parameters for `new_mask_from_indices`."""

    color: ColorType | None = None
    """Color to assign to the new mask (can be a hex string or a Color object)."""

    name: str | None = "New Mask"
    """Name to assign to the new mask."""

    idxs: list[int]
    """List of pixel indices to include in the new mask."""


@pipeline_step(
    step_name="new_mask_from_indices",
    params_validator=SingleModelParamsValidator(StepNewMaskFromIndicesParams),
    friendly_name="New Mask from Indices",
    step_category=StepClass.UTILITY,
)
def new_mask_from_indices(
    spectral_map: SpectralMap, res_id: str, idxs: list[int], name: str | None, color: ColorType | None = None
):
    """Create a new mask from a list of flat pixel indices.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    res_id : str
        Key under which the new mask is stored.
    idxs : list of int
        Flat pixel indices to include in the mask.
    name : str or None
        Display name for the new mask.
    color : ColorType or None, optional
        Colour to assign to the mask. If ``None``, a colour is generated automatically.
    """
    color_str: str | None = color.as_hex(format="long") if isinstance(color, Color) else color
    spectral_map.masks[res_id] = spectral_map.new_mask(
        name=name, color=color_str if color_str is not None else color_generator(None), idxs=idxs
    )


class StepMaskOperationParams(StepParams, ParamsSetResultId):
    """Parameters for `mask_operation`."""

    B: str
    """Identifier of the second mask to use in the operation."""

    A: str | None = None
    """Identifier of the first mask to use in the operation (if ``None``, the operation is applied to the whole map)."""

    operation: Literal["diff", "union", "intersection"] = "union"
    """Operation to perform on the masks (`"diff"`, `"union"`, `"intersection"`)."""


@pipeline_step(
    step_name="mask_operation",
    params_validator=SingleModelParamsValidator(StepMaskOperationParams),
    friendly_name="Mask Operation",
    step_category=StepClass.UTILITY,
)
def mask_operation(
    spectral_map: SpectralMap,
    *,
    B: str,
    A: str | None = None,
    operation: Literal["diff", "union", "intersection"],
    res_id: str,
):
    """Compute a Boolean operation between two masks and store the result.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    B : str
        Identifier of the second (right-hand) mask.
    A : str or None, optional
        Identifier of the first (left-hand) mask. If ``None``, the operation
        is applied over the entire map.
    operation : {'diff', 'union', 'intersection'}
        Boolean set operation to perform.
    res_id : str
        Key under which the resulting mask is stored.
    """
    new_mask = spectral_map.mask_operation(mask_b=B, mask_a=A, op=operation)
    new_mask.rename("Mask")
    new_mask.set_color(color_generator(None))
    new_mask.visible = False

    spectral_map.masks[res_id] = new_mask


class StepMaskThresholdParams(StepParams, ParamsSetResultId, ParamsAcceptRoIX):
    """Parameters for `mask_threshold`."""

    agg: str | None = None
    """Aggregation method to apply to the pixel spectra before thresholding (if ``None``, the default aggregation method of the map is used)."""

    vmin: float | None = None
    """Minimum threshold value (if ``None``, the minimum value of the aggregated spectrum is used)."""

    vmax: float | None = None
    """Maximum threshold value (if ``None``, the maximum value of the aggregated spectrum is used)."""


@pipeline_step(
    step_name="mask_threshold",
    params_validator=SingleModelParamsValidator(StepMaskThresholdParams),
    friendly_name="Create Mask from Thresholded Aggregated Spectral Data",
    step_category=StepClass.UTILITY,
)
def mask_threshold(
    spectral_map: SpectralMap,
    *,
    agg: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    roi_x: list | None = None,
    res_id: str,
):
    """Generate a mask by thresholding aggregated spectral intensity.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to generate the mask from.
    agg : str or None, optional
        Aggregation method to apply before thresholding. Defaults to ``"max"``.
    vmin : float or None, optional
        Lower threshold (inclusive). Defaults to ``0.0``.
    vmax : float or None, optional
        Upper threshold (inclusive). Defaults to ``1.0``.
    roi_x : list or None, optional
        Spectral region to aggregate over.
    res_id : str
        Key under which the resulting mask is stored.
    """
    mask = spectral_map.mask_threshold(
        agg=agg if agg is not None else "max",
        vmin=vmin if vmin is not None else 0.0,
        vmax=vmax if vmax is not None else 1.0,
        roi_x=roi_x,
    )
    mask.rename("Mask")
    mask.set_color(color_generator(None))
    mask.visible = False

    spectral_map.masks[res_id] = mask


class StepMaskFromImageParams(StepParams, ParamsSetResultId):
    """Parameters for `mask_from_image`."""

    image_id: str
    """Identifier of the image to use for mask generation."""

    threshold_down: float | None = None
    """Lower threshold value (if ``None``, the minimum value of the image is used)."""

    threshold_up: float | None = None
    """Upper threshold value (if ``None``, the maximum value of the image is used)."""

    relative: bool = False
    """If ``True``, the thresholds are interpreted as relative values (between 0 and 1) instead of absolute values."""

    invert: bool = False
    """If ``True``, invert the mask (i.e., select pixels outside the threshold range)."""


@pipeline_step(
    step_name="mask_from_image",
    params_validator=SingleModelParamsValidator(StepMaskFromImageParams),
    friendly_name="Create Mask from Thresholded Image",
    step_category=StepClass.UTILITY,
)
def mask_from_image(
    spectral_map: SpectralMap,
    *,
    res_id: str,
    threshold_down: float | None = None,
    threshold_up: float | None = None,
    relative: bool = False,
    invert: bool = False,
    image_id: str,
):
    """Generate a mask by thresholding the pixel values of an existing image.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    res_id : str
        Key under which the resulting mask is stored.
    threshold_down : float or None, optional
        Lower threshold value. ``None`` uses the image minimum.
    threshold_up : float or None, optional
        Upper threshold value. ``None`` uses the image maximum.
    relative : bool, optional
        If ``True``, thresholds are interpreted as fractions in ``[0, 1]``.
    invert : bool, optional
        If ``True``, selects pixels *outside* the threshold range.
    image_id : str
        Identifier of the source image.
    """
    new_mask = spectral_map.mask_from_image(
        image_id=image_id,
        threshold_down=threshold_down,
        threshold_up=threshold_up,
        relative=relative,
        invert=invert,
    )
    new_mask.visible = False
    spectral_map.masks[res_id] = new_mask


class StepImportMaskParams(StepParams, ParamsSetResultId):
    """Parameters for `import_mask`."""

    mask_path: str | None = None
    """Path to the input file containing the mask data."""


@pipeline_step(
    step_name="import_mask",
    params_validator=SingleModelParamsValidator(StepImportMaskParams),
    friendly_name="Import Mask",
    step_category=StepClass.UTILITY,
)
def import_mask(spectral_map: SpectralMap, mask_path: str, res_id: str):
    """Import a binary mask from an image file.

    The image is converted to a 1-bit (black-and-white) format and all
    non-zero pixels are added to the mask.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map to modify in place.
    mask_path : str
        Path to the source image file.
    res_id : str
        Key under which the resulting mask is stored.
    """
    im = np.array(Image.open(mask_path).convert(mode="1"))
    spectral_map.masks[res_id] = Mask(idxs=np.argwhere(im.ravel() > 0).ravel(), img_shape=spectral_map.map_shape)


class StepExportMaskAsImageParams(StepParams, ParamsOutputFile):
    """Parameters for `export_mask_as_image`."""

    mask: str | None = None
    """Identifier of the mask to export as an image."""

    color: ColorType | None = None
    """Color to assign to the mask in the exported image (can be a hex string or a Color object)."""


@pipeline_step(
    step_name="export_mask_as_image",
    params_validator=SingleModelParamsValidator(StepExportMaskAsImageParams),
    friendly_name="Export Mask as Image",
    step_category=StepClass.UTILITY,
)
def export_mask_as_image(spectral_map: SpectralMap, mask: str, out_file: str, color: ColorType | None = None):
    """Render a mask as a coloured PNG image and export it to a file.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map containing the mask to export.
    mask : str
        Identifier of the mask to render.
    out_file : str
        Output file path. If empty, defaults to ``<map_name>_<mask>.png``.
    color : ColorType or None, optional
        Override colour for the mask pixels. If ``None``, the mask's assigned
        colour is used.
    """
    from ramappy.core.images2d.renderer import Image2DRenderer

    out_file = out_file or f"{spectral_map.name}_{mask}.png"
    color_str: str | None
    if color is None:
        color_str = spectral_map.masks[mask].color
    elif isinstance(color, Color):
        color_str = color.as_hex(format="long")
    else:
        color_str = None
    Image2DRenderer.render(spectral_map.masks[mask].to_image(viz_rules={"cmap": color_str})).save(out_file)


class StepRefineMaskParams(StepParams):
    """Parameters for `refine_mask`."""

    mask_id: str
    """Identifier of the mask to refine."""

    morphology_minsize: int | None = None
    """Minimum size of connected components to keep (if ``None``, no size filtering is applied)."""

    erosion: int = 0
    """Number of pixels to erode the mask (if 0, no erosion is applied)."""


@pipeline_step(
    step_name="refine_mask",
    params_validator=SingleModelParamsValidator(StepRefineMaskParams),
    friendly_name="Refine Mask",
    step_category=StepClass.UTILITY,
)
def refine_mask(spectral_map: SpectralMap, mask_id: str, morphology_minsize: int | None, erosion: int) -> None:
    """Refine a mask by removing small regions and applying morphological erosion.

    Parameters
    ----------
    spectral_map : SpectralMap
        The spectral map whose mask is refined in place.
    mask_id : str
        Identifier of the mask to refine.
    morphology_minsize : int or None
        Minimum size of connected components to retain. ``None`` disables size
        filtering.
    erosion : int
        Number of pixels to erode from the mask boundary. ``0`` disables erosion.
    """
    spectral_map.masks[mask_id].refine(morphology_minsize=morphology_minsize, erosion=erosion)


__all__ = [
    "StepDeleteMaskParams",
    "StepDeleteMasksGroupParams",
    "StepExportMaskAsImageParams",
    "StepImportMaskParams",
    "StepMaskFromImageParams",
    "StepMaskOperationParams",
    "StepMaskThresholdParams",
    "StepNewMaskFromIndicesParams",
    "StepRefineMaskParams",
    "delete_mask",
    "delete_masks_group",
    "export_mask_as_image",
    "import_mask",
    "mask_from_image",
    "mask_operation",
    "mask_threshold",
    "new_mask_from_indices",
    "refine_mask",
]
