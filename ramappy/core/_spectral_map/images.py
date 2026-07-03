"""Image-related behaviors for :class:`SpectralMap <ramappy.core.spectral_map.SpectralMap>`.

This module is internal: public API remains in :mod:`ramappy.core.spectral_map`.

The goal is to keep :class:`SpectralMap <ramappy.core.spectral_map.SpectralMap>` modular while preserving its
public import path and behavior.
"""

from __future__ import annotations

import base64
import contextlib
import warnings
from io import BytesIO

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.integrate import simpson

from ramappy import const
from ramappy.core.collections import OrderedEntityMap
from ramappy.core.images2d import DataRules, Image2D, Image2DGroup, Image2DRenderer, VizRules
from ramappy.core.masks import Mask, MaskGroup
from ramappy.utils import blend_images, fwhm, rubberband, rubberband_batch


class _SpectralMapImagesMixin:
    """Internal mixin implementing image and composite-image utilities for SpectralMap."""

    images: OrderedEntityMap[Image2D]
    images_group: OrderedEntityMap[Image2DGroup]
    masks: OrderedEntityMap[Mask]
    masks_group: OrderedEntityMap[MaskGroup]
    data: np.ndarray
    x: np.ndarray
    map_shape: tuple[int, int]
    img_width: int
    img_height: int

    def get_image(self, image_id: str) -> Image2D:
        """Retrieve an image by ID."""
        return self.images[image_id]

    def delete_image(self, image_id: str):
        """Delete an image by ID."""
        group_id = self.find_group_given_image(image_id)
        self.images.pop(image_id, None)
        if group_id is not None:
            grp = self.images_group.get(group_id)
            if grp is not None and grp.elem_ids is not None:
                grp.elem_ids.remove(image_id)
                if len(grp.elem_ids) == 0:
                    self.images_group.pop(group_id)

    def update_images(self):
        """Update all images."""
        for image_id in self.images:
            self.update_image(image_id)

    def update_image(self, image_id: str):
        """Update a specific image by ID."""
        if not self.images[image_id].locked:
            self.images[image_id] = self.intensity_map(
                data_rules=self.images[image_id].data_rules,
                viz_rules=self.images[image_id].viz_rules,
                name=self.images[image_id].name,
                locked=self.images[image_id].locked,
                visible=self.images[image_id].visible,
            )

    def new_images_group(self, elem_ids: list[str] | None = None, **kwargs) -> Image2DGroup:
        """Create a new group of images."""
        return Image2DGroup(elem_ids=elem_ids, **kwargs)

    def get_images_group(self, group_id: str) -> Image2DGroup:
        """Retrieve an image group by ID."""
        if group_id in self.images_group:
            return self.images_group[group_id]
        raise ValueError(f"Group {group_id} not found")

    def find_group_given_image(self, image_id: str) -> str | None:
        """Find the group ID containing the given image ID."""
        return self.images[image_id].parent_group

    def delete_images_group(self, group_id: str):
        """Delete an image group by ID."""
        if group_id in self.images_group:
            grp = self.images_group.get(group_id)
            for image in grp.elem_ids if grp is not None and grp.elem_ids is not None else []:
                self.images.pop(image)

            self.images_group.pop(group_id)
        else:
            raise ValueError(f"Group {group_id} not found")

    def intensity_map(
        self,
        *,
        data_rules: DataRules | dict | None = None,
        viz_rules: VizRules | dict | None = None,
        name: str = "Spectral Projection Image",
        locked: bool = False,
        visible: bool = True,
    ) -> Image2D:
        """Return the 2D intensity map, wrapped in an :class:`Image2D <ramappy.core.images2d.Image2D>`."""
        data_rules = data_rules if isinstance(data_rules, DataRules) else DataRules.model_validate(data_rules or {})

        img = self.get_band_intensity(data_rules.mask, data_rules.agg, data_rules.roi_x_A, data_rules.remove_baseline)

        if data_rules.roi_x_B is not None:
            img_b = self.get_band_intensity(
                data_rules.mask, data_rules.agg, data_rules.roi_x_B, data_rules.remove_baseline
            )
            img /= img_b

        return Image2D(data=img, data_rules=data_rules, viz_rules=viz_rules, name=name, locked=locked, visible=visible)

    def get_band_intensity(
        self,
        mask: str | None = None,
        agg: str = "mean",
        roi_x: npt.ArrayLike | None = None,
        remove_baseline: bool | None = None,
        exclude_mask: str | None = None,
        **kwargs,
    ):
        """Calculate the intensity map for a specific band."""
        if exclude_mask is not None:
            mask_obj = self.get_mask(mask)
            mask_obj -= self.masks[exclude_mask]
            mask = mask_obj  # Update mask to the modified one

        idxs, x_idx, pixel_idx = self.get_indices(mask, roi_x)
        intensities = self.data[idxs]

        if remove_baseline and (isinstance(x_idx, slice) or x_idx.shape > (2,)):
            # otherwise baseline correction is meaningless
            # warning: we assume `roi_x` to contain a single region!
            # rubberband uses ConvexHull, which releases the GIL => use 'threads' ?
            # IMPORTANT: if idxs is pure slicing, `intensities` is a view into self.data.
            # We must not mutate the underlying data when computing images.
            if isinstance(pixel_idx, slice) and isinstance(x_idx, slice):
                intensities = intensities.copy()

            x = self.x[x_idx]
            if intensities.ndim == 2 and intensities.shape[const.Axis.SPECTRAL] > 2:
                intensities -= rubberband_batch(intensities, x)
            else:
                intensities -= self.apply_func(rubberband, data=intensities, by="pixel", x=x)

        if intensities.ndim > 1:
            if agg == "area" and (isinstance(x_idx, slice) or (not isinstance(x_idx, slice) and x_idx.shape > (1,))):
                intensities = simpson(intensities, x=self.x[x_idx], axis=const.Axis.SPECTRAL)
            elif agg == "max":
                intensities = np.max(intensities, axis=const.Axis.SPECTRAL)
            elif agg == "min":
                intensities = np.min(intensities, axis=const.Axis.SPECTRAL)
            elif agg == "fwhm":
                intensities = self.apply_func(fwhm, data=intensities, by="pixel")
            else:
                # default to mean (works even for a single index)
                intensities = np.mean(intensities, axis=const.Axis.SPECTRAL)

        if not isinstance(pixel_idx, slice):
            # i.e., we are using a mask
            mask_resolved: Mask = self.get_mask(mask if isinstance(mask, str) else None)
            img = np.full(self.map_shape, np.nan, dtype=intensities.dtype)
            img.ravel()[pixel_idx] = intensities
            img = np.ma.masked_array(
                img, mask=mask_resolved.get_2Dmask(invert=True)
            )  # a True value marks invalid pixels
        else:
            img = intensities.reshape(self.map_shape)

        return img

    def get_composite_image(
        self,
        images: list[str] | str | None = None,
        images_groups: list[str] | str | None = None,
        masks: list[str] | str | None = None,
        masks_groups: list[str] | str | None = None,
        exclude_masks: list[str] | str | bool = "0",
        background_color: str | tuple[int, int, int] | tuple[int, int, int, int] = "#ffffff00",
        blend_mode: const.BlendModes = const.BlendModes.screen,
        format: str | None = None,
        base64_encode: bool = False,
        max_size: tuple[int, int] | None = (512, 512),
        only_visible: bool = False,
    ) -> Image.Image | BytesIO | bytes:
        """Generate a composite image from multiple layers."""
        background = Image.new("RGBA", (self.img_width, self.img_height), color=background_color)

        def _get_rendered_array(im):
            rendered = Image2DRenderer.render(im).convert("RGBA")
            if rendered.size != (self.img_width, self.img_height):
                rendered = rendered.resize((self.img_width, self.img_height), resample=Image.Resampling.LANCZOS)
            return np.array(rendered) / 255.0

        def _visible_images():
            for im in self.images.values():
                if im.visible and (im.parent_group is None or self.images_group[im.parent_group].visible):
                    with contextlib.suppress(ValueError):
                        yield _get_rendered_array(im)

        if images is None and images_groups is None:
            imgs = list(_visible_images())
        else:
            imgs = []
            if isinstance(images, str):
                images = [images]
            elif images is None:
                images = []

            if images_groups is not None:
                if isinstance(images_groups, str):
                    images_groups = [images_groups]
                for group_id in images_groups:
                    if group_id in self.images_group:
                        # add all images in the group
                        images.extend(self.images_group[group_id].elem_ids or [])
                    else:
                        warnings.warn(f"Image group {group_id} not found, skipping", stacklevel=2)

            for img_id in images:
                if img_id in self.images and (not only_visible or self.images[img_id].visible):
                    with contextlib.suppress(ValueError):
                        imgs.append(_get_rendered_array(self.images[img_id]))
                else:
                    warnings.warn(f"Image {img_id} not found, skipping", stacklevel=2)

            if exclude_masks is not False:
                warnings.warn(
                    "The `exclude_masks` parameter is ignored when `images` or `images_groups` are specified.",
                    UserWarning,
                    stacklevel=2,
                )
            exclude_masks = True

        if len(imgs) == 0:
            res = background
        else:
            res_blend = blend_images(imgs, mode=blend_mode)
            assert res_blend is not None
            res = Image.alpha_composite(res_blend, background)

        if masks is not None or masks_groups is not None:
            if exclude_masks is not False:
                warnings.warn(
                    "The `exclude_masks` parameter is ignored when `masks` or `masks_groups` are specified. "
                    "Use `only_visible` to filter out invisible masks.",
                    UserWarning,
                    stacklevel=2,
                )
            exclude_masks = False

            if isinstance(masks, str):
                masks = [masks]
            elif masks is None:
                masks = []

            if masks_groups is not None:
                if isinstance(masks_groups, str):
                    masks_groups = [masks_groups]
                for group_id in masks_groups:
                    if group_id in self.masks_group:
                        masks.extend(self.masks_group[group_id].elem_ids or [])
                    else:
                        warnings.warn(f"Mask group {group_id} not found, skipping", stacklevel=2)
            for mask_id in masks:
                if mask_id in self.masks and (not only_visible or self.masks[mask_id].visible):
                    res = Image.alpha_composite(res, Image2DRenderer.render(self.masks[mask_id].to_image()))
                else:
                    warnings.warn(f"Mask {mask_id} not found, skipping", stacklevel=2)

        if exclude_masks is not True:
            if isinstance(exclude_masks, str):
                excluded_masks = [exclude_masks]
            elif isinstance(exclude_masks, list):
                excluded_masks = exclude_masks
            else:
                excluded_masks = []
            for k, m in self.masks.items():
                if (
                    m.visible
                    and k not in excluded_masks
                    and (m.parent_group is None or self.masks_group[m.parent_group].visible)
                ):
                    res = Image.alpha_composite(res, Image2DRenderer.render(m.to_image()))

        if max_size is not None and (self.img_width > max_size[0] or self.img_height > max_size[1]):
            res.thumbnail(max_size, Image.Resampling.LANCZOS)

        if format is not None:
            buffered = BytesIO()
            res.save(buffered, format=format)
            if base64_encode:
                return base64.b64encode(buffered.getvalue())
            return buffered

        return res
