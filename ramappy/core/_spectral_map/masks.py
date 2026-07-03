"""Mask-related behaviors for :class:`SpectralMap <ramappy.core.spectral_map.SpectralMap>`.

This module is internal: public API remains in :mod:`ramappy.core.spectral_map`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import matplotlib.colors as cls
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ramappy.core.collections import OrderedEntityMap
from ramappy.core.masks import Mask, MaskGroup
from ramappy.utils import color_generator


class _SpectralMapMasksMixin:
    """Internal mixin implementing mask/group utilities for SpectralMap."""

    # NOTE: these methods rely on attributes set by `SpectralMap.__init__`:
    # - self.masks: OrderedEntityMap[str, Mask]
    # - self.masks_group: OrderedEntityMap[str, MaskGroup]
    # - self.images: OrderedEntityMap[str, Image2D]
    # - self.images_group: OrderedEntityMap[str, Image2DGroup]
    # - self.map_shape

    from ramappy.core.images2d import Image2D

    masks: OrderedEntityMap[Mask]
    masks_group: OrderedEntityMap[MaskGroup]
    images: OrderedEntityMap[Image2D]
    map_shape: tuple[int, int]

    @abstractmethod
    def get_band_intensity(self, **kwargs) -> np.ndarray: ...
    @abstractmethod
    def get_image(self, image_id: str) -> Image2D: ...

    def update_masks_group_color(self, group: str | MaskGroup, colormap: str | None = None):
        """Update the color of a mask group."""
        if isinstance(group, str):
            group = self.get_masks_group(group)

        if colormap is None:
            colormap = group.color
        else:
            group.set_color(colormap)

        cmap = plt.get_cmap(group.color)
        n_masks = len(group.elem_ids or [])
        colors = cmap(np.linspace(0, 1, n_masks))

        for i, mask_id in enumerate(group.elem_ids or []):
            if mask_id not in self.masks:
                continue
            self.masks[mask_id].set_color(cls.rgb2hex(colors[i]))

    def new_masks_group(self, elem_ids: list[str] | None = None, **kwargs) -> MaskGroup:
        """Create a new group of masks."""
        new_group = MaskGroup(elem_ids=elem_ids, **kwargs)
        self.update_masks_group_color(new_group)
        return new_group

    def get_masks_group(self, group_id: str) -> MaskGroup:
        """Retrieve a mask group by ID."""
        if group_id in self.masks_group:
            return self.masks_group[group_id]
        raise ValueError(f"Group {group_id} not found")

    def find_group_given_mask(self, mask_id: str) -> str | None:
        """Find the group ID containing the given mask ID."""
        for group_key, group_val in self.masks_group.items():
            if mask_id in (group_val.elem_ids or []):
                return group_key
        return None

    def change_spectrum_agg(self, group_id: str, spectrum_agg: Literal["mean", "centroid"]):
        """Change the spectrum aggregation method for a mask group."""
        group = self.get_masks_group(group_id)
        group.spectrum_agg = spectrum_agg
        for key in group.elem_ids or []:
            self.get_mask(key).set_spectrum_agg(spectrum_agg)

    def delete_masks_group(self, group_id: str):
        """Delete a mask group by ID."""
        if group_id in self.masks_group:
            grp = self.masks_group[group_id]
            for mask in grp.elem_ids or []:
                self.delete_mask(mask)

            self.masks_group.pop(group_id)
        else:
            raise ValueError(f"Group {group_id} not found")

    def new_mask(self, idxs: npt.ArrayLike | None = None, **kwargs) -> Mask:
        """Create a new mask."""
        return Mask(idxs=idxs, img_shape=self.map_shape, **kwargs)

    def get_mask(self, mask_id: str | None = None) -> Mask:
        """Retrieve a mask by ID."""
        if mask_id is None:
            return self.new_mask()
        return self.masks[mask_id]

    def delete_mask(self, mask_id: str):
        """Delete a mask by ID."""
        if mask_id != "0":
            self.masks.pop(mask_id, None)
            for _key, img in self.images.items():
                if img.data_rules is not None and img.data_rules.mask == mask_id:
                    img.data_rules.mask = None
        else:
            raise ValueError("Cannot delete default selection mask.")

    def mask_operation(
        self,
        mask_b: str | Mask,
        mask_a: str | Mask | None = None,
        mask_c: str | Mask | None = None,
        op: Literal["diff", "union", "intersection"] = "diff",
    ) -> Mask:
        """Perform a binary operation between two masks."""
        mask_a_obj = mask_a if isinstance(mask_a, Mask) else self.get_mask(mask_a)
        mask_b_obj = mask_b if isinstance(mask_b, Mask) else self.get_mask(mask_b)

        if mask_c is None:
            mask_c_obj = self.new_mask(np.array([]), editable=True)
        else:
            mask_c_obj = mask_c if isinstance(mask_c, Mask) else self.get_mask(mask_c)

        if op == "diff":
            indices = np.setdiff1d(mask_a_obj.indices, mask_b_obj.indices)
            mask_c_obj.idxs = indices
        elif op == "union":
            indices = np.union1d(mask_a_obj.indices, mask_b_obj.indices)
            mask_c_obj.idxs = indices
        elif op == "intersection":
            indices = np.intersect1d(mask_a_obj.indices, mask_b_obj.indices)
            mask_c_obj.idxs = indices

        return mask_c_obj

    def mask_threshold(
        self,
        agg: str = "max",
        vmin: float = 0.0,
        vmax: float = 1.0,
        roi_x: npt.ArrayLike | None = None,
    ):
        """Create a new mask with the specified intensity threshold."""
        intensities = self.get_band_intensity(agg=agg, roi_x=roi_x).flatten()

        min_val = intensities.min(axis=None)
        interval = intensities.max(axis=None) - min_val
        min_intensity = min_val + vmin * interval
        max_intensity = min_val + vmax * interval

        mask = np.argwhere((intensities >= min_intensity) & (intensities <= max_intensity)).flatten()

        return self.new_mask(mask, editable=False, spectrum_agg=None)

    def mask_from_image(
        self,
        image_id: str,
        *,
        threshold_down: float | None = None,
        threshold_up: float | None = None,
        relative: bool = False,
        invert: bool = False,
    ) -> Mask:
        """Create a new mask from the specified image."""
        if threshold_down is None and threshold_up is None:
            raise ValueError("At least one threshold must be specified to create a mask")
        img = self.get_image(image_id)
        if img.data is None:
            raise ValueError(f"Image {image_id} does not have any data associated (RBG image?)")
        color = (img.viz_rules.cmap if img.viz_rules is not None else None) or color_generator()
        if relative:
            img_min, img_max = img.data_range  # type: ignore
            threshold_down = img_min + threshold_down * (img_max - img_min) if threshold_down else None
            threshold_up = img_min + threshold_up * (img_max - img_min) if threshold_up else None

        mask_down = (img.data <= threshold_down) if threshold_down else False
        mask_up = (img.data >= threshold_up) if threshold_up else False
        mask = np.flatnonzero((mask_down | mask_up) if not invert else (~mask_down & ~mask_up))

        return self.new_mask(mask, name=img.name, color=color)
