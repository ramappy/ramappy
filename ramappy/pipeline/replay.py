from __future__ import annotations

from typing import TYPE_CHECKING

from ramappy.core.collections import OrderedEntityMap
from ramappy.core.images2d import Image2D
from ramappy.core.masks import Mask
from ramappy.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from ramappy.core.spectral_map import SpectralMap


class ArtifactReplayer:
    """
    Given a SpectralMap that has been processed by a Pipeline,
    re-derive images and attempt to restore masks.
    """

    def replay_images(
        self,
        source: SpectralMap,
        images: OrderedEntityMap[Image2D],
        pipeline: Pipeline,
    ) -> OrderedEntityMap[Image2D]:
        """Re-derive all images with a source_step_id using their DataRules."""
        new_images = OrderedEntityMap[Image2D]()
        for img_id, img in images.items():
            if img.source_step_id is not None:
                # Find if the step still exists in the pipeline
                step_exists = any(s.step_id == img.source_step_id for s in pipeline.steps)
                if step_exists and img.data_rules:
                    # Re-derive image from data using rules

                    try:
                        new_img = source.intensity_map(
                            data_rules=img.data_rules,
                            viz_rules=img.viz_rules,
                            name=img.name,
                            locked=img.locked,
                            visible=img.visible,
                        )
                        new_img.source_step_id = img.source_step_id
                        new_images[img_id] = new_img
                        continue
                    except Exception:
                        pass
            # If not re-derivable or re-derivation failed, copy if locked or imported
            new_images[img_id] = img
        return new_images

    def replay_masks(
        self,
        source: SpectralMap,
        masks: OrderedEntityMap[Mask],
        pipeline: Pipeline,
        original_shape: tuple[int, int],
    ) -> tuple[OrderedEntityMap[Mask], list[str]]:
        """
        Re-derive algorithm masks by replaying generating steps.
        Attempt coordinate remapping for user-drawn masks.
        Returns (replayed_masks, list_of_dropped_mask_ids).
        """
        new_masks = OrderedEntityMap[Mask]()
        dropped = []

        current_shape = (source.img_height, source.img_width)
        same_shape = current_shape == original_shape

        for mask_id, mask in masks.items():
            if mask.source_step_id is not None:
                # TODO: Implement re-running the generating step
                # For now, we keep it if shape matches
                if same_shape:
                    new_masks[mask_id] = mask
                else:
                    dropped.append(mask_id)
            # User-drawn mask
            elif same_shape:
                new_masks[mask_id] = mask
            else:
                # TODO: Coordinate remapping for geometric transforms
                dropped.append(mask_id)

        return new_masks, dropped
