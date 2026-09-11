import copy
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import is_color_like
from scipy.ndimage import binary_erosion, generate_binary_structure, iterate_structure, label

from .metadata import MaskMetadata
from .spectrum import Spectrum


class Mask:
    """Mask / Spatial Region of Interest on the 2-D map."""

    def __init__(
        self,
        idxs: npt.ArrayLike | None = None,
        name: str = "Cursor",
        img_shape: tuple[int, int] | None = None,
        img_height: int | None = None,
        img_width: int | None = None,
        parent_group: str | None = None,
        comment: str | None = None,
        color: str | None = None,  # Handled dynamically below
        editable: bool | None = True,
        visible: bool | None = True,
        spectrum: Spectrum | None = None,  # for, e.g., cluster centroids
        spectrum_agg: Literal["mean", "centroid"] = "mean",  # preferred aggregate for spectrum
        metadata: MaskMetadata | dict | None = None,
        source_step_id: str | None = None,
    ):
        self.name = name
        self.source_step_id: str | None = source_step_id
        if img_shape is not None:
            self.set_mask_shape(*img_shape)
        else:
            if img_height is None or img_width is None:
                raise ValueError("Missing shape of mask")
            self.set_mask_shape(img_height, img_width)

        self.idxs = idxs

        self.spectrum = spectrum
        self.set_spectrum_agg(spectrum_agg)
        self._mask_metadata: MaskMetadata = MaskMetadata.from_any(copy.deepcopy(metadata))

        self.parent_group = parent_group

        self.comment = comment
        if color is None:
            color = "#5A3FD0" if name == "Cursor" else "#3B82F6"
        self.set_color(color)
        self.editable = editable
        self.visible = visible

    def __add__(self, other):
        if self.mask_shape != other.mask_shape:
            raise ValueError("Masks have different dimensions")
        idxs = np.union1d(self.idxs, other.idxs)
        return Mask(idxs, img_shape=self.mask_shape)

    def __or__(self, other):
        return self.__add__(other)

    def __and__(self, other):
        if self.mask_shape != other.mask_shape:
            raise ValueError("Masks have different dimensions")
        idxs = np.intersect1d(self.idxs, other.idxs)
        return Mask(idxs, img_shape=self.mask_shape)

    def __sub__(self, other):
        if self.mask_shape != other.mask_shape:
            raise ValueError("Masks have different dimensions")
        idxs = np.setdiff1d(self.idxs, other.idxs, assume_unique=True)
        return Mask(idxs, img_shape=self.mask_shape)

    def is_empty(self) -> bool:
        return self.idxs.size == 0

    @property
    def metadata(self) -> dict:
        """Additional metadata fields (`extra`) as a mutable dict."""
        return self._mask_metadata.extra

    @metadata.setter
    def metadata(self, value: MaskMetadata | dict | None):
        self._mask_metadata = MaskMetadata.from_any(copy.deepcopy(value))

    @property
    def mask_metadata(self) -> MaskMetadata:
        """Structured metadata model for this mask."""
        return self._mask_metadata

    @property
    def get_name(self):
        return self.name

    def rename(self, new_name):
        self.name = new_name

    def __str__(self):
        return self.name

    @property
    def idxs(self):
        return self._idxs

    @idxs.setter
    def idxs(self, idxs=None):
        if idxs is None:
            self._idxs = np.arange(self.img_height * self.img_width, dtype=np.intp)
        else:
            if isinstance(idxs, int):
                idxs = [idxs]
            self._idxs = np.array(idxs, dtype=np.intp)

    @property
    def indices(self):
        return self.idxs

    def update(self, idxs):
        self.idxs = idxs

    def unravel_index(self):
        """Return x,y indices for the 2-D map given row-major 1-D indices."""
        return np.unravel_index(self.idxs, (self.img_height, self.img_width))

    @property
    def mask_size(self):
        return self.idxs.size

    @property
    def mask_shape(self):
        """The size of the 2D image as height x width."""
        return self.img_height, self.img_width

    @mask_shape.setter
    def mask_shape(self, value: tuple[int, int]) -> None:
        img_height, img_width = value
        self.img_height, self.img_width = img_height, img_width

    def set_mask_shape(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

    def set_color(self, color):
        if color is not None and is_color_like(color):
            self.color = color
        else:
            self.color = "#3B82F6"  # CSS dodgerblue-like vibrant blue

    def set_spectrum_agg(self, spectrum_agg: Literal["mean", "centroid"] | None):
        self.spectrum_agg = spectrum_agg

    def get_2Dmask(self, invert=False, idxs=None):
        """Return the 2D (boolean) mask, having True on pixels belonging to the Mask."""
        if idxs is None:
            idxs = self.idxs
        boolmask = np.full(self.mask_shape, invert, dtype=bool)
        boolmask.ravel()[idxs] = not invert
        return boolmask

    def to_image(self, viz_rules: dict | None = None):
        from ramappy.core.images2d import Image2D

        if viz_rules is None:
            viz_rules = {"cmap": self.color}
        return Image2D(
            data=self.get_2Dmask(),
            viz_rules=viz_rules,
        )

    def refine(self, morphology_minsize: int | None = 64, erosion: int = 0) -> None:
        """Refine mask by removing small isolated regions and/or perform erosion"""
        # apply morphological operations to clean a mask (i.e., remove small objects and/or perform erosion)
        img = self.get_2Dmask()  # True for pixels in mask
        modified = False

        if erosion > 0:
            # perform an erosion operation with a "disk" of provided radius to improve the edges
            struct = iterate_structure(generate_binary_structure(2, 1), erosion).astype(np.uint8)
            img = binary_erosion(img, structure=struct, border_value=0).astype(bool)  # .ravel()
            modified = True

        if morphology_minsize is not None:
            connectivity_matrix, n_components = label(img, structure=generate_binary_structure(2, 1))
            if n_components > 1:  # `label` counts foreground components only, background is never one of them
                # we have more than 1 connected region (ie, maybe something other than the true fg)
                component_sizes = np.bincount(connectivity_matrix.ravel())

                too_small = component_sizes < morphology_minsize
                too_small_mask = too_small[connectivity_matrix]

                # TODO: fill small regions (holes), NOT in a corner
                img.ravel()[too_small_mask.ravel()] = False
                modified = True

        if modified:
            self.idxs = np.flatnonzero(img.ravel())


class MaskGroup:
    """Class representing group of masks and their properties.

    Useful for grouping mask created from clustering operations.

    """

    def __init__(
        self,
        elem_ids: list[str] | None = None,
        name: str = "Multimask",
        visible: bool = True,
        color: str = "tab10",
        spectrum_agg: str = "mean",
    ):
        """
        Parameters
        ----------
        elem_ids : list[str]
            list of ids (keys) of the masks part of the group.
        name: str
            Name of the group of masks.
        visible: bool
            Visibility attribute for the entire group of masks.
        color: str
            Name of the matplotlib color map. It will be used to define the colors of each mask in the group.
        spectrum_agg: str
            Default spectrum to show.
        """
        self.name = name
        self.elem_ids = elem_ids
        self.visible = visible
        self.set_color(color)
        self.spectrum_agg = spectrum_agg

    def set_color(self, color: str | None = "tab10"):
        if color in plt.colormaps():
            self.color = color
        else:
            self.color = "tab10"
