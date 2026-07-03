"""Pure 2-D image data container."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from PIL import Image

from ramappy.utils import minmax

from ..metadata import ImageMetadata
from ..spectrum import Spectrum
from .rules import DataRules, VizRules


class Image2D:
    """2-D image, pure data container.

    Stores raw float data, visualization rules, and metadata.
    **No rendering happens at construction time.**

    To produce a rendered `PIL.Image.Image` call
    `ramappy.core.images2d.Image2DRenderer.render`.

    Parameters
    ----------
    data : np.ndarray | None
        Raw float intensity data, shape ``(height, width)``.
    image : PIL.Image.Image | None
        Pre-existing raster image (for locked/imported images that have no
        underlying float data). At most one of *data* / *image* should be
        set.
    parent_group : str | None
        ID of the :class:`Image2DGroup <ramappy.core.images2d.group.Image2DGroup>` this image belongs to.
    data_rules : DataRules | dict | None
        Rules describing how the data was derived (aggregation, mask, ROI).
    viz_rules : VizRules | dict | None
        Rules describing how to colourise the float data.
    name : str
        Human-readable label.
    locked : bool
        If ``True`` the image is not recomputed when the parent
        :class:`SpectralMap <ramappy.core.spectral_map.SpectralMap>` is updated.
    visible : bool
        Whether the image should be included in compositing.
    spectrum : Spectrum | None
        A spectrum associated with this image.
    metadata : dict | None
        Arbitrary additional metadata.
    source_step_id : str | None
        ID of the :class:`ProcessingStepConfig <ramappy.core.processing_step.ProcessingStepConfig>` that
        generated this image, or ``None`` for locked/imported images.
    """

    _viz_rules: VizRules | None = None
    _data_rules: DataRules | None = None

    def __init__(
        self,
        data: np.ndarray | None = None,
        image: Image.Image | None = None,
        parent_group: str | None = None,
        data_rules: DataRules | dict | None = None,
        viz_rules: VizRules | dict | None = None,
        name: str = "",
        locked: bool = False,
        visible: bool = True,
        spectrum: Spectrum | None = None,
        metadata: ImageMetadata | dict[str, Any] | None = None,
        source_step_id: str | None = None,
    ) -> None:
        if data is not None and data.ndim > 2:
            raise ValueError(f"Expecting a flat 2-D image (single channel), got ndim={data.ndim}")

        self.data = data
        # _image is the pre-existing raster for locked/imported images.
        self._image: Image.Image | None = image
        self.name = name
        self.spectrum = spectrum
        self._image_metadata: ImageMetadata = ImageMetadata.from_any(copy.deepcopy(metadata))
        self.parent_group = parent_group
        self.locked = locked
        self.visible = visible
        self.source_step_id = source_step_id

        self.data_rules = data_rules
        self._viz_rules = viz_rules if isinstance(viz_rules, VizRules) else VizRules.model_validate(viz_rules or {})

    # Properties

    @property
    def width(self) -> int | None:
        """Width of the image in pixels (derived from data shape)."""
        return self.data.shape[1] if self.data is not None else (self._image.width if self._image else None)

    @property
    def height(self) -> int | None:
        """Height of the image in pixels (derived from data shape)."""
        return self.data.shape[0] if self.data is not None else (self._image.height if self._image else None)

    @property
    def viz_rules(self) -> VizRules | None:
        """Visualisation rules."""
        return self._viz_rules

    @viz_rules.setter
    def viz_rules(self, value: VizRules | dict | None) -> None:
        self._viz_rules = value if isinstance(value, VizRules) else VizRules.model_validate(value or {})

    @property
    def data_rules(self) -> DataRules | None:
        """Data derivation rules."""
        return self._data_rules

    @data_rules.setter
    def data_rules(self, value: DataRules | dict | None) -> None:
        self._data_rules = value if isinstance(value, DataRules) else DataRules.model_validate(value or {})

    @property
    def data_range(self) -> tuple[float, float] | None:
        """``(min, max)`` of the raw data, or ``None`` when no data is set."""
        if self.data is None:
            return None
        if not hasattr(self, "_data_range") or self._data_range is None:
            d = self.data.compressed() if isinstance(self.data, np.ma.MaskedArray) else self.data.ravel()
            if self.data.dtype == bool:
                self._data_range: tuple[float, float] = (0.0, 1.0)
            else:
                self._data_range = tuple(minmax(d))  # type: ignore[assignment]
        return self._data_range

    @property
    def metadata(self) -> dict[str, Any]:
        """Additional metadata fields (`extra`) as a mutable dict."""
        return self._image_metadata.extra

    @metadata.setter
    def metadata(self, value: ImageMetadata | dict[str, Any] | None) -> None:
        self._image_metadata = ImageMetadata.from_any(copy.deepcopy(value))

    @property
    def image_metadata(self) -> ImageMetadata:
        """Structured metadata model for this image layer."""
        return self._image_metadata
