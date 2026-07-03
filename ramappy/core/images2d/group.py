from __future__ import annotations


class Image2DGroup:
    """Class representing a group of images and their properties.

    Useful for grouping images created from PCA/MCR/etc.
    """

    def __init__(
        self,
        elem_ids: list[str] | None = None,
        name: str = "Group",
        visible: bool | None = True,
    ):
        self.name = name
        self.elem_ids = elem_ids
        self.visible = visible
