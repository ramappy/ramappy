from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from matplotlib import colormaps
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator
from pydantic_extra_types.color import Color

ColorType = Annotated[Color, PlainSerializer(lambda x: x.as_hex(format="long"), return_type=str)]
COLORMAPS_NAMES = StrEnum("COLORMAPS_NAMES", {cmap: cmap for cmap in colormaps})


class DataRules(BaseModel):
    mask: str | None = None
    roi_x_A: tuple[float, float] | None = None
    roi_x_B: tuple[float, float] | None = None
    agg: Literal["mean", "area", "max", "min", "fwhm"] = "mean"
    remove_baseline: bool | None = None

    model_config = ConfigDict(validate_assignment=True)

    def update(self, data: dict) -> DataRules:
        for k, v in data.items():
            setattr(self, k, v)
        return self


class VizRules(BaseModel):
    cmap: COLORMAPS_NAMES | ColorType = COLORMAPS_NAMES.turbo
    color2: ColorType | None = None
    invert: bool = False
    vmin: float | None = None
    vmax: float | None = None
    auto_levels: bool | None = None
    min: float | None = None
    max: float | None = None
    alpha: float | None = Field(ge=0, le=1, default=None)

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("color2", mode="before")
    @classmethod
    def validate_color2(cls, v):
        if v == "k":
            return Color("black")
        return v

    def update(self, data: dict) -> VizRules:
        """Update VizRules with new data."""
        for k, v in data.items():
            setattr(self, k, v)
        return self
