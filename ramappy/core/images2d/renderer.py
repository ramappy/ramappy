"""Stateless rendering helpers for :class:`Image2D <ramappy.core.images2d.Image2D>` and :class:`Mask <ramappy.core.masks.Mask>`.

Example
-------
>>> from ramappy.core.images2d import Image2D, Image2DRenderer
>>> rendered: PIL.Image.Image = Image2DRenderer.render(img)
>>> colorbar_svg: bytes = Image2DRenderer.render_colorbar(img)
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from ramappy.core.masks import Mask

import numpy as np
from matplotlib import colorbar, colors, ticker
from matplotlib import pyplot as plt
from PIL import Image

from ramappy.utils import create_cmap, minmax

from .image import Image2D
from .rules import VizRules


class Image2DRenderer:
    """Stateless renderer for :class:`Image2D <ramappy.core.images2d.Image2D>` objects.

    All methods are static, there is no instance state.
    """

    @staticmethod
    def render(img: Image2D, viz_rules: VizRules | dict | None = None) -> Image.Image:
        """Render *img*'s float data through its (or the supplied) colour map.

        Parameters
        ----------
        img : Image2D
            The image to render.
        viz_rules : VizRules | dict | None, optional
            Override the image's stored ``viz_rules``. If ``None`` the image's
            own ``viz_rules`` are used.

        Returns
        -------
        PIL.Image.Image
            RGBA raster image.

        Raises
        ------
        ValueError
            If *img* has no float data (locked / raster-only image).
        """
        if img.data is None:
            if img._image is not None:
                return img._image
            raise ValueError("Image2D has no data and no stored raster image, cannot render.")

        vr = Image2DRenderer._resolve_vr(img, viz_rules)
        cmap = Image2DRenderer._make_cmap(vr)

        raw = img.data.compressed() if isinstance(img.data, np.ma.MaskedArray) else img.data  # type: ignore
        dr = img.data_range
        if dr is None:
            dr = minmax(raw.ravel())
        _min, _max = dr
        interval = _max - _min if _max != _min else 1.0

        if vr.auto_levels:
            p5, p95 = np.percentile(raw[~np.isnan(raw)] if raw.dtype != bool else raw, [5, 95])
            min_viz, max_viz = (p5 - _min) / interval, (p95 - _min) / interval
        else:
            min_viz, max_viz = vr.vmin, vr.vmax

        v_min = _min if min_viz is None else _min + min_viz * interval
        v_max = _max if max_viz is None else _min + max_viz * interval

        if vr.min is not None and vr.min < _max:
            v_min = vr.min
        if vr.max is not None and vr.max > _min:
            v_max = vr.max

        norm = colors.Normalize(vmin=v_min, vmax=v_max)
        rgba = cmap(norm(img.data), bytes=True)
        rendered = Image.fromarray(rgba)

        if vr.alpha is not None:
            rendered = Image2DRenderer._apply_alpha(rendered, float(vr.alpha))

        return rendered

    @staticmethod
    def render_colorbar(img: Image2D, viz_rules: VizRules | dict | None = None) -> bytes:
        """Render a vertical SVG colourbar for *img*.

        Parameters
        ----------
        img : Image2D
            Source image (data range and viz_rules are read from it).
        viz_rules : VizRules | dict | None, optional
            Override viz_rules.

        Returns
        -------
        bytes
            Raw SVG bytes.
        """
        if img.data is None or img.data.dtype == bool:
            return b""

        vr = Image2DRenderer._resolve_vr(img, viz_rules)
        cmap = Image2DRenderer._make_cmap(vr)

        dr = img.data_range
        if dr is None:
            return b""
        _min, _max = dr
        interval = _max - _min if _max != _min else 1.0

        min_viz, max_viz = vr.vmin, vr.vmax
        v_min = _min if min_viz is None else _min + min_viz * interval
        v_max = _max if max_viz is None else _min + max_viz * interval
        if vr.min is not None and vr.min < _max:
            v_min = vr.min
        if vr.max is not None and vr.max > _min:
            v_max = vr.max

        norm = colors.Normalize(vmin=v_min, vmax=v_max)
        n_ticks = 7

        def ticks_formatter(x, pos):
            sign = "<" if pos == 0 and v_min != _min else ">" if pos == n_ticks - 1 and v_max != _max else ""
            if abs(x) < 0.1 or abs(x) > 999:
                a, b = f"{x:.2e}".split("e")
                return rf"${sign}{a} \times 10^{{{int(b)}}}$"
            return rf"${sign}{round(x, 2)}$"

        fig = plt.figure(figsize=(1.5, 3))
        ax = fig.add_axes((0, 0.05, 0.13, 1))
        ticks = np.linspace(v_min, v_max, n_ticks, endpoint=True)
        colorbar.ColorbarBase(
            ax,
            cmap=cmap,
            norm=norm,
            orientation="vertical",
            ticks=list(ticks),
            format=ticker.FuncFormatter(ticks_formatter),
        )
        buf = BytesIO()
        fig.savefig(buf, bbox_inches="tight", transparent=True, format="svg")
        plt.close(fig)
        return buf.getvalue()

    @staticmethod
    def render_mask(mask: Mask) -> Image.Image:  # type: ignore[name-defined]
        """Render *mask* as a coloured RGBA raster image.

        Parameters
        ----------
        mask : Mask
            The mask to render.

        Returns
        -------
        PIL.Image.Image
            RGBA raster where selected pixels carry the mask's colour and
            unselected pixels are fully transparent.
        """

        img_height, img_width = mask.mask_shape
        rgba = np.zeros((img_height, img_width, 4), dtype=np.uint8)

        if mask.color is not None:
            r, g, b, *a = colors.to_rgba(mask.color)
            alpha = int((a[0] if a else 1.0) * 255)
            mask_2d = mask.get_2Dmask()
            rgba[mask_2d, 0] = int(r * 255)
            rgba[mask_2d, 1] = int(g * 255)
            rgba[mask_2d, 2] = int(b * 255)
            rgba[mask_2d, 3] = alpha

        return Image.fromarray(rgba, mode="RGBA")

    # Helpers

    @staticmethod
    def _resolve_vr(img: Image2D, viz_rules: VizRules | dict | None) -> VizRules:
        """Return the effective :class:`VizRules <ramappy.core.images2d.VizRules>` for *img*, applying any override."""
        if viz_rules is None:
            return img.viz_rules or VizRules()
        if isinstance(viz_rules, VizRules):
            return viz_rules
        return VizRules.model_validate(viz_rules)

    @staticmethod
    def _make_cmap(vr: VizRules):
        """Build a matplotlib colormap from *vr*."""
        color = vr.cmap
        color_end = color.as_hex(format="long") if hasattr(color, "as_hex") else color
        color_start = vr.color2.as_hex(format="long") if vr.color2 is not None else None

        if isinstance(color_end, str) and color_end in plt.colormaps():
            cmap = plt.get_cmap(color_end)
        else:
            cmap = create_cmap(color_start=color_start, color_end=color_end)
        if vr.invert:
            cmap = cmap.reversed()
        return cmap

    @staticmethod
    def _apply_alpha(image: Image.Image, alpha: float) -> Image.Image:
        image = image.convert("RGBA")
        channel_a = image.getchannel("A")
        channel_a = channel_a.point(lambda _: int(255 * alpha))
        image.putalpha(channel_a)
        return image
