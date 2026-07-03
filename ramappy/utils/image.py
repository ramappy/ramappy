"""Image utilities."""

import base64
import functools
from io import BytesIO

import numpy as np
import PIL.Image
from matplotlib.colors import ColorConverter, LinearSegmentedColormap

from ramappy import const


def create_cmap(color_end: str | tuple, color_start: str | tuple | None = None) -> LinearSegmentedColormap:
    """Single/dual-color linear colormap, from alpha/color_start to color_end.

    Parameters
    ----------
    color_end : str or tuple
        End color.
    color_start : str or tuple, optional
        Start color, by default None.

    Returns
    -------
    LinearSegmentedColormap
        Colormap.
    """
    if isinstance(color_end, str):
        color_end = ColorConverter.to_rgb(color_end)

    if color_start is None:
        color_start = (*color_end, 0)
    elif isinstance(color_start, str):
        color_start = ColorConverter.to_rgb(color_start)

    return LinearSegmentedColormap.from_list("linear_cmap", [color_start, color_end])


def blend_images(
    imgs: list[np.ndarray], mode: const.BlendModes | str = const.BlendModes.screen
) -> PIL.Image.Image | None:
    """Blend multiple images.

    Parameters
    ----------
    imgs : list[np.ndarray]
        List of images as numpy arrays.
    mode : const.BlendModes or str, optional
        Blending mode, by default const.BlendModes.screen.

    Returns
    -------
    PIL.Image.Image or None
        Blended image.
    """
    # assumes all images are float arrays and of the same size
    # https://www.w3.org/TR/SVGCompositing/
    if len(imgs) == 0:
        return None
    if len(imgs) == 1:
        res = np.clip(imgs[0], 0.0, 1.0)
        return PIL.Image.fromarray((res * 255).astype(np.uint8)).convert("RGBA")

    mode_str = str(mode).lower()
    if mode_str == "screen":
        # Screen compositing: out = a + b - a*b  (applied iteratively for N layers)
        # Equivalent to 1 - ∏(1 - aᵢ), which is the W3C SVG "screen" blend.
        res = functools.reduce(lambda a, b: a + b - a * b, imgs)
    elif mode_str in {"plus", "add"}:
        res = np.sum(imgs, axis=0)
    elif mode_str == "multiply":
        res = np.prod(imgs, axis=0)
    else:
        raise NotImplementedError(f"Blend mode {mode} is not supported")

    res = np.clip(res, 0.0, 1.0)
    return PIL.Image.fromarray((res * 255).astype(np.uint8)).convert("RGBA")


def encode_image(image: PIL.Image.Image, format: str = "png", base64_encode: bool = True) -> bytes | BytesIO:
    """Encode an image.

    Parameters
    ----------
    image : PIL.Image.Image
        Image to encode.
    format : str, optional
        Image format, by default "png".
    base64_encode : bool, optional
        Whether to return base64 encoded bytes, by default True.

    Returns
    -------
    bytes or BytesIO
        Encoded image.
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    if base64_encode:
        return base64.b64encode(buffered.getvalue())
    return buffered


def decode_image(img: str) -> PIL.Image.Image:
    """Decode a base64-encoded image.

    Parameters
    ----------
    img : str
        Base64 encoded image string.

    Returns
    -------
    PIL.Image.Image
        Decoded image.
    """
    return PIL.Image.open(BytesIO(base64.b64decode(img)))
