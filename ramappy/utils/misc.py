"""Miscellaneous utilities."""

import random
import uuid

import matplotlib

# Precompute the tab10 palette once at import time (constant, never changes).
_TAB10_HEX: tuple[str, ...] = tuple(
    f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    for r, g, b, *_ in matplotlib.colormaps["tab10"].colors  # type: ignore
)
_N_TAB10 = len(_TAB10_HEX)


def generate_key() -> str:
    """Generate a string to be used as a key for dictionaries.

    Returns
    -------
    str
        UUID string.
    """
    return str(uuid.uuid4())


def generate_random_pixel(width: int, height: int) -> int:
    """Generate a random pixel index.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.

    Returns
    -------
    int
        Random pixel index.
    """
    return random.randint(0, width * height - 1)


def color_generator(i: int | None = None) -> str:
    """Generate a HEX color choosing from matplotlib's tab10 colormap.

    Parameters
    ----------
    i : int, optional
        Index of the color, by default None.

    Returns
    -------
    str
        Hex color code.
    """
    if i is None:
        i = random.randrange(_N_TAB10)

    return _TAB10_HEX[i % _N_TAB10]
