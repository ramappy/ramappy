"""Utility functions."""

from .array import aggregate, atleast_2d, is_sorted, merge_arrays, minmax
from .image import blend_images, create_cmap, decode_image, encode_image
from .metrics import (
    fwhm,
    pearson_correlation_coefficient,
    r2_score,
    spectral_angle_mapper,
    spectral_information_divergence,
)
from .misc import color_generator, generate_key, generate_random_pixel
from .roi import clean_roi, common_roi_overlap, find_nearest_x, find_nearest_x_many, get_x_regions, select_x_indices
from .smoothing import diff, diffmat, divdiffmat, rubberband, rubberband_batch, whittaker_smooth

__all__ = [
    "aggregate",
    "atleast_2d",
    "blend_images",
    "clean_roi",
    "color_generator",
    "common_roi_overlap",
    "create_cmap",
    "decode_image",
    "diff",
    "diffmat",
    "divdiffmat",
    "encode_image",
    "find_nearest_x",
    "find_nearest_x_many",
    "fwhm",
    "generate_key",
    "generate_random_pixel",
    "get_x_regions",
    "is_sorted",
    "merge_arrays",
    "minmax",
    "pearson_correlation_coefficient",
    "r2_score",
    "rubberband",
    "rubberband_batch",
    "select_x_indices",
    "spectral_angle_mapper",
    "spectral_information_divergence",
    "whittaker_smooth",
]
