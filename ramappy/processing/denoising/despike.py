"""Spike detection and optional correction processing step."""

import itertools
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field

from ramappy import const
from ramappy.core import SpectralMap
from ramappy.core.pipeline import (
    ParamsAcceptMask,
    ParamsAcceptRoIX,
    ParamsParallelProcessing,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)


def find_outliers(intensities, threshold=8.0, mad=False, axis: const.Axis = const.Axis.PIXEL) -> np.ndarray:
    """Detect which spectra contain spikes due to cosmic rays (if any).

    Two detection modes are available:

    **Z-score** (``mad=False``): a spectrum is flagged if any channel satisfies

    .. math::

        |y_i - \\bar{y}| > \\tau \\cdot \\sigma

    where :math:`\\bar{y}` and :math:`\\sigma` are the mean and standard
    deviation computed over the chosen *axis*.

    **MAD** (``mad=True``): a spectrum is flagged if any channel satisfies

    .. math::

        0.6745 \\cdot |y_i - \\tilde{y}| > \\tau \\cdot \\mathrm{MAD}

    where :math:`\\tilde{y}` is the median and :math:`\\mathrm{MAD}` is the
    median absolute deviation. The constant 0.6745 makes MAD a consistent
    estimator of :math:`\\sigma` under Gaussian noise.

    Parameters
    ----------
    intensities : array_like of shape (N, W)
        Intensity values to analyse for outliers.
    threshold : float, default=8.0
        Multiplier :math:`\\tau` for the outlier detection threshold.
    mad : bool, default=False
        If ``True``, use MAD-based detection; if ``False``, use Z-score.
    axis : const.Axis, default=const.Axis.PIXEL
        Axis along which to compute the reference statistics.
        ``const.Axis.PIXEL`` computes per-wavenumber statistics (recommended);
        ``const.Axis.SPECTRAL`` computes per-pixel statistics;
        ``None`` computes a single global statistic.

    Returns
    -------
    outliers : np.ndarray of int
        Flat indices (into the first axis of *intensities*) of pixels
        containing at least one spiked channel.
    """
    keepdims = axis is not None

    if mad:
        # Use robust statistics with MAD
        # Scale factor: Phi^-1(0.75) ≈ 0.67449, so MAD ≈ 0.67449 * sigma for normal data.
        scale = 0.67449
        median_int = np.median(intensities, axis=axis, keepdims=keepdims)
        median_err = np.abs(intensities - median_int)
        mad_threshold = threshold * np.median(median_err, axis=axis, keepdims=keepdims)
        spikes = scale * median_err > mad_threshold
    else:
        # Use Z-score based detection
        mean = np.mean(intensities, axis=axis, keepdims=keepdims)
        std = np.std(intensities, axis=axis, keepdims=keepdims)
        spikes = np.abs(intensities - mean) > threshold * std

    return np.flatnonzero(np.any(spikes, axis=const.Axis.SPECTRAL))


def _compute_single_median_joblib(data_slice: np.ndarray) -> np.ndarray:
    return np.median(data_slice, axis=0)


class StepDespikeParams(StepParams, ParamsAcceptMask, ParamsAcceptRoIX, ParamsParallelProcessing):
    """Parameters for the despike processing step."""

    method: Literal["normal", "mad"] = "normal"
    """Outlier detection method to use: 'normal' for Z-score, 'mad' for median absolute deviation."""

    threshold: float = Field(8, gt=0)
    """Threshold multiplier for outlier detection. Higher values result in fewer detected spikes."""

    first_diff: bool = True
    """If True, use the first discrete difference of the spectrum for spike detection; otherwise, use the raw spectrum."""

    axis: const.Axis = const.Axis.PIXEL
    """Axis along which to compute statistics for spike detection."""

    correct_spikes: bool = False
    """If True, attempt to correct detected spikes by replacing them with the median of neighboring pixels."""


@pipeline_step(
    step_name="despike",
    params_validator=SingleModelParamsValidator(StepDespikeParams),
    friendly_name="Detect and remove spikes",
    step_category=StepClass.PROCESSING,
    modifies_data=lambda params: params.correct_spikes,  # type: ignore
)
def despike(
    spectral_map: SpectralMap,
    *,
    method: str = "normal",
    threshold: float = 8.0,
    first_diff: bool = True,
    mask: str | None = None,
    roi_x: npt.ArrayLike | None = None,
    axis: const.Axis = const.Axis.PIXEL,
    correct_spikes: bool = False,
    n_jobs: int = -1,
):
    """Mark bad pixels (i.e., presenting spikes) that will be discarded from the following steps.

    Parameters
    ----------
    spectral_map : SpectralMap
        Hyperspectral image object to process.
    method : {"normal", "mad"}, default="normal"
        Outlier detection method:

        - "normal": Use Z-score based outlier detection, selecting pixels
          which have a Z-score exceeding threshold * std from the mean.
        - "mad": Similar to "normal", but use robust statistics for the
          central tendency (median) and dispersion (MAD).
    threshold : float, default=8.0
        Multiple of standard deviations/MAD to detect outliers.
        E.g., threshold=8 for "normal", threshold=15 for "mad".
    first_diff : bool, default=True
        Use the first discrete difference of the spectrum rather than the
        actual spectrum.
    mask : str or None, default=None
        Restrict search to this masked region in the 2-D spectral_map.
        If None, use the whole spectral_map.
    roi_x : array_like or None, default=None
        Restrict search to these regions in the Raman shift.
        If None, use the whole spectral axis.
        Example: roi_x=[[600,1800],[2800,3030]] or roi_x=[600,3030]
    axis : const.Axis, default=const.Axis.PIXEL
        Whether to compute statistics (mean, standard deviation, MAD) globally
        (None) or on a per-spectrum (const.Axis.SPECTRAL) or per-wavenumber
        (const.Axis.PIXEL) basis.
    correct_spikes : bool, default=False
        Try correcting any detected spikes by replacing with median of
        neighboring pixels.
    n_jobs : int, default=-1
        Number of CPU cores to use for parallel processing.
    """
    idxs, _, pixel_idx = spectral_map.get_indices(mask, roi_x)

    mad = method == "mad"  # if False, assume 'normal'

    data: np.ndarray = np.asarray(spectral_map.data)
    intensities = np.diff(data[idxs], axis=const.Axis.SPECTRAL) if first_diff else data[idxs]

    spikes = find_outliers(intensities, threshold, mad=mad, axis=axis)

    if not isinstance(pixel_idx, slice):
        # map spikes indices (relative to pixel_idx subset) to map indices
        spikes = pixel_idx[np.array(spikes)]

    if correct_spikes:
        # replace bad pixel spectra with the median of the neighbouring pixels
        iy, ix = np.unravel_index(spikes, (spectral_map.img_height, spectral_map.img_width))
        # get neighbours (3x3), don't remove spike itself (median can use spectral info not affected by spike)
        # mode='clip' in ravel_multi_index leaves dupes, manually use min/max
        ix_n = [range(max(0, i - 1), min(spectral_map.img_width, i + 2)) for i in ix]
        iy_n = [range(max(0, i - 1), min(spectral_map.img_height, i + 2)) for i in iy]

        n_idxs = [
            np.ravel_multi_index(
                np.array(list(itertools.product(y, x))).T.reshape(2, -1),
                dims=(spectral_map.img_height, spectral_map.img_width),
            )
            for (y, x) in zip(iy_n, ix_n, strict=False)
        ]

        data2: np.ndarray = np.asarray(spectral_map.data)
        neighbour_blocks = np.empty(len(n_idxs), dtype=object)
        for i, n_idx in enumerate(n_idxs):
            neighbour_blocks[i] = data2[n_idx, :]
        corrected = spectral_map.apply_func(
            _compute_single_median_joblib,
            data=neighbour_blocks,
            by="pixel",
            preserve_input_dtype=False,
            parallel=(n_jobs != 1),
        )
        np.asarray(spectral_map.data)[spikes, :] = corrected

    spectral_map.masks["spikes"] = spectral_map.new_mask(
        spikes, name="Spikes", editable=False, spectrum_agg=None, color="#FF00FF"
    )
