"""Baseline-correction processing step and helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pybaselines.api import Baseline
from pydantic import ConfigDict, Field

from ramappy.core import SpectralMap, Spectrum
from ramappy.core._spectral_map.functional import _PARALLEL_MIN_ITEMS
from ramappy.core.pipeline import (
    ParamsAcceptMask,
    ParamsAcceptRoIX,
    ParamsParallelProcessing,
    ParamsSeparateRegions,
    StepClass,
    StepParams,
    UnionParamsValidator,
    pipeline_step,
)
from ramappy.utils import rubberband, rubberband_batch

_PYBASELINES_BASELINE_CACHE_MAXSIZE = 16
_PYBASELINES_BASELINE_CACHE: OrderedDict[tuple, Baseline] = OrderedDict()


def _x_cache_key(x: np.ndarray) -> tuple:
    """Create a safe cache key for an x-axis view.

    We purposely key by the underlying buffer pointer + shape/strides/dtype so that we
    *only* reuse a cached Baseline instance when `x` points to the exact same memory
    region. This avoids correctness issues from approximate hashing of x values.
    """
    x = np.asarray(x)
    ptr = int(x.__array_interface__["data"][0])
    return (ptr, x.shape, x.strides, x.dtype.str)


def _get_cached_baseline_fitter(x: np.ndarray) -> Baseline:
    """Return a cached `pybaselines.api.Baseline` instance for the given x-axis."""
    key = _x_cache_key(x)
    fitter = _PYBASELINES_BASELINE_CACHE.get(key)
    if fitter is not None:
        _PYBASELINES_BASELINE_CACHE.move_to_end(key)
        return fitter

    fitter = Baseline(x_data=np.asarray(x), assume_sorted=True)
    _PYBASELINES_BASELINE_CACHE[key] = fitter
    _PYBASELINES_BASELINE_CACHE.move_to_end(key)
    if len(_PYBASELINES_BASELINE_CACHE) > _PYBASELINES_BASELINE_CACHE_MAXSIZE:
        _PYBASELINES_BASELINE_CACHE.popitem(last=False)
    return fitter


class StepCorrectBaselineParams(
    StepParams, ParamsAcceptMask, ParamsAcceptRoIX, ParamsSeparateRegions, ParamsParallelProcessing
):
    """Base parameter model for baseline correction variants."""

    force_nonnegative: bool = False
    """Whether to force the baseline-corrected spectra to be non-negative."""

    model_config = ConfigDict(extra="allow")


class StepCorrectBaselineParamsPLS(StepCorrectBaselineParams):
    """Parameters for penalized least-squares baseline methods."""

    method: Literal["arpls", "iarpls", "aspls", "airpls"] = "arpls"
    """Baseline correction method to use."""

    lambda_: float | None = None
    """Smoothing parameter for the baseline correction. If None, the method will choose a default value. Higher values result in smoother baselines. For pybaselines PLS methods, ``lambda_`` is mapped to pybaselines' ``lam``."""


class StepCorrectBaselineParamsPLSEta(StepCorrectBaselineParams):
    """Parameters for DRPLS baseline correction."""

    method: Literal["drpls"] = "drpls"
    """Baseline correction method to use."""

    lambda_: float | None = None
    """Smoothing parameter for the baseline correction. If None, the method will choose a default value. Higher values result in smoother baselines. For pybaselines PLS methods, ``lambda_`` is mapped to pybaselines' ``lam``."""

    eta: float | None = Field(default=None, ge=0)
    """Parameter for the DRPLS method."""


class StepCorrectBaselineParamsRubberband(StepCorrectBaselineParams):
    """Parameters for rubberband baseline correction."""

    method: Literal["rubberband"] = "rubberband"
    """Baseline correction method to use."""

    kind: Literal["linear", "slinear"] = "slinear"
    """Interpolation kind for the rubberband baseline correction. See `scipy.interpolate.interp1d` for details."""

    max_points: int | None = None
    """Maximum number of points to use for the rubberband baseline correction. If None, all points are used."""


class StepCorrectBaselineParamsFree(StepCorrectBaselineParams):
    """Parameters for parameter-free baseline methods."""

    method: Literal["beads", "snip"] = "snip"
    """Baseline correction method to use."""


class StepCorrectBaselineParamsPoly(StepCorrectBaselineParams):
    """Parameters for polynomial-based baseline methods."""

    method: Literal["poly", "imodpoly", "goldindec"] = "poly"
    """Baseline correction method to use."""

    poly_order: int | None = Field(ge=1)
    """Polynomial order for the baseline correction. If None, the method will choose a default value. Higher values result in more flexible baselines."""


@pipeline_step(
    step_name="correct_baseline",
    params_validator=UnionParamsValidator(
        Annotated[
            StepCorrectBaselineParamsPLS
            | StepCorrectBaselineParamsPLSEta
            | StepCorrectBaselineParamsFree
            | StepCorrectBaselineParamsRubberband
            | StepCorrectBaselineParamsPoly,
            Field(discriminator="method"),
        ]
    ),
    friendly_name="Correct Baseline",
    step_category=StepClass.PROCESSING,
    supports_preview=True,
    show_difference="baseline",
)
def correct_baseline(
    spectral_map: SpectralMap | Spectrum,
    method: str = "rubberband",
    mask: str | None = None,
    roi_x: npt.ArrayLike | None = None,
    force_nonnegative: bool = False,
    n_jobs: int = -1,
    **kwargs,
):
    """Perform a baseline correction of each spectrum.

    Parameters
    ----------
    spectral_map : SpectralMap or Spectrum
        Data container modified in place.
    method : {'arpls', 'drpls', 'iarpls', 'aspls', 'airpls', 'poly', 'imodpoly', 'goldindec', 'beads', 'snip', 'rubberband'}, default='rubberband'
        Method for the baseline correction. See the pybaselines documentation for details on each method (excluding 'rubberband').
    mask : str or None, default=None
        Optional spatial mask identifier.
    roi_x : array_like or None, default=None
        Restrict baseline correction to these spectral regions.
    force_nonnegative : bool, default=False
        Whether to force the baseline-corrected spectra to be non-negative.
    n_jobs : int, default=-1
        Number of CPU cores to use for parallel processing.
    **kwargs : dict, optional
        Method-specific options. For pybaselines PLS methods, ``lambda_`` is
        mapped to pybaselines' ``lam``.

    Raises
    ------
    ValueError
        If the method is not recognized.
    """
    idxs, x_idx, _pixel_idx = spectral_map.get_indices(mask, roi_x)
    intensities = spectral_map.data[idxs]

    x = spectral_map.x[x_idx]

    kwargs.pop("separate_regions", None)

    if method in {"arpls", "drpls", "iarpls", "aspls", "airpls", "poly", "imodpoly", "goldindec", "beads", "snip"}:
        if method in {"arpls", "drpls", "iarpls", "aspls", "airpls"}:
            kwargs["lam"] = kwargs.pop("lambda_", None)

        n_items = intensities.shape[0] if intensities.ndim > 1 else 1
        # `apply_func` itself falls back to a serial loop for small workloads (joblib
        # dispatch overhead dominates below `_PARALLEL_MIN_ITEMS`), so only bypass the
        # cached fitter when the call will actually run concurrently across threads/processes.
        if n_jobs != 1 and n_items >= _PARALLEL_MIN_ITEMS:
            # Each parallel call needs its own Baseline instance (not thread-safe to share).
            def _fit_pls(d: np.ndarray) -> np.ndarray:
                return getattr(Baseline(x_data=x, assume_sorted=True), method)(d, **kwargs)[0]

            baseline = spectral_map.apply_func(_fit_pls, data=intensities, by="pixel", parallel=True)
        else:
            baseline_fitter = _get_cached_baseline_fitter(x)
            func = getattr(baseline_fitter, method)
            baseline = spectral_map.apply_func(lambda d: func(d, **kwargs)[0], data=intensities, by="pixel")
    elif method == "rubberband":
        kind = kwargs.get("kind", "slinear")
        max_points = kwargs.get("max_points")

        if max_points is None and kind in {"linear", "slinear"} and intensities.ndim == 2:
            baseline = rubberband_batch(intensities, x)
        else:
            baseline = spectral_map.apply_func(
                rubberband, data=intensities, by="pixel", x=x, parallel=(n_jobs != 1), **kwargs
            )
    else:
        raise ValueError(f"Method {method} not recognized")

    # If idxs uses advanced indexing (e.g., masks or non-slice ROI), `arr[idxs] -= baseline`
    # triggers a read/modify/write temporary. `np.subtract.at` performs an in-place scatter
    # update and avoids that temporary.
    if isinstance(idxs, tuple) and not (isinstance(idxs[0], slice) and isinstance(idxs[1], slice)):
        np.subtract.at(spectral_map.data, idxs, baseline)
    else:
        spectral_map.data[idxs] -= baseline

    # Force non-negative values: shift the entire image by a single global offset so
    # that the most-negative value becomes zero.
    if method != "rubberband" and force_nonnegative:
        global_min = spectral_map.data[idxs].min()
        if global_min < 0:
            if isinstance(idxs, tuple) and not (isinstance(idxs[0], slice) and isinstance(idxs[1], slice)):
                np.subtract.at(spectral_map.data, idxs, global_min)
            else:
                spectral_map.data[idxs] -= global_min
