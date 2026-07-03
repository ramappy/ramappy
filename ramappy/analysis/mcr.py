"""Multivariate Curve Resolution (MCR) analysis step.

The step decomposes spectra into concentration maps and component spectra using
iterative MCR-ALS style optimization.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from pymcr.constraints import ConstraintNonneg, ConstraintNorm, ConstraintZeroEndPoints
from pymcr.mcr import McrAR
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge

from ramappy.analysis.common import MAX_IMG_TO_SHOW
from ramappy.core import SpectralMap, Spectrum
from ramappy.core.images2d import Image2D
from ramappy.core.masks import Mask
from ramappy.core.pipeline import (
    ParamsAcceptMask,
    ParamsAcceptRoIX,
    ParamsSetResultId,
    SingleModelParamsValidator,
    StepClass,
    StepParams,
    pipeline_step,
)
from ramappy.utils import color_generator, generate_key, rubberband

MCRConstraints = Literal["non-negative", "sum-to-1", "endpoints-to-0"]
MCRRegressors = Literal["OLS", "NNLS", "Ridge", "LASSO"]
MCRDType = Literal["float32", "float64"]


def mcr_constraint_map(constraint: MCRConstraints):
    """Map a constraint label to a `pymcr` constraint object.

    Parameters
    ----------
    constraint
        Human-readable constraint label.

    Returns
    -------
    object
        Instantiated `pymcr` constraint implementation.

    Raises
    ------
    ValueError
        If ``constraint`` is not recognized.
    """

    if constraint == "non-negative":
        return ConstraintNonneg()
    if constraint == "sum-to-1":
        return ConstraintNorm(copy=True)
    if constraint == "endpoints-to-0":
        return ConstraintZeroEndPoints()
    raise ValueError(f"Unknown constraint: {constraint}")


def mcr_regressor_map(regressor: MCRRegressors, kwargs: dict | None = None):
    """Resolve a regressor label into a concrete regression model.

    Parameters
    ----------
    regressor
        Regressor selection (`"OLS"`, `"NNLS"`, `"Ridge"`, `"LASSO"`).
    kwargs
        Optional keyword arguments used to configure compatible models.

    Returns
    -------
    object
        Regressor object accepted by ``pymcr.McrAR``.
    """

    kwargs = {} if kwargs is None else dict(kwargs)

    if regressor == "Ridge":
        positive = kwargs.pop("positive", False)
        alpha = kwargs.pop("alpha", 1)
        return Ridge(alpha=alpha, positive=positive, random_state=13)
    if regressor == "LASSO":
        positive = kwargs.pop("positive", False)
        alpha = kwargs.pop("alpha", 1)
        return Lasso(alpha=alpha, positive=positive, random_state=13, **kwargs)
    return regressor


class StepMCRParams(StepParams, ParamsAcceptMask, ParamsAcceptRoIX, ParamsSetResultId):
    """Parameters for the MCR step."""

    initial_estimate_method: Literal["SVD", "PCA", "reference_spectra"] | None = None
    """Strategy used to initialize component spectra. If `None`, the method is automatically selected based on the presence of `reference_spectra`."""

    reference_spectra: list[str] | None = None
    """Optional list of reference spectra (ids) used when ``initial_estimate_method='reference_spectra'``."""

    n_components: int | float | None = 0.9
    """Number of components (or explained-variance target for PCA init). Values ``<= 0`` disable PCA."""

    phantom_baseline_endmember: bool = False
    """Whether to append a baseline-like endmember for reference initialization."""

    concentration_regressor: MCRRegressors = "OLS"
    """Regressor used for concentration optimization step."""

    spectra_regressor: MCRRegressors = "OLS"
    """Regressor used for spectra optimization step."""

    concentration_regressor_kwargs: dict | None = None
    """Optional model-specific kwargs passed to concentration regressor constructor."""

    spectra_regressor_kwargs: dict | None = None
    """Optional model-specific kwargs passed to spectra regressor constructor."""

    concentration_constraints: list[MCRConstraints] | None = None
    """Optional list of constraints for the concentration optimization block."""

    spectra_constraints: list[MCRConstraints] | None = None
    """Optional list of constraints for the spectra optimization block."""

    work_dtype: MCRDType = "float32"
    """Working precision used for internal matrix operations (`"float64"` or `"float32"`)."""


@pipeline_step(
    step_name="mcr",
    params_validator=SingleModelParamsValidator(StepMCRParams),
    friendly_name="MCR",
    step_category=StepClass.ANALYSIS,
)
def mcr(
    spectral_map: SpectralMap,
    *,
    mask: str | None = None,
    roi_x: npt.ArrayLike | None = None,
    initial_estimate_method: Literal["SVD", "PCA", "reference_spectra"] | None = None,
    reference_spectra: list[Spectrum | str] | None = None,
    n_components: float | None = 0.9,
    phantom_baseline_endmember: bool = False,
    concentration_regressor: MCRRegressors = "OLS",
    spectra_regressor: MCRRegressors = "OLS",
    concentration_regressor_kwargs: dict | None = None,
    spectra_regressor_kwargs: dict | None = None,
    concentration_constraints: list[MCRConstraints] | None = None,
    spectra_constraints: list[MCRConstraints] | None = None,
    work_dtype: MCRDType = "float32",
    res_id: str | None = None,
):
    """Run MCR decomposition and persist outputs in the map.

    Parameters
    ----------
    spectral_map
        Input map on which decomposition is performed.
    mask
        Optional spatial mask identifier restricting included pixels.
    roi_x
        Optional spectral ROI used during decomposition.
    initial_estimate_method
        Strategy used to initialize component spectra.
    reference_spectra
        Optional list of reference spectra (objects or ids) used when
        ``initial_estimate_method='reference_spectra'``.
    n_components
        Number of components (or explained-variance target for PCA init).
    phantom_baseline_endmember
        Whether to append a baseline-like endmember for reference initialization.
    concentration_regressor, spectra_regressor
        Regressors used for concentration and spectra optimization steps.
    concentration_regressor_kwargs, spectra_regressor_kwargs
        Optional model-specific kwargs passed to regressor constructors.
    concentration_constraints, spectra_constraints
        Optional lists of constraints for each optimization block.
    work_dtype
        Working precision used for internal matrix operations (`"float64"` or
        `"float32"`).
    res_id
        Optional explicit id used for generated output group names.

    Raises
    ------
    ValueError
        If incompatible initialization inputs are provided.
    RuntimeError
        If optimization finishes without producing spectra.
    """

    if initial_estimate_method is None:
        initial_estimate_method = "SVD" if reference_spectra is None else "reference_spectra"
    elif initial_estimate_method != "reference_spectra" and reference_spectra is not None:
        raise ValueError(
            f"{initial_estimate_method} implies automatically set initial components, but `reference_spectra` was passed.",
        )
    elif initial_estimate_method == "reference_spectra":
        if reference_spectra is None or len(reference_spectra) == 0:
            raise ValueError(
                "`reference_spectra` must contain at least one spectrum when using reference_spectra initialization"
            )

        resolved_spectra: list[Spectrum] = [
            spectral_map.get_reference_spectrum(rs)[0] if isinstance(rs, str) else rs for rs in reference_spectra
        ]
        allow_multiple_spectra = len(resolved_spectra) == 1 and resolved_spectra[0].data.shape[0] < 10
        if not allow_multiple_spectra:
            resolved_spectra = [
                Spectrum(
                    x=rs.x,
                    data=rs.data.mean(axis=0, keepdims=True),
                    x_axis_unit=rs.x_axis_unit,
                    data_unit=rs.data_unit,
                    roi_x=rs.roi_x,
                )
                if rs.data.shape[0] > 1
                else rs
                for rs in resolved_spectra
            ]

        common_spc = spectral_map.align_external_spectrum(
            resolved_spectra[0],
            overlap=True,
            allow_multiple_spectra=allow_multiple_spectra,
        )
        for rs in resolved_spectra[1:]:
            common_spc = common_spc.align_external_spectrum(rs, overlap=True)

        initial_spectra = np.vstack(
            [common_spc.align_external_spectrum(rs, overlap=True).data for rs in resolved_spectra]
        )

        if roi_x is not None:
            warnings.warn(
                "A ROI was provided while using external reference spectra, resetting it to a ROI common to the spectra",
                stacklevel=2,
            )
        roi_x = common_spc.roi_x

    idxs, x_idx, pixel_idx = spectral_map.get_indices(mask, roi_x)
    work_np_dtype = np.float64 if work_dtype == "float64" else np.float32

    # Make the main work array contiguous: this improves performance for repeated linear solves
    # inside pymcr/sklearn, and avoids repeated internal conversions.
    intensities = np.asarray(np.asarray(spectral_map.data)[idxs], dtype=work_np_dtype, order="C")
    x_roi = spectral_map.x[x_idx]

    if concentration_constraints is None:
        concentration_constraints = []
    if spectra_constraints is None:
        spectra_constraints = []

    mcrar = McrAR(
        c_regr=mcr_regressor_map(concentration_regressor, concentration_regressor_kwargs),
        st_regr=mcr_regressor_map(spectra_regressor, spectra_regressor_kwargs),
        c_constraints=[mcr_constraint_map(c) for c in concentration_constraints],
        st_constraints=[mcr_constraint_map(c) for c in spectra_constraints],
    )

    if initial_estimate_method == "SVD":
        if not isinstance(n_components, int):
            raise ValueError("SVD initial guess needs an integer number of components")
        max_rank = min(intensities.shape)
        if not 1 <= n_components < max_rank:
            raise ValueError(f"SVD initial guess requires 1 <= n_components < {max_rank}, got {n_components}")
        _, _, vh = svds(intensities, k=n_components)
        initial_spectra = np.abs(vh)
        denom = float(initial_spectra.max())
        if denom > 0:
            initial_spectra = (initial_spectra / denom) * float(intensities.max())
    elif initial_estimate_method == "PCA":
        pca = PCA(n_components=n_components, random_state=0)
        initial_spectra = pca.fit_transform(intensities.T).T
    elif initial_estimate_method == "reference_spectra" and phantom_baseline_endmember:
        baseline = rubberband(intensities.mean(axis=0), x=x_roi)
        initial_spectra = np.vstack([initial_spectra, baseline])

    initial_spectra = np.asarray(initial_spectra, dtype=work_np_dtype, order="C")

    scores = mcrar.fit_transform(intensities, ST=initial_spectra)
    if mcrar.ST_opt_ is None:
        raise RuntimeError("MCR did not produce optimized spectra (ST_opt_ is None)")
    spectra = mcrar.ST_opt_.T

    group_key = res_id or generate_key()
    elem_ids: list[str] = []

    mask_2d = None
    if not isinstance(pixel_idx, slice):
        mask_obj: Mask | None
        mask_obj = spectral_map.get_mask(mask) if isinstance(mask, str) else None
        if mask_obj is None:
            raise ValueError("Mask indices were provided, but mask object could not be resolved")
        mask_2d = mask_obj.get_2Dmask(invert=True)

    data_unit = spectral_map.data_unit

    for i in reversed(range(spectra.shape[1])):
        if mask_2d is not None:
            data = np.full(spectral_map.map_shape, np.nan, dtype=scores.dtype)
            data.ravel()[pixel_idx] = scores[:, i]
            img = np.ma.masked_array(data, mask=mask_2d, fill_value=np.nan, copy=False)
        else:
            img = scores[:, i].reshape(spectral_map.map_shape)
        img_key = f"{group_key}_{i}"
        spectral_map.images[img_key] = Image2D(
            data=img,
            parent_group=group_key,
            data_rules=None,
            viz_rules={"cmap": color_generator(i), "color2": "k"},
            name=f"MCR {i + 1}",
            spectrum=Spectrum(
                data=spectra[:, i],
                x=x_roi,
                roi_x=roi_x,
                x_axis_unit=spectral_map.x_axis_unit,
                data_unit=data_unit,
                ignore_sort=True,
                name=f"MCR{i + 1}",
            ),
            metadata={"initial_estimate_method": initial_estimate_method},
            locked=True,
            visible=i < MAX_IMG_TO_SHOW,
        )
        elem_ids.append(img_key)

    spectral_map.images_group[group_key] = spectral_map.new_images_group(elem_ids, name="MCR")
