"""Shared helpers for analysis pipeline steps."""

from inspect import signature
from typing import Any

from sklearn.cluster import AgglomerativeClustering

MAX_IMG_TO_SHOW = 5


def _make_hierarchical_clustering(*, n_clusters: int) -> AgglomerativeClustering:
    """Build an AgglomerativeClustering estimator across sklearn API variants."""

    init_params: dict[str, Any] = {"n_clusters": n_clusters, "linkage": "ward"}
    estimator_params = signature(AgglomerativeClustering).parameters

    if "metric" in estimator_params:
        init_params["metric"] = "euclidean"
    else:
        init_params["affinity"] = "euclidean"

    return AgglomerativeClustering(**init_params)


def validate_pca_n_components(value):
    """Normalize/validate PCA `n_components` inputs.

    Parameters
    ----------
    value : int | float | None
        Candidate PCA component parameter.

    Returns
    -------
    int | float | None
        Normalized value preserving explained-variance ratios in ``(0, 1)``.

    Raises
    ------
    ValueError
        If ``value`` is negative.
    """

    if value is None:
        return None
    if value < 0:
        raise ValueError("Must be a non-negative number")
    if value >= 1:
        return int(value)
    return value
