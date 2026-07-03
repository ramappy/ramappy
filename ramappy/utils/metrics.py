"""Spectral metrics."""

import numpy as np
from scipy.signal import find_peaks, peak_widths


def spectral_angle_mapper(intensities: np.ndarray, reference_spectrum: np.ndarray, centre: bool = False) -> np.ndarray:
    """Spectral angle between each row of *intensities* and each row of *reference_spectrum*.

    The spectral angle :math:`\\theta` between a pixel spectrum :math:`\\mathbf{x}`
    and a reference spectrum :math:`\\mathbf{y}` is defined as:

    .. math::

        \\theta = \\arccos\\left( \\frac{\\mathbf{x} \\cdot \\mathbf{y}}{\\|\\mathbf{x}\\| \\|\\mathbf{y}\\|} \\right)

    If ``centre=True``, the mean intensity of each spectrum is subtracted from
    it prior to calculating the angle (mean-centred SAM).

    Parameters
    ----------
    intensities : np.ndarray of shape (N, W)
        Pixel or sample spectra.
    reference_spectrum : np.ndarray of shape (M, W)
        Reference spectra.
    centre : bool, optional
        Subtract the row mean before computing angles (mean-centred SAM).
        Default is ``False``.

    Returns
    -------
    np.ndarray of shape (N, M)
        Spectral angle in radians for each pixel-reference pair.
    """
    if centre:
        # mean 0
        intensities -= intensities.mean(axis=1, keepdims=True)
        reference_spectrum -= reference_spectrum.mean(axis=1, keepdims=True)

    dot_product = np.dot(intensities, reference_spectrum.T)
    norm_i = np.linalg.norm(intensities, axis=1, keepdims=True)  # (N, 1)
    norm_r = np.linalg.norm(reference_spectrum, axis=1, keepdims=True)  # (M, 1)
    # Guard against zero-norm spectra (e.g., all-zero background pixels)
    cosine_similarity = dot_product / np.maximum(norm_i * norm_r.T, np.spacing(1))
    cosine_similarity = np.clip(cosine_similarity, -1, 1)
    return np.arccos(cosine_similarity)


def pearson_correlation_coefficient(intensities: np.ndarray, reference_spectrum: np.ndarray) -> np.ndarray:
    """Pearson correlation between each row of *intensities* and each row of *reference_spectrum*.

    The Pearson correlation coefficient :math:`r` between a pixel spectrum :math:`\\mathbf{x}`
    and a reference spectrum :math:`\\mathbf{y}` is defined as:

    .. math::

        r = \\frac{(\\mathbf{x} - \\bar{\\mathbf{x}}) \\cdot (\\mathbf{y} - \\bar{\\mathbf{y}})}{\\|\\mathbf{x} - \\bar{\\mathbf{x}}\\| \\|\\mathbf{y} - \\bar{\\mathbf{y}}\\|}

    where :math:`\\bar{\\mathbf{x}}` and :math:`\\bar{\\mathbf{y}}` are the means of the spectra.

    Parameters
    ----------
    intensities : np.ndarray of shape (N, W)
        Pixel or sample spectra.
    reference_spectrum : np.ndarray of shape (M, W)
        Reference spectra.

    Returns
    -------
    np.ndarray of shape (N, M)
        Pearson correlation coefficient for each pixel-reference pair.
    """
    intensities = intensities - intensities.mean(axis=1, keepdims=True)
    reference_spectrum = reference_spectrum - reference_spectrum.mean(axis=1, keepdims=True)

    dot_product = np.dot(intensities, reference_spectrum.T)
    norm_i = np.linalg.norm(intensities, axis=1, keepdims=True)  # (N, 1)
    norm_r = np.linalg.norm(reference_spectrum, axis=1, keepdims=True)  # (M, 1)

    return dot_product / np.maximum(norm_i * norm_r.T, np.spacing(1))


def spectral_information_divergence(intensities: np.ndarray, reference_spectrum: np.ndarray) -> np.ndarray:
    """
    Computes the spectral information divergence between two vectors [Chang2000]_.

    For each pair of pixel spectrum ``p_i`` and reference spectrum ``q_j`` the
    SID is defined as the symmetric KL divergence between the spectral
    probability distributions:

    .. math::

        \\mathrm{SID}(p \\| q) = D_{KL}(p \\| q) + D_{KL}(q \\| p)

    where each spectrum is treated as an unnormalised probability vector (all
    values non-negative). A small positive constant is added *before*
    normalisation to avoid :math:`\\log(0)` and to keep the distributions
    strictly positive.

    Reference
    ---------
    .. [Chang2000] C.-I. Chang, "An Information-Theoretic Approach to Spectral
           Variability, Similarity, and Discrimination for Hyperspectral
           Image", IEEE TRANSACTIONS ON INFORMATION THEORY, VOL. 46, NO. 5,
           AUGUST 2000.

    """
    eps = np.spacing(1)
    # Add eps before normalising to keep distributions strictly positive
    p = (intensities + eps) / np.sum(intensities + eps, axis=1, keepdims=True)  # (N, W)
    q = (reference_spectrum + eps) / np.sum(reference_spectrum + eps, axis=1, keepdims=True)  # (M, W)

    log_p = np.log(p)  # (N, W)
    log_q = np.log(q)  # (M, W)

    H_p = np.sum(p * log_p, axis=1, keepdims=True)  # (N, 1)
    kl_pq = H_p - p @ log_q.T  # (N, M)

    H_q = np.sum(q * log_q, axis=1, keepdims=True)  # (M, 1)
    kl_qp = H_q.T - (q @ log_p.T).T  # (N, M)

    return kl_pq + kl_qp  # (N, M)


def r2_score(intensities: np.ndarray, reference_spectrum: np.ndarray) -> np.ndarray:
    """Computes the Coefficient of Determination (R² or r²).

    For each pixel spectrum in ``intensities`` (shape ``(N, W)``) and each
    reference in ``reference_spectrum`` (shape ``(M, W)``), returns the R²
    score measuring how well the pixel resembles the reference:

    .. math::

        R^2_{ij} = 1 - \\frac{\\sum_k (r_{jk} - s_{ik})^2}
                              {\\sum_k (r_{jk} - \\bar{r}_j)^2}

    where :math:`r_j` is the j-th reference spectrum and :math:`s_i` is the
    i-th pixel spectrum. A score of 1 indicates a perfect match; lower values
    indicate poorer agreement.

    Parameters
    ----------
    intensities : np.ndarray of shape (N, W)
        Pixel spectra to evaluate.
    reference_spectrum : np.ndarray of shape (M, W)
        Reference spectra.

    Returns
    -------
    np.ndarray of shape (N, M)
        R² scores.
    """
    # Per-reference total variance: (M,)
    ref_mean = reference_spectrum.mean(axis=1, keepdims=True)  # (M, 1)
    ss_total = np.sum((reference_spectrum - ref_mean) ** 2, axis=1)  # (M,)

    # Per-pixel x per-reference residual sum of squares: (N, M)
    # ||a - b||^2 = ||a||^2 - 2 a·b^T + ||b||^2
    ss_residuals = (
        np.sum(intensities**2, axis=1, keepdims=True)  # (N, 1)
        - 2.0 * intensities @ reference_spectrum.T  # (N, M)
        + np.sum(reference_spectrum**2, axis=1, keepdims=True).T  # (1, M)
    )  # (N, M)

    return 1.0 - ss_residuals / np.maximum(ss_total, np.spacing(1))  # (N, M)


def fwhm(y: np.ndarray, peak: str | int = "max") -> float:
    """Full width at half maximum of a peak in *y*.

    Parameters
    ----------
    y : np.ndarray
        1-D spectrum.
    peak : str or int, optional
        Which peak to measure. ``"max"`` (default) uses the tallest peak;
        an integer selects the peak at that index.

    Returns
    -------
    float
        FWHM in sample units (index-space).
    """
    if isinstance(peak, int):
        highest_peak = peak
    else:
        # "max": find local peaks and pick the tallest; fall back to argmax if
        # no local peak is detected (e.g., monotonic or flat signal).
        peaks, _ = find_peaks(y)
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(y))])
        highest_peak = int(peaks[np.argmax(y[peaks])])
    fwhm = peak_widths(y, [highest_peak], rel_height=0.5)
    return fwhm[0][0]
