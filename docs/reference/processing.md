# Processing Algorithms

`ramappy.processing` provides the signal-processing functions used by the built-in pipeline steps. These functions operate directly on `SpectralMap` objects and are also individually importable for use outside of a pipeline.

For the full auto-generated API reference see {py:mod}`ramappy.processing`.

```{seealso}
Each function below has a corresponding pipeline step. See [Pipeline Steps](steps.md) for step names, parameter models, and YAML configuration examples.
```

---

## Baseline Correction

{py:mod}`ramappy.processing.baseline` implements multiple baseline estimation strategies. The `correct_baseline` pipeline step dispatches to this module based on the `method` parameter.

Supported methods: `arpls`, `airpls`, `iarpls` (PLS-based), `poly` (polynomial), and free-form anchor-point interpolation.

**Full reference:** {py:mod}`ramappy.processing.baseline`

---

## Spectral Operations

### Cropping

{py:mod}`ramappy.processing.crop_spectral` truncates the spectral axis to one or more regions of interest.

**Full reference:** {py:mod}`ramappy.processing.crop_spectral`

### Resampling

{py:func}`ramappy.processing.resample.resample` resamples the spectral axis onto a target grid defined by a step size (`spectrum_step`), reference spectrum ID (`ref_spectrum_id`), or explicit grid (`x_grid`) using interpolation.

**Full reference:** {py:mod}`ramappy.processing.resample`

### Intensity Normalization

{py:func}`ramappy.processing.normalize.normalize_intensities` normalizes spectra by peak value, area, L1/L2 norm, or a reference wavenumber.

**Full reference:** {py:mod}`ramappy.processing.normalize`

---

## Denoising

### Cosmic Ray Removal

{py:func}`ramappy.processing.denoising.despike.despike` detects and corrects cosmic-ray spikes using robust outlier statistics across neighbouring pixels.

**Full reference:** {py:mod}`ramappy.processing.denoising.despike`

### Spectral Smoothing

{py:func}`ramappy.processing.denoising.smooth_spectral.smooth_spectral` applies Whittaker or Savitzky–Golay smoothers along the spectral axis.

**Full reference:** {py:mod}`ramappy.processing.denoising.smooth_spectral`

### Spatial Smoothing

{py:func}`ramappy.processing.denoising.smooth_spatial.smooth_spatial` applies median or Gaussian filters over the spatial dimensions.

**Full reference:** {py:mod}`ramappy.processing.denoising.smooth_spatial`

### SVD Denoising

{py:func}`ramappy.processing.denoising.svd.smooth_svd` denoises by truncating the singular value decomposition of the spectral data matrix, retaining only the most significant components.

**Full reference:** {py:mod}`ramappy.processing.denoising.svd`

---

## Geometric Transformations

{py:mod}`ramappy.processing.geometric` groups the spatial-crop, flip, and rotate operations that modify the map's spatial layout without altering spectral data.

**Full reference:** {py:mod}`ramappy.processing.geometric`
