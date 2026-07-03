# Analysis Algorithms

`ramappy.analysis` provides multivariate and statistical analysis algorithms that operate on `SpectralMap` objects. Each algorithm is also exposed as a pipeline step.

For the full auto-generated API reference see {py:mod}`ramappy.analysis`.

```{seealso}
Each algorithm below has a corresponding pipeline step. See [Pipeline Steps](steps.md) for step names, parameter models, and YAML configuration examples.
```

---

## Decomposition

### Principal Component Analysis (PCA)

{py:func}`ramappy.analysis.pca.principal_components` reduces dimensionality by projecting the spectral data onto its principal components. Returns score images and loading spectra.

**Full reference:** {py:mod}`ramappy.analysis.pca`

### Multivariate Curve Resolution (MCR-ALS)

{py:func}`ramappy.analysis.mcr.mcr` decomposes a mixed spectral dataset into pure component spectra and their spatial abundance maps using alternating least-squares.

**Full reference:** {py:mod}`ramappy.analysis.mcr`

### N-FINDR Endmember Extraction

{py:func}`ramappy.analysis.nfindr.step.nfindr` implements the N-FINDR algorithm to identify spectrally pure endmember pixels directly from the data, without external references.

**Full reference:** {py:mod}`ramappy.analysis.nfindr`

---

## Clustering

### Spectral Clustering

{py:func}`ramappy.analysis.cluster.cluster` applies k-means or agglomerative hierarchical clustering to the spectral data matrix, grouping pixels by spectral similarity.

**Full reference:** {py:mod}`ramappy.analysis.cluster`

### Substrate / Background Identification

{py:func}`ramappy.analysis.substrate.substrate_extraction` identifies substrate and foreground regions through spectral clustering, producing a spatial mask and an aggregated substrate spectrum.

**Full reference:** {py:mod}`ramappy.analysis.substrate`

---

## Fitting

### Spectral Fitting

{py:func}`ramappy.analysis.spectral_fitting.spectral_fitting` fits a linear combination of reference spectra to each pixel using non-negative least squares, producing per-pixel abundance maps for each reference component.

**Full reference:** {py:mod}`ramappy.analysis.spectral_fitting`
