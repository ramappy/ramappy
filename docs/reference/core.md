# Core Data Model

The `ramappy.core` package defines the primary data containers and abstractions used throughout the library. All IO readers, processing steps, and analysis algorithms operate on these types.

For the full auto-generated API reference see {py:mod}`ramappy.core`.

---

## SpectralMap

{py:class}`ramappy.core.spectral_map.SpectralMap` is the main container for a hyperspectral dataset. It stores spectral intensities alongside the spatial grid, the spectral calibration axis, masks, images, and metadata.

Key attributes and methods:

- `cube`: reshape the flat 2-D data array into a `(height, width, bands)` cube.
- `get_spectrum()`: aggregate all pixels into a single representative spectrum.
- `get_indices()`: resolve a named mask and spectral ROI to index arrays.

**Full reference:** {py:class}`ramappy.core.spectral_map.SpectralMap`

## Spectrum

{py:class}`ramappy.core.spectrum.Spectrum` holds a single 1-D spectrum with its calibration axis and unit metadata.

**Full reference:** {py:class}`ramappy.core.spectrum.Spectrum`

## Masks

{py:mod}`ramappy.core.masks` defines the named-mask system that restricts pipeline steps to spatial subregions of a map.

**Full reference:** {py:mod}`ramappy.core.masks`

## 2D Images

{py:mod}`ramappy.core.images2d` holds the container types for 2-D reference images
({py:class}`ramappy.core.images2d.image.Image2D`) and groups of images
({py:class}`ramappy.core.images2d.group.ImageGroup`).

**Full reference:** {py:mod}`ramappy.core.images2d`

## Collections

{py:mod}`ramappy.core.collections` provides typed sequence containers for
`SpectralMap` and `Spectrum` objects.

**Full reference:** {py:mod}`ramappy.core.collections`
