# Utilities

Helper modules used throughout `ramappy` for array manipulation, region-of-interest indexing, image compositing, and signal processing utilities.

For the full auto-generated API reference see {py:mod}`ramappy.utils`.

---

## Array Utilities

{py:mod}`ramappy.utils.array` provides general-purpose NumPy array helpers — axis permutations, padding, binning, and other transformations — used across processing steps.

**Full reference:** {py:mod}`ramappy.utils.array`

---

## Region of Interest (ROI)

{py:mod}`ramappy.utils.roi` implements 1-D and 2-D region-of-interest slicing utilities that translate physical coordinate ranges into index masks, used by `crop_spectral`, `crop_spatial`, and related steps.

**Full reference:** {py:mod}`ramappy.utils.roi`

---

## Image Utilities

{py:mod}`ramappy.utils.image` provides helpers for converting spectral data to displayable RGB images, applying colormaps, and compositing multiple image layers into a single output.

**Full reference:** {py:mod}`ramappy.utils.image`

---

## Smoothing

{py:mod}`ramappy.utils.smoothing` implements the core 1-D smoothing algorithms (Whittaker smoother and Savitzky–Golay filter) used by the `smooth_spectral` pipeline step.

**Full reference:** {py:mod}`ramappy.utils.smoothing`

---

## Metrics

{py:mod}`ramappy.utils.metrics` provides spectral quality and distance metrics used internally by denoising and analysis algorithms.

**Full reference:** {py:mod}`ramappy.utils.metrics`
