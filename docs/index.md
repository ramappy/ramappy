# ramappy

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

`ramappy` is a Python library for reading, processing, and analyzing hyperspectral datasets, with
a focus on Raman spectroscopy and imaging. It provides typed core data structures, composable
processing/analysis steps, extensible IO format registries, and project/session primitives for
backend applications.

- **Data Containers**: {py:class}`~ramappy.core.spectral_map.SpectralMap` and
  {py:class}`~ramappy.core.spectrum.Spectrum`, with coordinate tracking, spatial grids, metadata,
  and unit labels.
- **Pipelines**: composable, serializable sequences of preprocessing and analysis steps.
- **IO**: readers/writers for proprietary formats (WITec, Renishaw, Horiba) alongside standard
  ones (HDF5, Zarr, MATLAB, CSV).
- **Processing & Analysis**: baseline correction, denoising, clustering, MCR, N-FINDR, and more.

```{toctree}
:maxdepth: 1
:hidden:
:caption: Getting Started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Reference

reference/steps
reference/io-formats
reference/index
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Guide

user-guide/building-pipelines
user-guide/extending-ramappy
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Examples & CLI

examples/index
cli/batch
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: About

license
```

## Architecture overview

The codebase is organized around a few stable layers:

1. **Core domain** ({py:mod}`ramappy.core`) — {py:class}`~ramappy.core.spectral_map.SpectralMap`
   (map-level container: `n_pixels × n_spectral` plus spatial geometry),
   {py:class}`~ramappy.core.spectrum.Spectrum` (single/multi-pixel spectra), and supporting
   entities (masks, images, grouped/ordered collections).
2. **Pipeline layer** ({py:mod}`ramappy.core.pipeline`, {py:mod}`ramappy.pipeline`) — step schemas,
   the step registry/decorator, and middleware-based execution. See [Building Pipelines](user-guide/building-pipelines.md).
3. **IO layer** ({py:mod}`ramappy.io`) — format registries for readers/writers, decorator-based
   registration, automatic format guessing by extension/sniffer. See [IO Formats](reference/io-formats.md).
4. **Project/session layer** ({py:mod}`ramappy.project`) — a backend-ready wrapper for data +
   pipeline + versioned assets.

## Usage examples

Create a map from a 3-D cube `(height, width, n_spectral)`:

```python
import numpy as np
from ramappy import SpectralMap

x = np.linspace(100.0, 1800.0, 512)
cube = np.random.rand(10, 10, 512).astype(np.float32)  # (height, width, n_spectral)

hsi = SpectralMap.from_cube(cube, spectral_axis=x)
```

Read a file from disk, and export processed data back out:

```python
from ramappy.io import read_file, export_spectra

hsi = read_file("sample.zarr", format="auto", as_hsi=True)
export_spectra(hsi, out_file="out.zarr", format_name="zarr", format_params={})
```

See [Quickstart](getting-started/quickstart.md) for a complete pipeline example.

## Getting Started

New to `ramappy`? Install it and run your first pipeline.

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)

## Guide

Build pipelines, and extend `ramappy` with your own steps, IO formats, and plugins.

- [Building Pipelines](user-guide/building-pipelines.md)
- [Extending ramappy](user-guide/extending-ramappy.md)

## Reference

Every pipeline step, IO format, and the full API.

- [Pipeline Steps](reference/steps.md)
- [IO Formats](reference/io-formats.md)
- [Reference index](reference/index.md)

## Examples & CLI

End-to-end examples and the batch command-line tool.

- [Examples](examples/index.md)
- [ramappy-batch](cli/batch.md)

## License

`ramappy` is licensed under the [GNU General Public License, Version 3](license.md).

