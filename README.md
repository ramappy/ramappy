<p align="center">
  <a href="https://ramapp.io"><img width="75%" src="./docs/assets/ramappy_logo.svg" alt="ramappy"></a>
</p>

# ramappy

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CI](https://github.com/ramappy/ramappy/actions/workflows/ci.yml/badge.svg)](https://github.com/ramappy/ramappy/actions)

`ramappy` is a Python toolkit for hyperspectral spectroscopy processing and analysis,
with a strong focus on Raman maps.


It provides:

- typed core data structures (`SpectralMap`, `Spectrum`)
- composable processing and analysis steps
- extensible IO format registries
- project/session primitives for backend applications

## Architecture overview

The codebase is organized around a few stable layers:

1. **Core domain (`ramappy.core`)**
   - `SpectralMap`: map-level container (`n_pixels × n_spectral` + spatial geometry)
   - `Spectrum`: spectrum container (single or multi-pixel)
   - entities (`Mask`, `Image2D`, grouped/ordered collections)

2. **Pipeline layer (`ramappy.core.pipeline`, `ramappy.pipeline`)**
   - step schemas (`StepParams`)
   - step registry/decorator (`@pipeline_step`)
   - middleware-based execution (`StepRunner`)

3. **IO layer (`ramappy.io`)**
   - format registries for readers/writers
   - decorator-based registration (`@input_format`, `@output_format`)
   - automatic format guessing by extension/sniffer

4. **Project/session layer (`ramappy.project`)**
   - backend-ready wrapper for data + pipeline + versioned assets

## Installation

`ramappy` is not yet published on PyPI; install it directly from Git with [uv](https://docs.astral.sh/uv/)
(recommended) or `pip`:

```bash
uv add git+https://github.com/ramappy/ramappy.git
# or
pip install git+https://github.com/ramappy/ramappy.git
```

To pin to a specific commit, tag, or branch, append `@<ref>` to the URL, e.g.,
`git+https://github.com/ramappy/ramappy.git@v0.1.0`.

## Core abstractions

### `SpectralMap`

Main map object containing:

- spectral axis (`x`)
- flattened spectral data matrix (`data`)
- spatial shape (`img_width`, `img_height`)
- unit metadata (`x_axis_unit`, `data_unit`)
- optional masks/images/spectra/history

### `Spectrum`

Spectrum object used for:

- extracted map spectra (cursor, mask averages, etc.)
- standalone imported spectra
- plotting/export and alignment operations

### IO registries

IO formats are registered centrally through decorators:

- `@input_format(...)` → reader registration
- `@output_format(...)` → writer registration

This keeps extension handling and parameter validation declarative.

## Usage examples

### Create a map from arrays

**From a 3-D cube** `(height, width, n_spectral)`: preferred when your data is already spatially arranged:

```python
import numpy as np
from ramappy import SpectralMap

x = np.linspace(100.0, 1800.0, 512)
cube = np.random.rand(10, 10, 512).astype(np.float32)  # (height, width, n_spectral)

hsi = SpectralMap.from_cube(cube, spectral_axis=x)
```

**From a 2-D flat array** `(n_pixels, n_spectral)`: when data arrives row-major (all pixels concatenated):

```python
import numpy as np
from ramappy import SpectralMap

x = np.linspace(100.0, 1800.0, 512)
data = np.random.rand(100, 512).astype(np.float32)  # (n_pixels, n_spectral)

hsi = SpectralMap(x=x, data=data, img_width=10, img_height=10)
# img_width * img_height must equal data.shape[0]
```

Both representations are interchangeable at runtime: `hsi.data` is the flat view and `hsi.cube` is the
zero-copy 3-D view `(height, width, n_spectral)`.

### Read data from file

```python
from ramappy.io import read_file

hsi = read_file("sample.zarr", format="auto", as_hsi=True)
```

### Export map data

```python
from ramappy.io import export_spectra

export_spectra(hsi, out_file="out.zarr", format_name="zarr", format_params={})
```

## Command-line interface

Installing `ramappy` exposes the `ramappy-batch` command for running a pipeline over a set of files and
collecting all results into a single aggregated output file.

```
ramappy-batch [src_files ...] dest_dir -c config.yaml [-p N] [-w]
```

| Argument | Description |
|---|---|
| `src_files` | One or more input files, or a single directory (all files with the format's extension are used) |
| `dest_dir` | Output directory. The aggregated result is written to `dest_dir/<config_stem>.<ext>` |
| `-c`, `--config_file` | YAML pipeline config file (required) |
| `-p`, `--parallel` | Number of parallel workers; `-1` = all CPUs (default), `0` = serial |
| `-w`, `--overwrite` | Overwrite the output file if it already exists |

### Config file format

```yaml
input:
  format: renishaw_wdf # any registered input format (renishaw_wdf, csv, zarr, …)
  format_params: {}    # format-specific reader options

pipeline:
  - name: correct_baseline
    params:
      method: rubberband

output:
  format: parquet      # csv (default), parquet, or feather
  transpose: false     # set true to stack results vertically instead of aligning columns

general:               # optional section
  mask_dir: path/to/masks   # directory containing pre-computed masks named <stem>_mask_<id>.png
```

### Example

```bash
# Process all .wdf files in raw_data/, write results to results/
ramappy-batch raw_data/ results/ -c pipeline.yaml

# Serial processing, overwrite existing output
ramappy-batch sample_*.wdf results/ -c pipeline.yaml -p 0 -w
```

## Development guidelines

### Environment setup

Prefer `uv`:

```bash
uv sync --all-extras
```

Or editable install:

```bash
pip install -e .
```

## Documentation (auto-generated)

API docs are generated from docstrings using **Sphinx**.

### Build the documentation

To build the documentation locally, run:

```bash
# Sync dependencies
uv sync --extra docs

# Build HTML documentation
cd docs
make html
```

The output HTML files will be generated in `docs/_build/html/`. You can open `docs/_build/html/index.html` in your web browser to view the built site.

Configuration lives in `docs/conf.py`; manually written documentation pages live under `docs/reference/`
(e.g., `core.md`, `processing.md`), while `docs/reference/api/` holds the auto-generated API reference.

## License

`ramappy` is released under the [GNU General Public License v3.0](LICENSE).
