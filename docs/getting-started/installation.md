# Installation

`ramappy` is not yet published on PyPI, install it directly from Git. [uv](https://docs.astral.sh/uv/)
is the recommended tool, but a plain `pip` also works.

## Install uv

If you don't already have `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other
methods (`pipx`, `brew`, `winget`, ...).

## Install ramappy

Into an existing project managed by `uv`:

```bash
uv add git+https://github.com/ramappy/ramappy.git
```

As a standalone tool / one-off environment:

```bash
uv pip install git+https://github.com/ramappy/ramappy.git
```

With plain `pip`:

```bash
pip install git+https://github.com/ramappy/ramappy.git
```

To pin to a specific commit, tag, or branch, append `@<ref>` to the URL, e.g.,
`git+https://github.com/ramappy/ramappy.git@v0.1.0`.

## Optional dependencies

Some functionality in **RamAppy** relies on optional dependencies. Install the relevant extras only if you need the associated features.

### Pandas export

Exporting data to a `pandas.DataFrame` requires the `pandas` extra, which installs both **pandas** and **PyArrow**.

```bash
uv add "ramappy[pandas] @ git+https://github.com/ramappy/ramappy.git"
```

### Intel® Extension for Scikit-learn

Support for the **Intel® Extension for Scikit-learn** is enabled by default through the `acceleration` dependency group. It provides hardware-optimized implementations of many scikit-learn algorithms when available, while remaining fully compatible with the standard scikit-learn API.

## Batch CLI

Installing `ramappy` also installs the `ramappy-batch` console script, used to run a
pipeline over many files in parallel. See [ramappy-batch](../cli/batch.md) for details.

## Development install

To work on `ramappy` itself, clone the repository and let `uv` set up a local environment with
the `dev` and `docs` extras:

```bash
git clone https://github.com/ramappy/ramappy.git
cd ramappy
uv sync --extra dev --extra docs
```

Next: [Quickstart](quickstart.md).
