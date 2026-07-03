# Extending ramappy

`ramappy` resolves the string `name` in a step or IO config to an actual Python callable through
two global registries. Understanding them is the key to extending `ramappy` without modifying its
source — either locally in your own scripts, or by shipping an installable plugin package.

## Pipeline step registry

{py:class}`~ramappy.core.pipeline.PipelineStepRegistry` maps a step name string (e.g.
`"correct_baseline"`) to a {py:class}`~ramappy.core.pipeline.PipelineStep` descriptor: the
function that implements it, its Pydantic params model, its category, and display metadata.
Steps register themselves via the {py:func}`~ramappy.core.pipeline.pipeline_step` decorator —
importing the module that defines a step is enough to register it.

```python
from ramappy.core.pipeline import PipelineStepRegistry

sorted(PipelineStepRegistry.available_steps)  # every registered step name
PipelineStepRegistry.available_steps["correct_baseline"].params_validator
```

Built-in steps live under {py:mod}`ramappy.pipeline.steps` and
{py:mod}`ramappy.analysis`/{py:mod}`ramappy.processing`; see [Pipeline Steps](../reference/steps.md)
for the full curated list.

## IO format registries

{py:class}`~ramappy.io.core.InputFormatRegistry` and {py:class}`~ramappy.io.core.OutputFormatRegistry`
work the same way for file IO: a `format_name` string maps to an
{py:class}`~ramappy.io.core.InputFormat`/{py:class}`~ramappy.io.core.OutputFormat` descriptor via
the {py:func}`~ramappy.io.core.input_format`/{py:func}`~ramappy.io.core.output_format` decorators.

```python
from ramappy.io.core import InputFormatRegistry

InputFormatRegistry.list_formats()          # every registered input format name
InputFormatRegistry.get_format("csv").read  # the reader for that format
```

See [IO Formats](../reference/io-formats.md) for the full list.

## Why a registry?

Both registries decouple *configuration* (a plain string + a dict of params, safe to store in
YAML/JSON) from *implementation* (a Python function). This is what lets pipelines and file format
choices be fully serializable, and lets plugins register new steps/formats simply by being
installed.

## Writing a custom step

A pipeline step is a plain function plus a Pydantic params model, registered with the
{py:func}`~ramappy.core.pipeline.pipeline_step` decorator. Once registered (i.e. once the module
defining it has been imported), it's usable by name in any `Pipeline`, procedural or YAML.

```python
from ramappy.core.pipeline import StepParams, StepClass, SingleModelParamsValidator, pipeline_step


class StepScaleIntensityParams(StepParams):
    """Parameters for `scale_intensity`."""

    factor: float = 1.0


@pipeline_step(
    step_name="scale_intensity",
    params_validator=SingleModelParamsValidator(StepScaleIntensityParams),
    friendly_name="Scale Intensity",
    step_category=StepClass.UTILITY,
    modifies_data=True,
)
def scale_intensity(spectral_map, factor: float = 1.0):
    """Multiply all intensities by a constant factor."""
    spectral_map.data = spectral_map.data * factor
```

This step is now usable exactly like any built-in one:

```python
from ramappy.pipeline import Pipeline
from ramappy.core.pipeline import ProcessingStepConfig

pipeline = Pipeline(steps=[ProcessingStepConfig(name="scale_intensity", params={"factor": 2.0})])
pipeline.run(hsi)
```

### Key pieces

- **`StepParams` subclass** — a Pydantic model listing the step's configurable parameters. Mix in
  {py:class}`~ramappy.core.pipeline.ParamsAcceptRoIX`, {py:class}`~ramappy.core.pipeline.ParamsRequireMask`,
  etc. (see [Pipeline Steps](../reference/steps.md#common-parameter-mixins)) to accept a spectral
  ROI or restrict the step to a named mask.
- **`params_validator`** — {py:class}`~ramappy.core.pipeline.SingleModelParamsValidator` for a
  single params model; {py:class}`~ramappy.core.pipeline.UnionParamsValidator` if the step accepts
  more than one shape of params (e.g., `method`-dependent parameter sets).
- **`step_category`** — one of the {py:class}`~ramappy.core.pipeline.StepClass` values
  (`processing`, `analysis`, `visualization`, `utility`); used for grouping in
  [Pipeline Steps](../reference/steps.md).
- **the function itself** — receives the `SpectralMap`/`Spectrum` as its first argument plus the
  validated params as keyword arguments, and mutates it in place.

## Writing a custom IO format

An IO format is a reader and/or writer function plus a Pydantic params model, registered with the
{py:func}`~ramappy.io.core.input_format`/{py:func}`~ramappy.io.core.output_format` decorators.

```python
from ramappy.io.core import IOParams, input_format


class DemoNpyParams(IOParams):
    """Parameters for the demo `.npy` reader."""

    img_width: int | None = None
    img_height: int | None = None


@input_format(
    "demo_npy",
    friendly_name="Demo NumPy binary",
    extensions={"npy"},
    format_params_model=DemoNpyParams,
)
def read_demo_npy(filepath_or_buffer, *, img_width=None, img_height=None):
    import numpy as np
    from ramappy.core.spectral_map import SpectralMap

    arr = np.load(filepath_or_buffer)
    x = np.arange(arr.shape[-1], dtype=float)
    return SpectralMap(x=x, data=arr.reshape(-1, arr.shape[-1]), img_width=img_width, img_height=img_height)
```

Once imported, `"demo_npy"` is usable anywhere a format name is accepted:

```python
from ramappy.io.core import InputFormatRegistry

fmt = InputFormatRegistry.get_format("demo_npy")
hsi = fmt.read("map.npy", params={"img_width": 2, "img_height": 2})
```

or through the high-level {py:func}`ramappy.io.common.read_file` entrypoint, which dispatches by
file extension to whichever registered format claims it (using `sniffer` to disambiguate when
several formats share an extension).

### Key pieces

- **`IOParams` subclass** — the Pydantic model describing this format's parameters. Mix in
  {py:class}`~ramappy.io.core.IOParamsAsHsi` to expose the `as_hsi` toggle other readers use.
- **`extensions`** — file extensions this format claims; used by
  {py:meth}`~ramappy.io.core.InputFormatRegistry.from_extension` to auto-detect a format from a
  filename when none is given explicitly.
- **`sniffer`** — optional callable used to disambiguate when multiple formats share an extension
  (e.g., `matlab` vs `matlab_witec`, both `.mat`).
- **the reader/writer function** — reads from `filepath_or_buffer` and returns a
  `SpectralMap`/`Spectrum` (readers), or accepts one and writes it out (writers).

## The plugin system

Steps and IO formats don't have to live inside the `ramappy` package itself. A separate,
independently versioned Python package can register new steps/formats simply by being installed
alongside `ramappy` — no code changes to `ramappy` required.

Plugin discovery uses standard Python [entry points](https://packaging.python.org/en/latest/specifications/entry-points/)
under one of three groups, matching the {py:class}`PluginGroup <ramappy.core.plugin_factory.PluginGroup>`
values: `ramappy.plugins.io`, `ramappy.plugins.processing`, `ramappy.plugins.steps`. A plugin
declares one or more entry points in its `pyproject.toml` pointing at a module (or attribute) to
import — the import itself is what triggers registration via `@pipeline_step`/`@input_format`/
`@output_format`.
