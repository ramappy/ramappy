# Building Pipelines

A {py:class}`~ramappy.pipeline.pipeline.Pipeline` is an ordered list of
{py:class}`~ramappy.core.pipeline.ProcessingStepConfig` objects: a step `name` (must match a
registered [step name](../reference/steps.md)) plus a `params` mapping validated against that
step's Pydantic params model. You can build one directly in Python, or load one from YAML/JSON —
both produce the exact same `Pipeline`:

```{admonition} Where to find step names and IO formats
:class: tip

The full, exhaustive list of every `name` you can use in a step is [Pipeline Steps](../reference/steps.md).
For reading/writing data to disk (see [Running against real data](#running-against-real-data)
below), the full list of supported formats is [IO Formats](../reference/io-formats.md).
```

::::{tab-set}
:::{tab-item} Procedural
```python
from ramappy.pipeline import Pipeline
from ramappy.core.pipeline import ProcessingStepConfig

pipeline = Pipeline(
    steps=[
        ProcessingStepConfig(name="crop_spectral", params={"roi_x": [600, 1800]}),
        ProcessingStepConfig(name="correct_baseline", params={"method": "arpls", "lambda_": 1e6}),
        ProcessingStepConfig(name="smooth_spectral", params={"method": "whittaker", "lambda_": 30, "d": 2}),
    ]
)
pipeline.run(hsi)  # modifies hsi in place
```
:::
:::{tab-item} YAML
```yaml
# pipeline.yaml
steps:
  - name: crop_spectral
    params: {roi_x: [600, 1800]}
  - name: correct_baseline
    params: {method: arpls, lambda_: 1.0e6}
  - name: smooth_spectral
    params: {method: whittaker, lambda_: 30, d: 2}
```
```python
from ramappy.pipeline import Pipeline

pipeline = Pipeline.from_yaml("pipeline.yaml")
pipeline.run(hsi)
```
:::
::::

```{seealso}
[Extending ramappy](extending-ramappy.md) explains how step names and IO format names resolve to
Python callables at runtime, and how to register your own. [ramappy-batch](../cli/batch.md)
documents the separate (but related) YAML shape used by the batch CLI, which wraps the same
`steps` list inside a larger `input`/`pipeline`/`output` document.
```

## Serializing a pipeline you built in Python

{py:meth}`Pipeline.to_yaml() <ramappy.pipeline.pipeline.Pipeline.to_yaml>` is safe for ordinary
pipeline configuration data, including array-valued step parameters such as `roi_x`.
It serializes each step via `model_dump(mode="json")` and writes YAML with the standard safe
serializer, so the result can be read back with {py:meth}`Pipeline.from_yaml() <ramappy.pipeline.pipeline.Pipeline.from_yaml>` without hand-editing the file.

## Running against real data

`Pipeline.run()` operates on an in-memory {py:class}`~ramappy.core.spectral_map.SpectralMap`.
To read one from disk first, use {py:func}`ramappy.io.common.read_file` — see
[IO Formats](../reference/io-formats.md) for the list of supported formats.
