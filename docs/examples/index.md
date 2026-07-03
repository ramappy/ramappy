# Examples

Runnable, end-to-end examples. Each one is shown both as plain Python (procedural) and as a YAML
pipeline definition (declarative) — pick whichever fits your workflow; both produce the same
`Pipeline`.

## Baseline correction + smoothing

Crop to the fingerprint region, remove the fluorescence background, and smooth the spectra:

::::{tab-set}
:::{tab-item} Procedural
```python
import numpy as np
from ramappy.core import SpectralMap
from ramappy.pipeline import Pipeline
from ramappy.core.pipeline import ProcessingStepConfig

# Dummy 5x5 map, 200 spectral bands, standing in for a real reader (see IO Formats)
wavenumbers = np.linspace(100, 1800, 200)
hsi = SpectralMap(x=wavenumbers, data=np.random.rand(25, 200), img_width=5, img_height=5)

pipeline = Pipeline(
    steps=[
        ProcessingStepConfig(name="crop_spectral", params={"roi_x": [600, 1800]}),
        ProcessingStepConfig(name="correct_baseline", params={"method": "arpls", "lambda_": 1e6}),
        ProcessingStepConfig(name="smooth_spectral", params={"method": "whittaker", "lambda_": 30, "d": 2}),
    ]
)
pipeline.run(hsi)
print(hsi.cube.shape)  # (5, 5, 141) -- fewer bands after cropping
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

See [Pipeline Steps](../reference/steps.md) for every step used above (parameters, defaults, and
the underlying function), and [Building Pipelines](../user-guide/building-pipelines.md) for more
on the procedural/YAML equivalence.

## Batch processing many files

For running the same pipeline over a whole directory of files from the command line, see
[ramappy-batch](../cli/batch.md).
