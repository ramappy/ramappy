# Quickstart

## 1. Basic data manipulation

Create a {py:class}`~ramappy.core.spectral_map.SpectralMap` from a NumPy array and inspect its properties:

```python
import numpy as np
from ramappy.core import SpectralMap

# Create a dummy 5x5 hyperspectral map with 200 spectral bands
wavenumbers = np.linspace(100, 1800, 200)
random_intensities = np.random.rand(25, 200)

hsi = SpectralMap(
    x=wavenumbers,
    data=random_intensities,
    img_width=5,
    img_height=5,
    name="Example Raman Map",
)

# Access the 3D data cube (height, width, spectral bands)
cube = hsi.cube
print(f"Cube shape: {cube.shape}")  # (5, 5, 200)

# Extract the mean spectrum
mean_spec = hsi.get_spectrum(agg="mean")
idx = np.argmin(np.abs(mean_spec.spectral_axis - 1000))
print(f"Mean intensity near 1000 cm⁻¹: {mean_spec.data[0, idx]}")
```

## 2. Composing and executing pipelines

Define a sequence of preprocessing steps and execute them on a dataset. See
[Building Pipelines](../user-guide/building-pipelines.md) for the equivalent YAML form.

```python
from ramappy.pipeline import Pipeline
from ramappy.core.pipeline import ProcessingStepConfig

pipeline = Pipeline(
    steps=[
        # Crop to the fingerprint region [600, 1800] cm⁻¹
        ProcessingStepConfig(name="crop_spectral", params={"roi_x": [600, 1800]}),
        # Apply baseline subtraction
        ProcessingStepConfig(name="correct_baseline", params={"method": "arpls", "lambda_": 1e6}),
        # Smooth spectrally using Whittaker's method
        ProcessingStepConfig(name="smooth_spectral", params={"method": "whittaker", "lambda_": 30, "d": 2}),
    ]
)

# Run the pipeline in-place on our dataset
pipeline.run(hsi)
print(f"Processed cube shape: {hsi.cube.shape}")
```

See [Pipeline Steps](../reference/steps.md) for the full list of available steps and their parameters.

## 3. Batch processing (CLI)

Run a pipeline on multiple files in parallel and aggregate the outputs using the `ramappy-batch` tool:

```bash
ramappy-batch ./raw_data/ ./processed_output/ -c pipeline_config.yaml --parallel -1
```

See [ramappy-batch](../cli/batch.md) for the full CLI reference and configuration format.

Next: [Building Pipelines](../user-guide/building-pipelines.md), [Examples](../examples/index.md).
