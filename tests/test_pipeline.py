import numpy as np

from ramappy.core import SpectralMap
from ramappy.core.pipeline import ProcessingStepConfig
from ramappy.pipeline.pipeline import Pipeline


def test_pipeline_basic_run():
    x = np.linspace(100, 1800, 200)
    intensities = np.ones((25, 200), dtype=np.float32)
    hsi = SpectralMap(x=x, data=intensities, img_width=5, img_height=5)

    # Create a pipeline with a simple math operation (subtract a constant)
    # The math_operation step uses: operand, agg_operand, mask, op, roi_x
    step = ProcessingStepConfig(
        step_id="math_1",
        name="math_operation",
        params={"B": 1.0, "operation": "sub", "A": None, "aggregate_B": "mean", "roi_x": None},
    )

    pipeline = Pipeline(steps=[step])
    out = pipeline.run(hsi)

    assert np.allclose(out.data, 0.0)
    assert len(out.history.steps) == 1


def test_pipeline_serialization(tmp_path):
    yaml_path = tmp_path / "pipeline.yaml"
    step = ProcessingStepConfig(step_id="math_1", name="math_operation", params={"B": 2.0, "operation": "sub"})
    pipeline = Pipeline(input_format="witec", steps=[step])

    pipeline.to_yaml(yaml_path)
    assert yaml_path.exists()

    loaded = Pipeline.from_yaml(yaml_path)
    assert loaded.input_format == "witec"
    assert len(loaded.steps) == 1
    assert loaded.steps[0].name == "math_operation"
    assert loaded.steps[0].params.B == 2.0


def test_pipeline_serialization_uses_safe_yaml_for_array_parameters(tmp_path):
    yaml_path = tmp_path / "pipeline_with_roi.yaml"
    step = ProcessingStepConfig(
        name="math_operation",
        params={"B": 2.0, "operation": "sub", "roi_x": [[100.0, 200.0]]},
    )
    pipeline = Pipeline(steps=[step])

    serialized = pipeline.to_yaml(yaml_path)

    assert "!!python" not in serialized
    loaded = Pipeline.from_yaml(yaml_path)
    np.testing.assert_allclose(loaded.steps[0].params.roi_x, [[100.0, 200.0]])
