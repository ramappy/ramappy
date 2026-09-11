import numpy as np
import pytest
import yaml
from PIL import Image

from ramappy.cli import batch
from ramappy.core.pipeline import ParamsOutputFile, PipelineStepRegistry
from ramappy.pipeline.configuration import BatchConfig
from ramappy.pipeline.steps.masks import StepImportMaskParams


def _write_csv(path, x, pixel_values):
    """Write a minimal wide-format CSV: first column is x, remaining columns are pixels."""
    rows = [[xi, *vals] for xi, vals in zip(x, zip(*pixel_values, strict=True), strict=True)]
    with path.open("w") as f:
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def _runtime_config(pipeline_steps):
    raw_config = {
        "input": {"format": "csv", "format_params": {"img_width": 2, "img_height": 1, "header": None}},
        "pipeline": pipeline_steps,
    }
    return BatchConfig.model_validate(raw_config).build_runtime_config()


def test_batch_main_end_to_end_single_spectrum_output(tmp_path, monkeypatch):
    """Regression test: main() must not re-validate an already-built RuntimePipelineConfig."""
    x = [100.0, 120.0, 140.0, 160.0]
    csv_path = tmp_path / "sample.csv"
    _write_csv(csv_path, x, [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": {"format": "csv", "format_params": {"img_width": 2, "img_height": 1, "header": None}},
                "pipeline": [{"name": "math_operation", "params": {"B": 1.0, "operation": "sub"}}],
                "output": {"single_spectrum": True, "as_dataframe": "polars"},
            }
        )
    )

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        ["ramappy-batch", str(csv_path), str(out_dir), "-c", str(config_path)],
    )

    batch.main()

    assert (out_dir / "config.csv").exists()


def test_inject_masks_raises_clear_error_on_unmatched_filename(tmp_path):
    runtime_config = _runtime_config([{"name": "math_operation", "params": {"B": 1.0, "operation": "sub"}}])
    (tmp_path / "sample_mask_.png").touch()  # empty mask name: glob matches, regex (\w+) does not

    with pytest.raises(ValueError, match="does not match the expected"):
        batch.inject_masks(runtime_config, tmp_path, "sample")


def test_inject_masks_builds_valid_step_tuples_and_survives_inject_outpath(tmp_path):
    runtime_config = _runtime_config([{"name": "math_operation", "params": {"B": 1.0, "operation": "sub"}}])

    mask_path = tmp_path / "sample_mask_foo.png"
    Image.fromarray(np.zeros((2, 2), dtype=bool)).save(mask_path)

    batch.inject_masks(runtime_config, tmp_path, "sample")

    step, params = runtime_config.steps[0]
    assert step is PipelineStepRegistry.get_step("import_mask")
    assert isinstance(params, StepImportMaskParams)
    assert params.res_id == "foo"
    assert params.mask_path == str(mask_path)

    # Must not crash mixing the injected step with the pre-existing (non-output) step.
    batch.inject_outpath(runtime_config, tmp_path / "out")
    assert not isinstance(params, ParamsOutputFile) or params.out_file is not None
