"""Pipeline execution helpers."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import polars as pl
import yaml

from ramappy.core.pipeline import HistoryLog, PipelineStep, PipelineStepRegistry, ProcessingStepConfig, StepParams
from ramappy.io import export_spectra
from ramappy.pipeline.configuration import PipelineConfigLike, RuntimeGeneralConfig, ensure_runtime_config
from ramappy.pipeline.context import StepContext, StepRunner
from ramappy.pipeline.steps import register_all

register_all()


@dataclass
class Pipeline:
    """
    Ordered sequence of data-transform steps.
    Contains NO image/mask creation logic.
    Portable: serializable to YAML/JSON, runnable on any SpectralMap.
    """

    input_format: str | None = None  # format name used to load the data
    input_params: dict[str, Any] | None = None
    steps: list[ProcessingStepConfig] = field(default_factory=list)

    def run(
        self,
        data: SpectralMap,
        *,
        on_progress: Callable[[str, float], None] | None = None,
        stop_after: str | None = None,  # step_id to stop at (for partial replay)
        runner: StepRunner | None = None,
    ) -> SpectralMap:
        """Run all steps in sequence. Returns the modified SpectralMap."""
        runner = runner or StepRunner()

        for config in self.steps:
            ctx = StepContext(data=data, step_config=config)
            runner.run(ctx)

            if on_progress:
                on_progress(config.name, 1.0)
            if stop_after and config.step_id == stop_after:
                break
        return data

    def to_yaml(self, path: Path | None = None) -> str:
        """Serialize to YAML."""
        data = {
            "input_format": self.input_format,
            "input_params": self.input_params,
            # mode="json" invokes each field's JSON serializer (e.g., NumpyArrayROI -> list),
            # keeping the dump plain-YAML-safe instead of falling back to unsafe !!python tags
            "steps": [s.model_dump(mode="json") for s in self.steps],
        }
        res = yaml.safe_dump(data, allow_unicode=True, default_flow_style=False)
        if path:
            Path(path).write_text(res, encoding="utf-8")
        return res

    @classmethod
    def from_yaml(cls, path: Path | str) -> Pipeline:
        """Load from YAML."""
        content = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        steps = [ProcessingStepConfig.model_validate(s) for s in data.get("steps", [])]
        return cls(input_format=data.get("input_format"), input_params=data.get("input_params"), steps=steps)

    @classmethod
    def from_history(cls, log: HistoryLog) -> Pipeline:
        """Upgrade a legacy HistoryLog to a Pipeline (strips change tracking)."""
        input_format = None
        input_params = None
        if log.input:
            input_format = log.input.get("format_name")
            input_params = log.input.get("format_params")

        return cls(
            input_format=input_format,
            input_params=input_params if isinstance(input_params, dict) else None,
            steps=[
                s if isinstance(s, ProcessingStepConfig) else ProcessingStepConfig.model_validate(s) for s in log.steps
            ],
        )


if TYPE_CHECKING:
    from ramappy.core.spectral_map import SpectralMap


MaskAgg = str | None
MaskSpec = dict[str, MaskAgg] | Sequence[str] | set[str] | str | None
ExportResult = pl.DataFrame | BytesIO | PathLike[str] | str | None


def process_pipeline(
    filepath_or_buffer: str,
    config: PipelineConfigLike,
) -> SpectralMap | ExportResult:
    """Batch process a file, using the provided config.

    Parameters
    ----------
    filepath_or_buffer : str
        Input path or buffer pointing to the dataset to process.
    config : PipelineConfigLike
        Pipeline definition, either as a config object or plain mapping.

    Returns
    -------
    SpectralMap | polars.DataFrame | pandas.DataFrame | BytesIO | os.PathLike[str] | str | None
        The mutated `HSI` instance when no export step is configured, a DataFrame-like
        object when exporting single spectra, or an exported artifact path/handle when
        writing datasets via `ramappy.io.export_spectra`.

    Raises
    ------
    FileNotFoundError
        If the reader cannot load the requested file/buffer.
    ValueError
        If the pipeline contains no steps.
    """
    mpl.use("Agg")

    runtime = ensure_runtime_config(config)

    hsi_or_spectrum = runtime.reader.read(filepath_or_buffer, runtime.reader_params, as_hsi=True)
    spectral_map = cast("SpectralMap", hsi_or_spectrum)
    if spectral_map is None:
        msg = "Couldn't import file"
        raise FileNotFoundError(msg)

    steps = runtime.steps
    if not steps:
        msg = "No steps provided in the pipeline"
        raise ValueError(msg)

    _run_steps(spectral_map, steps)

    metadata = _collect_metadata(filepath_or_buffer, runtime.general)
    output_config = _prepare_output_config(runtime.output, metadata, runtime.general)
    if output_config is None:
        return spectral_map

    export_spectrum_step = PipelineStepRegistry.get_step("export_spectrum")
    return _execute_export(spectral_map, output_config, export_spectrum_step)


def _run_steps(spectral_map: SpectralMap, steps: Sequence[tuple[PipelineStep, StepParams]]) -> None:
    for step, step_params in steps:
        step.run(spectral_map, step_params, track_changes=False)


def _collect_metadata(filepath_or_buffer: str, general: RuntimeGeneralConfig | None) -> dict[str, Any]:
    if general is None or general.metadata_reader is None:
        return {}
    try:
        return general.metadata_reader(filepath_or_buffer)
    except ValueError:
        return {}


def _prepare_output_config(
    output_config: dict[str, Any] | None,
    metadata: dict[str, Any],
    general: RuntimeGeneralConfig | None,
) -> dict[str, Any] | None:
    if output_config is None:
        return None
    prepared = copy.deepcopy(output_config)
    if metadata:
        additional = prepared.get("additional_info") or {}
        prepared["additional_info"] = {**metadata, **additional}
    if general and general.inject:
        prepared.setdefault("general", general.inject)
    return prepared


def _execute_export(
    spectral_map: SpectralMap,
    output_config: dict[str, Any],
    export_step: PipelineStep,
) -> ExportResult:
    single_spectrum = output_config.pop("single_spectrum", False)
    if single_spectrum:
        output_config.setdefault("as_dataframe", "polars")
        return _export_single_spectrum(spectral_map, output_config, export_step)
    output_config.setdefault("format", None)
    return export_spectra(spectral_map, **output_config)


def _export_single_spectrum(
    spectral_map: SpectralMap,
    config: dict[str, Any],
    export_step: PipelineStep,
) -> ExportResult:
    mask_spec = config.get("mask")
    if isinstance(mask_spec, (dict, list, tuple, set)):
        frames: list[pl.DataFrame] = []
        for mask_name, agg in _iter_mask_requests(mask_spec, config.get("agg")):
            current_cfg = copy.deepcopy(config)
            current_cfg["mask"] = mask_name
            if agg is not None:
                current_cfg["agg"] = agg
            frames.append(cast(pl.DataFrame, _call_export_step(spectral_map, current_cfg, export_step)))
        how: Any = "vertical" if config.get("transpose") else "align"
        return pl.concat(frames, how=how)
    return _call_export_step(spectral_map, config, export_step)


def _iter_mask_requests(mask_spec: MaskSpec, agg_default: MaskAgg) -> Iterator[tuple[str, MaskAgg]]:
    if isinstance(mask_spec, dict):
        for mask_name, agg in mask_spec.items():
            yield str(mask_name), agg if isinstance(agg, str) else None
        return
    if isinstance(mask_spec, str) or mask_spec is None:
        if mask_spec is not None:
            yield mask_spec, agg_default
        return
    if isinstance(mask_spec, Iterable):
        for mask_name in mask_spec:
            yield mask_name, agg_default


def _call_export_step(
    spectral_map: SpectralMap,
    config: dict[str, Any],
    export_step: PipelineStep,
) -> ExportResult:
    params = export_step.validate_params(config)
    return export_step.func(spectral_map, **params.model_dump())
