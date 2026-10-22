"""Utilities for describing and validating batch pipeline configurations."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ramappy.core.pipeline import PipelineStep, PipelineStepRegistry, StepParams
from ramappy.io.core import InputFormat, InputFormatRegistry, IOParams

MetadataReader = Callable[[str], dict[str, Any]]


def _metadata_from_filename(filepath: str) -> dict[str, Any]:
    return {"filename": Path(filepath).stem}


def _metadata_from_regex(filepath: str, pattern: re.Pattern[str]) -> dict[str, Any]:
    match = pattern.match(Path(filepath).stem)
    if match is None:
        return {}
    return match.groupdict()


_BUILTIN_METADATA_READERS: dict[str, MetadataReader] = {
    "filename": _metadata_from_filename,
    "stem": _metadata_from_filename,
}


class BatchInputConfig(BaseModel):
    format: str = "csv"
    format_params: dict[str, Any] = Field(default_factory=dict)

    def resolve_reader(self) -> InputFormat:
        reader = InputFormatRegistry.get_format(self.format)
        if reader is None:
            available = ", ".join(sorted(InputFormatRegistry.list_formats()))
            msg = f"Input format '{self.format}' is not supported. Available formats: {available}."
            raise ValueError(msg)
        return reader

    def validated_params(self, reader: InputFormat | None = None) -> IOParams:
        reader = reader or self.resolve_reader()
        return reader.validate_params(self.format_params)


class BatchOutputConfig(BaseModel):
    format: str | None = None
    single_spectrum: bool = False
    transpose: bool = False
    as_dataframe: bool | str = False
    additional_info: dict[str, str | float] | None = None

    model_config = ConfigDict(extra="allow")


class BatchGeneralConfig(BaseModel):
    metadata_reader: str | dict[str, str] | None = None
    inject: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    def build_runtime_config(self) -> RuntimeGeneralConfig | None:
        reader = self._resolve_metadata_reader()
        payload = self.inject or None
        if reader is None and payload is None:
            return None
        return RuntimeGeneralConfig(metadata_reader=reader, inject=payload)

    def _resolve_metadata_reader(self) -> MetadataReader | None:
        reader_cfg = self.metadata_reader
        if reader_cfg is None:
            return None
        if isinstance(reader_cfg, str):
            if reader_cfg in _BUILTIN_METADATA_READERS:
                return _BUILTIN_METADATA_READERS[reader_cfg]
            supported = ", ".join(sorted(_BUILTIN_METADATA_READERS))
            msg = f"Unknown metadata_reader '{reader_cfg}'. Supported built-ins: {supported}"
            raise ValueError(msg)
        if isinstance(reader_cfg, dict):
            regex = reader_cfg.get("regex")
            if not regex:
                msg = "metadata_reader mapping requires a 'regex' entry"
                raise ValueError(msg)
            pattern = re.compile(regex)
            return lambda filepath: _metadata_from_regex(filepath, pattern)
        msg = f"Unsupported metadata_reader configuration: {reader_cfg!r}"
        raise TypeError(msg)


class BatchConfig(BaseModel):
    input: BatchInputConfig
    pipeline: list[dict[str, Any]] = Field(default_factory=list)
    output: BatchOutputConfig | None = None
    general: BatchGeneralConfig = Field(default_factory=BatchGeneralConfig)

    model_config = ConfigDict(extra="allow")

    def build_runtime_config(
        self,
        *,
        reader: InputFormat | None = None,
        reader_params: IOParams | None = None,
    ) -> RuntimePipelineConfig:
        reader = reader or self.input.resolve_reader()
        if reader_params is not None:
            params = reader_params.model_copy(deep=True)
        else:
            params = self.input.validated_params(reader)
        steps = validate_pipeline(self.pipeline)
        output_cfg = self.output.model_dump(mode="python", exclude_none=True) if self.output is not None else None
        general_cfg = self.general.build_runtime_config()
        return RuntimePipelineConfig(
            reader=reader,
            reader_params=params,
            steps=steps,
            output=output_cfg,
            general=general_cfg,
        )


@dataclass(slots=True)
class RuntimeGeneralConfig:
    metadata_reader: MetadataReader | None = None
    inject: Mapping[str, Any] | None = None


@dataclass(slots=True)
class RuntimePipelineConfig:
    reader: InputFormat
    reader_params: IOParams
    steps: list[tuple[PipelineStep, StepParams]]
    output: dict[str, Any] | None = None
    general: RuntimeGeneralConfig | None = None


def _validate_single_step(step: dict[str, Any]) -> tuple[PipelineStep, StepParams]:
    try:
        current = PipelineStepRegistry.get_step(step["name"])
        params = current.validate_params(step.get("params", {}))
    except KeyError as exc:
        msg = f"Step definition must include a 'name'. Invalid step: {step!r}"
        raise ValueError(msg) from exc
    else:
        return current, params


def validate_pipeline(pipeline: Iterable[dict[str, Any]]) -> list[tuple[PipelineStep, StepParams]]:
    return [_validate_single_step(step) for step in pipeline]


PipelineConfigLike = BatchConfig | RuntimePipelineConfig | Mapping[str, Any]


def ensure_runtime_config(config: PipelineConfigLike) -> RuntimePipelineConfig:
    if isinstance(config, RuntimePipelineConfig):
        return config
    if isinstance(config, BatchConfig):
        return config.build_runtime_config()
    if isinstance(config, Mapping):
        model = BatchConfig.model_validate(config)
        return model.build_runtime_config()
    msg = f"Unsupported pipeline configuration type: {type(config)!r}"
    raise TypeError(msg)


def _namespace_refs(node: Any, prefix: str) -> Any:
    """Prefix every local ``$defs`` reference so sub-schemas can be merged without name clashes."""
    if isinstance(node, dict):
        return {
            k: (
                f"#/$defs/{prefix}.{v.removeprefix('#/$defs/')}"
                if k == "$ref" and isinstance(v, str)
                else _namespace_refs(v, prefix)
            )
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_namespace_refs(v, prefix) for v in node]
    return node


def _embed_schema(schema: dict[str, Any], prefix: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Move *schema*'s ``$defs`` into *defs* under *prefix* and return the rewritten schema."""
    schema = _namespace_refs(schema, prefix)
    for name, sub in schema.pop("$defs", {}).items():
        defs[f"{prefix}.{name}"] = sub
    return schema


def _is_core(func: Callable[..., Any]) -> bool:
    return func.__module__.split(".")[0] == "ramappy"


def batch_config_json_schema(*, include_plugins: bool = True) -> dict[str, Any]:
    """Return a JSON Schema (draft 2020-12) for batch pipeline configuration files.

    The schema is built from the registered input formats and pipeline steps.
    With ``include_plugins=False`` only those defined in the ``ramappy`` package
    are included, which gives a reproducible schema regardless of installed plugins.
    """
    formats = {n: f for n, f in InputFormatRegistry.formats.items() if include_plugins or _is_core(f._read_func)}
    steps = {n: s for n, s in PipelineStepRegistry.available_steps.items() if include_plugins or _is_core(s.func)}
    schema = BatchConfig.model_json_schema()
    defs: dict[str, Any] = schema.setdefault("$defs", {})
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "ramappy batch pipeline configuration"

    # Per-format validation of input.format_params
    input_def = defs["BatchInputConfig"]
    input_def["allOf"] = [
        {
            "if": {"properties": {"format": {"const": name}}, "required": ["format"]},
            "then": {
                "properties": {
                    "format_params": _embed_schema(fmt.format_params_model.model_json_schema(), f"input.{name}", defs)
                }
            },
        }
        for name, fmt in sorted(formats.items())
    ]
    input_def["properties"]["format"] = {
        "type": "string",
        "default": "csv",
        "enum": sorted(formats),
    }

    # Pipeline steps: one branch per registered step, discriminated by name
    branches = []
    for name, step in sorted(steps.items()):
        params = _embed_schema(step.params_validator.json_schema(), f"step.{name}", defs)
        branch: dict[str, Any] = {
            "title": step.friendly_name or name,
            "type": "object",
            "properties": {"name": {"const": name}, "params": params},
            "required": ["name"],
        }
        if step.step_category is not None:
            branch["description"] = f"{step.step_category.value} step"
        branches.append(branch)
    schema["properties"]["pipeline"]["items"] = {"oneOf": branches}
    return schema


__all__ = [
    "BatchConfig",
    "BatchGeneralConfig",
    "BatchInputConfig",
    "BatchOutputConfig",
    "PipelineConfigLike",
    "RuntimeGeneralConfig",
    "RuntimePipelineConfig",
    "batch_config_json_schema",
    "ensure_runtime_config",
    "validate_pipeline",
]


if __name__ == "__main__":
    import json
    import sys

    # Usage: python -m ramappy.pipeline.configuration [OUTPUT_FILE]  (core steps/formats only)
    text = json.dumps(batch_config_json_schema(include_plugins=False), indent=2, sort_keys=True) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
