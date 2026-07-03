from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    GetJsonSchemaHandler,
    SerializeAsAny,
    TypeAdapter,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import TypedDict

from ramappy.core.exceptions import PipelineError
from ramappy.utils import generate_key, is_sorted

# Type aliases for spectral data

SpectralAxis = npt.ArrayLike
"""Type alias for the spectral axis array (x)."""

SpectralData = npt.ArrayLike
"""Type alias for intensity data arrays (n_pixels, n_spectral)."""


class Changes(TypedDict):
    """Change set produced by an :class:`ramappy.core.collections.EntityTransaction`."""

    new: set[str]
    changed: set[str]
    deleted: set[str]


class StepResult(BaseModel):
    masks: Changes | None = None
    masks_group: Changes | None = None
    images: Changes | None = None
    images_group: Changes | None = None
    spectra: Changes | None = None
    modified_data: bool | None = None
    results: dict[str, Any] = Field(default_factory=dict)


class NumpyArrayROI:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetJsonSchemaHandler,
    ) -> core_schema.CoreSchema:
        roi_schema = core_schema.union_schema(
            [
                core_schema.tuple_schema(
                    [
                        core_schema.float_schema(),
                        core_schema.float_schema(),
                    ]
                ),
                core_schema.list_schema(
                    core_schema.tuple_schema(
                        [
                            core_schema.float_schema(),
                            core_schema.float_schema(),
                        ]
                    )
                ),
                core_schema.is_instance_schema(np.ndarray),
                core_schema.none_schema(),
            ]
        )

        return core_schema.no_info_after_validator_function(
            cls._to_numpy,
            roi_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
                when_used="json",
            ),
        )

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None

        # canonical internal representation: (N, 2) float ndarray
        return np.atleast_2d(np.asarray(value, dtype=float))

    @staticmethod
    def _serialize(value: np.ndarray | None) -> list | None:
        if value is None:
            return None

        return value.tolist()

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        roi_pair_schema: JsonSchemaValue = {
            "type": "array",
            "prefixItems": [
                {"type": "number"},
                {"type": "number"},
            ],
            "minItems": 2,
            "maxItems": 2,
        }

        return {
            "anyOf": [
                roi_pair_schema,
                {
                    "type": "array",
                    "items": roi_pair_schema,
                },
                {"type": "null"},
            ]
        }


def roi_x_valid(roi: Any, handler: ValidatorFunctionWrapHandler) -> np.ndarray | None:
    roi = handler(roi)
    if roi is None:
        return None
    roi = np.atleast_2d(roi)
    # Sort by upper endpoint
    roi = roi[np.argsort(roi[:, 1])]
    if not is_sorted(roi.ravel()):
        raise ValueError("Regions must not overlap")
    return roi


class ParamsAcceptRoIX(BaseModel):
    """Parameters that accept a spectral ROI (x-axis) for processing."""

    roi_x: NumpyArrayROI | None = None
    """Spectral region(s) of interest (ROI) for processing. ROI is a 2D array of shape (N, 2), where each row is a pair of start and end values for the spectral region. The regions must not overlap and must be sorted by their start values."""

    _validate_roi_x = field_validator("roi_x", mode="wrap")(roi_x_valid)


class ParamsRequireRoIX(BaseModel):
    """Parameters that require a spectral ROI (x-axis) for processing."""

    roi_x: NumpyArrayROI
    """Spectral region(s) of interest (ROI) for processing. ROI is a 2D array of shape (N, 2), where each row is a pair of start and end values for the spectral region. The regions must not overlap and must be sorted by their start values."""

    _validate_roi_x = field_validator("roi_x", mode="wrap")(roi_x_valid)


class ParamsAcceptMask(BaseModel):
    """Parameters that accept a mask for processing."""

    mask: str | None = None
    """Name of the mask to use for processing. If None, no mask will be applied."""


class ParamsRequireMask(BaseModel):
    """Parameters that require a mask for processing."""

    mask: str
    """Name of the mask to use for processing. This parameter is required."""


class ParamsSetResultId(BaseModel):
    """Parameters that allow setting a result ID for processing."""

    res_id: str | None = Field(default_factory=generate_key, alias="id")
    """Result ID to assign to the output of the processing step. If None, a unique ID will be generated automatically."""


class ParamsSeparateRegions(BaseModel):
    """Parameters that allow separating processing by regions."""

    separate_regions: bool = False
    """Whether to process each region separately."""


class ParamsOutputFile(BaseModel):
    """Parameters that allow specifying an output file for processing."""

    out_file: FilePath | Path | None = None
    """Output file path for saving the results of the processing step. If None, the results will not be saved to a file."""


class ParamsParallelProcessing(BaseModel):
    """Parameters that allow specifying parallel processing options."""

    n_jobs: int = -1
    """Number of CPU cores to use for parallel processing. Use -1 to use all available cores."""


class StepParams(BaseModel):
    """Base class for step parameters. All step parameter models should inherit from this class."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StepClass(StrEnum):
    """Category label for a :class:`ramappy.core.pipeline.PipelineStep`."""

    PROCESSING = "processing"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    UTILITY = "utility"
    EXPORT = "export"
    IO = "io"


class ParamsValidator(Protocol):
    def validate(
        self,
        params: dict[str, Any] | bytes | StepParams,
    ) -> StepParams: ...

    def json_schema(self, **kwargs) -> dict[str, Any]: ...


class SingleModelParamsValidator:
    """Validator for a simple StepParams model."""

    def __init__(self, model: type[StepParams]):
        self.model = model

    def validate(
        self,
        params: dict[str, Any] | bytes | StepParams,
    ) -> StepParams:
        if isinstance(params, self.model):
            return params

        if isinstance(params, bytes):
            return self.model.model_validate_json(params)

        return self.model.model_validate(params)

    def json_schema(self, **kwargs):
        return self.model.model_json_schema(**kwargs)


class UnionParamsValidator:
    """Validator for a StepParams model with, e.g., a discriminated union."""

    def __init__(self, schema: Any):
        self.adapter = TypeAdapter(schema)

    def validate(
        self,
        params: dict[str, Any] | bytes | StepParams,
    ) -> StepParams:
        if isinstance(params, StepParams):
            return params

        if isinstance(params, bytes):
            return self.adapter.validate_json(params)

        return self.adapter.validate_python(params)

    def json_schema(self, **kwargs):
        return self.adapter.json_schema(**kwargs)


@dataclass
class PipelineStep:
    """Definition of a single pipeline step, as registered by :func:`pipeline_step`."""

    step_name: str
    """Unique name for the step."""

    params_validator: ParamsValidator
    """Validator for the step parameters."""

    friendly_name: str | None
    """Human-readable name for the step. If None, a title-cased version of *step_name* will be used."""

    supports_preview: bool
    """Whether the step supports previewing its effects before execution."""

    func: Callable
    """The function that implements the step's processing logic."""

    step_category: StepClass
    """Category of the step (e.g., processing, analysis, visualization)."""

    # interactive_only: bool
    modifies_data: Callable[[StepParams], bool] | bool | None = None
    """Whether the step modifies the underlying data. If a callable is provided, it will be called with the step parameters to determine if the data is modified."""

    modifies_spectral_axis: Callable[[StepParams], bool] | bool = False
    """Whether the step modifies the spectral axis. If a callable is provided, it will be called with the step parameters to determine if the spectral axis is modified."""

    modifies_images: bool = False  # if it _specifically_ modifies images (e.g., map_rotation, flipping), ie, data is just reshaped or similar and not directly modified
    """Whether the step modifies images (e.g., map rotation, flipping)."""

    show_difference: bool | Literal["baseline"] = False
    """Whether to show the difference after the step is executed. If 'baseline', it will show the difference from the baseline."""

    # _pre_run: Callable | None = None
    # _post_run: Callable | None = None

    preview_pre_run: Callable[[Any], StepParams] | None = None
    """A callable that takes the data and returns the step parameters for previewing the step's effects."""

    def __post_init__(self):
        if self.friendly_name is None:
            self.friendly_name = self.step_name.replace("_", " ").title()

    def _modifies_data(self, params: StepParams) -> bool:
        if isinstance(self.modifies_data, bool):
            return self.modifies_data
        if callable(self.modifies_data):
            return self.modifies_data(params)
        return self.step_category == StepClass.PROCESSING

    def _modifies_spectral_axis(self, params: StepParams) -> bool:
        if isinstance(self.modifies_spectral_axis, bool):
            return self.modifies_spectral_axis
        if callable(self.modifies_spectral_axis):
            return self.modifies_spectral_axis(params)
        return False

    def validate_params(
        self,
        params: dict[str, Any] | bytes | StepParams,
        *,
        strict: bool = True,
    ) -> StepParams:

        if isinstance(params, StepParams):
            return params

        try:
            return self.params_validator.validate(params)

        except ValidationError as e:
            if strict:
                raise PipelineError(e, input_data=params, model=None) from e
            return params  # type: ignore

    def __repr__(self):
        return f"<Step {self.friendly_name} ({self.step_name})>"

    def run(
        self,
        spectral_map,  # SpectralMap or Spectrum
        params: dict[str, Any] | StepParams,
        *,
        store: Any = None,
        update_images: bool = True,
        show_difference: bool | Literal["baseline"] | None = None,
        **kwargs,
    ) -> Any:
        """Execute this step on *spectral_map*.

        Parameters
        ----------
        spectral_map : SpectralMap | Spectrum
            The data object to process.
        params : dict | StepParams
            Step parameters. Validated automatically.
        store : any, optional
            Zarr project store for incremental persistence.
        update_images : bool
            If ``True`` (default), recompute all unlocked images after the
            step runs.
        show_difference : bool | 'baseline' | None
            Override the step's default ``show_difference`` flag.
        """
        from ramappy.pipeline.context import StepContext, StepRunner

        params = self.validate_params(params)

        runner = StepRunner()
        step_config = ProcessingStepConfig(name=self.step_name, params=params)
        ctx = StepContext(data=spectral_map, step_config=step_config, store=store, options=kwargs)

        # Carry show_difference into ctx.options so middleware can read it.
        show_diff = self.show_difference if show_difference is None else show_difference
        ctx.options["show_difference"] = show_diff

        result = runner.run(ctx)

        if (self._modifies_data(params) or self.modifies_images) and update_images:
            spectral_map.update_images()

        return result


class PipelineStepRegistry:
    """Registry for available pipeline steps."""

    available_steps: ClassVar[dict[str, PipelineStep]] = {}

    steps_classes: ClassVar[dict[StepClass, set[str]]] = defaultdict(set)

    @classmethod
    def register(
        cls,
        step_name: str,
        *,
        params_validator: ParamsValidator,
        friendly_name: str | None = None,
        func: Callable,
        supports_preview: bool = False,
        step_category: StepClass,
        modifies_data: Callable[[StepParams], bool] | bool | None = None,
        modifies_spectral_axis: Callable[[StepParams], bool] | bool = False,
        modifies_images: bool = False,
        show_difference: bool | Literal["baseline"] = False,
        preview_pre_run: Callable | None = None,
    ) -> None:
        cls.available_steps[step_name] = PipelineStep(
            step_name=step_name,
            params_validator=params_validator,
            friendly_name=friendly_name,
            func=func,
            supports_preview=supports_preview,
            step_category=step_category,
            modifies_data=modifies_data,
            modifies_spectral_axis=modifies_spectral_axis,
            modifies_images=modifies_images,
            show_difference=show_difference,
            preview_pre_run=preview_pre_run,
        )
        cls.steps_classes[step_category].add(step_name)

    @classmethod
    def get_step(cls, step_name: str) -> PipelineStep:
        try:
            return cls.available_steps[step_name]
        except KeyError as err:
            raise ValueError(
                f"Step {step_name} not found. Available steps: {list(cls.available_steps.keys())}"
            ) from err


def pipeline_step(
    step_name: str,
    *,
    params_validator: ParamsValidator,
    friendly_name: str | None = None,
    supports_preview: bool = False,
    step_category: StepClass,
    modifies_data: Callable[[StepParams], bool] | bool | None = None,
    modifies_spectral_axis: Callable[[StepParams], bool] | bool = False,
    modifies_images: bool = False,
    show_difference: bool | Literal["baseline"] = False,
    preview_pre_run: Callable[[Any], StepParams] | None = None,
):
    """Decorator to register a new step class

    Parameters
    ----------
    step_name : str
        Unique name for the step.
    params_validator : ParamsValidator
        Validator for the step parameters.
    friendly_name : str, optional
        Human-readable name for the step. If None, a title-cased version of *step_name* will be used.
    supports_preview : bool
        Whether the step supports previewing its effects before execution.
    step_category : StepClass
        Category of the step (e.g., processing, analysis, visualization).
    modifies_data : bool or Callable[[StepParams], bool], optional
        Whether the step modifies the underlying data. If a callable is provided, it will be called with the step parameters to determine if the data is modified.
    modifies_spectral_axis : bool or Callable[[StepParams], bool], optional
        Whether the step modifies the spectral axis. If a callable is provided, it will be called with the step parameters to determine if the spectral axis is modified.
    modifies_images : bool
        Whether the step modifies images (e.g., map rotation, flipping).
    show_difference : bool or 'baseline'
        Whether to show the difference after the step is executed. If 'baseline', it will show the difference from the baseline.
    preview_pre_run : Callable[[Any], StepParams], optional
        A callable that takes the data and returns the step parameters for previewing the step's effects
    """

    def decorator(func: Callable):
        PipelineStepRegistry.register(
            step_name=step_name,
            params_validator=params_validator,
            friendly_name=friendly_name,
            func=func,
            supports_preview=supports_preview,
            step_category=step_category,
            modifies_data=modifies_data,
            modifies_spectral_axis=modifies_spectral_axis,
            modifies_images=modifies_images,
            show_difference=show_difference,
            preview_pre_run=preview_pre_run,
        )
        return func

    return decorator


class ProcessingStepConfig(BaseModel):
    step_id: str = Field(default_factory=generate_key)
    """Unique identifier of this *step instance* within a pipeline (auto-generated)."""

    name: str
    """Name of the registered step to run (must match a key in `PipelineStepRegistry.available_steps`,
    i.e., a step's `step_name`; see the "Step Name" column in the Pipeline Steps reference docs)."""

    params: SerializeAsAny[StepParams | dict[str, Any]]
    results: dict[str, Any] | None = None
    changes: StepResult | None = Field(default=None, exclude=True)

    # Class variable to store subclasses
    _subclasses: ClassVar[dict[str, type[ProcessingStepConfig]]] = {}

    @model_validator(mode="before")
    @classmethod
    def _resolve_params_model(cls, data: Any) -> Any:
        if isinstance(data, dict):
            name = data.get("name")
            params = data.get("params")

            if name and isinstance(params, dict):
                step = PipelineStepRegistry.get_step(name)
                data["params"] = step.params_validator.validate(params)

        return data


class HistoryInput(TypedDict):
    """Configuration input history metadata."""

    format_name: str | None
    format_params: dict[str, Any] | BaseModel | None


class HistoryLog(BaseModel):
    input: HistoryInput | None = None
    steps: list[ProcessingStepConfig | dict[str, Any]] = Field(default_factory=list)

    def set_input_config(self, input_format: str | None = None, input_config: dict[str, Any] | BaseModel | None = None):
        self.input = HistoryInput(format_name=input_format, format_params=input_config)

    def append(self, step: ProcessingStepConfig):
        self.steps.append(step)

    def get_last_changes(self) -> StepResult:
        if not self.steps:
            return StepResult()

        last_step = self.steps[-1]
        if isinstance(last_step, ProcessingStepConfig):
            return last_step.changes if last_step.changes is not None else StepResult()

        if isinstance(last_step, dict):
            changes = last_step.get("changes")
            if changes is None:
                return StepResult()
            try:
                return StepResult.model_validate(changes)
            except ValidationError:
                return StepResult()

        return StepResult()

    def gen_pipeline(self, format_name: Literal["dict", "json", "yaml"] = "dict") -> dict | str:
        if format_name == "json":
            return self.model_dump_json()
        pipeline = self.model_dump(mode="json")
        if format_name == "yaml":
            return yaml.safe_dump(pipeline, allow_unicode=True, default_flow_style=False)
        return pipeline
