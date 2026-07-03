from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ramappy.core.pipeline import PipelineStepRegistry, ProcessingStepConfig, StepResult
from ramappy.utils.roi import clean_roi, common_roi_overlap, get_x_regions

if TYPE_CHECKING:
    from ramappy.core.spectral_map import SpectralMap
    from ramappy.io.zarr.store import ZarrProjectStore


@dataclass
class StepContext:
    """State shared across the middleware chain during a single step execution."""

    data: SpectralMap
    step_config: ProcessingStepConfig
    store: ZarrProjectStore | None = None
    options: dict[str, Any] = field(default_factory=dict)

    # Internal state for middleware
    start_time: float = field(default_factory=time.time)
    transactions: dict[str, Any] = field(default_factory=dict)
    changes: StepResult = field(default_factory=StepResult)
    results: dict[str, Any] = field(default_factory=dict)


Middleware = Callable[[StepContext, Callable[[StepContext], Any]], Any]


def transaction_middleware(ctx: StepContext, next_step: Callable[[StepContext], Any]) -> Any:
    """Starts transactions on SpectralMap collections and commits them after."""
    # Start transactions
    ctx.transactions = getattr(ctx.data, "begin_transaction", lambda: {})()

    try:
        result = next_step(ctx)

        # Collect changes from transactions
        ctx.changes = StepResult(
            masks=ctx.transactions["masks"].commit() if "masks" in ctx.transactions else None,
            masks_group=ctx.transactions["masks_group"].commit() if "masks_group" in ctx.transactions else None,
            images=ctx.transactions["images"].commit() if "images" in ctx.transactions else None,
            images_group=ctx.transactions["images_group"].commit() if "images_group" in ctx.transactions else None,
            spectra=ctx.transactions["spectra"].commit() if "spectra" in ctx.transactions else None,
            results=ctx.results,
        )
        return result
    except Exception:
        # Rollback all
        for tx in ctx.transactions.values():
            tx.rollback()
        raise


def persistence_middleware(ctx: StepContext, next_step: Callable[[StepContext], Any]) -> Any:
    """If a store is present, save changed entities incrementally."""
    result = next_step(ctx)

    if ctx.store and ctx.changes:
        # Incremental save logic
        if ctx.changes.masks:
            for m_id in ctx.changes.masks["new"] | ctx.changes.masks["changed"]:
                ctx.store.write_mask(m_id, ctx.data.masks[m_id])
            for m_id in ctx.changes.masks["deleted"]:
                ctx.store.delete_mask(m_id)

        if ctx.changes.images:
            for i_id in ctx.changes.images["new"] | ctx.changes.images["changed"]:
                ctx.store.write_image(i_id, ctx.data.images[i_id])
            for i_id in ctx.changes.images["deleted"]:
                ctx.store.delete_image(i_id)

        if ctx.changes.spectra:
            for s_id in ctx.changes.spectra["new"] | ctx.changes.spectra["changed"]:
                ctx.store.write_spectrum(s_id, ctx.data.spectra[s_id])
            for s_id in ctx.changes.spectra["deleted"]:
                ctx.store.delete_spectrum(s_id)

        # Handle intensities modification if flag is set
        if ctx.changes.modified_data:
            ctx.store.write_intensities(ctx.data.cube)

        # Flush global attributes (including updated history with results).
        project = getattr(ctx.data, "_project", None)
        pipeline = getattr(project, "pipeline", None) if project else None

        ctx.store.flush_attrs(
            pipeline=pipeline,  # pipeline could be None here if not project-aware
            name=ctx.data.name or "",
            data=ctx.data,
        )

    return result


def history_middleware(ctx: StepContext, next_step: Callable[[StepContext], Any]) -> Any:
    """Appends the step to the SpectralMap's history log."""
    result = next_step(ctx)

    # Update config with captured changes before appending
    ctx.step_config.changes = ctx.changes
    ctx.step_config.results = ctx.changes.results
    ctx.data.history.append(ctx.step_config)

    return result


def core_middleware(ctx: StepContext, next_step: Callable[[StepContext], Any]) -> Any:
    """Execute the underlying step function.

    Contract:
    ``step(spectral_map: SpectralMap, **params) -> SpectralMap``.

    For transition safety, ``None`` and ``dict`` returns are still accepted.
    """
    step = PipelineStepRegistry.get_step(ctx.step_config.name)
    params = dict(ctx.step_config.params)

    separate_regions = params.get("separate_regions", False)

    sig = inspect.signature(step.func)
    takes_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if "separate_regions" not in sig.parameters and not takes_kwargs:
        params.pop("separate_regions", None)

    wants_framework_split = separate_regions and "separate_regions" not in sig.parameters

    if wants_framework_split:
        base_regions = get_x_regions(ctx.data.x)

        user_roi = params.get("roi_x")
        roi_list = common_roi_overlap(base_regions, clean_roi(user_roi)) if user_roi is not None else base_regions

        # remove it from params since we are handling it
        params.pop("separate_regions", None)

        res = None
        for roi in roi_list:
            params["roi_x"] = [roi]
            res = step.func(ctx.data, **params)

            if res is not None and hasattr(res, "data") and hasattr(res, "x"):
                ctx.data = res

            if res is not None:
                if isinstance(res, dict):
                    ctx.results.update(res)
                else:
                    ctx.results["value"] = res
    else:
        # We pass track_changes=False because we handle it in transaction_middleware
        res = step.func(ctx.data, **params)

        if res is not None and hasattr(res, "data") and hasattr(res, "x"):
            ctx.data = res

        if res is not None:
            if isinstance(res, dict):
                ctx.results.update(res)
            else:
                ctx.results["value"] = res

    return res


def modification_middleware(ctx: StepContext, next_step: Callable[[StepContext], Any]) -> Any:
    """Detects if spectral data were modified based on step metadata."""
    step = PipelineStepRegistry.get_step(ctx.step_config.name)
    modifies_data = step._modifies_data(ctx.step_config.params)

    if modifies_data and ctx.options.get("save_old_data", False):
        extra_info = getattr(ctx.data, "_extra_info", None)
        if not isinstance(extra_info, dict):
            extra_info = {}
            ctx.data._extra_info = extra_info  # type: ignore
        snapshot = ctx.data.data.copy()
        ctx.data.aux_data["old_data"] = snapshot
        # Backward compatibility for existing consumers.
        ctx.data.aux_data["data_previous"] = snapshot
        extra_info["x_old"] = ctx.data.x.copy()

    result = next_step(ctx)

    if modifies_data:
        ctx.changes.modified_data = True

    if (
        (modifies_data or getattr(step, "modifies_images", False))
        and ctx.options.get("update_images", False)
        and hasattr(ctx.data, "update_images")
    ):
        ctx.data.update_images()

    return result


class StepRunner:
    """Orchestrates middleware chain execution."""

    def __init__(self, middlewares: list[Middleware] | None = None):
        # Explicit execution order, outermost to innermost.
        # - persistence: writes final state to store
        # - history: appends step history
        # - modification: sets modified_data / optional image updates
        # - transaction: collects map-level entity changes atomically
        # - core: executes the registered step function
        self.middlewares = middlewares or [
            persistence_middleware,
            history_middleware,
            modification_middleware,
            transaction_middleware,
            core_middleware,
        ]

    def run(self, ctx: StepContext) -> Any:
        def wrap(m_idx: int) -> Callable[[StepContext], Any]:
            if m_idx >= len(self.middlewares):
                return lambda c: None  # Should not be reached

            current_m = self.middlewares[m_idx]
            if m_idx == len(self.middlewares) - 1:
                return lambda c: current_m(c, lambda _: None)

            return lambda c: current_m(c, wrap(m_idx + 1))

        return wrap(0)(ctx)
