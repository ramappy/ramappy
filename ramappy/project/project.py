from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ramappy.core.pipeline import PipelineStep, StepParams
from ramappy.pipeline.pipeline import Pipeline
from ramappy.project.versions import AssetVersions

if TYPE_CHECKING:
    from ramappy.core.spectral_map import SpectralMap
    from ramappy.core.spectrum import Spectrum
    from ramappy.io.zarr.store import ZarrProjectStore


@dataclass
class RamAppProject:
    """Backend session wrapper around a spectral dataset and its pipeline.

    Parameters
    ----------
    data : SpectralMap
        Active project dataset.
    pipeline : Pipeline
        Portable sequence of processing/analysis steps.
    project_id : str
        Stable project identifier.
    preview_step : tuple[PipelineStep, StepParams] | None, optional
        Optional in-memory preview step not yet committed to history.
    asset_versions : AssetVersions
        Version counters for cacheable rendered assets.
    last_access : datetime.datetime
        Last access timestamp.
    """

    data: SpectralMap
    pipeline: Pipeline  # the portable, reusable pipeline
    project_id: str
    preview_step: tuple[PipelineStep, StepParams] | None = None
    asset_versions: AssetVersions = field(default_factory=AssetVersions)
    last_access: datetime.datetime = field(default_factory=datetime.datetime.now)

    _mean_spectrum_cache: Spectrum | None = field(default=None, repr=False)
    _store: ZarrProjectStore | None = field(default=None, repr=False)

    def __post_init__(self):
        # Link data to project for middleware access
        self.data._project = self

    def export_pipeline(self, path: Path | str) -> None:
        """Export the current pipeline as a portable YAML file."""
        self.pipeline.to_yaml(Path(path))

    def apply_pipeline_to(self, other: SpectralMap) -> SpectralMap:
        """Run this project's pipeline on a different map."""
        return self.pipeline.run(other)
