from ramappy.core.plugin_factory import discover_and_import_plugins
from ramappy.pipeline.configuration import validate_pipeline
from ramappy.pipeline.pipeline import Pipeline, process_pipeline

discover_and_import_plugins("processing")


__all__ = [
    "Pipeline",
    "process_pipeline",
    "validate_pipeline",
]
