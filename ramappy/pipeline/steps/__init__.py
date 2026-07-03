"""Built-in pipeline steps.

These modules are imported for their side effects: the ``@pipeline_step`` decorators
register each step in the global registry.

Keep imports lightweight where possible; prefer local imports inside step functions
for optional/heavy dependencies.
"""

from __future__ import annotations

from ramappy.core.plugin_factory import discover_and_import_plugins


def register_all() -> None:
    """Import all built-in step modules to register them."""

    # Import order is not important for correctness, but we keep it stable.
    import ramappy.analysis as _analysis  # noqa: F401
    import ramappy.processing as _processing  # noqa: F401

    from . import exporting as _exporting  # noqa: F401
    from . import external_spectra as _spectra  # noqa: F401
    from . import images as _images  # noqa: F401
    from . import masks as _masks  # noqa: F401
    from . import math_ops as _math_ops  # noqa: F401
    from . import plotting as _plotting  # noqa: F401

    # Optional external step plugins (entry points: ramappy.plugins.steps)
    discover_and_import_plugins("steps")


__all__ = ["register_all"]
