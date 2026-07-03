"""Utility functions for handling optional dependencies."""

from importlib.util import find_spec


def _require_pandas_dependencies() -> None:
    """Raise ImportError if the optional pandas dependencies are unavailable."""

    if find_spec("pandas") is None or find_spec("pyarrow") is None:
        raise ImportError(
            "Converting to pandas requires the optional 'pandas' dependency group "
            "(pandas and pyarrow). Install it with:\n\n"
            "    uv sync --group pandas"
        )
