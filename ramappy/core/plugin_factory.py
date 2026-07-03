from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from typing import Literal

PluginGroup = Literal["io", "processing", "steps"]

_loaded_groups: set[PluginGroup] = set()


def discover_and_import_plugins(group: PluginGroup) -> None:
    """Import entry-point plugins for the given ``ramappy.plugins.*`` group.

    Plugin imports are cached per-group to avoid repeated load side effects.
    """

    if group in _loaded_groups:
        return

    discovered_plugins = sorted(entry_points(group=f"ramappy.plugins.{group}"), key=lambda ep: ep.name)
    for plugin in discovered_plugins:
        try:
            plugin.load()
        except Exception as exc:  # pragma: no cover - external plugin failure
            warnings.warn(
                f"Failed loading ramappy plugin '{plugin.name}' in group '{group}': {exc}",
                stacklevel=2,
            )

    _loaded_groups.add(group)


def discover_and_import_many(groups: tuple[PluginGroup, ...]) -> None:
    """Import multiple plugin groups."""

    for group in groups:
        discover_and_import_plugins(group)
