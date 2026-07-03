# ruff: noqa: I001

"""Parsing helpers for the ownformat table layout."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


COL_MATCH = re.compile(
    r"^(?P<type>mask|image)\[(?P<id>.*?)\]: (?P<name>.*?)(?: \(group\[(?P<g_id>.*?)\]: (?P<g_name>.*?)\))?$"
)


@dataclass(frozen=True, slots=True)
class ParsedItemColumn:
    kind: str  # "mask" | "image"
    item_id: str
    item_name: str
    group_id: str | None
    group_name: str | None


def parse_item_column(col_name: str) -> ParsedItemColumn | None:
    """Parse a metadata column name for a mask/image element.

    Returns None if the column doesn't match the expected naming convention.
    """
    m = COL_MATCH.search(col_name)
    if m is None:
        return None
    gd = m.groupdict()
    kind = gd.get("type")
    item_id = gd.get("id")
    item_name = gd.get("name")
    if kind not in {"mask", "image"} or item_id is None or item_name is None:
        raise ValueError(f"Ownformat column parse error: {col_name!r}")
    return ParsedItemColumn(
        kind=kind,
        item_id=item_id,
        item_name=item_name,
        group_id=gd.get("g_id"),
        group_name=gd.get("g_name"),
    )


def split_spectral_and_metadata_columns(columns: list[str]) -> tuple[list[str], np.ndarray, list[str]]:
    """Split columns into spectral (numeric headers) and metadata.

    Convention: spectral columns are contiguous from the beginning, and their
    header names are the x-axis values (floats) serialized as strings.
    """
    if not columns:
        raise ValueError("Ownformat: empty table")

    wn: list[float] = []
    sep = 0
    for i, c in enumerate(columns):
        try:
            wn.append(float(c))
            sep = i + 1
        except (TypeError, ValueError):
            break

    if sep == 0:
        raise ValueError(
            "Ownformat: no spectral columns found. Expecting numeric header names at the beginning of the table."
        )

    spectral_cols = columns[:sep]
    meta_cols = columns[sep:]
    return spectral_cols, np.asarray(wn, dtype=float), meta_cols
