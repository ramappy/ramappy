"""Ownformat IO implementation.

This subpackage contains the implementation for the "ownformat" import/export format.
The public (registered) entry points remain in `ramappy.io.custom` for backward
compatibility.

The core functions here are intentionally small and testable.
"""

from .reader import read_ownformat
from .writer import write_ownformat

__all__ = ["read_ownformat", "write_ownformat"]
