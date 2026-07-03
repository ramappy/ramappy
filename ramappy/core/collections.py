"""Typed, ordered, change-tracked mapping used throughout ramappy."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import TypedDict, TypeVar

T = TypeVar("T")


class Changes(TypedDict):
    """Change set returned by :meth:`EntityTransaction.commit <ramappy.core.collections.EntityTransaction.commit>`."""

    new: set[str]
    changed: set[str]
    deleted: set[str]


class EntityTransaction[T]:
    """Tracks changes made to an :class:`OrderedEntityMap <ramappy.core.collections.OrderedEntityMap>` within a context block.

    Obtained by calling :meth:`OrderedEntityMap.begin_transaction <ramappy.core.collections.OrderedEntityMap.begin_transaction>`. The
    transaction object records which keys were added, modified, and deleted
    during the block.

    On clean exit the changes are **committed** (they remain in the map).
    On exception the changes are **rolled back** automatically.

    Parameters
    ----------
    owner : OrderedEntityMap
        The map this transaction is attached to.

    Examples
    --------
    >>> m: OrderedEntityMap[str] = OrderedEntityMap()
    >>> with m.begin_transaction() as tx:
    ...     m["a"] = "hello"
    >>> tx.added
    {'a': 'hello'}

    >>> m2: OrderedEntityMap[str] = OrderedEntityMap({"a": "original"})
    >>> try:
    ...     with m2.begin_transaction():
    ...         m2["a"] = "changed"
    ...         raise RuntimeError("oops")
    ... except RuntimeError:
    ...     pass
    >>> m2["a"]  # rolled back
    'original'
    """

    def __init__(self, owner: OrderedEntityMap[T]) -> None:
        self._owner = owner
        self._snapshot: dict[str, T] = dict(owner._data)
        self._snapshot_order: list[str] = list(owner._order)

        # Change sets, populated on commit
        self._added: dict[str, T] = {}
        self._modified: dict[str, T] = {}
        self._deleted: set[str] = set()
        self._committed = False

        # Context manager protocol

    def commit(self) -> Changes:
        """Manually commit the transaction and return the changes.

        This performs the same logic as a clean context manager exit.
        """
        if self._committed:
            return {
                "new": set(self._added.keys()),
                "changed": set(self._modified.keys()),
                "deleted": self._deleted,
            }

        # Compute change sets
        for key, value in self._owner._data.items():
            if key not in self._snapshot:
                self._added[key] = value
            elif self._snapshot[key] is not value:
                self._modified[key] = value
        for key in self._snapshot:
            if key not in self._owner._data:
                self._deleted.add(key)

        self._committed = True
        return {
            "new": set(self._added.keys()),
            "changed": set(self._modified.keys()),
            "deleted": self._deleted,
        }

    def rollback(self) -> None:
        """Manually roll back the transaction.

        This performs the same logic as an exception during a context block.
        """
        self._owner._data = self._snapshot
        self._owner._order = self._snapshot_order
        self._committed = False

    def __enter__(self) -> EntityTransaction[T]:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        if exc_type is not None:
            # Exception: roll back
            self.rollback()
        elif not self._committed:
            # Clean exit: commit
            self.commit()

    @property
    def added(self) -> dict[str, T]:
        """Keys/values that were **inserted** during the transaction."""
        return dict(self._added)

    @property
    def modified(self) -> dict[str, T]:
        """Keys whose values were **changed** during the transaction."""
        return dict(self._modified)

    @property
    def deleted(self) -> set[str]:
        """Keys that were **removed** during the transaction."""
        return set(self._deleted)

    @property
    def is_empty(self) -> bool:
        """``True`` if nothing changed during the transaction."""
        return not (self._added or self._modified or self._deleted)


class OrderedEntityMap[T](MutableMapping[str, T]):
    """Ordered, change-tracked mapping that replaces ``JournalOrderedDict``.

    Maintains insertion order (like a regular Python ``dict``). Change
    tracking is done via explicit :meth:`ramappy.core.collections.OrderedEntityMap.begin_transaction` context managers.

    Parameters
    ----------
    items : dict[str, T] | None
        Optional initial contents.

    Examples
    --------
    >>> masks: OrderedEntityMap[Mask] = OrderedEntityMap()
    >>> with masks.begin_transaction() as tx:
    ...     masks["m1"] = Mask(...)
    ...     masks["m2"] = Mask(...)
    >>> list(tx.added.keys())
    ['m1', 'm2']
    """

    def __init__(self, items: dict[str, T] | None = None) -> None:
        self._data: dict[str, T] = {}
        self._order: list[str] = []
        if items:
            for k, v in items.items():
                self[k] = v

    def __getitem__(self, key: str) -> T:
        return self._data[key]

    def __setitem__(self, key: str, value: T) -> None:
        if key not in self._data:
            self._order.append(key)
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]
        self._order.remove(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        inner = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"{type(self).__name__}({{{inner}}})"

    def set_order(self, keys: list[str]) -> None:
        """Reorder the map according to *keys*.

        Parameters
        ----------
        keys : list[str]
            Must contain exactly the same keys as the current map (no extras,
            no missing).

        Raises
        ------
        ValueError
            If *keys* does not match the current key set.
        """
        if set(keys) != set(self._order):
            raise ValueError(f"Key mismatch: provided {set(keys)}, have {set(self._order)}")
        self._order = list(keys)

    def begin_transaction(self) -> EntityTransaction[T]:
        """Return an :class:`ramappy.core.collections.EntityTransaction` context manager.

        Use as::

            with entity_map.begin_transaction() as tx:
                entity_map["key"] = value
            # tx.added, tx.modified, tx.deleted available here
        """
        return EntityTransaction(self)

    @property
    def ordered_keys(self) -> list[str]:
        """Backward-compatibility: return the internal order list."""
        return self._order

    @ordered_keys.setter
    def ordered_keys(self, value: list[str]) -> None:
        """Backward-compatibility: set the internal order list."""
        self.set_order(value)

    def get_nth_item(self, n: int) -> T:
        """Return the *n*-th item in insertion order (0-indexed)."""
        return self._data[self._order[n]]
