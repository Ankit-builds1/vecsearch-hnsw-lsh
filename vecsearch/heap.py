"""Binary heaps written by hand (no ``heapq``).

The heap is stored in a plain python list, using the usual array layout:

    parent(i) = (i - 1) // 2
    left(i)   = 2 * i + 1
    right(i)  = 2 * i + 2

Every entry is a ``(key, item)`` pair. ``MinHeap`` keeps the smallest key on
top and ``MaxHeap`` keeps the largest key on top.

Implementation trick: both heaps share one min-heap implementation. A
MaxHeap stores ``-key`` internally and flips the sign back when returning,
so the sift loops are just plain ``<`` comparisons (this matters a lot for
speed in Python, since HNSW does millions of heap operations).

These are used for:
  * top-k selection (keep a MaxHeap of size k, pop the worst when it overflows)
  * HNSW beam search (MinHeap of candidates + MaxHeap of current results)
"""

from __future__ import annotations

from typing import Any, Generic, Iterable, Iterator, TypeVar

import numpy as np

T = TypeVar("T")


class _BinaryHeap(Generic[T]):
    """Array-backed binary min-heap over ``_sign * key``.

    Space: O(n) for n stored entries.
    """

    _sign: float = 1.0

    def __init__(self, items: Iterable[tuple[float, T]] | None = None) -> None:
        """Create a heap, optionally from existing ``(key, item)`` pairs.

        Building from items uses bottom-up heapify.
        Time: O(n)   Space: O(n).
        """
        s = self._sign
        self._data: list[tuple[float, T]] = []
        if items is not None:
            self._data = [(s * float(k), v) for k, v in items]
            for i in range(len(self._data) // 2 - 1, -1, -1):
                self._sift_down(i)

    # ------------------------------------------------------------ helpers
    def _sift_up(self, i: int) -> None:
        """Move entry i up until its parent is not bigger.

        Time: O(log n)   Space: O(1).
        """
        data = self._data
        entry = data[i]
        key = entry[0]
        while i > 0:
            parent = (i - 1) >> 1
            if key < data[parent][0]:
                data[i] = data[parent]  # shift parent down, place entry at the end
                i = parent
            else:
                break
        data[i] = entry

    def _sift_down(self, i: int) -> None:
        """Move entry i down until both children are not smaller.

        Time: O(log n)   Space: O(1).
        """
        data = self._data
        n = len(data)
        entry = data[i]
        key = entry[0]
        while True:
            child = 2 * i + 1
            if child >= n:
                break
            right = child + 1
            if right < n and data[right][0] < data[child][0]:
                child = right
            if data[child][0] < key:
                data[i] = data[child]
                i = child
            else:
                break
        data[i] = entry

    # ------------------------------------------------------------ public API
    def push(self, key: float, item: T) -> None:
        """Insert ``item`` with priority ``key``.

        Time: O(log n)   Space: O(1) amortized.
        """
        self._data.append((self._sign * key, item))
        self._sift_up(len(self._data) - 1)

    def pop(self) -> tuple[float, T]:
        """Remove and return the top ``(key, item)``.

        Time: O(log n)   Space: O(1).

        Raises:
            IndexError: if the heap is empty.
        """
        data = self._data
        if not data:
            raise IndexError("pop from empty heap")
        last = data.pop()
        if data:
            top = data[0]
            data[0] = last
            self._sift_down(0)
        else:
            top = last
        return self._sign * top[0], top[1]

    def peek(self) -> tuple[float, T]:
        """Return the top ``(key, item)`` without removing it.

        Time: O(1)   Space: O(1).

        Raises:
            IndexError: if the heap is empty.
        """
        if not self._data:
            raise IndexError("peek from empty heap")
        k, v = self._data[0]
        return self._sign * k, v

    def peek_key(self) -> float:
        """Only the top key (slightly faster than ``peek()[0]``). O(1)."""
        return self._sign * self._data[0][0]

    def pushpop(self, key: float, item: T) -> tuple[float, T]:
        """Push then pop in one step (faster than doing both separately).

        If the new entry would itself be the top, it is returned directly
        and the heap is not touched.

        Time: O(log n)   Space: O(1).
        """
        s = self._sign
        sk = s * float(key)
        data = self._data
        if data and data[0][0] < sk:
            top = data[0]
            data[0] = (sk, item)
            self._sift_down(0)
            return s * top[0], top[1]
        return float(key), item

    def to_sorted_list(self) -> list[tuple[float, T]]:
        """Return all entries ordered from top to bottom (heap is unchanged).

        Time: O(n log n)   Space: O(n).
        """
        s = self._sign
        return [(s * k, v) for k, v in sorted(self._data, key=lambda e: e[0])]

    def __len__(self) -> int:
        """Number of entries. O(1)."""
        return len(self._data)

    def __bool__(self) -> bool:
        """True if not empty. O(1)."""
        return bool(self._data)

    def __iter__(self) -> Iterator[tuple[float, T]]:
        """Iterate in internal (array) order, NOT sorted order. O(n)."""
        s = self._sign
        return ((s * k, v) for k, v in self._data)

    def __repr__(self) -> str:
        """Short description like ``MinHeap(size=3)``. O(1)."""
        return f"{type(self).__name__}(size={len(self)})"


class MinHeap(_BinaryHeap[T]):
    """Binary min-heap: ``peek()`` / ``pop()`` give the smallest key.

    push O(log n), pop O(log n), peek O(1), space O(n).
    """

    _sign = 1.0


class MaxHeap(_BinaryHeap[T]):
    """Binary max-heap: ``peek()`` / ``pop()`` give the largest key.

    Stores negated keys in a min-heap internally.
    push O(log n), pop O(log n), peek O(1), space O(n).
    """

    _sign = -1.0


def k_smallest(keys: np.ndarray | list[float], items: Any, k: int) -> list[tuple[float, Any]]:
    """Return the ``k`` pairs with the smallest keys, sorted ascending.

    Keeps a MaxHeap of size k: the top is the *worst* of the current best k,
    so a new key only gets in if it beats that one.

    Time: O(n log k)   Space: O(k).

    Args:
        keys: n distances.
        items: n matching items (ids). Anything indexable.
        k: how many to keep. ``k <= 0`` gives an empty list.
    """
    if k <= 0:
        return []
    keys = keys.tolist() if isinstance(keys, np.ndarray) else list(keys)
    heap: MaxHeap[Any] = MaxHeap()
    for i, key in enumerate(keys):
        if len(heap) < k:
            heap.push(key, items[i])
        elif key < heap.peek_key():
            heap.pushpop(key, items[i])
    return list(reversed(heap.to_sorted_list()))
