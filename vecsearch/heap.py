"""Binary heaps written by hand (no ``heapq``).

The heap is stored in a plain python list, using the usual array layout:

    parent(i) = (i - 1) // 2
    left(i)   = 2 * i + 1
    right(i)  = 2 * i + 2

Every entry is a ``(key, item)`` pair. ``MinHeap`` keeps the smallest key on
top and ``MaxHeap`` keeps the largest key on top. Internally ``MaxHeap`` is
just a ``MinHeap`` that compares with the opposite sign, so both share the
same sift-up / sift-down code.

These are used for:
  * top-k selection (keep a MaxHeap of size k, pop the worst when it overflows)
  * HNSW beam search (MinHeap of candidates + MaxHeap of current results)
"""

from __future__ import annotations

from typing import Any, Generic, Iterable, Iterator, TypeVar

import numpy as np

T = TypeVar("T")


class _BinaryHeap(Generic[T]):
    """Array-backed binary heap. Smaller ``_sign * key`` goes on top.

    Space: O(n) for n stored entries.
    """

    _sign: float = 1.0

    def __init__(self, items: Iterable[tuple[float, T]] | None = None) -> None:
        """Create a heap, optionally from existing ``(key, item)`` pairs.

        Building from items uses bottom-up heapify.
        Time: O(n)   Space: O(n).
        """
        self._data: list[tuple[float, T]] = []
        if items is not None:
            self._data = [(float(k), v) for k, v in items]
            for i in range(len(self._data) // 2 - 1, -1, -1):
                self._sift_down(i)

    # ------------------------------------------------------------ helpers
    def _less(self, i: int, j: int) -> bool:
        """True if entry i should sit above entry j. O(1)."""
        return self._sign * self._data[i][0] < self._sign * self._data[j][0]

    def _swap(self, i: int, j: int) -> None:
        """Swap two entries. O(1)."""
        self._data[i], self._data[j] = self._data[j], self._data[i]

    def _sift_up(self, i: int) -> None:
        """Move entry i up until the heap property holds.

        Time: O(log n)   Space: O(1).
        """
        while i > 0:
            parent = (i - 1) // 2
            if self._less(i, parent):
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        """Move entry i down until the heap property holds.

        Time: O(log n)   Space: O(1).
        """
        n = len(self._data)
        while True:
            left = 2 * i + 1
            right = left + 1
            best = i
            if left < n and self._less(left, best):
                best = left
            if right < n and self._less(right, best):
                best = right
            if best == i:
                return
            self._swap(i, best)
            i = best

    # ------------------------------------------------------------ public API
    def push(self, key: float, item: T) -> None:
        """Insert ``item`` with priority ``key``.

        Time: O(log n)   Space: O(1) amortized.
        """
        self._data.append((float(key), item))
        self._sift_up(len(self._data) - 1)

    def pop(self) -> tuple[float, T]:
        """Remove and return the top ``(key, item)``.

        Time: O(log n)   Space: O(1).

        Raises:
            IndexError: if the heap is empty.
        """
        if not self._data:
            raise IndexError("pop from empty heap")
        top = self._data[0]
        last = self._data.pop()
        if self._data:
            self._data[0] = last
            self._sift_down(0)
        return top

    def peek(self) -> tuple[float, T]:
        """Return the top ``(key, item)`` without removing it.

        Time: O(1)   Space: O(1).

        Raises:
            IndexError: if the heap is empty.
        """
        if not self._data:
            raise IndexError("peek from empty heap")
        return self._data[0]

    def pushpop(self, key: float, item: T) -> tuple[float, T]:
        """Push then pop in one step (faster than doing both separately).

        If the new entry would itself be the top, it is returned directly
        and the heap is not touched.

        Time: O(log n)   Space: O(1).
        """
        key = float(key)
        if self._data and self._sign * self._data[0][0] < self._sign * key:
            top = self._data[0]
            self._data[0] = (key, item)
            self._sift_down(0)
            return top
        return key, item

    def to_sorted_list(self) -> list[tuple[float, T]]:
        """Return all entries ordered from top to bottom (heap is unchanged).

        Time: O(n log n)   Space: O(n).
        """
        return sorted(self._data, key=lambda e: self._sign * e[0])

    def __len__(self) -> int:
        """Number of entries. O(1)."""
        return len(self._data)

    def __bool__(self) -> bool:
        """True if not empty. O(1)."""
        return bool(self._data)

    def __iter__(self) -> Iterator[tuple[float, T]]:
        """Iterate in internal (array) order, NOT sorted order. O(n)."""
        return iter(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={len(self)})"


class MinHeap(_BinaryHeap[T]):
    """Binary min-heap: ``peek()`` / ``pop()`` give the smallest key.

    push O(log n), pop O(log n), peek O(1), space O(n).
    """

    _sign = 1.0


class MaxHeap(_BinaryHeap[T]):
    """Binary max-heap: ``peek()`` / ``pop()`` give the largest key.

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
    heap: MaxHeap[Any] = MaxHeap()
    for i in range(len(keys)):
        key = float(keys[i])
        if len(heap) < k:
            heap.push(key, items[i])
        elif key < heap.peek()[0]:
            heap.pushpop(key, items[i])
    return list(reversed(heap.to_sorted_list()))
