"""Exact (brute force) nearest neighbour search.

This is the simplest possible index: keep every vector and compare the
query against all of them. It is slow for big data but always correct,
so it is used as the ground truth for measuring recall of LSH and HNSW.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseIndex
from .heap import k_smallest


class BruteForceIndex(BaseIndex):
    """Flat index that scans all n vectors for every query.

    add:    O(m * d) amortized
    search: O(n * d + n)  with argpartition, or O(n * d + n log k) with the heap
    space:  O(n * d)

    Args:
        dim: vector dimension.
        metric: "l2", "cosine" or "ip".
        use_heap: if True, top-k selection uses our own MaxHeap
            (``heap.k_smallest``) instead of ``np.argpartition``. The result is
            the same; argpartition is just faster because it runs in C.
    """

    def __init__(self, dim: int, metric: str = "l2", use_heap: bool = False) -> None:
        super().__init__(dim, metric)
        self.use_heap = use_heap

    def add(self, vectors: np.ndarray, ids: np.ndarray | list[int] | None = None) -> None:
        """Append vectors to the flat array.

        Time: O(m * d) amortized   Space: O(m * d).
        """
        X = self._check_vectors(vectors)
        if X.shape[0] == 0:
            return
        new_ids = self._make_ids(X.shape[0], ids)
        self._store.append(X)
        self._ids.extend(new_ids)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Exact k nearest neighbours.

        Time: O(n * d) for distances + O(n) (argpartition) or O(n log k) (heap)
        Space: O(n) for the distance array.
        """
        q = self._check_query(query)
        n = len(self)
        if n == 0 or k <= 0:
            return self._empty_result()
        k = min(k, n)
        dists = self._batch_dist(q, self.vectors)

        if self.use_heap:
            best = k_smallest(dists, np.arange(n), k)
            pos = np.array([p for _, p in best], dtype=np.int64)
        else:
            if k < n:
                pos = np.argpartition(dists, k - 1)[:k]
            else:
                pos = np.arange(n)
            pos = pos[np.argsort(dists[pos], kind="stable")]

        ids = np.asarray(self._ids, dtype=np.int64)[pos]
        return ids, dists[pos].astype(np.float32)

    def _get_state(self) -> dict[str, Any]:
        return {
            "params": {"dim": self.dim, "metric": self.metric, "use_heap": self.use_heap},
            "arrays": {"vectors": self.vectors.copy(), "ids": self.ids},
            "extra": {},
        }

    def _set_state(self, state: dict[str, Any]) -> None:
        arrays = state["arrays"]
        self._store.append(arrays["vectors"].astype(np.float32))
        self._ids = [int(i) for i in arrays["ids"]]
