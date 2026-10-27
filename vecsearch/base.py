"""Common base class for every index in the project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .distance import METRICS, get_batch_distance, get_distance


class VectorStore:
    """Growable 2-D float32 array (like a dynamic array / ArrayList).

    When it runs out of room the capacity is doubled, so appending m vectors
    costs O(m * d) amortized instead of re-copying everything on every add.

    Space: O(capacity * d), capacity < 2n.
    """

    def __init__(self, dim: int, capacity: int = 1024) -> None:
        """Allocate an empty store. Time/Space: O(capacity * d)."""
        self.dim = dim
        self._buf = np.zeros((max(1, capacity), dim), dtype=np.float32)
        self._n = 0

    def append(self, X: np.ndarray) -> None:
        """Append rows of ``X`` (m, d).

        Time: O(m * d) amortized (O(n * d) when a resize happens).
        """
        m = X.shape[0]
        need = self._n + m
        if need > self._buf.shape[0]:
            new_cap = self._buf.shape[0]
            while new_cap < need:
                new_cap *= 2
            new_buf = np.zeros((new_cap, self.dim), dtype=np.float32)
            new_buf[: self._n] = self._buf[: self._n]
            self._buf = new_buf
        self._buf[self._n : need] = X
        self._n = need

    @property
    def data(self) -> np.ndarray:
        """View of the filled rows, shape (n, d). O(1), no copy."""
        return self._buf[: self._n]

    def __len__(self) -> int:
        """Number of stored rows. O(1)."""
        return self._n


class BaseIndex(ABC):
    """Abstract vector index. Stores vectors in a VectorStore: O(n * d) space.

    Subclasses must implement ``add`` and ``search`` plus the two state hooks
    used by persistence. Results are always returned as two arrays
    ``(ids, distances)`` sorted by increasing distance.
    """

    def __init__(self, dim: int, metric: str = "l2") -> None:
        """Validate parameters and set up storage. O(1)."""
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}, expected one of {METRICS}")
        self.dim = dim
        self.metric = metric
        self._dist = get_distance(metric)
        self._batch_dist = get_batch_distance(metric)
        self._store = VectorStore(dim)
        self._ids: list[int] = []  # internal position -> external id

    # ---------------------------------------------------------------- helpers
    def _check_vectors(self, X: np.ndarray) -> np.ndarray:
        """Make sure ``X`` is a float32 matrix with the right number of columns. O(m*d)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dimension {self.dim}, got shape {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("vectors contain NaN or inf")
        return X

    def _check_query(self, q: np.ndarray) -> np.ndarray:
        """Make sure ``q`` is one vector of the right dimension. O(d)."""
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"expected query of dimension {self.dim}, got {q.shape[0]}")
        return q

    def _make_ids(self, m: int, ids: np.ndarray | list[int] | None) -> list[int]:
        """Default ids are 0, 1, 2, ... continuing from the current size. O(m)."""
        if ids is None:
            start = len(self._ids)
            return list(range(start, start + m))
        out = [int(i) for i in ids]
        if len(out) != m:
            raise ValueError("ids must have the same length as vectors")
        return out

    @staticmethod
    def _empty_result() -> tuple[np.ndarray, np.ndarray]:
        """Empty (ids, distances) pair returned for an empty index or k <= 0. O(1)."""
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    @property
    def vectors(self) -> np.ndarray:
        """All stored vectors (n, d), in insertion order. O(1) view."""
        return self._store.data

    @property
    def ids(self) -> np.ndarray:
        """External ids in insertion order. O(n)."""
        return np.asarray(self._ids, dtype=np.int64)

    # ---------------------------------------------------------------- API
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: np.ndarray | list[int] | None = None) -> None:
        """Insert vectors (m, d) with optional external integer ids.

        Cost depends on the index: O(m * d) for brute force up to
        O(m * log n * ef_construction * M * d) for HNSW.
        """

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(ids, distances)`` of the k nearest neighbours of ``query``.

        Cost depends on the index: O(n * d) for brute force, much less for LSH / HNSW.

        If the index has fewer than k items, fewer results are returned.
        """

    def search_batch(self, queries: np.ndarray, k: int = 10) -> list[tuple[np.ndarray, np.ndarray]]:
        """Run ``search`` for each row of ``queries``.

        Time: m * cost(search)   Space: O(m * k).
        """
        Q = np.asarray(queries, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[None, :]
        return [self.search(q, k) for q in Q]

    def __len__(self) -> int:
        """Number of stored vectors. O(1)."""
        return len(self._ids)

    def __repr__(self) -> str:
        """Short description, e.g. ``HNSWIndex(dim=128, metric='l2', size=10000)``. O(1)."""
        return f"{type(self).__name__}(dim={self.dim}, metric={self.metric!r}, size={len(self)})"

    # ---------------------------------------------------------------- persistence
    @abstractmethod
    def _get_state(self) -> dict[str, Any]:
        """Return everything needed to rebuild the index (arrays + plain python). O(n * d)."""

    @abstractmethod
    def _set_state(self, state: dict[str, Any]) -> None:
        """Restore the index from a dict produced by ``_get_state``. O(n * d)."""

    def save(self, path: str | Path) -> Path:
        """Save the index to a folder. See ``vecsearch.persistence``. O(n * d + graph size)."""
        from .persistence import save_index

        return save_index(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseIndex":
        """Load an index previously written by ``save``. O(n * d + graph size)."""
        from .persistence import load_index

        index = load_index(path)
        if not isinstance(index, cls):
            raise TypeError(f"file contains a {type(index).__name__}, not a {cls.__name__}")
        return index
