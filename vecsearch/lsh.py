"""Locality Sensitive Hashing with random hyperplanes (SimHash).

Idea: pick ``n_bits`` random hyperplanes. For a vector x, each hyperplane
gives one bit (which side of the plane x is on). The bits together form a
hash code. Two vectors with a small angle between them agree on most bits,
so they tend to fall into the same bucket.

    P[bit is equal] = 1 - angle(a, b) / pi

One table with many bits is very selective (few false positives, but misses
a lot), so we use ``n_tables`` independent tables and take the union of the
buckets the query lands in. Candidates are then re-ranked with the exact
distance.

Multi-probe: optionally also look at the buckets whose code differs from the
query code in exactly one bit. This boosts recall without adding tables.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseIndex
from .heap import k_smallest


class LSHIndex(BaseIndex):
    """Random hyperplane LSH index with multiple hash tables.

    Let L = n_tables, K = n_bits, C = number of candidates collected.

    add:    O(m * L * K * d)   (one matrix product per batch)
    search: O(L * K * d) to hash + O(L * P) bucket lookups + O(C * d) re-rank
            + O(C log k) top-k,  where P = 1 + K if multi-probe else 1
    space:  O(n * d) vectors + O(n * L) bucket entries + O(L * K * d) planes

    Args:
        dim: vector dimension.
        metric: distance used for re-ranking ("l2", "cosine", "ip").
        n_tables: number of independent hash tables (L).
        n_bits: hyperplanes per table (K). Code fits in a python int.
        multi_probe: also probe buckets at Hamming distance 1.
        center: subtract the mean of the first added batch before hashing.
            Hyperplanes go through the origin, so for data that is all in one
            region (e.g. SIFT, all values >= 0) centering makes the bits useful.
        seed: random seed for the hyperplanes.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        n_tables: int = 10,
        n_bits: int = 12,
        multi_probe: bool = True,
        center: bool = True,
        seed: int = 42,
    ) -> None:
        """Draw the random hyperplanes and create empty tables. O(L * K * d) time and space."""
        super().__init__(dim, metric)
        if n_tables <= 0 or n_bits <= 0:
            raise ValueError("n_tables and n_bits must be positive")
        if n_bits > 62:
            raise ValueError("n_bits must be <= 62 so the code fits in an int64")
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.multi_probe = multi_probe
        self.center = center
        self.seed = seed

        rng = np.random.default_rng(seed)
        # planes[t] has shape (n_bits, dim)
        self._planes = rng.normal(size=(n_tables, n_bits, dim)).astype(np.float32)
        self._powers = (1 << np.arange(n_bits, dtype=np.int64)).astype(np.int64)
        self._mean: np.ndarray | None = None
        # one dict per table: hash code -> list of internal positions
        self._tables: list[dict[int, list[int]]] = [dict() for _ in range(n_tables)]

    # ---------------------------------------------------------------- hashing
    def _hash(self, X: np.ndarray) -> np.ndarray:
        """Hash codes for every row of X in every table.

        Returns an int64 array of shape (n_tables, m).
        Time: O(m * L * K * d)   Space: O(m * L * K).
        """
        if self._mean is not None:
            X = X - self._mean
        # (L, K, d) x (d, m) -> (L, K, m)
        bits = np.einsum("lkd,md->lkm", self._planes, X) > 0
        return np.einsum("lkm,k->lm", bits.astype(np.int64), self._powers)

    def _probe_codes(self, code: int) -> list[int]:
        """The query bucket plus (if multi-probe) its K one-bit neighbours. O(K)."""
        if not self.multi_probe:
            return [code]
        return [code] + [code ^ (1 << b) for b in range(self.n_bits)]

    # ---------------------------------------------------------------- API
    def add(self, vectors: np.ndarray, ids: np.ndarray | list[int] | None = None) -> None:
        """Hash and insert vectors into every table.

        Time: O(m * L * K * d)   Space: O(m * (d + L)).
        """
        X = self._check_vectors(vectors)
        if X.shape[0] == 0:
            return
        new_ids = self._make_ids(X.shape[0], ids)
        if self.center and self._mean is None:
            self._mean = X.mean(axis=0).astype(np.float32)

        start = len(self._ids)
        codes = self._hash(X)
        for t in range(self.n_tables):
            table = self._tables[t]
            for j, code in enumerate(codes[t].tolist()):
                bucket = table.get(code)
                if bucket is None:
                    table[code] = [start + j]
                else:
                    bucket.append(start + j)

        self._store.append(X)
        self._ids.extend(new_ids)

    def candidates(self, query: np.ndarray) -> np.ndarray:
        """Internal positions of all vectors sharing a probed bucket with ``query``.

        Time: O(L * K * d + L * P + C)   Space: O(C).
        """
        q = self._check_query(query)
        codes = self._hash(q[None, :])[:, 0].tolist()
        found: set[int] = set()
        for t, code in enumerate(codes):
            table = self._tables[t]
            for c in self._probe_codes(code):
                bucket = table.get(c)
                if bucket:
                    found.update(bucket)
        return np.fromiter(found, dtype=np.int64, count=len(found))

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Approximate k nearest neighbours.

        Collect candidates from the buckets, compute exact distances to just
        those, then pick the k best with a MaxHeap.

        Time: O(L * K * d + C * d + C log k)   Space: O(C).
        """
        q = self._check_query(query)
        if len(self) == 0 or k <= 0:
            return self._empty_result()
        cand = self.candidates(q)
        if cand.size == 0:
            return self._empty_result()
        dists = self._batch_dist(q, self.vectors[cand])
        best = k_smallest(dists, cand, k)
        ids_arr = np.asarray(self._ids, dtype=np.int64)
        ids = ids_arr[[p for _, p in best]]
        return ids, np.array([d for d, _ in best], dtype=np.float32)

    def bucket_stats(self) -> dict[str, float]:
        """Average / max bucket size over all tables (useful for tuning). O(L * B)."""
        sizes = [len(b) for table in self._tables for b in table.values()]
        if not sizes:
            return {"buckets": 0, "avg_size": 0.0, "max_size": 0}
        return {"buckets": len(sizes), "avg_size": float(np.mean(sizes)), "max_size": int(max(sizes))}

    # ---------------------------------------------------------------- persistence
    def _get_state(self) -> dict[str, Any]:
        """Params, arrays (vectors, ids, planes, mean) and bucket tables. O(n * d) for the copy."""
        arrays = {"vectors": self.vectors.copy(), "ids": self.ids, "planes": self._planes}
        if self._mean is not None:
            arrays["mean"] = self._mean
        return {
            "params": {
                "dim": self.dim,
                "metric": self.metric,
                "n_tables": self.n_tables,
                "n_bits": self.n_bits,
                "multi_probe": self.multi_probe,
                "center": self.center,
                "seed": self.seed,
            },
            "arrays": arrays,
            "extra": {"tables": self._tables},
        }

    def _set_state(self, state: dict[str, Any]) -> None:
        """Restore vectors, planes and buckets saved by ``_get_state``. O(n * d)."""
        arrays = state["arrays"]
        self._store.append(arrays["vectors"].astype(np.float32))
        self._ids = [int(i) for i in arrays["ids"]]
        self._planes = arrays["planes"].astype(np.float32)
        self._mean = arrays["mean"].astype(np.float32) if "mean" in arrays else None
        self._tables = state["extra"]["tables"]
