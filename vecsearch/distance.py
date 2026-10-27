"""Distance functions used by every index.

All functions return a *distance*, meaning a smaller value = more similar.
That way every index can just look for the k smallest values no matter
which metric is used:

    l2      -> euclidean distance            ||a - b||
    cosine  -> cosine distance               1 - (a.b / (|a| |b|))
    ip      -> negative inner product        -(a.b)

There is a single-pair version (a, b) and a batched version (q, X) of each.
The batched versions are fully vectorized with NumPy, which is the main
reason the pure-Python indexes are usable at all.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

SingleFn = Callable[[np.ndarray, np.ndarray], float]
BatchFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

METRICS = ("l2", "cosine", "ip")

_EPS = 1e-12


# ---------------------------------------------------------------- single pair

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean (L2) distance between two vectors.

    Time: O(d)   Space: O(d) for the temporary difference vector.
    """
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.dot(diff, diff)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 - cosine similarity. Range is [0, 2].

    Zero vectors are treated as having similarity 0 (distance 1).

    Time: O(d)   Space: O(1) extra.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < _EPS:
        return 1.0
    return 1.0 - float(np.dot(a, b)) / denom


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Negative dot product, so that a bigger dot product = smaller distance.

    Time: O(d)   Space: O(1) extra.
    """
    return -float(np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))


# ---------------------------------------------------------------- batched

def euclidean_batch(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """L2 distance from one query ``q`` (d,) to every row of ``X`` (n, d).

    Uses the expansion ||x - q||^2 = ||x||^2 - 2 x.q + ||q||^2 so the whole
    thing is one matrix-vector product instead of a python loop.

    Time: O(n * d)   Space: O(n) for the output.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    sq = np.einsum("ij,ij->i", X, X) - 2.0 * (X @ q) + np.dot(q, q)
    np.maximum(sq, 0.0, out=sq)  # tiny negatives from float rounding
    return np.sqrt(sq)


def cosine_batch(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cosine distance from ``q`` (d,) to every row of ``X`` (n, d).

    Time: O(n * d)   Space: O(n).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1) * np.linalg.norm(q)
    sims = np.where(norms > _EPS, (X @ q) / np.maximum(norms, _EPS), 0.0)
    return (1.0 - sims).astype(np.float32)


def dot_product_batch(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Negative dot product from ``q`` (d,) to every row of ``X`` (n, d).

    Time: O(n * d)   Space: O(n).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    return -(X @ np.asarray(q, dtype=np.float32))


def pairwise(Q: np.ndarray, X: np.ndarray, metric: str = "l2") -> np.ndarray:
    """Full distance matrix between queries ``Q`` (m, d) and data ``X`` (n, d).

    Used for computing ground truth in tests/benchmarks.

    Time: O(m * n * d)   Space: O(m * n).
    """
    Q = np.asarray(Q, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if metric == "l2":
        sq = (
            np.einsum("ij,ij->i", Q, Q)[:, None]
            - 2.0 * (Q @ X.T)
            + np.einsum("ij,ij->i", X, X)[None, :]
        )
        return np.sqrt(np.maximum(sq, 0.0))
    if metric == "cosine":
        qn = np.linalg.norm(Q, axis=1, keepdims=True)
        xn = np.linalg.norm(X, axis=1, keepdims=True)
        denom = np.maximum(qn @ xn.T, _EPS)
        return 1.0 - (Q @ X.T) / denom
    if metric == "ip":
        return -(Q @ X.T)
    raise ValueError(f"unknown metric {metric!r}, expected one of {METRICS}")


_SINGLE: dict[str, SingleFn] = {"l2": euclidean, "cosine": cosine, "ip": dot_product}
_BATCH: dict[str, BatchFn] = {"l2": euclidean_batch, "cosine": cosine_batch, "ip": dot_product_batch}


def get_distance(metric: str) -> SingleFn:
    """Return the single-pair distance function for ``metric``. O(1)."""
    try:
        return _SINGLE[metric]
    except KeyError:
        raise ValueError(f"unknown metric {metric!r}, expected one of {METRICS}") from None


def get_batch_distance(metric: str) -> BatchFn:
    """Return the batched (one-to-many) distance function for ``metric``. O(1)."""
    try:
        return _BATCH[metric]
    except KeyError:
        raise ValueError(f"unknown metric {metric!r}, expected one of {METRICS}") from None
