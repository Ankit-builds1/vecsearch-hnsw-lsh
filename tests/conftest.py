import sys
from pathlib import Path

import numpy as np
import pytest

# make "import vecsearch" work without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecsearch.distance import pairwise  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def make_clustered(n: int, dim: int, n_clusters: int = 20, seed: int = 0) -> np.ndarray:
    """Gaussian blobs - closer to real data than uniform noise."""
    r = np.random.default_rng(seed)
    centers = r.normal(0, 10, size=(n_clusters, dim))
    labels = r.integers(0, n_clusters, size=n)
    return (centers[labels] + r.normal(0, 1.5, size=(n, dim))).astype(np.float32)


def recall_at_k(index, data: np.ndarray, queries: np.ndarray, k: int, metric: str = "l2") -> float:
    """Average fraction of the true top-k that the index found."""
    truth = np.argsort(pairwise(queries, data, metric), axis=1)[:, :k]
    hits = 0
    for q, t in zip(queries, truth):
        ids, _ = index.search(q, k)
        hits += len(set(ids.tolist()) & set(t.tolist()))
    return hits / (len(queries) * k)
