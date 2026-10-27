import numpy as np
import pytest

from vecsearch import BruteForceIndex
from vecsearch.base import VectorStore
from vecsearch.distance import METRICS, pairwise


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("use_heap", [False, True])
def test_exact_results(metric, use_heap, rng):
    X = rng.normal(size=(300, 16)).astype(np.float32)
    Q = rng.normal(size=(10, 16)).astype(np.float32)
    idx = BruteForceIndex(16, metric=metric, use_heap=use_heap)
    idx.add(X)
    truth = np.argsort(pairwise(Q, X, metric), axis=1)[:, :5]
    for q, t in zip(Q, truth):
        ids, dists = idx.search(q, 5)
        assert ids.tolist() == t.tolist()
        assert np.all(np.diff(dists) >= -1e-6)  # sorted ascending


def test_finds_itself(rng):
    X = rng.normal(size=(100, 8))
    idx = BruteForceIndex(8)
    idx.add(X)
    ids, dists = idx.search(X[37], 1)
    assert ids[0] == 37
    assert dists[0] == pytest.approx(0.0, abs=1e-3)


def test_empty_index():
    idx = BruteForceIndex(4)
    ids, dists = idx.search(np.ones(4), 5)
    assert len(ids) == 0 and len(dists) == 0
    assert len(idx) == 0


def test_k_larger_than_n(rng):
    idx = BruteForceIndex(3)
    idx.add(rng.normal(size=(4, 3)))
    ids, _ = idx.search(np.zeros(3), 10)
    assert sorted(ids.tolist()) == [0, 1, 2, 3]


def test_k_zero(rng):
    idx = BruteForceIndex(3)
    idx.add(rng.normal(size=(4, 3)))
    assert len(idx.search(np.zeros(3), 0)[0]) == 0


def test_custom_ids_and_incremental_add(rng):
    idx = BruteForceIndex(2)
    idx.add(np.array([[0, 0], [10, 10]]), ids=[100, 200])
    idx.add(np.array([[1, 1]]))  # default id continues from current size -> 2
    assert len(idx) == 3
    ids, _ = idx.search(np.array([0.9, 0.9]), 3)
    assert ids.tolist() == [2, 100, 200]


def test_bad_shapes():
    idx = BruteForceIndex(4)
    with pytest.raises(ValueError):
        idx.add(np.ones((3, 5)))
    with pytest.raises(ValueError):
        idx.search(np.ones(3), 1)
    with pytest.raises(ValueError):
        idx.add(np.ones((2, 4)), ids=[1])
    with pytest.raises(ValueError):
        idx.add(np.array([[np.nan, 0, 0, 0]]))


def test_bad_constructor_args():
    with pytest.raises(ValueError):
        BruteForceIndex(0)
    with pytest.raises(ValueError):
        BruteForceIndex(4, metric="hamming")


def test_single_vector_add():
    idx = BruteForceIndex(3)
    idx.add(np.array([1.0, 2.0, 3.0]))
    assert len(idx) == 1


def test_search_batch(rng):
    X = rng.normal(size=(50, 4))
    idx = BruteForceIndex(4)
    idx.add(X)
    res = idx.search_batch(X[:3], 1)
    assert [r[0][0] for r in res] == [0, 1, 2]


def test_vector_store_grows():
    s = VectorStore(dim=2, capacity=1)
    for i in range(100):
        s.append(np.array([[i, i]], dtype=np.float32))
    assert len(s) == 100
    assert s.data[57].tolist() == [57.0, 57.0]
