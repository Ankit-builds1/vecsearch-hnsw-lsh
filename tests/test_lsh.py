import numpy as np
import pytest

from vecsearch import LSHIndex
from vecsearch.distance import METRICS

from conftest import make_clustered, recall_at_k


@pytest.fixture(scope="module")
def data():
    X = make_clustered(3000, 32, seed=1)
    Q = make_clustered(50, 32, seed=1)[:50] + np.random.default_rng(5).normal(0, 0.5, (50, 32)).astype(np.float32)
    return X, Q


def test_recall_vs_brute_force(data):
    X, Q = data
    idx = LSHIndex(32, n_tables=12, n_bits=10, multi_probe=True)
    idx.add(X)
    r = recall_at_k(idx, X, Q, k=10)
    assert r >= 0.80, f"recall too low: {r:.3f}"


def test_more_tables_give_better_recall(data):
    X, Q = data
    few = LSHIndex(32, n_tables=2, n_bits=14, multi_probe=False)
    many = LSHIndex(32, n_tables=16, n_bits=14, multi_probe=False)
    few.add(X)
    many.add(X)
    assert recall_at_k(many, X, Q, 10) > recall_at_k(few, X, Q, 10)


def test_multi_probe_helps(data):
    X, Q = data
    plain = LSHIndex(32, n_tables=4, n_bits=14, multi_probe=False)
    probe = LSHIndex(32, n_tables=4, n_bits=14, multi_probe=True)
    plain.add(X)
    probe.add(X)
    assert recall_at_k(probe, X, Q, 10) >= recall_at_k(plain, X, Q, 10)


def test_scans_fewer_than_all(data):
    X, Q = data
    idx = LSHIndex(32, n_tables=8, n_bits=12)
    idx.add(X)
    sizes = [len(idx.candidates(q)) for q in Q]
    assert np.mean(sizes) < len(X)  # otherwise it is just brute force


def test_finds_exact_duplicate(rng):
    X = rng.normal(size=(500, 16))
    idx = LSHIndex(16, n_tables=4, n_bits=8)
    idx.add(X)
    ids, dists = idx.search(X[123], 1)
    assert ids[0] == 123
    assert dists[0] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("metric", METRICS)
def test_all_metrics_run(metric, rng):
    X = rng.normal(size=(200, 8))
    idx = LSHIndex(8, metric=metric, n_tables=4, n_bits=6)
    idx.add(X)
    ids, dists = idx.search(X[0], 5)
    assert 0 < len(ids) <= 5
    assert np.all(np.diff(dists) >= -1e-6)


def test_empty_index():
    idx = LSHIndex(4)
    ids, _ = idx.search(np.ones(4), 3)
    assert len(ids) == 0
    assert idx.bucket_stats()["buckets"] == 0


def test_k_larger_than_n(rng):
    idx = LSHIndex(4, n_tables=3, n_bits=2)
    idx.add(rng.normal(size=(5, 4)))
    ids, _ = idx.search(np.zeros(4), 50)
    assert len(ids) <= 5
    assert len(set(ids.tolist())) == len(ids)  # no duplicates


def test_custom_ids(rng):
    X = rng.normal(size=(20, 4))
    idx = LSHIndex(4, n_tables=3, n_bits=3)
    idx.add(X, ids=list(range(1000, 1020)))
    ids, _ = idx.search(X[3], 1)
    assert ids[0] == 1003


def test_hash_is_deterministic(rng):
    X = rng.normal(size=(50, 8))
    a = LSHIndex(8, seed=7)
    b = LSHIndex(8, seed=7)
    a.add(X)
    b.add(X)
    assert a._tables == b._tables


def test_invalid_params():
    with pytest.raises(ValueError):
        LSHIndex(4, n_tables=0)
    with pytest.raises(ValueError):
        LSHIndex(4, n_bits=100)
