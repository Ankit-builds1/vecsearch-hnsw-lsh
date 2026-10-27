import sys
from pathlib import Path

import numpy as np
import pytest

from vecsearch import HNSWIndex
from vecsearch.distance import METRICS

from conftest import make_clustered, recall_at_k


@pytest.fixture(scope="module")
def built():
    # queries must come from the same clusters as the data, so split one sample
    data = make_clustered(2060, 32, seed=3)
    X, Q = data[:2000], data[2000:]
    idx = HNSWIndex(32, M=12, ef_construction=100, ef_search=50)
    idx.add(X)
    return idx, X, Q


def test_recall_clustered(built):
    idx, X, Q = built
    r = recall_at_k(idx, X, Q, 10)
    assert r >= 0.95, f"recall {r:.3f}"


def test_recall_uniform_random(rng):
    # uniform random data is the hard case for every ANN method
    X = rng.random((2000, 16), dtype=np.float32)
    Q = rng.random((50, 16), dtype=np.float32)
    idx = HNSWIndex(16, M=16, ef_construction=100, ef_search=100)
    idx.add(X)
    assert recall_at_k(idx, X, Q, 10) >= 0.90


def test_higher_ef_does_not_hurt(built):
    idx, X, Q = built
    old = idx.ef_search
    idx.ef_search = 10
    low = recall_at_k(idx, X, Q, 10)
    idx.ef_search = 200
    high = recall_at_k(idx, X, Q, 10)
    idx.ef_search = old
    assert high >= low


def test_degree_limits(built):
    idx, _, _ = built
    for layer, g in enumerate(idx._layers):
        cap = idx.M_max0 if layer == 0 else idx.M
        assert all(len(n) <= cap for n in g.values()), f"layer {layer} over capacity"
        # no self loops, all neighbours exist on the same layer
        for u, neigh in g.items():
            assert u not in neigh
            assert all(v in g for v in neigh)


def test_layers_are_nested(built):
    idx, _, _ = built
    for l in range(1, idx.max_level + 1):
        assert set(idx.layer_nodes(l)) <= set(idx.layer_nodes(l - 1))
    # entry point lives on the top layer
    assert idx.entry_point in idx.layer_nodes(idx.max_level)


def test_level_distribution_decays(built):
    idx, X, _ = built
    stats = idx.graph_stats()["layers"]
    assert stats[0]["nodes"] == len(X)
    counts = [s["nodes"] for s in stats]
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    # with M=12 about 1/12 of nodes should reach layer 1
    assert 0.03 < counts[1] / counts[0] < 0.2


def test_finds_itself(built):
    idx, X, _ = built
    for i in [0, 500, 1999]:
        ids, d = idx.search(X[i], 1)
        assert ids[0] == i
        assert d[0] == pytest.approx(0.0, abs=1e-3)


def test_results_sorted_and_unique(built):
    idx, _, Q = built
    ids, d = idx.search(Q[0], 20)
    assert len(ids) == 20 and len(set(ids.tolist())) == 20
    assert np.all(np.diff(d) >= -1e-6)


@pytest.mark.parametrize("metric", METRICS)
def test_all_metrics(metric, rng):
    X = rng.normal(size=(300, 8))
    idx = HNSWIndex(8, metric=metric, M=8, ef_construction=50)
    idx.add(X)
    Q = rng.normal(size=(20, 8))
    assert recall_at_k(idx, X, Q, 5, metric) >= 0.9


def test_empty_index():
    idx = HNSWIndex(4)
    ids, _ = idx.search(np.ones(4), 3)
    assert len(ids) == 0
    assert idx.max_level == -1
    assert idx.search_with_trace(np.ones(4))["path"] == []


def test_single_element():
    idx = HNSWIndex(3)
    idx.add(np.array([[1.0, 2.0, 3.0]]), ids=[99])
    ids, _ = idx.search(np.zeros(3), 5)
    assert ids.tolist() == [99]


def test_k_larger_than_n(rng):
    idx = HNSWIndex(4, M=4)
    idx.add(rng.normal(size=(7, 4)))
    ids, _ = idx.search(np.zeros(4), 100)
    assert sorted(ids.tolist()) == list(range(7))


def test_incremental_adds_match_single_add(rng):
    X = rng.normal(size=(200, 8))
    a = HNSWIndex(8, M=8, seed=1)
    a.add(X)
    b = HNSWIndex(8, M=8, seed=1)
    b.add(X[:50])
    b.add(X[50:])
    assert a._layers == b._layers


def test_custom_ids(rng):
    X = rng.normal(size=(50, 4))
    idx = HNSWIndex(4, M=4)
    idx.add(X, ids=np.arange(50) * 10)
    assert idx.search(X[7], 1)[0][0] == 70


def test_trace(built):
    idx, _, Q = built
    t = idx.search_with_trace(Q[0], 5)
    layers = [l for l, _ in t["path"]]
    assert layers[0] == idx.max_level and layers[-1] == 0
    assert layers == sorted(layers, reverse=True)  # only goes down
    assert len(t["expanded"]) > 0
    assert t["ids"].tolist() == idx.search(Q[0], 5)[0].tolist()


def test_invalid_params():
    with pytest.raises(ValueError):
        HNSWIndex(4, M=1)
    with pytest.raises(ValueError):
        HNSWIndex(4, ef_search=0)


# ------------------------------------------------------------------ SIFT-10K

SIFT_DIR = Path(__file__).resolve().parents[1] / "data" / "siftsmall"


@pytest.mark.skipif(not (SIFT_DIR / "siftsmall_base.fvecs").exists(),
                    reason="SIFT-10K not downloaded (run benchmarks/datasets.py)")
def test_sift10k_recall_at_10():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from datasets import load_sift10k

    ds = load_sift10k(verbose=False)
    idx = HNSWIndex(128, M=16, ef_construction=100, ef_search=50)
    idx.add(ds.base)
    hits = 0
    for q, gt in zip(ds.queries, ds.ground_truth):
        ids, _ = idx.search(q, 10)
        hits += len(set(ids.tolist()) & set(gt[:10].tolist()))
    recall = hits / (10 * len(ds.queries))
    assert recall >= 0.90, f"SIFT-10K recall@10 = {recall:.3f}"
