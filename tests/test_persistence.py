import json

import numpy as np
import pytest

from vecsearch import BruteForceIndex, HNSWIndex, LSHIndex
from vecsearch.persistence import load_index, save_index


def build(cls, X, **kw):
    idx = cls(X.shape[1], **kw)
    idx.add(X)
    return idx


INDEXES = [
    (BruteForceIndex, {}),
    (LSHIndex, {"n_tables": 4, "n_bits": 8}),
    (HNSWIndex, {"M": 8, "ef_construction": 50}),
]


@pytest.mark.parametrize("cls,kw", INDEXES)
def test_roundtrip_same_results(cls, kw, rng, tmp_path):
    X = rng.normal(size=(300, 16)).astype(np.float32)
    Q = rng.normal(size=(10, 16)).astype(np.float32)
    idx = build(cls, X, **kw)
    idx.save(tmp_path / "idx")
    loaded = cls.load(tmp_path / "idx")

    assert type(loaded) is cls
    assert len(loaded) == len(idx)
    assert np.array_equal(loaded.vectors, idx.vectors)
    for q in Q:
        a_ids, a_d = idx.search(q, 5)
        b_ids, b_d = loaded.search(q, 5)
        assert a_ids.tolist() == b_ids.tolist()
        assert np.allclose(a_d, b_d)


@pytest.mark.parametrize("cls,kw", INDEXES)
def test_can_keep_adding_after_load(cls, kw, rng, tmp_path):
    X = rng.normal(size=(200, 8)).astype(np.float32)
    full = build(cls, X, **kw)

    half = build(cls, X[:100], **kw)
    save_index(half, tmp_path / "half")
    resumed = load_index(tmp_path / "half")
    resumed.add(X[100:])

    assert len(resumed) == 200
    q = X[150]
    assert resumed.search(q, 5)[0].tolist() == full.search(q, 5)[0].tolist()


def test_hnsw_graph_identical(rng, tmp_path):
    idx = build(HNSWIndex, rng.normal(size=(150, 8)), M=6)
    idx.save(tmp_path / "h")
    loaded = HNSWIndex.load(tmp_path / "h")
    assert loaded._layers == idx._layers
    assert loaded.entry_point == idx.entry_point
    assert loaded.max_level == idx.max_level


def test_custom_ids_survive(rng, tmp_path):
    X = rng.normal(size=(20, 4))
    idx = BruteForceIndex(4)
    idx.add(X, ids=list(range(500, 520)))
    idx.save(tmp_path / "b")
    loaded = BruteForceIndex.load(tmp_path / "b")
    assert loaded.ids.tolist() == list(range(500, 520))


def test_empty_index_roundtrip(tmp_path):
    for cls, kw in INDEXES:
        idx = cls(4, **kw)
        idx.save(tmp_path / cls.__name__)
        loaded = cls.load(tmp_path / cls.__name__)
        assert len(loaded) == 0
        assert len(loaded.search(np.ones(4), 3)[0]) == 0


def test_metric_and_params_saved(rng, tmp_path):
    idx = build(HNSWIndex, rng.normal(size=(30, 4)), metric="cosine", M=5, ef_search=33)
    idx.save(tmp_path / "c")
    header = json.loads((tmp_path / "c" / "header.json").read_text())
    assert header["class"] == "HNSWIndex"
    loaded = HNSWIndex.load(tmp_path / "c")
    assert loaded.metric == "cosine" and loaded.M == 5 and loaded.ef_search == 33


def test_wrong_class_raises(rng, tmp_path):
    build(BruteForceIndex, rng.normal(size=(10, 4))).save(tmp_path / "b")
    with pytest.raises(TypeError):
        HNSWIndex.load(tmp_path / "b")


def test_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "nope")


def test_bad_version(rng, tmp_path):
    folder = save_index(build(BruteForceIndex, rng.normal(size=(5, 4))), tmp_path / "v")
    h = json.loads((folder / "header.json").read_text())
    h["format_version"] = 999
    (folder / "header.json").write_text(json.dumps(h))
    with pytest.raises(ValueError):
        load_index(folder)


def test_unknown_class(rng, tmp_path):
    folder = save_index(build(BruteForceIndex, rng.normal(size=(5, 4))), tmp_path / "u")
    h = json.loads((folder / "header.json").read_text())
    h["class"] = "AnnoyIndex"
    (folder / "header.json").write_text(json.dumps(h))
    with pytest.raises(ValueError):
        load_index(folder)
