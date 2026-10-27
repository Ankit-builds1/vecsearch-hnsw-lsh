"""Tests for the demo backend, using the offline embedder (no network needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from search_engine import Corpus, HashingEmbedder, _synthetic_headlines, build_engine  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    corpus = _synthetic_headlines(400, seed=1)
    corpus.texts.append("Quantum computer solves chemistry puzzle")
    corpus.labels.append("Sci/Tech")
    return build_engine(corpus, HashingEmbedder(), use_cache=False)


def test_hashing_embedder_normalized():
    emb = HashingEmbedder(64).encode(["hello world", "another sentence", ""])
    assert emb.shape == (3, 64)
    assert np.allclose(np.linalg.norm(emb[:2], axis=1), 1.0, atol=1e-5)
    assert np.all(np.isfinite(emb))


def test_all_indexes_built(engine):
    assert set(engine.indexes) == {"HNSW", "LSH", "BruteForce"}
    assert all(len(i) == 401 for i in engine.indexes.values())


def test_exact_match_ranks_first(engine):
    for name in ("BruteForce", "HNSW"):
        hits, ms = engine.search("Quantum computer solves chemistry puzzle", name, k=3)
        assert hits[0].text == "Quantum computer solves chemistry puzzle"
        assert hits[0].similarity == pytest.approx(1.0, abs=1e-4)
        assert ms >= 0


def test_hnsw_agrees_with_exact(engine):
    q = "Central bank cuts interest rates in Europe"
    exact = {h.text for h in engine.search(q, "BruteForce", 10)[0]}
    approx = {h.text for h in engine.search(q, "HNSW", 10, ef=100)[0]}
    assert len(exact & approx) >= 7


def test_similarities_sorted(engine):
    hits, _ = engine.search("team wins", "BruteForce", 10)
    sims = [h.similarity for h in hits]
    assert sims == sorted(sims, reverse=True)


def test_clean_text():
    from search_engine import clean_text

    assert clean_text("NASA #39;s new  rocket") == "NASA's new rocket"
    assert clean_text("AT&amp;T \\ profits") == "AT&T profits"


def test_synthetic_corpus_shape():
    c = _synthetic_headlines(50)
    assert isinstance(c, Corpus) and len(c.texts) == 50 == len(c.labels)
