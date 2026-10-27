import numpy as np
import pytest

from vecsearch import distance as D


def test_euclidean_known_value():
    assert D.euclidean(np.array([0, 0]), np.array([3, 4])) == pytest.approx(5.0)


def test_cosine_known_values():
    a = np.array([1.0, 0.0])
    assert D.cosine(a, np.array([2.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
    assert D.cosine(a, np.array([0.0, 5.0])) == pytest.approx(1.0)
    assert D.cosine(a, np.array([-1.0, 0.0])) == pytest.approx(2.0)


def test_cosine_zero_vector_is_safe():
    assert D.cosine(np.zeros(3), np.ones(3)) == pytest.approx(1.0)
    out = D.cosine_batch(np.zeros(3), np.ones((4, 3)))
    assert np.all(np.isfinite(out))


def test_dot_product_is_negated():
    assert D.dot_product(np.array([1, 2]), np.array([3, 4])) == pytest.approx(-11.0)


@pytest.mark.parametrize("metric", D.METRICS)
def test_batch_matches_single(metric, rng):
    q = rng.normal(size=16)
    X = rng.normal(size=(50, 16))
    single = D.get_distance(metric)
    batch = D.get_batch_distance(metric)
    expected = np.array([single(q, x) for x in X])
    assert np.allclose(batch(q, X), expected, atol=1e-4)


@pytest.mark.parametrize("metric", D.METRICS)
def test_pairwise_matches_batch(metric, rng):
    Q = rng.normal(size=(5, 8))
    X = rng.normal(size=(30, 8))
    M = D.pairwise(Q, X, metric)
    batch = D.get_batch_distance(metric)
    for i, q in enumerate(Q):
        assert np.allclose(M[i], batch(q, X), atol=1e-4)


@pytest.mark.parametrize("metric", D.METRICS)
def test_batch_empty_matrix(metric):
    out = D.get_batch_distance(metric)(np.ones(4), np.empty((0, 4)))
    assert out.shape == (0,)


def test_unknown_metric():
    with pytest.raises(ValueError):
        D.get_distance("manhattan")
    with pytest.raises(ValueError):
        D.get_batch_distance("manhattan")
    with pytest.raises(ValueError):
        D.pairwise(np.ones((1, 2)), np.ones((1, 2)), "manhattan")


def test_euclidean_batch_never_negative(rng):
    X = rng.normal(size=(100, 32)).astype(np.float32)
    d = D.euclidean_batch(X[0], X)
    assert d.min() >= 0
    assert d[0] == pytest.approx(0.0, abs=1e-3)
