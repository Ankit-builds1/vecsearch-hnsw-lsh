import numpy as np
import pytest

from vecsearch.heap import MaxHeap, MinHeap, k_smallest


def drain(h):
    out = []
    while h:
        out.append(h.pop()[0])
    return out


def test_minheap_pops_in_ascending_order(rng):
    keys = rng.normal(size=200).tolist()
    h = MinHeap()
    for i, k in enumerate(keys):
        h.push(k, i)
    assert drain(h) == sorted(keys)


def test_maxheap_pops_in_descending_order(rng):
    keys = rng.normal(size=200).tolist()
    h = MaxHeap()
    for i, k in enumerate(keys):
        h.push(k, i)
    assert drain(h) == sorted(keys, reverse=True)


def test_heapify_constructor(rng):
    keys = rng.integers(0, 50, size=100).tolist()
    h = MinHeap([(k, None) for k in keys])
    assert len(h) == 100
    assert drain(h) == sorted(keys)


def test_peek_does_not_remove():
    h = MaxHeap()
    h.push(1, "a")
    h.push(5, "b")
    assert h.peek() == (5.0, "b")
    assert len(h) == 2


def test_empty_heap_errors():
    with pytest.raises(IndexError):
        MinHeap().pop()
    with pytest.raises(IndexError):
        MaxHeap().peek()


def test_pushpop():
    h = MinHeap([(3, "c"), (1, "a"), (2, "b")])
    # new key smaller than top -> returned straight away
    assert h.pushpop(0, "z") == (0.0, "z")
    # new key bigger than top -> old top comes out
    assert h.pushpop(10, "x") == (1.0, "a")
    assert drain(h) == [2.0, 3.0, 10.0]
    # pushpop on an empty heap just hands the item back
    assert MinHeap().pushpop(4, "q") == (4.0, "q")


def test_duplicate_keys():
    h = MinHeap()
    for i in range(10):
        h.push(1.0, i)
    assert sorted(item for _, item in h.to_sorted_list()) == list(range(10))


def test_to_sorted_list_keeps_heap():
    h = MaxHeap([(2, "b"), (9, "z"), (4, "d")])
    assert [k for k, _ in h.to_sorted_list()] == [9.0, 4.0, 2.0]
    assert len(h) == 3


def test_k_smallest(rng):
    keys = rng.normal(size=500)
    items = np.arange(500)
    got = k_smallest(keys, items, 10)
    expected = np.argsort(keys)[:10]
    assert [i for _, i in got] == expected.tolist()


def test_k_smallest_edge_cases():
    assert k_smallest([3.0, 1.0], [0, 1], 0) == []
    # k bigger than n just returns everything sorted
    assert [i for _, i in k_smallest([3.0, 1.0, 2.0], [0, 1, 2], 10)] == [1, 2, 0]
    assert k_smallest([], [], 5) == []
