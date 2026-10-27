# Progress Log

## Stage 1 - Setup + core building blocks (done)
- Project structure, `.venv` (Python 3.12), `requirements.txt`, `.gitignore`, git repo.
- `distance.py`: euclidean / cosine / dot product, single-pair + vectorized batch + pairwise matrix.
  All return a distance (smaller = closer) so every index can do "k smallest".
- `heap.py`: hand written array-based binary `MinHeap` / `MaxHeap` (push, pop, peek, pushpop,
  O(n) heapify) and `k_smallest` top-k helper.
- `base.py`: `BaseIndex` abstract class + `VectorStore` (dynamic array with capacity doubling).
- `brute_force.py`: `BruteForceIndex` exact search (argpartition or own heap for top-k).
- Tests: `test_distance.py`, `test_heap.py`, `test_brute_force.py` - 41 passed.

## Stage 2 - LSH index (done)
- `lsh.py`: `LSHIndex` with random hyperplane hashing, L tables x K bits, `dict` buckets,
  data centering (needed for non-negative data like SIFT), optional 1-bit multi-probe,
  exact re-rank of candidates + top-k with own MaxHeap.
- Tests check recall against brute force (Recall@10 = 1.0 on clustered 3000x32 while
  scanning ~14% of the points), more tables -> better recall, multi-probe, edge cases.
- Total: 54 tests passing.
