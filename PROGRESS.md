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

## Stage 3 - HNSW index (done)
- `hnsw.py`: `HNSWIndex` with params M, ef_construction, ef_search; random level
  `floor(-ln(U) / ln(M))`; greedy descent on upper layers; beam search on layer 0 with
  MinHeap (candidates) + MaxHeap (results); neighbour selection heuristic (Alg. 4) with
  keepPrunedConnections; M_max0 = 2M on layer 0; `search_with_trace` for visualization.
- Speed work after profiling: heap stores the sign-adjusted key (no compare method calls),
  heuristic reads a python list instead of a numpy matrix, direct L2 for short neighbour lists.
  Build got ~2.6x faster.
- `benchmarks/datasets.py`: SIFT-10K downloaded OK from the INRIA FTP server
  (fallback to synthetic clustered data is implemented, not needed).
- **SIFT-10K Recall@10 (M=16, ef_construction=200): ef=10 -> 0.952, ef=50 -> 0.999, ef=100 -> 1.0.**
  Target (>= 0.90) met without extra tuning.
- Bug found in a test (queries drawn from different clusters than the data) and fixed.
- Total: 73 tests passing (includes a SIFT-10K recall test, skipped if data is missing).
