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

## Stage 4 - Persistence (done)
- `persistence.py`: index saved as a folder: `header.json` (class, version, params),
  `arrays.npz` (numpy arrays, loaded with `allow_pickle=False`), `structure.pkl`
  (HNSW layers / LSH buckets / RNG state). `index.save(path)` / `Cls.load(path)`.
- RNG state is saved, so an HNSW index can be loaded and extended and gives exactly the
  same graph as building it in one go.
- Tests: round trip for all three indexes, add-after-load, empty index, custom ids,
  wrong class / missing files / bad version errors. Total: 87 tests passing.

## Stage 5 - Benchmarks (done)
- `benchmarks/run_benchmarks.py`: BruteForce / LSH / HNSW (ours) vs FAISS Flat / LSH / HNSW.
  FAISS pinned to 1 thread and queried one vector at a time (same conditions as ours).
  Measures build time, QPS, Recall@10, distance computations per query, memory, disk size.
- Experiments: main table, HNSW sweep (M x ef_search), LSH sweep (L x K), FAISS HNSW ef sweep,
  scaling up to n = 50K (synthetic clustered 128-d, since SIFT-10K only has 10K vectors).
- Outputs in `benchmarks/results/`: 5 CSVs, `summary.json`, 5 PNG charts. `--charts-only` redraws charts.
- Key numbers (SIFT-10K): HNSW Recall@10 0.999 (ef=50) / 1.000 (ef>=80) using ~650 distance
  computations per query instead of 10,000. FAISS HNSW 0.995 at the same settings. Our LSH 0.959
  (L=16,K=12) up to 0.997 (L=32,K=12); FAISS LSH 0.691.
- Scaling: brute force is faster below ~10K vectors (one BLAS call vs python overhead), at 50K our
  HNSW is ~2x faster than brute force with recall 0.998 while computing ~1% of the distances.
- Issue found + fixed: the first run had noisy single-shot timings and stopped at n=10K. Now best of 3
  repeats, and scaling goes to 50K.

## Stage 6 - Streamlit demo (done)
- `app/search_engine.py` (backend, no UI) + `app/streamlit_app.py` (UI).
- Corpus: first 5,000 headlines of the AG News test set (downloaded OK). Embeddings:
  sentence-transformers all-MiniLM-L6-v2 (384-d). Fallbacks implemented (synthetic headlines,
  offline hashing embedder) but not needed.
- Sidebar: switch HNSW / LSH / BruteForce, k, ef_search; table comparing time + recall of all three.
- Embeddings and indexes cached in `app/cache/` using our own save/load (HNSW loads in 0.02 s
  instead of ~40 s rebuild).
- Fixes during testing: AG News `#39;` entities cleaned; single-shot timings replaced by median of 5;
  LSH config tuned by measurement on the real embeddings (L=12,K=10 gave recall 0.64 ->
  L=32,K=10 gives 0.91; HNSW gives 0.995). Cache folder names now include index params.
- Verified in the browser: app launches, no server errors, queries return relevant results.
  Screenshots in `docs/screenshots/`.
- Tests: `tests/test_search_engine.py` (offline embedder). Total 94 tests passing.

## Stage 7 - HNSW visualization notebook (done)
- `notebooks/hnsw_visualization.ipynb` (executed, outputs saved): 300 random 2-D points, M=4.
  1. nodes per layer vs theory n / M^l (observed 300, 69, 17, 5, 2 vs expected 300, 75, 19, 5, 1)
  2. the graph drawn on every layer
  3. search path of one query: greedy descent through the layers + beam expansion on layer 0
  4. effect of ef_search on recall and work (done in 16-D because 2-D is too easy: recall is 1.0
     even at the smallest beam). Recall 0.70 (ef=10) -> 0.97 (ef=50) -> 1.00 (ef=200).

## Stage 8 - Documentation (done)
- Final full benchmark run; main table now shows best-accuracy settings too (HNSW ef=100 -> recall 1.000,
  LSH L=32 K=12 -> 0.997). The HNSW sweep was re-run with 5 timing repeats because one run had CPU
  noise (added `--only` flag to re-run single experiments, `--out` to write elsewhere).
- `README.md`: overview, two mermaid diagrams (architecture + HNSW query flow), setup, usage,
  full results tables + charts, screenshots.
- `docs/REPORT.md`: abstract, introduction, problem statement, data structures, algorithms with
  complexity tables, implementation, results (6 subsections with real numbers), conclusion,
  future scope, 10 references.
- `docs/VIVA_NOTES.md`: 35 questions with short answers.
