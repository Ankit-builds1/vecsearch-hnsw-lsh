# VecSearch: A Vector Similarity Search Engine Built from Scratch

**Course:** Data Structures and Algorithms - Project Report
**Author:** Ankit Dash
**Tools:** Python 3.12, NumPy (core), pytest, FAISS (only for comparison), Streamlit, sentence-transformers
**Live demo:** https://vecsearch-ankit.streamlit.app/
**Source code:** https://github.com/Ankit-builds1/vecsearch-hnsw-lsh

---

## Abstract

Modern applications such as semantic search, recommendation and image retrieval represent items as
high-dimensional vectors and need to find the vectors nearest to a query very quickly. This project
implements three nearest neighbour search indexes from scratch in Python and NumPy: an exact
**Brute Force** index, **Locality Sensitive Hashing (LSH)** with random hyperplanes, and a
**Hierarchical Navigable Small World (HNSW)** graph. Supporting data structures (binary min/max heaps,
a dynamic array, hash table buckets and a layered adjacency-list graph) are also written by hand.
The indexes are benchmarked on the standard SIFT-10K dataset against Facebook's FAISS library. Our HNSW
reaches **Recall@10 = 0.999** at ef_search = 50 and **1.000** at ef_search = 100, slightly above FAISS HNSW with
the same parameters (0.995), while computing only about 650 distances per query instead of 10,000. Our LSH reaches
Recall@10 = 0.997 with 32 hash tables. A Streamlit application
demonstrates semantic search over 5,000 news headlines using the indexes.

---

## 1. Introduction

A *vector embedding* is a list of numbers that describes an object such that similar objects get similar
vectors. A sentence model like all-MiniLM-L6-v2 turns "stock market crash" and "shares fall sharply"
into vectors that are close together even though they share no words. Searching by meaning therefore
becomes a geometric problem: **find the k vectors nearest to the query vector**.

The obvious solution compares the query with every stored vector. This is exact but costs O(n * d) per
query, which does not scale to millions of vectors. Approximate Nearest Neighbour (ANN) algorithms trade a
small amount of accuracy for a large reduction in work. The two most influential families are hashing
(LSH) and proximity graphs (HNSW), and both are built around classic data structures, which makes them a
good DSA project.

## 2. Problem Statement

Given a dataset X = {x1, ..., xn} of d-dimensional vectors, a distance function dist and a query q, return
the k vectors with the smallest dist(q, xi).

Goals of the project:
1. Implement Brute Force, LSH and HNSW indexes with a common interface (`add`, `search`, `save`, `load`)
   using only Python and NumPy.
2. Implement the needed data structures myself (no `heapq`, no ANN libraries).
3. Measure build time, query speed, Recall@10 and memory, and compare with FAISS.
4. Study the parameter trade-offs (M and ef_search for HNSW, number of tables and bits for LSH).
5. Build a working semantic search demo.

## 3. Data Structures Used

| Data structure | Where | Why |
|---|---|---|
| **Binary MinHeap / MaxHeap** (array based) | `heap.py`, used by all indexes | O(log n) insert / extract, O(1) peek. Top-k selection and HNSW beam search. |
| **Dynamic array** (capacity doubling) | `base.py` `VectorStore` | Amortized O(1) append of vectors while keeping one contiguous NumPy block for fast vectorized math. |
| **Hash table** (Python `dict`) of lists | `lsh.py` buckets | Average O(1) lookup of the bucket for a hash code. One table per hash function group. |
| **Bit codes** (integers) | `lsh.py` | K hyperplane signs packed into one integer; multi-probe flips single bits with XOR. |
| **Layered graph as adjacency lists** (`list[dict[int, list[int]]]`) | `hnsw.py` | Layer l -> node -> neighbour list. O(1) neighbour access, O(n * M) memory. |
| **Hash set** | `hnsw.py` search | O(1) "already visited" check during graph search. |
| **Skip-list style levels** | `hnsw.py` | Exponentially decaying number of nodes per layer gives logarithmic search. |

### 3.1 Binary heap
The heap is a complete binary tree stored in an array: the parent of index i is (i - 1) / 2 and its children
are 2i + 1 and 2i + 2. `push` appends and sifts up, `pop` moves the last element to the root and sifts down.
Both are O(log n). Building from a list uses bottom-up heapify in O(n). A MaxHeap is implemented as a MinHeap
over negated keys so both share one implementation.

**Top-k with a MaxHeap.** To keep the k smallest distances, keep a MaxHeap of size k. Its root is the worst of
the current best k. A new distance enters only if it is smaller than the root. This costs O(n log k) time and O(k)
memory, compared with O(n log n) for sorting.

### 3.2 Dynamic array
Vectors are stored in one NumPy matrix. When it is full, a new matrix of double the capacity is allocated and the
old rows are copied. Over n appends the total copying is at most 2n rows, so each append costs O(1) amortized.

## 4. Algorithms and Complexity Analysis

Notation: n vectors, d dimensions, k results.

### 4.1 Distance functions
Euclidean ||a - b||, cosine distance 1 - a.b / (|a||b|), and negative dot product. All are "smaller is closer".
The batched L2 uses ||x - q||^2 = ||x||^2 - 2 x.q + ||q||^2, so distances to all rows are one matrix-vector product.
Cost O(n * d).

### 4.2 Brute Force (exact)
Compute the distance to every vector, then select the k smallest (with `np.argpartition` in O(n), or with our own
heap in O(n log k)).

| Operation | Time | Space |
|---|---|---|
| add m vectors | O(m * d) amortized | O(n * d) |
| search | O(n * d) | O(n) |

### 4.3 Locality Sensitive Hashing (random hyperplanes)
For each of L tables, K random Gaussian hyperplanes are drawn. The hash of x is the K-bit code
`h(x) = [sign(r1 . x), ..., sign(rK . x)]`. For two vectors with angle theta, each bit agrees with
probability `p = 1 - theta / pi`, so the probability they share a bucket in at least one of L tables is
`1 - (1 - p^K)^L`. Larger K makes buckets more selective, larger L raises recall.

* **Centering:** hyperplanes pass through the origin, but SIFT vectors are all non-negative. The mean of the data
  is subtracted before hashing, otherwise most bits are constant.
* **Multi-probe:** the query also checks the K buckets at Hamming distance 1 (code XOR 2^b).
* **Re-ranking:** candidates from all probed buckets are united, exact distances are computed only for them,
  and the top k are selected with a MaxHeap.

| Operation | Time | Space |
|---|---|---|
| add m vectors | O(m * L * K * d) | O(n * L) bucket entries + O(L * K * d) planes |
| search | O(L * K * d) hashing + O(L * K) probes + O(C * d) re-rank + O(C log k), C = candidates | O(C) |

### 4.4 Hierarchical Navigable Small World graph
Following Malkov and Yashunin (2016):

**Level assignment.** Each new node gets level `l = floor(-ln(U) * mL)` with U uniform in (0, 1] and
`mL = 1 / ln(M)`. Then `P(level >= l) = M^-l`: each layer holds about 1/M of the nodes below it, like a skip list,
and the expected number of layers is O(log n).

**Search (Algorithm 5).**
1. Start at the global entry point on the top layer.
2. On every layer above 0: *greedy search*, i.e. repeatedly move to the neighbour closest to the query until no
   neighbour is closer.
3. On layer 0: *beam search* with width ef. A MinHeap holds candidates to expand (closest first) and a MaxHeap
   holds the best ef results found so far. A candidate is popped, its unvisited neighbours are evaluated in one
   vectorized call, and any neighbour better than the current worst result is pushed into both heaps. The loop
   stops when the closest candidate is farther than the worst result.
4. Return the k best of the ef results.

**Insertion (Algorithm 1).** Greedy descent to the new node's level, then on each layer from there down to 0 run
the beam search with ef_construction, select M neighbours with the heuristic, add links in both directions, and
shrink any neighbour list that now exceeds its limit (M on upper layers, 2M on layer 0).

**Neighbour selection heuristic (Algorithm 4).** Candidates are visited from closest to farthest. A candidate e is
kept only if it is closer to the new node than to every neighbour already kept. Otherwise e is "covered" by an
existing neighbour and is pruned (pruned ones fill remaining slots, `keepPrunedConnections`). This keeps links
pointing in different directions and preserves bridges between clusters.

| Operation | Time | Space |
|---|---|---|
| insert one vector | O(log n * ef_construction * M * d) expected | O(M * level) edges |
| search | O(log n * M * d) upper layers + O(ef * M * d) layer 0 | O(ef + visited) |
| total memory | | O(n * d + n * M) |

### 4.5 Summary

| Index | Build | Query | Exact? | Main parameters |
|---|---|---|---|---|
| Brute Force | O(n d) | O(n d) | yes | - |
| LSH | O(n L K d) | O(L K d + C d) | no | L tables, K bits, multi-probe |
| HNSW | O(n log n * ef_c * M * d) | ~O(log n * M d + ef M d) | no | M, ef_construction, ef_search |

## 5. Implementation

```
vecsearch/
  distance.py     single, batched and pairwise distances (l2, cosine, ip)
  heap.py         MinHeap, MaxHeap, k_smallest
  base.py         BaseIndex (abstract) + VectorStore (dynamic array)
  brute_force.py  BruteForceIndex
  lsh.py          LSHIndex
  hnsw.py         HNSWIndex (+ search_with_trace for visualization)
  persistence.py  save_index / load_index
tests/            pytest suite (94 tests)
benchmarks/       datasets.py, run_benchmarks.py, results/
app/              search_engine.py (backend), streamlit_app.py (UI)
notebooks/        hnsw_visualization.ipynb
```

**Common interface.** Every index extends `BaseIndex` and implements `add(vectors, ids=None)`,
`search(query, k) -> (ids, distances)`, `save(path)` / `load(path)` and `__len__`. Results are always sorted by
increasing distance, and requests with k > n return all n vectors.

**Making Python fast enough.** A first profile of HNSW construction showed most time was in Python overhead,
not in distance math. Three changes made the build about 2.6x faster:
1. the heap stores the sign-adjusted key so sift loops use plain `<` instead of a method call (it was called
   13 million times while building 3,000 points);
2. the neighbour heuristic converts the candidate distance matrix to a Python list before reading single elements
   (indexing NumPy scalars one by one is slow);
3. distances from the query to a node's neighbours are always computed in one NumPy call.

**Persistence.** An index is saved as a folder: `header.json` (class, format version, parameters), `arrays.npz`
(vectors, ids, LSH planes) and `structure.pkl` (HNSW layers, LSH buckets, random generator state). Saving the RNG
state means a loaded HNSW index can be extended and still produce exactly the same graph as an uninterrupted build
(this is tested).

**Testing.** The pytest suite covers: heap ordering against `sorted`, heapify, pushpop; distance functions against
known values and batch-vs-single agreement; empty index, k = 0, k > n, wrong dimensions, NaN input; recall of LSH and
HNSW against brute force; HNSW graph invariants (degree limits, nested layers, no self loops, level distribution);
save/load round trips; and the demo backend with an offline embedder.

## 6. Results

### 6.1 Setup
* **Dataset:** SIFT-10K (siftsmall, INRIA TEXMEX): 10,000 base vectors and 100 query vectors, 128 dimensions,
  with the official ground truth. For the scaling test (6.5), synthetic clustered 128-d data up to 50,000 vectors
  is used, with ground truth computed by brute force.
* **Metric:** Euclidean distance, k = 10.
* **Fairness:** every index is queried one vector at a time on a single thread. FAISS is pinned to 1 thread with
  `faiss.omp_set_num_threads(1)`. Each timing is the best of several repeats.
* **Measures:** build time, queries per second (QPS), Recall@10, distance computations per query (a
  machine-independent measure of work) and memory. For our indexes, memory is the real size of all Python/NumPy
  objects; for FAISS it is the serialized index size.
* **Machine:** Windows 11, Python 3.12, NumPy 2.5, FAISS 1.15 (CPU).

### 6.2 Main comparison (SIFT-10K)

| Index | Build (s) | QPS | Latency (ms) | Recall@10 | Distances / query | Memory (MB) |
|---|---:|---:|---:|---:|---:|---:|
| BruteForce (ours) | 0.002 | 1,283 | 0.78 | 1.000 | 10,000 | 8.3 |
| LSH, L=32, K=12 (ours) | 0.12 | 484 | 2.07 | 0.997 | 2,629 | 28.4 |
| HNSW, M=16, ef=50 (ours) | 42.3 | 963 | 1.04 | 0.999 | 652 | 11.8 |
| HNSW, M=16, ef=100 (ours) | 42.3 | 610 | 1.64 | 1.000 | 997 | 11.8 |
| FAISS Flat | 0.002 | 7,153 | 0.14 | 1.000 | 10,000 | 4.9 |
| FAISS LSH (1024 bits) | 0.12 | 11,900 | 0.08 | 0.691 | - | 1.7 |
| FAISS HNSW, M=16, ef=50 | 1.04 | 24,278 | 0.04 | 0.995 | 582 | 6.3 |

![Comparison](../benchmarks/results/comparison_bars.png)

**Observations**
1. **HNSW is accurate and does little work.** With ef_search = 50 it finds 99.9% of the true neighbours while
   computing 652 distances per query, 6.5% of what brute force needs. With ef_search = 100 recall is a perfect 1.000.
2. **Our HNSW is implemented correctly.** With the same M and ef it reaches the same recall as FAISS
   (0.999 vs 0.995) and does a similar number of distance computations (652 vs 582).
3. **The speed gap with FAISS is a language gap, not an algorithm gap.** FAISS HNSW is about 25x faster per query
   because the graph walk and heaps run in compiled C++ with SIMD, while ours run in the Python interpreter.
4. **Brute force is fastest at n = 10K.** It is a single matrix-vector product executed by BLAS, so for small n
   its O(n) cost is cheaper than HNSW's Python overhead. Section 6.5 shows where this changes.
5. **LSH needs many tables for high recall.** 32 tables give 0.997 recall but scan 26% of the data and use the most
   memory. Our LSH beats FAISS IndexLSH (0.691) because candidates are re-ranked with the exact distance, while
   FAISS's IndexLSH ranks by Hamming distance between binary codes.
6. **Build time:** HNSW is by far the slowest to build (42 s with ef_construction = 200), since every insertion is a
   search plus neighbour selection. Brute force and LSH just store or hash the vectors.

### 6.3 Recall vs speed

![Recall vs QPS](../benchmarks/results/recall_vs_qps.png)

Each blue point is our HNSW with a different ef_search. Increasing ef moves along the curve toward perfect recall
at lower speed. LSH points (different L and K) are always below and to the left of the HNSW curve: for any target
recall, HNSW is faster.

### 6.4 Parameter study

**HNSW (Recall@10 on SIFT-10K, ef_construction = 100):**

| M | ef=10 | ef=20 | ef=40 | ef=80 | ef=160 | ef=320 | Build (s) | Memory (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.749 | 0.872 | 0.936 | 0.969 | 0.992 | 0.996 | 23.3 | 10.4 |
| 8 | 0.881 | 0.954 | 0.985 | 0.996 | 0.998 | 0.998 | 23.7 | 10.9 |
| 16 | 0.956 | 0.984 | 0.998 | 1.000 | 1.000 | 1.000 | 26.5 | 11.8 |
| 32 | 0.985 | 0.998 | 0.999 | 1.000 | 1.000 | 1.000 | 50.2 | 13.7 |

![HNSW sweep](../benchmarks/results/hnsw_param_sweep.png)

* **ef_search** is the main accuracy knob: query latency grows about linearly with it (0.46 ms at ef=10 to
  4.1 ms at ef=320 for M=16), and recall saturates at 1.000 from ef=80.
* **M** controls the graph: with M=4 the graph is too sparse (recall only 0.749 at ef=10), while M=16-32 is
  already near perfect at small ef. M=32 doubles the build time compared with M=16 for a small gain, so M=16
  is a good default, which matches the recommendation in the HNSW paper.

**LSH (K = 12 bits, multi-probe on):**

| Tables L | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|
| Recall@10 | 0.439 | 0.632 | 0.820 | 0.959 | 0.997 |
| % of data scanned | 2.7 | 5.1 | 9.3 | 18.1 | 26.3 |

![LSH sweep](../benchmarks/results/lsh_sweep.png)

More bits per table (K) make buckets smaller and faster but miss more neighbours; more tables (L) recover the
recall. This matches the collision probability 1 - (1 - p^K)^L from Section 4.3.

### 6.5 Scaling with dataset size

| n | Brute force (ms) | HNSW (ms) | HNSW Recall@10 | HNSW distances / query | HNSW layers |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.06 | 0.67 | 1.000 | 448 | 3 |
| 2,000 | 0.10 | 0.58 | 1.000 | 397 | 3 |
| 5,000 | 0.30 | 0.51 | 1.000 | 252 | 4 |
| 10,000 | 0.94 | 0.65 | 1.000 | 226 | 4 |
| 20,000 | 1.45 | 0.87 | 1.000 | 299 | 4 |
| 50,000 | 3.47 | 1.92 | 0.998 | 555 | 5 |

![Scaling](../benchmarks/results/scaling.png)

Brute force time grows linearly with n, as expected from O(n * d). HNSW work stays in the hundreds of distance
computations even at 50,000 vectors (about 1% of brute force), and the number of layers grows only from 3 to 5,
like log n. Our Python HNSW becomes faster than NumPy brute force at about 10,000 vectors, and at 50,000 it is
**1.8x faster** with 0.998 recall. With more data the gap keeps growing.

### 6.6 Semantic search demo

The Streamlit app embeds the first 5,000 AG News headlines with all-MiniLM-L6-v2 (384 dimensions) and searches them
with cosine distance. For example, the query "space mission to mars" returns headlines such as "Mission accomplished:
US astronauts are home at last" and "Mars gases reveal enticing clues for life", although they share few words with
the query. HNSW returns the same top 10 as exact search.

An interesting finding came from tuning LSH on these text embeddings. Measured on 200 queries:

| Index | Recall@10 |
|---|---:|
| LSH, L=12, K=10 (first guess) | 0.642 |
| LSH, L=32, K=10 (used in the app) | 0.912 |
| HNSW, M=16, ef=50 | 0.995 |

LSH is much weaker on text embeddings than on SIFT. The reason is the angle between neighbours: the nearest
headlines typically have cosine similarity around 0.5 (about 60 degrees), so each hash bit agrees with probability
1 - 60/180 = 0.67, and all 10 bits agree only 0.67^10 = 1.8% of the time. LSH therefore needs many tables, while
HNSW is unaffected because it follows actual neighbour links instead of hashing. Mean-centering the embeddings
before hashing was also tested and did not help.

![App](screenshots/app_search.jpg)

## 7. Conclusion

All three indexes were implemented from scratch with the supporting data structures, and they behave as theory
predicts. Brute force is exact and, for 10K vectors, very fast because it is one BLAS call. LSH reduces the number of
vectors that must be checked but needs many tables (memory) to reach high recall. HNSW gives the best accuracy per
unit of work: on SIFT-10K it finds the true top 10 almost perfectly while computing only a small fraction of the
distances, and its work per query grows very slowly with n while brute force grows linearly. Compared with FAISS,
our HNSW reaches the same recall at the same parameters, which confirms the algorithm is implemented correctly; the
large speed gap comes from C++/SIMD versus interpreted Python, not from the algorithm.

## 8. Future Scope

* Move the hot loops (beam search, heuristic) to Cython, Numba or C++ to remove interpreter overhead.
* Support deletion (mark-as-deleted plus periodic repair of neighbour lists).
* Product Quantization (PQ) to compress vectors 10-30x, and IVF (inverted file) indexes for comparison.
* Batched and multithreaded queries; parallel graph construction with fine-grained locks.
* Filtered search (e.g. "only Sports headlines") and hybrid keyword + vector search.
* Scale experiments to SIFT-1M and GIST-1M.

## 9. References

1. Y. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical
   Navigable Small World graphs," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2018
   (arXiv:1603.09320, 2016).
2. M. S. Charikar, "Similarity estimation techniques from rounding algorithms," *STOC*, 2002.
3. P. Indyk and R. Motwani, "Approximate nearest neighbors: towards removing the curse of dimensionality,"
   *STOC*, 1998.
4. Q. Lv, W. Josephson, Z. Wang, M. Charikar and K. Li, "Multi-probe LSH: efficient indexing for high-dimensional
   similarity search," *VLDB*, 2007.
5. H. Jegou, M. Douze and C. Schmid, "Product quantization for nearest neighbor search," *IEEE TPAMI*, 2011
   (source of the SIFT-10K / TEXMEX dataset, http://corpus-texmex.irisa.fr/).
6. J. Johnson, M. Douze and H. Jegou, "Billion-scale similarity search with GPUs," *IEEE Transactions on Big Data*,
   2019 (FAISS).
7. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," *EMNLP*, 2019.
8. X. Zhang, J. Zhao and Y. LeCun, "Character-level convolutional networks for text classification," *NeurIPS*,
   2015 (AG News dataset).
9. T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 3rd ed., MIT Press, 2009
   (heaps, amortized analysis of dynamic arrays, hashing).
10. W. Pugh, "Skip lists: a probabilistic alternative to balanced trees," *Communications of the ACM*, 1990.

---

Copyright © 2026 Ankit Dash. All rights reserved.
