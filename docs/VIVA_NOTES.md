# Viva Notes - VecSearch

Short answers to questions I expect in the viva. Grouped by topic.

---

## A. The problem

**1. What is vector similarity search?**
Given a set of n vectors and a query vector, find the k vectors that are closest to the query
(k nearest neighbours, k-NN). Text, images and audio can all be turned into vectors
(embeddings), so "find similar items" becomes "find nearest vectors".

**2. Why not just compare the query with every vector?**
That is the brute force method. It costs O(n * d) per query. For 10K vectors it is fine, but for
100 million vectors of 768 dimensions it is far too slow for real time search. Approximate
methods (LSH, HNSW) look at only a small part of the data.

**3. What does "approximate" nearest neighbour (ANN) mean?**
The index may miss some of the true nearest neighbours in exchange for a big speed-up.
We measure how good it is with recall.

**4. What is Recall@10?**
For each query: (number of the true top-10 that the index returned) / 10, averaged over all
queries. Recall@10 = 1.0 means the index found exactly the true top 10.

**5. What is the curse of dimensionality?**
In high dimensions distances between random points become very similar, space partitioning
trees (like KD-trees) stop pruning and degrade to brute force. That is why graph and hashing
methods are used for d > ~20.

---

## B. Distance metrics

**6. Which distance functions did you implement?**
Euclidean (L2), cosine distance (1 - cosine similarity) and negative dot product. All return
"smaller = more similar", so every index just looks for the k smallest values.

**7. When is cosine better than Euclidean?**
When only the direction matters, not the length, e.g. sentence embeddings. For unit-length
vectors they give the same ranking because ||a - b||^2 = 2 - 2 cos(a, b).

**8. How did you make the distance computation fast in Python?**
Vectorization with NumPy. For L2 against a whole matrix I use
||x - q||^2 = ||x||^2 - 2 x.q + ||q||^2, which is one matrix-vector product in C instead of a Python loop.

---

## C. Heap

**9. Why did you need a heap?**
For top-k selection and for the HNSW search. A heap gives the smallest (or largest) element in
O(1) and insert/delete in O(log n).

**10. How is a binary heap stored?**
In a plain array. For index i: parent = (i - 1) // 2, left child = 2i + 1, right child = 2i + 2.
It is a complete binary tree, so there are no gaps and no pointers are needed.

**11. Explain push and pop.**
Push: append at the end, then sift up (swap with parent while smaller). Pop: take the root,
move the last element to the root, then sift down (swap with the smaller child). Both O(log n)
because the tree height is log n.

**12. How do you get top-k with a heap? Why a MaxHeap for the k smallest?**
Keep a MaxHeap of size k. Its top is the worst of the best k found so far. A new distance only
enters if it is smaller than the top, then the top is removed. Total O(n log k) time, O(k) space,
better than sorting everything (O(n log n)).

**13. How did you implement MaxHeap and MinHeap without duplicating code?**
The MaxHeap stores -key inside a min-heap and flips the sign back when returning. Same sift code.

**14. What is heapify and its complexity?**
Building a heap from an array by sifting down every non-leaf node from the bottom up. It is O(n),
not O(n log n), because most nodes are near the bottom and only move a little.

---

## D. LSH

**15. What is Locality Sensitive Hashing?**
A family of hash functions where similar items collide (land in the same bucket) with high
probability and different items with low probability. The opposite of normal hashing, which tries to
spread similar keys apart.

**16. How does random hyperplane LSH work?**
Pick K random hyperplanes through the origin. Each one gives one bit: 1 if the vector is on its
positive side. The K bits form the bucket key. For two vectors with angle theta,
P(same bit) = 1 - theta / pi, so close vectors share most bits.

**17. Why multiple hash tables?**
With K bits, one table is very strict and true neighbours often fall into a different bucket. Using L
independent tables and taking the union of the buckets raises the chance that a true neighbour shares
at least one bucket: 1 - (1 - p^K)^L.

**18. What is the effect of K and L?**
More bits K = smaller buckets = faster but lower recall. More tables L = higher recall but more memory
and more candidates to check. My LSH sweep chart shows this trade-off.

**19. What is multi-probe LSH?**
Besides the exact bucket, also look at buckets whose code differs by one bit (Hamming distance 1).
This gives better recall without adding more tables (less memory).

**20. Why did you subtract the mean before hashing?**
The hyperplanes pass through the origin. SIFT vectors are all non-negative, so they sit in one corner
and most hyperplanes put all of them on the same side, which makes the bits useless. Centering moves the
data around the origin.

**21. What data structure stores the buckets?**
A Python dict (hash table) per table: bucket code -> list of vector ids. Lookup is O(1) on average.

---

## E. HNSW

**22. What is HNSW in one sentence?**
A multi-layer proximity graph: the top layers are sparse with long links (like express lanes of a skip
list), the bottom layer has every vector linked to its close neighbours, and search walks from the top
down towards the query.

**23. How is the level of a new node chosen?**
level = floor(-ln(U) * mL), U uniform in (0, 1], mL = 1 / ln(M). So P(level >= l) = M^-l, each layer has
about 1/M of the nodes of the layer below. Expected number of layers is O(log n). Same idea as a skip list.

**24. Explain the search algorithm.**
Start at the entry point on the top layer. On every upper layer do greedy search: move to a neighbour if it
is closer to the query, stop at a local minimum, then go down one layer. On layer 0 do a beam search with
width ef: a MinHeap of candidates to expand (closest first) and a MaxHeap of the best ef results. Stop when
the closest candidate is worse than the worst result. Return the best k.

**25. What do M, ef_construction and ef_search control?**
M = links per node (layer 0 allows 2M). Bigger M = better recall, more memory, slower build.
ef_construction = beam width while inserting, better graph quality but slower build.
ef_search = beam width at query time. It is the main speed vs recall knob and can be changed without
rebuilding.

**26. What is the neighbour selection heuristic and why is it needed?**
Instead of just taking the M closest candidates, a candidate is only kept if it is closer to the new node
than to every neighbour already kept. This avoids all links pointing into the same cluster and keeps
"bridge" edges between clusters, so the graph stays navigable.

**27. What happens when a node has too many links?**
When a new node links to an existing node, the existing node may go over its limit (M or 2M). Then its
neighbour list is shrunk again with the same heuristic.

**28. Time and space complexity of HNSW?**
Search: about O(log n) hops, each costing O(M * d), so roughly O(ef * M * d + log n * M * d).
Insert: O(log n * ef_construction * M * d). Space: O(n * d) for vectors plus O(n * M) for edges.

**29. HNSW vs LSH, which is better and why?**
In my benchmark HNSW gives much higher recall at the same speed. LSH only uses the hash bits (a rough
partition of space) while HNSW uses the actual geometry of the data through its graph. LSH is simpler, has
cheap inserts, and has theoretical guarantees, but it needs many tables for high recall.

---

## F. Implementation and results

**30. Why is FAISS faster than your HNSW if the algorithm is the same?**
FAISS is written in C++ with SIMD distance code and no interpreter overhead. My code runs the graph walk and
the heap in Python, so every step has interpreter overhead. The number of distance computations is similar,
so recall is comparable, but the constant factor is much bigger. This is why the comparison focuses on
recall and on how time grows with n, not just raw speed.

**31. How did you test correctness?**
pytest: unit tests for heaps (pop order vs sorted), distance functions (batch vs single, known values),
edge cases (empty index, k > n, k = 0, wrong dimensions, NaN), recall of LSH/HNSW against brute force,
graph invariants (degree limits, nested layers, no self loops) and save/load round trips.

**32. How are indexes saved?**
A folder with a JSON header (class, version, parameters), an .npz file with NumPy arrays and a pickle file
with the graph/bucket dictionaries and the random generator state. With the RNG state saved, an index can
be loaded and extended and still build exactly the same graph.

**33. Why did LSH work well on SIFT (0.997 recall) but worse on the sentence embeddings (0.91)?**
For text embeddings the nearest neighbours are still far apart in angle (cosine similarity about 0.5, roughly
60 degrees). One hash bit then agrees with probability 1 - 60/180 = 0.67, so 10 bits all agree only
0.67^10 = 1.8% of the time per table. That is why LSH needed 32 tables there. HNSW is not affected because it
follows real neighbour links (0.995 recall on the same queries).

**34. What are your key results in one line?**
On SIFT-10K: HNSW recall@10 0.999 at ef=50 (1.000 at ef=100) using ~650 distance computations instead of
10,000, LSH 0.997 with 32 tables, and at 50K vectors our Python HNSW is 1.8x faster than NumPy brute force.

**35. What would you improve next?**
Write the hot loops in Cython/Numba or C++, add deletions, product quantization to compress vectors,
multithreaded queries and batched search, and an IVF index for comparison.
