"""Hierarchical Navigable Small World (HNSW) graph index.

Based on: Malkov & Yashunin, "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs" (2016).

How it works (short version):

* Every vector is a node. Each node gets a random level L drawn from an
  exponential distribution, so most nodes only live on layer 0, a few reach
  layer 1, fewer reach layer 2 ... (just like a skip list).
* On every layer a node is linked to (at most) M close neighbours.
  Layer 0 allows 2*M links.
* Search starts at the single entry point on the top layer. On upper layers
  we walk greedily towards the query (move to a neighbour if it is closer).
  On layer 0 we do a beam search that keeps the ``ef`` best nodes found so
  far, using a MinHeap of candidates to expand and a MaxHeap of results.
* Neighbours are chosen with the "heuristic" from the paper: a candidate is
  only kept if it is closer to the new node than to any neighbour already
  kept. This spreads links in different directions and keeps the graph
  navigable in clustered data.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import BaseIndex
from .distance import pairwise
from .heap import MaxHeap, MinHeap


class HNSWIndex(BaseIndex):
    """HNSW approximate nearest neighbour index.

    n = number of vectors, d = dimension, M = links per node.

    add (one vector):  O(log n * ef_construction * M * d) expected
    search:            O(log n * M * d) for the upper layers +
                       O(ef_search * M * d) on layer 0  ->  roughly O(log n) hops
    space:             O(n * d) vectors + O(n * M) edges

    Args:
        dim: vector dimension.
        metric: "l2", "cosine" or "ip".
        M: number of neighbours chosen per node per layer (layer 0 keeps up to 2*M).
        ef_construction: beam width used while building. Higher = better graph, slower build.
        ef_search: beam width used while searching. Higher = better recall, slower query.
        keep_pruned: after the heuristic, top up the neighbour list with pruned
            candidates until it has M entries (``keepPrunedConnections`` in the paper).
        seed: seed for the random level generator.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        keep_pruned: bool = True,
        seed: int = 42,
    ) -> None:
        """Validate parameters and create an empty graph. O(1)."""
        super().__init__(dim, metric)
        if M < 2:
            raise ValueError("M must be at least 2")
        if ef_construction < 1 or ef_search < 1:
            raise ValueError("ef_construction and ef_search must be >= 1")
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.keep_pruned = keep_pruned
        self.seed = seed
        self._mL = 1.0 / math.log(M)
        self._rng = np.random.default_rng(seed)

        # _layers[l][node] = list of neighbour positions on layer l
        self._layers: list[dict[int, list[int]]] = []
        self._levels: list[int] = []
        self._entry: int | None = None
        self._max_level: int = -1
        # how many query-to-node distances were computed (for benchmarks; reset freely)
        self.distance_count = 0

    # ================================================================ helpers
    def _random_level(self) -> int:
        """Draw a level from floor(-ln(U) * mL), U ~ Uniform(0, 1]. O(1)."""
        u = 1.0 - self._rng.random()  # in (0, 1], avoids log(0)
        return int(-math.log(u) * self._mL)

    def _dists(self, q: np.ndarray, nodes: list[int]) -> list[float]:
        """Distances from q to a list of nodes in one vectorized call. O(len(nodes) * d).

        For L2 the direct ``sqrt(sum((x - q)^2))`` is used because neighbour
        lists are short (<= 2M rows) and it needs fewer numpy calls than the
        norm expansion in ``euclidean_batch``.
        """
        self.distance_count += len(nodes)
        X = self._store.data[nodes]
        if self.metric == "l2":
            diff = X - q
            return np.sqrt(np.einsum("ij,ij->i", diff, diff)).tolist()
        return self._batch_dist(q, X).tolist()

    def _greedy_search(self, q: np.ndarray, entry: int, entry_dist: float, layer: int,
                       trace: list[tuple[int, int]] | None = None) -> tuple[int, float]:
        """Greedy walk on one upper layer: keep moving to the closest neighbour.

        Stops when no neighbour is closer than the current node (local minimum).

        Time: O(hops * M * d), hops is small (about log n overall).
        Space: O(M).
        """
        cur, cur_d = entry, entry_dist
        graph = self._layers[layer]
        if trace is not None:
            trace.append((layer, cur))
        changed = True
        while changed:
            changed = False
            neigh = graph.get(cur, [])
            if not neigh:
                break
            for n, d in zip(neigh, self._dists(q, neigh)):
                if d < cur_d:
                    cur, cur_d, changed = n, d, True
            if changed and trace is not None:
                trace.append((layer, cur))
        return cur, cur_d

    def _search_layer(self, q: np.ndarray, entry_points: list[tuple[float, int]], ef: int,
                      layer: int, visited_out: list[int] | None = None) -> list[tuple[float, int]]:
        """Beam search on one layer (Algorithm 2 in the paper).

        candidates: MinHeap -> always expand the closest unexpanded node.
        results:    MaxHeap of size ef -> top is the worst of the current best.
        Stop when the closest candidate is further than the worst result.

        Time: O(ef * M * d) roughly (each expansion = one batched distance call)
        Space: O(ef + visited).

        Returns:
            list of (distance, node) sorted ascending, length <= ef.
        """
        graph = self._layers[layer]
        visited: set[int] = {n for _, n in entry_points}
        candidates: MinHeap[int] = MinHeap(entry_points)
        results: MaxHeap[int] = MaxHeap(entry_points)
        while len(results) > ef:
            results.pop()

        worst = results.peek_key()
        while candidates:
            c_dist, c = candidates.pop()
            if c_dist > worst:
                break  # everything left is worse than our worst result
            if visited_out is not None:
                visited_out.append(c)
            neigh = [n for n in graph.get(c, ()) if n not in visited]
            if not neigh:
                continue
            visited.update(neigh)
            for n, d in zip(neigh, self._dists(q, neigh)):
                if d < worst or len(results) < ef:
                    candidates.push(d, n)
                    if len(results) < ef:
                        results.push(d, n)
                    else:
                        results.pushpop(d, n)  # drop the current worst
                    worst = results.peek_key()

        return list(reversed(results.to_sorted_list()))

    def _select_neighbors(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Neighbour selection heuristic (Algorithm 4 in the paper).

        Walk through candidates from closest to furthest. Keep candidate e only
        if it is closer to the base node than to every already kept neighbour;
        otherwise e is "covered" by a kept neighbour and gets pruned.

        Time: O(C^2 * d) to build the CxC distance matrix + O(C * m) for the loop.
        Space: O(C^2).

        Args:
            candidates: (distance to base node, node), sorted ascending.
            m: max neighbours to return.
        """
        if len(candidates) <= m:
            return [n for _, n in candidates]
        nodes = [n for _, n in candidates]
        vecs = self._store.data[nodes]
        # plain python lists are much faster to index one element at a time
        between = pairwise(vecs, vecs, self.metric).tolist()

        kept: list[int] = []      # positions inside `nodes`
        pruned: list[int] = []
        for i, (d_base, _) in enumerate(candidates):
            if len(kept) >= m:
                break
            row = between[i]
            for j in kept:
                if row[j] <= d_base:  # e is closer to a kept neighbour -> prune
                    pruned.append(i)
                    break
            else:
                kept.append(i)

        if self.keep_pruned:
            for i in pruned:
                if len(kept) >= m:
                    break
                kept.append(i)
        return [nodes[i] for i in kept]

    def _shrink(self, node: int, layer: int, m_max: int) -> None:
        """Re-select ``node``'s neighbours on ``layer`` if it has more than m_max.

        Time: O(m_max^2 * d)   Space: O(m_max^2).
        """
        neigh = self._layers[layer][node]
        if len(neigh) <= m_max:
            return
        d = self._dists(self._store.data[node], neigh)
        cand = sorted(zip(d, neigh))
        self._layers[layer][node] = self._select_neighbors(cand, m_max)

    def _insert(self, pos: int) -> None:
        """Insert the vector at internal position ``pos`` into the graph (Algorithm 1).

        Time: O(log n * ef_construction * M * d) expected.
        Space: O(M * level) new edges.
        """
        q = self._store.data[pos]
        level = self._random_level()
        self._levels.append(level)
        while len(self._layers) <= level:
            self._layers.append({})

        if self._entry is None:
            for l in range(level + 1):
                self._layers[l][pos] = []
            self._entry, self._max_level = pos, level
            return

        # 1) greedy descent through the layers above the new node's level
        ep = self._entry
        ep_d = self._dists(q, [ep])[0]
        for l in range(self._max_level, level, -1):
            ep, ep_d = self._greedy_search(q, ep, ep_d, l)

        # 2) on each layer from min(level, max_level) down to 0, find + link neighbours
        entry_points = [(ep_d, ep)]
        for l in range(min(level, self._max_level), -1, -1):
            found = self._search_layer(q, entry_points, self.ef_construction, l)
            neighbours = self._select_neighbors(found, self.M)
            self._layers[l][pos] = list(neighbours)
            m_max = self.M_max0 if l == 0 else self.M
            for n in neighbours:
                self._layers[l][n].append(pos)
                self._shrink(n, l, m_max)
            entry_points = found

        # new node is higher than everything so far -> becomes the entry point
        if level > self._max_level:
            for l in range(self._max_level + 1, level + 1):
                self._layers[l][pos] = []
            self._entry, self._max_level = pos, level

    # ================================================================ API
    def add(self, vectors: np.ndarray, ids: np.ndarray | list[int] | None = None) -> None:
        """Insert vectors one by one into the graph.

        Time: O(m * log n * ef_construction * M * d) expected.
        Space: O(m * (d + M)).
        """
        X = self._check_vectors(vectors)
        if X.shape[0] == 0:
            return
        new_ids = self._make_ids(X.shape[0], ids)
        start = len(self._store)
        self._store.append(X)
        self._ids.extend(new_ids)
        for pos in range(start, start + X.shape[0]):
            self._insert(pos)

    def search(self, query: np.ndarray, k: int = 10, ef: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Approximate k nearest neighbours (Algorithm 5).

        Args:
            query: (d,) vector.
            k: number of results.
            ef: beam width for this query, defaults to ``self.ef_search``.
                Always at least k.

        Time: O(log n * M * d + ef * M * d)   Space: O(ef + visited).
        """
        q = self._check_query(query)
        if self._entry is None or k <= 0:
            return self._empty_result()
        found = self._search(q, max(ef or self.ef_search, k))
        best = found[:k]
        ids = np.asarray(self._ids, dtype=np.int64)[[n for _, n in best]]
        return ids, np.array([d for d, _ in best], dtype=np.float32)

    def _search(self, q: np.ndarray, ef: int, trace: list[tuple[int, int]] | None = None,
                visited_out: list[int] | None = None) -> list[tuple[float, int]]:
        """Shared search routine (greedy on upper layers, beam on layer 0).

        Time: O(log n * M * d + ef * M * d)   Space: O(ef + visited).
        """
        assert self._entry is not None
        ep = self._entry
        ep_d = self._dists(q, [ep])[0]
        for l in range(self._max_level, 0, -1):
            ep, ep_d = self._greedy_search(q, ep, ep_d, l, trace)
        if trace is not None:
            trace.append((0, ep))
        return self._search_layer(q, [(ep_d, ep)], ef, 0, visited_out)

    def search_with_trace(self, query: np.ndarray, k: int = 10, ef: int | None = None) -> dict[str, Any]:
        """Like ``search`` but also returns the path taken (for visualization).

        Time: same as ``search``, O(log n * M * d + ef * M * d). Space: O(ef + visited).

        Returns a dict with:
            ids, distances: the results (internal positions == ids if default ids used)
            path: list of (layer, node) visited by the greedy descent
            expanded: nodes expanded on layer 0, in order
        """
        q = self._check_query(query)
        if self._entry is None:
            ids, dists = self._empty_result()
            return {"ids": ids, "distances": dists, "path": [], "expanded": []}
        trace: list[tuple[int, int]] = []
        expanded: list[int] = []
        found = self._search(q, max(ef or self.ef_search, k), trace, expanded)[:k]
        return {
            "ids": np.asarray(self._ids, dtype=np.int64)[[n for _, n in found]],
            "distances": np.array([d for d, _ in found], dtype=np.float32),
            "path": trace,
            "expanded": expanded,
        }

    # ================================================================ introspection
    @property
    def max_level(self) -> int:
        """Highest layer in the graph (-1 when empty). O(1)."""
        return self._max_level

    @property
    def entry_point(self) -> int | None:
        """Internal position of the entry point node. O(1)."""
        return self._entry

    def layer_edges(self, layer: int) -> list[tuple[int, int]]:
        """All directed edges (u, v) on one layer, mostly for plotting. O(edges)."""
        if layer >= len(self._layers):
            return []
        return [(u, v) for u, neigh in self._layers[layer].items() for v in neigh]

    def layer_nodes(self, layer: int) -> list[int]:
        """Nodes present on a layer. O(nodes on layer)."""
        if layer >= len(self._layers):
            return []
        return list(self._layers[layer].keys())

    def graph_stats(self) -> dict[str, Any]:
        """Nodes and average out-degree per layer. O(n * M)."""
        out = []
        for l, g in enumerate(self._layers):
            deg = [len(v) for v in g.values()]
            out.append({"layer": l, "nodes": len(g), "avg_degree": float(np.mean(deg)) if deg else 0.0})
        return {"max_level": self._max_level, "layers": out}

    # ================================================================ persistence
    def _get_state(self) -> dict[str, Any]:
        """Params, vectors/ids and the graph (layers, levels, entry, RNG state). O(n * d)."""
        return {
            "params": {
                "dim": self.dim,
                "metric": self.metric,
                "M": self.M,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "keep_pruned": self.keep_pruned,
                "seed": self.seed,
            },
            "arrays": {"vectors": self.vectors.copy(), "ids": self.ids},
            "extra": {
                "layers": self._layers,
                "levels": self._levels,
                "entry": self._entry,
                "max_level": self._max_level,
                "rng_state": self._rng.bit_generator.state,
            },
        }

    def _set_state(self, state: dict[str, Any]) -> None:
        """Restore vectors and graph saved by ``_get_state``. O(n * d)."""
        arrays, extra = state["arrays"], state["extra"]
        self._store.append(arrays["vectors"].astype(np.float32))
        self._ids = [int(i) for i in arrays["ids"]]
        self._layers = extra["layers"]
        self._levels = list(extra["levels"])
        self._entry = extra["entry"]
        self._max_level = int(extra["max_level"])
        self._rng.bit_generator.state = extra["rng_state"]
