"""Benchmark BruteForce vs LSH vs HNSW (ours) against FAISS.

Measures, for each index:
    build time (s), queries per second (single query at a time),
    Recall@10 against the dataset ground truth, memory (MB),
    distance computations per query (machine independent measure of work)

Experiments:
    1. main comparison table          -> results/comparison.csv
    2. HNSW sweep over M and ef_search -> results/hnsw_sweep.csv
    3. LSH sweep over tables and bits  -> results/lsh_sweep.csv
    4. FAISS HNSW ef_search sweep      -> results/faiss_hnsw_sweep.csv
    5. scaling with n up to 50K        -> results/scaling.csv (synthetic data, SIFT-10K is too small)
    + PNG charts for all of them, and results/summary.json

FAISS is pinned to 1 thread and queried one vector at a time, so it is
compared on the same terms as our single-threaded python code.

Usage:
    python benchmarks/run_benchmarks.py           # full run (~5-8 min)
    python benchmarks/run_benchmarks.py --quick   # smaller sweeps
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from datasets import Dataset, load_dataset  # noqa: E402
from vecsearch import BruteForceIndex, HNSWIndex, LSHIndex  # noqa: E402
from vecsearch.distance import pairwise  # noqa: E402

try:
    import faiss  # type: ignore

    faiss.omp_set_num_threads(1)
    HAVE_FAISS = True
except ImportError:  # pragma: no cover
    HAVE_FAISS = False

RESULTS = ROOT / "benchmarks" / "results"
K = 10


# ===================================================================== measuring

def deep_size(obj: Any, seen: set[int] | None = None) -> int:
    """Real memory used by a python object graph (numpy arrays counted by nbytes)."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return sys.getsizeof(obj) if obj.base is not None else obj.nbytes + 112
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_size(x, seen) for x in obj)
    elif hasattr(obj, "__dict__"):
        size += deep_size(vars(obj), seen)
    return size


def our_memory_mb(index) -> float:
    """Memory of one of our indexes (skips shared function references)."""
    keep = {k: v for k, v in vars(index).items() if not callable(v)}
    return deep_size(keep) / 2**20


def disk_mb(index) -> float:
    tmp = Path(tempfile.mkdtemp())
    try:
        index.save(tmp / "idx")
        return sum(f.stat().st_size for f in (tmp / "idx").iterdir()) / 2**20
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def recall(found: list[np.ndarray], gt: np.ndarray, k: int = K) -> float:
    hits = sum(len(set(f[:k].tolist()) & set(g[:k].tolist())) for f, g in zip(found, gt))
    return hits / (k * len(gt))


def time_queries(search: Callable[[np.ndarray], np.ndarray], queries: np.ndarray,
                 repeats: int = 1) -> tuple[list[np.ndarray], float]:
    """Run every query one at a time. Returns (results, queries per second)."""
    for q in queries[:5]:  # warm-up
        search(q)
    best = float("inf")
    found: list[np.ndarray] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        found = [search(q) for q in queries]
        best = min(best, time.perf_counter() - t0)
    return found, len(queries) / best


# ===================================================================== FAISS wrappers

def faiss_search_fn(index, k: int = K) -> Callable[[np.ndarray], np.ndarray]:
    def fn(q: np.ndarray) -> np.ndarray:
        _, ids = index.search(q[None, :].astype(np.float32), k)
        return ids[0]
    return fn


def faiss_mb(index) -> float:
    return faiss.serialize_index(index).nbytes / 2**20


# ===================================================================== experiments

def run_main(ds: Dataset, quick: bool) -> pd.DataFrame:
    d = ds.base.shape[1]
    rows = []

    def ours(name: str, make: Callable[[], Any], efs: list[int] | None = None, repeats: int = 3) -> None:
        """Build once; for HNSW report one row per ef_search value (same graph)."""
        print(f"  {name} ...", flush=True)
        t0 = time.perf_counter()
        idx = make()
        idx.add(ds.base)
        build = time.perf_counter() - t0
        mem, disk = our_memory_mb(idx), disk_mb(idx)
        for ef in efs or [None]:
            if ef is not None:
                idx.ef_search = ef
            label = f"{name}, ef={ef})" if ef is not None else name
            found, qps = time_queries(lambda q: idx.search(q, K)[0], ds.queries, repeats)
            rows.append({
                "method": label, "library": "VecSearch (ours)", "build_s": build, "qps": qps,
                "latency_ms": 1000 / qps, f"recall@{K}": recall(found, ds.ground_truth),
                "dist_per_query": distances_per_query(idx, ds.queries),
                "memory_mb": mem, "disk_mb": disk,
            })

    ours("BruteForce", lambda: BruteForceIndex(d))
    # L=32, K=12 was the most accurate LSH setting in the sweep that still scans < 30% of the data
    ours("LSH (L=32, K=12)", lambda: LSHIndex(d, n_tables=32, n_bits=12, multi_probe=True))
    ours("HNSW (M=16", lambda: HNSWIndex(d, M=16, ef_construction=100 if quick else 200),
         efs=[50, 100], repeats=2)

    if HAVE_FAISS:
        def theirs(name: str, make: Callable[[], Any], train: bool = False) -> None:
            print(f"  {name} ...", flush=True)
            t0 = time.perf_counter()
            idx = make()
            if train:
                idx.train(ds.base)
            idx.add(ds.base)
            build = time.perf_counter() - t0
            found, qps = time_queries(faiss_search_fn(idx), ds.queries, repeats=5)
            mb = faiss_mb(idx)
            dpq = float("nan")
            if isinstance(idx, faiss.IndexFlat):
                dpq = float(len(ds.base))
            elif isinstance(idx, faiss.IndexHNSWFlat):
                try:  # FAISS keeps a global counter of HNSW distance computations
                    faiss.cvar.hnsw_stats.reset()
                    faiss_search_fn(idx)(ds.queries[0])
                    faiss.cvar.hnsw_stats.reset()
                    for q in ds.queries:
                        faiss_search_fn(idx)(q)
                    dpq = faiss.cvar.hnsw_stats.ndis / len(ds.queries)
                except AttributeError:
                    pass
            rows.append({
                "method": name, "library": "FAISS", "build_s": build, "qps": qps,
                "latency_ms": 1000 / qps, f"recall@{K}": recall(found, ds.ground_truth),
                "dist_per_query": dpq, "memory_mb": mb, "disk_mb": mb,
            })

        theirs("FAISS Flat", lambda: faiss.IndexFlatL2(d))
        theirs("FAISS LSH (1024 bits)", lambda: faiss.IndexLSH(d, 1024), train=True)

        def faiss_hnsw():
            h = faiss.IndexHNSWFlat(d, 16)
            h.hnsw.efConstruction = 100 if quick else 200
            h.hnsw.efSearch = 50
            return h

        theirs("FAISS HNSW (M=16, ef=50)", faiss_hnsw)

    return pd.DataFrame(rows)


def distances_per_query(index, queries: np.ndarray) -> float:
    """Average number of query-to-vector distance computations (machine independent work)."""
    if isinstance(index, BruteForceIndex):
        return float(len(index))
    if isinstance(index, LSHIndex):
        return float(np.mean([len(index.candidates(q)) for q in queries]))
    if isinstance(index, HNSWIndex):
        index.distance_count = 0
        for q in queries:
            index.search(q, K)
        return index.distance_count / len(queries)
    return float("nan")


def run_hnsw_sweep(ds: Dataset, quick: bool) -> pd.DataFrame:
    """Sweep M (build) x ef_search (query)."""
    d = ds.base.shape[1]
    Ms = [8, 16] if quick else [4, 8, 16, 32]
    efs = [10, 20, 50, 100] if quick else [10, 20, 40, 80, 160, 320]
    sweep = []
    for M in Ms:
        print(f"  HNSW M={M} ...", flush=True)
        idx = HNSWIndex(d, M=M, ef_construction=100)
        t0 = time.perf_counter()
        idx.add(ds.base)
        build = time.perf_counter() - t0
        for ef in efs:
            found, qps = time_queries(lambda q: idx.search(q, K, ef=ef)[0], ds.queries, repeats=5)
            idx.distance_count = 0
            for q in ds.queries:
                idx.search(q, K, ef=ef)
            sweep.append({"M": M, "ef_construction": 100, "ef_search": ef, "build_s": build,
                          "qps": qps, "latency_ms": 1000 / qps,
                          f"recall@{K}": recall(found, ds.ground_truth),
                          "dist_per_query": idx.distance_count / len(ds.queries),
                          "memory_mb": our_memory_mb(idx)})
    return pd.DataFrame(sweep)


def run_scaling(quick: bool) -> tuple[pd.DataFrame, str]:
    """How query time and work grow with n: brute force vs HNSW (M=16, ef=50).

    Uses synthetic clustered 128-d data so we can go past the 10K vectors of SIFT-10K.
    The HNSW index is built incrementally and measured at each checkpoint.
    """
    from datasets import make_synthetic

    n_max = 10_000 if quick else 50_000
    ds = make_synthetic(n=n_max, n_queries=100, dim=128, n_clusters=100, seed=1)
    checkpoints = [c for c in (1000, 2000, 5000, 10_000, 20_000, 50_000) if c <= n_max]
    d = ds.base.shape[1]
    hnsw = HNSWIndex(d, M=16, ef_construction=100, ef_search=50)
    brute = BruteForceIndex(d)
    rows, start, build = [], 0, 0.0
    for n in checkpoints:
        print(f"  scaling n={n:,} ...", flush=True)
        chunk = ds.base[start:n]
        t0 = time.perf_counter()
        hnsw.add(chunk)
        build += time.perf_counter() - t0
        brute.add(chunk)
        start = n
        gt = np.argsort(pairwise(ds.queries, ds.base[:n]), axis=1)[:, :K]
        found, qps_h = time_queries(lambda q: hnsw.search(q, K)[0], ds.queries, repeats=3)
        _, qps_b = time_queries(lambda q: brute.search(q, K)[0], ds.queries, repeats=3)
        rows.append({"n": n, "hnsw_ms": 1000 / qps_h, "brute_ms": 1000 / qps_b,
                     "hnsw_recall": recall(found, gt),
                     "hnsw_dist_per_query": distances_per_query(hnsw, ds.queries),
                     "brute_dist_per_query": float(n), "hnsw_build_s_total": build,
                     "max_level": hnsw.max_level})
    return pd.DataFrame(rows), "synthetic clustered 128-d data"


def run_lsh_sweep(ds: Dataset, quick: bool) -> pd.DataFrame:
    d = ds.base.shape[1]
    rows = []
    tables = [4, 16] if quick else [2, 4, 8, 16, 32]
    bits = [8, 12, 16]
    print("  LSH sweep ...", flush=True)
    for L in tables:
        for Kb in bits:
            t0 = time.perf_counter()
            idx = LSHIndex(d, n_tables=L, n_bits=Kb, multi_probe=True)
            idx.add(ds.base)
            build = time.perf_counter() - t0
            found, qps = time_queries(lambda q: idx.search(q, K)[0], ds.queries, repeats=3)
            cand = float(np.mean([len(idx.candidates(q)) for q in ds.queries]))
            rows.append({"n_tables": L, "n_bits": Kb, "build_s": build, "qps": qps,
                         "latency_ms": 1000 / qps, f"recall@{K}": recall(found, ds.ground_truth),
                         "avg_candidates": cand, "scanned_pct": 100 * cand / len(ds.base)})
    return pd.DataFrame(rows)


def run_faiss_hnsw_sweep(ds: Dataset, quick: bool) -> pd.DataFrame:
    if not HAVE_FAISS:
        return pd.DataFrame()
    d = ds.base.shape[1]
    h = faiss.IndexHNSWFlat(d, 16)
    h.hnsw.efConstruction = 100
    h.add(ds.base)
    rows = []
    for ef in ([10, 20, 50, 100] if quick else [10, 20, 40, 80, 160, 320]):
        h.hnsw.efSearch = ef
        found, qps = time_queries(faiss_search_fn(h), ds.queries, repeats=5)
        rows.append({"M": 16, "ef_search": ef, "qps": qps, "latency_ms": 1000 / qps,
                     f"recall@{K}": recall(found, ds.ground_truth)})
    return pd.DataFrame(rows)


# ===================================================================== charts

# chart colours: first three categorical slots (validated colour-blind safe as a set)
C_OURS, C_FAISS, C_LSH = "#2a78d6", "#eb6834", "#1baf7a"
C_NEUTRAL = "#8a8985"
M_RAMP = {4: "#86b6ef", 8: "#3987e5", 16: "#1c5cab", 32: "#0d366b"}  # ordinal blue ramp
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False,
    })


def _label_end(ax, x, y, text, color=INK2, dx=6, dy=0) -> None:
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                color=color, fontsize=9, va="center")


def make_charts(main: pd.DataFrame, sweep: pd.DataFrame, lsh: pd.DataFrame,
                fh: pd.DataFrame, scaling: pd.DataFrame, ds_name: str,
                scaling_name: str = "synthetic data") -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    rk = f"recall@{K}"
    out = []

    # 1. recall vs QPS trade-off -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    s16 = sweep[sweep.M == 16].sort_values("ef_search")
    ax.plot(s16[rk], s16.qps, "-o", color=C_OURS, label="HNSW, ours (M=16)",
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    for _, r in s16.iterrows():
        ax.annotate(f"ef={int(r.ef_search)}", (r[rk], r.qps), xytext=(-7, 0),
                    textcoords="offset points", fontsize=7.5, color=INK2, ha="right", va="center")
    if len(fh):
        fh2 = fh.sort_values("ef_search")
        ax.plot(fh2[rk], fh2.qps, "-s", color=C_FAISS, label="HNSW, FAISS (M=16)",
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        _label_end(ax, fh2[rk].iloc[-1], fh2.qps.iloc[-1], "FAISS HNSW", INK)
    ax.scatter(lsh[rk], lsh.qps, marker="^", s=55, color=C_LSH, edgecolor=SURFACE,
               linewidth=1.2, label="LSH, ours (L x K grid)", zorder=3)
    ax.annotate("LSH (ours)", (lsh[rk].min(), lsh.qps[lsh[rk].idxmin()]), xytext=(0, 10),
                textcoords="offset points", fontsize=9, color=INK, ha="center")
    bf = main[main.method == "BruteForce"].iloc[0]
    ax.scatter([bf[rk]], [bf.qps], marker="D", s=70, color=C_NEUTRAL, zorder=4, label="Brute force, ours")
    ax.annotate("Brute force\n(exact)", (bf[rk], bf.qps), xytext=(10, 0), textcoords="offset points",
                fontsize=9, color=INK2, va="center")
    ax.annotate("Our HNSW", (s16[rk].iloc[0], s16.qps.iloc[0]), xytext=(-10, 12),
                textcoords="offset points", fontsize=9, color=INK, ha="right")
    ax.set_xlim(right=1.07)
    ax.set_yscale("log")
    ax.set_xlabel(f"Recall@{K}")
    ax.set_ylabel("Queries per second (log scale, 1 thread)")
    ax.set_title(f"Recall vs speed on {ds_name}")
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "recall_vs_qps.png", dpi=150)
    plt.close(fig)
    out.append("recall_vs_qps.png")

    # 2. HNSW M sweep ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for M, g in sweep.groupby("M"):
        g = g.sort_values("ef_search")
        c = M_RAMP.get(int(M), C_OURS)
        for ax, col in ((ax1, rk), (ax2, "latency_ms")):
            ax.plot(g.ef_search, g[col], "-o", color=c, label=f"M = {int(M)}",
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        # direct labels on the left panel where curves are still apart (ef = 10)
        ax1.annotate(f"M={int(M)}", (g.ef_search.iloc[0], g[rk].iloc[0]), xytext=(-8, 0),
                     textcoords="offset points", ha="right", va="center", fontsize=8.5, color=INK)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks(sorted(sweep.ef_search.unique()))
        ax.set_xticklabels([str(int(e)) for e in sorted(sweep.ef_search.unique())])
        ax.minorticks_off()
        ax.set_xlabel("ef_search (log)")
    ax1.set_ylabel(f"Recall@{K}")
    ax1.set_title("Accuracy")
    ax1.margins(x=0.15)
    ax2.set_ylabel("Query latency (ms)")
    ax2.set_title("Speed")
    ax2.legend(loc="upper left", fontsize=9)
    fig.suptitle("HNSW parameter sweep: M (links per node) and ef_search (beam width)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "hnsw_param_sweep.png", dpi=150)
    plt.close(fig)
    out.append("hnsw_param_sweep.png")

    # 3. comparison bars (small multiples, one measure each) ---------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    m = main.iloc[::-1]  # first method on top
    panels = [("build_s", "Build time (s)", "{:.3g}", True), ("qps", "Queries / second", "{:,.0f}", True),
              (rk, f"Recall@{K}", "{:.3f}", False), ("memory_mb", "Memory (MB)", "{:.1f}", False)]
    for ax, (col, title, fmt, logx) in zip(axes.flat, panels):
        colors = [C_FAISS if lib == "FAISS" else C_OURS for lib in m.library]
        ax.barh(m.method, m[col], color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        for y, v in enumerate(m[col]):
            ax.annotate(fmt.format(v), (v, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        if logx:
            ax.set_xscale("log")
        if col == rk:
            ax.set_xlim(0, 1.15)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_OURS),
                        plt.Rectangle((0, 0), 1, 1, color=C_FAISS)],
               labels=["VecSearch (ours)", "FAISS"], loc="upper right", ncol=2, fontsize=9)
    fig.suptitle(f"Index comparison on {ds_name} (k={K}, 1 thread)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RESULTS / "comparison_bars.png", dpi=150)
    plt.close(fig)
    out.append("comparison_bars.png")

    # 4. scaling with n (two panels: wall clock time, and machine independent work) --
    if len(scaling):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sc = scaling.sort_values("n")
        panels = [(axes[0], "brute_ms", "hnsw_ms", "Query latency (ms)", "Time per query"),
                  (axes[1], "brute_dist_per_query", "hnsw_dist_per_query",
                   "Distance computations per query (log)", "Work per query")]
        for ax, bcol, hcol, ylabel, title in panels:
            ax.plot(sc.n, sc[bcol], "-D", color=C_NEUTRAL, label="Brute force  O(n)",
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.plot(sc.n, sc[hcol], "-o", color=C_OURS, label="HNSW (M=16, ef=50)  ~O(log n)",
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            _label_end(ax, sc.n.iloc[-1], sc[bcol].iloc[-1], "Brute force", INK)
            _label_end(ax, sc.n.iloc[-1], sc[hcol].iloc[-1], "HNSW", INK)
            ax.set_xscale("log")
            ax.set_xlabel("Number of vectors n (log)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.margins(x=0.18)
        axes[1].set_yscale("log")
        axes[0].legend(loc="upper left", fontsize=8.5)
        fig.suptitle(f"Scaling with dataset size ({scaling_name})", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(RESULTS / "scaling.png", dpi=150)
        plt.close(fig)
        out.append("scaling.png")

    # 5. LSH: recall vs fraction scanned -----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ramp = {8: "#86b6ef", 12: "#3987e5", 16: "#1c5cab"}
    for Kb, g in lsh.groupby("n_bits"):
        g = g.sort_values("n_tables")
        ax.plot(g.scanned_pct, g[rk], "-^", color=ramp.get(int(Kb), C_LSH), label=f"K = {int(Kb)} bits",
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        _label_end(ax, g.scanned_pct.iloc[-1], g[rk].iloc[-1], f"K={int(Kb)}", INK)
    ax.set_xscale("log")
    ax.set_xlabel("% of dataset scanned per query (log)")
    ax.set_ylabel(f"Recall@{K}")
    ax.set_title("LSH: more tables (L) -> more candidates -> higher recall")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "lsh_sweep.png", dpi=150)
    plt.close(fig)
    out.append("lsh_sweep.png")
    return out


# ===================================================================== main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller parameter sweeps")
    ap.add_argument("--synthetic", action="store_true", help="skip SIFT, use synthetic data")
    ap.add_argument("--charts-only", action="store_true", help="redraw charts from the saved CSVs")
    ap.add_argument("--only", nargs="+", choices=["main", "hnsw", "lsh", "faiss", "scaling"],
                    help="re-run only these experiments, reuse the saved CSVs for the rest")
    ap.add_argument("--out", type=Path, default=None,
                    help="write results to this folder instead of benchmarks/results")
    args = ap.parse_args()

    global RESULTS
    if args.out is not None:
        RESULTS = args.out.resolve()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.charts_only:
        summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
        read = lambda name: pd.read_csv(RESULTS / name)  # noqa: E731
        charts = make_charts(read("comparison.csv"), read("hnsw_sweep.csv"), read("lsh_sweep.csv"),
                             read("faiss_hnsw_sweep.csv"), read("scaling.csv"), summary["dataset"],
                             "synthetic clustered 128-d data")
        print("redrew", charts)
        return
    ds = load_dataset("synthetic" if args.synthetic else "sift")
    print(ds)
    t_all = time.perf_counter()

    # each experiment: (key, csv file, function). With --only, the others are read from disk.
    experiments = [
        ("main", "comparison.csv", lambda: run_main(ds, args.quick)),
        ("hnsw", "hnsw_sweep.csv", lambda: run_hnsw_sweep(ds, args.quick)),
        ("lsh", "lsh_sweep.csv", lambda: run_lsh_sweep(ds, args.quick)),
        ("faiss", "faiss_hnsw_sweep.csv", lambda: run_faiss_hnsw_sweep(ds, args.quick)),
        ("scaling", "scaling.csv", lambda: run_scaling(args.quick)[0]),
    ]
    frames = {}
    for i, (key, csv_name, fn) in enumerate(experiments, 1):
        if args.only and key not in args.only:
            frames[key] = pd.read_csv(RESULTS / csv_name)
            continue
        print(f"[{i}/{len(experiments)}] {key}")
        frames[key] = fn()
        frames[key].to_csv(RESULTS / csv_name, index=False, float_format="%.4f")
    main_df = frames["main"]
    charts = make_charts(main_df, frames["hnsw"], frames["lsh"], frames["faiss"], frames["scaling"],
                         ds.name, "synthetic clustered 128-d data")

    summary = {
        "dataset": ds.name, "n": int(ds.base.shape[0]), "dim": int(ds.base.shape[1]),
        "n_queries": int(ds.queries.shape[0]), "k": K, "quick": args.quick,
        "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
        "faiss": getattr(faiss, "__version__", None) if HAVE_FAISS else None,
        "total_runtime_s": round(time.perf_counter() - t_all, 1),
        "comparison": main_df.round(4).to_dict(orient="records"), "charts": charts,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.set_option("display.width", 160)
    print("\n" + main_df.round(3).to_string(index=False))
    print(f"\nsaved CSVs + {len(charts)} charts to {RESULTS} in {summary['total_runtime_s']} s")


if __name__ == "__main__":
    main()
