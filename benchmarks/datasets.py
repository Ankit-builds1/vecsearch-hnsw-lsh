"""Dataset loading for the benchmarks.

Primary dataset: SIFT-10K ("siftsmall") from the INRIA TEXMEX corpus
    10,000 base vectors, 100 queries, 128 dims, with ground truth.
    http://corpus-texmex.irisa.fr/

If the download fails (no internet, FTP blocked, ...), we fall back to
synthetic data: Gaussian clusters in 128 dims with the same sizes, and
compute the ground truth ourselves with brute force.
"""

from __future__ import annotations

import io
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecsearch.distance import pairwise  # noqa: E402

DATA_DIR = ROOT / "data"
SIFT_URLS = [
    "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
]


@dataclass
class Dataset:
    name: str
    base: np.ndarray          # (n, d) float32
    queries: np.ndarray       # (m, d) float32
    ground_truth: np.ndarray  # (m, >=100) int64, true nearest neighbour ids
    metric: str = "l2"

    def __repr__(self) -> str:
        return (f"Dataset({self.name}: base={self.base.shape}, "
                f"queries={self.queries.shape}, gt={self.ground_truth.shape})")


# --------------------------------------------------------------------- fvecs reader

def read_fvecs(raw: bytes) -> np.ndarray:
    """Parse the .fvecs format: each row = int32 dim followed by dim float32 values."""
    a = np.frombuffer(raw, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)


def read_ivecs(raw: bytes) -> np.ndarray:
    """Parse the .ivecs format (same layout as fvecs but int32 values)."""
    a = np.frombuffer(raw, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().astype(np.int64)


# --------------------------------------------------------------------- SIFT

def _download(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed urls)
        return resp.read()


def load_sift10k(verbose: bool = True) -> Dataset:
    """Download (once) and load SIFT-10K. Raises RuntimeError if not possible."""
    folder = DATA_DIR / "siftsmall"
    files = {"base": "siftsmall_base.fvecs", "query": "siftsmall_query.fvecs",
             "gt": "siftsmall_groundtruth.ivecs"}

    if not all((folder / f).exists() for f in files.values()):
        folder.mkdir(parents=True, exist_ok=True)
        last_err: Exception | None = None
        for url in SIFT_URLS:
            try:
                if verbose:
                    print(f"downloading {url} ...")
                raw = _download(url)
                with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
                    for member in tar.getmembers():
                        name = Path(member.name).name
                        if name in files.values():
                            f = tar.extractfile(member)
                            assert f is not None
                            (folder / name).write_bytes(f.read())
                if all((folder / f).exists() for f in files.values()):
                    break
            except Exception as e:  # noqa: BLE001 - any failure -> try next / fallback
                last_err = e
                if verbose:
                    print(f"  failed: {e}")
        else:
            raise RuntimeError(f"could not download SIFT-10K: {last_err}")

    base = read_fvecs((folder / files["base"]).read_bytes())
    queries = read_fvecs((folder / files["query"]).read_bytes())
    gt = read_ivecs((folder / files["gt"]).read_bytes())
    return Dataset("SIFT-10K", base, queries, gt, "l2")


# --------------------------------------------------------------------- synthetic

def make_synthetic(n: int = 10_000, n_queries: int = 100, dim: int = 128,
                   n_clusters: int = 50, seed: int = 0) -> Dataset:
    """Clustered Gaussian data + brute force ground truth (top 100)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, size=(n_clusters, dim)).astype(np.float32) * 4
    labels = rng.integers(0, n_clusters, size=n + n_queries)
    X = centers[labels] + rng.normal(0, 1, size=(n + n_queries, dim)).astype(np.float32)
    base, queries = X[:n], X[n:]
    gt = np.argsort(pairwise(queries, base, "l2"), axis=1)[:, :100]
    return Dataset(f"Synthetic-{n // 1000}K-{dim}d", base, queries, gt, "l2")


def load_dataset(prefer: str = "sift", verbose: bool = True) -> Dataset:
    """SIFT-10K if possible, otherwise synthetic data of the same size."""
    if prefer == "sift":
        try:
            return load_sift10k(verbose)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"SIFT-10K unavailable ({e}); using synthetic fallback")
    return make_synthetic()


if __name__ == "__main__":
    print(load_dataset())
