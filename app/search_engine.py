"""Backend for the semantic search demo (no Streamlit code in here).

Pipeline:
    sentences --(all-MiniLM-L6-v2)--> 384-d embeddings --> our indexes --> top-k

Corpus: first 5,000 headlines of the AG News test set (4 categories). If the
download fails we generate a small synthetic headline corpus instead.
Embeddings + built indexes are cached in app/cache/ so the app starts fast
after the first run (the cache uses our own save/load code).
"""

from __future__ import annotations

import csv
import html
import io
import os
import re
import sys
import time
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecsearch import BaseIndex, BruteForceIndex, HNSWIndex, LSHIndex  # noqa: E402

CACHE = ROOT / "app" / "cache"
AG_NEWS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
AG_LABELS = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_NAMES = ("HNSW", "LSH", "BruteForce")


@dataclass
class Corpus:
    texts: list[str]
    labels: list[str]
    source: str


# ===================================================================== corpus

def _synthetic_headlines(n: int, seed: int = 0) -> Corpus:
    """Template-based fallback corpus, used only if AG News can't be downloaded."""
    rng = np.random.default_rng(seed)
    topics = {
        "Sports": (["Local team", "Star striker", "Olympic sprinter", "Veteran coach", "Rookie pitcher"],
                   ["wins championship", "breaks world record", "signs new contract", "injured before final",
                    "retires after long career"]),
        "Business": (["Tech giant", "Oil prices", "Central bank", "Retail chain", "Startup"],
                     ["reports record profit", "cuts interest rates", "announces layoffs", "raises funding",
                      "shares fall sharply"]),
        "Sci/Tech": (["Researchers", "Space agency", "New smartphone", "AI model", "Chip maker"],
                     ["discover new planet", "launches satellite", "beats human experts", "unveils faster processor",
                      "fixes security flaw"]),
        "World": (["Prime minister", "United Nations", "Protesters", "Election officials", "Rescue teams"],
                  ["call for ceasefire", "sign peace deal", "gather in capital", "count final votes",
                   "reach flood victims"]),
    }
    places = ["in Europe", "in Asia", "this week", "amid criticism", "after talks", "", "on Monday", "in India"]
    texts, labels = [], []
    names = list(topics)
    for _ in range(n):
        lab = names[rng.integers(len(names))]
        subj, verb = topics[lab]
        texts.append(f"{subj[rng.integers(len(subj))]} {verb[rng.integers(len(verb))]} "
                     f"{places[rng.integers(len(places))]}".strip())
        labels.append(lab)
    return Corpus(texts, labels, "synthetic headlines (AG News download failed)")


def clean_text(text: str) -> str:
    """AG News stores some characters as broken HTML entities, e.g. "NASA #39;s" -> "NASA's"."""
    text = re.sub(r"\s?#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = html.unescape(text.replace("\\", " "))
    return re.sub(r"\s+", " ", text).strip()


def load_corpus(n: int = 5000, timeout: float = 30.0) -> Corpus:
    """First ``n`` AG News headlines (cached to disk), synthetic fallback."""
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / "ag_news_test.csv"
    try:
        if not local.exists():
            with urllib.request.urlopen(AG_NEWS_URL, timeout=timeout) as r:  # noqa: S310 - fixed url
                local.write_bytes(r.read())
        texts, labels = [], []
        reader = csv.reader(io.StringIO(local.read_text(encoding="utf-8")))
        for row in reader:
            if len(row) < 2:
                continue
            title = clean_text(row[1])
            if title:
                texts.append(title)
                labels.append(AG_LABELS.get(int(row[0]), "?"))
            if len(texts) >= n:
                break
        return Corpus(texts, labels, f"AG News test set ({len(texts)} headlines)")
    except Exception:  # noqa: BLE001 - offline etc.
        return _synthetic_headlines(n)


# ===================================================================== embedder

class HashingEmbedder:
    """Offline fallback: hashed bag of words + character trigrams, L2-normalized.

    Much weaker than a real sentence model (no understanding of synonyms),
    but lets the app and tests run without internet.
    """

    name = "hashing bag-of-words (offline fallback)"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        grams = [w[i:i + 3] for w in words for i in range(max(1, len(w) - 2))]
        return words + grams

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in self._features(t):
                h = zlib.crc32(f.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


class SentenceEmbedder:
    """Wrapper around sentence-transformers all-MiniLM-L6-v2 (384-d)."""

    name = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: list[str], batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress_bar).astype(np.float32)


def load_embedder(allow_model: bool = True) -> HashingEmbedder | SentenceEmbedder:
    if allow_model:
        try:
            return SentenceEmbedder()
        except Exception:  # noqa: BLE001 - no internet / no torch
            pass
    return HashingEmbedder()


# ===================================================================== engine

@dataclass
class SearchHit:
    rank: int
    text: str
    label: str
    similarity: float


@dataclass
class SearchEngine:
    corpus: Corpus
    embedder: HashingEmbedder | SentenceEmbedder
    embeddings: np.ndarray
    indexes: dict[str, BaseIndex] = field(default_factory=dict)
    build_times: dict[str, float] = field(default_factory=dict)
    last_checked: int = 0
    _proj: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    def vectors_checked(self, q: np.ndarray, index_name: str, k: int = 10, ef: int | None = None) -> int:
        """How many stored vectors the index compared with the query (its real work)."""
        index = self.indexes[index_name]
        if isinstance(index, HNSWIndex):
            index.distance_count = 0
            index.search(q, k, ef=ef)
            return index.distance_count
        if isinstance(index, LSHIndex):
            return int(len(index.candidates(q)))
        return len(index)

    def projection(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2-D PCA of all embeddings, for the headline map.

        Returns (points (n, 2), mean (d,), components (d, 2)) so queries can be
        projected the same way. Computed once with an SVD.
        """
        if self._proj is None:
            mean = self.embeddings.mean(axis=0)
            _, _, vt = np.linalg.svd(self.embeddings - mean, full_matrices=False)
            comps = vt[:2].T
            self._proj = ((self.embeddings - mean) @ comps, mean, comps)
        return self._proj

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([query])[0]

    def search(self, query: str, index_name: str = "HNSW", k: int = 10,
               ef: int | None = None) -> tuple[list[SearchHit], float]:
        """Returns hits and search time in ms (embedding time not included)."""
        q = self.embed_query(query)
        return self.search_vector(q, index_name, k, ef)

    def search_vector(self, q: np.ndarray, index_name: str = "HNSW", k: int = 10,
                      ef: int | None = None, repeats: int = 5) -> tuple[list[SearchHit], float]:
        """Search with a precomputed query vector.

        The search is run ``repeats`` times and the median time is reported, because a
        single sub-millisecond measurement is very noisy (cold caches, other processes).
        """
        index = self.indexes[index_name]
        times = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            if isinstance(index, HNSWIndex):
                ids, dists = index.search(q, k, ef=ef)
            else:
                ids, dists = index.search(q, k)
            times.append(time.perf_counter() - t0)
        ms = float(np.median(times)) * 1000
        self.last_checked = self.vectors_checked(q, index_name, k, ef)
        hits = [SearchHit(r + 1, self.corpus.texts[i], self.corpus.labels[i], 1.0 - float(d))
                for r, (i, d) in enumerate(zip(ids.tolist(), dists.tolist()))]
        return hits, ms


def _cache_key(corpus: Corpus, embedder_name: str) -> str:
    h = zlib.crc32(("\n".join(corpus.texts) + embedder_name).encode())
    return f"{len(corpus.texts)}_{h:08x}"


def build_engine(corpus: Corpus | None = None, embedder=None, use_cache: bool = True,
                 hnsw_params: dict | None = None, progress=None) -> SearchEngine:
    """Embed the corpus and build all three indexes (cosine metric).

    ``progress`` is an optional callback ``progress(fraction, message)``.
    """
    def report(f: float, msg: str) -> None:
        if progress:
            progress(f, msg)

    corpus = corpus or load_corpus()
    embedder = embedder or load_embedder()
    key = _cache_key(corpus, embedder.name)
    folder = CACHE / key
    emb_file = folder / "embeddings.npy"

    report(0.05, f"embedding {len(corpus.texts)} sentences with {embedder.name} ...")
    if use_cache and emb_file.exists():
        emb = np.load(emb_file)
    else:
        emb = embedder.encode(corpus.texts, batch_size=128)
        if use_cache:
            folder.mkdir(parents=True, exist_ok=True)
            np.save(emb_file, emb)

    dim = emb.shape[1]
    params = {"M": 16, "ef_construction": 100, "ef_search": 50, **(hnsw_params or {})}
    # LSH settings picked by measuring recall on these embeddings: semantic neighbours are
    # ~60 degrees apart, so each bit only agrees ~67% of the time and LSH needs many tables.
    # L=32, K=10 gave recall@10 ~0.91 while scanning ~38% of the corpus.
    lsh_params = {"n_tables": 32, "n_bits": 10, "multi_probe": True, "center": False}
    makers = {
        "BruteForce": lambda: BruteForceIndex(dim, metric="cosine"),
        "LSH": lambda: LSHIndex(dim, metric="cosine", **lsh_params),
        "HNSW": lambda: HNSWIndex(dim, metric="cosine", **params),
    }
    cache_names = {
        "BruteForce": "BruteForce",
        "LSH": f"LSH_{lsh_params['n_tables']}_{lsh_params['n_bits']}_{int(lsh_params['center'])}",
        "HNSW": f"HNSW_{params['M']}_{params['ef_construction']}",
    }
    engine = SearchEngine(corpus, embedder, emb)
    for i, name in enumerate(INDEX_NAMES):
        report(0.2 + 0.25 * i, f"building {name} index ...")
        path = folder / cache_names[name]
        t0 = time.perf_counter()
        if use_cache and (path / "header.json").exists():
            engine.indexes[name] = BaseIndex.load(path)
            if isinstance(engine.indexes[name], HNSWIndex):
                engine.indexes[name].ef_search = params["ef_search"]
        else:
            idx = makers[name]()
            idx.add(emb)
            engine.indexes[name] = idx
            if use_cache:
                idx.save(path)
        engine.build_times[name] = time.perf_counter() - t0
    report(1.0, "ready")
    return engine
