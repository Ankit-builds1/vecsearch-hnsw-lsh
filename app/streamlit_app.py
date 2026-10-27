"""VecSearch semantic search demo.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from search_engine import INDEX_NAMES, build_engine  # noqa: E402

st.set_page_config(page_title="VecSearch - Semantic Search", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def get_engine():
    bar = st.progress(0.0, text="starting ...")
    engine = build_engine(progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
    bar.empty()
    return engine


st.title("VecSearch")
st.caption("Semantic search over news headlines using vector indexes written from scratch "
           "(Brute Force, LSH, HNSW) in Python + NumPy.")

with st.spinner("Loading model, embedding corpus and building indexes (first run takes a minute) ..."):
    engine = get_engine()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    index_name = st.radio("Index", INDEX_NAMES, index=0,
                          help="HNSW = graph search, LSH = hashing, BruteForce = exact scan")
    k = st.slider("Number of results (k)", 1, 30, 10)
    ef = st.slider("HNSW ef_search", 10, 400, 50, step=10,
                   help="Beam width. Higher = more accurate but slower.",
                   disabled=index_name != "HNSW")
    compare = st.checkbox("Compare all three indexes", value=True)

    st.divider()
    st.subheader("About the data")
    st.write(f"**Corpus:** {engine.corpus.source}")
    st.write(f"**Embedding model:** {engine.embedder.name} ({engine.embeddings.shape[1]}-d)")
    st.write(f"**Vectors indexed:** {len(engine.embeddings):,}")
    hnsw = engine.indexes["HNSW"]
    st.write(f"**HNSW:** M={hnsw.M}, ef_construction={hnsw.ef_construction}, layers={hnsw.max_level + 1}")
    st.caption("Index load/build time: " +
               ", ".join(f"{n} {t:.2f}s" for n, t in engine.build_times.items()))

# ------------------------------------------------------------------ main
examples = ["stock market crash fears", "football team wins the cup", "new vaccine for disease",
            "space mission to mars", "peace talks between countries"]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["query"] = ex

query = st.text_input("Search query", key="query", placeholder="e.g. oil prices are going up")

if query.strip():
    hits, ms = engine.search(query, index_name, k, ef=ef)
    st.subheader(f"Top {len(hits)} results from {index_name}  -  {ms:.2f} ms")
    if not hits:
        st.info("No results (LSH found no candidates in the probed buckets - try more tables).")
    else:
        df = pd.DataFrame([{"Rank": h.rank, "Headline": h.text, "Category": h.label,
                            "Similarity": round(h.similarity, 4)} for h in hits])
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={"Similarity": st.column_config.ProgressColumn(
                         "Similarity", min_value=0.0, max_value=1.0, format="%.3f")})

    if compare:
        st.subheader("Index comparison for this query")
        q = engine.embed_query(query)
        exact, _ = engine.search_vector(q, "BruteForce", k)
        exact_set = {h.text for h in exact}
        rows = []
        for name in INDEX_NAMES:
            r, t = engine.search_vector(q, name, k, ef=ef)
            rows.append({"Index": name, "Search time (ms)": round(t, 3),
                         f"Recall@{k} vs exact": round(len({h.text for h in r} & exact_set) / max(1, len(exact_set)), 3),
                         "Top result": r[0].text if r else "-"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Search time is the median of 5 runs and excludes the time to embed the query. "
                   "Recall compares each index to the exact brute force answer.")
else:
    st.info("Type a query above or click one of the examples.")
