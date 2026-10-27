"""VecSearch semantic search demo.

Run from the project root (so the theme in .streamlit/config.toml is used):
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

from search_engine import INDEX_NAMES, build_engine  # noqa: E402

st.set_page_config(page_title="VecSearch", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- design tokens
INK = "#E8ECF7"          # light text on the dark indigo background
MUTED = "#A9B4D0"
CATEGORY_COLORS = {      # identity colours (dark-mode steps), always shown next to the category name
    "World": "#3987E5",
    "Sports": "#1BC48A",
    "Business": "#F0B020",
    "Sci/Tech": "#EE7FAA",
}
INDEX_COLORS = {"HNSW": "#3987E5", "LSH": "#1BC48A", "BruteForce": "#F07A45"}


def tint(hex_color: str, alpha: float) -> str:
    """'#RRGGBB' -> 'rgba(r, g, b, alpha)' for soft backgrounds."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


INDEX_BLURB = {
    "HNSW": "Walks a layered graph of nearby headlines",
    "LSH": "Looks only in matching hash buckets",
    "BruteForce": "Compares the query with every headline",
}

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Public+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root { --ink: #E8ECF7; --muted: #A9B4D0; --accent: #5B9BEA; --line: #2E3B78; --surface: #18214A; }
/* deep indigo night background with two soft colour glows */
.stApp { background:
  radial-gradient(900px 500px at 85% -10%, rgba(238,127,170,.16), transparent 60%),
  radial-gradient(900px 600px at 10% 0%, rgba(57,135,229,.22), transparent 60%),
  #0F1533; }
header[data-testid="stHeader"] { background: transparent; }

.stApp :is(p, li, label, input, textarea, button, td, th),
.stApp div[data-testid="stMarkdownContainer"] { font-family: "Public Sans", system-ui, sans-serif; }

.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* masthead: indigo banner with the four category colours as a strip */
.vs-masthead { background: linear-gradient(135deg, #24327A 0%, #1B2559 60%, #2A2466 100%);
  border: 1px solid #34449A; border-radius: 18px; padding: 2rem 2.2rem 1.6rem; margin-bottom: 1.2rem; position: relative; overflow: hidden; }
.vs-masthead h1 {
  font-family: "Newsreader", Georgia, serif; font-weight: 600; font-size: 3.6rem;
  line-height: 1.02; letter-spacing: -0.02em; color: #FFFFFF; margin: 0 0 .5rem 0; padding: 0;
}
.vs-masthead p { color: #C9D3EC; font-size: 1.06rem; max-width: 62ch; margin: 0 0 1.2rem 0; }
.vs-strip { display: flex; gap: .45rem; flex-wrap: wrap; }
.vs-strip span { display: inline-flex; align-items: center; gap: .45rem; padding: .3rem .8rem; border-radius: 999px;
  background: rgba(255,255,255,.08); color: #FFFFFF; font-size: .88rem; }
.vs-strip span i { width: .7rem; height: .7rem; border-radius: 50%; background: var(--cat); display: inline-block; }
.vs-masthead .vs-bands { position: absolute; top: 0; right: 0; bottom: 0; width: 14px; display: flex; flex-direction: column; }
.vs-masthead .vs-bands b { flex: 1; }

/* the search box is the hero */
div[data-testid="stTextInput"] input {
  font-size: 1.3rem; padding: .95rem 1.1rem; border-radius: 12px;
  border: 1.5px solid var(--line); background: var(--surface); color: var(--ink);
}
div[data-testid="stTextInput"] input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(91,155,234,.30); }
div[data-testid="stTextInput"] input::placeholder { color: #7E8AAE; }

/* example queries as quiet chips */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
  border-radius: 999px; border: 1px solid var(--line); background: var(--surface);
  color: var(--ink); font-size: .9rem; padding: .25rem .9rem; min-height: 2.2rem;
}
div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover { border-color: var(--accent); color: var(--accent); }

.vs-status { color: var(--muted); font-size: .98rem; margin: .6rem 0 .3rem 0; display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
.vs-status b { color: var(--ink); font-weight: 600; }
.vs-idx { color: #FFFFFF; background: var(--idx); border-radius: 999px; padding: .15rem .75rem; font-weight: 600; font-size: .9rem; }

/* ranked results: each row carries its category colour */
.vs-result { display: grid; grid-template-columns: 3.2rem 1fr; gap: .9rem; padding: .9rem 1rem .9rem .6rem; margin: 0 0 .55rem 0;
  border-left: 5px solid var(--cat); border-radius: 10px; background: var(--tint); }
.vs-rank { font-family: "Newsreader", Georgia, serif; font-size: 2rem; line-height: 1; color: var(--cat); text-align: right; }
.vs-headline { font-family: "Newsreader", Georgia, serif; font-size: 1.32rem; line-height: 1.3; color: var(--ink); }
.vs-meta { display: flex; align-items: center; gap: .8rem; margin-top: .4rem; font-size: .86rem; color: var(--muted); }
.vs-cat { display: inline-flex; align-items: center; gap: .4rem; }
.vs-cat i { width: .62rem; height: .62rem; border-radius: 50%; background: var(--cat); display: inline-block; }
.vs-bar { flex: 0 0 120px; height: 6px; border-radius: 3px; background: rgba(255,255,255,.12); overflow: hidden; }
.vs-bar span { display: block; height: 100%; background: var(--cat); border-radius: 3px; }

/* tabs */
div[data-baseweb="tab-list"] button p { font-size: 1rem; font-weight: 500; }
section[data-testid="stSidebar"] { background: #141C42; border-right: 1px solid #263270; }

.vs-empty { padding: 2.2rem 0 1rem; color: var(--muted); font-size: 1.05rem; max-width: 60ch; }
.vs-note { color: var(--muted); font-size: .9rem; max-width: 75ch; }

section[data-testid="stSidebar"] .vs-side-h { font-family: "Newsreader", Georgia, serif; font-size: 1.35rem; color: var(--ink); margin: .2rem 0 .3rem; }
</style>
"""
# a blank line would end the HTML block in Streamlit's markdown and print the CSS as text
st.markdown("\n".join(line for line in CSS.splitlines() if line.strip()), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_engine():
    bar = st.progress(0.0, text="Starting ...")
    engine = build_engine(progress=lambda f, m: bar.progress(min(f, 1.0), text=m.capitalize()))
    bar.empty()
    return engine


@st.cache_data(show_spinner=False)
def load_benchmark_table() -> pd.DataFrame | None:
    path = ROOT / "benchmarks" / "results" / "comparison.csv"
    return pd.read_csv(path) if path.exists() else None


with st.spinner("Loading the model and the indexes. The first start takes about a minute."):
    engine = get_engine()
N = len(engine.embeddings)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="vs-side-h">Search settings</div>', unsafe_allow_html=True)
    index_name = st.radio("Index", INDEX_NAMES, index=0,
                          captions=[INDEX_BLURB[n] for n in INDEX_NAMES])
    k = st.slider("Results to show", 1, 30, 10)
    ef = st.slider("HNSW beam width (ef_search)", 10, 400, 50, step=10,
                   help="How many candidates HNSW keeps while searching. "
                        "Wider is more accurate and slower.",
                   disabled=index_name != "HNSW")

    st.divider()
    st.markdown('<div class="vs-side-h">About this demo</div>', unsafe_allow_html=True)
    hnsw = engine.indexes["HNSW"]
    st.markdown(
        f"""
- **{N:,} news headlines** from {engine.corpus.source.split('(')[0].strip()}
- Each headline is a **{engine.embeddings.shape[1]}-number vector** made by {engine.embedder.name}
- HNSW graph: **{hnsw.max_level + 1} layers**, M = {hnsw.M}
- All three indexes are written from scratch in Python and NumPy
"""
    )

# ---------------------------------------------------------------- masthead + search
strip = "".join(f'<span style="--cat:{c}"><i></i>{name}</span>' for name, c in CATEGORY_COLORS.items())
bands = "".join(f'<b style="background:{c}"></b>' for c in CATEGORY_COLORS.values())
st.markdown(
    f'<div class="vs-masthead"><div class="vs-bands">{bands}</div><h1>VecSearch</h1>'
    f'<p>Search {N:,} news headlines by meaning. Your words don\'t have to match the headline, '
    f'because every headline is stored as a vector and the index finds the closest ones.</p>'
    f'<div class="vs-strip">{strip}</div></div>',
    unsafe_allow_html=True,
)

query = st.text_input("Search headlines", key="query", label_visibility="collapsed",
                      placeholder="Try: oil prices are going up")

examples = ["stock market crash", "team wins the cup", "new vaccine",
            "mission to mars", "peace talks"]
for col, ex in zip(st.columns(len(examples)), examples):
    col.button(ex, use_container_width=True, on_click=lambda e=ex: st.session_state.update(query=e))

query = (query or "").strip()
q_vec = engine.embed_query(query) if query else None
hits, ms = (engine.search_vector(q_vec, index_name, k, ef=ef) if query else ([], 0.0))
checked = engine.last_checked if query else 0

if query:
    st.markdown(
        f'<p class="vs-status"><span class="vs-idx" style="--idx:{INDEX_COLORS[index_name]}">{index_name}</span>'
        f'<span>found {len(hits)} headlines in <b>{ms:.2f} ms</b>, '
        f'comparing the query with <b>{checked:,}</b> of {N:,} stored vectors '
        f'({100 * checked / N:.0f}%).</span></p>',
        unsafe_allow_html=True,
    )

tab_results, tab_map, tab_compare, tab_bench = st.tabs(
    ["Results", "Headline map", "Compare indexes", "Benchmarks"])

# ---------------------------------------------------------------- results
with tab_results:
    if not query:
        st.markdown('<div class="vs-empty">Type a topic or a question above, or pick one of the '
                    'examples. Results show up here, ranked by how close their meaning is to '
                    'your query.</div>', unsafe_allow_html=True)
    elif not hits:
        st.markdown('<div class="vs-empty">LSH found no headlines in the buckets it checked. '
                    'Switch to HNSW or BruteForce in the sidebar to search this query.</div>',
                    unsafe_allow_html=True)
    else:
        rows = []
        for h in hits:
            color = CATEGORY_COLORS.get(h.label, MUTED)
            width = max(0.0, min(1.0, h.similarity)) * 100
            rows.append(
                f'<div class="vs-result" style="--cat:{color};--tint:{tint(color, 0.16)}">'
                f'<div class="vs-rank">{h.rank}</div><div>'
                f'<div class="vs-headline">{html.escape(h.text)}</div>'
                f'<div class="vs-meta"><span class="vs-cat" style="--cat:{color}"><i></i>{h.label}</span>'
                f'<span class="vs-bar" title="cosine similarity"><span style="width:{width:.0f}%"></span></span>'
                f'<span>similarity {h.similarity:.3f}</span></div></div></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

# ---------------------------------------------------------------- headline map
with tab_map:
    pts, mean, comps = engine.projection()
    df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "headline": engine.corpus.texts,
                       "category": engine.corpus.labels})
    color = alt.Color("category:N", title="Category",
                      scale=alt.Scale(domain=list(CATEGORY_COLORS), range=list(CATEGORY_COLORS.values())),
                      legend=alt.Legend(orient="top", direction="horizontal", titleOrient="left",
                                        symbolOpacity=1, symbolSize=120, labelFontSize=13))
    base = (alt.Chart(df).mark_circle(size=14, opacity=0.38)
            .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), color=color,
                    tooltip=["headline:N", "category:N"]))
    layers = [base]
    if query and hits:
        qx, qy = ((q_vec - mean) @ comps).tolist()
        text_to_pos = {t: i for i, t in enumerate(engine.corpus.texts)}
        res = pd.DataFrame([{"x": pts[text_to_pos[h.text], 0], "y": pts[text_to_pos[h.text], 1],
                             "headline": h.text, "category": h.label, "rank": h.rank,
                             "similarity": round(h.similarity, 3), "qx": qx, "qy": qy} for h in hits])
        layers += [
            alt.Chart(res).mark_rule(color=INK, opacity=0.35, strokeWidth=1.2)
              .encode(x="qx:Q", y="qy:Q", x2="x:Q", y2="y:Q"),
            alt.Chart(res).mark_circle(size=120, opacity=1, stroke=INK, strokeWidth=1.5)
              .encode(x="x:Q", y="y:Q", color=color,
                      tooltip=["rank:Q", "headline:N", "category:N", "similarity:Q"]),
            alt.Chart(pd.DataFrame({"x": [qx], "y": [qy], "label": ["your query"]}))
              .mark_point(shape="diamond", size=320, filled=True, color=INK)
              .encode(x="x:Q", y="y:Q", tooltip=["label:N"]),
            alt.Chart(pd.DataFrame({"x": [qx], "y": [qy], "label": [f'"{query}"']}))
              .mark_text(align="left", dx=14, dy=-12, fontSize=14, font="Newsreader", color=INK)
              .encode(x="x:Q", y="y:Q", text="label:N"),
        ]
    chart = alt.layer(*layers).properties(height=560).configure_view(strokeWidth=0).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.markdown(
        '<p class="vs-note">Every dot is one headline. Its 384 numbers are squashed down to 2 with PCA so '
        'they fit on screen, which is why sports, business, world and science news form loose regions. '
        'Squashing loses information, so some true neighbours look further apart here than they really '
        'are. Scroll to zoom, drag to pan, hover to read a headline.</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- compare indexes
with tab_compare:
    if not query:
        st.markdown('<div class="vs-empty">Search for something first. This tab then runs the same '
                    'query through all three indexes and compares speed, work and accuracy.</div>',
                    unsafe_allow_html=True)
    else:
        exact, _ = engine.search_vector(q_vec, "BruteForce", k)
        exact_set = {h.text for h in exact}
        rows = []
        for name in INDEX_NAMES:
            r, t = engine.search_vector(q_vec, name, k, ef=ef)
            rows.append({"Index": name, "Time (ms)": round(t, 2),
                         "Headlines checked": engine.last_checked,
                         "Share of corpus": 100 * engine.last_checked / N,
                         "Recall": len({h.text for h in r} & exact_set) / max(1, len(exact_set))})
        cmp_df = pd.DataFrame(rows)

        def bars(col: str, title: str, fmt: str) -> alt.LayerChart:
            b = alt.Chart(cmp_df).encode(
                y=alt.Y("Index:N", sort=list(INDEX_NAMES), title=None),
                x=alt.X(f"{col}:Q", title=None, axis=alt.Axis(grid=True, gridColor="#2E3B78")))
            idx_color = alt.Color("Index:N", legend=None, scale=alt.Scale(
                domain=list(INDEX_COLORS), range=list(INDEX_COLORS.values())))
            return (b.mark_bar(size=18, cornerRadiusEnd=4).encode(color=idx_color)
                    + b.mark_text(align="left", dx=5, color=INK).encode(text=alt.Text(f"{col}:Q", format=fmt))
                    ).properties(title=title, height=170)

        c1, c2 = st.columns(2)
        c1.altair_chart(bars("Time (ms)", "Time per search (ms, median of 5)", ".2f"), use_container_width=True)
        c2.altair_chart(bars("Headlines checked", "Headlines compared with the query", ",d"),
                        use_container_width=True)

        st.dataframe(
            cmp_df, hide_index=True, use_container_width=True,
            column_config={
                "Share of corpus": st.column_config.ProgressColumn("Share of corpus checked", format="%.0f%%",
                                                                   min_value=0.0, max_value=100.0),
                "Recall": st.column_config.NumberColumn(f"Recall@{k} vs exact", format="%.2f"),
                "Headlines checked": st.column_config.NumberColumn(format="%d"),
            },
        )
        h_row = cmp_df[cmp_df.Index == "HNSW"].iloc[0]
        st.markdown(
            f'<p class="vs-note">HNSW returned {h_row["Recall"] * 100:.0f}% of the exact top {k} while '
            f'checking {h_row["Share of corpus"]:.0f}% of the headlines. Brute force always checks all '
            f'{N:,}, so it is exact by definition and is used as the reference here.</p>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------- benchmarks
with tab_bench:
    table = load_benchmark_table()
    if table is None:
        st.markdown('<div class="vs-empty">No benchmark results yet. Run '
                    '<code>python benchmarks/run_benchmarks.py</code> to create them.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<p class="vs-note">Measured on SIFT-10K (10,000 image descriptors with 128 numbers each, '
                    '100 queries) on one CPU thread. FAISS is Facebook\'s C++ library, included only '
                    'for comparison.</p>', unsafe_allow_html=True)
        show = table[["method", "library", "recall@10", "latency_ms", "dist_per_query", "build_s", "memory_mb"]].copy()
        # FAISS LSH does not report a distance count
        show["dist_per_query"] = show["dist_per_query"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "n/a")
        st.dataframe(
            show, hide_index=True, use_container_width=True,
            column_config={
                "method": "Index", "library": "Library",
                "recall@10": st.column_config.ProgressColumn("Recall@10", format="%.3f",
                                                             min_value=0.0, max_value=1.0),
                "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.2f"),
                "dist_per_query": st.column_config.TextColumn("Distances per query"),
                "build_s": st.column_config.NumberColumn("Build (s)", format="%.2f"),
                "memory_mb": st.column_config.NumberColumn("Memory (MB)", format="%.1f"),
            },
        )
        c1, c2 = st.columns(2)
        img = ROOT / "benchmarks" / "results"
        if (img / "recall_vs_qps.png").exists():
            c1.image(str(img / "recall_vs_qps.png"), caption="Recall vs speed for every setting tested")
        if (img / "scaling.png").exists():
            c2.image(str(img / "scaling.png"),
                     caption="Brute force grows with n, HNSW stays nearly flat")
