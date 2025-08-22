import os
import hashlib
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv

# 🔹 Load environment variables from backend/.env explicitly
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "backend", ".env")
load_dotenv(dotenv_path=dotenv_path)

st.set_page_config(page_title="PubMed Semantic Search (UI)", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("PubMed Semantic Search")
st.caption("Frontend that calls FastAPI backend. Top-5 summarization visualized beneath results.")

# ✅ Debugging: show if keys are loaded (remove in production)
with st.sidebar:
    st.subheader("🔑 Environment Debug")
    st.write("NCBI_EMAIL:", os.getenv("NCBI_EMAIL"))
    st.write("GEMINI_API_KEY:", "✅ Loaded" if os.getenv("GEMINI_API_KEY") else "❌ Not Found")
    st.write("BACKEND_URL:", BACKEND_URL)

    # --- Your existing sidebar settings ---
    st.subheader("Settings")
    ncbi_email = st.text_input("Entrez Email (recommended)")
    ncbi_api_key = st.text_input("Entrez API Key (optional)", type="password")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)
    model_label = st.selectbox("Embedding model", [
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "dmis-lab/biobert-base-cased-v1.1"
    ])
    retmax = st.slider("Number of PubMed articles to fetch", 10, 5000, 50, step=10)
    top_k = st.slider("Top-N results to show", 5, 100, 10)
    use_mesh = st.checkbox("Use MeSH mapping", value=True)
    use_gemini_summary = st.checkbox("Use Gemini summarizer (if available)", value=False)
    gemini_key = st.text_input(
        "Gemini API Key (optional)", 
        type="password", 
        placeholder="Set GEMINI_API_KEY env var or paste here"
    )
    clear_cache = st.button("Clear cache")

if clear_cache:
    st.experimental_memo_clear()
    st.success("Cache cleared.")

query = st.text_input("Enter your medical query", placeholder="e.g., heart attack symptoms")
do_search = st.button("Search")

if do_search and query.strip():
    payload = {
        "query": query,
        "retmax": retmax,
        "top_k": top_k,
        "model_name": model_label,
        "use_mesh": use_mesh,
        "use_gemini_summary": use_gemini_summary,
        "gemini_api_key": gemini_key or None
    }
    with st.spinner("Calling backend search..."):
        try:
            resp = requests.post(f"{backend_url.rstrip('/')}/search", json=payload, timeout=500)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    results = data.get("results", [])
    total = data.get("total_fetched", 0)
    st.write(f"Fetched {total} articles from PubMed — displaying top {len(results)} results.")

    # Show results cards
    for i, r in enumerate(results, start=1):
        st.markdown(f"### {i}. [{r.get('title')}]({r.get('url')})")
        meta = []
        if r.get("journal"):
            meta.append(r.get("journal"))
        if r.get("year"):
            meta.append(str(r.get("year")))
        if r.get("authors"):
            meta.append("Authors: " + ", ".join(r.get("authors")[:3]) + (", et al." if len(r.get("authors")) > 3 else ""))
        st.write(" • ".join(meta))
        st.write(r.get("abstract")[:800] + ("…" if len(r.get("abstract")) > 800 else ""))
        st.write(f"Similarity: **{r.get('score'):.4f}**")
        st.markdown("---")

    # Visualize top-5 summaries (if present)
    if results:
        top5 = results[:5]
        st.header("Top-5 Summaries")
        cols = st.columns(2)
        for idx, r in enumerate(top5, start=1):
            col = cols[(idx - 1) % 2]
            with col:
                st.markdown(f"#### {idx}. [{r.get('title')}]({r.get('url')})")
                summary = r.get("summary")
                if summary:
                    st.write(summary)
                else:
                    st.write("No summary available.")
                with st.expander("Show full abstract"):
                    st.write(r.get("abstract"))

    # CSV download
    if results:
        df = pd.DataFrame(results)
        st.download_button(
            "Download results (CSV)", 
            data=df.to_csv(index=False).encode("utf-8"), 
            file_name="pubmed_results.csv", 
            mime="text/csv"
        )
else:
    st.info("Enter a query and press Search.")

