# backend/search_logic.py
import re
from typing import List, Dict, Optional
from pubmed_fetcher import fetch_pubmed_articles, PubMedArticle
from embedder import TextEmbedder
from vector_store import build_faiss_index, search_index
import numpy as np
import os
from gemini_integration import call_gemini_summarize

# small synonym & MeSH maps (extend as needed)
GENERAL_SYNS = {
    "heart": ["cardiac", "cardio"],
    "attack": ["infarction"],
    "stroke": ["cva", "cerebrovascular accident"]
}
MEDICAL_SYNS = {
    "myocardial": ["cardiac"],
    "infarction": ["heart attack"],
    "diabetes": ["type 2 diabetes", "t2d"]
}
MESH_MAP = {
    "heart attack": ["Myocardial Infarction"],
    "diabetes": ["Diabetes Mellitus"],
    "stroke": ["Stroke"]
}

def tokenize_terms(query: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\-\']+", query)

def expand_term_with_synonyms(term: str) -> List[str]:
    term_l = term.lower()
    syns = []
    syns.extend(GENERAL_SYNS.get(term_l, []))
    syns.extend(MEDICAL_SYNS.get(term_l, []))
    if term_l in MESH_MAP:
        syns.extend(MESH_MAP[term_l])
    return [term] + syns

def mesh_terms_for_query(tokens: List[str]) -> List[str]:
    mesh_terms = []
    joined = " ".join(tokens).lower()
    for k, v in MESH_MAP.items():
        if k in joined:
            mesh_terms.extend(v)
    return mesh_terms

def build_boolean_query(query: str, use_mesh: bool = True) -> str:
    tokens = tokenize_terms(query)
    if not tokens:
        return query
    groups = []
    for t in tokens:
        expanded = expand_term_with_synonyms(t)
        expanded_quoted = []
        for e in expanded:
            if " " in e:
                expanded_quoted.append(f"\"{e}\"")
            else:
                expanded_quoted.append(e)
        group = "(" + " OR ".join(expanded_quoted) + ")"
        groups.append(group)
    if use_mesh:
        mesh = mesh_terms_for_query(tokens)
        if mesh:
            mesh_group = "(" + " OR ".join([f"\"{m}\"[MeSH Terms]" for m in mesh]) + ")"
            groups.append(mesh_group)
    return " AND ".join(groups) if groups else query

def run_search_pipeline(
    query: str,
    retmax: int = 200,
    top_k: int = 10,
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    ncbi_email: Optional[str] = None,
    ncbi_api_key: Optional[str] = None,
    use_mesh: bool = True,
    use_gemini_summary: bool = False,
    gemini_api_key: Optional[str] = None
) -> Dict:
    boolean_query = build_boolean_query(query, use_mesh=use_mesh)
    articles = fetch_pubmed_articles(boolean_query, retmax=retmax, email=ncbi_email, api_key=ncbi_api_key)
    if not articles:
        return {"original_query": query, "boolean_query": boolean_query, "total_fetched": 0, "results": []}

    texts = [a.abstract for a in articles if isinstance(a.abstract, str) and a.abstract.strip()]
    keep_indices = [i for i, a in enumerate(articles) if isinstance(a.abstract, str) and a.abstract.strip()]
    if not texts:
        return {"original_query": query, "boolean_query": boolean_query, "total_fetched": len(articles), "total_with_abstracts": 0, "results": []}

    embedder = TextEmbedder(model_name=model_name)
    doc_embeddings = embedder.encode(texts, batch_size=8, normalize=True)
    index = build_faiss_index(doc_embeddings)
    q_emb = embedder.encode([query], batch_size=1, normalize=True)
    scores, inds = search_index(index, q_emb, top_k=top_k)

    results = []
    for score, local_idx in zip(scores[0], inds[0]):
        if local_idx < 0 or local_idx >= len(keep_indices):
            continue
        global_idx = keep_indices[local_idx]
        art = articles[global_idx]
        results.append({
            "pmid": art.pmid,
            "title": art.title,
            "abstract": art.abstract,
            "url": art.url,
            "journal": art.journal,
            "year": art.year,
            "authors": art.authors,
            "score": float(score)
        })

    # Top-5 summaries
    summaries = []
    if use_gemini_summary:
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    else:
        api_key = None

    for r in results[:5]:
        try:
            s = call_gemini_summarize(api_key, r["title"], r["abstract"])
        except Exception as e:
            print("Summarization failed:", e)
            s = (r["abstract"] or "")[:400] + "..."
        r["summary"] = s

    return {
        "original_query": query,
        "boolean_query": boolean_query,
        "total_fetched": len(articles),
        "total_with_abstracts": len(texts),
        "results": results
    }
