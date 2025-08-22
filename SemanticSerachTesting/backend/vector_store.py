# backend/vector_store.py
import numpy as np
import faiss

def build_faiss_index(embeddings: np.ndarray):
    """
    Build IndexFlatIP using inner-product on normalized vectors.
    Input: embeddings (N, D) float32 and already L2-normalized.
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def search_index(index, query_embeddings: np.ndarray, top_k: int = 10):
    if query_embeddings.dtype != np.float32:
        query_embeddings = query_embeddings.astype(np.float32)
    scores, indices = index.search(query_embeddings, top_k)
    return scores, indices
