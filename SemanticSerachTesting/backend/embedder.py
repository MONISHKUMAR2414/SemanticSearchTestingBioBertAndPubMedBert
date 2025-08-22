# backend/embedder.py
from __future__ import annotations
from typing import Iterable, List, Optional
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

class TextEmbedder:
    """
    Transformer-based embedder using mean-pooling. No sentence-transformers dependency.
    """
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", device: Optional[str] = None, max_length: int = 512):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)

    @torch.inference_mode()
    def encode(self, texts: Iterable[str], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        texts_list: List[str] = [t if isinstance(t, str) else str(t) for t in texts]
        if len(texts_list) == 0:
            return np.zeros((0, self._model.config.hidden_size), dtype=np.float32)

        all_embeddings = []
        for start in range(0, len(texts_list), batch_size):
            batch = texts_list[start : start + batch_size]
            inputs = self._tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            token_embeddings = outputs.last_hidden_state
            sentence_embeddings = _mean_pool(token_embeddings, inputs["attention_mask"])
            if normalize:
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.append(sentence_embeddings.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.vstack(all_embeddings)
