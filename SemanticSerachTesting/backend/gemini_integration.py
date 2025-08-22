# backend/gemini_integration.py
"""
Adapter for Gemini (Google Generative) summarization with local fallback.
Set GEMINI_API_KEY and GEMINI_ENDPOINT if you have access to Google Generative API.
If not, the function falls back to a local HF summarizer (smaller model).
"""

import os
import requests
from typing import Optional

_local_summarizer = None

def _init_local_summarizer():
    global _local_summarizer
    if _local_summarizer is None:
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            _local_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
        except Exception as e:
            print("Local summarizer initialization failed:", e)
            _local_summarizer = None

def call_gemini_summarize(api_key: Optional[str], title: str, abstract: str, model: Optional[str] = None) -> str:
    """
    Summarize using Gemini if api_key provided; otherwise use local fallback.
    Returns a concise summary string.
    """
    model = model or os.getenv("GEMINI_MODEL", "gemini-1.5")
    endpoint = os.getenv("GEMINI_ENDPOINT")
    text = (title or "") + "\n\n" + (abstract or "")
    prompt = (
        "Summarize the following PubMed article for a researcher in 3 short bullet points (<=25 words each), "
        "then give a 1-line key takeaway. Keep language concise and technical when appropriate.\n\n"
        f"Article:\n{text}"
    )

    if api_key and endpoint:
        try:
            url = f"{endpoint}/{model}:generate"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "prompt": {"text": prompt},
                "temperature": 0.0,
                "maxOutputTokens": 350
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            j = resp.json()
            # Try to extract text
            if isinstance(j, dict):
                if "candidates" in j and isinstance(j["candidates"], list) and j["candidates"]:
                    return j["candidates"][0].get("content", "").strip()
                # try search for first string
                def find_text(obj):
                    if isinstance(obj, str):
                        return obj
                    if isinstance(obj, dict):
                        for v in obj.values():
                            r = find_text(v)
                            if r:
                                return r
                    if isinstance(obj, list):
                        for it in obj:
                            r = find_text(it)
                            if r:
                                return r
                    return None
                candidate = find_text(j)
                if candidate:
                    return candidate.strip()
            print("Gemini response unexpected; falling back to local summarizer.")
        except Exception as e:
            print("Gemini call failed:", type(e).__name__, e)

    # Local fallback
    try:
        _init_local_summarizer()
        if _local_summarizer:
            input_text = text if len(text) <= 3000 else text[:3000]
            out = _local_summarizer(input_text, max_length=110, min_length=30, do_sample=False)
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return out[0].get("summary_text", "").strip()
            return str(out)[:800]
    except Exception as e:
        print("Local summarizer failed:", e)

    # Last fallback: naive truncation and bullets
    short = (abstract or "")[:450]
    sents = [s.strip() for s in short.split(". ") if s.strip()]
    bullets = []
    for s in sents[:3]:
        bullets.append(f"- {s.strip()}.")
    takeaway = sents[0][:120] + "..." if sents else ""
    return "\n".join(bullets) + ("\n\nKey takeaway: " + takeaway if takeaway else "")
