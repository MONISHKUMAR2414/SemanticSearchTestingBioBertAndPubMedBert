from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from search_logic import run_search_pipeline

# 🔹 Always load backend/.env explicitly
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

app = FastAPI(title="PubMed Semantic Search API")

# --- Debug endpoints (you can remove later if not needed) ---
@app.get("/ping")
def ping():
    return {"message": "Backend is alive!"}

@app.get("/keys")
def get_keys():
    return {
        "NCBI_EMAIL": os.getenv("NCBI_EMAIL"),
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
        "BACKEND_URL": os.getenv("BACKEND_URL", "http://localhost:8000")
    }
# -----------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    retmax: Optional[int] = 200
    top_k: Optional[int] = 10
    model_name: Optional[str] = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    use_mesh: Optional[bool] = True
    use_gemini_summary: Optional[bool] = False
    gemini_api_key: Optional[str] = None

@app.post("/search")
def search(req: SearchRequest):
    ncbi_email = os.getenv("NCBI_EMAIL")
    if not ncbi_email:
        raise HTTPException(status_code=400, detail="NCBI_EMAIL environment variable not set.")

    ncbi_api_key = os.getenv("NCBI_API_KEY")

    try:
        out = run_search_pipeline(
            query=req.query,
            retmax=req.retmax,
            top_k=req.top_k,
            model_name=req.model_name,
            ncbi_email=ncbi_email,
            ncbi_api_key=ncbi_api_key,
            use_mesh=req.use_mesh,
            use_gemini_summary=req.use_gemini_summary,
            gemini_api_key=req.gemini_api_key
        )
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
