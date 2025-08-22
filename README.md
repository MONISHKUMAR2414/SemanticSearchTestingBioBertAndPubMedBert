🧬 PUBMED SEMANTIC SEARCH

A powerful tool to search PubMed articles using semantic embeddings!
Built with FastAPI + Streamlit, this app allows you to fetch, rank, and summarize PubMed articles using BioBERT or PubMedBERT, with optional MeSH mapping and Gemini summarization.

✨ KEY FEATURES

🔹 Natural Language Search – Ask in plain English, get relevant PubMed articles.

🔹 Article Details – Fetch title, abstract, authors, journal, year, and URL.

🔹 Multiple Embedding Models – Supports BioBERT and PubMedBERT.

🔹 MeSH Mapping – Optional semantic expansion for better results.

🔹 Gemini Summarization – Optional AI-generated summaries.

🔹 Interactive UI – View top-N results with abstracts and summaries.

🔹 CSV Export – Download all results for offline analysis.


Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux / Mac
.venv\Scripts\activate     # Windows


Install dependencies
pip install -r requirements.txt


Before running the Back End juss verify you are in  PS D:\SemanticSerachTesting\backend> uvicorn main:app --reload --port 8000
crt cd dir so 



Test Backend:
Visit http://localhost:8000/ping → should return:

{ "message": "Backend is alive!" }


then create new terminal for frontEnd running 

(.venv) PS D:\SemanticSerachTesting\frontend> streamlit run app.py --server.port 8501

📝 HOW TO USE

Step 1: Enter a medical query (e.g., "back pain").

Step 2: Set retmax – number of PubMed articles to fetch.

Step 3: Set top_k – number of top results to display.

Step 4: Enable MeSH mapping or Gemini summarization (optional).

Step 5: View top articles, abstracts, summaries, and download CSV.

<img width="1916" height="970" alt="Screenshot 2025-08-22 131007" src="https://github.com/user-attachments/assets/c0150bb5-fe4a-4523-9f2a-24b78e7fddcf" />

<img width="1896" height="965" alt="Screenshot 2025-08-22 130931" src="https://github.com/user-attachments/assets/054f8371-21a2-475d-ac2c-d60614cf41a9" />


🤝 CONTRIBUTING

Fork repository → create feature branch → make changes → push → open pull request.

Keep commits clear and descriptive.

📄 LICENSE

MIT License © 2025
