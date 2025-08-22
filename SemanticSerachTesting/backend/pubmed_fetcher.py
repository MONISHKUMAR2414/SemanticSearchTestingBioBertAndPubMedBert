# backend/pubmed_fetcher.py
from dataclasses import dataclass
from typing import List, Optional
from Bio import Entrez
import re

@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    url: str
    journal: Optional[str] = None
    year: Optional[str] = None
    authors: Optional[List[str]] = None

def _parse_article(entry: dict) -> Optional[PubMedArticle]:
    try:
        medline = entry.get("MedlineCitation", {})
        pmid_raw = medline.get("PMID")
        pmid = pmid_raw.get("#text") if isinstance(pmid_raw, dict) else str(pmid_raw)
        if not pmid:
            return None

        article = medline.get("Article", {})
        title = article.get("ArticleTitle") or ""
        abstract_parts = article.get("Abstract", {}).get("AbstractText")
        abstract = ""

        if isinstance(abstract_parts, list):
            texts = []
            for x in abstract_parts:
                if hasattr(x, "content"):
                    texts.append(str(x.content))
                else:
                    texts.append(str(x))
            abstract = " ".join(texts)
        elif isinstance(abstract_parts, str):
            abstract = abstract_parts
        elif abstract_parts:
            abstract = str(abstract_parts)

        abstract = re.sub(r"<[^>]+>", "", abstract)
        abstract = re.sub(r"\s+", " ", abstract).strip()

        journal_info = article.get("Journal", {}).get("Title")
        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year") or pub_date.get("MedlineDate")
        authors_list = article.get("AuthorList") or []
        authors = []
        for a in authors_list:
            last = a.get("LastName")
            fore = a.get("ForeName") or a.get("Initials")
            if last and fore:
                authors.append(f"{fore} {last}")
            elif last:
                authors.append(last)

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        return PubMedArticle(
            pmid=str(pmid),
            title=title if isinstance(title, str) else str(title),
            abstract=abstract,
            url=url,
            journal=journal_info if isinstance(journal_info, str) else None,
            year=str(year) if year else None,
            authors=authors if authors else None,
        )
    except Exception as e:
        print("Parsing error:", e)
        return None

def fetch_pubmed_articles(query: str, retmax: int = 50, email: Optional[str] = None, api_key: Optional[str] = None) -> List[PubMedArticle]:
    """
    Search and fetch PubMed articles; returns PubMedArticle objects.
    """
    Entrez.email = (email or "").strip() or "pubmed-semantic@example.com"
    if api_key:
        Entrez.api_key = api_key

    handle = Entrez.esearch(db="pubmed", term=query, retmax= retmax)
    record = Entrez.read(handle)
    handle.close()
    id_list = record.get("IdList", [])
    if not id_list:
        return []

    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="xml")
    fetch_record = Entrez.read(handle)
    handle.close()

    articles = []
    for entry in fetch_record.get("PubmedArticle", []):
        parsed = _parse_article(entry)
        if parsed:
            articles.append(parsed)
    return articles
