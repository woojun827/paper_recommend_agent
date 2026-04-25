from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

from .config import SETTINGS
from .models import Paper
from .utils import clip, current_year, ensure_dir, hash_key, normalize_space


class CachedSession:
    def __init__(self, cache_subdir: str = "http") -> None:
        self.session = requests.Session()
        self.cache_dir = ensure_dir(SETTINGS.cache_dir / cache_subdir)

    def get_json(self, url: str, *, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict:
        key = hash_key([url, json.dumps(params or {}, sort_keys=True), json.dumps(headers or {}, sort_keys=True)])
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        response = self.session.get(url, params=params, headers=headers, timeout=SETTINGS.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.1)
        return data


class SemanticScholarClient:
    BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(self) -> None:
        self.http = CachedSession("semantic_scholar")

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if SETTINGS.semantic_scholar_api_key:
            headers["x-api-key"] = SETTINGS.semantic_scholar_api_key
        return headers

    def search_papers(self, query: str, limit: int = 10) -> List[Paper]:
        fields = ",".join([
            "paperId", "title", "abstract", "year", "authors", "venue", "citationCount",
            "influentialCitationCount", "referenceCount", "fieldsOfStudy", "tldr", "url",
            "externalIds", "openAccessPdf"
        ])
        data = self.http.get_json(
            f"{self.BASE}/paper/search",
            params={"query": query, "limit": limit, "fields": fields},
            headers=self._headers(),
        )
        papers: List[Paper] = []
        for item in data.get("data", []):
            external_ids = item.get("externalIds") or {}
            oa_pdf = item.get("openAccessPdf") or {}
            papers.append(
                Paper(
                    source="semantic_scholar",
                    paper_id=item.get("paperId", ""),
                    semantic_scholar_id=item.get("paperId", ""),
                    title=normalize_space(item.get("title", "")),
                    abstract=normalize_space(item.get("abstract", "")),
                    year=item.get("year"),
                    venue=normalize_space(item.get("venue", "")),
                    authors=[a.get("name", "") for a in item.get("authors", []) if a.get("name")],
                    citation_count=item.get("citationCount") or 0,
                    influential_citation_count=item.get("influentialCitationCount") or 0,
                    reference_count=item.get("referenceCount") or 0,
                    fields_of_study=item.get("fieldsOfStudy") or [],
                    tldr=normalize_space((item.get("tldr") or {}).get("text", "")),
                    url=item.get("url", ""),
                    pdf_url=(oa_pdf or {}).get("url", ""),
                    doi=(external_ids.get("DOI") or "").lower(),
                )
            )
        return papers


class OpenAlexClient:
    BASE = "https://api.openalex.org"

    def __init__(self) -> None:
        self.http = CachedSession("openalex")

    def _params(self, extra: Optional[Dict] = None) -> Dict:
        params = dict(extra or {})
        if SETTINGS.openalex_api_key:
            params["api_key"] = SETTINGS.openalex_api_key
        return params

    def search_papers(self, query: str, limit: int = 10) -> List[Paper]:
        data = self.http.get_json(
            f"{self.BASE}/works",
            params=self._params({"search": query, "per-page": limit}),
            headers={"Accept": "application/json"},
        )
        papers: List[Paper] = []
        for item in data.get("results", []):
            papers.append(self._parse_work(item))
        return papers

    def _parse_work(self, item: Dict) -> Paper:
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        authorships = item.get("authorships") or []
        concepts = item.get("keywords") or []
        ids = item.get("ids") or {}
        abstract = self._decode_inverted_index(item.get("abstract_inverted_index") or {})
        return Paper(
            source="openalex",
            openalex_id=item.get("id", ""),
            paper_id=item.get("id", ""),
            title=normalize_space(item.get("display_name", "")),
            abstract=normalize_space(abstract),
            year=item.get("publication_year"),
            venue=normalize_space(source.get("display_name", "")),
            authors=[a.get("author", {}).get("display_name", "") for a in authorships if a.get("author")],
            citation_count=item.get("cited_by_count") or 0,
            influential_citation_count=0,
            reference_count=item.get("referenced_works_count") or 0,
            fields_of_study=[
                t.get("display_name", "") for t in (item.get("topics") or []) if t.get("display_name")
            ],
            keywords=[c.get("display_name", "") for c in concepts if c.get("display_name")],
            url=item.get("id", ""),
            pdf_url=(primary_location.get("pdf_url") or ""),
            doi=(ids.get("doi") or "").replace("https://doi.org/", "").lower(),
        )

    @staticmethod
    def _decode_inverted_index(index: Dict[str, List[int]]) -> str:
        if not index:
            return ""
        max_pos = max((max(v) for v in index.values() if v), default=-1)
        if max_pos < 0:
            return ""
        words = [""] * (max_pos + 1)
        for token, positions in index.items():
            for pos in positions:
                if 0 <= pos < len(words):
                    words[pos] = token
        return " ".join(words)


class ArxivClient:
    BASE = "https://export.arxiv.org/api/query"

    def __init__(self) -> None:
        self.session = requests.Session()

    def search_papers(self, query: str, limit: int = 5) -> List[Paper]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        response = self.session.get(self.BASE, params=params, timeout=SETTINGS.timeout_seconds)
        response.raise_for_status()
        text = response.text
        return self._parse_atom(text)

    def _parse_atom(self, xml_text: str) -> List[Paper]:
        import xml.etree.ElementTree as ET

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(xml_text)
        papers: List[Paper] = []
        for entry in root.findall("atom:entry", ns):
            title = normalize_space(entry.findtext("atom:title", default="", namespaces=ns))
            summary = normalize_space(entry.findtext("atom:summary", default="", namespaces=ns))
            entry_id = entry.findtext("atom:id", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns)
            year = None
            if published and len(published) >= 4:
                try:
                    year = int(published[:4])
                except ValueError:
                    year = None
            authors = [normalize_space(a.findtext("atom:name", default="", namespaces=ns)) for a in entry.findall("atom:author", ns)]
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            papers.append(
                Paper(
                    source="arxiv",
                    paper_id=entry_id,
                    title=title,
                    abstract=summary,
                    year=year,
                    venue="arXiv",
                    authors=[a for a in authors if a],
                    citation_count=0,
                    influential_citation_count=0,
                    fields_of_study=[normalize_space(entry.find("arxiv:primary_category", ns).attrib.get("term", ""))] if entry.find("arxiv:primary_category", ns) is not None else [],
                    url=entry_id,
                    pdf_url=pdf_url,
                )
            )
        return papers


def merge_papers(*paper_groups: Iterable[Paper]) -> List[Paper]:
    merged: Dict[str, Paper] = {}
    for group in paper_groups:
        for paper in group:
            key = paper.key()
            if key not in merged:
                merged[key] = paper
                continue
            existing = merged[key]
            if not existing.abstract and paper.abstract:
                existing.abstract = paper.abstract
            if not existing.tldr and paper.tldr:
                existing.tldr = paper.tldr
            if not existing.pdf_url and paper.pdf_url:
                existing.pdf_url = paper.pdf_url
            if not existing.url and paper.url:
                existing.url = paper.url
            existing.citation_count = max(existing.citation_count, paper.citation_count)
            existing.influential_citation_count = max(existing.influential_citation_count, paper.influential_citation_count)
            if not existing.venue and paper.venue:
                existing.venue = paper.venue
            if not existing.year and paper.year:
                existing.year = paper.year
            existing.fields_of_study = list(dict.fromkeys(existing.fields_of_study + paper.fields_of_study))
            existing.keywords = list(dict.fromkeys(existing.keywords + paper.keywords))
            if paper.openalex_id and not existing.openalex_id:
                existing.openalex_id = paper.openalex_id
            if paper.semantic_scholar_id and not existing.semantic_scholar_id:
                existing.semantic_scholar_id = paper.semantic_scholar_id
            if paper.doi and not existing.doi:
                existing.doi = paper.doi
            if paper.authors and not existing.authors:
                existing.authors = paper.authors
    return list(merged.values())
