from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ReadPaper:
    title: str
    note: str = ""
    liked: bool = True


@dataclass
class UserProfile:
    interest_keywords: List[str]
    read_papers: List[ReadPaper] = field(default_factory=list)
    difficulty_target: str = "intermediate"
    freshness_mode: str = "balanced"  # classic | balanced | latest
    language: str = "ko"
    top_k: int = 5


@dataclass
class Paper:
    source: str
    paper_id: str = ""
    title: str = ""
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    authors: List[str] = field(default_factory=list)
    citation_count: int = 0
    influential_citation_count: int = 0
    reference_count: int = 0
    fields_of_study: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    tldr: str = ""
    url: str = ""
    pdf_url: str = ""
    doi: str = ""
    openalex_id: str = ""
    semantic_scholar_id: str = ""
    score_details: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    estimated_difficulty: str = "intermediate"
    estimated_difficulty_score: float = 0.5
    rationale: str = ""

    def short_authors(self, max_authors: int = 3) -> str:
        if not self.authors:
            return ""
        if len(self.authors) <= max_authors:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:max_authors]) + f" 외 {len(self.authors) - max_authors}명"

    def key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.lower()}"
        if self.semantic_scholar_id:
            return f"s2:{self.semantic_scholar_id}"
        if self.openalex_id:
            return f"oa:{self.openalex_id}"
        return self.title.strip().lower()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationResult:
    query: str
    expanded_queries: List[str]
    candidates: List[Paper]
    recommended: List[Paper]
    run_path: str = ""


@dataclass
class AnalysisResult:
    paper: Paper
    summary: str
    why_read: List[str]
    prerequisites: List[str]
    reading_order: List[str]
    what_to_focus: List[str]
    caution_points: List[str]
    comparison_to_history: str
    output_markdown: str
