from __future__ import annotations

import math
from typing import Iterable, List

from .models import Paper, ReadPaper, UserProfile
from .utils import current_year, overlap_score, tokenize


BEGINNER_HINTS = {"survey", "tutorial", "review", "overview", "introduction", "primer"}
ADVANCED_HINTS = {"theorem", "proof", "provable", "optimization", "variational", "bayesian", "diffusion", "alignment", "scaling"}


def estimate_difficulty(paper: Paper) -> tuple[str, float]:
    title = (paper.title or "").lower()
    abstract = (paper.abstract or "").lower()
    score = 0.5

    if any(word in title for word in BEGINNER_HINTS):
        score -= 0.22
    if any(word in abstract for word in BEGINNER_HINTS):
        score -= 0.12
    if any(word in title for word in ADVANCED_HINTS):
        score += 0.18
    if any(word in abstract for word in ADVANCED_HINTS):
        score += 0.14
    if len(abstract.split()) > 220:
        score += 0.10
    if paper.citation_count > 1200:
        score -= 0.05
    if paper.venue.lower() in {"nature", "science"}:
        score += 0.05

    score = max(0.0, min(1.0, score))
    if score < 0.34:
        return "beginner", score
    if score < 0.67:
        return "intermediate", score
    return "advanced", score


def _difficulty_target_value(level: str) -> float:
    mapping = {"beginner": 0.15, "intermediate": 0.5, "advanced": 0.85}
    return mapping.get((level or "intermediate").lower(), 0.5)


def difficulty_match(paper_score: float, target_level: str) -> float:
    target_value = _difficulty_target_value(target_level)
    return max(0.0, 1.0 - abs(paper_score - target_value) / 0.7)


def prominence_score(paper: Paper) -> float:
    age = max(1, current_year() - (paper.year or current_year()) + 1)
    velocity = paper.citation_count / age
    raw = math.log1p(paper.citation_count) + 0.8 * math.log1p(paper.influential_citation_count) + 0.5 * math.log1p(velocity)
    return raw


def freshness_score(paper: Paper, freshness_mode: str) -> float:
    if not paper.year:
        return 0.3
    delta = max(0, current_year() - paper.year)
    recent = max(0.0, 1.0 - delta / 10)
    classic = min(1.0, delta / 10)
    mode = (freshness_mode or "balanced").lower()
    if mode == "latest":
        return recent
    if mode == "classic":
        return classic
    return 0.55 * recent + 0.45 * classic


def history_similarity(paper: Paper, read_papers: Iterable[ReadPaper]) -> float:
    paper_tokens = tokenize(paper.title + " " + paper.abstract)
    best = 0.0
    for rp in read_papers:
        rp_tokens = tokenize(rp.title + " " + rp.note)
        if not rp_tokens:
            continue
        overlap = len(set(paper_tokens) & set(rp_tokens)) / max(1, len(set(rp_tokens)))
        best = max(best, overlap)
    return best


def rank_papers(papers: List[Paper], profile: UserProfile, query: str) -> List[Paper]:
    query_terms = tokenize(" ".join(profile.interest_keywords + [query]))
    prominences = []
    for paper in papers:
        label, score = estimate_difficulty(paper)
        paper.estimated_difficulty = label
        paper.estimated_difficulty_score = score
        prominences.append(prominence_score(paper))

    min_prom = min(prominences) if prominences else 0.0
    max_prom = max(prominences) if prominences else 1.0
    denom = max(1e-9, max_prom - min_prom)

    for paper in papers:
        relevance = overlap_score(query_terms, paper.title, paper.abstract, paper.fields_of_study + paper.keywords)
        prom = (prominence_score(paper) - min_prom) / denom
        diff_fit = difficulty_match(paper.estimated_difficulty_score, profile.difficulty_target)
        freshness = freshness_score(paper, profile.freshness_mode)
        history_sim = history_similarity(paper, profile.read_papers)
        novelty = 1.0 - history_sim

        if history_sim > 0.88:
            novelty *= 0.4
        if paper.title.lower() in {rp.title.lower() for rp in profile.read_papers}:
            novelty = 0.0

        final = 0.42 * relevance + 0.24 * prom + 0.16 * diff_fit + 0.10 * novelty + 0.08 * freshness
        if any(k in (paper.title + " " + paper.abstract).lower() for k in ["survey", "tutorial", "review"]) and profile.difficulty_target != "advanced":
            final += 0.04
        paper.score_details = {
            "relevance": round(relevance, 4),
            "prominence": round(prom, 4),
            "difficulty_match": round(diff_fit, 4),
            "novelty": round(novelty, 4),
            "freshness": round(freshness, 4),
            "history_similarity": round(history_sim, 4),
        }
        paper.final_score = round(final, 6)

    return sorted(papers, key=lambda p: p.final_score, reverse=True)
