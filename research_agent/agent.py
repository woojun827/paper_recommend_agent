from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from .config import SETTINGS
from .llm import LLMClient
from .models import AnalysisResult, Paper, ReadPaper, RecommendationResult, UserProfile
from .prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    RECOMMENDER_SYSTEM_PROMPT,
    build_analysis_prompt,
    build_recommendation_prompt,
)
from .scholarly_clients import ArxivClient, OpenAlexClient, SemanticScholarClient, merge_papers
from .scorer import rank_papers
from .utils import clip, ensure_dir, format_bullets, normalize_space, save_json, slugify, tokenize


class ResearchPaperAgent:
    def __init__(self) -> None:
        self.semantic = SemanticScholarClient()
        self.openalex = OpenAlexClient()
        self.arxiv = ArxivClient()
        self.llm = LLMClient()
        self.runs_dir = ensure_dir(Path("runs"))

    def expand_queries(self, query: str, profile: UserProfile) -> List[str]:
        base = normalize_space(query)
        keywords = [k.strip() for k in profile.interest_keywords if k.strip()]
        expansions = [base]
        if keywords:
            expansions.append(normalize_space(base + " " + " ".join(keywords[:3])))
        tokens = tokenize(base)
        if len(tokens) >= 2:
            expansions.append(" ".join(tokens[: min(4, len(tokens))]))
        if profile.freshness_mode == "classic":
            expansions.append(normalize_space(base + " foundational survey"))
        elif profile.freshness_mode == "latest":
            expansions.append(normalize_space(base + " recent benchmark"))
        else:
            expansions.append(normalize_space(base + " survey benchmark"))
        return list(dict.fromkeys([e for e in expansions if e]))

    def recommend(self, query: str, profile: UserProfile) -> RecommendationResult:
        expanded_queries = self.expand_queries(query, profile)
        ss_results: List[Paper] = []
        oa_results: List[Paper] = []
        arxiv_results: List[Paper] = []

        for q in expanded_queries[:4]:
            try:
                ss_results.extend(self.semantic.search_papers(q, limit=8))
            except Exception:
                pass
            try:
                if SETTINGS.openalex_api_key:
                    oa_results.extend(self.openalex.search_papers(q, limit=8))
            except Exception:
                pass
            try:
                arxiv_results.extend(self.arxiv.search_papers(q, limit=4))
            except Exception:
                pass

        merged = merge_papers(ss_results, oa_results, arxiv_results)
        ranked = rank_papers(merged, profile, query)
        shortlist = ranked[: SETTINGS.top_n_from_heuristic]

        recommended = shortlist[: profile.top_k]
        llm_json = self.llm.generate_json(
            RECOMMENDER_SYSTEM_PROMPT,
            build_recommendation_prompt(query, profile, shortlist, profile.top_k),
        )
        if llm_json and isinstance(llm_json.get("recommended"), list):
            llm_items = llm_json["recommended"]
            chosen: List[Paper] = []
            for item in llm_items:
                idx = item.get("candidate_no")
                if not isinstance(idx, int):
                    continue
                if 1 <= idx <= len(shortlist):
                    paper = shortlist[idx - 1]
                    paper.rationale = item.get("reason", "")
                    chosen.append(paper)
            if chosen:
                dedup = []
                seen = set()
                for paper in chosen:
                    if paper.key() not in seen:
                        seen.add(paper.key())
                        dedup.append(paper)
                recommended = dedup[: profile.top_k]

        run_path = self._save_recommendation_run(query, expanded_queries, ranked, recommended, profile)
        return RecommendationResult(
            query=query,
            expanded_queries=expanded_queries,
            candidates=ranked,
            recommended=recommended,
            run_path=run_path,
        )

    def analyze_selected_paper(self, selected_paper: Paper, profile: UserProfile) -> AnalysisResult:
        llm_json = self.llm.generate_json(
            ANALYSIS_SYSTEM_PROMPT,
            build_analysis_prompt(selected_paper, profile),
        )
        if llm_json:
            summary = llm_json.get("summary", "")
            why_read = llm_json.get("why_read", []) or []
            prerequisites = llm_json.get("prerequisites", []) or []
            reading_order = llm_json.get("reading_order", []) or []
            what_to_focus = llm_json.get("what_to_focus", []) or []
            caution_points = llm_json.get("caution_points", []) or []
            comparison_to_history = llm_json.get("comparison_to_history", "")
        else:
            summary, why_read, prerequisites, reading_order, what_to_focus, caution_points, comparison_to_history = self._fallback_analysis(selected_paper, profile)

        md = self._render_markdown_analysis(selected_paper, summary, why_read, prerequisites, reading_order, what_to_focus, caution_points, comparison_to_history)
        return AnalysisResult(
            paper=selected_paper,
            summary=summary,
            why_read=why_read,
            prerequisites=prerequisites,
            reading_order=reading_order,
            what_to_focus=what_to_focus,
            caution_points=caution_points,
            comparison_to_history=comparison_to_history,
            output_markdown=md,
        )

    def _fallback_analysis(self, paper: Paper, profile: UserProfile) -> Tuple[str, List[str], List[str], List[str], List[str], List[str], str]:
        summary = (
            f"'{paper.title}'은/는 {paper.year or '연도 미상'}년에 발표된 연구로, "
            f"주제상 {', '.join(paper.fields_of_study[:3]) or '관련 분야'} 맥락에서 읽을 가치가 있습니다. "
            f"초록 기준으로 보면 {clip(paper.abstract, 220)}"
        )
        why_read = [
            "현재 관심 키워드와의 관련성이 높음",
            f"인용 수 기준 참고 가치가 있음 (citation={paper.citation_count})",
            f"난이도는 {paper.estimated_difficulty}로 추정되어 현재 목표 수준과의 적합성을 확인 가능",
        ]
        prerequisites = [
            "제목과 초록에 등장하는 핵심 용어 3~5개를 먼저 정리",
            "기본 문제 설정(task setting)과 평가 지표가 무엇인지 미리 확인",
            "동일 주제의 survey/tutorial 논문이 있으면 먼저 읽기",
        ]
        reading_order = [
            "초록과 TL;DR로 문제 정의 파악",
            "서론에서 왜 이 문제가 중요한지 확인",
            "방법(Method)에서 기존 접근과의 차이를 한 줄로 요약",
            "실험(Experiment)에서 어떤 데이터/지표를 썼는지 체크",
            "결론에서 한계와 후속 과제를 기록",
        ]
        what_to_focus = [
            "이 논문이 해결하려는 정확한 문제는 무엇인가",
            "기존 방법 대비 무엇이 새롭다고 주장하는가",
            "실험 설계가 주장과 잘 연결되는가",
            "내 연구/관심 주제에 바로 연결할 수 있는 부분은 무엇인가",
        ]
        caution_points = [
            "현재 구현은 메타데이터 중심이므로 본문 세부 실험 수치까지는 알 수 없음",
            "인용 수가 높다고 해서 항상 가장 쉬운 입문 논문은 아님",
        ]
        history_titles = [rp.title for rp in profile.read_papers]
        comparison_to_history = (
            "이전 읽은 논문과의 직접 비교는 제한적이지만, "
            f"현재 히스토리({'; '.join(history_titles[:3]) or '없음'})를 기준으로 보면 "
            "기초-응용 사이의 연결 논문인지, 혹은 새로운 세부 분야로 확장하는 논문인지 확인하는 방식으로 읽는 것이 좋습니다."
        )
        return summary, why_read, prerequisites, reading_order, what_to_focus, caution_points, comparison_to_history

    def _render_markdown_analysis(
        self,
        paper: Paper,
        summary: str,
        why_read: List[str],
        prerequisites: List[str],
        reading_order: List[str],
        what_to_focus: List[str],
        caution_points: List[str],
        comparison_to_history: str,
    ) -> str:
        return f"""# Paper Reading Guide

## Selected Paper
- **Title**: {paper.title}
- **Year**: {paper.year or 'Unknown'}
- **Venue**: {paper.venue or 'Unknown'}
- **Authors**: {paper.short_authors() or 'Unknown'}
- **URL**: {paper.url or 'Unknown'}
- **PDF**: {paper.pdf_url or 'Unknown'}
- **Estimated Difficulty**: {paper.estimated_difficulty}
- **Citations**: {paper.citation_count}
- **Influential Citations**: {paper.influential_citation_count}

## Summary
{summary}

## Why Read This Paper
{format_bullets(why_read)}

## Recommended Prerequisites
{format_bullets(prerequisites)}

## Suggested Reading Order
{format_bullets(reading_order)}

## What To Focus On
{format_bullets(what_to_focus)}

## Caution Points
{format_bullets(caution_points)}

## Comparison To Previously Read Papers
{comparison_to_history}
"""

    def export_analysis(self, analysis: AnalysisResult, base_name: str | None = None) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = base_name or slugify(analysis.paper.title)
        path = self.runs_dir / f"{stamp}_{name}_analysis.md"
        path.write_text(analysis.output_markdown, encoding="utf-8")
        return str(path)

    def _save_recommendation_run(
        self,
        query: str,
        expanded_queries: List[str],
        ranked: List[Paper],
        recommended: List[Paper],
        profile: UserProfile,
    ) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.runs_dir / f"{stamp}_{slugify(query)}_recommendations.json"
        payload = {
            "query": query,
            "expanded_queries": expanded_queries,
            "profile": asdict(profile),
            "recommended": [p.to_dict() for p in recommended],
            "ranked": [p.to_dict() for p in ranked[:30]],
        }
        save_json(payload, path)
        return str(path)
