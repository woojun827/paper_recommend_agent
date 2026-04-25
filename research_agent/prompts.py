from __future__ import annotations

import json
from typing import List

from .models import Paper, UserProfile
from .utils import clip


RECOMMENDER_SYSTEM_PROMPT = """
You are a senior graduate research mentor and literature scout.
Your job is to recommend must-read papers for a graduate student.
Use ONLY the provided paper metadata.
Never invent claims, datasets, results, or prerequisites that are not supported by the metadata.
If information is missing, say it is unknown.
Return valid JSON only.
""".strip()


ANALYSIS_SYSTEM_PROMPT = """
You are a graduate-level paper reading coach.
Use ONLY the provided metadata and user history.
Do not fabricate experimental details or claims not present in the metadata.
Give practical reading guidance in Korean.
Return valid JSON only.
""".strip()


# Co-STAR-inspired structured prompt: Context / Objective / Style / Tone / Audience / Response

def build_recommendation_prompt(query: str, profile: UserProfile, candidates: List[Paper], top_k: int) -> str:
    condensed = []
    for idx, p in enumerate(candidates, start=1):
        condensed.append(
            {
                "candidate_no": idx,
                "title": p.title,
                "year": p.year,
                "venue": p.venue,
                "authors": p.authors[:5],
                "citation_count": p.citation_count,
                "influential_citation_count": p.influential_citation_count,
                "fields_of_study": p.fields_of_study,
                "tldr": clip(p.tldr, 180),
                "abstract": clip(p.abstract, 500),
                "estimated_difficulty": p.estimated_difficulty,
                "heuristic_score": round(p.final_score, 4),
                "url": p.url,
            }
        )
    prompt = {
        "Context": {
            "task": "Recommend must-read papers for the user's current study need.",
            "user_query": query,
            "interest_keywords": profile.interest_keywords,
            "previously_read_papers": [rp.title for rp in profile.read_papers],
            "difficulty_target": profile.difficulty_target,
            "freshness_mode": profile.freshness_mode,
            "language": profile.language,
        },
        "Objective": {
            "goal": f"Select the best {top_k} papers from the candidate pool.",
            "selection_rules": [
                "Balance topical relevance, prominence, and difficulty fit.",
                "Prefer must-read or foundational papers when appropriate.",
                "Avoid recommending papers too similar to what the user has already read unless they are truly essential.",
                "Do not claim a paper is seminal unless the metadata strongly supports it."
            ]
        },
        "Style": "Compact, practical, advisor-like.",
        "Tone": "Professional but supportive.",
        "Audience": "A graduate student planning the next paper-study session.",
        "Response": {
            "format": {
                "recommended": [
                    {
                        "candidate_no": 1,
                        "reason": "why it matches the user's need",
                        "fit_summary": "1-2 sentence summary",
                        "reading_priority": "high|medium",
                        "expected_takeaway": "what the user will learn",
                        "difficulty_note": "difficulty guidance"
                    }
                ],
                "query_reframing": ["optional refined search phrase 1", "optional refined search phrase 2"],
                "advice": "short advice on how to proceed"
            },
            "json_only": True
        },
        "Candidates": condensed,
        "FewShotExample": {
            "recommended": [
                {
                    "candidate_no": 2,
                    "reason": "Directly matches the user's current topic and is widely recognized as a starting point.",
                    "fit_summary": "Good bridge paper between fundamentals and current interest.",
                    "reading_priority": "high",
                    "expected_takeaway": "Understand the core task setup and standard evaluation lens.",
                    "difficulty_note": "Manageable for an intermediate student if read after one survey paper."
                }
            ],
            "query_reframing": [
                "retrieval augmented generation evaluation survey",
                "foundational papers on retrieval augmented generation"
            ],
            "advice": "Read one survey or foundational paper first, then move to the most recent strong benchmark paper."
        }
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def build_analysis_prompt(selected_paper: Paper, profile: UserProfile) -> str:
    prompt = {
        "Context": {
            "paper": {
                "title": selected_paper.title,
                "year": selected_paper.year,
                "venue": selected_paper.venue,
                "authors": selected_paper.authors,
                "citation_count": selected_paper.citation_count,
                "influential_citation_count": selected_paper.influential_citation_count,
                "fields_of_study": selected_paper.fields_of_study,
                "tldr": selected_paper.tldr,
                "abstract": clip(selected_paper.abstract, 1800),
                "url": selected_paper.url,
            },
            "user_profile": {
                "interest_keywords": profile.interest_keywords,
                "previously_read_papers": [rp.title for rp in profile.read_papers],
                "difficulty_target": profile.difficulty_target,
                "language": profile.language,
            }
        },
        "Objective": {
            "goal": "Explain why this paper matters and provide a practical reading path.",
            "rules": [
                "Do not invent details not present in title/abstract/TLDR/metadata.",
                "Focus on how the student should read the paper, not just what it is.",
                "Be explicit about unknowns when metadata is insufficient."
            ]
        },
        "Style": "Actionable and study-oriented.",
        "Tone": "Mentoring tone, concise but not shallow.",
        "Audience": "Graduate student preparing a paper study session.",
        "Response": {
            "format": {
                "summary": "short paragraph",
                "why_read": ["bullet"],
                "prerequisites": ["bullet"],
                "reading_order": ["bullet"],
                "what_to_focus": ["bullet"],
                "caution_points": ["bullet"],
                "comparison_to_history": "short paragraph"
            },
            "json_only": True
        }
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)
