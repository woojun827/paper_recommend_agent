from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from research_agent.agent import ResearchPaperAgent
from research_agent.models import ReadPaper, UserProfile
from research_agent.utils import load_json, normalize_space


def parse_history(path: str | None) -> tuple[List[str], List[ReadPaper], str, str]:
    if not path:
        return [], [], "intermediate", "balanced"
    data = load_json(path)
    interest_keywords = data.get("interest_keywords", []) or []
    read_papers = [ReadPaper(**rp) for rp in (data.get("read_papers", []) or [])]
    difficulty = data.get("difficulty_target", "intermediate")
    freshness = data.get("freshness_mode", "balanced")
    return interest_keywords, read_papers, difficulty, freshness


def build_profile(args: argparse.Namespace) -> UserProfile:
    hist_keywords, hist_read, hist_diff, hist_fresh = parse_history(args.history)
    cli_keywords = [k.strip() for k in (args.keywords or "").split(",") if k.strip()]
    keywords = cli_keywords or hist_keywords or [normalize_space(args.query)]
    read_papers = hist_read
    difficulty = args.difficulty or hist_diff
    freshness = args.freshness or hist_fresh
    return UserProfile(
        interest_keywords=keywords,
        read_papers=read_papers,
        difficulty_target=difficulty,
        freshness_mode=freshness,
        language=args.language,
        top_k=args.top_k,
    )


def print_recommendations(result) -> None:
    print("\n=== 추천 논문 ===")
    for idx, paper in enumerate(result.recommended, start=1):
        print(f"\n[{idx}] {paper.title}")
        print(f"  - Year/Venue: {paper.year or 'Unknown'} / {paper.venue or 'Unknown'}")
        print(f"  - Authors: {paper.short_authors() or 'Unknown'}")
        print(f"  - Citations: {paper.citation_count} | Influential: {paper.influential_citation_count}")
        print(f"  - Difficulty: {paper.estimated_difficulty} | Score: {paper.final_score:.4f}")
        if paper.rationale:
            print(f"  - Why: {paper.rationale}")
        print(f"  - URL: {paper.url or 'Unknown'}")

    print(f"\n추천 결과 저장: {result.run_path}")


def interactive_mode(agent: ResearchPaperAgent) -> None:
    print("=== 대학원생용 연구 보조 GPT ===")
    query = input("현재 찾고 싶은 주제/질문: ").strip()
    keywords = input("관심 키워드(콤마로 구분, 비워도 됨): ").strip()
    difficulty = input("원하는 난이도(beginner/intermediate/advanced) [intermediate]: ").strip() or "intermediate"
    freshness = input("논문 성향(classic/balanced/latest) [balanced]: ").strip() or "balanced"
    top_k = int(input("추천 개수 [5]: ").strip() or "5")
    profile = UserProfile(
        interest_keywords=[k.strip() for k in keywords.split(",") if k.strip()] or [query],
        read_papers=[],
        difficulty_target=difficulty,
        freshness_mode=freshness,
        language="ko",
        top_k=top_k,
    )
    result = agent.recommend(query, profile)
    print_recommendations(result)
    choice = input("\n분석할 논문 번호를 선택하세요 (그만하려면 Enter): ").strip()
    if not choice:
        return
    idx = int(choice) - 1
    if idx < 0 or idx >= len(result.recommended):
        print("잘못된 번호입니다.")
        return
    analysis = agent.analyze_selected_paper(result.recommended[idx], profile)
    path = agent.export_analysis(analysis)
    print("\n=== 읽기 가이드 ===")
    print(analysis.output_markdown)
    print(f"\n분석 결과 저장: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Graduate Research Assistant GPT")
    sub = parser.add_subparsers(dest="command")

    recommend = sub.add_parser("recommend", help="추천 논문 생성")
    recommend.add_argument("--query", required=True)
    recommend.add_argument("--keywords", default="")
    recommend.add_argument("--history", default=None, help="history_example.json 같은 사용자 히스토리 파일")
    recommend.add_argument("--difficulty", default=None, choices=["beginner", "intermediate", "advanced"])
    recommend.add_argument("--freshness", default=None, choices=["classic", "balanced", "latest"])
    recommend.add_argument("--top-k", type=int, default=5)
    recommend.add_argument("--language", default="ko")
    recommend.add_argument("--analyze-rank", type=int, default=None, help="추천 후 바로 분석할 논문 순위")

    analyze = sub.add_parser("analyze", help="recommend 결과 파일에서 선택 논문 분석")
    analyze.add_argument("--run-json", required=True)
    analyze.add_argument("--rank", type=int, required=True)
    analyze.add_argument("--history", default=None)
    analyze.add_argument("--language", default="ko")

    args = parser.parse_args()
    agent = ResearchPaperAgent()

    if not args.command:
        interactive_mode(agent)
        return

    if args.command == "recommend":
        profile = build_profile(args)
        result = agent.recommend(args.query, profile)
        print_recommendations(result)
        if args.analyze_rank is not None:
            idx = args.analyze_rank - 1
            if 0 <= idx < len(result.recommended):
                analysis = agent.analyze_selected_paper(result.recommended[idx], profile)
                path = agent.export_analysis(analysis)
                print("\n=== 읽기 가이드 ===")
                print(analysis.output_markdown)
                print(f"\n분석 결과 저장: {path}")
        return

    if args.command == "analyze":
        run_data = load_json(args.run_json)
        recommended = run_data.get("recommended", [])
        idx = args.rank - 1
        if idx < 0 or idx >= len(recommended):
            raise SystemExit("rank가 추천 목록 범위를 벗어났습니다.")
        hist_keywords, hist_read, hist_diff, hist_fresh = parse_history(args.history)
        profile = UserProfile(
            interest_keywords=hist_keywords or [run_data.get("query", "")],
            read_papers=hist_read,
            difficulty_target=hist_diff,
            freshness_mode=hist_fresh,
            language=args.language,
            top_k=5,
        )
        paper_dict = recommended[idx]
        from research_agent.models import Paper
        paper = Paper(**paper_dict)
        analysis = agent.analyze_selected_paper(paper, profile)
        path = agent.export_analysis(analysis)
        print(analysis.output_markdown)
        print(f"\n분석 결과 저장: {path}")


if __name__ == "__main__":
    main()
