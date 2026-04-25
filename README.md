# 대학원생용 연구 보조 GPT

논문 스터디를 할 때 다음 문제가 자주 생깁니다.

- 지금 내 관심 키워드와 **정말 직접적으로 연결되는 논문**이 무엇인지 찾기 어려움
- 너무 최신 논문만 보면 기초가 비고, 너무 고전만 보면 현재 트렌드를 놓치기 쉬움
- **저명성(인용 수, 영향력)** 과 **난이도** 를 같이 고려해 논문을 고르기 어려움
- 이미 읽은 논문과 너무 비슷한 논문을 또 읽게 되는 경우가 많음

이 프로젝트는 위 문제를 해결하기 위한 **연구 보조 에이전트**입니다.

핵심 기능:
1. 관심 키워드 + 이전에 읽은 논문 + 원하는 난이도 + 논문 성향(classic/balanced/latest)을 입력받음
2. Semantic Scholar / OpenAlex / arXiv에서 후보 논문을 수집함
3. 관련성, 저명성, 난이도 적합도, 새로움, 최신성 기준으로 점수화함
4. 상위 논문을 추천하고, 사용자가 하나를 선택하면 **읽기 방향성**까지 제시함
5. 결과를 JSON/Markdown으로 저장하여 GitHub/Notion/과제 보고서에 재사용 가능

---

## 1. 왜 이 주제가 과제와 맞는가

사진 속 과제 설명 기준으로 보면, 이 프로젝트는 아래 항목과 직접 맞닿아 있습니다.

- **주제 2: LLM을 활용한 (전공) 연구 보조 파이프라인 구축 및 평가**
  - 논문 탐색 → 후보 수집 → 점수화 → 추천 → 논문 읽기 가이드 생성이라는 파이프라인을 실제로 구현
- **주제 3: 전공 특화 GPTs 개발 프로젝트**
  - 일반 챗봇이 아니라, 대학원생의 논문 탐색/스터디라는 특정 업무에 특화된 GPT

또한 평가 항목 중 중요한 부분에도 잘 대응합니다.

- 개념 연계: Prompt Engineering, LLM, AI Design & Control, Responsible AI 반영
- 결과물 완성도: 실제 실행 가능한 코드 제공
- 비판적 성찰: 환각, 최신성, 인용 수 편향, 메타데이터 한계 등을 명시 가능
- 논리적 구성: 문제 정의 → 설계 → 구현 → 평가 → 한계 → 개선안으로 쓰기 쉬움

---

## 2. 강의록과의 직접 연계 포인트

### A. Prompt Engineering
이 프로젝트의 핵심은 “좋은 논문 추천”을 그냥 모델에게 막 묻는 것이 아니라,
**목적성 / 맥락 제공 / 구체성 / 예시 / 구조화** 를 반영한 프롬프트로 추천 품질을 높이는 데 있습니다.

구현 반영 방식:
- 역할(Role): `senior graduate research mentor`
- 맥락(Context): 관심 키워드, 이전에 읽은 논문, 난이도 목표, 최신/고전 선호
- 목적(Objective): must-read 논문 추천
- 구조화(Response): JSON만 반환하도록 강제
- 예시(Few-shot): 추천 JSON 예시 포함
- 단계화: 먼저 heuristic shortlist, 이후 LLM 정제

즉, 수업에서 말한 “질문을 잘 설계하는 것이 AI를 효과적으로 활용하는 핵심 전략”을 직접 시스템으로 옮긴 형태입니다.

### B. LLM
LLM은 단순 검색기가 아니라, 후보군 중 무엇이 더 적절한지 **설명 가능한 추천 사유**와
**읽기 방향성**을 생성하는 데 사용됩니다.

구현 반영 방식:
- 후보 논문을 먼저 메타데이터 기반으로 압축
- 컨텍스트 윈도우 비용과 정보 손실을 줄이기 위해 top-N shortlist만 LLM에 전달
- LLM이 추천 이유와 읽기 가이드를 생성
- OpenAI 키가 없을 경우 heuristic-only 모드로도 동작

### C. AI Design & Control
강의에서 강조한 것처럼, 모델 성능이 곧 시스템 성능은 아닙니다.
이 프로젝트는 “프롬프트만 잘 쓰는 챗봇”이 아니라, **입력-맥락-출력-검증** 루프를 갖춘 구조로 설계했습니다.

구현 반영 방식:
- 입력 정규화: query / history / difficulty / freshness 분리
- 출력 포맷 강제: JSON only
- 1차 heuristic ranking → 2차 LLM recommendation
- JSON 파싱 실패 시 fallback
- 추천 결과, 점수, 쿼리 확장 결과를 run 파일로 저장
- human-in-the-loop: 최종 논문 선택은 사용자

### D. Responsible AI
연구 보조 시스템에서는 “그럴듯한 환각”이 특히 위험합니다.
그래서 본 시스템은 **메타데이터 밖의 정보는 모른다고 말하도록** 설계했습니다.

구현 반영 방식:
- system prompt에서 “provided metadata only” 명시
- 실험 세부 수치/기여점 과장 금지
- 없는 정보는 unknown 처리
- 로그 저장으로 추적 가능성 확보

---

## 3. 시스템 구조

```text
[사용자 입력]
  └─ query, 관심 키워드, 이전에 읽은 논문, 난이도, freshness
        ↓
[Query Expansion]
        ↓
[Scholarly Retrieval]
  ├─ Semantic Scholar
  ├─ OpenAlex (optional)
  └─ arXiv
        ↓
[Merge & Deduplicate]
        ↓
[Heuristic Scoring]
  ├─ relevance
  ├─ prominence
  ├─ difficulty_match
  ├─ novelty
  └─ freshness
        ↓
[LLM Re-ranking / Explanation] (optional)
        ↓
[Top-K Recommendation]
        ↓
[Human Selection]
        ↓
[Reading Guide Generation]
        ↓
[JSON/Markdown Export]
```

---

## 4. 추천 점수는 어떻게 계산되는가

최종 점수는 대략 아래 조합입니다.

- **관련성(relevance)**: 관심 키워드가 제목/초록/분야와 얼마나 겹치는가
- **저명성(prominence)**: citation_count, influential_citation_count, citation velocity
- **난이도 적합도(difficulty match)**: beginner/intermediate/advanced와의 차이
- **새로움(novelty)**: 이미 읽은 논문과 얼마나 덜 겹치는가
- **최신성(freshness)**: classic / balanced / latest 선호 반영

주의:
- 인용 수가 높다고 무조건 쉬운 논문은 아님
- 너무 최신 논문은 인용 수가 낮아 과소평가될 수 있음
- 그래서 classic/balanced/latest 모드를 별도로 둠

---

## 5. 설치

```bash
cd research_paper_agent
python -m venv .venv
```

### Windows PowerShell
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### macOS / Linux
```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env`에서 API 키를 넣으면 LLM 추천 품질이 올라갑니다.

- `OPENAI_API_KEY`: 선택 사항. 있으면 추천 이유/읽기 가이드가 더 자연스러워짐
- `SEMANTIC_SCHOLAR_API_KEY`: 선택 사항. 안정성 향상용
- `OPENALEX_API_KEY`: 선택 사항. OpenAlex 검색 사용 시 필요

OpenAI 키가 없으면 **heuristic-only 모드**로도 동작합니다.

---

## 6. 실행 방법

### 6-1. 대화형 실행
```bash
python app.py
```

### 6-2. 추천만 실행
```bash
python app.py recommend \
  --query "retrieval augmented generation evaluation" \
  --history data/history_example.json \
  --top-k 5
```

### 6-3. 추천 후 바로 1위 논문 분석
```bash
python app.py recommend \
  --query "vision-language model survey" \
  --history data/history_example.json \
  --top-k 5 \
  --analyze-rank 1
```

### 6-4. 이미 저장된 추천 결과에서 2위 논문 분석
```bash
python app.py analyze \
  --run-json runs/20260424_123456_rag-evaluation_recommendations.json \
  --rank 2 \
  --history data/history_example.json
```

---

## 7. 입력 히스토리 파일 형식

`data/history_example.json` 참고.

```json
{
  "interest_keywords": ["retrieval augmented generation", "LLM evaluation"],
  "read_papers": [
    {
      "title": "Attention Is All You Need",
      "note": "Transformer 기본 구조는 이해함",
      "liked": true
    }
  ],
  "difficulty_target": "intermediate",
  "freshness_mode": "balanced"
}
```

---

## 8. 산출물

실행하면 `runs/` 폴더에 아래 파일이 저장됩니다.

- `*_recommendations.json`
  - 추천 논문 목록
  - 점수 상세
  - 쿼리 확장 결과
- `*_analysis.md`
  - 선택 논문 읽기 가이드
  - 왜 읽어야 하는지
  - 선행지식
  - 읽는 순서
  - 주의점

이 파일은 과제 제출용 GitHub/Notion 정리에도 바로 활용 가능합니다.

---

## 9. 이 구현의 장점

- 실제로 돌릴 수 있는 코드
- LLM 키가 없어도 최소 기능 작동
- 점수 기반 + LLM 설명 기반의 **하이브리드 추천 구조**
- Human-in-the-loop 구조
- 결과 로그 저장
- 과제용 보고서/발표로 연결하기 쉬움

---

## 10. 한계

- 현재는 **논문 본문 전체 PDF 파싱**까지는 하지 않음
- 난이도 추정은 heuristic이므로 완벽하지 않음
- 저명성 판단이 인용 수에 일부 의존하므로 신생 분야에는 불리할 수 있음
- OpenAlex는 API 키가 있어야 더 안정적으로 사용 가능
- LLM이 없으면 읽기 가이드의 자연스러움은 다소 떨어질 수 있음

---

## 11. 다음 개선안

- PDF 업로드 후 본문 section-level 분석
- citation graph 기반 “읽는 순서” 자동 생성
- 분야별 난이도 추정 모델 추가
- Streamlit UI 추가
- BibTeX / APA export 지원
- 사용자가 “좋아요/별로” 피드백을 남기면 다음 추천에 반영

---

## 12. 추천 발표 포인트

발표에서는 아래 흐름으로 가면 좋습니다.

1. 문제 상황: 논문은 많지만 지금 나에게 맞는 must-read를 고르기 어려움
2. 기존 방식 한계: 검색 결과 나열, 인용 수만 보기, ChatGPT에 그냥 물어보기
3. 제안 시스템: Retrieval + Scoring + Prompt Engineering + Reading Guide
4. 강의 연계: Prompt Engineering / LLM / AI Design & Control / Responsible AI
5. 데모: 키워드 입력 → 추천 → 선택 → 읽기 가이드 생성
6. 한계와 개선안

