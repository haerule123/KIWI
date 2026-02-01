# KIWI

앱 리뷰를 **수집 → 문장 분리 → ABSA(속성·감성 분석) → 보고서 생성(L1~L4) → RAG 기반 질의응답**으로 연결하고,
Django 대시보드에서 보고서를 조회/저장/PDF로 내려받을 수 있게 만든 리뷰 분석 플랫폼입니다.

---

## 한 줄 요약

- Google Play Store 리뷰를 MySQL에 적재하고, 문장 단위로 쪼갠 뒤 ABSA로 라벨링/스코어링하여 보고서를 만들고, 그 보고서를 벡터화해 RAG 챗으로 질문에 답하는 시스템

---

## 문제 정의

- 앱 리뷰는 양이 많고(지속적으로 누적), **버전별/기간별 이슈**가 섞여 있어 수작업으로 핵심을 뽑기 어렵습니다.
- 단순 키워드 검색만으로는 “왜 불만이 늘었는지”, “어느 기능이 문제인지” 같은 **맥락 기반 질문**에 답하기 어렵습니다.

---

## 전체 아키텍처(개요)

```text

Track A) 실제 보고서 생성/서비스 파이프라인 (주로 Google Play Store 기반)

[Crawling]
	- Google Play Store: crawling/crawler.py
				|
				v
[MySQL]
	app / version / review
				|
				v
[Sentence Split]
	review -> review_line (문장 단위)
	- RAG-for-report/kssds_line_splitter.py (DB 저장)
				|
				v
[ABSA]
	review_line -> analysis (aspect/sentiment + score)
	- fine_tuning/model2.py (학습)
	- fine_tuning/predict2.py (DB에 결과 적재)
				|
				v
[Report Generation]
	분석 결과 기반 리포트 생성 -> analytics(an_text)
	- RAG-for-report/create_report_with_rag_and_llm_with_gpu.py
				|
				v
[Vectorization]
	analytics(an_text) -> Chroma(Vector DB)
	- rag/ingest_db.py
				|
				v
[RAG Q&A]
	- rag/main.py (Hybrid Retriever + Rerank + MoA)
	- front/mysite/main/ai_service.py (Django 연동, Streaming)
				|
				v
[Web]
	- front/mysite (Django): 대시보드/저장/마이페이지/PDF 다운로드/AI 챗


Track B) 메타데이터 태깅/분류 모델 학습용 데이터 수집 (iOS App Store 기반)

[Crawling]
	- iOS App Store: app_store_cralwer/app_store_crawler.py
				|
				v
[Preprocess]
	- app_store_cralwer/spliter.py (CSV -> 문장 CSV)
				|
				v
[LLM Labeling]
	- app_store_cralwer/labeling.py (Gemini로 aspect/sentiment 라벨링)
	- app_store_cralwer/combine.py (결과 JSON 병합)
```

---

## 핵심 기능

### 1) 리뷰 수집 및 DB 적재

- Google Play 리뷰 수집 → MySQL 적재: `crawling/crawler.py`
	- 앱 메타데이터(장르/개발자/아이콘 등) + 버전 + 리뷰를 upsert/insert
	- `reviewId` 중복 방지(`INSERT IGNORE`)
- iOS App Store 리뷰 수집 → CSV 저장: `app_store_cralwer/app_store_crawler.py`
	- 실제 보고서 생성 파이프라인에 직접 투입하기보다는, **메타데이터 태깅/분류 모델 훈련용 데이터셋**을 만들기 위한 수집 용도

### 2) 문장 분리(전처리)

- DB의 `review`를 문장 단위(`review_line`)로 분리: `RAG-for-report/kssds_line_splitter.py`
	- 반복 문자 정규화 후(Ko soynlp), KSSDS로 문장 분리
- iOS CSV 리뷰를 문장 단위 CSV로 분리: `app_store_cralwer/spliter.py` (훈련 데이터 전처리)

### 3) ABSA(Aspect-Based Sentiment Analysis) 파인튜닝/추론

- 멀티태스크(Aspect 7클래스 + Sentiment 3클래스) 분류 헤드
- 클래스 불균형 대응을 위한 **Weighted Loss** 적용
- (데이터 구축) iOS App Store 리뷰를 수집/문장 분리 후 Gemini로 1차 라벨링하여 학습 데이터로 활용: `app_store_cralwer/labeling.py`
- 학습: `fine_tuning/model2.py` (base: `beomi/KcELECTRA-base-v2022`)
- 추론/DB 적재: `fine_tuning/predict2.py` → `analysis` 테이블에 결과 저장

### 4) 리포트 생성(L1~L4) 및 저장

- 분석된 문장/리뷰를 기반으로 LLM으로 보고서 생성 후 `analytics` 테이블에 저장
	- 예: 장르/버전 단위로 대상 버전 선정 → 보고서 생성 → DB 저장
	- 스크립트: `RAG-for-report/create_report_with_rag_and_llm_with_gpu.py`

### 5) RAG 질의응답(검색 + 생성)

- **Hybrid 검색**: Vector Search + BM25
- **Cross-Encoder Rerank**로 최종 컨텍스트 품질 개선
- (실험) **MoA(Mixture of Agents)**로 초안 생성 후 Judge LLM이 최종 종합
	- 콘솔 챗봇: `rag/main.py`
	- DB→벡터DB 적재: `rag/ingest_db.py`

### 6) Django 대시보드(제품화)

- 메인 대시보드(로그인 기반): `front/mysite/main/views.py`
	- 사용자 권한(회사 이메일 도메인 기반) 앱만 노출
	- 리포트 마크다운을 HTML 렌더링, 저장 여부 표시
	- 리포트 PDF 다운로드(`pdfkit`) 지원
	- RAG 챗 API 제공(Streaming)
- 마이페이지: `front/mysite/mypage/views.py` (프로필/비밀번호/탈퇴, 저장 리포트 목록)

---

## 기술 스택

- **Backend/Web**: Django, Daphne(ASGI), MySQL
- **Crawling**: google-play-scraper, requests/pandas
- **NLP/ML**: PyTorch, HuggingFace Transformers, scikit-learn
- **RAG**: LangChain, ChromaDB
- **Embedding/Rerank**: BGE-M3, bge-reranker 계열 / CrossEncoder
- **LLM API**: Gemini, Groq (모듈별로 사용)

---

## 빠른 실행(로컬)

> 이 레포는 실험 모듈이 여러 개라 “한 번에 전부”보다, **파이프라인 단위로** 실행하는 방식이 안정적입니다.

### 0) MySQL 스키마 생성

- 스키마/테이블 생성 SQL은 `crawling/README.md`에 정리돼 있습니다.

### 1) 환경 변수(.env)

루트(또는 각 모듈 디렉터리)에 `.env`를 만들고 최소 아래 값들을 설정합니다.

```env
# 공통 DB
host=localhost
port=3306
user=...
passwd=...
dbname=kiwi

# LLM API (모듈별로 사용)
GEMINI_API_KEY=...
GOOGLE_API_KEY=...   # (app_store_cralwer/labeling.py)
GROQ_API_KEY=...

# Django
SECRET_KEY=...       # front/mysite/mysite/settings.py
DB_ENGINE=django.db.backends.mysql
DB_NAME=kiwi
DB_USER=...
DB_PASSWORD=...
DB_HOST=localhost
DB_PORT=3306
```

### 2) (선택) Google Play 리뷰 수집 → DB 적재

```bash
python crawling/crawler.py
```

### 3) (선택) 리뷰 문장 분리 → review_line 생성

```bash
python RAG-for-report/kssds_line_splitter.py
```

### 4) (선택) ABSA 모델 학습 및 분석 결과 적재

```bash
python fine_tuning/model2.py
python fine_tuning/predict2.py
```

### 5) (선택) 보고서 생성 → analytics 저장

```bash
python RAG-for-report/create_report_with_rag_and_llm_with_gpu.py
```

### 6) (선택) analytics 벡터화(ChromaDB)

```bash
python rag/ingest_db.py
```

### 7) Django 웹 실행

```bash
cd front/mysite
python manage.py runserver
```

---

## 실행/환경 주의사항

- **Vector DB 경로(PERSIST_DIRECTORY)**: 모듈별로 벡터DB 경로가 다를 수 있습니다. 실행 전 각 스크립트/서비스의 상단 설정을 확인하세요.
- **PDF 다운로드**: `front/mysite/main/views.py`는 `pdfkit`을 사용합니다. 로컬에서 PDF 생성이 필요하면 `wkhtmltopdf` 설치가 추가로 필요할 수 있습니다.
- **RAG-for-report 스크립트**: GPU/안정성 최적화 실험이 포함돼 있어 의존성이 많습니다. 자세한 설치/실행은 `RAG-for-report/README.md`를 기준으로 맞추는 것을 권장합니다.

---

## 폴더 안내

- `crawling/`: Google Play 리뷰 수집 및 MySQL 스키마 문서
- `app_store_cralwer/`: iOS App Store 리뷰 수집/문장분리/LLM 라벨링 파이프라인(ABSA 데이터 생성용)
- `fine_tuning/`: KcELECTRA 기반 ABSA 파인튜닝/추론
- `RAG-for-report/`: 리뷰 문장 벡터화 및 LLM 기반 리포트 생성(고성능/실험 스크립트 포함)
- `rag/`: analytics(보고서) 기반 고급 RAG(Q&A) 구현
- `front/`: Django 웹(대시보드/마이페이지/AI 챗) + HTML 템플릿

---

## 포트폴리오 관점에서의 포인트

- “데이터 수집 → 정제 → 모델링 → 리포트 생성 → 검색/생성 → 웹 제품화”로 끝까지 연결
- 불균형 데이터(Aspect/Sentiment) 대응을 위해 **Weighted Loss** 적용
- RAG에서 **Hybrid Retriever + Reranking + (실험) MoA**로 답변 품질을 단계적으로 개선
- 보고서를 Markdown으로 저장하고, 웹에서 HTML/PDF로 변환해 사용자 경험까지 고려

---

## 성과(정성/정량)

- iOS App Store 기반 LLM 라벨링 파이프라인(Gemini)으로 분류 모델 학습 데이터를 구축했고, 샘플 검수 기준 **체감 정확도 90%+**를 확인

---