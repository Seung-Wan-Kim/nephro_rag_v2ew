# 🧠 Nephrology RAG System (신장내과 질병군 문서 기반 질의응답 시스템)

본 프로젝트는 다음과 같은 기능을 포함합니다:

## ✅ 주요 기능
1. **자연어 기반 질의응답 (RAG)**
   - AKI, CKD, Nephrotic Syndrome, Glomerulonephritis, Electrolyte Disorders에 대한 질의 가능
   - Ko-SBERT 기반 문서 임베딩, FAISS 검색 사용
   - 참조 문서 출처 표시 기능 포함

2. **수치 기반 질병 유사도 분석**
   - Creatinine, eGFR, 단백뇨, Albumin 수치를 입력 받아 질병군 판단
   - 중증도 간략 진단 결과 출력

## 🗂 디렉토리 구조
- `app/`: Streamlit 실행 파일
- `vector_store_xxx/`: 질병군별 FAISS 벡터 저장소
- `docs_ko/`: 각 질병군의 요약 문서 (.md)
- `.streamlit/`: Streamlit 설정파일
- `create_embeddings_*.py`: 각 질병군의 md 파일 임베딩 스크립트

## 🚀 실행 방법
```bash
cd nephro_rag_v2e
streamlit run app/app_rag.py
```

## 🧪 필요 패키지
`requirements.txt` 참조