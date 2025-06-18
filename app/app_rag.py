import streamlit as st
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import os

# -----------------------------
# 임베딩 모델
embedding_model = SentenceTransformer("jhgan/ko-sbert-nli")

# -----------------------------
# 벡터 경로 판단
def get_vector_path_from_question(question: str) -> str:
    if "급성" in question or "AKI" in question:
        return "vector_store_aki_md_ko"
    elif "만성" in question or "CKD" in question:
        return "vector_store_ckd_md_ko"
    elif "신증후군" in question or "Nephrotic" in question:
        return "vector_store_ns_md_ko"
    elif "사구체" in question or "Glomerulonephritis" in question:
        return "vector_store_gn_md_ko"
    elif "전해질" in question or "Electrolyte" in question:
        return "vector_store_electrolyte_md_ko"
    return ""

def load_vector_db(vector_path: str):
    if not os.path.exists(vector_path):
        return None
    return FAISS.load_local(vector_path, embedding_model)

# -----------------------------
# 수치 기반 분석
def analyze_values(creatinine, eGFR, proteinuria, albumin):
    results = []
    if creatinine and creatinine > 1.5:
        results.append("🔴 **AKI 가능성**: 크레아티닌 상승")
    if eGFR and eGFR < 60:
        results.append("🟠 **CKD 가능성**: eGFR 감소")
    if albumin and albumin < 3.0 and proteinuria and proteinuria >= 3.5:
        results.append("🟡 **Nephrotic Syndrome 가능성**: 심한 단백뇨 + 저알부민혈증")
    if not results:
        results.append("✅ 특별한 이상 소견 없음 (입력된 값 기준)")
    return results

# -----------------------------
# Streamlit UI
st.title("🧠 신장내과 RAG 시스템 (자연어 + 수치 기반)")

tab1, tab2 = st.tabs(["💬 자연어 질의", "🧪 수치 기반 분석"])

# -----------------------------
with tab1:
    question = st.text_input("질문을 입력하세요:", "")
    if st.button("질문하기"):
        vector_path = get_vector_path_from_question(question)
        if not vector_path:
            st.error("❌ 질병군을 자동 판단할 수 없습니다.")
        else:
            db = load_vector_db(vector_path)
            if db is None:
                st.error(f"❌ 벡터 DB를 찾을 수 없습니다: {vector_path}")
            else:
                retrieved_docs = db.similarity_search(question, k=3)
                combined_content = "\n".join([doc.page_content for doc in retrieved_docs])
                st.markdown("### 🧠 응답")
                st.write(combined_content)
                st.markdown("### 📚 참조 문서")
                for i, doc in enumerate(retrieved_docs, 1):
                    source = doc.metadata.get("source", "출처 미상")
                    st.markdown(f"{i}. `{source}`")

# -----------------------------
with tab2:
    st.markdown("아래 수치를 입력하시면 질병 가능성을 분석합니다.")
    creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=20.0, step=0.1)
    eGFR = st.number_input("eGFR (mL/min/1.73m²)", min_value=0.0, max_value=150.0, step=1.0)
    proteinuria = st.number_input("단백뇨 (g/day)", min_value=0.0, max_value=20.0, step=0.1)
    albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=6.0, step=0.1)

    if st.button("수치 분석하기"):
        results = analyze_values(creatinine, eGFR, proteinuria, albumin)
        st.markdown("### 🧾 분석 결과")
        for res in results:
            st.markdown(res)