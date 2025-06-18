
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# 환경 변수 로드 (.env 파일 사용 시)
load_dotenv()

# Ko-SBERT 임베딩 모델
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

# 벡터 경로 결정 함수
def get_vector_path_from_question(question):
    question = question.lower()
    if "급성" in question or "aki" in question:
        return "vector_store_aki_md_ko/"
    elif "만성" in question or "ckd" in question:
        return "vector_store_ckd_md_ko/"
    elif "신증후군" in question or "nephrotic" in question:
        return "vector_store_ns_md_ko/"
    elif "사구체" in question or "glomerulo" in question:
        return "vector_store_gn_md_ko/"
    elif "전해질" in question or "electrolyte" in question:
        return "vector_store_electrolyte_md_ko/"
    else:
        return None

# 벡터 DB 로드
def load_vector_db(vector_path):
    return FAISS.load_local(vector_path, embedding_model)

# Streamlit UI
st.set_page_config(page_title="Nephrology RAG System", layout="wide")
st.title("🧠 신장내과 질환 문서기반 질의응답 시스템")

tab1, tab2 = st.tabs(["🔍 문서 기반 질의", "🧪 혈액검사 수치 기반 분석"])

with tab1:
    user_question = st.text_input("질문을 입력하세요:", key="question_input")

    if user_question:
        vector_path = get_vector_path_from_question(user_question)
        if not vector_path or not os.path.exists(vector_path):
            st.error(f"❌ 해당 질병군에 대한 벡터 데이터가 존재하지 않습니다: {vector_path}")
        else:
            db = load_vector_db(vector_path)
            docs = db.similarity_search(user_question, k=3)
            st.markdown("### 📄 검색 결과")
            if docs:
                for i, doc in enumerate(docs):
                    source = doc.metadata.get("source", "출처 없음")
        st.markdown(f"**{i+1}.** `{source}`")

        st.markdown(f"**{i+1}.** `{source}`\n\n{doc.page_content}")
            else:
                st.warning("⚠️ 문서에서 관련된 정보를 찾을 수 없습니다.")

with tab2:
    st.markdown("### 🧪 혈액검사 수치를 입력하세요:")

    col1, col2, col3, col4 = st.columns(4)
    fields = {
        "Creatinine": col1,
        "eGFR": col2,
        "Albumin": col3,
        "Proteinuria": col4,
        "BUN": col1,
        "Na": col2,
        "K": col3,
        "Cl": col4,
        "CO2": col1,
        "Ca": col2,
        "Phosphorus": col3,
        "Hb": col4,
        "PTH": col1,
        "VitaminD": col2,
        "ALP": col3,
        "LDH": col4
    }

    input_values = {}
    for test, col in fields.items():
        val = col.text_input(test, key=f"input_{test}")
        input_values[test] = val

    st.markdown("#### 결과 분석")

    if st.button("🔍 수치 기반 분석 실행"):
        valid_inputs = {k: float(v) for k, v in input_values.items() if v.strip() != ""}
        if not valid_inputs:
            st.warning("❗ 수치를 하나 이상 입력해 주세요.")
        else:
            score = 0
            if float(valid_inputs.get("Creatinine", 0)) > 1.5:
                score += 1
            if float(valid_inputs.get("eGFR", 100)) < 60:
                score += 1
            if float(valid_inputs.get("Albumin", 5.0)) < 3.5:
                score += 1
            if float(valid_inputs.get("Proteinuria", 0)) > 1:
                score += 1

            if score == 0:
                st.info("⚖️ 현재 입력된 수치로는 명확한 질병 관련성은 낮습니다.")
            elif score == 1:
                st.info("🔎 일부 수치에서 이상이 있습니다. 추가 검사가 필요할 수 있습니다.")
            else:
                st.error("🚨 신장 질환이 의심됩니다. 추가적인 정밀 진단이 필요합니다.")

# 문서 출력
if docs_and_sources:
    for i, (doc, source) in enumerate(docs_and_sources):
        st.markdown(f"**{i+1}.** `{source}`")
        if doc.page_content.strip():
            st.write(doc.page_content)
        else:
            st.write('해당 문서에서 답변을 찾을 수 없습니다.')
else:
    st.warning("🔍 문서에서 관련 정보를 찾을 수 없습니다.")
