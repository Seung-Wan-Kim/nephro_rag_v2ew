import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 문서 기반 응답을 위한 기본 설정
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def load_vector_db(vector_path):
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    return FAISS.load_local(vector_path, embedding_model)

def get_vector_path_from_question(question):
    question = question.lower()
    if "aki" in question or "급성" in question:
        return "vector_store_aki_md_ko"
    elif "ckd" in question or "만성" in question:
        return "vector_store_ckd_md_ko"
    elif "신증후군" in question or "nephrotic" in question:
        return "vector_store_ns_md_ko"
    elif "사구체" in question or "glomerulo" in question:
        return "vector_store_gn_md_ko"
    elif "전해질" in question or "electrolyte" in question:
        return "vector_store_electrolyte_md_ko"
    return None

st.set_page_config(page_title="Nephrology RAG System", layout="wide")
st.title("🔍 신장내과 질환 진단 지원 시스템")

tab1, tab2 = st.tabs(["🧠 문서 기반 질의응답", "🧪 혈액검사 수치 기반 분석"])

with tab1:
    question = st.text_input("질문을 입력하세요", "")
    if question:
        vector_path = get_vector_path_from_question(question)
        if not vector_path or not os.path.exists(vector_path):
            st.error(f"❌ 해당 질병군에 대한 벡터 데이터가 존재하지 않습니다: {vector_path}")
        else:
            with st.spinner("🔍 문서 검색 중..."):
                try:
                    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
                    db = FAISS.load_local(vector_path, embedding_model)
                    retriever = db.as_retriever()
                    docs = retriever.get_relevant_documents(question)
                    if not docs:
                        st.warning("📄 관련 문서를 찾을 수 없습니다.")
                    else:
                        for i, doc in enumerate(docs):
                            source = doc.metadata.get("source", "알 수 없음")
                            st.markdown(f"**{i+1}.** `{source}`

{doc.page_content}")
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")

with tab2:
    st.markdown("🚧 **수치 기반 분석 기능은 현재 개발 중입니다. 향후 업데이트 예정입니다.**")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    input_fields = [
        "Creatinine", "eGFR", "BUN", "Albumin", "Proteinuria",
        "Na", "K", "Cl", "CO2", "Ca",
        "IP", "Hb", "PTH", "Vitamin D", "ALP",
        "LDH", "Lactate", "WBC", "Platelet", "CRP"
    ]

    values = {}
    for i, field in enumerate(input_fields):
        with [col1, col2, col3, col4, col5][i % 5]:
            values[field] = st.text_input(field, "")

    st.button("진단 분석 시작", help="해당 기능은 추후 제공 예정입니다.")