import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document

# 질병군별 벡터 DB 경로 자동 선택 함수
def get_vector_path_from_question(question: str) -> str:
    question = question.lower()
    if "aki" in question or "급성" in question or "신손상" in question:
        return "vector_store_aki_md_ko/"
    elif "ckd" in question or "만성" in question or "콩팥병" in question:
        return "vector_store_ckd_md_ko/"
    elif "nephrotic" in question or "신증후군" in question:
        return "vector_store_ns_md_ko/"
    elif "glomerulo" in question or "사구체" in question:
        return "vector_store_gn_md_ko/"
    elif "electrolyte" in question or "전해질" in question:
        return "vector_store_electrolyte_md_ko/"
    else:
        return ""

# 벡터 DB 로드
def load_vector_db(vector_path):
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    return FAISS.load_local(vector_path, embedding_model)

# Streamlit 앱 시작
st.set_page_config(layout="wide")
st.title("🧠 신장내과 질병군 문서 기반 질의응답 시스템")

# 사용자 입력
question = st.text_input("질문을 입력하세요 (예: 급성신손상이란?)")

if question:
    vector_path = get_vector_path_from_question(question)

    if not vector_path or not os.path.exists(vector_path):
        st.error(f"❌ 해당 질병군에 대한 벡터 데이터가 존재하지 않습니다: {vector_path}")
    else:
        db = load_vector_db(vector_path)
        docs = db.similarity_search(question, k=3)

        if docs:
            st.subheader("🔍 검색된 문서 기반 응답")
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "출처 없음")
                content = doc.page_content.strip()
                if content:
                    st.markdown(f"**{i+1}.** `{source}`

{content}")
                else:
                    st.warning("⚠️ 문서에서 관련된 정보를 찾을 수 없습니다.")
        else:
            st.warning("⚠️ 관련 문서를 찾을 수 없습니다.")
