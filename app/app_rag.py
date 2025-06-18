import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# 수치 기반 분석용 라이브러리
import pandas as pd

# ---------------------------
# 벡터 로딩 함수
# ---------------------------
def load_vector_db(vector_path):
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
    return FAISS.load_local(vector_path, embedding_model, allow_dangerous_deserialization=True)

# ---------------------------
# 질병군별 경로 판단
# ---------------------------
def get_vector_path_from_question(question):
    if "급성" in question or "AKI" in question:
        return "vector_store_aki_md_ko/"
    elif "만성" in question or "CKD" in question:
        return "vector_store_ckd_md_ko/"
    elif "신증후군" in question or "단백뇨" in question:
        return "vector_store_ns_md_ko/"
    elif "사구체신염" in question or "Glomerulonephritis" in question:
        return "vector_store_gn_md_ko/"
    elif "전해질" in question or "Electrolyte" in question:
        return "vector_store_electrolyte_md_ko/"
    else:
        return None

# ---------------------------
# 수치 기반 분석 함수
# ---------------------------
def analyze_lab_values(inputs):
    messages = []
    recommendations = []
    score = 0

    if inputs["Creatinine"] > 1.2:
        messages.append("크레아티닌 수치가 높습니다. AKI 가능성을 고려하세요.")
        score += 20
    if inputs["eGFR"] < 60:
        messages.append("eGFR이 낮습니다. CKD 가능성이 있습니다.")
        score += 20
    if inputs["Albumin"] < 3.0:
        messages.append("알부민이 낮습니다. Nephrotic Syndrome을 고려해보세요.")
        score += 10
    if inputs["Proteinuria"] > 1.0:
        messages.append("단백뇨 수치가 높습니다. 사구체 질환 가능성이 있습니다.")
        score += 10
    if inputs["BUN"] > 20:
        messages.append("BUN 수치가 상승했습니다. 신기능 저하를 의심할 수 있습니다.")
        score += 10
    if inputs["Na"] < 135 or inputs["Na"] > 145:
        messages.append("나트륨 수치 이상. 전해질 이상 가능성 있음.")
        score += 10
    if inputs["K"] < 3.5 or inputs["K"] > 5.0:
        messages.append("칼륨 수치 이상. 전해질 이상 가능성 있음.")
        score += 10
    if inputs["Ca"] < 8.5 or inputs["Ca"] > 10.5:
        messages.append("칼슘 수치 이상. 전해질 이상 가능성 있음.")
        score += 5
    if inputs["Phosphorus"] > 4.5:
        messages.append("인 수치가 높습니다. CKD-MBD 가능성 있음.")
        score += 5

    if score == 0:
        recommendations.append("더 많은 혈액검사 항목을 입력하시면 정확도가 향상됩니다.")

    return messages, recommendations, score

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🩺 신장내과 문서 기반 질의응답 + 수치 기반 분석")

# 수치 입력창
st.header("🧪 혈액검사 수치 입력")
default_vals = {
    "Creatinine": 1.0, "eGFR": 90, "Albumin": 4.0, "Proteinuria": 0.0,
    "BUN": 15, "Na": 140, "K": 4.0, "Cl": 100, "CO2": 24,
    "Ca": 9.5, "Phosphorus": 3.5, "Hb": 13, "PTH": 40, "VitaminD": 30,
    "ALP": 70, "LDH": 180
}
user_inputs = {}

cols = st.columns(4)
keys = list(default_vals.keys())
for i, key in enumerate(keys):
    with cols[i % 4]:
        user_inputs[key] = st.number_input(f"{key}", value=default_vals[key])

if st.button("🔍 수치 기반 분석 실행"):
    messages, recs, score = analyze_lab_values(user_inputs)
    st.subheader("🔎 분석 결과")
    for msg in messages:
        st.info(msg)
    for rec in recs:
        st.warning(rec)
    st.success(f"예상 유사도 점수: {score}점 (100점 만점 기준)")

# 자연어 질문
st.header("💬 질의응답 (RAG)")
question = st.text_input("질문을 입력하세요", placeholder="예: 급성 신손상의 정의는?")

if st.button("📚 문서 기반 응답 생성"):
    vector_path = get_vector_path_from_question(question)
    if vector_path is None or not os.path.exists(vector_path):
        st.error(f"❌ 해당 질병군에 대한 벡터 데이터가 존재하지 않습니다: {vector_path}")
    else:
        st.info("⛔ 현재 API 키 없이 LLM 호출이 비활성화되어 있습니다. 향후 OpenAI 등 연동 필요")
