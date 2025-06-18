
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# 경로 설정
doc_folder = "docx_ko/aki"
vector_store_path = "vector_store_aki_ko"

# LangChain-compatible Ko-SBERT 임베딩 모델 래퍼
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

# 문서 수집 및 Document 객체로 변환
documents = []
for filename in os.listdir(doc_folder):
    if filename.endswith(".md") or filename.endswith(".txt"):
        filepath = os.path.join(doc_folder, filename)
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
            documents.append(Document(page_content=content, metadata={"source": filename}))

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
split_docs = text_splitter.split_documents(documents)

# 벡터 저장소 생성 및 저장
db = FAISS.from_documents(split_docs, embedding_model)
db.save_local(vector_store_path)

print(f"✅ 벡터 저장소 생성 완료: {vector_store_path}")
