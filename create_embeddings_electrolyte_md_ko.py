# create_embeddings_electrolyte_md_ko.py

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

import os

# 임베딩 모델 불러오기
embedding_model = SentenceTransformer("jhgan/ko-sbert-nli")

# 문서 경로
file_path = "docs_ko/electrolyte/electrolyte_summary.md"

# 문서 로딩
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()

# 문서 청크 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
split_docs = text_splitter.split_documents(documents)

# 벡터 저장소 생성
db = FAISS.from_texts([doc.page_content for doc in split_docs], embedding_model)

# 디렉토리 생성 및 저장
vector_path = "vector_store_electrolyte_md_ko"
os.makedirs(vector_path, exist_ok=True)
db.save_local(vector_path)

print("✅ Electrolyte Disorders 임베딩 완료!")
