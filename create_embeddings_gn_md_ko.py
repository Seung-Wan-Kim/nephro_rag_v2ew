from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# 파일 경로 설정
doc_path = "docs_ko/gn/gn_summary.md"
persist_directory = "vector_store_gn_md_ko"

# 문서 로드
loader = TextLoader(doc_path, encoding="utf-8")
documents = loader.load()

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
split_docs = text_splitter.split_documents(documents)

# Ko-SBERT 임베딩 모델 로드
embedding_model = SentenceTransformer("jhgan/ko-sbert-nli")

# FAISS 벡터 저장소 생성
db = FAISS.from_texts([doc.page_content for doc in split_docs], embedding_model)

# 벡터 저장
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
db.save_local(persist_directory)

print("Glomerulonephritis 문서 임베딩 완료!")