from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
import pickle

# 파일 경로
doc_path = "ns_summary.md"
vector_dir = "vector_store_ns_md_ko"
os.makedirs(vector_dir, exist_ok=True)

# 문서 로드 및 분할
loader = TextLoader(doc_path, encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# Ko-SBERT 임베딩 모델
embedding_model = SentenceTransformer("jhgan/ko-sbert-nli")

# 벡터 저장소 생성
db = FAISS.from_texts([doc.page_content for doc in split_docs], embedding_model)
db.save_local(vector_dir)

# pickle 저장
with open(os.path.join(vector_dir, "index.pkl"), "wb") as f:
    pickle.dump(db, f)

print("✅ NS 임베딩 생성 완료:", vector_dir)