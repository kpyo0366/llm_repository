import os
import shutil
import sqlite3
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.chat_models import ChatOllama

# ✅ 1️⃣ PDF 파일 경로 설정
file_path = "절대경로"  # 절대 경로로 변경
## 모빌리티 관련 대한민국 정부문서를 활용하였습니다.
if not os.path.exists(file_path):
    raise FileNotFoundError(f"🚨 PDF 파일이 존재하지 않습니다: {file_path}")

# ✅ 2️⃣ PDF 문서 로드
loader = PyPDFLoader(file_path)
pages = loader.load()
if not pages:
    raise ValueError("🚨 PDF 문서에서 내용을 불러오지 못했습니다!")

# ✅ 3️⃣ 문서 분할 (텍스트 청크 단위)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)
docs = text_splitter.split_documents(pages)

# ✅ 4️⃣ ChromaDB 저장 경로 설정 및 기존 데이터 삭제
chroma_persist_directory = os.path.abspath("./chroma_db")
shutil.rmtree(chroma_persist_directory, ignore_errors=True)  # 기존 DB 삭제
os.makedirs(chroma_persist_directory, exist_ok=True)  # 새 폴더 생성
os.chmod(chroma_persist_directory, 0o777)  # 권한 설정

# ✅ 5️⃣ SQLite DB 파일이 생성되었는지 확인 후 WAL 모드 활성화
db_path = os.path.join(chroma_persist_directory, "chroma.sqlite")
if os.path.exists(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")

# ✅ 6️⃣ 임베딩 모델 로드
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ 7️⃣ ChromaDB 벡터 저장소 생성 (`allow_reset` 제거)
vectorstore = Chroma.from_documents(
    docs,
    embedding=embedding_model,
    persist_directory=chroma_persist_directory,
)

print("✅ 벡터 저장소 생성 완료! (ChromaDB 사용)")

# ✅ 8️⃣ 검색 기능 설정 (유사도 높은 5개 문서 검색)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ✅ 9️⃣ Ollama 모델 로드
llm = ChatOllama(model="gemma2:2b", temperature=0)

# ✅ 🔟 RAG 프롬프트 설정
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''
prompt = ChatPromptTemplate.from_template(template)

# ✅ 1️⃣1️⃣ 문서 포맷팅 함수
def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# ✅ 1️⃣2️⃣ RAG Chain 연결
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ✅ 1️⃣3️⃣ 질문 실행
query = "모빌리티 자동차국 책임자는 누구인가요?"
response = rag_chain.invoke(query)

# ✅ 1️⃣4️⃣ 결과 출력
print("🔹 검색된 답변:")
print(response)
