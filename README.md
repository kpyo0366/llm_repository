# llm_repository

LLM 기반 RAG QA 시스템 구축
이 프로젝트는 LLM (Large Language Model)과 RAG (Retrieval-Augmented Generation) 기술을 활용하여 PDF 문서에서 정보를 검색하고 질의응답을 수행하는 시스템입니다.

✅ 주요 기능:

PDF 문서 로드 및 텍스트 추출: PyPDFLoader를 사용하여 문서를 읽고 텍스트를 추출
텍스트 분할 및 벡터 임베딩: RecursiveCharacterTextSplitter로 텍스트를 청크 단위로 나누고, HuggingFaceEmbeddings를 통해 벡터화
ChromaDB 기반 검색: Chroma를 활용하여 벡터 검색을 수행하고 관련 문서를 검색
Ollama LLM을 통한 자연어 질의응답: ChatOllama 모델을 적용하여 검색된 문맥을 기반으로 질문에 대한 답변을 생성
최적화된 데이터베이스 설정: SQLite의 WAL 모드를 활성화하여 데이터 저장 성능을 향상
이 시스템을 활용하면 문서 기반의 지식 검색 및 Q&A 기능을 자동화할 수 있으며, 연구 논문, 기술 문서, 정책 보고서 등의 방대한 데이터를 효율적으로 탐색할 수 있습니다. 🚀
