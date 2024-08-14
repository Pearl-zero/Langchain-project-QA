# embedding_saver.py
from langchain_upstage import UpstageEmbeddings
from langchain.vectorstores import Chroma
from text_splitter import TextSplitter

class EmbeddingSaver(TextSplitter):
    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        self.embeddings = None

    def create_embeddings(self):
        # Upstage 임베딩 생성
        self.embeddings = UpstageEmbeddings(
            api_key=self.api_key,
            model="solar-embedding-1-large"
        )

    def save_embeddings(self, docs):
        # 텍스트 목록 정의
        texts = self.txt_split(docs)

        # 임베딩하여 벡터 DB에 저장
        try:
            vectordb = Chroma.from_documents(
                texts,
                self.embeddings,
                persist_directory=self.CHROMA_PERSIST_DIR,
                collection_name=self.CHROMA_COLLECTION_NAME,
            )
            print('임베딩 저장 성공')
        except Exception as e:
            print(f"Error occurred: {e}")