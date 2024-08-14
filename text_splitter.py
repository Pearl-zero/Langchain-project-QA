# text_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from data_loader import DataLoader

class TextSplitter(DataLoader):
    def __init__(self):
        super().__init__()

    def txt_split(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)

        return texts