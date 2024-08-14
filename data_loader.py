# data_loader.py
import os
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, NotebookLoader
from langchain_community.document_loaders import DirectoryLoader

class DataLoader:
    def __init__(self):
        self.CUR_DIR = None
        self.CHROMA_PERSIST_DIR = None
        self.CHROMA_COLLECTION_NAME = None

    def set_dir_path(self, dir, file_name, chroma_name):
        self.CUR_DIR = dir
        self.CHROMA_PERSIST_DIR = os.path.join(self.CUR_DIR, file_name)
        self.CHROMA_COLLECTION_NAME = chroma_name

    def load_data(self):
        LOADER_DICT = {
            "pdf": PyPDFLoader,
            "txt": TextLoader,
            "md": UnstructuredMarkdownLoader,
            "ipynb": NotebookLoader
        }

        if not self.CUR_DIR:
            raise ValueError("Directory path is not set")

        loader = LOADER_DICT.get('pdf')
        if loader is None:
            raise ValueError("Not supported file type")

        directory_loader = DirectoryLoader(self.CUR_DIR, glob="*.pdf", loader_cls=loader)
        docs = directory_loader.load()

        return docs