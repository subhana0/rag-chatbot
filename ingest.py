import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = "data"
VECTOR_DIR = "vectorstore"


def load_docs():
    docs = []
    path = Path(DATA_DIR)

    for f in path.glob("*.pdf"):
        docs.extend(PyPDFLoader(str(f)).load())

    for f in path.glob("*.txt"):
        docs.extend(TextLoader(str(f), encoding="utf-8").load())

    return docs


def build():
    docs = load_docs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)
    db.save_local(VECTOR_DIR)

    print("FAISS index created")


if __name__ == "__main__":
    build()