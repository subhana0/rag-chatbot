import os
import sys
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_DIR = "vectorstore"


def load_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    return loader.load()


def add_document(file_path: str):
    print(f"\n📄 Loading file: {file_path}")

    documents = load_file(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"🔹 Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("🔹 Loading existing FAISS index...")

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("🔹 Adding new documents to index...")

    vectorstore.add_documents(chunks)

    vectorstore.save_local(VECTORSTORE_DIR)

    print(f"✅ Successfully added {len(chunks)} chunks from {file_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python ingest_incremental.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print("❌ File not found")
        sys.exit(1)

    add_document(file_path)