import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

load_dotenv()

VECTOR_DIR = "vectorstore"


# ----------------------------
# DOCUMENT LOADER (local, safe)
# ----------------------------
def load_documents(data_dir="data"):
    docs = []

    for file in Path(data_dir).glob("*"):
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        elif file.suffix == ".docx":
            loader = Docx2txtLoader(str(file))
        else:
            continue

        docs.extend(loader.load())

    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


# ----------------------------
# HYBRID RETRIEVER
# ----------------------------
def create_hybrid_retriever(vectorstore, chunks):
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )


# ----------------------------
# RERANKER
# ----------------------------
def apply_reranking(retriever):
    compressor = FlashrankRerank(top_n=4)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )


# ----------------------------
# LOAD CHAIN
# ----------------------------
def load_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # load docs for BM25
    docs = load_documents("data")
    chunks = split_documents(docs)

    # hybrid retrieval
    retriever = create_hybrid_retriever(db, chunks)

    # reranking (SAFE WRAP)
    retriever = apply_reranking(retriever)

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    return chain


# ----------------------------
# ASK FUNCTION
# ----------------------------
def ask(chain, question):
    res = chain.invoke({"question": question})

    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in res["source_documents"]
    })

    return {
        "answer": res["answer"],
        "sources": sources
    }