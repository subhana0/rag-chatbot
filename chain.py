import os
import re
import time
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

from better_profanity import profanity

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

VECTOR_DIR = "vectorstore"


# =========================================================
# DOCUMENT LOADER
# =========================================================
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


# =========================================================
# PHASE 7: RETRIEVAL
# =========================================================
def create_hybrid_retriever(vectorstore, chunks):
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )


def apply_reranking(retriever):
    compressor = FlashrankRerank(top_n=4)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )


# =========================================================
# PHASE 8: GUARDRAILS
# =========================================================

class RateLimiter:
    def __init__(self, max_requests=20, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, user_id="default"):
        now = time.time()

        self.requests[user_id] = [
            t for t in self.requests[user_id]
            if now - t < self.window
        ]

        if len(self.requests[user_id]) >= self.max_requests:
            return False, "Rate limit exceeded. Try again later."

        self.requests[user_id].append(now)
        return True, ""


class InputGuardrails:
    MAX_LEN = 1000
    MIN_LEN = 2

    PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above|system)\s+instructions",
        r"you\s+are\s+now",
        r"jailbreak",
        r"system\s*:",
        r"disregard",
    ]

    def __init__(self):
        profanity.load_censor_words()
        self.compiled = [re.compile(p, re.I) for p in self.PATTERNS]

    def validate(self, text):
        text = text.strip()

        if len(text) < self.MIN_LEN:
            return False, "Query too short"

        if len(text) > self.MAX_LEN:
            return False, "Query too long"

        if profanity.contains_profanity(text):
            return False, "Inappropriate language detected"

        for p in self.compiled:
            if p.search(text):
                return False, "Prompt injection detected"

        return True, text


class TopicGuardrail:
    BLOCK_PATTERNS = [
        r"write.*(poem|story|joke|song)",
        r"capital of",
        r"how to hack",
        r"make.*weapon",
    ]

    def __init__(self):
        self.compiled = [re.compile(p, re.I) for p in self.BLOCK_PATTERNS]

    def check(self, text):
        for p in self.compiled:
            if p.search(text):
                return False, "Out of scope question"
        return True, ""


class OutputGuardrails:
    LEAK_PATTERNS = [
        r"as an ai",
        r"system prompt",
        r"instructions are",
    ]

    def __init__(self):
        self.compiled = [re.compile(p, re.I) for p in self.LEAK_PATTERNS]
        profanity.load_censor_words()

    def validate(self, text):
        for p in self.compiled:
            if p.search(text):
                return "I can only answer based on the provided documents."

        return profanity.censor(text)


# =========================================================
# INIT GUARDRAILS
# =========================================================
rate_limiter = RateLimiter()
input_guard = InputGuardrails()
topic_guard = TopicGuardrail()
output_guard = OutputGuardrails()


# =========================================================
# LOAD CHAIN
# =========================================================
def load_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # BM25 data
    docs = load_documents("data")
    chunks = split_documents(docs)

    # Hybrid retrieval
    retriever = create_hybrid_retriever(db, chunks)

    # Reranking
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


# =========================================================
# ASK FUNCTION (PHASE 8 PROTECTED)
# =========================================================
def ask(chain, question, user_id="default"):

    # 1. Rate limit
    allowed, msg = rate_limiter.is_allowed(user_id)
    if not allowed:
        return {"answer": msg, "sources": [], "blocked": True}

    # 2. Input check
    ok, cleaned = input_guard.validate(question)
    if not ok:
        return {"answer": cleaned, "sources": [], "blocked": True}

    # 3. Topic check
    ok, msg = topic_guard.check(cleaned)
    if not ok:
        return {"answer": msg, "sources": [], "blocked": True}

    # 4. RAG
    res = chain.invoke({"question": cleaned})

    # 5. Sources
    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in res["source_documents"]
    })

    # 6. Output filter
    final_answer = output_guard.validate(res["answer"])

    return {
        "answer": final_answer,
        "sources": sources,
        "blocked": False
    }