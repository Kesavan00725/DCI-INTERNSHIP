import os
import glob
from typing import List, Dict, Any, Tuple

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from google import genai
from google.genai import errors as genai_errors, types

# --------------------------------------------------
# 1. Config & Gemini client
# --------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment or .env")

client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CHUNK_SIZE = 600        # characters per chunk
CHUNK_OVERLAP = 120     # overlap characters


# --------------------------------------------------
# 2. Data loading & chunking
# --------------------------------------------------
def load_documents(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load all .txt and .md files from data_dir.
    """
    docs: List[Dict[str, Any]] = []

    file_paths = glob.glob(os.path.join(data_dir, "*.txt")) + \
                 glob.glob(os.path.join(data_dir, "*.md"))

    for idx, path in enumerate(file_paths):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                text = f.read()

        docs.append(
            {
                "id": f"doc_{idx}",
                "filename": os.path.basename(path),
                "text": text,
            }
        )

    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks


# --------------------------------------------------
# 3. Embedding & in-memory vector store
# --------------------------------------------------
class ChunkRecord(BaseModel):
    id: str
    doc_id: str
    filename: str
    text: str
    embedding: List[float]


VECTOR_STORE: List[ChunkRecord] = []


def embed_text(text: str) -> List[float]:
    """
    Embed text using Gemini embeddings.
    """
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        return response.embeddings[0].values
    except genai_errors.ClientError as e:
        raise RuntimeError(f"Gemini embedding error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected embedding error: {e}")


def build_vector_store() -> None:
    """
    Load docs, chunk them, embed chunks, fill VECTOR_STORE.
    """
    global VECTOR_STORE

    print("🚀 Building vector store from 'data/'...")
    print("🔍 Loading documents from:", DATA_DIR)

    docs = load_documents(DATA_DIR)
    print(f"📄 Found {len(docs)} documents")

    if not docs:
        print("⚠ No documents found.")
        VECTOR_STORE = []
        return

    records: List[ChunkRecord] = []

    for doc in docs:
        print(f"➡ Processing file: {doc['filename']}")
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"   └─ Created {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            try:
                emb = embed_text(chunk)
            except Exception as e:
                print("❌ Embedding failed:", e)
                continue

            records.append(
                ChunkRecord(
                    id=f"{doc['id']}_chunk_{idx}",
                    doc_id=doc["id"],
                    filename=doc["filename"],
                    text=chunk,
                    embedding=emb,
                )
            )

    VECTOR_STORE = records
    print(f"✅ VECTOR_STORE size: {len(VECTOR_STORE)}")
    print("✅ Vector store ready.")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    """
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_similar_chunks(
    query: str,
    top_k: int = 3,
) -> List[Tuple[float, ChunkRecord]]:
    """
    Embed query, compute cosine similarity with all chunks,
    return top_k (similarity, ChunkRecord) pairs.
    """
    if not VECTOR_STORE:
        raise RuntimeError(
            "VECTOR_STORE is empty. Make sure 'data/' has documents and "
            "the app successfully built the index at startup."
        )

    query_emb = np.array(embed_text(query), dtype="float32")

    scored: List[Tuple[float, ChunkRecord]] = []
    for rec in VECTOR_STORE:
        emb = np.array(rec.embedding, dtype="float32")
        score = cosine_sim(query_emb, emb)
        scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# --------------------------------------------------
# 4. FastAPI schemas
# --------------------------------------------------
class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class SourcePassage(BaseModel):
    chunk_id: str
    filename: str
    similarity: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourcePassage]


# --------------------------------------------------
# 5. FastAPI app + UI route
# --------------------------------------------------
app = FastAPI(title="Gemini RAG Application", version="1.0.0")

# CORS (helpful if you later host frontend separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    build_vector_store()


@app.get("/", response_class=FileResponse)
def serve_ui():
    """
    Serve the index.html UI.
    """
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    """
    RAG endpoint:
    - Retrieve top_k similar chunks
    - Ask Gemini to answer using only those chunks
    - Return answer + source passages
    """
    question = payload.question.strip()
    top_k = max(1, payload.top_k)

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        top_scored = search_similar_chunks(question, top_k=top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not top_scored:
        raise HTTPException(
            status_code=500,
            detail="No indexed chunks available. Did the startup indexing fail?",
        )

    # Build context for Gemini
    context_parts: List[str] = []
    for rank, (score, rec) in enumerate(top_scored, start=1):
        context_parts.append(
            f"[{rank}] (similarity={score:.3f}, file={rec.filename})\n{rec.text}"
        )
    context = "\n\n".join(context_parts)

    system_instruction = (
        "You are a helpful assistant that answers questions ONLY using "
        "the provided context. If the answer is not clearly in the context, "
        "say you don't know. Do not hallucinate. Be concise and accurate."
    )

    rag_prompt = (
        f"{system_instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer in a short paragraph."
    )

    # Call Gemini
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=rag_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=256,
            ),
        )
        answer_text = (response.text or "").strip()
    except genai_errors.ClientError as e:
        if e.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exhausted or unavailable for this project. "
                    "Check your plan and billing in Google AI Studio."
                ),
            )
        raise HTTPException(
            status_code=500,
            detail=f"Gemini API error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error calling Gemini: {str(e)}",
        )

    if not answer_text:
        answer_text = "I couldn't generate an answer."

    # Build sources payload
    sources: List[SourcePassage] = []
    for score, rec in top_scored:
        sources.append(
            SourcePassage(
                chunk_id=rec.id,
                filename=rec.filename,
                similarity=float(score),
                text=rec.text,
            )
        )

    return AskResponse(answer=answer_text, sources=sources)
