"""

 app.py - MiniRAG Web Server                             
  FastAPI web interface for the RAG pipeline              


ENDPOINTS:
  GET  /          -> health check (is the server alive?)
  GET  /status    -> shows which environment and models are active
  POST /ask       -> ask a question against the document

HOW TO RUN LOCALLY:
  uvicorn app:app --reload

HOW IT RUNS ON RENDER:
  uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

from config import CONFIG, show_config
from importlib import import_module

# Import each lesson
_chunking      = import_module("chunking")
_embedding     = import_module("embedding")
_vector_search = import_module("vector_search")
_prompt        = import_module("prompt_builder")
_pipeline      = import_module("rag_pipeline")

chunk_text        = _chunking.chunk_text
build_vector_store = _vector_search.build_vector_store
search            = _vector_search.search
build_rag_prompt  = _prompt.build_rag_prompt
generate_answer   = _pipeline.generate_answer

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MiniRAG",
    description="A simple Retrieval-Augmented Generation API",
    version="1.0.0",
)

# ── Build the vector store once at startup ────────────────────────────────────
# We do this once when the server starts so every request is fast.
# In a real system you would load this from a database.

print("\nMiniRAG starting up...")
show_config()

print("Loading document...")
with open("document.txt", "r") as f:
    document = f.read()

print("Chunking...")
CHUNKS = chunk_text(document, chunk_size=100, overlap=20)
print(f"  -> {len(CHUNKS)} chunks")

print("Building vector store (this may take a moment)...")
STORE = build_vector_store(CHUNKS)
print(f"  -> {len(STORE)} vectors ready")
print("\nMiniRAG is ready!\n")


# ── Request / Response models

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 2          # how many chunks to retrieve

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]      # the chunks that were used to answer
    environment: str
    llm_model: str


# ── Endpoints

@app.get("/")
def health_check():
    """Quick check that the server is alive."""
    return {
        "status": "ok",
        "project": "MiniRAG",
        "message": "Send a POST request to /ask with a question"
    }


@app.get("/status")
def status():
    """Show which environment and models are currently active."""
    return {
        "environment": CONFIG["env"],
        "llm_model":   CONFIG["llm_model"],
        "embed_model": CONFIG["embed_model"],
        "chunks_loaded": len(STORE),
        "document_words": len(document.split()),
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    Ask a question against the loaded document.

    Body:
        {
            "question": "What are the two phases of RAG?",
            "top_k": 2
        }
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Retrieve relevant chunks
    results    = search(request.question, STORE, top_k=request.top_k)
    top_chunks = [(r["score"], r["text"]) for r in results]

    # Build prompt and generate answer
    prompt = build_rag_prompt(request.question, top_chunks)
    answer = generate_answer(prompt)

    return AnswerResponse(
        question    = request.question,
        answer      = answer,
        sources     = [r["text"] for r in results],
        environment = CONFIG["env"],
        llm_model   = CONFIG["llm_model"],
    )




if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)