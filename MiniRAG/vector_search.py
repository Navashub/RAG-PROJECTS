"""

  LESSON 3  VECTOR SEARCH  (multi-environment version)   
  Local  → Ollama  (nomic-embed-text)                     
  Render → sentence-transformers (nomic-embed-text-v1.5)  


WHAT IS VECTOR SEARCH?
  We embed every chunk and store it.
  When a question comes in, we embed it with the SAME model.
  We compare the question vector to every chunk vector.
  The chunks with the highest similarity score are returned.

THIS IS YOUR MINI VECTOR DATABASE.
  Real tools like ChromaDB or FAISS do exactly this —
  just faster and at larger scale.

NOTE ON EMBEDDING:
  This file imports embed_text from 2_embedding.py.
  That means it automatically uses:
    - Ollama (nomic-embed-text) when MINIRAG_ENV=local
    - sentence-transformers   when MINIRAG_ENV=production
  No changes needed here when switching environments.
"""

import numpy as np
from config import CONFIG, show_config

# ── Import chunking from Lesson 1 ─────────────────────────────────────────────
from importlib import import_module
_chunking = import_module("chunking")
chunk_text = _chunking.chunk_text

# ── Import embedding from Lesson 2 (environment-aware) ───────────────────────
_embedding = import_module("embedding")
embed_text         = _embedding.embed_text
cosine_similarity  = _embedding.cosine_similarity


# ── Core functions ────────────────────────────────────────────────────────────

def build_vector_store(chunks: list[str]) -> list[dict]:
    """
    Embed every chunk and store it with its text.
    This is your in-memory vector database.

    Returns a list of dicts: [{ "text": str, "vector": np.ndarray }, ...]
    """
    print(f"  Building vector store from {len(chunks)} chunks...")
    store = []
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk)           # ← uses Ollama or sentence-transformers
        store.append({"text": chunk, "vector": vector})
        print(f"    [{i+1}/{len(chunks)}] embedded ✓")
    return store


def search(query: str, store: list[dict], top_k: int = 3) -> list[dict]:
    """
    Find the most relevant chunks for a given query.

    Args:
        query:  The user's question as a string.
        store:  The vector store built by build_vector_store().
        top_k:  How many top results to return.

    Returns:
        A sorted list of dicts: [{ "score": float, "text": str }, ...]
    """
    # Embed the question with the SAME model used for the chunks
    query_vector = embed_text(query)

    # Score every chunk against the query
    results = []
    for item in store:
        score = cosine_similarity(query_vector, item["vector"])
        results.append({"score": score, "text": item["text"]})

    # Best match first
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  LESSON 3: VECTOR SEARCH")
    print("=" * 60)
    show_config()

    with open("document.txt", "r") as f:
        document = f.read()

    print("[1] Chunking the document...")
    chunks = chunk_text(document)
    print(f"    → {len(chunks)} chunks\n")

    print("[2] Building the vector store...")
    store = build_vector_store(chunks)
    print(f"    → {len(store)} vectors stored\n")

    questions = [
        "What are the two phases of RAG?",
        "How does cosine similarity work?",
        "Why is chunk size important?",
    ]

    for question in questions:
        print("=" * 60)
        print(f" Question: {question}")
        print("=" * 60)

        results = search(question, store, top_k=3)

        for rank, result in enumerate(results, start=1):
            print(f"\n  #{rank}  Score: {result['score']:.4f}")
            print(f"  {result['text'][:200]}...")

        print()

    print(f"""
   1. embed_text() here uses: {'Ollama' if CONFIG['env'] == 'local' else 'sentence-transformers'}
      — same function, different backend depending on environment
   2. We score every chunk against the query vector
   3. top_k controls how many chunks go into the final prompt
   4. Real vector DBs (ChromaDB, FAISS, Pinecone) do this same
      math — just much faster with indexing tricks
""")