"""

 LESSON 3 - VECTOR SEARCH                                
 Goal: Given a question, find the most relevant chunks   


WHAT IS VECTOR SEARCH?
  We have a list of embedded chunks (from Lesson 2).
  We embed the user's question using the same model.
  We compare the question vector to every chunk vector.
  The chunks with the highest similarity score are returned.

THIS IS YOUR MINI VECTOR DATABASE.
  Real tools like ChromaDB or FAISS do the same thing —
  just faster and at larger scale. Understanding this file
  means you understand what those tools do under the hood.

WHAT YOU WILL SEE WHEN YOU RUN THIS FILE:
  - All chunks embedded and stored
  - A question embedded and compared to each chunk
  - Ranked results showing which chunks matched best

Requirements:
  pip install anthropic numpy
  export ANTHROPIC_API_KEY="sk-ant-..."
"""

import numpy as np
import anthropic

# Re-use the chunking function from Lesson 1
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        boundary = text.rfind(". ", start, end)
        if boundary != -1:
            end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

client = anthropic.Anthropic()

def embed_text(text: str) -> np.ndarray:
    response = client.embeddings.create(model="voyage-3", input=text)
    return np.array(response.embeddings[0].embedding, dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─ The core of this lesson 

def build_vector_store(chunks: list[str]) -> list[dict]:
    """
    Embed every chunk and store it with its text.
    This is your in-memory vector database.

    Returns a list of dicts, each with:
        { "text": str, "vector": np.ndarray }
    """
    print(f"  Building vector store from {len(chunks)} chunks...")
    store = []
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk)
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
        A list of dicts: [{ "score": float, "text": str }, ...]
    """
    # Embed the question using the SAME model as the documents
    query_vector = embed_text(query)

    # Score every chunk
    results = []
    for item in store:
        score = cosine_similarity(query_vector, item["vector"])
        results.append({"score": score, "text": item["text"]})

    # Sort by score, highest first
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]



if __name__ == "__main__":

    print("=" * 60)
    print("  LESSON 3: VECTOR SEARCH")
    print("=" * 60)

    # Load and chunk the document
    with open("document.txt", "r") as f:
        document = f.read()

    print("\n[1] Chunking the document...")
    chunks = chunk_text(document)
    print(f"    → {len(chunks)} chunks\n")

    print("[2] Embedding all chunks (building the vector store)...")
    store = build_vector_store(chunks)
    print(f"    → Vector store ready with {len(store)} entries\n")

    # Now test several questions
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

    print("""
   1. We embed the QUERY the same way as the documents.
   2. We score every chunk — the closer the score to 1.0, the better the match.
   3. top_k controls how many chunks we send to the LLM.
      More chunks = more context, but also a longer (more expensive) prompt.
   4. Real vector databases (ChromaDB, FAISS, Pinecone) do exactly this —
      they just do it much faster using clever indexing algorithms.
""")