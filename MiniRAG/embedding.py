"""

  LESSON 2 - EMBEDDING  (multi-environment version)       
  Local  → Ollama  (nomic-embed-text)                     
  Render → sentence-transformers (nomic-embed-text-v1.5)  


WHY TWO EMBEDDING BACKENDS?
  Groq is great for LLM inference but does NOT offer an embedding API.
  So we use:
    - Ollama locally      → fast, uses your GPU/CPU, no extra install
    - sentence-transformers on Render → pure Python, no server needed,
      runs the same nomic-embed-text model weights directly

  The vectors produced are compatible — same model, same dimensions.
"""

import numpy as np
import requests
from config import CONFIG, show_config


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL - Ollama embedding
# ═══════════════════════════════════════════════════════════════════════════════

def embed_with_ollama(text: str) -> np.ndarray:
    """
    Call the local Ollama server to get an embedding.
    Ollama must be running: `ollama serve`
    Model must be pulled:   `ollama pull nomic-embed-text`
    """
    url = f"{CONFIG['ollama_base_url']}/api/embeddings"
    payload = {
        "model": CONFIG["embed_model"],
        "prompt": text,
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    vector = response.json()["embedding"]
    return np.array(vector, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION — sentence-transformers embedding (runs on Render, no server)
# ═══════════════════════════════════════════════════════════════════════════════

def embed_with_sentence_transformers(text: str) -> np.ndarray:
    """
    Use the sentence-transformers library to embed text locally.
    No server needed — the model runs in-process.

    Install: pip install sentence-transformers
    First run downloads the model weights (~270 MB, cached after that).
    """
    # Lazy import so local users don't need to install it
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed.\n"
            "Run: pip install sentence-transformers"
        )

    # Cache the model so we don't reload it on every call
    if not hasattr(embed_with_sentence_transformers, "_model"):
        print("  Loading embedding model (first time only)...")
        embed_with_sentence_transformers._model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
        )

    model = embed_with_sentence_transformers._model
    vector = model.encode(text, normalize_embeddings=True)
    return np.array(vector, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE - one function, works in both environments
# ═══════════════════════════════════════════════════════════════════════════════

def embed_text(text: str) -> np.ndarray:
    """
    Convert text to an embedding vector.
    Automatically uses the right backend based on config.py.
    """
    if CONFIG["env"] == "local":
        return embed_with_ollama(text)
    else:
        return embed_with_sentence_transformers(text)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Range: -1.0 to 1.0."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


if __name__ == "__main__":

    print("=" * 60)
    print("  LESSON 2: EMBEDDING")
    print("=" * 60)
    show_config()

    # Show what a vector looks like ───
    sample = "RAG retrieves documents to help an LLM answer questions."
    print(f" Embedding: '{sample}'\n")

    vector = embed_text(sample)
    print(f" Vector (first 10 of {len(vector)} values):")
    print("  ", vector[:10])

    # Similarity comparison 
    print("\n\n SIMILARITY COMPARISON")
    print("-" * 60)

    pairs = [
        (
            "RAG retrieves documents to help an LLM answer questions.",
            "Retrieval-Augmented Generation uses external knowledge.",
            "Similar (same topic)",
        ),
        (
            "RAG retrieves documents to help an LLM answer questions.",
            "My favourite food is jollof rice.",
            "nrelated",
        ),
        (
            "Vector databases store embeddings for fast search.",
            "Embeddings are stored in vector databases for retrieval.",
            "Very similar (same sentence, rearranged)",
        ),
    ]

    for a, b, label in pairs:
        vec_a = embed_text(a)
        vec_b = embed_text(b)
        score = cosine_similarity(vec_a, vec_b)
        print(f"\n  {label}")
        print(f"  A: \"{a[:65]}\"")
        print(f"  B: \"{b[:65]}\"")
        print(f"  Similarity: {score:.4f}")

    print("""
   - Local:  Ollama serves the model like a local API server
   - Render: sentence-transformers runs the model directly in Python
   - Same model name, same vector dimensions → fully compatible
   - You can switch environments without changing any other file
""")