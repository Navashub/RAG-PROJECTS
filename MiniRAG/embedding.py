"""
╔══════════════════════════════════════════════════════════╗
║  LESSON 2 — EMBEDDING                                    ║
║  Goal: Turn text into numbers a computer can compare     ║
╚══════════════════════════════════════════════════════════╝

WHY DO WE EMBED?
  Computers can't compare text directly — "cat" and "kitten" look
  completely different as strings. But if we convert them to numbers
  (vectors), we can MEASURE how similar they are mathematically.

WHAT IS A VECTOR?
  A vector is just a list of numbers, e.g. [0.12, -0.45, 0.87, ...]
  An embedding model produces ~1024 of these numbers for any piece of text.
  Similar meanings → similar numbers → vectors that are "close" in space.

WHAT YOU WILL SEE WHEN YOU RUN THIS FILE:
  - The raw vector output for a sentence
  - A side-by-side comparison of similar vs different sentences
  - Why the same embedding model must be used for docs AND queries

Requirements:
  pip install anthropic numpy
  export ANTHROPIC_API_KEY="sk-ant-..."
"""

import numpy as np
import anthropic

client = anthropic.Anthropic()


def embed_text(text: str) -> np.ndarray:
    """
    Convert a piece of text into an embedding vector.

    Args:
        text: Any string of text.

    Returns:
        A numpy array of floats — the embedding vector.
    """
    response = client.embeddings.create(
        model="voyage-3",
        input=text,
    )
    vector = response.embeddings[0].embedding
    return np.array(vector, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measure how similar two vectors are.
    Score of 1.0 = identical direction (same meaning)
    Score of 0.0 = unrelated
    Score of -1.0 = opposite meaning
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))




if __name__ == "__main__":

    print("=" * 60)
    print("  LESSON 2: EMBEDDING")
    print("=" * 60)

    #  Show what a vector looks like 
    sample = "RAG retrieves documents to help an LLM answer questions."
    print(f"\n Input text:\n   '{sample}'\n")

    vector = embed_text(sample)
    print(f" Embedding vector (first 10 of {len(vector)} numbers):")
    print("  ", vector[:10])
    print(f"\n   → Shape: {vector.shape}  |  Type: {vector.dtype}\n")

    # Compare similar vs different sentences 
    print("-" * 60)
    print(" SIMILARITY COMPARISON")
    print("-" * 60)

    pairs = [
        # (sentence_a, sentence_b, expected relationship)
        (
            "RAG retrieves documents to help an LLM answer questions.",
            "Retrieval-Augmented Generation uses external knowledge.",
            " Very similar (same topic, different wording)",
        ),
        (
            "RAG retrieves documents to help an LLM answer questions.",
            "The cat sat on the mat.",
            " Unrelated (completely different topic)",
        ),
        (
            "The weather is sunny today.",
            "It is a bright and sunny day.",
            "Very similar (same meaning, different words)",
        ),
        (
            "Vector databases store embeddings for fast retrieval.",
            "My favourite food is jollof rice.",
            " Unrelated",
        ),
    ]

    for a, b, label in pairs:
        vec_a = embed_text(a)
        vec_b = embed_text(b)
        score = cosine_similarity(vec_a, vec_b)
        print(f"\n  {label}")
        print(f"  A: \"{a[:60]}\"")
        print(f"  B: \"{b[:60]}\"")
        print(f"  Similarity score: {score:.4f}")

    print("""
   1. Similar meaning → score close to 1.0
   2. Unrelated text  → score close to 0.0
   3. The SAME embedding model must be used for documents AND queries.
      Mixing models is like translating with two different dictionaries — 
      the numbers won't be compatible.
""")