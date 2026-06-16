# vector_store.py
# ─────────────────────────────────────────────────────────
# Responsible for ONE thing: storing vectors and finding
# the most similar ones to a query.
#
# This is your "vector database" — the heart of RAG.
#
# Real vector databases (ChromaDB, Pinecone, Qdrant, Weaviate)
# do exactly what this class does, plus:
#   - Persist data to disk or cloud
#   - Approximate nearest-neighbour (ANN) search using HNSW graphs
#     so similarity search is O(log n) instead of O(n)
#   - Metadata filtering ("only search chunks from 2024 documents")
#   - Horizontal scaling across many machines
#
# Building it from scratch here means you understand what
# "add a vector database" actually means — no black box.
# ─────────────────────────────────────────────────────────

import numpy as np

from config import TOP_K


class VectorStore:
    """
    An in-memory vector store backed by plain Python lists.

    Storage layout (three parallel lists — same index = same record):
        self.texts     → ["chunk text 1",  "chunk text 2",  ...]
        self.vectors   → [np.array([...]), np.array([...]), ...]
        self.metadata  → [{"source": ...}, {"source": ...}, ...]
    """

    def __init__(self):
        self.texts:    list[str]        = []
        self.vectors:  list[np.ndarray] = []
        self.metadata: list[dict]       = []

    # ── Writing ───────────────────────────────────────────

    def add(self, text: str, vector: list[float], metadata: dict = None) -> None:
        """
        Store a single chunk with its embedding and optional metadata.

        Args:
            text:     The raw chunk string (returned in search results).
            vector:   The embedding produced by embedder.embed(text).
            metadata: Anything useful — source file, page number, date, etc.
        """
        self.texts.append(text)
        self.vectors.append(np.array(vector, dtype=np.float32))
        self.metadata.append(metadata or {})

    # ── Reading ───────────────────────────────────────────

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Measure the similarity between two vectors using the angle between them.

        Formula:  cos(θ) = (A · B) / (|A| × |B|)

        Why cosine and not Euclidean distance?
            Cosine ignores vector length — only the direction matters.
            A long document and a short one on the same topic will still
            score high similarity even though their raw magnitudes differ.

        Return value:
             1.0 → identical direction (same meaning)
             0.0 → perpendicular (unrelated)
            -1.0 → opposite direction (opposite meaning)
        """
        dot_product = np.dot(vec_a, vec_b)
        magnitude   = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

        if magnitude == 0:
            return 0.0

        return float(dot_product / magnitude)

    def search(self, query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
        """
        Find the top_k chunks most similar to the query vector.

        This is brute-force O(n) — we compare the query to every
        stored vector. Fine for hundreds or thousands of chunks.
        For millions of vectors, a real ANN index is needed.

        Args:
            query_vector: Embedding of the user's question.
            top_k:        How many results to return.

        Returns:
            List of result dicts, sorted by score descending:
            [
                {
                    "text":     "the chunk content",
                    "score":    0.87,
                    "metadata": {"source": "policy.txt", "chunk_index": 2}
                },
                ...
            ]
        """
        q = np.array(query_vector, dtype=np.float32)

        # Score every stored vector against the query
        scored = [
            (i, self.cosine_similarity(q, v))
            for i, v in enumerate(self.vectors)
        ]

        # Sort highest score first, take top_k
        scored.sort(key=lambda pair: pair[1], reverse=True)

        results = []
        for idx, score in scored[:top_k]:
            results.append({
                "text":     self.texts[idx],
                "score":    round(score, 4),
                "metadata": self.metadata[idx],
            })

        return results

    # ── Inspection ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.texts)

    def stats(self) -> dict:
        """Return a summary — useful for debugging and logging."""
        return {
            "total_chunks":       len(self.texts),
            "embedding_dimension": len(self.vectors[0]) if self.vectors else 0,
        }