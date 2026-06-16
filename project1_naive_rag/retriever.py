# retriever.py
# ─────────────────────────────────────────────────────────
# Responsible for ONE thing: given a question, find the
# most relevant chunks from the vector store.
#
# The Retriever sits between the user's question and the
# vector store. It handles:
#   1. Embedding the question
#   2. Calling the vector store's search
#   3. Returning clean, ranked results
#
# Why a separate Retriever class?
#   In later projects we'll upgrade retrieval (hybrid search,
#   reranking, query expansion) without touching anything else.
#   The interface stays the same; the internals change.
#   That's the Open/Closed Principle — open for extension,
#   closed for modification.
# ─────────────────────────────────────────────────────────

from embedder import embed
from vector_store import VectorStore
from config import TOP_K


class Retriever:
    """
    Wraps a VectorStore to provide question → relevant chunks retrieval.

    Usage:
        retriever = Retriever(store)
        results   = retriever.retrieve("What is the remote work policy?")
    """

    def __init__(self, store: VectorStore, top_k: int = TOP_K):
        self.store = store
        self.top_k = top_k

    def retrieve(self, question: str) -> list[dict]:
        """
        Find the most relevant chunks for a given question.

        Args:
            question: The user's natural-language question.

        Returns:
            List of result dicts from VectorStore.search():
            [{"text": ..., "score": ..., "metadata": ...}, ...]
        """
        # Embed the question into the same vector space as the chunks.
        # This is the KEY insight of semantic search:
        # question and relevant chunks will point in similar directions.
        question_vector = embed(question)

        results = self.store.search(query_vector=question_vector, top_k=self.top_k)

        return results

    def retrieve_with_log(self, question: str) -> list[dict]:
        """
        Same as retrieve(), but prints what was found.
        Useful during development and for understanding system behaviour.
        """
        print(f"\n  🔍 Retrieving for: '{question}'")

        results = self.retrieve(question)

        for i, result in enumerate(results, start=1):
            source  = result["metadata"].get("source", "unknown")
            score   = result["score"]
            preview = result["text"][:70].replace("\n", " ")
            print(f"    {i}. [score={score}] [{source}] {preview}...")

        return results