# rag_pipeline.py
# ─────────────────────────────────────────────────────────
# The orchestrator. This file's job is to wire all the
# components together and expose a clean interface.
#
# Notice what this file does NOT do:
#   - It doesn't know how chunking works (chunker.py)
#   - It doesn't know how embeddings work (embedder.py)
#   - It doesn't know how storage works (vector_store.py)
#   - It doesn't know how the LLM is called (generator.py)
#
# It only knows the ORDER of operations and delegates
# each step to the right specialist module.
#
# This is the Facade pattern — a simple interface over a
# complex subsystem. main.py calls this; it doesn't need
# to know any of the internals.
# ─────────────────────────────────────────────────────────

from chunker import chunk_documents
from embedder import embed_batch
from vector_store import VectorStore
from retriever import Retriever
from generator import generate_answer


class RAGPipeline:
    """
    A complete Retrieval-Augmented Generation pipeline.

    Two phases:
        1. index(documents)  — build the searchable knowledge base
        2. query(question)   — answer a question using that knowledge base

    Usage:
        pipeline = RAGPipeline()
        pipeline.index(DOCUMENTS)
        result = pipeline.query("What is the remote work policy?")
        print(result["answer"])
    """

    def __init__(self):
        self.store     = VectorStore()
        self.retriever = None   # created after indexing

    # ── Phase 1: Indexing ─────────────────────────────────

    def index(self, documents: list[dict]) -> None:
        """
        Process documents into a searchable vector store.

        Steps:
            1. Chunk each document into smaller pieces
            2. Embed every chunk (text → vector)
            3. Store (chunk, vector, metadata) in the vector store
            4. Initialise the retriever on top of the store

        Args:
            documents: List of {"source": str, "text": str} dicts.
        """
        print("── Indexing Phase ───────────────────────────────────")

        # Step 1: Chunk
        print("\n[1/3] Chunking documents...")
        chunks = chunk_documents(documents)
        print(f"      Created {len(chunks)} chunks from {len(documents)} documents")

        # Step 2: Embed
        print("\n[2/3] Embedding chunks...")
        texts    = [c["text"] for c in chunks]
        vectors  = embed_batch(texts)

        # Step 3: Store
        print("\n[3/3] Storing in vector store...")
        for chunk, vector in zip(chunks, vectors):
            self.store.add(
                text     = chunk["text"],
                vector   = vector,
                metadata = {
                    "source":      chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                },
            )

        # Step 4: Wire up retriever
        self.retriever = Retriever(self.store)

        stats = self.store.stats()
        print(f"\n✅ Indexing complete.")
        print(f"   Chunks stored:       {stats['total_chunks']}")
        print(f"   Embedding dimension: {stats['embedding_dimension']}")
        print("─" * 52)

    # ── Phase 2: Querying ─────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Answer a question using the indexed knowledge base.

        Steps:
            1. Retrieve the most relevant chunks for the question
            2. Generate an answer grounded in those chunks

        Args:
            question: A natural-language question from the user.

        Returns:
            {
                "question":  the original question,
                "answer":    the LLM's grounded response,
                "sources":   list of source filenames used,
                "chunks":    the raw retrieved chunk dicts,
            }
        """
        if self.retriever is None:
            raise RuntimeError("Call index() before query().")

        print(f"\n── Query ────────────────────────────────────────────")
        print(f"   Q: {question}")

        # Step 1: Retrieve
        chunks = self.retriever.retrieve_with_log(question)

        # Step 2: Generate
        print("\n   Generating answer...")
        answer = generate_answer(question, chunks)

        sources = list({c["metadata"].get("source", "unknown") for c in chunks})

        result = {
            "question": question,
            "answer":   answer,
            "sources":  sources,
            "chunks":   chunks,
        }

        print(f"\n   Answer:\n  {answer}")
        print(f"\n  Sources used: {', '.join(sources)}")
        print("─" * 52)

        return result