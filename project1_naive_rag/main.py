# main.py
# ─────────────────────────────────────────────────────────
# Entry point only. This file should be thin — its only job
# is to kick things off and print results.
#
# Rule of thumb for main.py:
#   If you find yourself writing logic here, it probably
#   belongs in one of the specialist modules instead.
# ─────────────────────────────────────────────────────────

from documents import DOCUMENTS
from rag_pipeline import RAGPipeline


def main():
    print("=" * 52)
    print("  Project 1 — Naive RAG from Scratch")
    print("  Stack: Ollama + nomic-embed-text + llama3.2")
    print("=" * 52)

    # ── Build the pipeline ────────────────────────────────
    pipeline = RAGPipeline()
    pipeline.index(DOCUMENTS)

    # ── Run queries ───────────────────────────────────────
    questions = [
        # Tests a clear policy fact
        "How many days per week can I work remotely?",

        # Tests a numeric detail buried in benefits text
        "What happens to my vacation days after 5 years?",

        # Tests a technical standard
        "How fast must API endpoints respond?",

        # Tests a nuanced rule (new employees)
        "Can I work remotely if I just started last week?",

        # Tests graceful out-of-scope handling
        "What is the company's policy on bringing pets to the office?",
    ]

    results = []
    for question in questions:
        result = pipeline.query(question)
        results.append(result)

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  Summary")
    print("=" * 52)
    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer'][:120]}...")
        print(f"   Sources: {', '.join(r['sources'])}")


if __name__ == "__main__":
    main()