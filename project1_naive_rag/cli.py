# cli.py
# ─────────────────────────────────────────────────────────
# An interactive entry point: load real documents from a
# folder, index them once, then let the user ask unlimited
# questions in a loop until they choose to exit.
#
# This is the file you run day-to-day. main.py still exists
# as a fixed demo/regression test — useful for confirming
# nothing broke after you change code.
#
# Rule of thumb (same as main.py):
#   This file should stay thin. It loads data, drives a loop,
#   and prints things. Real logic lives in the pipeline.
# ─────────────────────────────────────────────────────────

from document_loader import load_documents_from_folder
from rag_pipeline import RAGPipeline

DATA_FOLDER = "data"


def print_welcome():
    print("=" * 52)
    print("  RAG Assistant - Interactive Mode")
    print("  Stack: Ollama + nomic-embed-text + llama3.2")
    print("=" * 52)


def print_help():
    print("\nCommands:")
    print("  Type a question and press Enter to get an answer.")
    print("  Type 'sources' to see which files are loaded.")
    print("  Type 'exit' or 'quit' to leave.\n")


def run_question_loop(pipeline: RAGPipeline, loaded_sources: list[str]):
    """
    The interactive loop: read a question, answer it, repeat.

    Args:
        pipeline:       An already-indexed RAGPipeline.
        loaded_sources: Filenames that were loaded, for the 'sources' command.
    """
    while True:
        question = input("\n Your question: ").strip()

        if not question:
            continue  # ignore empty input, ask again

        if question.lower() in ("exit", "quit"):
            print("\n Goodbye!")
            break

        if question.lower() == "sources":
            print("\n Loaded documents:")
            for source in loaded_sources:
                print(f"   - {source}")
            continue

        result = pipeline.query(question)
        # pipeline.query() already prints the answer and sources,
        # so there's nothing extra to do here — this is the benefit
        # of putting that logging inside the pipeline once.


def main():
    print_welcome()

    # ── Load documents from disk ──────────────────────────
    print(f"\n Loading documents from '{DATA_FOLDER}/'...")
    try:
        documents = load_documents_from_folder(DATA_FOLDER)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ {e}")
        return

    print(f"   Found {len(documents)} file(s):")
    for doc in documents:
        print(f"   - {doc['source']}")

    # ── Index once ─────────────────────────────────────────
    pipeline = RAGPipeline()
    pipeline.index(documents)

    # ── Interactive loop ──────────────────────────────────
    print_help()
    sources = [doc["source"] for doc in documents]
    run_question_loop(pipeline, sources)


if __name__ == "__main__":
    main()