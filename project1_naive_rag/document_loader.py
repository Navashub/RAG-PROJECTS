# document_loader.py
# ─────────────────────────────────────────────────────────
# Responsible for ONE thing: reading raw documents from disk
# and turning them into the {"source", "text"} shape that
# the rest of the pipeline already expects.
#
# Kept deliberately simple for Project 1: plain .txt files only.
# Project 1's goal is to learn chunking, embedding, and retrieval —
# not file parsing. We'll handle PDFs/DOCX in a later project once
# the core RAG concepts are second nature.
# ─────────────────────────────────────────────────────────

import os


def load_documents_from_folder(folder_path: str) -> list[dict]:
    """
    Read every .txt file in a folder and return them as documents.

    Args:
        folder_path: Path to a folder containing .txt files.

    Returns:
        List of {"source": filename, "text": file_contents} dicts —
        the exact shape chunker.chunk_documents() expects.

    Raises:
        FileNotFoundError: If the folder doesn't exist.
        ValueError: If the folder has no .txt files.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Folder not found: '{folder_path}'. "
            f"Create it and add some .txt files first."
        )

    documents = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(".txt"):
            continue  # only .txt for Project 1

        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            print(f"    Skipping '{filename}' — file is empty")
            continue

        documents.append({
            "source": filename,
            "text":   text,
        })

    if not documents:
        raise ValueError(
            f"No .txt files found in '{folder_path}'. "
            f"Add at least one document and try again."
        )

    return documents