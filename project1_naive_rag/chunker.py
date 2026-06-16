# chunker.py
# ─────────────────────────────────────────────────────────
# Responsible for ONE thing: splitting raw text into chunks.
#
# Why chunking matters:
#   Embedding models have a token limit. More importantly,
#   embedding a whole document averages out all its meaning —
#   you lose the ability to pinpoint the ONE relevant sentence.
#   Smaller chunks = more precise retrieval.
#
# The trade-off:
#   Too small → a chunk has no useful context on its own
#   Too large → retrieval becomes imprecise (too much mixed meaning)
#   Sweet spot → 150–500 chars for most use cases
# ─────────────────────────────────────────────────────────

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping fixed-size chunks.

    Args:
        text:       The raw text to split.
        chunk_size: Max characters per chunk.
        overlap:    How many characters the next chunk re-uses
                    from the tail of the previous chunk.

    Returns:
        A list of non-empty text chunks.

    Example with chunk_size=20, overlap=5:
        text  = "AAAAABBBBBCCCCCDDDDDEEEEE"
        chunk 1 → "AAAAABBBBBCCCCCDDDDDE"   (positions 0..20)
        chunk 2 → "DDDDDEEEEE..."            (positions 15..35)
                   ^^^^^ shared overlap keeps context across the boundary
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:  # ignore empty / whitespace-only slices
            chunks.append(chunk)

        # Advance by (chunk_size - overlap) so next chunk
        # re-uses the last `overlap` characters of this one
        start += chunk_size - overlap

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk every document and attach metadata to each chunk.

    Args:
        documents: List of {"source": str, "text": str} dicts.

    Returns:
        List of chunk dicts:
        {
            "text":        the chunk string,
            "source":      original document source,
            "chunk_index": position of this chunk within its document
        }
    """
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])

        for i, chunk_text_value in enumerate(chunks):
            all_chunks.append({
                "text":        chunk_text_value,
                "source":      doc["source"],
                "chunk_index": i,
            })

    return all_chunks