
"""

 LESSON 1 - CHUNKING                                     
  Goal: Split a document into small, overlapping pieces   


WHY DO WE CHUNK?
  LLMs have a limit on how much text they can read at once.
  We break the document into small pieces so we can pick
  ONLY the relevant ones later instead of sending everything.

WHY OVERLAP?
  If a sentence falls right at the boundary between two chunks,
  we would cut it in half and lose its meaning. Overlap prevents that.
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks by WORDS, not characters.
    Word-based splitting is more natural and avoids cutting mid-word.

    Args:
        text:       The full document as a string.
        chunk_size: Number of WORDS per chunk (not characters).
        overlap:    Number of WORDS to repeat between chunks.

    Returns:
        A list of text strings.
    """
    words = text.split()          # split into individual words
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        # Move forward by (chunk_size - overlap) words
        # This guarantees start always moves forward
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    with open("document.txt", "r") as f:
        document = f.read()

    print("=" * 60)
    print("  LESSON 1: CHUNKING")
    print("=" * 60)

    word_count = len(document.split())
    print("\nDocument stats:")
    print("  Characters :", len(document))
    print("  Words      :", word_count)

    # Show chunks at different sizes so students see the effect
    print("\nHOW CHUNK SIZE AFFECTS THE NUMBER OF CHUNKS")
    print("-" * 60)
    print(f"  {'chunk_size (words)':<22} {'overlap':<10} {'chunks':<10}")
    print(f"  {'-'*20:<22} {'-'*8:<10} {'-'*8:<10}")

    for size, ov in [(50, 10), (100, 20), (200, 40)]:
        result = chunk_text(document, chunk_size=size, overlap=ov)
        print(f"  {size:<22} {ov:<10} {len(result):<10}")

    # Show the actual chunks at default settings
    print("\n")
    chunks = chunk_text(document, chunk_size=100, overlap=20)
    print(f"Chunks at chunk_size=100 words, overlap=20 words:")
    print("-" * 60)

    for i, chunk in enumerate(chunks):
        words_in_chunk = len(chunk.split())
        print(f"\nChunk {i + 1}  ({words_in_chunk} words)")
        print(chunk)
        print("-" * 60)

    print("""
Teaching Points:
  - We split by WORDS not characters (more natural boundaries)
  - chunk_size=100 words is roughly 2-4 sentences
  - overlap=20 words means the last 20 words of one chunk
    appear at the start of the next chunk
  - start += chunk_size - overlap always moves forward (no infinite loop)
""")