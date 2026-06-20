"""
╔══════════════════════════════════════════════════════════╗
║  LESSON 1 — CHUNKING                                     ║
║  Goal: Split a document into small, overlapping pieces   ║
╚══════════════════════════════════════════════════════════╝

WHY DO WE CHUNK?
  LLMs have a limit on how much text they can read at once (context window).
  Also, sending the whole document every time is wasteful and slow.
  We break it into small pieces so we can pick ONLY the relevant ones later.

WHY OVERLAP?
  If a sentence falls right at the boundary between two chunks,
  we'd cut it in half and lose its meaning. Overlap prevents that.

WHAT YOU WILL SEE WHEN YOU RUN THIS FILE:
  - The original document
  - Each chunk printed with its number and character count
  - A comparison of different chunk sizes
"""



def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text:       The full document as a string.
        chunk_size: How many characters per chunk (300 ≈ 2-3 sentences).
        overlap:    How many characters to repeat between chunks.

    Returns:
        A list of text strings.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

       
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to break at a sentence boundary (period + space)
        # so we don't cut mid-sentence
        boundary = text.rfind(". ", start, end)
        if boundary != -1:
            end = boundary + 1  

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
 
        start = next_start
 
        # Stop if we have reached the end
        if end >= len(text):
            break

    return chunks



if __name__ == "__main__":

    # Load the shared document
    with open("document.txt", "r") as f:
        document = f.read()

    print("=" * 60)
    print("  LESSON 1: CHUNKING")
    print("=" * 60)

    print("\n Original document length:", len(document), "characters\n")

    chunks = chunk_text(document, chunk_size=300, overlap=50)

    print(f" Split into {len(chunks)} chunks (chunk_size=300, overlap=50)\n")
    print("-" * 60)

    for i, chunk in enumerate(chunks):
        print(f"\n Chunk {i + 1}  ({len(chunk)} chars)")
        print(chunk)
        print("-" * 60)

   # comparing chunks
    print("\n\n HOW CHUNK SIZE AFFECTS THE NUMBER OF CHUNKS")
    print("-" * 60)
    print(f"  {'chunk_size':<15} {'overlap':<10} {'# chunks':<10}")
    print(f"  {'-'*13:<15} {'-'*8:<10} {'-'*8:<10}")

    for size, ov in [(100, 20), (300, 50), (500, 80), (1000, 100)]:
        result = chunk_text(document, chunk_size=size, overlap=ov)
        print(f"  {size:<15} {ov:<10} {len(result):<10}")

    print("""
   - Small chunks  → more precise retrieval, but may lose context
   - Large chunks  → more context, but may include irrelevant text
   - Good default: 300-500 characters with ~50 character overlap
""")