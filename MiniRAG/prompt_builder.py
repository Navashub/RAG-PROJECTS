"""

  LESSON 4 — PROMPT BUILDING                              
  Goal: Combine retrieved chunks + question into a prompt 


THIS IS THE HEART OF RAG.
  The prompt is what actually changes the LLM's behaviour.
  Without a good prompt, even perfect retrieval gives bad answers.

THE KEY INSTRUCTION IS:
  "Answer using ONLY the context below."
  This one line is what prevents hallucination.
  The LLM is not allowed to guess -it must cite what it was given.

WHAT YOU WILL SEE WHEN YOU RUN THIS FILE:
  - A RAG prompt printed clearly 
  - A comparison: prompt WITH context vs prompt WITHOUT context
  - Why the structure of the prompt matters

No API calls needed for this lesson - it's pure Python!
"""


def build_rag_prompt(question: str, chunks: list[tuple[float, str]]) -> str:
    """
    Build the RAG prompt that will be sent to the LLM.

    Args:
        question: The user's question.
        chunks:   A list of (similarity_score, chunk_text) tuples.

    Returns:
        A formatted string ready to send to the LLM.
    """
    # Format each retrieved chunk with its relevance score
    context_block = "\n\n---\n\n".join(
        f"[Relevance: {score:.2f}]\n{text}"
        for score, text in chunks
    )

    # The prompt has 3 clear sections:
    #   1. Instructions to the LLM  ← tells it how to behave
    #   2. Context                  ← the retrieved knowledge
    #   3. Question                 ← what the user actually asked
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not use any outside knowledge.

=== CONTEXT (retrieved from the document) ===

{context_block}

=== USER QUESTION ===

{question}

=== YOUR ANSWER ==="""

    return prompt


def build_plain_prompt(question: str) -> str:
    """
    A prompt WITHOUT any retrieved context - for comparison.
    This is what a normal (non-RAG) LLM call looks like.
    """
    return f"""You are a helpful assistant.

=== USER QUESTION ===

{question}

=== YOUR ANSWER ==="""




if __name__ == "__main__":

    print("=" * 60)
    print("  LESSON 4: PROMPT BUILDING")
    print("=" * 60)

    # Pretend these came from Lesson 3's vector search
    fake_retrieved_chunks = [
        (0.91, "The RAG pipeline has two main phases. The first is indexing: "
               "documents are split into smaller chunks, each chunk is converted "
               "into a numerical vector using an embedding model, and these vectors "
               "are stored in a vector database."),
        (0.85, "The second phase is retrieval and generation: when a user asks a "
               "question, the question is also converted into an embedding. The system "
               "then searches the vector database for chunks whose embeddings are most "
               "similar to the question embedding."),
    ]

    question = "What are the two main phases of RAG?"

    #  Show the RAG prompt 
    print("\n RAG PROMPT (with retrieved context):\n")
    print("-" * 60)
    rag_prompt = build_rag_prompt(question, fake_retrieved_chunks)
    print(rag_prompt)
    print("-" * 60)

    # ─Show the plain prompt for comparison 
    print("\n\n PLAIN PROMPT (no context — what a normal LLM call looks like):\n")
    print("-" * 60)
    plain_prompt = build_plain_prompt(question)
    print(plain_prompt)
    print("-" * 60)

    # Prompt stats
    print("\n\n PROMPT COMPARISON")
    print(f"  {'Type':<25} {'Characters':<15} {'Lines':<10}")
    print(f"  {'-'*23:<25} {'-'*13:<15} {'-'*8:<10}")
    print(f"  {'RAG prompt':<25} {len(rag_prompt):<15} {len(rag_prompt.splitlines()):<10}")
    print(f"  {'Plain prompt':<25} {len(plain_prompt):<15} {len(plain_prompt.splitlines()):<10}")

    print("""
   1. The RAG prompt is longer because it includes the context.
      Every extra character costs tokens - so we only include TOP chunks.

   2. "Answer ONLY from the context" is the anti-hallucination instruction.
      Without it, the LLM will happily mix context with its own (possibly
      wrong) training knowledge.

   3. If the answer isn't in the retrieved chunks, a good RAG system should
      say "I don't know" - not invent something. Test this in Lesson 5!

   4. Prompt structure matters:
        INSTRUCTIONS → CONTEXT → QUESTION → ANSWER PLACEHOLDER
      This order helps the LLM understand the task before seeing the data.
""")