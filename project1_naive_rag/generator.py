# generator.py
# ─────────────────────────────────────────────────────────
# Responsible for ONE thing: take a question + retrieved
# chunks → build a prompt → call the LLM → return the answer.
#
# This is the "G" in RAG (Generation).
#
# The prompt is arguably the most important engineering
# decision in a RAG system:
#   - How do you present the context?
#   - How do you instruct the LLM to stay grounded?
#   - How do you handle cases where context is insufficient?
#
# All of that is controlled right here in one place.
# ─────────────────────────────────────────────────────────

import json
import urllib.request

from config import OLLAMA_BASE_URL, CHAT_MODEL


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Construct the full prompt we send to the LLM.

    Structure:
        [System instruction — tells the LLM its role and constraints]
        [Context block    — the retrieved chunks, numbered and sourced]
        [Question         — the user's actual question]
        [Answer cue       — triggers the response]

    Why include the source in the context block?
        It lets the LLM (and the user) trace where each fact came from.
        In a production system you'd show these citations in the UI.

    Why say "ONLY use the context below"?
        Without this, the LLM will freely mix retrieved facts with its
        own training knowledge — you lose control over accuracy.
        Grounding = the LLM only answers from what you give it.

    Args:
        question:         The user's question.
        retrieved_chunks: List of {"text", "score", "metadata"} dicts.

    Returns:
        A complete prompt string ready to send to the LLM.
    """
    # Format each chunk with a label so the LLM can reference it
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk["metadata"].get("source", "unknown")
        score  = chunk["score"]
        text   = chunk["text"].strip()
        context_blocks.append(
            f"[Source {i}: {source} | relevance score: {score}]\n{text}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant for TechCorp employees.
Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say:
"I don't have enough information in my knowledge base to answer that."
Do not use any knowledge from outside the provided context.
Be concise and cite which source your answer comes from.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    return prompt


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the prompt and call the LLM to generate an answer.

    Args:
        question:         The user's question.
        retrieved_chunks: Results from the retriever.

    Returns:
        The LLM's response as a plain string.
    """
    prompt = build_prompt(question, retrieved_chunks)

    payload = json.dumps({
        "model":  CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    request = urllib.request.Request(
        url     = f"{OLLAMA_BASE_URL}/api/generate",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )

    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read())

    return data["response"].strip()