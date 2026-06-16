# embedder.py
# ─────────────────────────────────────────────────────────
# Responsible for ONE thing: converting text → vectors.
#
# What is an embedding?
#   A list of floats (e.g. 768 numbers) that encodes
#   the *meaning* of a piece of text. The model is trained
#   so that similar meanings produce similar vectors.
#
#   "dog" and "puppy"  → vectors close together (small angle)
#   "dog" and "invoice" → vectors far apart (large angle)
#
# Why does this enable search?
#   Instead of matching keywords, we match *meaning*.
#   A query like "can I bring my pet?" will find chunks
#   mentioning "dog" or "animal" even without those exact words.
#
# This module talks to Ollama's /api/embeddings endpoint.
# Swapping to OpenAI or another provider = change this file only.
# ─────────────────────────────────────────────────────────

import json
import urllib.request

from config import OLLAMA_BASE_URL, EMBEDDING_MODEL


def embed(text: str) -> list[float]:
    """
    Embed a single string using the configured Ollama model.

    Args:
        text: Any string — a chunk, a query, a sentence.

    Returns:
        A list of floats representing the text in vector space.
        With nomic-embed-text this is 768 dimensions.

    Raises:
        urllib.error.URLError: If Ollama is not running.
        KeyError: If the response shape changes (model version issue).
    """
    payload = json.dumps({
        "model":  EMBEDDING_MODEL,
        "prompt": text,
    }).encode("utf-8")

    request = urllib.request.Request(
        url     = f"{OLLAMA_BASE_URL}/api/embeddings",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )

    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read())

    return data["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings, one by one.

    NOTE: Ollama does not yet support true batch embedding in one request.
    We call embed() per text. In production with OpenAI you'd batch these
    into a single API call to reduce latency and cost.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors, in the same order as the input.
    """
    embeddings = []

    for i, text in enumerate(texts):
        print(f"    Embedding {i + 1}/{len(texts)}: {text[:50]}...")
        vector = embed(text)
        embeddings.append(vector)

    return embeddings