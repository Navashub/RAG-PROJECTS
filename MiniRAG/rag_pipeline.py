"""

  LESSON 5 - FULL RAG PIPELINE  (multi-environment)       
  Local  → Ollama  (llama3.2)                             
  Render → Groq    (llama3-8b-8192)                       


WHAT'S NEW VS THE SINGLE-FILE VERSION:
  - generate_answer() detects the environment automatically
  - Local:  sends the prompt to Ollama's /api/chat endpoint
  - Render: sends the prompt to Groq (OpenAI-compatible API)
  - Everything else (chunking, embedding, search, prompt) is unchanged
"""

import requests
from config import CONFIG, show_config
from importlib import import_module

# Import each lesson's function
_chunking      = import_module("chunking")
_embedding     = import_module("embedding")
_vector_search = import_module("vector_search")
_prompt        = import_module("prompt_builder")

chunk_text         = _chunking.chunk_text
embed_text         = _embedding.embed_text
build_vector_store = _vector_search.build_vector_store
search             = _vector_search.search
build_rag_prompt   = _prompt.build_rag_prompt



# LOCAL — Ollama LLM
# ═══════════════════════════════════════════════════════════════════════════════

def generate_with_ollama(prompt: str) -> str:
    """
    Send a prompt to the local Ollama server and return the response.
    Uses the /api/chat endpoint with a single user message.
    """
    url = f"{CONFIG['ollama_base_url']}/api/chat"
    payload = {
        "model": CONFIG["llm_model"],   # llama3.2
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()



# PRODUCTION — Groq LLM
# ═══════════════════════════════════════════════════════════════════════════════

def generate_with_groq(prompt: str) -> str:
    """
    Send a prompt to Groq's API (OpenAI-compatible) and return the response.
    Groq's free tier is generous — fast inference on llama3-8b-8192.

    Docs: https://console.groq.com/docs/openai
    """
    api_key = CONFIG.get("groq_api_key")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set.\n"
            "Add it as an environment variable on Render."
        )

    url = f"{CONFIG['groq_base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CONFIG["llm_model"],   # llama3-8b-8192
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.2,             # low temp = more factual, less creative
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()



# PUBLIC INTERFACE — one function, works in both environments
# ═══════════════════════════════════════════════════════════════════════════════

def generate_answer(prompt: str) -> str:
    """
    Generate an answer using the right LLM backend.
    Automatically uses Ollama locally and Groq on Render.
    """
    if CONFIG["env"] == "local":
        return generate_with_ollama(prompt)
    else:
        return generate_with_groq(prompt)


# ── Pipeline

def ask(question: str, store: list[dict], top_k: int = 3) -> str:
    """Ask a question against the pre-built vector store."""
    results    = search(question, store, top_k=top_k)
    top_chunks = [(r["score"], r["text"]) for r in results]
    prompt     = build_rag_prompt(question, top_chunks)
    answer     = generate_answer(prompt)
    return answer




if __name__ == "__main__":

    print("=" * 60)
    print("  LESSON 5: FULL RAG PIPELINE")
    print("=" * 60)
    show_config()

    # One-time setup
    print(" Loading document...")
    with open("document.txt", "r") as f:
        document = f.read()

    print("  Chunking...")
    chunks = chunk_text(document)
    print(f"   → {len(chunks)} chunks\n")

    print(" Building vector store...")
    store = build_vector_store(chunks)
    print(f"   → {len(store)} vectors stored\n")

    # Questions
    questions = [
        "What are the two main phases of RAG?",
        "What does cosine similarity measure?",
        "Why do organisations prefer RAG over retraining their models?",
        # This is NOT in the document — watch the model say "I don't know"
        "Who invented RAG and in which year?",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f" {question}")
        print("=" * 60)
        answer = ask(question, store, top_k=2)
        print(f" {answer}")

    print("\n\n Done!")
    print(f"\n   Running on: {CONFIG['env'].upper()}")
    print(f"   LLM used  : {CONFIG['llm_model']}")
    print(f"   Embed model: {CONFIG['embed_model']}\n")