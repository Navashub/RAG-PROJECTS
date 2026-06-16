# config.py
# ─────────────────────────────────────────────────────────
# ALL configuration lives here.
#
# Why a dedicated config file?
#   - You change settings in ONE place, not hunting through code
#   - Easy to swap models (e.g. local → cloud) without touching logic
#   - In production this would load from environment variables
#     using python-dotenv or a secrets manager
# ─────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"

# The model that turns text into vectors (numbers)
EMBEDDING_MODEL = "nomic-embed-text:latest"

# The model that reads context and writes the answer
CHAT_MODEL = "llama3.2:latest"

# ── Chunking ─────────────────────────────────────────────
# How many characters per chunk.
# Too small → chunks lose context. Too large → retrieval is imprecise.
CHUNK_SIZE = 200

# How many characters the next chunk re-uses from the previous one.
# Prevents meaning being lost at chunk boundaries.
CHUNK_OVERLAP = 40

# ── Retrieval ────────────────────────────────────────────
# How many chunks to pull back per query.
# More chunks = more context for the LLM, but also more noise.
TOP_K = 3