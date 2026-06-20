"""                      
  Tells every other file which models to use              


HOW IT WORKS:
  We check one environment variable: MINIRAG_ENV
  - Not set / "local"  → use Ollama (your laptop)
  - "production"       → use Groq (Render deployment)

HOW TO SET IT:

  Local (you never need to set it — "local" is the default):
    No action needed. Just run your scripts.

  On Render (set this in your service's Environment Variables):
    MINIRAG_ENV=production
    GROQ_API_KEY=your_groq_key_here
"""

import os

ENV = os.getenv("MINIRAG_ENV", "local").lower()  # default = local

# ── Local config (Ollama) 
LOCAL_CONFIG = {
    "env":             "local",
    "embed_model":     "nomic-embed-text",   # ollama pull nomic-embed-text
    "llm_model":       "llama3.2",           # ollama pull llama3.2
    "ollama_base_url": "http://localhost:11434",
}

# ── Production config (Groq) 
# Groq's free tier gives you fast inference on open models.
# For embeddings, Groq doesn't offer an embedding API,
# so we use a lightweight local sentence-transformers fallback
# OR you can use Ollama on Render too (see note below).
#
# Best free stack on Render:
#   Embeddings → nomic-embed-text via ollama  (if you run ollama on Render)
#             OR sentence-transformers         (pure Python, no server needed)
#   LLM        → llama3-8b-8192 via Groq      (fast, generous free tier)

PRODUCTION_CONFIG = {
    "env":          "production",
    "embed_model":  "nomic-embed-text",          # via sentence-transformers (no server)
    "llm_model":    "llama3-8b-8192",            # Groq model string
    "groq_api_key": os.getenv("GROQ_API_KEY"),   # set this on Render
    "groq_base_url": "https://api.groq.com/openai/v1",
}

\
CONFIG = LOCAL_CONFIG if ENV == "local" else PRODUCTION_CONFIG


#  print which environment is active 
def show_config():
    print(f"\n MiniRAG Config")
    print(f"   Environment : {CONFIG['env']}")
    print(f"   LLM model   : {CONFIG['llm_model']}")
    print(f"   Embed model : {CONFIG['embed_model']}")
    if ENV == "local":
        print(f"   Ollama URL  : {CONFIG['ollama_base_url']}")
    else:
        key = CONFIG.get("groq_api_key")
        print(f"   Groq key    : {' set' if key else ' NOT SET — add GROQ_API_KEY on Render'}")
    print()


if __name__ == "__main__":
    show_config()