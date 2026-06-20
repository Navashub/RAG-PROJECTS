"""

  config.py , Environment detector                        
  Reads .env and sets the right models for each env       


HOW IT WORKS:
  1. Loads your .env file
  2. Reads MINIRAG_ENV to know which environment you're in
  3. Builds the correct config for that environment

YOUR .ENV FILES:
  .env.local      → local laptop with Ollama
  .env.production → Render with Groq
  .env.example    → safe template to commit to GitHub
  .env            → the active file (copy from one of the above)
"""

import os

# ─ Load .env file (if it exists) 
# We do this manually so we don't need python-dotenv as a dependency.
# If you prefer, you can: pip install python-dotenv
# and replace this block with: from dotenv import load_dotenv; load_dotenv()

def _load_env_file(path: str = ".env"):
    """Read a .env file and set variables into os.environ."""
    if not os.path.exists(path):
        return  # no .env file — that's fine, use system env vars
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip()
            if key and key not in os.environ:  # don't override system vars
                os.environ[key] = value

_load_env_file()  # load .env on import



ENV = os.getenv("MINIRAG_ENV", "local").lower()

# ── Local config (Ollama) 
LOCAL_CONFIG = {
    "env":             "local",
    "embed_model":     os.getenv("EMBED_MODEL", "nomic-embed-text"),
    "llm_model":       os.getenv("LLM_MODEL",   "llama3.2"),
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
}

# ── Production config (Groq) 
PRODUCTION_CONFIG = {
    "env":            "production",
    "embed_model":    os.getenv("EMBED_MODEL",   "nomic-embed-text-v1.5"),
    "llm_model":      os.getenv("LLM_MODEL",     "llama3-8b-8192"),
    "groq_api_key":   os.getenv("GROQ_API_KEY"),
    "groq_base_url":  "https://api.groq.com/openai/v1",
}

# ─ Active config 
CONFIG = LOCAL_CONFIG if ENV == "local" else PRODUCTION_CONFIG


def show_config():
    print(f"\n  MiniRAG Config")
    print(f"   Environment  : {CONFIG['env']}")
    print(f"   Embed model  : {CONFIG['embed_model']}")
    print(f"   LLM model    : {CONFIG['llm_model']}")
    if ENV == "local":
        print(f"   Ollama URL   : {CONFIG['ollama_base_url']}")
    else:
        key = CONFIG.get("groq_api_key")
        status = " set" if key else " NOT SET — add GROQ_API_KEY to .env or Render"
        print(f"   Groq API key : {status}")
    print()


if __name__ == "__main__":
    show_config()