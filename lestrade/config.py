import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index/index.faiss")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EXTERNAL_MODELS = os.getenv("EXTERNAL_MODELS", "gpt-4o,gpt-4o-mini,gpt-4-turbo").split(",")
KB_DIRS = os.getenv("KB_DIRS", "")
WATCH_EXT = os.getenv("WATCH_EXT", ".txt,.md").split(",")
WATCH_INTERVAL = int(os.getenv("WATCH_INTERVAL", "300"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")

HTML_FORMAT = os.getenv("HTML_FORMAT", "true").lower() in ("1", "true", "yes")

FALLBACK_MESSAGE_EN = os.getenv(
    "FALLBACK_MESSAGE_EN",
    "I couldn't find relevant information in my knowledge base to answer your question. "
    "Please try rephrasing or ask a different question.",
)
FALLBACK_MESSAGE_ZH = os.getenv(
    "FALLBACK_MESSAGE_ZH",
    "我没有找到相关信息来回答您的问题。请尝试换一种方式提问，或咨询其他问题。",
)

CHUNKING_PLUGIN = os.getenv(
    "LESTRADE_CHUNKING",
    "",
)
RETRIEVAL_PLUGIN = os.getenv(
    "LESTRADE_RETRIEVAL",
    "",
)
RESPONSE_PLUGIN = os.getenv(
    "LESTRADE_RESPONSE",
    "",
)
INGESTION_PLUGIN = os.getenv(
    "LESTRADE_INGESTION",
    "",
)
