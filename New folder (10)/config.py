import os
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
for _d in (DATA_DIR, UPLOAD_DIR):
    os.makedirs(_d, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "studybuddy.db")
QDRANT_PATH = os.path.join(DATA_DIR, "qdrant")
QDRANT_URL = os.getenv("SB_QDRANT_URL", "")
CACHE_DIR = os.path.join(DATA_DIR, "llm_cache")

RERANK_MODEL = os.getenv("SB_RERANK_MODEL", "BAAI/bge-reranker-base")
COLLECTION = os.getenv("SB_COLLECTION", "studybuddy")
EMBED_MODEL = os.getenv("SB_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

LLM_PROVIDER = os.getenv("SB_LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.getenv("SB_GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("SB_GROQ_MODEL", "llama-3.3-70b-versatile")
OLLAMA_MODEL = os.getenv("SB_OLLAMA_MODEL", "llama3.1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

OCR_ENGINE = os.getenv("SB_OCR_ENGINE", "easyocr")

CHILD_SIZE = 500
CHILD_OVERLAP = 80
PARENT_MAX = 3000
CANDIDATES = 20
TOP_K_FINAL = 5
TOP_K_SUMMARY = 12
MAX_RETRIES = 2