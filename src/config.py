from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_DIR = STORAGE_DIR / "faiss_index"

#DEFAULT_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 4