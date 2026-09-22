"""
Shared fixtures for all eval tests.
"""
import json
import logging
import pytest
from pathlib import Path
from src.vectorstore import get_embeddings, build_faiss_index
from config import CHUNKING_STRATEGIES
from src.loader import load_provider_documents
from src.config import INDEX_DIR
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
K = 5  # Number of docs to retrieve


@pytest.fixture(scope="session")
def golden_dataset():
    """Load golden dataset once per session."""
    with open(GOLDEN_DATASET_PATH) as f:
        data = json.load(f)
    logger.info("Loaded golden dataset | queries=%s", len(data))
    return data


@pytest.fixture(scope="session")
def raw_documents():
    """Load raw documents once per session."""
    from src.loader import load_documents  # Your existing loader
    docs = load_documents()
    logger.info("Loaded raw documents | count=%s", len(docs))
    return docs


@pytest.fixture(scope="session")
def embedding_model():
    """Shared embedding model."""
    from config import DEFAULT_EMBEDDING_MODEL
    return get_embeddings(DEFAULT_EMBEDDING_MODEL)


@pytest.fixture(scope="session", params=list(CHUNKING_STRATEGIES.keys()))
def chunked_vectorstore(request, raw_documents, embedding_model):
    """
    Parameterized fixture: builds a vectorstore for each chunking strategy.
    Tests using this fixture will run once per strategy automatically.
    """
    strategy_name = request.param
    config = CHUNKING_STRATEGIES[strategy_name]
    logger.info("Building vectorstore | chunking_strategy=%s", strategy_name)
    if not INDEX_DIR.exists():
            docs = load_provider_documents(raw_documents)
            vectorstore, chunks = build_faiss_index(
                    docs=docs,
                    embedding_model=embedding_model,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.overlap,
                    index_dir=INDEX_DIR)
   
    return {
        "strategy_name": strategy_name,
        "config": config,
        "vectorstore": vectorstore,
        "chunks": chunks,
    }