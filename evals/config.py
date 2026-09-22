from dataclasses import dataclass
from typing import List

@dataclass
class ChunkingConfig:
    """Chunking strategy configuration."""
    chunk_size: int
    overlap: int
    strategy: str  # "simple", "recursive", "semantic"
    
    def __str__(self):
        return f"{self.strategy}_size{self.chunk_size}_overlap{self.overlap}"

@dataclass
class RetrieverConfig:
    """Retriever configuration."""
    method: str  # "bm25", "semantic", "hybrid"
    top_k: int = 5
    fetch_k: int = 20
    
    def __str__(self):
        return f"{self.method}_k{self.top_k}"

# Test configurations
CHUNKING_STRATEGIES = [
    ChunkingConfig(chunk_size=256, overlap=50, strategy="simple"),
    ChunkingConfig(chunk_size=512, overlap=100, strategy="simple"),
    ChunkingConfig(chunk_size=600, overlap=100, strategy="simple"),  # Current
    ChunkingConfig(chunk_size=1024, overlap=200, strategy="simple"),
    ChunkingConfig(chunk_size=256, overlap=50, strategy="recursive"),
]

RETRIEVER_METHODS = [
    RetrieverConfig(method="bm25", top_k=5),
    RetrieverConfig(method="semantic", top_k=5),
    RetrieverConfig(method="hybrid", top_k=5),
]