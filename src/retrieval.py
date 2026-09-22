"""
Retrieval module for the application.
V2:
 - Support multiple retrieval methods: BM25, semantic, hybrid
 - Run similarity search with optional BM25
 - Filter by provider metadata
 - Return top k results
"""

from langchain_core.documents import Document
from typing import List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Retriever:
    """Multi-method retriever supporting BM25, semantic, and hybrid search."""
    
    def __init__(self, vectorstore, bm25_retriever=None):
        """
        Initialize retriever.
        
        Args:
            vectorstore: LangChain vectorstore for semantic search
            bm25_retriever: Optional BM25 retriever for keyword search
        """
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        logger.info("Retriever initialized | has_bm25=%s", self.bm25_retriever is not None)
    
    def retrieve_semantic(self, 
                         query: str,
                         provider: Optional[str] = None,
                         k: int = 5,
                         fetch_k: int = 12) -> List[Document]:
        """
        Semantic (embedding-based) retrieval.
        
        Args:
            query: Search query
            provider: Filter by provider (None or "All" = no filter)
            k: Return top k results
            fetch_k: Fetch this many before filtering
        
        Returns:
            List of relevant documents
        """
        logger.debug("semantic_retrieval | query=%s | provider=%s | k=%s", 
                    query[:50], provider, k)
        
        # Fetch more results than needed
        docs = self.vectorstore.similarity_search(query, k=fetch_k)
        logger.debug("semantic_search returned %s documents", len(docs))
        
        # Filter by provider if specified
        filtered_docs = self._filter_by_provider(docs, provider)
        
        logger.info("semantic_retrieval complete | returned=%s | provider=%s",
                   len(filtered_docs[:k]), provider)
        return filtered_docs[:k]
    
    def retrieve_bm25(self,
                     query: str,
                     provider: Optional[str] = None,
                     k: int = 5,
                     fetch_k: int = 12) -> List[Document]:
        """
        BM25 (keyword-based) retrieval.
        
        Args:
            query: Search query
            provider: Filter by provider
            k: Return top k results
            fetch_k: Fetch this many before filtering
        
        Returns:
            List of relevant documents
        """
        if not self.bm25_retriever:
            logger.warning("BM25 retriever not initialized, falling back to semantic")
            return self.retrieve_semantic(query, provider, k, fetch_k)
        
        logger.debug("bm25_retrieval | query=%s | provider=%s | k=%s",
                    query[:50], provider, k)
        
        # Fetch more results than needed
        docs = self.bm25_retriever.invoke(query)
        logger.debug("bm25_search returned %s documents", len(docs))
        
        # Filter by provider if specified
        filtered_docs = self._filter_by_provider(docs, provider)
        
        logger.info("bm25_retrieval complete | returned=%s | provider=%s",
                   len(filtered_docs[:k]), provider)
        return filtered_docs[:k]
    
    def retrieve_hybrid(self,
                       query: str,
                       provider: Optional[str] = None,
                       k: int = 5,
                       fetch_k: int = 12,
                       alpha: float = 0.5) -> List[Document]:
        """
        Hybrid retrieval combining BM25 and semantic search.
        Uses Reciprocal Rank Fusion (RRF) to combine results.
        
        Args:
            query: Search query
            provider: Filter by provider
            k: Return top k results
            fetch_k: Fetch this many before filtering
            alpha: Weight for semantic results (1-alpha for BM25)
                   0.5 = equal weight, 1.0 = semantic only, 0.0 = BM25 only
        
        Returns:
            List of relevant documents combined from both methods
        """
        if not self.bm25_retriever:
            logger.warning("BM25 retriever not initialized, using semantic only")
            return self.retrieve_semantic(query, provider, k, fetch_k)
        
        logger.debug("hybrid_retrieval | query=%s | provider=%s | k=%s | alpha=%.2f",
                    query[:50], provider, k, alpha)
        
        # Get results from both methods
        bm25_docs = self.bm25_retriever.invoke(query)
        semantic_docs = self.vectorstore.similarity_search(query, k=fetch_k)
        
        logger.debug("hybrid | bm25=%s docs, semantic=%s docs",
                    len(bm25_docs), len(semantic_docs))
        
        # Combine using Reciprocal Rank Fusion
        combined = self._reciprocal_rank_fusion(
            bm25_docs, 
            semantic_docs, 
            alpha=alpha
        )
        
        # Filter by provider
        filtered_docs = self._filter_by_provider(combined, provider)
        
        logger.info("hybrid_retrieval complete | returned=%s | provider=%s",
                   len(filtered_docs[:k]), provider)
        return filtered_docs[:k]
    
    @staticmethod
    def _filter_by_provider(docs: List[Document], 
                           provider: Optional[str] = None) -> List[Document]:
        """
        Filter documents by provider metadata.
        
        Args:
            docs: List of documents to filter
            provider: Provider to filter by (None or "All" = no filter)
        
        Returns:
            Filtered list of documents
        """
        if not provider or provider.lower() == "all":
            return docs
        
        filtered = [
            doc for doc in docs
            if doc.metadata.get("provider", "").lower() == provider.lower()
        ]
        
        logger.debug("filter_by_provider | input=%s | provider=%s | output=%s",
                    len(docs), provider, len(filtered))
        return filtered
    
    @staticmethod
    def _reciprocal_rank_fusion(bm25_docs: List[Document],
                                semantic_docs: List[Document],
                                alpha: float = 0.5,
                                k: int = 60) -> List[Document]:
        """
        Combine BM25 and semantic results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank))
        
        Args:
            bm25_docs: Results from BM25 search
            semantic_docs: Results from semantic search
            alpha: Weight for semantic (1-alpha for BM25)
            k: Parameter for RRF formula (default 60)
        
        Returns:
            Combined and ranked list of documents
        """
        combined_scores = {}
        
        # Score BM25 results
        for rank, doc in enumerate(bm25_docs, 1):
            doc_id = Retriever._get_doc_id(doc)
            rrf_score = (1 - alpha) / (k + rank)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score
        
        # Score semantic results
        for rank, doc in enumerate(semantic_docs, 1):
            doc_id = Retriever._get_doc_id(doc)
            rrf_score = alpha / (k + rank)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score
        
        # Create mapping of doc_id to document
        doc_map = {}
        for doc in bm25_docs + semantic_docs:
            doc_id = Retriever._get_doc_id(doc)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Sort by combined score
        sorted_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return documents in sorted order
        return [doc_map[doc_id] for doc_id, score in sorted_docs]
    
    @staticmethod
    def _get_doc_id(doc: Document) -> str:
        """
        Get unique document ID for deduplication.
        Uses doc_id metadata if available, otherwise uses content hash.
        """
        if "doc_id" in doc.metadata:
            return doc.metadata["doc_id"]
        return hash(doc.page_content)

def format_docs_for_context(docs: List[Document]) -> str:
    """
    Format documents for context in a prompt.
    
    Args:
        docs: List of documents to format
    
    Returns:
        Formatted string with document context
    """
    formatted_docs = []
    for i, doc in enumerate(docs, start=1):
        provider = doc.metadata.get("provider", "Unknown Provider")
        file_name = doc.metadata.get("file_name", "Unknown File")
        page = doc.metadata.get("page", "Unknown Page")
        content = doc.page_content
        
        formatted_doc = (
            f"[Source {i}] Provider: {provider}\n"
            f"File: {file_name}\n"
            f"Page: {page}\n"
            f"Content:\n{content}\n"
        )
        formatted_docs.append(formatted_doc)
    
    logger.debug("format_docs_for_context | docs=%s | total_length=%s",
                len(formatted_docs), sum(len(doc) for doc in formatted_docs))
    return "\n\n".join(formatted_docs)


def format_sources(docs: List[Document]) -> list:
    """
    Format sources for citation in a prompt.
    
    Args:
        docs: List of documents to format as sources
    
    Returns:
        List of source dictionaries with metadata
    """
    sources = []
    for i, doc in enumerate(docs, start=1):
        source = {
            "id": i,
            "provider": doc.metadata.get("provider", "Unknown Provider"),
            "file_name": doc.metadata.get("file_name", "Unknown File"),
            "page": doc.metadata.get("page", "Unknown Page"),
            "source": doc.metadata.get("source", "Unknown Source"),
            "preview": doc.page_content[:300]
        }
        sources.append(source)
    
    logger.debug("format_sources | count=%s", len(sources))
    return sources