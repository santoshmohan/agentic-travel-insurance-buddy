import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import DEFAULT_CHAT_MODEL
from src.retrieval import Retriever, format_docs_for_context, format_sources

import logging
import time

logger = logging.getLogger(__name__)
load_dotenv()

SYSTEM_PROMPT ="""
You are a Travel Insurance Buddy, a helpful assistant that answers questions 
using the provided insurance documents.

Rules:
 - Answer only using the reterieved context.
 - Do not make up coverage or exclusions.
 - Be cautious and precise.
 - Always mention the provider, file, and page references when available.
 - This is general information only, not personal financial or claims advice.
"""

def get_llm(model_name:str = DEFAULT_CHAT_MODEL)-> ChatNVIDIA:
    """Get the LLM instance using the specified model."""
    api_key = os.getenv("NVIDIA_API_KEY")
    logger.info("Choosen model %s", model_name)
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not set in the environment variables.")
    
    llm= ChatNVIDIA(
                    model=model_name,
                    api_key=api_key, 
                    temperature=1,
                    top_p=0.95,
                    max_tokens=16384,
                    reasoning_budget=16384,
                    timeout =200)
    logger.info("LLM initialsed successfully")
    return llm

def answer_question(vectorstore,
                    question: str,
                    provider: str = "All",
                    k: int = 5,
                    retrieval_method: str="hybrid",
                    bm25_retrierver=None):
    """
    Answer a question using the vectorstore and retrieved documents.
    
    Args:
        vectorstore: LangChain vectorstore for semantic search
        question: User's question
        provider: Filter by provider ("All" = no filter)
        k: Number of documents to retrieve
        retrieval_method: "semantic", "bm25", or "hybrid"
        bm25_retriever: Optional BM25 retriever for keyword search
    
    Returns:
        Dict with answer, sources, and metadata
    """
    
    logger.info("answer_question called | provider=%s | k=%s | method=%s | question=%s",
                provider, k, retrieval_method, question[:100])

    start_time = time.perf_counter()

    try:
        retriever = Retriever(vectorstore,bm25_retrierver)
        retrieval_start = time.perf_counter()
        match retrieval_method:
            case "bm25":
                docs = retriever.retrieve_bm25(question,provider,k, fetch_k=max(12, k * 3))
            case "semantic":
                docs = retriever.retrieve_semantic(question,provider,k, fetch_k=max(12, k * 3))
            case "hybrid":
                docs = retriever.retrieve_hybrid(question,provider,k, fetch_k=max(12, k * 3))
            case _:
                logging.error("Unknown retrieval method: %s", retrieval_method)
                raise ValueError(f"Unknown retrieval method: {retrieval_method}")
       
        retrieval_time = time.perf_counter() - retrieval_start

        logger.info("retrieval complete | docs=%s | retrieval_time=%.3fs | method=%s",
                    len(docs), retrieval_time, retrieval_method)

        if docs:
            for i, doc in enumerate(docs[:3], start=1):
                metadata = getattr(doc, "metadata", {})
                logger.debug("doc_%s | provider=%s | file=%s",
                             i, metadata.get("provider"), metadata.get("file_name"))

        if not docs:
            logger.warning("No relevant documents found | provider=%s | question=%s",
                           provider, question[:200])
            return {
                "answer": "No relevant documents found for the given question.",
                "context": [],
                "sources": [],
                "retrieval_method": retrieval_method,
                "latency_ms": (time.perf_counter() - start_time) * 1000
            }

        context = format_docs_for_context(docs)
        logger.debug("context built | context_length=%s", len(context))

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",
             "Provider filter: {provider}\n\n"
             "Question: {question}\n\n"
             "Retrieved context:\n{context}\n\n"
             "Give a concise answer with source references.")
        ])

        llm = get_llm()
        chain = prompt | llm

        llm_start = time.perf_counter()
        response = invoke_with_retry(chain, {
            "provider": provider,
            "question": question,
            "context": context
        })
        llm_time = time.perf_counter() - llm_start

        total_time = time.perf_counter() - start_time

        logger.info("LLM response complete | llm_time=%.3fs | total_time=%.3fs",
                    llm_time, total_time)

        return {
            "answer": response.content,
            "sources": format_sources(docs),
            "retrieval_method": retrieval_method,
            "latency_ms": total_time * 1000,
        }

    except Exception:
        logger.exception("answer_question failed | provider=%s | question=%s",
                         provider, question[:200])
        raise
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)

def invoke_with_retry(chain, inputs):
    """Invoke LLM chain with automatic retry on timeout."""
    logger.debug("invoking LLM chain")
    return chain.invoke(inputs)