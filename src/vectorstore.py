from pathlib import Path
from typing import List
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    INDEX_DIR, 
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, 
    DEFAULT_EMBEDDING_MODEL
)

logging = logging.getLogger(__name__)

def get_embeddings(model_name:str = DEFAULT_EMBEDDING_MODEL)-> HuggingFaceEmbeddings:
    """Get embeddings using the specified model."""
    logging.info("Getting Embeddings for model: %s", model_name)
    embeddings =  HuggingFaceEmbeddings(model_name=model_name)
    logging.info("Get Embeddings for model successful")
    return embeddings

def split_documents(docs:List[Document],
                    chunk_size:int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap:int = DEFAULT_CHUNK_OVERLAP)-> List[Document]:
    """Split documents into smaller chunks."""
    logging.info("Splittig text based on chunksize %i and with overlap %i", chunk_size,chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)

def build_faiss_index(docs:List[Document],
                      embedding_model:str = DEFAULT_EMBEDDING_MODEL,
                      chunk_size:int = DEFAULT_CHUNK_SIZE,
                      chunk_overlap:int = DEFAULT_CHUNK_OVERLAP,
                      index_dir:Path = INDEX_DIR
                      )-> FAISS:
    """Build a FAISS index from the provided documents and embeddings."""
    logging.info("Building FAISS index...")
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = get_embeddings(embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    logging.info("Building FAISS index successful")
    return vectorstore, chunks

def load_faiss_index(index_dir:Path = INDEX_DIR,
                     embedding_model:str = DEFAULT_EMBEDDING_MODEL)-> FAISS:
    """Load a FAISS index from the specified directory."""
    logging.info("Loading FAISS index...")
    if not index_dir.exists():
        raise FileNotFoundError(f"The specified index directory does not exist: {index_dir}")
    
    embeddings = get_embeddings(embedding_model)
    vectorstore = FAISS.load_local(str(index_dir), 
                                   embeddings,
                                   allow_dangerous_deserialization=True)
    logging.info("Loading FAISS index successful")   
    return vectorstore