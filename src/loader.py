"""
    Module for loading provider documents from a specified data directory.
    Scans provider folders
    Loads documents from supported file types (.pdf, .txt, .md)
    Attaches metadata to each document including provider name, file name, document type, and source path.
"""

import os
from pathlib import Path
from typing import List
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.config import DATA_DIR

logger = logging.getLogger(__name__)
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md"]

def infer_doc_types(file_name:str)-> str:
    """Infer the document type based on the file name."""
    logger.info("file type:%s", file_name)
    lower_file_name = file_name.lower()
    if "pds" in lower_file_name:
        return "pdf"
    if "claim" in lower_file_name:
        return "claims"
    return "other"

def load_provider_documents(data_dir:Path=DATA_DIR) -> List[Document]:
    """Load provider documents from the specified data directory."""
    
    if not data_dir.exists():
        raise FileNotFoundError(f"The specified data directory does not exist: {data_dir}")
    
    all_documents: List[Document] = []
    
    provider_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    
    for provider_dir in provider_dirs:
        provider = provider_dir.name
        logger.info("Provider: %s",provider)
        for file_path in provider_dir.glob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_FILE_TYPES:
                continue
            file_name = file_path.name
            doc_type = infer_doc_types(file_name)
            
            if file_path.suffix.lower() == ".pdf":
                logger.info("LOADING PDF file: %s", file_path)
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                logger.info("LOADING PDF file successful")
            else:
                logger.info("LOADING TEXT file: %s", file_path)
                loader = TextLoader(str(file_path))
                docs = loader.load()
                logger.info("LOADING text file successful")
                
            for doc in docs:
                doc.metadata["provider"] = provider
                doc.metadata["file_name"] = file_name
                doc.metadata["doc_type"] = doc_type
                doc.metadata["source"] = str(file_path)
                if "page" not in doc.metadata:
                    doc.metadata["page"] = None

            all_documents.extend(docs)

    return all_documents

def list_available_providers(data_dir: Path = DATA_DIR)-> list[str]:
    """List all available providers in the specified data directory."""
    
    if not data_dir.exists():
        raise FileNotFoundError(f"The specified data directory does not exist: {data_dir}")
    
    provider_dirs = [p.name for p in data_dir.iterdir() if p.is_dir()]
    logger.info("Provider directories : %s", provider_dirs)
    return provider_dirs

def list_available_retrieval_methods() ->List[str]:
    return ["hybrid", "semantic", "bm25"]