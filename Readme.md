# Travel Insurance Buddy

A provider-aware travel insurance assistant built with Streamlit and a retrieval-augmented generation pipeline over policy documents and claims content.

## What the project does

The app loads policy PDFs and text documents from provider folders under `data/`, builds a FAISS vector index, and lets a user ask questions about coverage, exclusions, and claims. The retrieval layer supports:

- semantic search using embeddings
- BM25 keyword search
- hybrid retrieval combining both

The app is designed to answer questions such as:

- Does TickInsurance cover lost baggage?
- What is the medical evacuation limit for TickInsurance?
- Are pre-existing conditions covered?
- What exclusions apply to a claim?

## Current project structure

```text
.
├── app.py                     # Streamlit UI
├── ingest.py                  # Index build helper
├── Readme.md                  # Project overview
├── requirement.txt            # Python dependencies
├── data/                      # Provider data folders
├── eval/                      # Legacy compatibility import package
├── evals/                     # Evaluation utilities and metrics
├── golden-dataset/            # Golden dataset JSON
├── src/
│   ├── chains.py              # RAG answer pipeline
│   ├── config.py              # App configuration
│   ├── loader.py              # Document loading and provider discovery
│   ├── logging_config.py      # Logging setup
│   ├── retrieval.py           # Retriever implementation and formatting helpers
│   └── vectorstore.py         # FAISS index + embeddings setup
├── storage/
│   └── faiss_index/           # Built FAISS index artifacts
├── tests/
│   ├── test_rag_evals.py      # End-to-end RAG evaluation tests
│   └── test_retrieval_evals.py
└── .venv/
```

## High-level flow

```text
Provider documents in data/
   ↓
Document loading and metadata tagging
   ↓
Chunking + embedding generation
   ↓
FAISS vector store
   ↓
Retriever (semantic / bm25 / hybrid)
   ↓
LLM answer generation
   ↓
Streamlit chat interface
```

## Features

- provider-aware filtering in the app
- configurable retrieval method from the sidebar
- document metadata for provider, file, type, and source
- FAISS-backed storage for efficient similarity search
- evaluation harness for retrieval and RAG quality
- golden dataset used for repeatable benchmarking

## Local setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

If you are using the project’s current environment, the commands used here are:

```bash
source .venv/bin/activate
python -m pytest
```

## Run the app

```bash
source .venv/bin/activate
streamlit run app.py
```

## Run the ingestion/build step

```bash
source .venv/bin/activate
python ingest.py
```

This builds the FAISS index from the documents in `data/` and saves it under `storage/faiss_index/`.

## Evaluation

The project includes retrieval and end-to-end evaluation tests under `tests/` and metric utilities under `evals/`.

Run the evaluation suite with:

```bash
source .venv/bin/activate
python -m pytest tests/test_retrieval_evals.py -q
python -m pytest tests/test_rag_evals.py -q
```

The evaluation flow measures:

- NDCG@5
- MRR
- precision@5
- recall@5
- context relevance
- faithfulness

## Notes

This project is a local research and evaluation prototype for provider-specific insurance Q&A. It is not a production claims advisor and should not be treated as a substitute for the full policy wording or legal/financial advice.