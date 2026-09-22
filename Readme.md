## Travel Insurance buddy

I built a provider-aware travel insurance assistant using Streamlit, LangChain, NVIDIA endpoints, and RAG over PDS/claims documents. It can answer coverage questions, find exclusions, and generate claim evidence checklists with citations.

## High-level flow

```
Provider folders
   ↓
Load PDFs / docs
   ↓
Chunk + metadata tagging
   ↓
FAISS vector store
   ↓
LangChain tools
   ├── search_provider_docs
   ├── coverage_checker
   ├── exclusion_finder
   └── claim_checklist_builder
   ↓
Agent
   ↓
Streamlit chat UI
```

## Installation

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Example Queries
“Does TickInsurance cover lost baggage?”
“What exclusions apply to medical claims under 1Cover?”
“What does 1Cover say about cancellation claims?”
“What claim documents are mentioned for baggage theft?”
“Is trip cancellation covered?”