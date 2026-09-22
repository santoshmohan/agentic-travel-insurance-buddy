from src.loader import load_provider_documents
from src.vectorstore import build_faiss_index


def main():
    print("Loading provider documents...")
    docs = load_provider_documents()

    if not docs:
        raise ValueError("No supported documents found under data/")

    print(f"Loaded {len(docs)} raw documents/pages")

    vectorstore, chunks = build_faiss_index(docs)

    print(f"Created {len(chunks)} chunks")
    print("FAISS index saved to storage/faiss_index/")


if __name__ == "__main__":
    main()