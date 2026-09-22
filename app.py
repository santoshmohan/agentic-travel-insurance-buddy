import streamlit as stlit
import logging
import warnings
from src.loader import list_available_providers, load_provider_documents, list_available_retrieval_methods
from src.vectorstore import build_faiss_index, load_faiss_index
from src.chains import answer_question
from src.retrieval import format_sources
from src.config import INDEX_DIR
from src.logging_config import setup_logging


setup_logging()
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

stlit.set_page_config(
    page_title="Travel Insurance Buddy",
    page_icon="✈️",
    layout="wide",
)

def init_session_state():
    if "messages" not in stlit.session_state:
        stlit.session_state.messages = [
            {
            "role":"assistant",
            "content":(
                    "Hi, I’m Travel Insurance Buddy. "
                    "Ask me about coverage, exclusions, or claims based on the indexed provider documents."
            )
            }
        ]
    
    if "vectorstore_ready" not in stlit.session_state:
        stlit.session_state.vectorstore_ready=False
    
    if "bm25_retriever" not in stlit.session_state:
        stlit.session_state.bm25_retriever = None

    if "vectorstore" not in stlit.session_state:
        stlit.session_state.vectorstore = None
    
    if "last_sources" not in stlit.session_state:
        stlit.session_state.last_sources = []


def build_or_load_index(rebuild: bool = False):
    if rebuild or not INDEX_DIR.exists():
        docs = load_provider_documents()
        vectorstore, chunks = build_faiss_index(docs)
        stlit.session_state.vectorstore = vectorstore
        stlit.session_state.vectorstore_ready = True
        return len(docs), len(chunks), "rebuilt"

    vectorstore = load_faiss_index()
    stlit.session_state.vectorstore = vectorstore
    stlit.session_state.vectorstore_ready = True
    return None, None, "loaded"

def main():
    logger.info("Starting Travel Insurance Buddy app")
    init_session_state()

    stlit.title("✈️ Travel Insurance Buddy")
    stlit.caption("Provider-aware RAG assistant for travel insurance PDS and claims documents.")

    stlit.warning(
        "This tool provides general information based on uploaded policy documents and "
        "is not personal financial advice or claims advice. Coverage depends on the full "
        "policy wording, exclusions, limits, and your circumstances."
    )

    providers = ["All"] + list_available_providers()
    retrieval_methods = list_available_retrieval_methods()

    with stlit.sidebar:
        stlit.header("Settings")

        selected_provider = stlit.selectbox(
            "Choose provider",
            options=providers,
            index=0,
        )
        
        retrieval_method = stlit.selectbox(
            "Retrieval Method",
            options=retrieval_methods,
            help="hybrid=BM25+semantic, semantic=embedding-based, bm25=keyword-based"
        )

        if stlit.button("Load / Build Knowledge Base", use_container_width=True):
            with stlit.spinner("Preparing vector index..."):
                doc_count, chunk_count, mode = build_or_load_index(rebuild=not INDEX_DIR.exists())

            if mode == "rebuilt":
                stlit.success(f"Index built. Loaded {doc_count} docs/pages and {chunk_count} chunks.")
            else:
                stlit.success("Index loaded from storage.")

        if stlit.button("Rebuild Index", use_container_width=True):
            with stlit.spinner("Rebuilding vector index..."):
                doc_count, chunk_count, mode = build_or_load_index(rebuild=True)
            stlit.success(f"Rebuilt index with {doc_count} docs/pages and {chunk_count} chunks.")

        stlit.markdown("---")
        stlit.markdown("### Available providers")
        if len(providers) > 1:
            for provider in providers[1:]:
                stlit.write(f"- {provider}")
        else:
            stlit.write("No provider folders found in `data/`.")

    for msg in stlit.session_state.messages:
        with stlit.chat_message(msg["role"]):
            stlit.markdown(msg["content"])

    user_input = stlit.chat_input("Ask about coverage, exclusions, or claims...")

    if user_input:
        stlit.session_state.messages.append({"role": "user", "content": user_input})

        with stlit.chat_message("user"):
            stlit.markdown(user_input)

        with stlit.chat_message("assistant"):
            if not stlit.session_state.vectorstore_ready:
                answer = "Please click **Load / Build Knowledge Base** firstlit."
                stlit.markdown(answer)
                stlit.session_state.last_sources = []
            else:
                with stlit.spinner("Searching policy documents..."):
                    print("***********")
                    print(selected_provider)
                    print("***********")
                    logger.info("User submitted question=%s | provider=%s | method=%s | k=%s", user_input, selected_provider, retrieval_method, 4)
                    result = answer_question(
                        vectorstore=stlit.session_state.vectorstore,
                        question=user_input,
                        provider=selected_provider,
                        k=4,
                        retrieval_method=retrieval_method,
                        bm25_retrierver=stlit.session_state.bm25_retriever
                    )
                    logger.info("Question answered | method=%s | latency_ms=%.1f",
                       retrieval_method, result.get("latency_ms", 0))
                answer = result["answer"]
                stlit.markdown(answer)

                stlit.session_state.last_sources = result["sources"]

                if stlit.session_state.last_sources:
                    with stlit.expander("View retrieved sources"):
                        for i, src in enumerate(stlit.session_state.last_sources, start=1):
                            stlit.markdown(
                                f"**Source {i}**  \n"
                                f"- Provider: `{src['provider']}`  \n"
                                f"- File: `{src['file_name']}`  \n"
                                f"- Page: `{src['page']}`"
                            )
                            stlit.caption(src["preview"])

        stlit.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()