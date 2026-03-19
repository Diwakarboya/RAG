import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_app import (
    build_vector_store,
    load_vector_store,
    build_qa_components,
    run_query,
    load_documents,
    GoogleGenerativeAIEmbeddings,
)


load_dotenv()


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def main():
    st.set_page_config(page_title="PDF RAG with Gemini", page_icon="📄", layout="wide")

    st.title("PDF RAG with Gemini 2.5 Flash")
    st.write("Upload a PDF, build a local FAISS index, and ask questions about it.")

    if not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "GOOGLE_API_KEY is not set. Please add it to your .env file "
            "or environment and restart the app."
        )
        return

    with st.sidebar:
        st.header("1. Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        default_db_path = "faiss_index"
        db_path = st.text_input(
            "Vector DB path (folder)",
            value=default_db_path,
            help="Local directory where the FAISS index will be stored/loaded.",
        )

        rebuild = st.checkbox(
            "Rebuild index from PDF",
            value=False,
            help="If checked, re-embeds and rebuilds the FAISS index from the uploaded PDF.",
        )

        build_clicked = st.button("Build / Load Index")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.llm = None
        st.session_state.retriever = None
        st.session_state.current_db_path = None

    status_placeholder = st.empty()

    if build_clicked:
        if not uploaded_file:
            st.warning("Please upload a PDF first.")
        else:
            try:
                status_placeholder.info("Saving uploaded PDF...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                status_placeholder.info("Loading and chunking document...")
                docs = load_documents(tmp_path)

                embeddings = get_embeddings()

                if rebuild or not os.path.isdir(db_path):
                    status_placeholder.info("Building FAISS index (this may take a bit)...")
                    vector_store = build_vector_store(docs, embeddings, db_path)
                else:
                    status_placeholder.info("Loading existing FAISS index from disk...")
                    vector_store = load_vector_store(db_path, embeddings)

                llm, retriever = build_qa_components(vector_store)

                st.session_state.vector_store = vector_store
                st.session_state.llm = llm
                st.session_state.retriever = retriever
                st.session_state.current_db_path = db_path

                status_placeholder.success("Index ready. You can now ask questions.")
            except Exception as e:
                status_placeholder.error(f"Error while building/loading index: {e}")

    st.header("2. Ask a question")
    question = st.text_input("Your question", value="")

    if st.button("Ask") and question:
        if not st.session_state.vector_store or not st.session_state.llm:
            st.warning("Please upload a PDF and build/load the index first.")
        else:
            try:
                with st.spinner("Thinking with Gemini 2.5 Flash..."):
                    answer, sources = run_query(
                        st.session_state.llm, st.session_state.retriever, question
                    )

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Source chunks")
                for i, doc in enumerate(sources, start=1):
                    st.markdown(f"**Chunk {i}** - {doc.metadata.get('source', 'uploaded_pdf')}")
                    st.write(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))
                    st.markdown("---")
            except Exception as e:
                st.error(f"Error during question answering: {e}")


if __name__ == "__main__":
    main()

