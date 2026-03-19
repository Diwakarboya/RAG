import argparse
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document path does not exist: {path}")
    if not path.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported. Please provide a .pdf file.")
    return PyPDFLoader(path).load()


def build_vector_store(documents, embeddings: GoogleGenerativeAIEmbeddings, db_path: str) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    vector_store.save_local(db_path)
    return vector_store


def load_vector_store(db_path: str, embeddings: GoogleGenerativeAIEmbeddings) -> FAISS:
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)


def build_qa_components(vector_store: FAISS):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return llm, retriever


def run_query(llm: ChatGoogleGenerativeAI, retriever, query: str):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [
        (
            "system",
            "You are a helpful assistant. Answer the user's question "
            "using ONLY the provided context. If the answer is not in the "
            "context, say you don't know.",
        ),
        (
            "user",
            f"Context:\n{context}\n\nQuestion: {query}",
        ),
    ]

    response = llm.invoke(messages)
    answer = response.content
    return answer, docs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple LangChain+FAISS RAG app using OpenAI ChatGPT."
    )
    parser.add_argument(
        "document",
        help="Path to a PDF document to index (e.g. C:\\RAG\\monopoly.pdf)",
    )
    parser.add_argument(
        "--db-path",
        default="faiss_index",
        help="Directory on disk to store/load the FAISS vector DB (default: faiss_index)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding the FAISS index from the PDF even if a local index exists.",
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Single question to ask. If omitted, you'll be prompted interactively.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Create a .env file with GOOGLE_API_KEY=your_key."
        )

    args = parse_args()

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    if args.rebuild_index or not os.path.isdir(args.db_path):
        print(f"Loading document from {args.document} ...")
        docs = load_documents(args.document)

        print("Splitting into chunks and building FAISS index ...")
        vector_store = build_vector_store(docs, embeddings, args.db_path)
    else:
        print(f"Loading existing FAISS index from '{args.db_path}' ...")
        vector_store = load_vector_store(args.db_path, embeddings)

    print("Creating retrieval-augmented generation components ...")
    llm, retriever = build_qa_components(vector_store)

    if args.query:
        answer, sources = run_query(llm, retriever, args.query)
        print("\nAnswer:\n" + answer)
        print("\nSources:")
        for i, doc in enumerate(sources, start=1):
            print(
                f"[{i}] {doc.metadata.get('source', 'unknown')} "
                f"(chunk length={len(doc.page_content)})"
            )
    else:
        print("\nRAG ready. Type your questions (blank line to exit).\n")
        while True:
            query = input("Question: ").strip()
            if not query:
                break
            answer, sources = run_query(llm, retriever, query)
            print("\nAnswer:\n" + answer)
            print("\nSources:")
            for i, doc in enumerate(sources, start=1):
                print(
                    f"[{i}] {doc.metadata.get('source', 'unknown')} "
                    f"(chunk length={len(doc.page_content)})"
                )
            print()


if __name__ == "__main__":
    main()