import argparse
import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


def load_documents(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f\"Document path does not exist: {path}\")
    return TextLoader(path, encoding=\"utf-8\").load()


def build_vector_store(documents) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[\"\\n\\n\", \"\\n\", \". \", \" \", \"\"],
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=\"text-embedding-3-small\")
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return vector_store


def build_qa_chain(vector_store: FAISS):
    llm = ChatOpenAI(model=\"gpt-4.1-mini\", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={\"k\": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type=\"stuff\",
    )
    return qa_chain


def run_query(qa_chain, query: str):
    result = qa_chain.invoke({\"query\": query})
    answer = result[\"result\"]
    sources = result.get(\"source_documents\", [])
    return answer, sources


def parse_args():
    parser = argparse.ArgumentParser(
        description=\"Simple LangChain+FAISS RAG app using OpenAI ChatGPT.\"
    )
    parser.add_argument(
        \"document\",
        help=\"Path to a text/markdown document to index (e.g. docs/my_doc.txt)\",
    )
    parser.add_argument(
        \"-q\",
        \"--query\",
        help=\"Single question to ask. If omitted, you'll be prompted interactively.\",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    if not os.getenv(\"OPENAI_API_KEY\"):
        raise RuntimeError(
            \"OPENAI_API_KEY is not set. Create a .env file with OPENAI_API_KEY=your_key.\"
        )

    args = parse_args()

    print(f\"Loading document from {args.document} ...\")
    docs = load_documents(args.document)

    print(\"Splitting into chunks and building FAISS index ...\")
    vector_store = build_vector_store(docs)

    print(\"Creating retrieval-augmented generation chain ...\")
    qa_chain = build_qa_chain(vector_store)

    if args.query:
        answer, sources = run_query(qa_chain, args.query)
        print(\"\\nAnswer:\\n\" + answer)
        print(\"\\nSources:\")
        for i, doc in enumerate(sources, start=1):
            print(f\"[{i}] {doc.metadata.get('source', 'unknown')} (chunk length={len(doc.page_content)})\")
    else:
        print(\"\\nRAG ready. Type your questions (blank line to exit).\\n\")
        while True:
            query = input(\"Question: \").strip()
            if not query:
                break
            answer, sources = run_query(qa_chain, query)
            print(\"\\nAnswer:\\n\" + answer)
            print(\"\\nSources:\")
            for i, doc in enumerate(sources, start=1):
                print(f\"[{i}] {doc.metadata.get('source', 'unknown')} (chunk length={len(doc.page_content)})\")
            print()


if __name__ == \"__main__\":
    main()

