import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_app import (
    build_vector_store,
    load_vector_store,
    build_qa_components,
    run_query,
)


@pytest.fixture
def fake_documents():
    return [
        Document(page_content="First test chunk", metadata={"source": "test.pdf"}),
        Document(page_content="Second test chunk", metadata={"source": "test.pdf"}),
    ]


def test_build_vector_store_calls_faiss_and_saves(tmp_path, fake_documents):
    db_path = tmp_path / "faiss_index"

    fake_embeddings = MagicMock()

    with patch("rag_app.FAISS") as faiss_cls:
        fake_vs = MagicMock()
        faiss_cls.from_documents.return_value = fake_vs

        vs = build_vector_store(fake_documents, fake_embeddings, str(db_path))

        faiss_cls.from_documents.assert_called_once()
        args, kwargs = faiss_cls.from_documents.call_args
        assert len(args[0]) == len(fake_documents)
        assert kwargs["embedding"] is fake_embeddings

        fake_vs.save_local.assert_called_once_with(str(db_path))
        assert vs is fake_vs


def test_load_vector_store_uses_faiss_load_local(tmp_path):
    db_path = tmp_path / "faiss_index"
    fake_embeddings = MagicMock()

    with patch("rag_app.FAISS") as faiss_cls:
        fake_vs = MagicMock()
        faiss_cls.load_local.return_value = fake_vs

        vs = load_vector_store(str(db_path), fake_embeddings)

        faiss_cls.load_local.assert_called_once_with(
            str(db_path), fake_embeddings, allow_dangerous_deserialization=True
        )
        assert vs is fake_vs


def test_build_qa_components_creates_llm_and_retriever():
    fake_vs = MagicMock()
    fake_retriever = MagicMock()
    fake_vs.as_retriever.return_value = fake_retriever

    with patch("rag_app.ChatGoogleGenerativeAI") as chat_cls:
        fake_llm = MagicMock()
        chat_cls.return_value = fake_llm

        llm, retriever = build_qa_components(fake_vs)

        chat_cls.assert_called_once()
        _, kwargs = chat_cls.call_args
        assert kwargs["model"] == "gemini-2.5-flash"
        assert kwargs["temperature"] == 0

        fake_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 4})
        assert llm is fake_llm
        assert retriever is fake_retriever


def test_run_query_uses_retriever_and_llm(fake_documents):
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="fake answer")

    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = fake_documents

    answer, sources = run_query(fake_llm, fake_retriever, "What is this?")

    fake_retriever.invoke.assert_called_once_with("What is this?")
    fake_llm.invoke.assert_called_once()

    assert answer == "fake answer"
    assert sources == fake_documents

