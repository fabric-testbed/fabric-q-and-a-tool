"""
Smoke tests to verify that critical dependencies import correctly.

These tests catch breaking changes from dependency upgrades (e.g., langchain
major version bumps) without requiring a running server or external services.

Run with: pytest tests/test_imports.py
"""


def test_langchain_core_imports():
    """Verify langchain_core classes used by the app are importable."""
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate


def test_langgraph_imports():
    """Verify langgraph classes used by the app are importable."""
    from langgraph.graph import START, StateGraph


def test_langchain_ollama_imports():
    """Verify langchain_ollama is importable."""
    from langchain_ollama import ChatOllama


def test_langchain_chroma_imports():
    """Verify langchain_chroma is importable."""
    from langchain_chroma import Chroma


def test_langchain_huggingface_imports():
    """Verify langchain_huggingface is importable."""
    from langchain_huggingface import HuggingFaceEmbeddings


def test_document_creation():
    """Verify Document can be instantiated with expected fields."""
    from langchain_core.documents import Document

    doc = Document(page_content="test content", metadata={"source": "test"})
    assert doc.page_content == "test content"
    assert doc.metadata["source"] == "test"


def test_prompt_template():
    """Verify PromptTemplate works with the pattern used in app.py."""
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(
        "Question: {question}\nContext: {context}"
    )
    result = prompt.invoke({"question": "What is FABRIC?", "context": "FABRIC is a testbed."})
    assert "What is FABRIC?" in result.text
    assert "FABRIC is a testbed." in result.text


def test_state_graph_construction():
    """Verify StateGraph can be built with the pattern used in app.py."""
    from typing_extensions import List, TypedDict
    from langchain_core.documents import Document
    from langgraph.graph import START, StateGraph

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", lambda state: {"context": []})
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    assert graph is not None
