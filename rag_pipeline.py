"""
RAG Pipeline Module

This module contains the core Retrieval-Augmented Generation (RAG) pipeline logic,
including document retrieval, reranking, and response generation using LangGraph.

The pipeline is model-agnostic and can work with any LLM backend (OpenAI, Ollama, vLLM, etc.)
as long as it provides an `.invoke()` method compatible with LangChain's interface.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, START
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from utils.helpers.rerank_helpers import calculate_document_scores, attach_scores_to_documents
from utils.logging_setup import get_logger


# State definition for LangGraph state machine
class State(TypedDict):
    """
    State TypedDict for the RAG pipeline

    Attributes:
        question: User's query
        context: Retrieved/reranked documents
        answer: LLM-generated response
    """
    question: str
    context: List[Document]
    answer: str


def run_rag_pipeline(
    query: str,
    vectorstore,
    llm,
    prompt_template: str,
    num_docs: int,
    retrieval_k: int = 20,
    enable_reranking: bool = True
) -> dict:
    """
    Execute the complete RAG pipeline: retrieve → rerank → generate

    This function orchestrates the three-stage RAG pipeline using LangGraph:
    1. Retrieve: Fetch top-k documents from vectorstore using similarity search
    2. Rerank: Score and rerank documents using BAAI/bge-reranker-v2-m3, keep top-N
    3. Generate: Send reranked documents + query to LLM for final response

    Arguments:
        query: User's question
        vectorstore: Initialized ChromaDB vectorstore instance
        llm: Initialized LLM instance (must have .invoke() method)
        prompt_template: System prompt template string with {question} and {context} placeholders
        num_docs: Number of documents to keep after reranking (tool-specific: QA=6, CG=4)
        retrieval_k: Number of documents to retrieve initially (default: 20)

    Returns:
        dict with keys:
            - 'answer': str (LLM response content or error message)
            - 'context': List[Document] (reranked documents used for generation)
            - 'question': str (original query for reference)
            - 'success': bool (whether generation succeeded)
            - 'error': str | None (error message if failed, None otherwise)

    Example:
        >>> from rag_pipeline import run_rag_pipeline
        >>> from utils.helpers.initialize_dependencies import initialize_model, initialize_retriever
        >>>
        >>> vectorstore = initialize_retriever('./data/vectordbs/qa_tool/')
        >>> llm = initialize_model('gpt-4o-mini', 0.2)
        >>>
        >>> result = run_rag_pipeline(
        ...     query="What is FABRIC?",
        ...     vectorstore=vectorstore,
        ...     llm=llm,
        ...     prompt_template="Answer: {question}\\n\\nContext: {context}",
        ...     num_docs=6
        ... )
        >>>
        >>> if result['success']:
        ...     print(result['answer'])
    """
    logger = get_logger(__name__)

    # Define pipeline stage functions as closures to capture dependencies
    # This allows them to access vectorstore, llm, num_docs, etc. from outer scope

    def retrieve(state: State) -> dict:
        """
        Retrieves relevant documents from vectorstore using similarity search

        Arguments:
            state: Current state of the LLM application

        Returns:
            dict with 'context' key containing retrieved documents
        """
        k = retrieval_k if enable_reranking else num_docs
        retrieved_docs = vectorstore.similarity_search(state["question"], k=k)
        logger.info(f"{k} documents retrieved from vectorstore")
        return {"context": retrieved_docs}

    def rerank(state: State) -> dict:
        """
        Reranks the retrieved documents based on relevance to the query

        Uses BAAI/bge-reranker-v2-m3 model to score query-document pairs,
        then sorts by score and keeps top-N documents.

        Arguments:
            state: Current state of the LLM application

        Returns:
            dict with 'context' key containing top-N reranked documents
        """
        query = state["question"]
        docs = state["context"]
        pairs = [(query, doc.page_content) for doc in docs]

        # Calculate relevance scores and attach to documents
        scores = calculate_document_scores(pairs)
        reranked_docs = attach_scores_to_documents(scores, docs)

        logger.info(f"Documents reranked, keeping top {num_docs}")
        return {"context": reranked_docs[:num_docs]}

    def generate(state: State) -> dict:
        """
        Generates the response to user query using LLM

        Constructs prompt from template with reranked documents as context,
        then invokes LLM to generate final response.

        Arguments:
            state: Current state of the LLM application

        Returns:
            dict with 'answer' key containing LLM response object
        """
        prompt = PromptTemplate.from_template(prompt_template)
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response}

    # Execute the RAG pipeline with error handling
    try:
        # Build LangGraph state machine
        graph_builder = StateGraph(State)

        # Register nodes and define execution flow
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_edge(START, "retrieve")

        if enable_reranking:
            graph_builder.add_node("rerank", rerank)
            graph_builder.add_edge("retrieve", "rerank")

        # Compile and execute graph
        graph = graph_builder.compile()
        rag_state = graph.invoke({"question": query})

        # Execute generate stage separately (maintained for consistency with original)
        final_state = generate(rag_state)
        logger.info("LLM inference successful")

        # Extract response content (handle both string and object responses)
        answer_obj = final_state['answer']
        if isinstance(answer_obj, str):
            answer_text = answer_obj
        elif hasattr(answer_obj, 'content'):
            answer_text = answer_obj.content
        else:
            # Fallback: convert to string
            answer_text = str(answer_obj)

        # Return success result
        return {
            'answer': answer_text,
            'context': rag_state['context'],
            'question': query,
            'success': True,
            'error': None
        }

    except Exception as e:
        # Log error and return failure result
        logger.error(f"RAG pipeline failed: {str(e)}")
        return {
            'answer': "(No response provided)",
            'context': [],
            'question': query,
            'success': False,
            'error': str(e)
        }
