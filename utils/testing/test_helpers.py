"""Helper functions for running RAG pipeline tests."""

from typing import Dict, Any
from rag_pipeline import run_rag_pipeline
from utils.helpers.initialize_dependencies import initialize_model, initialize_retriever


def run_test(config: dict, query: str) -> Dict[str, Any]:
    """
    Run single test using rag_pipeline.

    Args:
        config: Loaded configuration dictionary
        query: Question string

    Returns:
        Result dict from rag_pipeline.run_rag_pipeline()
        Contains: question, answer, context, success, error
    """
    # Initialize components from config
    vectorstore = initialize_retriever(config['rag_config']['vectorstore']['path'])
    llm = initialize_model(
        config['rag_config']['llm']['model'],
        config['rag_config']['llm']['temperature']
    )

    # Run RAG pipeline
    result = run_rag_pipeline(
        query=query,
        vectorstore=vectorstore,
        llm=llm,
        prompt_template=config['rag_config']['prompt']['template'],
        num_docs=config['rag_config']['retrieval']['rerank_top_n'],
        retrieval_k=config['rag_config']['retrieval']['initial_k']
    )

    # Add config metadata
    result['config_name'] = config['name']

    return result


def print_result(result: dict, show_context: bool = True) -> None:
    """
    Pretty-print test result to console.

    Args:
        result: Result dictionary from run_test()
        show_context: If True, display context documents
    """
    print(f"\n{'='*80}")
    print(f"Config: {result.get('config_name', 'Unknown')}")
    print(f"Question: {result['question']}")
    print(f"Success: {result.get('success', False)}")

    if result.get('error'):
        print(f"\nError: {result['error']}")
    else:
        print(f"\nAnswer:\n{result['answer']}")

    if show_context and result.get('context'):
        print(f"\nContext Documents ({len(result['context'])} docs):")
        for i, doc in enumerate(result['context'], 1):
            source = doc.metadata.get('source', 'Unknown')
            # Truncate long source paths
            if len(source) > 70:
                source = '...' + source[-67:]
            print(f"  {i}. {source}")

    print(f"{'='*80}\n")


def print_comparison(query: str, results: list) -> None:
    """
    Print side-by-side comparison of multiple test results for the same query.

    Args:
        query: The question being tested
        results: List of result dicts from different configs
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    for result in results:
        config_name = result.get('config_name', 'Unknown')
        print(f"[{config_name}]")

        if result.get('error'):
            print(f"  Error: {result['error']}")
        else:
            # Truncate long answers for comparison view
            answer = result['answer']
            if len(answer) > 250:
                answer = answer[:247] + "..."
            print(f"  Answer: {answer}")
            print(f"  Context docs: {len(result.get('context', []))}")

        print()

    print(f"{'='*80}\n")
