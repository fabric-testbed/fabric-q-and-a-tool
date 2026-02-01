#!/usr/bin/env python3
"""
Single Configuration Test Script

Tests a single RAG configuration with multiple queries.

Usage:
    python single_config_test.py [--config PATH] [--no-context]

    --config: Path to YAML config file (default: ../../configs/test_configs/baseline.yaml)
    --no-context: Hide context documents in output
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.testing.yaml_config_loader import load_config
from utils.testing.test_helpers import run_test, print_result
import pandas as pd


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test a single RAG configuration')
    parser.add_argument(
        '--config',
        default='configs/test_configs/baseline.yaml',
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--no-context',
        action='store_true',
        help='Hide context documents in output'
    )
    args = parser.parse_args()

    # ========== CONFIGURATION ==========
    CONFIG_FILE = args.config

    test_queries = [
        "What is FABRIC?",
        "How do I create a slice?",
        "Explain the FablibManager API"
    ]

    SHOW_CONTEXT = not args.no_context
    # ===================================

    # Load configuration
    print("Loading configuration...")
    config = load_config(CONFIG_FILE)

    print(f"\nTesting with: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Model: {config['rag_config']['llm']['model']}")
    print(f"Temperature: {config['rag_config']['llm']['temperature']}")
    print(f"VectorDB: {config['rag_config']['vectorstore']['path']}")
    print(f"Rerank top N: {config['rag_config']['retrieval']['rerank_top_n']} docs")
    print()

    # Run tests for each query
    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Testing query...")

        result = run_test(config, query)
        results.append(result)

        print_result(result, show_context=SHOW_CONTEXT)

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    summary_data = []
    for result in results:
        summary_data.append({
            'Query': result['question'][:50] + '...' if len(result['question']) > 50 else result['question'],
            'Success': result.get('success', False),
            'Context Docs': len(result.get('context', [])),
            'Answer Length': len(result.get('answer', ''))
        })

    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    print()


if __name__ == '__main__':
    main()
