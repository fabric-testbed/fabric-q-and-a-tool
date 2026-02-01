#!/usr/bin/env python3
"""
Simple test script for making POST requests to app.py Flask endpoint.

Usage:
    python utils/testing/test_api.py --query "What is FABRIC?" --tool-type "QA"
    python utils/testing/test_api.py -q "Generate code" -t "code_generation" -k "your-api-key"
"""

import requests
import json
import argparse
import os
import sys
from typing import Dict, Optional

# Add project root to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config import HOST, PORT


def make_api_request(
    query: str,
    tool_type: str,
    api_key: str,
    host: str,
    port: int
) -> tuple[Optional[Dict], int, Optional[str]]:
    """
    Make POST request to Flask app.

    Args:
        query: Question to ask
        tool_type: Tool type (QA or code_generation)
        api_key: API key for authentication
        host: Server host
        port: Server port

    Returns:
        Tuple of (response_data, status_code, error_message)
    """
    url = f"http://{host}:{port}/"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "tool_type": tool_type
    }

    try:
        response = requests.post(url, json=payload, headers=headers)

        # Try to parse JSON response
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = {"raw_response": response.text}

        return data, response.status_code, None

    except requests.exceptions.ConnectionError:
        return None, 0, f"Connection error: Could not connect to {url}. Is the Flask app running?"
    except requests.exceptions.Timeout:
        return None, 0, "Request timed out"
    except Exception as e:
        return None, 0, f"Unexpected error: {str(e)}"


def print_response(
    response_data: Optional[Dict],
    status_code: int,
    error: Optional[str],
    query: str,
    tool_type: str
) -> None:
    """
    Pretty-print the API response.

    Args:
        response_data: Response JSON data
        status_code: HTTP status code
        error: Error message if any
        query: Original query
        tool_type: Tool type used
    """
    print("\n" + "="*80)
    print("API TEST RESULT")
    print("="*80)

    print(f"\nQuery: {query}")
    print(f"Tool Type: {tool_type}")
    print(f"Status Code: {status_code}")

    if error:
        print(f"\nERROR: {error}")
        print("\nMake sure the Flask app is running:")
        print("  python app.py")
    elif response_data:
        if status_code == 200:
            print(f"\nModel Used: {response_data.get('model_used', 'N/A')}")
            print(f"\nResponse:")
            print("-" * 80)
            print(response_data.get('response', response_data.get('raw_response', 'No response')))
            print("-" * 80)
        else:
            print(f"\nError Response:")
            print(json.dumps(response_data, indent=2))

    print("\n" + "="*80 + "\n")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test script for app.py Flask API endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python utils/testing/test_api.py -q "What is FABRIC?" -t "QA" -k "your-api-key"

  # Using environment variable for API key
  export API_KEY="your-api-key"
  python utils/testing/test_api.py -q "What is FABRIC?" -t "QA"

  # Custom host/port
  python utils/testing/test_api.py -q "Test" -t "QA" -k "key" --host localhost --port 5000
        """
    )

    parser.add_argument(
        '-q', '--query',
        required=True,
        help='Question to ask the API'
    )

    parser.add_argument(
        '-t', '--tool-type',
        required=True,
        choices=['QA', 'code_generation'],
        help='Tool type: QA or code_generation'
    )

    parser.add_argument(
        '-k', '--api-key',
        default=os.environ.get('API_KEY'),
        help='API key for authentication (or set API_KEY env variable)'
    )

    parser.add_argument(
        '--host',
        default=HOST,
        help=f'Server host (default: {HOST})'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=PORT,
        help=f'Server port (default: {PORT})'
    )

    args = parser.parse_args()

    # Validate API key
    if not args.api_key:
        print("ERROR: API key required. Provide via --api-key or API_KEY environment variable.")
        sys.exit(1)

    # Make request
    response_data, status_code, error = make_api_request(
        query=args.query,
        tool_type=args.tool_type,
        api_key=args.api_key,
        host=args.host,
        port=args.port
    )

    # Print results
    print_response(
        response_data=response_data,
        status_code=status_code,
        error=error,
        query=args.query,
        tool_type=args.tool_type
    )

    # Exit with appropriate code
    sys.exit(0 if status_code == 200 else 1)


if __name__ == '__main__':
    main()
