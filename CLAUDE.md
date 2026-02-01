# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the FABRIC Q&A tool - a production RAG (Retrieval-Augmented Generation) application for answering questions about FABRIC testbed. The application supports two modes:
- **QA Tool**: General question answering about FABRIC using knowledge base and forums
- **Code Generation Tool**: Generates Python code examples for FABRIC testbed operations

## Tech Stack

- **Framework**: Flask API with single POST endpoint
- **RAG Pipeline**: LangGraph state machine orchestrating retrieve → rerank → generate stages
- **Vector Database**: ChromaDB with `all-mpnet-base-v2` embeddings
- **Reranking**: BAAI/bge-reranker-v2-m3 model
- **LLM Support**: OpenAI (gpt-4o-mini) and Ollama models
- **Key Dependencies**: langchain, sentence-transformers, torch, transformers

## Architecture

### Core RAG Pipeline (`rag_pipeline.py`)
The pipeline uses LangGraph to orchestrate three stages:

1. **Retrieve** (`retrieval_k=20` docs): Fetch documents from ChromaDB using similarity search
2. **Rerank** (`num_docs=6` for QA, `4` for code_gen): Score with BAAI/bge-reranker-v2-m3, keep top-N
3. **Generate**: Send reranked context + prompt to LLM for final response

The pipeline is **model-agnostic** - any LLM with `.invoke()` method (LangChain compatible) can be used.

### Request Flow (`app.py`)
1. Validate API key (X-API-KEY header)
2. Validate request params (query, tool_type)
3. Assign hyperparameters based on tool_type (model, db_loc, template, num_docs, temp)
4. Initialize vectorstore and LLM
5. Execute RAG pipeline
6. Clean and format response
7. Log query-response to custom log

### Configuration (`config.py`)
All configuration is loaded from `.env` file:
- API keys (FLASK_SECRET_KEY, OPEN_AI_SECRET)
- Vector DB paths (QA_DB_FILE, CG_DB_FILE)
- LLM models (QA_MODEL, CG_MODEL)
- System prompts (QA_PROMPT, CG_PROMPT)
- Logging (LOG_DIR)
- Deployment (HOST, PORT)

Tool-specific hyperparameters are hardcoded in `config.py`:
- QA: 6 docs, temp=0.2
- Code Gen: 4 docs, temp=0

### Logging (`utils/logging_setup.py`)
Two separate logs per session (timestamped):
- **Main log**: Standard Python logging with file/line info
- **Query log**: Custom level (25) for formatted query-response pairs

Access via `get_logger(__name__)` in any module.

### Helper Functions (`utils/helpers/`)
- **initialize_dependencies.py**: Initialize LLM, retriever, and reranker tokenizer
- **rerank_helpers.py**: Calculate relevance scores and attach to documents
- **assign_params.py**: Map tool_type → (model, db_loc, template, num_docs, temp)
- **validate_api_call.py**: Validate API key and request parameters
- **print_helpers.py**: Format responses for user (clean markdown, add context sources)

## Development Commands

### Environment Setup
```bash
# Using uv (recommended)
uv sync

# For CUDA support (Grace Hopper or GPU servers)
uv sync --extra cuda
```

### Running the Application
```bash
# Standard run (uses HOST and PORT from .env)
python app.py

# Debug mode with custom settings
python -m flask --app app run --debug --host=0.0.0.0 --port=5000
```

### Testing the API
```bash
# Test QA tool
python utils/testing/test_api.py -q "What is FABRIC?" -t "QA" -k "your-api-key"

# Test code generation tool
python utils/testing/test_api.py -q "Create a slice" -t "code_generation" -k "your-api-key"

# Using environment variable for API key
export API_KEY="your-api-key"
python utils/testing/test_api.py -q "Test query" -t "QA"
```

### Vector Database Management
Vector databases are created/updated using Jupyter notebooks in `notebooks/`:

1. **01_data_preprocessing/**: Clean CSV/PDF source files
2. **02_vectordb_creation/**: Build ChromaDB databases with embeddings
3. **03_analysis/**: Evaluate retrieval quality metrics
4. **04_testing/**: End-to-end RAG pipeline testing

After creating new databases, update `.env` to point to new locations:
```bash
QA_DB_FILE=./data/vectordbs/qa_tool/
CG_DB_FILE=./data/vectordbs/code_gen/
```

## Critical Technical Notes

### GPU Requirement
The sentence-transformer and reranker models require GPU for acceptable performance. Ensure the deployment environment has CUDA support.

### Model-Specific Formatting
Response formatting in `utils/helpers/print_helpers.py` assumes gpt-4o-mini output format. If using different models, adjust formatting logic accordingly.

### Reranker Initialization
`initialize_tokenizer()` in `rerank_helpers.py` loads BAAI/bge-reranker-v2-m3 on every call. For production, consider caching the tokenizer/model globally to avoid repeated initialization overhead.

### LLM Backend Flexibility
To add new LLM backends:
1. Update `initialize_model()` in `utils/helpers/initialize_dependencies.py`
2. Ensure the model has `.invoke(messages)` method (LangChain compatible)
3. Update `assign_params.py` to map tool_type → new model

## File Organization

```
fabric-q-and-a-tool/
├── app.py                    # Flask API entry point
├── rag_pipeline.py           # Core RAG logic (LangGraph state machine)
├── config.py                 # Environment & hyperparameter config
├── utils/
│   ├── logging_setup.py      # Dual logger setup (main + query logs)
│   ├── helpers/
│   │   ├── initialize_dependencies.py  # LLM, retriever, tokenizer init
│   │   ├── rerank_helpers.py           # Document scoring & sorting
│   │   ├── assign_params.py            # Tool-specific hyperparams
│   │   ├── validate_api_call.py        # Request validation
│   │   └── print_helpers.py            # Response formatting
│   └── testing/
│       └── test_api.py       # CLI tool for API testing
├── notebooks/                # VectorDB creation workflow
│   ├── 01_data_preprocessing/
│   ├── 02_vectordb_creation/
│   ├── 03_analysis/
│   └── 04_testing/
└── data/                     # Local data (gitignored)
    ├── raw/                  # Source CSV/PDF files
    ├── processed/            # Cleaned data
    └── vectordbs/            # ChromaDB databases
        ├── qa_tool/
        └── code_gen/
```

## Environment Variables Reference

Required `.env` variables:
- **FLASK_SECRET_KEY**: Flask session secret
- **OPEN_AI_SECRET**: OpenAI API key
- **QA_DB_FILE**: Path to Q&A tool vector database
- **CG_DB_FILE**: Path to code generation tool vector database
- **QA_MODEL**: LLM model name for Q&A (e.g., "gpt-4o-mini")
- **CG_MODEL**: LLM model name for code generation
- **QA_PROMPT**: System prompt template with {question} and {context} placeholders
- **CG_PROMPT**: Code generation prompt template
- **LOG_DIR**: Directory for log files
- **HOST**: Server host (e.g., "0.0.0.0")
- **PORT**: Server port (e.g., "5000")

Example system prompts can be found in README.md.
