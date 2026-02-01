# FABRIC VectorDB Development Notebooks

This directory contains Jupyter notebooks for developing, analyzing, and testing the vector databases used by the FABRIC Q&A Tool. These notebooks support the data pipeline that powers the RAG (Retrieval Augmented Generation) system.

## Directory Structure

```
notebooks/
├── README.md                      # This file
├── 01_data_preprocessing/         # Data cleaning and preparation
├── 02_vectordb_creation/          # ChromaDB building and configuration
├── 03_analysis/                   # Retrieval quality and embedding analysis
└── 04_testing/                    # End-to-end RAG pipeline testing
```

## Purpose of Each Directory

### 01_data_preprocessing/
**Purpose**: Clean and prepare raw data (CSV, PDF files) for vectorization

**Typical workflows**:
- Extract text from PDFs (FABRIC documentation, manuals)
- Parse CSV files (forum posts, FAQs, knowledge base exports)
- Clean and normalize text (remove duplicates, fix encoding, standardize formatting)
- Split documents into chunks appropriate for embedding
- Generate metadata for each document chunk

**Output**: Processed data ready for embedding → saved to `data/processed/`

### 02_vectordb_creation/
**Purpose**: Build and configure ChromaDB vector databases from processed data

**Typical workflows**:
- Initialize ChromaDB collections for QA and Code Generation tools
- Generate embeddings using HuggingFace models (all-mpnet-base-v2)
- Configure embedding dimensions and distance metrics
- Batch insert document embeddings into ChromaDB
- Create indexes for efficient similarity search
- Validate database integrity and completeness

**Output**: ChromaDB databases → saved to `data/vectordbs/qa_tool/` and `data/vectordbs/code_gen/`

### 03_analysis/
**Purpose**: Evaluate retrieval quality and analyze embedding performance

**Typical workflows**:
- Test similarity search with sample queries
- Analyze retrieval precision/recall metrics
- Visualize embedding space (t-SNE, UMAP plots)
- Compare different embedding models or chunking strategies
- Identify gaps in knowledge base coverage
- Benchmark retrieval latency and throughput

**Output**: Analysis reports, visualizations, performance metrics

### 04_testing/
**Purpose**: End-to-end testing of the RAG pipeline

**Typical workflows**:
- Test full RAG workflow: retrieve → rerank → generate
- Validate reranking model performance (BAAI/bge-reranker-v2-m3)
- Test LLM response quality with different prompt templates
- Compare QA vs Code Generation tool performance
- Integration tests with production Flask app
- Regression tests when updating vector databases

**Output**: Test results, validation reports, example Q&A pairs

## Workflow: Creating/Updating Vector Databases

Follow this numbered workflow when creating or updating vector databases:

### 1. Data Preprocessing (01_data_preprocessing/)
```bash
# Place raw files in data/raw/
cp /path/to/fabric_docs.pdf data/raw/
cp /path/to/forum_posts.csv data/raw/

# Run preprocessing notebooks to clean and prepare data
# Output goes to data/processed/
```

### 2. VectorDB Creation (02_vectordb_creation/)
```bash
# Run database creation notebooks
# Reads from data/processed/
# Creates ChromaDB collections in data/vectordbs/
```

### 3. Analysis (03_analysis/)
```bash
# Validate retrieval quality
# Test sample queries
# Identify coverage gaps
```

### 4. Testing (04_testing/)
```bash
# Run end-to-end RAG pipeline tests
# Verify integration with Flask app
```

### 5. Update Environment Variables
Once vector databases are created, update your `.env` file:
```bash
QA_DB_FILE=./data/vectordbs/qa_tool/
CG_DB_FILE=./data/vectordbs/code_gen/
```

## Dependencies

### Production Dependencies (defined in pyproject.toml)
- `chromadb` - Vector database
- `sentence-transformers` - HuggingFace embeddings
- `langchain` - RAG framework
- `transformers` - Reranking models
- `torch` - PyTorch for model inference

### Additional Development Dependencies (for notebooks)
You may need to install additional packages for data analysis:

```bash
pip install jupyter notebook pandas matplotlib seaborn plotly scikit-learn nltk pypdf2
```

Or create a separate requirements file for development:
```bash
# notebooks/requirements-dev.txt
jupyter>=1.0.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.3.0
nltk>=3.8.0
pypdf2>=3.0.0
```

## Environment Setup

### 1. Install Dependencies
```bash
# Using uv (recommended)
uv sync

# For CUDA support (Grace Hopper or GPU servers)
uv sync --extra cuda
```

### 2. Start Jupyter
```bash
jupyter notebook notebooks/
```

### 3. GPU Requirements
Note: The sentence-transformer module requires a GPU for optimal performance. Ensure you're running on a machine with CUDA support.

## Environment Variables

Your notebooks should reference the same environment variables as the production app. See the main `.env` file or the main README.md for the complete list.

**Key variables for notebooks**:
```bash
# Vector database locations
QA_DB_FILE=./data/vectordbs/qa_tool/
CG_DB_FILE=./data/vectordbs/code_gen/

# LLM configuration (for testing)
OPEN_AI_SECRET=<your-openai-key>
QA_MODEL=gpt-4o-mini
CG_MODEL=gpt-4o-mini

# System prompts (for testing RAG pipeline)
QA_PROMPT="You are an AI Help Desk assistant for FABRIC..."
CG_PROMPT="You are an AI Code Assistant..."
```

## Relationship to Production Pipeline

### Current Status (Development/Alpha Production)
- Notebooks are used for experimentation and creating vector databases
- Generated databases are copied to production environment manually
- Notebooks serve as documentation for the data pipeline

### Future Formalization (Production Pipeline)
When ready to formalize the data pipeline for production:
1. Extract reusable logic from notebooks into Python modules
2. Create `scripts/` directory with production data pipeline scripts
3. Add automated tests for data processing steps
4. Set up CI/CD for automatic vector database updates
5. Notebooks remain as documentation and development sandbox

## Best Practices

1. **Version Control**: Commit notebooks with cleared outputs (use `jupyter nbconvert --clear-output`)
2. **Data Isolation**: Keep all data files in `data/` directory (gitignored)
3. **Reproducibility**: Document random seeds, model versions, and hyperparameters in notebooks
4. **Modularity**: Create reusable helper functions for common operations
5. **Documentation**: Add markdown cells explaining each major step
6. **Testing**: Validate outputs at each stage before proceeding to next notebook

## Troubleshooting

### GPU/CUDA Issues
If you encounter GPU errors with sentence-transformers:
```python
# Force CPU mode in notebooks
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

### ChromaDB Persistence Issues
Ensure ChromaDB paths are absolute or relative to project root:
```python
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./data/vectordbs/qa_tool/",
    settings=Settings(anonymized_telemetry=False)
)
```

### Import Errors
If you can't import from `utils/` in notebooks:
```python
import sys
sys.path.append('..')  # Add project root to path
from utils.helpers.rerank_helpers import get_rerank_model
```

## Questions or Issues?

For questions about the notebooks or data pipeline, please refer to the main project documentation or contact the FABRIC Q&A tool development team.
