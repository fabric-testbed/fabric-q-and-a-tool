# RAG Pipeline Testing Notebooks

This directory contains interactive Jupyter notebooks for testing different RAG configurations manually.

## Template Pattern

The testing notebooks use a **template pattern** to keep git history clean:

- **`.template.ipynb` files** - Committed to git with example configurations
- **`.ipynb` files** - Your working copies (ignored by git, not committed)

**First time setup:**
```bash
# If working copies don't exist, copy from templates:
cp compare_configs.template.ipynb compare_configs.ipynb
cp single_config_test.template.ipynb single_config_test.ipynb
```

**Benefits:**
- Modify configs and queries in your working copies without cluttering git history
- Template versions preserve the notebook structure for reference
- Similar to `.env.example` and `.env` pattern

## Quick Start

1. **Test a single configuration:**
   ```bash
   jupyter notebook single_config_test.ipynb
   ```
   - Edit `CONFIG_FILE` to choose which config to test
   - Modify `test_queries` list with your questions
   - Run all cells to see results

2. **Compare multiple configurations:**
   ```bash
   jupyter notebook compare_configs.ipynb
   ```
   - Edit `CONFIG_FILES` list to select configs to compare
   - Modify `test_queries` with your test questions
   - Run all cells to see side-by-side comparison

## Available Test Configurations

Located in `../../configs/test_configs/`:

- `baseline.yaml` - Production QA tool configuration (gpt-4o-mini, 6 docs)
- `code_gen_baseline.yaml` - Production code generation configuration (gpt-4o-mini, 4 docs)
- `gpt4o_test.yaml` - Test with GPT-4o (full model)
- `ollama_llama.yaml` - Test with local Ollama Llama 3.1 model
- `num_docs_comparison.yaml` - Compare different numbers of context documents (2 vs 4 vs 6)

## Creating New Test Configurations

1. Copy an existing YAML file from `configs/test_configs/`
2. Modify the configuration parameters:
   - `llm.model` - Model name (e.g., "gpt-4o", "llama3.1")
   - `llm.temperature` - Temperature (0 = deterministic, higher = more creative)
   - `vectorstore.path` - Path to vector database
   - `prompt.template` - System prompt (use `${VAR}` for .env variables)
   - `retrieval.rerank_top_n` - Number of docs after reranking
3. Save with a descriptive name
4. Reference it in the notebooks

## Testing Different Scenarios

### Test Different Models
Change the `llm.model` field:
- OpenAI: `gpt-4o-mini`, `gpt-4o`, etc.
- Ollama: `llama3.1`, `mistral`, etc.

### Test Different VectorDBs
Change the `vectorstore.path` field:
- QA tool: `./data/vectordbs/qa_tool/`
- Code generation: `./data/vectordbs/code_gen/`

### Test Different Prompts
Change the `prompt.template` field:
- Use `${QA_PROMPT}` to reference .env variable
- Or provide a literal string

### Test Different Hyperparameters
Change retrieval parameters:
- `retrieval.initial_k` - Initial retrieval count (default: 20)
- `retrieval.rerank_top_n` - Final docs after reranking (QA: 6, CG: 4)
- `llm.temperature` - Temperature (QA: 0.2, CG: 0)

## Example Workflows

### Workflow 1: Quick Single Config Test
```python
# In single_config_test.ipynb
CONFIG_FILE = "../../configs/test_configs/baseline.yaml"
test_queries = ["What is FABRIC?", "How do I create a slice?"]
# Run all cells
```

### Workflow 2: Model Comparison
```python
# In compare_configs.ipynb
CONFIG_FILES = [
    "../../configs/test_configs/baseline.yaml",      # gpt-4o-mini
    "../../configs/test_configs/gpt4o_test.yaml"     # gpt-4o
]
test_queries = ["What is FABRIC?"]
# Run all cells to see side-by-side comparison
```

### Workflow 3: Hyperparameter Tuning
```python
# Create 3 config files with different rerank_top_n values (2, 4, 6)
# In compare_configs.ipynb
CONFIG_FILES = [
    "../../configs/test_configs/num_docs_2.yaml",
    "../../configs/test_configs/num_docs_4.yaml",
    "../../configs/test_configs/num_docs_6.yaml"
]
# Run all cells
```

## Troubleshooting

### "Config file not found"
- Check that the path to the config YAML is correct
- Paths are relative to the notebook location

### "Environment variable not found"
- Ensure `.env` file exists in project root
- Check that the variable name in the config matches `.env`

### "VectorDB path not found"
- Verify the vector database exists at the specified path
- Check `data/vectordbs/` directory

### Ollama model errors
- Ensure Ollama is running: `ollama serve`
- Pull the model first: `ollama pull llama3.1`
