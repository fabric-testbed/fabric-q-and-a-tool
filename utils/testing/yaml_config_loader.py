"""Load and parse YAML test configurations."""

import yaml
import os
from dotenv import load_dotenv
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Load YAML config and resolve ${ENV_VAR} references.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with resolved configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid or env variable not found
    """
    load_dotenv()  # Load .env file

    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Get config file's directory for resolving relative paths
    config_dir = Path(config_path).parent.resolve()

    # Load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve ${VAR_NAME} references in prompt template
    prompt_template = config['rag_config']['prompt']['template']
    if isinstance(prompt_template, str) and prompt_template.startswith('${') and prompt_template.endswith('}'):
        var_name = prompt_template.strip('${}')
        resolved_value = os.getenv(var_name)
        if resolved_value is None:
            raise ValueError(f"Environment variable '{var_name}' not found in .env file")
        config['rag_config']['prompt']['template'] = resolved_value

    # Resolve relative vectordb path relative to config file
    vectordb_path = config['rag_config']['vectorstore']['path']
    if not os.path.isabs(vectordb_path):
        # Relative path - resolve it relative to config file's directory
        resolved_path = (config_dir / vectordb_path).resolve()
        config['rag_config']['vectorstore']['path'] = str(resolved_path)

    # Validate required fields
    _validate_config(config)

    return config


def _validate_config(config: dict) -> None:
    """
    Validate configuration has required fields.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required fields are missing
    """
    required_fields = [
        ('name', config),
        ('rag_config', config),
        ('vectorstore', config.get('rag_config', {})),
        ('llm', config.get('rag_config', {})),
        ('prompt', config.get('rag_config', {})),
        ('retrieval', config.get('rag_config', {})),
    ]

    for field_name, parent_dict in required_fields:
        if field_name not in parent_dict:
            raise ValueError(f"Missing required field: {field_name}")

    # Validate VectorDB path exists
    vectordb_path = config['rag_config']['vectorstore']['path']
    if not os.path.exists(vectordb_path):
        raise FileNotFoundError(f"VectorDB path not found: {vectordb_path}")
