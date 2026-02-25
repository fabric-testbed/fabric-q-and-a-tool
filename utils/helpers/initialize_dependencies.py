from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chat_models import init_chat_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from config import OPEN_AI_SECRET, OLLAMA_BASE_URL, OLLAMA_VERIFY_SSL
from utils.logging_setup import get_logger

logger = get_logger(__name__)

def initialize_model(model:str, temp: float):
    """
    Intialize the LLM chosen for inference 
    Arguments:
        - model: the model chosen for inference 
        - temp: the temperature setting for the model 
    Returns:
        - llm: an initialized instance of the LLM 
    """
    if model == "gpt-4o-mini":
        openai_secret = OPEN_AI_SECRET
        llm = init_chat_model("gpt-4o-mini", model_provider="openai",
                                openai_api_key=openai_secret, temperature=temp)
    else:
        default_window = 16384
        kwargs = dict(model=model, num_ctx=default_window, temperature=temp)
        if OLLAMA_BASE_URL:
            kwargs['base_url'] = OLLAMA_BASE_URL
        if not OLLAMA_VERIFY_SSL:
            kwargs['client_kwargs'] = {'verify': False}
        llm = OllamaLLM(**kwargs)

    logger.info(f"{model} successfully initialized")

    return llm

def initialize_retriever(db_loc: str):
    """
    Initialize the retriever with the given vectorstore
    Arguments:
        - db_loc: the location of the vectorstore to use
    Returns:
        - vectorstore: an initialized instance of the retriever
    """
    embedding_model = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
    vectorstore = Chroma(persist_directory=db_loc, embedding_function=embedding_model)

    logger.info(f"Retriever initialized with {db_loc}")

    return vectorstore

_tokenizer_cache = None

def initialize_tokenizer():
    global _tokenizer_cache
    if _tokenizer_cache is not None:
        return _tokenizer_cache

    # Load model and tokenizer (runs once, then cached)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    tokenizer_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3").to(device)
    tokenizer_model.eval()

    _tokenizer_cache = (device, tokenizer, tokenizer_model)
    logger.info("Reranker tokenizer and model loaded and cached")
    return _tokenizer_cache
