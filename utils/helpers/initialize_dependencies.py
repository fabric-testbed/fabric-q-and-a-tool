from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chat_models import init_chat_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from config import OPEN_AI_SECRET
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
        llm = OllamaLLM(model=model, num_ctx=default_window, temperature = temp)

    logger.info(f"{model} successfully initialized")

    return llm

def initialize_retreiver(db_loc: str):
    """
    Initialize the retreiver with the given vectostore
    Arguments:
        - db_loc: the location of the vectorstore to use
    Returns:
        - vectostore: an initialized instance of the retriever
    """
    embedding_model = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
    vectorstore = Chroma(persist_directory=db_loc, embedding_function=embedding_model)

    logger.info(f"Retriever initialized with {db_loc}")

    return vectorstore

def initialize_tokenizer():
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    tokenizer_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3").to(device)
    tokenizer_model.eval()

    return device, tokenizer, tokenizer_model
