import torch 
from langchain_core.documents import Document

from utils.helpers.initialize_dependencies import initialize_tokenizer
from utils.logging_setup import get_logger

logger = get_logger(__name__)

def calculate_document_scores(pairs: tuple) -> list | float:
    """
    Calculates the scores of each document retreived from the vectostore
    Arguments: 
        - paris: tuple containing user query and the content of the document
    Returns: 
        - scores: a list or float(if there's only document) of scores for each document 
    """
    device, tokenizer, tokenizer_model = initialize_tokenizer()

    with torch.no_grad():
        inputs = tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = tokenizer_model(**inputs)
        scores = outputs.logits.squeeze().tolist()
    
    logger.info("Document scores for reranking calculated successfully")

    return scores

def attach_scores_to_documents(scores: float | list, docs: list[Document]) -> list[Document]:
    """
    Attaches the score of each document in the document's metadata
    Arguments:
        - scores: a list or float(if there's only document) of scores for each document 
        - docs: the list of documents 
    Returns:
        - reranked_docs: documents, with scores attached in metadata
    """
    # Attach rerank scores and sort
    if isinstance(scores, float):  # for single doc case
        scores = [scores]

    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = score
    
    reranked_docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
    
    logger.info("Reranking scores attached to documents successfully")

    return reranked_docs