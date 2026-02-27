import re 
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from langchain_core.documents import Document

from utils.logging_setup import get_logger

logger = get_logger(__name__)

def clean_response(response: dict | str) -> str:
    """
    Cleans the LLM response using the print helper functions 
    Arguments:
        - response: raw response produced by invoking the LLM generation 
    Returns: 
        - res: the answer part of the LLM response, cleaned as needed
    """
    # **** clean up steps currently assume model is gpt-4o-mini ****
    res = ""

    # Get the answer content - handle both string and object with .content property
    answer = response["answer"]
    if isinstance(answer, str):
        answer_content = answer
    else:
        # If it's an object with .content property, use that
        answer_content = answer.content if hasattr(answer, 'content') else str(answer)

    if answer_content[3:11] == "markdown":
        res += remove_last_backtick_block(clean_markdown_ticks(answer_content))
    else:
        res += answer_content

    logger.info("Final response is cleaned")

    return res

def filter_responses_and_add_context(response: str, context: list[Document], tool_type: str) -> str:
    """
    Adds context to LLM answer based on whether LLM could help the user or not
    Arguments: 
        - response: cleaned answer from LLM 
        - context: documents used as reference by LLM 
        - tool_type: type of tool for this run 
    Returns: 
        - response: final response with context (given that LLM was able to help)
    """
    # Load lightweight embedding model for filtering out LLM responses where LLM cannot help 
    filter_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    cant_help_emb = filter_model.encode('I cannot help you with that', convert_to_tensor=True)

    res_emb = filter_model.encode(response, convert_to_tensor=True)
    sim = cos_sim(res_emb, cant_help_emb).item()
    logger.info(f"Similarity score between LLM response and 'cannot help' is {sim}")
     # If similarity is low, we assume LLM could help the user
    if sim < 0.70:
        # Only add the context for FABRIC questions
        context_sources = [doc.metadata.get('title', doc.metadata.get('source', 'unknown')) for doc in context]
        logger.info(f"Context appended to response ({len(context)} sources): {context_sources}")
        response += print_context_list(context, tool_type=tool_type)
    else:
        logger.info(f"Similarity score {sim:.4f} >= 0.70 threshold; LLM cannot help, no context appended")
    
    return response

def clean_markdown_ticks(text: str) -> str:
    """
    Removes the ```markdown in first list and corresponding ``` in last line
    Arguments: 
        - text: text to process
    Returns: 
        - lines: text, with first line removed
    """
    lines = text.splitlines(True)
    if lines:
        lines.pop(0)
        lines.pop(-1)
        
    return "".join(lines)

def print_context_list(contexts: list[Document], tool_type: str) -> str:
    """
    Returns the source of the documents retreieved
    Arguments: 
        - Contexts: the documents retrevied 
    Returns: 
        - Sources: the source of the documents retrieved
    """
    sources_with_urls = []

    for document in contexts:
        key = ''
        # If code generation, get physical file loc
        if tool_type.lower() == "code generation":
            source = os.path.basename(document.metadata['source']).replace("py", "ipynb")
            key = 'url'
        # If QA, get title of article/post
        else:
            source = document.metadata['title']
            key = 'source'

        url = document.metadata[key]
        sources_with_urls.append(f"[{source}]({url})")
        
    final = "\n".join(f"- {link}" for link in sources_with_urls)
        
    return  "\n\n" + "## Sources\n" + final + "\n\n --- \n\n"

def remove_last_backtick_block(content: str) -> str:
    """
    Removes the second of two consecutive triple-backtick blocks 
    (```), if they appear with just a newline between them.
    Arguments: 
        - content: text to be cleaned up 
    Returns:
        - cleaned_text: text with necessary corrections
    """
    pattern = r"```[\t ]*\n```[\t ]*\n?"
    cleaned_text = re.sub(pattern, "```\n", content)

    return cleaned_text
