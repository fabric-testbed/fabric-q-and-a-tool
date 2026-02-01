import requests
from flask import Response

from config import FLASK_SECRET_KEY
from utils.logging_setup import get_logger

logger = get_logger(__name__)

def validate_api_key(user_key: str) -> Response | None:
    """
    Check whether API key passed in by user matched the app's secret key
    Arguments: 
        - user_key: key passed in by user in API call 
    Returns: 
        - On success: nothing is returned 
        - On failure: a Response with status 403 
    """
    if user_key != FLASK_SECRET_KEY:
        return Response("Invalid API key\n", status=403)
    
    logger.info(f"API key validated")
    
def validate_api_params(query: str, tool_type: str) -> Response | tuple:
    """
    Checks whether user passed in a query (required) and tool_type (optional)
    Arguments: 
        - query: result of trying to extract query from user's API call
        - tool_type: result of trying to extract tool_type from user's API call
    Returns: 
        - On success (query exists): the query and tool_type as a tuple 
        - On failture (query doesn't exist): a Response with status 400 
    """
    if not query: 
        logger.info(f"Bad request with no query")
        return Response('No query specified\n', status=400)
    if not tool_type:
        tool_type = 'QA'

    logger.info(f"Query: {query}\nTool type: {tool_type}")

    return query, tool_type

