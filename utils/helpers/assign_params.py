from config import (
        CG_MODEL, QA_MODEL, CG_DB_FILE, QA_DB_FILE, CG_PROMPT, 
        QA_PROMPT, CG_DOCS, QA_DOCS, CG_TEMP, QA_TEMP
)
from utils.logging_setup import get_logger

logger = get_logger(__name__)

def assign_params_by_tool(tool_type: str) -> tuple:
    """
    Assign model to use for inference, location of vectorstore, system prompt template, 
    no.of documents to reference, and temperature of the model based on the type of tool. 
    Arguments: 
        - tool_type: string that represents the type of tool user would like to use
    Returns:
        - A tuple of all the parameters mentioned earlier. 
    """
    if tool_type.lower() == "code generation":
        model = CG_MODEL
        db_loc = CG_DB_FILE
        template = CG_PROMPT
        num_docs = CG_DOCS
        temp = CG_TEMP
    else:
        model = QA_MODEL
        db_loc = QA_DB_FILE
        template = QA_PROMPT
        num_docs = QA_DOCS
        temp = QA_TEMP

    logger.info("Hyperparameters assigned")
    return (model, db_loc, template, num_docs, temp)

