import utils.compat  # must precede any chromadb imports

from flask import (
    Flask, request, render_template, session,
    redirect, url_for, jsonify, flash, send_file, Response
)

from utils.logging_setup import setup_logging
from utils.helpers.validate_api_call import validate_api_key, validate_api_params
from utils.helpers.assign_params import assign_params_by_tool
from utils.helpers.initialize_dependencies import initialize_model, initialize_retriever, initialize_tokenizer
from utils.helpers.print_helpers import clean_response, filter_responses_and_add_context
from config import HOST, PORT
from rag_pipeline import run_rag_pipeline

app = Flask(__name__)

# Set up logging
setup_logging(app)

# Pre-load reranker model into cache so the first request isn't slow
initialize_tokenizer()

@app.route('/', methods=['POST'])
def generate_response() -> str:
    """
    Returns the LLM response for the user query
    Arguments:
        - query: the user question, collected by gradio textbox
        - tool_type: the type of tool, QA or code generation, chosen by user
    Returns:
        - res: a string that consists the output to print for user query (including sources and URLs)
    """
    # Validate API key
    api_key = request.headers.get("X-API-KEY")
    valid = validate_api_key(api_key)
    if valid: return valid 

    # Validate API params 
    data = request.get_json()
    query = data.get('query')
    tool_type = data.get('tool_type')
    query, tool_type = validate_api_params(query, tool_type)

    # Assign hyperparameters based on tool type 
    model, db_loc, template, num_docs, temp = assign_params_by_tool(tool_type)

    # Intialize vectorstore and LLM
    vectorstore = initialize_retriever(db_loc)
    llm = initialize_model(model, temp)

    # Execute RAG pipeline
    result = run_rag_pipeline(
        query=query,
        vectorstore=vectorstore,
        llm=llm,
        prompt_template=template,
        num_docs=num_docs,
        retrieval_k=20
    )

    # Handle pipeline result
    if result['success']:
        response = {'answer': result['answer']}
        context = result['context']
    else:
        response = {'answer': "(No response provided)"}
        context = []
        app.logger.error(f"RAG pipeline error: {result['error']}")

    # Clean and filter the response
    cleaned_response = clean_response(response)
    final_response = filter_responses_and_add_context(cleaned_response, context, tool_type)

    # Add results to the log
    app.logger.custom(f"QUERY: {query}\nRESPONSE: {final_response}\nMODEL: {model}\nTOOL: {tool_type}\n\n -- \n\n")

    return jsonify({
        'query': query,
        'response': final_response,
        'model_used': model
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host=HOST, port=PORT)
