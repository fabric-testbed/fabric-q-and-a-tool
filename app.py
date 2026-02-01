from flask import (
    Flask, request, render_template, session,
    redirect, url_for, jsonify, flash, send_file, Response
)
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph

from utils.logging_setup import setup_logging 
from utils.helpers.validate_api_call import validate_api_key, validate_api_params
from utils.helpers.assign_params import assign_params_by_tool
from utils.helpers.initialize_dependencies import initialize_model, initialize_retreiver
from utils.helpers.print_helpers import clean_response, filter_responses_and_add_context
from utils.helpers.rerank_helpers import calculate_document_scores, attach_scores_to_documents
from config import State, HOST, PORT

app = Flask(__name__)

# Set up logging 
setup_logging(app)

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
    vectorstore = initialize_retreiver(db_loc)
    llm = initialize_model(model, temp)

    # Build prompt from template
    prompt = PromptTemplate.from_template(template)
    # ----------------------------------------------------------------------------------------------------- #
    # -------------------------------------- Define Application Steps ------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    def retrieve(state: State, k=20):
        """
        Retrieves relevant documents from vectorstore
        Arguments:
            - state: this is the current state of the LLM application
            - k: this is the no.of documents to retreive, given by the user
        Returns:
            - context: documents retrieved, a piece of the state that will be merged to application state
        """
        retrieved_docs = vectorstore.similarity_search(state["question"], k=k)

        app.logger.info(f"{k} documents retreived from vectostore")

        return {"context": retrieved_docs}

    # --- Define the rerank function ---
    def rerank(state: State) -> dict:
        """
        Reranks the documents retrieved based on relevance to the query
        Arguments:
            - state: this is the current state of the LLM application
        Returns:
            - context: reranked documents, a piece of the state that will be merged to application state
        """
        query = state["question"]
        docs = state["context"]
        pairs = [(query, doc.page_content) for doc in docs]

        # Calculate and attach document scores
        scores = calculate_document_scores(pairs)
        reranked_docs = attach_scores_to_documents(scores, docs)

        return {"context": reranked_docs[:num_docs]}

    def generate(state: State):
        """
        Generates the response to user query
        Arguments:
            - state: this is the current state of the LLM application
        Returns:
            - answer: answer generated, a piece of the state that will be merged to application state
        """
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response}

    # Create the state graph
    graph_builder = StateGraph(State)

    # Register the nodes with names
    graph_builder.add_node("retrieve", lambda state: retrieve(state, k=20))
    graph_builder.add_node("rerank", rerank)

    # Define the execution flow
    graph_builder.add_edge(START, "retrieve")  # Start at retrieve
    graph_builder.add_edge("retrieve", "rerank")

    graph = graph_builder.compile()
    rag_state = graph.invoke({"question": query})

    # Just run this final step to generate the response
    response = ""
    try:
        response = generate(rag_state)
        app.logger.info("LLM inference successful")
    except Exception as e:
        response['answer'] = "(No response provided)"
        app.logger.info(f"Error occured during LLM inference: {e}")

    # Clean and filter the response 
    cleaned_response = clean_response(response)
    # final_response = cleaned_response
    final_response = filter_responses_and_add_context(cleaned_response, rag_state["context"], tool_type)

    # Add results to the log
    app.logger.custom(f"QUERY: {query}\nRESPONSE: {final_response}\nMODEL: {model}\nTOOL: {tool_type}\n\n -- \n\n")

    return jsonify({
        'query': query,
        'response': final_response,
        'model_used': model
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host=HOST, port=PORT)
