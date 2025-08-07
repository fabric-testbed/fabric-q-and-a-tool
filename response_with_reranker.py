import getpass
import os
import configparser
import requests        
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chat_models import init_chat_model

from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def generate_response(query, db_loc, model, temp, num_docs):

    print(f"Query: {query}")
    print(f"Database:  {db_loc}")
    print(f"Model: {model}")
    print(f"Temperature: {temp}")
    print(f"N of Documents: {num_docs}")
    
    # Specify Vetorstore and create a retriever

    embedding_model = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
    vectorstore = Chroma(persist_directory=db_loc,
          embedding_function=embedding_model)
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Specify the LLM model
    if model == "gpt-4o-mini":
        
        config = configparser.ConfigParser()
        config.read('fabric_ai.conf')
        openai_secret = config['API_KEYS']['openai_key']

        
        llm = init_chat_model("gpt-4o-mini", model_provider="openai", 
                               openai_api_key=openai_secret)
    
    else:
        # Set context window
        context_nums = {
            #"codestral": 32768,
            "codestral": 16384,
            
            } 

        llm = OllamaLLM(model=model, num_ctx=8192,
                    temperature = temp) # higher more creative, lower coherent
       

    template = """You are an AI Help Desk assistant. Use the following information to answer
    the question below.
    
    {context}
    
    Question: On FABRIC Testbed, {question} 
    
    Here is the answer based on the given information: """
    
    
    prompt = PromptTemplate.from_template(template)


    # --- Load model and tokenizer once ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3").to(device)
    model.eval()
    
    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str
    
    
    # Define application steps
    def retrieve(state: State, k):
        retrieved_docs = vectorstore.similarity_search(state["question"], k=200) 

        for i, doc in enumerate(retrieved_docs[:6]):
            print(f"{i+1}. Source: {doc.metadata['source']}")
            
        return {"context": retrieved_docs}

    # --- Define the rerank function ---
    def rerank(state: State) -> dict:
        query = state["question"]
        docs = state["context"]
        pairs = [(query, doc.page_content) for doc in docs]
    
        with torch.no_grad():
            inputs = tokenizer.batch_encode_plus(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            scores = outputs.logits.squeeze().tolist()
    
        # Attach rerank scores and sort
        if isinstance(scores, float):  # for single doc case
            scores = [scores]
    
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = score
    
        reranked_docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)

        for i, doc in enumerate(reranked_docs[:6]):
            print(f"{i+1}. Source: {doc.metadata['source']}, Score: {doc.metadata['rerank_score']:.4f}")
            
        return {"context": reranked_docs[:num_docs]}
    
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response}
    
    
    # Create the state graph
    graph_builder = StateGraph(State)
    
    # Register the nodes with names
    graph_builder.add_node("retrieve", lambda state: retrieve(state, k=100))
    graph_builder.add_node("rerank", rerank)
    graph_builder.add_node("generate", generate)
    
    # Define the execution flow
    graph_builder.add_edge(START, "retrieve")  # Start at retrieve
    graph_builder.add_edge("retrieve", "rerank")
    #graph_builder.add_edge("retrieve", "generate")  # Pass retrieved docs to generate
    graph_builder.add_edge("rerank", "generate")

    graph = graph_builder.compile()
    
    response = graph.invoke({"question": query})
    
    # for debugging
    print(response)
    print(response["context"])


    def print_context_list(contexts):
        sources = []
        for document in contexts:
            sources.append(document.metadata['source'])

        return  "\n\n ----\n\n" + "## Sources:\n\n" + str(sources)

    
    if model == "gpt-4o-mini":
        res = response["answer"].content + print_context_list(response["context"])

    elif model=="deepseek-r1":
        answer = response["answer"]
        res = answer.replace('</think>', ' ').replace('<think>', ' ') + print_context_list(response["context"])
        
    else:
        res = response["answer"] + print_context_list(response["context"])

    print(res)
    return res

