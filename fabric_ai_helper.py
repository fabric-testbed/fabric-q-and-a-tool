import gradio as gr
import random
import time
import response as response
import configparser

config = configparser.ConfigParser()
config.read('fabric_ai.conf')

def chat_response(question, model, db, temperature, doc_count):

    if db == "Forum only":
        db_loc = config['VectorDB']['forum_db_loc']
    elif db == "KB only":
        db_loc = config['VectorDB']['kb_db_loc']
    elif db == "KB and Forum":
        db_loc = config['VectorDB']['kb_forum_db_loc']
    else:
        db_loc = None

    ai_response = response.generate_response(question, db_loc, model, temperature, doc_count) 

    return ai_response
    

def authenticate(username, password):
    
    print(username, password)
    print(config['USERS'][username])

    return True if (password == config['USERS'][username]) else False


with gr.Blocks(theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg)) as demo:

    model_choices = ["phi4",
                     "llama3.3",
                     "gemma3:1b", 
                     "deepseek-r1", 
                     "mistral-large",
                     "gpt-4o-mini"]

    gr.Markdown("# 🤖 Knowledge Base Bot for FABRIC")

    with gr.Row():
        question = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
        with gr.Column():
            model = gr.Dropdown(choices=model_choices, 
                                label="Choose Model", 
                                value="phi4")
            db = gr.Dropdown(choices = ["No Database", "Forum only", "KB and Forum", "KB only"],
                              label="Which document Database to use?",
                              value="No Database")


    temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.1, 
                            label="Temperature (Randomness)")
    doc_count = gr.Slider(0, 8, value=4, step=1,
                          label="number of documents to retrieve")

    submit_btn = gr.Button("Generate Response")
    output = gr.Markdown(label="LLM Response")

    submit_btn.click(chat_response, 
                    inputs=[question, model, db, temperature, doc_count], 
                    outputs=output)

#demo.launch(share=True)
demo.launch(server_name=config['SERVER']['host_url'], server_port=7861,  auth=authenticate)
