import dotenv
dotenv.load_dotenv()

import gradio as gr
from model.bot import ChatBot
from model.localbot import LocalBot


title = "THDC AI Assistant"
description = """
This is a chatbot that explain about THDC.
"""
css = """.toast-wrap { display: none !important } """

bot = ChatBot()
async def predict(message, history):
    for token in bot.chat_stream(message):
        yield token

chatbot_stream = gr.Chatbot(avatar_images=('public/user.png', 'public/thdc.png'), bubble_full_width = False)
chat_interface_gpt = gr.ChatInterface(predict, 
                 title=title, 
                 description=description, 
                 textbox=gr.Textbox(),
                 chatbot=chatbot_stream,
                 css=css, ) 

localBot = LocalBot()
async def predict_local(message, history):
    for token in localBot.chat_stream(message):
        yield token

chatbot_stream_local = gr.Chatbot(avatar_images=('public/user.png', 'public/thdc.png'), bubble_full_width = False)
chat_interface_local = gr.ChatInterface(predict_local, 
                 title=title, 
                 description=description, 
                 textbox=gr.Textbox(),
                 chatbot=chatbot_stream_local,
                 css=css, ) 

# Gradio Demo 
with gr.Blocks() as demo:

    with gr.Tab("gpt-4-1106-preview"):
        chat_interface_gpt.render()
    with gr.Tab("llama-2-13b.Q4_K_M"):
        chat_interface_local.render()

    
if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7861, debug=True, share=True)