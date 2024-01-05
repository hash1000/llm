import dotenv
dotenv.load_dotenv()

import gradio as gr
from model.bot import ChatBot


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
# chatbot_stream = gr.Chatbot(avatar_images=('/root/THDC/backend/public/user.png', '/root/THDC/backend/public/thdc.png'), bubble_full_width = False)
chat_interface_stream = gr.ChatInterface(predict, 
                 title=title, 
                 description=description, 
                 textbox=gr.Textbox(),
                 chatbot=chatbot_stream,
                 css=css, ) 

# Gradio Demo 
with gr.Blocks() as demo:

    with gr.Tab("Streaming"):
        # chatbot_stream.like(vote, None, None)
        chat_interface_stream.render()

      
if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7861, debug=True, share=True)