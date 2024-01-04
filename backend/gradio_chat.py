import dotenv
dotenv.load_dotenv()

import asyncio
import gradio as gr
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.pinecone import Pinecone
import logging
import os
import pinecone
from typing import Awaitable

from model.bot import ChatBot



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pinecone.Index(index_name)

openai_api_key = os.getenv("OPENAI_API_KEY")
temperature = os.getenv("TEMPERATURE")
model = os.getenv("MODEL")

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Pinecone(index, embeddings, "text")
retriever = vectorstore.as_retriever()

title = "THDC AI Assistant"
description = """
This is a chatbot that explain about THDC.
"""
css = """.toast-wrap { display: none !important } """

bot = ChatBot()

async def predict(message, history):
    for token in bot.chat_stream(message):
        yield token

# No Stream    
def predict_batch(message, history):
    answering_llm = ChatOpenAI(
        temperature=temperature,
        model=model,
        streaming=False,
        verbose=True,
        callbacks=None,
        openai_api_key=openai_api_key,
    )
    
    doc_chain = load_qa_chain(
        answering_llm, chain_type="stuff", prompt=create_prompt_template(), verbose=True
    )
    
    question_llm = ChatOpenAI(
        temperature=temperature,
        model=model,
        streaming=False,
        verbose=True,
        callbacks=None,
        openai_api_key=openai_api_key,
    )

    qa = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=LLMChain(
            llm=question_llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True),
        combine_docs_chain=doc_chain,  # pyright: ignore reportPrivateUsage=none
        verbose=True,
        # rephrase_question=False,
    )
    
    history = history[-min(len(history), 3):]
    
    model_response = qa(
        {
            "question": message,
            "chat_history": [(pair[0], pair[1]) for pair in history]
        }
    )
    print(model_response)
    return model_response['answer']


chatbot_stream = gr.Chatbot(avatar_images=('/root/THDC/backend/public/user.png', '/root/THDC/backend/public/thdc.png'), bubble_full_width = False)
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