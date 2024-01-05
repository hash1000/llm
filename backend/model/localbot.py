import os
from llama_cpp import Llama
from model.bot import ChatBot

class LocalBot(ChatBot):
    def __init__(self):
        super().__init__()
        model_path = os.getenv("MODEL_PATH")
        self.llm = Llama(model_path=model_path, chat_format="llama-2")
    
    def chat_stream(self, query):
        self.history.append({"role": "user", "content": query})
        if len(self.history) > 5:
            self.history = self.history[-5:]
        messages = [{'role': 'system', 'content': self.system_str}] + self.history
        
        knowledge_prompt = self.get_knowledge_prompt(self.query_db(query))
        messages.append(knowledge_prompt)
        
        response = self.llm.create_chat_completion(
            messages = messages,
            temperature = 0.7,
            max_tokens = 800,
            stream = True
        )
        
        output = ""
        for chunk in response:
            data = chunk['choices'][0]['delta']
            if 'content' in data and data['content'] is not None:
                output += data['content']
                yield output
        
        self.history.append({"role": "assistant", "content": output})