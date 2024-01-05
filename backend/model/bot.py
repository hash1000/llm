import os
import openai
import pinecone
from cryptography.fernet import Fernet

from embedding.embedding import embed_text


client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatBot():

    def __init__(self):
        self.private = os.getenv("PRIVATE", "false").lower() == "true"
        self.llm_model = os.getenv("LLM_MODEL")
        
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV"),
        )
        self.pinecone_index = pinecone.Index(self.index_name)
        self.top_k = 5
        
        self.system_str = (
            "You are a helpful customer assistant from THDC (Total Health Dental Care).\n"
            "You need to answer the customer questions based on the company's data from knowledge base and conversation history.\n"
        )
        self.history = []
        
        key_string = os.getenv("ENCRYPTION_KEY")
        self.cipher_suite = Fernet(key_string.encode())

    def query_db(self, query):
        query_vector = embed_text(query)
        results = self.pinecone_index.query(query_vector, top_k=self.top_k)
        matches = results.to_dict()['matches']
        ids = [match['id'] for match in matches]
        data = self.pinecone_index.fetch(ids).to_dict()['vectors']
        descriptions = []
        for id in ids:
            descriptions.append(data[id]["metadata"])
        
        text = ""
        for description in descriptions:
            text = text + "<document>\n"
            for key, value in description.items():
                text = text + f"{key}: {self.cipher_suite.decrypt(value.encode()).decode()}\n"
            text = text + "</document>\n\n"
        return text
    
    def get_knowledge_prompt(self, knowledge_text):
        role_content = "You are Total Health Dental Care, the personal AI medical assistant, specializes in dental care. It provides clear, simplified explanations of what test results mean, relating them to possible medical conditions in easy-to-understand language, avoiding medical jargon. THDC suggests specific, straightforward questions for patients to ask their doctors, enhancing their understanding and communication. When presented with symptoms, THDC offers a list of potential causes in simple terms and engages in focused dialogue to refine these possibilities. It encourages consulting healthcare professionals for definitive diagnoses and provides questions to facilitate patient-doctor discussions. THDC gathers basic information from clients, such as age, gender, important past medical history, and inquires about any available lab or imaging studies, ensuring a more comprehensive and tailored consultation. THDC maintains a professional demeanor, strictly discussing medical topics and referring to healthcare professionals as needed. It avoids non-medical discussions, speculation, and personal opinions, relying on factual information from reliable medical textbooks and the user's uploaded medical documents."

        prompt_text = f"""
# Who you are
{role_content}

# Principles that must be followed
0. Your answer must be easily understandable to patients!
1. Your answer shouldn't be too long.
2. There should be good spacing between each category and each section in your answer.
3. Your answer must be based on the following documents and conversation history.

# Documents you must be based on
<documents>
{knowledge_text}
</documents>
        """
        return {"role": "system", "content": prompt_text}
    
    def chat_stream(self, query):
        self.history.append({"role": "user", "content": query})
        if len(self.history) > 5:
            self.history = self.history[-5:]
        messages = [{'role': 'system', 'content': self.system_str}] + self.history
        
        knowledge_prompt = self.get_knowledge_prompt(self.query_db(query))
        messages.append(knowledge_prompt)
        
        response = client.chat.completions.create(
            model = self.llm_model,
            messages = messages,
            temperature = 0.7,
            max_tokens = 800,
            stream = True
        )
        
        output = ""
        for chunk in response:
            data = chunk.choices[0].delta
            if hasattr(data, 'content') and data.content is not None:
                output += data.content
                yield output
        
        self.history.append({"role": "assistant", "content": output})

