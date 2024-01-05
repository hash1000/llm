import os
import openai
from transformers import AutoModel


def embed_text(text):
    private = os.getenv("PRIVATE", "false").lower() == "true"
    if private: 
        embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL")
        model = AutoModel.from_pretrained(embedding_model, trust_remote_code=True)
        return model.encode([text])[0].tolist()
    else:
        embedding_model = os.getenv("EMBEDDING_MODEL")
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return openai_client.embeddings.create(input=text, model="text-embedding-ada-002").data[0].embedding

