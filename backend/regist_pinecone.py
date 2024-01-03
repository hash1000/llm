import uuid
from cryptography.fernet import Fernet
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
import os
import openai
import pinecone
from tqdm import tqdm

from parsers.doc import process_doc
from parsers.docx import process_docx
from parsers.pdf import process_pdf
from parsers.powerpoint import process_powerpoint
from parsers.xlsx import process_xlsx

ALLOWED_FILES = [
    ".pdf",
    ".doc",
    ".docx",
    ".pptx",
    ".xls",
    ".xlsx",
]

file_processors = {
    ".pdf": process_pdf,
    ".pptx": process_powerpoint,
    ".doc": process_doc,
    ".docx": process_docx,
    ".xlsx": process_xlsx,
    ".xls": process_xlsx,
}


# client = openai.OpenAI(api_key=os.getenv('OPENAI_KEY'))

# def get_embedding(text, model="text-embedding-ada-002"):
#    text = text.replace("\n", " ")
#    return client.embeddings.create(input = [text], model=model).data[0].embedding


# def embedding(sentences):
#     if isinstance(sentences, str):
#         return get_embedding(sentences, model=embedding_model)
#     elif isinstance(sentences, list):
#         result = []
#         for sentence in sentences:
#             result.append(get_embedding(sentence, model=embedding_model))
#         return result


def construct_knowledgebase(dataset_folder):
    # Retrieve the key from environment variable
    key_string = os.getenv("ENCRYPTION_KEY")
    key = key_string.encode()
    
    # Initialize Fernet with the key
    cipher_suite = Fernet(key)

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),
    )
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pinecone.Index(index_name)
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    # vectorstore = Pinecone(index, embeddings, "text")
    
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    pinecone.create_index(index_name, dimension=1536, metric="cosine")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for root, dirs, files in os.walk(dataset_folder):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.endswith(tuple(ALLOWED_FILES)):
                print(file)
                file_extension = os.path.splitext(file)[1]
                documents = file_processors[file_extension](os.path.join(root, file))
                docs = text_splitter.split_documents(documents)
                id_list = []
                embedding_list = []
                metadata_list = []
                for doc in docs:
                    id_list.append(str(uuid.uuid4()))
                    metadata_list.append({"text": cipher_suite.encrypt(doc.page_content.encode()).decode()})
                embedding_list = embeddings.embed_documents([doc.page_content for doc in docs])
                index.upsert(vectors=list(zip(id_list, embedding_list, metadata_list)))
                # vectorstore.add_documents(documents=docs, index_name=index_name)


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    construct_knowledgebase('./data')