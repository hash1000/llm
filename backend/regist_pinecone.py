import os
import uuid
from cryptography.fernet import Fernet
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pinecone
from tqdm import tqdm

from embedding.embedding import embed_text
from parsers.csv import process_csv
from parsers.doc import process_doc
from parsers.docx import process_docx
from parsers.pdf import process_pdf
from parsers.powerpoint import process_powerpoint
from parsers.xlsx import process_xlsx

ALLOWED_FILES = [
    ".csv", 
    ".pdf",
    ".doc",
    ".docx",
    ".pptx",
    ".xls",
    ".xlsx",
]

file_processors = {
    ".csv": process_csv,
    ".pdf": process_pdf,
    ".pptx": process_powerpoint,
    ".doc": process_doc,
    ".docx": process_docx,
    ".xlsx": process_xlsx,
    ".xls": process_xlsx,
}

def construct_knowledgebase(dataset_folder):
    private = os.getenv("PRIVATE", "false").lower() == "true"

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
    
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    
    if private:
        dimention = int(os.getenv("DIMENSION", 1536))
        pinecone.create_index(index_name, dimension=dimention, metric="cosine")
    else:
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
                embedding_list = [embed_text(doc.page_content) for doc in docs]
                index.upsert(vectors=list(zip(id_list, embedding_list, metadata_list)))


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    construct_knowledgebase('./data')