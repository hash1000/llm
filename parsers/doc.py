from langchain.document_loaders import UnstructuredWordDocumentLoader

from .common import process_file

def process_doc(file_path):
    return process_file(
        file_path=file_path,
        loader_class=UnstructuredWordDocumentLoader
    )