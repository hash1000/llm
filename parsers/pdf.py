from langchain.document_loaders import PyMuPDFLoader  # UnstructuredPDFLoader
from .common import process_file


def process_pdf(file_path):
    return process_file(
        file_path=file_path,
        loader_class=PyMuPDFLoader
    )