from langchain.document_loaders import Docx2txtLoader

from .common import process_file

def process_docx(file_path):
    return process_file(
        file_path=file_path,
        loader_class=Docx2txtLoader
    )