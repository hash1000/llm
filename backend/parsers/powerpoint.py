from langchain.document_loaders import UnstructuredPowerPointLoader

from .common import process_file

def process_powerpoint(file_path):
    return process_file(
        file_path=file_path,
        loader_class=UnstructuredPowerPointLoader
    )