from langchain.document_loaders import UnstructuredExcelLoader

from .common import process_file

def process_xlsx(file_path):
    return process_file(
        file_path=file_path,
        loader_class=UnstructuredExcelLoader
    )