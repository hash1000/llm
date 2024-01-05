from langchain.document_loaders import CSVLoader
from .common import process_file


def process_csv(file_path):
    return process_file(
        file_path=file_path,
        loader_class=CSVLoader,
    )
