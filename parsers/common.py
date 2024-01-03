def process_file(file_path, loader_class):
    loader = loader_class(file_path)
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        documents = []
    return documents