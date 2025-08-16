from langchain_community.document_loaders import PyPDFLoader
import tempfile

def parse_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    return docs