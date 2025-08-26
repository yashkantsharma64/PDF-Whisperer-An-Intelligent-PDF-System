# import necessary libraries
from langchain_community.document_loaders import PyPDFLoader
import tempfile

# function to parse the uploaded PDF file
def parse_pdf(uploaded_file):
    # upload the file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    # load the PDF file content
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    # return pdf file content
    return docs