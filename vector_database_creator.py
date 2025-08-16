from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def create_vector_store(all_splits):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma.from_documents(
        documents=all_splits,
        collection_name="pdf_embeddings_collection",
        embedding=embeddings,
        persist_directory="./vector_db",
    )
    return vector_store