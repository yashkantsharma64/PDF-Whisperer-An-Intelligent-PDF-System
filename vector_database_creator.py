from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# function to create vector store from the text chunks
def create_vector_store(all_splits):
    # create huggungface embeddings object
    # the embeddings model used here is "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # create vector store using FAISS
    vector_store = FAISS.from_documents(
        all_splits,
        embeddings,
    )
    # return vector_store 
    vector_store.save_local("faiss_index")
