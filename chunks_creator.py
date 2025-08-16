from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=700,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits