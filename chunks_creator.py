from langchain.text_splitter import RecursiveCharacterTextSplitter

# function to create chunks of text from the PDF document content
def create_chunks(docs):
    # create langchain text splitter object
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=700,
        add_start_index=True,
    )
    # split the text into chunks
    all_splits = text_splitter.split_documents(docs)
    # return the splitted chunks
    return all_splits