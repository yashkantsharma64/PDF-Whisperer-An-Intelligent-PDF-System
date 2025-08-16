from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import Optional, List
import google.generativeai as genai
from dotenv import load_dotenv
import os
import streamlit as st
from pdf_parser import parse_pdf
from chunks_creator import create_chunks
from vector_database_creator import create_vector_store
from create_langchain_pipeline import create_chain

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Custom LangChain wrapper for Gemini due to protobuf issue
class GeminiLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = model.generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self):
        return {"name": "gemini-custom-rest"}

    @property
    def _llm_type(self):
        return "custom_gemini"

# Define Map and Combine Prompts
# Map prompt: **MUST** use "context" as input variable!
map_prompt_template = PromptTemplate(
    template=(
        "You are an expert assistant. Given the following document chunk and "
        "the user's question, provide a helpful and factual answer based only on this chunk.\n\n"
        "Question: {question}\n\n"
        "Chunk:\n{context}"
    ),
    input_variables=["question", "context"]
)

# Combine prompt: **MUST** use "summaries" as input variable for the reduce step!
combine_prompt_template = PromptTemplate(
    template=(
        "You are an expert assistant. Review the following partial answers and combine them into a single, complete answer for the user's question.\n\n"
        "Question: {question}\n\n"
        "Partial Answers:\n{summaries}\n\n"
        "Final Answer:"
    ),
    input_variables=["question", "summaries"]
)

load_dotenv()
st.title("PDF Whisperer - An Intelligent PDF System")
loader = st.empty()
query = st.text_input("Enter your question about the PDF document:")
st.write("Answer:")
st.sidebar.title("Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Select file", type=["pdf"], key="pdf_uploader")
process_pdf = st.sidebar.button("Ask PDF")

if process_pdf and uploaded_pdf is not None:
    # Load the PDF/document
    loader.write("Loading PDF...")
    docs = parse_pdf(uploaded_pdf)

    # Split the documents into chunks
    loader.warning("Creating chunks...")
    all_splits = create_chunks(docs)
    
    # Create Embeddings and Vector Store
    print("Creating vector store...")
    loader.warning("Creating Vector store...")
    vector_store = create_vector_store(all_splits)
    
    # Create Gemini LLM instance
    llm = GeminiLLM()

    # Create RetrievalQAWithSourcesChain with map_reduce method
    print("Creating chain...")
    chain = create_chain(llm, vector_store, map_prompt_template, combine_prompt_template)
    print("Chain created successfully.")

    # Run the Query
    if query:
        # chain.max_tokens_limit = 800
        print("Running query...")
        result = chain.invoke({"question": query})
        loader.success("Query answered successfully!...✅")
        st.write(result["answer"])
    else:
        st.write("Please enter a question to get an answer from the PDF document.")

        