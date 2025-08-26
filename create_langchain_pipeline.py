from langchain.chains import RetrievalQAWithSourcesChain

# function to create the langchain retrieval chain
def create_chain(llm, vector_store, map_prompt_template, combine_prompt_template):
    # create QA chain object using map_reduce method
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type="map_reduce",
        chain_type_kwargs={
            "question_prompt": map_prompt_template,
            "combine_prompt": combine_prompt_template,
        }
    )
    # return the chain object
    return chain