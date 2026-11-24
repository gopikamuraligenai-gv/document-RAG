
def create_retriever(vector_store, k=4):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )
