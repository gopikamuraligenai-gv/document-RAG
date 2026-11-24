
from langchain_community.vectorstores import FAISS


class MMRRetriever:

    def __init__(
        self,
        vectorstore,
        k=5,
        fetch_k=15,
        lambda_mult=0.5
    ):
        """
        vectorstore: Existing LangChain FAISS vector store
        k: Number of final documents
        fetch_k: Number of candidate documents
        lambda_mult: Balance between relevance and diversity
        """

        self.vectorstore = vectorstore
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult

    def retrieve(self, query):

        documents = self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult
        )

        return documents


def display_documents(documents):

    print("\nMMR RETRIEVED DOCUMENTS")
    print("=" * 50)

    for index, doc in enumerate(documents, start=1):

        print(f"\nDocument {index}")

        print("-" * 40)

        print("Content:")

        print(doc.page_content[:500])

        print("\nMetadata:")

        print(doc.metadata)


# Example integration:
#
# from app.config import initialize_embeddings
#
# embeddings = initialize_embeddings()
#
# vectorstore = FAISS.load_local(
#     "faiss_index",
#     embeddings,
#     allow_dangerous_deserialization=True
# )
#
# retriever = MMRRetriever(
#     vectorstore=vectorstore,
#     k=5,
#     fetch_k=15,
#     lambda_mult=0.5
# )
#
# query = "What is the loan amount and interest rate?"
#
# documents = retriever.retrieve(query)
#
# display_documents(documents)
