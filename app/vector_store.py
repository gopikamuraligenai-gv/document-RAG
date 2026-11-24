
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def create_vector_store(
    chunks: list[Document],
    embeddings,
    save_path: str = "faiss_index"
):
    """
    Create a FAISS vector store from document chunks
    and save it locally.
    """

    if not chunks:
        raise ValueError("No document chunks provided.")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    Path(save_path).mkdir(parents=True, exist_ok=True)

    vector_store.save_local(save_path)

    print(f"FAISS vector store created and saved at: {save_path}")

    return vector_store


def load_vector_store(
    embeddings,
    save_path: str = "faiss_index"
):
    """
    Load an existing FAISS vector store.
    """

    if not Path(save_path).exists():
        raise FileNotFoundError(
            f"FAISS index not found: {save_path}"
        )

    vector_store = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS vector store loaded successfully.")

    return vector_store


def create_retriever(vector_store, k: int = 4):
    """
    Create a retriever that returns the top-k relevant chunks.
    """

    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )
