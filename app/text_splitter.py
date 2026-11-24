
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> list[Document]:
    """
    Split loaded documents into smaller chunks
    using LangChain's RecursiveCharacterTextSplitter.
    """

    if not documents:
        raise ValueError("No documents provided for splitting.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Original sections: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    return chunks
