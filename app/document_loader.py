
from pathlib import Path

from docx import Document as DocxDocument
from langchain_core.documents import Document


def load_docx(file_path: str) -> list[Document]:
    """
    Load text from a DOCX file and convert it into
    LangChain Document objects.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() != ".docx":
        raise ValueError("Please provide a .docx file.")

    doc = DocxDocument(str(path))

    documents = []

    # Extract paragraphs
    for index, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text.strip()

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "type": "paragraph",
                        "paragraph_number": index
                    }
                )
            )

    # Extract tables
    for table_index, table in enumerate(doc.tables):
        rows = []

        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))

        table_text = "\n".join(rows).strip()

        if table_text:
            documents.append(
                Document(
                    page_content=table_text,
                    metadata={
                        "source": path.name,
                        "type": "table",
                        "table_number": table_index
                    }
                )
            )

    if not documents:
        raise ValueError("No readable text found in the DOCX file.")

    print(f"Successfully loaded: {path.name}")
    print(f"Extracted {len(documents)} text sections.")

    return documents
