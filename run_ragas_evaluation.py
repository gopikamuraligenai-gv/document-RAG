
from pathlib import Path

from app.document_loader import load_docx
from app.text_splitter import split_documents
from app.embeddings import create_embeddings
from app.vector_store import create_vector_store, create_retriever
from app.rag_chain import create_rag_chain

from ragas_evaluation import RAGASEvaluator
from evaluation_results import save_evaluation_results
from langchain_ollama import ChatOllama


BASE_DIR = Path(__file__).resolve().parent

DOCUMENT_PATH = BASE_DIR / "documents" / "reference_project.docx"
FAISS_PATH = BASE_DIR / "faiss_index"


def main():

    print("\n=== ACTUAL RAGAS EVALUATION ===")

    # 1. Load actual DOCX
    print("\n1. Loading reference document...")
    documents = load_docx(str(DOCUMENT_PATH))

    # 2. Split document
    print("\n2. Splitting document...")
    chunks = split_documents(documents)

    # 3. Initialize embeddings
    print("\n3. Initializing embeddings...")
    embeddings = create_embeddings()

    # 4. Create FAISS vector store
    print("\n4. Creating FAISS vector store...")
    vector_store = create_vector_store(
        chunks, embeddings, str(FAISS_PATH)
    )

    # 5. Create retriever
    print("\n5. Creating retriever...")
    retriever = create_retriever(vector_store, k=4)

    # 6. Initialize RAG chain
    print("\n6. Initializing Ollama RAG chain...")
    ask_question = create_rag_chain(retriever)

    # 7. Initialize evaluation LLM
    evaluator_llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    evaluator = RAGASEvaluator(
        llm=evaluator_llm,
        embeddings=embeddings
    )

    # 8. Evaluation questions
    questions = [
        "What is the purpose of the project?",
        "What are the main components of the system?",
        "How does the RAG pipeline work?",
        "Which technologies and models are used?",
        "How are the AI agents coordinated?"
    ]

    answers = []
    contexts = []

    # 9. Retrieve actual contexts and generate answers
    print("\n7. Generating actual answers and contexts...")

    for question in questions:

        print(f"\nQuestion: {question}")

        retrieved_docs = retriever.invoke(question)

        actual_contexts = [
            doc.page_content for doc in retrieved_docs
        ]

        answer = ask_question(question)

        if not isinstance(answer, str):
            answer = str(answer)

        answers.append(answer)
        contexts.append(actual_contexts)

        print(f"Answer: {answer}")
        print(f"Retrieved chunks: {len(actual_contexts)}")

    # 10. Ground truths
    # Replace these with verified reference answers
    # from your actual reference_project.docx.
    ground_truths = [
        "Describe the project's purpose based on the document.",
        "Describe the main system components based on the document.",
        "Explain the RAG pipeline described in the document.",
        "Identify the technologies and models listed in the document.",
        "Explain how the AI agents are coordinated according to the document."
    ]

    # 11. Run RAGAS
    print("\n8. Running RAGAS evaluation...")

    results = evaluator.evaluate(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )

    # 12. Save evaluation results
    print("\n9. Saving results...")

    save_evaluation_results(
        results,
        filename="ragas_evaluation_results.csv"
    )

    print("\n=== RAGAS EVALUATION COMPLETED ===")


if __name__ == "__main__":
    main()
