
from pathlib import Path
from langchain_ollama import ChatOllama

from app.document_loader import load_docx
from app.text_splitter import split_documents
from app.embeddings import create_embeddings
from app.vector_store import create_vector_store, create_retriever
from app.rag_chain import create_rag_chain

from ragas_evaluation import RAGASEvaluator, display_results


# Project paths
BASE_DIR = Path(__file__).resolve().parent

DOCUMENT_PATH = BASE_DIR / "documents" / "reference_project.docx"
FAISS_PATH = BASE_DIR / "faiss_index"
RESULT_PATH = BASE_DIR / "ragas_results.csv"


# Add questions and their correct reference answers
# based on your actual DOCX document.
EVALUATION_CASES = [
    {
        "question": "What is the main objective of this document?",
        "ground_truth": "ADD THE CORRECT ANSWER FROM YOUR DOCUMENT"
    },
    {
        "question": "What are the key topics discussed in this document?",
        "ground_truth": "ADD THE CORRECT ANSWER FROM YOUR DOCUMENT"
    }
]


def main():

    print("=" * 60)
    print(" RAGAS EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load DOCX
    print("\n1. Loading document...")

    documents = load_docx(str(DOCUMENT_PATH))

    # 2. Split documents
    print("\n2. Splitting documents...")

    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    # 3. Initialize embeddings
    print("\n3. Creating embeddings...")

    embeddings = create_embeddings()

    # 4. Create FAISS vector store
    print("\n4. Creating FAISS vector store...")

    vector_store = create_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        save_path=str(FAISS_PATH)
    )

    # 5. Create retriever
    print("\n5. Creating retriever...")

    retriever = create_retriever(
        vector_store,
        k=4
    )

    # 6. Create RAG chain
    print("\n6. Initializing Ollama Llama 3.2...")

    rag_chain = create_rag_chain(retriever)

    # 7. Initialize RAGAS evaluator
    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    evaluator = RAGASEvaluator(
        llm=llm,
        embeddings=embeddings
    )

    # 8. Generate actual RAG answers
    print("\n7. Generating answers...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for case in EVALUATION_CASES:

        question = case["question"]

        print(f"\nQuestion: {question}")

        result = rag_chain(question)

        questions.append(result["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(case["ground_truth"])

        print(f"Answer: {result['answer']}")

    # 9. Run RAGAS evaluation
    print("\n8. Running RAGAS evaluation...")

    results = evaluator.evaluate(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )

    # 10. Display results
    display_results(results)

    # 11. Save results
    result_df = results.to_pandas()

    result_df.to_csv(
        RESULT_PATH,
        index=False
    )

    print("\nEvaluation completed!")
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
