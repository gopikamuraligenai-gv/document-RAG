
from llm import initialize_llm
from retriever import TFIDFRetriever


class RAGEngine:

    def __init__(self, documents, k=5):

        # Initialize Ollama Llama 3.2
        self.llm = initialize_llm()

        # Initialize TF-IDF retriever
        self.retriever = TFIDFRetriever(
            documents=documents,
            k=k
        )

    # Retrieve relevant document chunks
    def retrieve_documents(self, question):

        results = self.retriever.retrieve(question)

        return results

    # Generate answer using Ollama
    def generate_answer(self, question, results):

        contexts = [
            result["document"].page_content
            if hasattr(result["document"], "page_content")
            else str(result["document"])
            for result in results
        ]

        context = "\n\n".join(contexts)

        prompt = f"""
You are a financial document assistant.

Answer the question using ONLY the provided context.
Do not invent or assume information.

If the answer is not available, say:
Information not found in the document.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content

    # Execute complete RAG pipeline
    def query(self, question):

        results = self.retrieve_documents(question)

        answer = self.generate_answer(
            question,
            results
        )

        contexts = [
            result["document"].page_content
            if hasattr(result["document"], "page_content")
            else str(result["document"])
            for result in results
        ]

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "retrieval_scores": [
                result["score"] for result in results
            ]
        }
