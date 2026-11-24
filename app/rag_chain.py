
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_rag_chain(retriever):
    """
    Create a RAG chain using FAISS and Ollama Llama 3.2.

    Returns:
        A function that retrieves document chunks,
        generates answers, and returns structured output
        for RAGAS evaluation.
    """

    # 1. Initialize Ollama LLM
    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    # 2. Create prompt template
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful document question-answering assistant.

Instructions:
- Answer using ONLY the provided context.
- Do not use outside knowledge.
- Do not make up information.
- If the answer is unavailable in the context, say:
  "I could not find this information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # 3. Create generation chain
    generation_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # 4. Retrieve and generate answer
    def retrieve_and_answer(question: str) -> dict:

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Retrieve relevant chunks from FAISS
        documents = retriever.invoke(question)

        # Extract actual document contexts
        contexts = [
            doc.page_content
            for doc in documents
            if doc.page_content and doc.page_content.strip()
        ]

        # Extract source information
        sources = sorted({
            str(doc.metadata.get("source", "Unknown"))
            for doc in documents
        })

        # Handle empty retrieval
        if not contexts:
            return {
                "question": question,
                "answer": (
                    "I could not find this information "
                    "in the uploaded document."
                ),
                "contexts": [],
                "sources": []
            }

        # Combine retrieved chunks
        context = "\n\n".join(contexts)

        # Generate answer using Ollama
        answer = generation_chain.invoke({
            "context": context,
            "question": question
        })

        # Return structured output for RAGAS
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "sources": sources
        }

    return retrieve_and_answer
