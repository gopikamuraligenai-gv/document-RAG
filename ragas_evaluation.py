
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


class RAGASEvaluator:

    def __init__(self, llm, embeddings):

        """
        Initialize RAGAS evaluator.

        llm:
            Existing LangChain LLM.

        embeddings:
            Existing LangChain embedding model.
        """

        self.llm = LangchainLLMWrapper(llm)

        self.embeddings = LangchainEmbeddingsWrapper(
            embeddings
        )

        # 1. Context Precision
        self.context_precision = LLMContextPrecisionWithReference(
            llm=self.llm
        )

        # 2. Context Recall
        self.context_recall = LLMContextRecall(
            llm=self.llm
        )

        # 3. Faithfulness
        self.faithfulness = Faithfulness(
            llm=self.llm
        )

        # 4. Answer Relevancy
        self.answer_relevancy = ResponseRelevancy(
            llm=self.llm,
            embeddings=self.embeddings
        )

    def evaluate(
        self,
        questions,
        answers,
        contexts,
        ground_truths
    ):

        # Validate dataset

        if not (
            len(questions)
            == len(answers)
            == len(contexts)
            == len(ground_truths)
        ):
            raise ValueError(
                "All input lists must have equal lengths."
            )

        if len(questions) == 0:
            raise ValueError(
                "Evaluation dataset cannot be empty."
            )

        # Create evaluation dataset

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

        dataset = Dataset.from_dict(data)

        print("\nStarting RAGAS Evaluation...")
        print("=" * 50)

        # Execute all four metrics

        results = evaluate(
            dataset=dataset,
            metrics=[
                self.context_precision,
                self.context_recall,
                self.faithfulness,
                self.answer_relevancy
            ]
        )

        return results


def display_results(results):

    print("\nRAGAS EVALUATION RESULTS")
    print("=" * 50)

    result_df = results.to_pandas()

    print(result_df.to_string(index=False))

    print("\nAVERAGE SCORES")
    print("=" * 50)

    average_scores = result_df.mean(
        numeric_only=True
    )

    for metric, score in average_scores.items():

        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":

    print("RAGAS EVALUATION MODULE")
    print("=" * 50)

    print("Four evaluation metrics:")
    print("1. Context Precision")
    print("2. Context Recall")
    print("3. Faithfulness")
    print("4. Answer Relevancy")

    print("\nWaiting for existing LangChain LLM and embeddings.")
