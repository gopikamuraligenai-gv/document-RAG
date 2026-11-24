
import pandas as pd
import os


def save_evaluation_results(results, filename="ragas_evaluation_results.csv"):

    # Convert RAGAS results into DataFrame
    df = results.to_pandas()

    print("\n" + "=" * 60)
    print("          RAGAS EVALUATION RESULTS")
    print("=" * 60)

    # Display individual question results
    print(df.to_string(index=False))

    # Calculate average scores
    metrics = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy"
    ]

    print("\n" + "=" * 60)
    print("          AVERAGE METRIC SCORES")
    print("=" * 60)

    for metric in metrics:

        if metric in df.columns:

            score = df[metric].mean()

            print(f"{metric}: {score:.4f}")

        else:

            print(f"{metric}: Not available")

    # Save CSV file
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        filename
    )

    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("Evaluation results saved successfully!")
    print("File location:", output_path)
    print("=" * 60)

    return df


if __name__ == "__main__":

    print("This file is a reusable evaluation-results module.")
    print("Run run_ragas_evaluation.py to generate results.")
