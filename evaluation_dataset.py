
# evaluation_dataset.py

"""
Evaluation dataset for Financial Document RAG.

This dataset contains:
1. Questions
2. Expected answers (ground truth)
3. Relevant document contexts

Use the same dataset for:
- LangChain RAG
- Without LangChain RAG
"""


EVALUATION_DATASET = [

    {
        "question": "What is the loan interest rate?",

        "ground_truth": (
            "The loan interest rate is 9%."
        ),

        "relevant_contexts": [
            "The loan agreement specifies an interest rate of 9%."
        ]
    },

    {
        "question": "What is the repayment period?",

        "ground_truth": (
            "The repayment period is 10 years."
        ),

        "relevant_contexts": [
            "The loan must be repaid over a period of 10 years."
        ]
    },

    {
        "question": "What is the total loan amount?",

        "ground_truth": (
            "The total approved loan amount is 50 lakh rupees."
        ),

        "relevant_contexts": [
            "The borrower has been approved for a total loan amount of INR 50 lakh."
        ]
    },

    {
        "question": "What happens if the borrower misses a repayment?",

        "ground_truth": (
            "The borrower may be charged a late payment penalty "
            "according to the loan agreement."
        ),

        "relevant_contexts": [
            "Late repayment may result in additional penalty charges "
            "as specified in the loan agreement."
        ]
    },

    {
        "question": "What is the purpose of the loan?",

        "ground_truth": (
            "The loan is intended to support business expansion."
        ),

        "relevant_contexts": [
            "The borrower intends to use the approved loan amount "
            "for business expansion."
        ]
    }

]


def get_questions():
    return [
        item["question"]
        for item in EVALUATION_DATASET
    ]


def get_ground_truths():
    return [
        item["ground_truth"]
        for item in EVALUATION_DATASET
    ]


def get_relevant_contexts():
    return [
        item["relevant_contexts"]
        for item in EVALUATION_DATASET
    ]


if __name__ == "__main__":

    print("FINANCIAL RAG EVALUATION DATASET")
    print("=" * 45)

    print("Total Questions:", len(EVALUATION_DATASET))

    for index, item in enumerate(EVALUATION_DATASET, start=1):

        print(f"\nQuestion {index}: {item['question']}")
        print("Ground Truth:", item["ground_truth"])
        print("Relevant Contexts:", item["relevant_contexts"])
