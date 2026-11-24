
def calculate_precision_at_k(retrieved_ids, relevant_ids, k=5):
    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        doc_id in relevant_ids for doc_id in retrieved
    )

    return relevant_count / len(retrieved)


def calculate_recall_at_k(retrieved_ids, relevant_ids, k=5):
    if not relevant_ids:
        return 0.0

    retrieved = retrieved_ids[:k]

    relevant_count = sum(
        doc_id in relevant_ids for doc_id in retrieved
    )

    return relevant_count / len(relevant_ids)


def calculate_hit_rate(retrieved_ids, relevant_ids, k=5):
    if not relevant_ids:
        return 0.0

    return float(
        any(doc_id in relevant_ids for doc_id in retrieved_ids[:k])
    )


def evaluate_retrieval(retrieved_ids, relevant_ids, k=5):

    precision = calculate_precision_at_k(
        retrieved_ids, relevant_ids, k
    )

    recall = calculate_recall_at_k(
        retrieved_ids, relevant_ids, k
    )

    hit_rate = calculate_hit_rate(
        retrieved_ids, relevant_ids, k
    )

    print("\n========== RETRIEVAL EVALUATION ==========")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"Hit Rate@{k}: {hit_rate:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate
    }


if __name__ == "__main__":

    retrieved_ids = [
        "chunk_1",
        "chunk_2",
        "chunk_3",
        "chunk_4",
        "chunk_5"
    ]

    relevant_ids = [
        "chunk_1",
        "chunk_3",
        "chunk_6"
    ]

    evaluate_retrieval(
        retrieved_ids,
        relevant_ids,
        k=5
    )
