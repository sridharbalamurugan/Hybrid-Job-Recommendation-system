import numpy as np
from sklearn.metrics import ndcg_score
from src.pipelines.retrival import retrieve_hybrid


def precision_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    return len([job for job in retrieved_k if job in relevant_set]) / k


def recall_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    return len([job for job in retrieved_k if job in relevant_set]) / len(relevant_set)


def evaluate_recommendations(user_job_dict, model_retrieval_fn, k=5):
    """
    Parameters:
        user_job_dict (dict): {user_query: [relevant_job_ids]}
        model_retrieval_fn (func): function that takes a query and returns ranked job_ids
        k (int): cutoff rank

    Returns:
        dict: avg precision@k, recall@k, ndcg@k
    """
    precisions, recalls, ndcgs = [], [], []

    for query, relevant in user_job_dict.items():
        retrieved = model_retrieval_fn(query)

        precisions.append(precision_at_k(relevant, retrieved, k))
        recalls.append(recall_at_k(relevant, retrieved, k))

        # Binary relevance array for sklearn ndcg_score
        y_true = np.zeros((1, len(retrieved)))
        y_true[0, :k] = [1 if job in relevant else 0 for job in retrieved[:k]]
        y_score = np.linspace(1, 0, len(retrieved)).reshape(1, -1)  # higher rank = higher score

        ndcgs.append(ndcg_score(y_true, y_score, k=k))

    return {
        f"Precision@{k}": round(np.mean(precisions), 4),
        f"Recall@{k}": round(np.mean(recalls), 4),
        f"NDCG@{k}": round(np.mean(ndcgs), 4)
    }


# Example usage
if __name__ == "__main__":
    # Simulated ground truth and fake retrieval function
    user_job_dict = {
        "python data analysis": ["job1", "job3", "job5"],
        "frontend react": ["job7", "job8"]
    }

    def mock_retrieval_fn(query):
        return ["job1", "job2", "job3", "job4", "job5"]

    results = evaluate_recommendations(user_job_dict, mock_retrieval_fn, k=3)
    print("\n Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score}")
