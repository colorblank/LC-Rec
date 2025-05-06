import math
from typing import List, Dict, Optional, Tuple, Set


def get_topk_results(
    predictions: List[str],
    scores: List[float],
    targets: List[str],
    k: int,
    all_items: Optional[Set[str]] = None,
) -> List[List[int]]:
    """Processes predictions and scores to generate top-k results against targets.

    Args:
        predictions (List[str]): A list of predicted item strings.
        scores (List[float]): A list of scores corresponding to the predictions.
        targets (List[str]): A list of target item strings.
        k (int): The number of top results to consider for each target.
        all_items (Optional[Set[str]], optional): A set of all valid items. If provided,
            predictions not in this set will be heavily penalized. Defaults to None.

    Returns:
        List[List[int]]: A list of lists, where each inner list contains binary indicators
                         (1 for hit, 0 for miss) for the top-k predictions against the corresponding target.
    """
    results = []
    B = len(targets)
    # Extract item IDs from predictions (assuming format like "Response: item_id")
    predictions = [
        _.split("Response:")[-1].strip().replace(" ", "") for _ in predictions
    ]

    # Penalize predictions not in the valid item set
    if all_items is not None:
        # Create a mutable copy of scores to modify
        mutable_scores = list(scores)
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                mutable_scores[i] = -1000.0  # Use a float for consistency
        scores = mutable_scores  # Update scores with the modified list

    for b in range(B):
        # Get predictions and scores for the current batch item
        batch_seqs = predictions[b * k : (b + 1) * k]
        batch_scores = scores[b * k : (b + 1) * k]

        # Pair predictions with scores and sort by score in descending order
        pairs: List[Tuple[str, float]] = list(zip(batch_seqs, batch_scores))
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        target_item = targets[b]
        one_results = []
        # Generate binary hit/miss indicators for the sorted predictions
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results


def get_metrics_results(
    topk_results: List[List[int]], metrics: List[str]
) -> Dict[str, float]:
    """Calculates various evaluation metrics based on top-k results.

    Args:
        topk_results (List[List[int]]): The processed top-k results from get_topk_results.
        metrics (List[str]): A list of metric names to calculate (e.g., ['hit@10', 'ndcg@10']).

    Returns:
        Dict[str, float]: A dictionary mapping metric names to their calculated values.

    Raises:
        NotImplementedError: If an unsupported metric name is provided.
    """
    res: Dict[str, float] = {}
    num_results = len(topk_results)
    if num_results == 0:
        return {m: 0.0 for m in metrics}  # Return 0 for all metrics if no results

    for m in metrics:
        metric_lower = m.lower()
        if metric_lower.startswith("hit@"):
            k = int(metric_lower.split("@")[1])
            res[m] = hit_k(topk_results, k) / num_results
        elif metric_lower.startswith("ndcg@"):
            k = int(metric_lower.split("@")[1])
            res[m] = ndcg_k(topk_results, k) / num_results
        else:
            raise NotImplementedError(f"Metric '{m}' not implemented.")

    return res


def ndcg_k(topk_results: List[List[int]], k: int) -> float:
    """Calculates the Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        topk_results (List[List[int]]): The processed top-k results.
        k (int): The cutoff value for NDCG calculation.

    Returns:
        float: The total NDCG score across all results.
    """
    total_ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        dcg = 0.0
        for i in range(len(res)):
            # IDCG is 1 for a single relevant item at the top
            dcg += res[i] / math.log2(i + 2)  # Use log base 2 for standard NDCG
        # Ideal DCG (IDCG) for a single relevant item is 1 (1 / log2(1+1))
        # Since we only have one target item per row, IDCG is 1 if hit@k > 0, else 0.
        # However, the division by num_results in get_metrics_results handles the normalization.
        # Here we just sum the DCG values.
        total_ndcg += dcg
    return total_ndcg


def hit_k(topk_results: List[List[int]], k: int) -> float:
    """Calculates the Hit Rate (HR) at k.

    Args:
        topk_results (List[List[int]]): The processed top-k results.
        k (int): The cutoff value for Hit Rate calculation.

    Returns:
        float: The total number of hits across all results.
    """
    total_hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            total_hit += 1.0  # Use float for consistency
    return total_hit
