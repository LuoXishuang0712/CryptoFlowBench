from __future__ import annotations

from typing import Any

from .models import ACTIONS, AlgorithmOutput


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _set_metrics(predicted: set[str], gold: set[str]) -> dict[str, float]:
    true_positive = len(predicted & gold)
    precision = _safe_div(true_positive, len(predicted))
    recall = _safe_div(true_positive, len(gold))
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "over_expansion_rate": _safe_div(len(predicted - gold), len(predicted)),
    }


def _gold_actions(task: dict[str, Any]) -> dict[str, str]:
    answer = task.get("answer_value") or {}
    if isinstance(answer, dict) and isinstance(answer.get("candidate_actions"), dict):
        return {str(key): str(value) for key, value in answer["candidate_actions"].items()}
    if isinstance(answer, dict) and isinstance(answer.get("state_action_labels"), dict):
        return {
            str(edge_id): str(action)
            for actions in answer["state_action_labels"].values()
            if isinstance(actions, dict)
            for edge_id, action in actions.items()
        }
    if isinstance(answer, dict) and all(str(value) in ACTIONS for value in answer.values()):
        return {str(key): str(value) for key, value in answer.items()}
    return {}


def _gold_follow(task: dict[str, Any], actions: dict[str, str]) -> set[str]:
    answer = task.get("answer_value") or {}
    if isinstance(answer, dict):
        for key in ("correct_follow_edges", "missing_next_edges", "gold_follow_edges"):
            if isinstance(answer.get(key), list):
                return {str(value) for value in answer[key]}
    return {edge_id for edge_id, action in actions.items() if action == "follow"}


def _macro_f1(gold: dict[str, str], predicted: dict[str, str]) -> float:
    active = [action for action in ACTIONS if action in set(gold.values()) | set(predicted.values())]
    scores: list[float] = []
    for action in active:
        tp = sum(1 for key, value in gold.items() if value == action and predicted.get(key) == action)
        fp = sum(1 for key, value in predicted.items() if value == action and gold.get(key) != action)
        fn = sum(1 for key, value in gold.items() if value == action and predicted.get(key) != action)
        scores.append(_safe_div(2 * tp, 2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def _follow_ece(samples: list[tuple[float, float]], bins: int = 10) -> float:
    if not samples:
        return 0.0
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            sample
            for sample in samples
            if lower <= sample[0] <= upper and (index == bins - 1 or sample[0] < upper)
        ]
        if bucket:
            confidence = sum(sample[0] for sample in bucket) / len(bucket)
            accuracy = sum(sample[1] for sample in bucket) / len(bucket)
            error += len(bucket) / len(samples) * abs(confidence - accuracy)
    return error


def evaluate_task(task: dict[str, Any], output: AlgorithmOutput) -> dict[str, Any]:
    task_type = str(task.get("task_type") or "")
    predictions = {item.edge_id: item for item in output.edge_predictions}
    predicted_actions = {edge_id: item.action for edge_id, item in predictions.items()}
    gold_actions = _gold_actions(task)
    gold_follow = _gold_follow(task, gold_actions)
    predicted_follow = set(output.selected_follow_edges)
    metrics: dict[str, Any] = {
        "task_type": task_type,
        "candidate_count": len(predictions),
        "gold_follow_count": len(gold_follow),
        "predicted_follow_count": len(predicted_follow),
    }
    all_predictions = [
        prediction
        for mode_predictions in output.edge_predictions_by_mode.values()
        for prediction in mode_predictions
        if prediction.edge_id in gold_actions
    ] or [prediction for prediction in output.edge_predictions if prediction.edge_id in gold_actions]
    if all_predictions:
        metrics["brier_score"] = sum(
            sum(
                (
                    prediction.action_probabilities.get(action, 0.0)
                    - (1.0 if gold_actions[prediction.edge_id] == action else 0.0)
                )
                ** 2
                for action in ACTIONS
            )
            for prediction in all_predictions
        ) / len(all_predictions)
        metrics["ece"] = _follow_ece(
            [
                (
                    prediction.follow_probability,
                    1.0 if gold_actions[prediction.edge_id] == "follow" else 0.0,
                )
                for prediction in all_predictions
            ]
        )

    if task_type == "edge_action_classification":
        correct = sum(
            1 for edge_id, action in gold_actions.items() if predicted_actions.get(edge_id) == action
        )
        metrics.update(_set_metrics(predicted_follow, gold_follow))
        metrics["action_accuracy"] = _safe_div(correct, len(gold_actions))
        metrics["action_macro_f1"] = _macro_f1(gold_actions, predicted_actions)
        if predictions and gold_actions:
            metrics["brier_score"] = sum(
                sum(
                    (
                        prediction.action_probabilities.get(action, 0.0)
                        - (1.0 if gold_actions.get(edge_id) == action else 0.0)
                    )
                    ** 2
                    for action in ACTIONS
                )
                for edge_id, prediction in predictions.items()
            ) / len(predictions)

    elif task_type == "path_completion":
        free_paths = output.paths_by_mode.get("free_rollout", output.paths)
        free_follow = {edge_id for path in free_paths for edge_id in path.edge_ids}
        metrics.update({f"path_{key}": value for key, value in _set_metrics(free_follow, gold_follow).items()})
        ranked = sorted(output.edge_predictions, key=lambda item: (-item.follow_probability, item.edge_id))
        metrics["mrr"] = next(
            (1.0 / rank for rank, item in enumerate(ranked, 1) if item.edge_id in gold_follow),
            0.0,
        )
        gold_paths = {
            tuple(path) for path in (task.get("answer_value") or {}).get("gold_paths") or []
        }
        predicted_paths = {tuple(path.edge_ids) for path in free_paths}
        metrics["complete_path_success"] = (
            bool(predicted_paths & gold_paths)
            if gold_paths
            else bool(gold_follow) and gold_follow <= free_follow
        )
        teacher_predictions = output.edge_predictions_by_mode.get("teacher_forced", [])
        teacher_actions = {item.edge_id: item.action for item in teacher_predictions}
        metrics["teacher_forced_next_step_accuracy"] = (
            sum(teacher_actions.get(edge_id) == action for edge_id, action in gold_actions.items())
            / len(gold_actions)
            if gold_actions
            else 0.0
        )
        for key, value in output.budget_usage_by_mode.get("free_rollout", {}).items():
            metrics[f"free_rollout_{key}"] = value

    elif task_type == "subgraph_noise_filtering":
        answer = task.get("answer_value") or {}
        gold_seed_nodes = {
            str(value).lower() for value in answer.get("gold_seed_nodes") or []
        }
        if not gold_seed_nodes:
            legacy_seed = str((task.get("graph_context") or {}).get("current_node") or "").lower()
            gold_seed_nodes = {legacy_seed} if legacy_seed else set()
        ranked_nodes = [item.node_id.lower() for item in output.node_predictions]
        selected_seeds = {value.lower() for value in output.selected_seed_nodes}
        metrics["gold_seed_available"] = bool(gold_seed_nodes)
        metrics["seed_hit_at_1"] = (
            bool(ranked_nodes) and ranked_nodes[0] in gold_seed_nodes
            if gold_seed_nodes
            else not selected_seeds
        )
        metrics["seed_mrr"] = next(
            (1.0 / rank for rank, node in enumerate(ranked_nodes, 1) if node in gold_seed_nodes),
            0.0,
        )
        predicted_paths = output.paths_by_mode.get("predicted_seed", output.paths)
        oracle_paths = output.paths_by_mode.get("oracle_seed", [])
        predicted_path_edges = {edge_id for path in predicted_paths for edge_id in path.edge_ids}
        oracle_path_edges = {edge_id for path in oracle_paths for edge_id in path.edge_ids}
        predicted_metrics = _set_metrics(predicted_path_edges, gold_follow)
        oracle_metrics = _set_metrics(oracle_path_edges, gold_follow)
        metrics.update({f"path_{key}": value for key, value in predicted_metrics.items()})
        metrics["predicted_seed_end_to_end_f1"] = predicted_metrics["f1"]
        metrics["oracle_seed_end_to_end_f1"] = oracle_metrics["f1"]
        metrics["none_of_above_accuracy"] = (
            not predicted_path_edges and not selected_seeds if not gold_follow else None
        )
        for mode, usage in output.budget_usage_by_mode.items():
            for key, value in usage.items():
                metrics[f"{mode}_{key}"] = value

    return {
        "scorer": "comparison_experiment_deterministic_v1",
        "metrics": metrics,
        "passed": _primary_pass(task_type, metrics),
    }


def _primary_pass(task_type: str, metrics: dict[str, Any]) -> bool:
    if task_type == "edge_action_classification":
        return bool(metrics.get("recall", 0.0) > 0.0) if metrics.get("gold_follow_count") else bool(
            metrics.get("predicted_follow_count") == 0
        )
    if task_type == "path_completion":
        return bool(metrics.get("complete_path_success"))
    if task_type == "subgraph_noise_filtering":
        seed_ok = metrics.get("seed_hit_at_1") if metrics.get("gold_seed_available") else True
        path_ok = (
            metrics.get("path_recall", 0.0) > 0.0
            if metrics.get("gold_follow_count")
            else metrics.get("predicted_follow_count") == 0
        )
        return bool(seed_ok and path_ok)
    return False
