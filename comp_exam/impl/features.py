from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from .models import ACTIONS, CandidateEdge, normalize_node


EDGE_FEATURE_NAMES = (
    "log_amount",
    "log_edge_count",
    "log_tx_count",
    "amount_share_from_source",
    "amount_rank_from_source",
    "time_rank_from_source",
    "src_is_active",
    "src_out_degree",
    "src_in_degree",
    "dst_out_degree",
    "dst_in_degree",
    "edge_type_native",
    "edge_type_token",
    "edge_type_other",
    "hop_from_seed",
    "is_self_loop",
)

NODE_FEATURE_NAMES = (
    "log_out_degree",
    "log_in_degree",
    "log_out_amount",
    "log_in_amount",
    "log_max_out_amount",
    "log_max_in_amount",
    "out_counterparty_count",
    "in_counterparty_count",
    "out_in_amount_ratio",
    "out_in_degree_ratio",
    "candidate_seed_declared",
)

_FLOAT32_MAX = 3.4028235e38


def _safe_ratio(numerator: float, denominator: float) -> float:
    ratio = numerator / denominator if denominator else 0.0
    if not math.isfinite(ratio):
        return 0.0 if ratio < 0.0 else _FLOAT32_MAX
    return min(max(ratio, -_FLOAT32_MAX), _FLOAT32_MAX)


def _safe_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _log1p(value: Any) -> float:
    return math.log1p(max(0.0, _safe_number(value)))


def _edge_order(edge: CandidateEdge) -> tuple[float, float, str]:
    timestamp = _safe_number(edge.metadata.get("min_timestamp"))
    block_number = _safe_number(edge.metadata.get("min_block_number"))
    return (
        timestamp if timestamp > 0.0 else float("inf"),
        block_number if block_number > 0.0 else float("inf"),
        edge.candidate_id,
    )


def edge_feature_vector(
    edge: CandidateEdge,
    edges: list[CandidateEdge],
    active_nodes: Iterable[str],
) -> list[float]:
    active = {normalize_node(value) for value in active_nodes if value}
    src = normalize_node(edge.src)
    dst = normalize_node(edge.dst)
    outgoing = [item for item in edges if normalize_node(item.src) == src]
    total_out = sum(item.amount for item in outgoing)
    amount_ranked = sorted(outgoing, key=lambda item: (-item.amount, item.candidate_id))
    time_ranked = sorted(outgoing, key=_edge_order)
    amount_rank = next(
        (index for index, item in enumerate(amount_ranked, 1) if item.candidate_id == edge.candidate_id),
        len(outgoing) or 1,
    )
    time_rank = next(
        (index for index, item in enumerate(time_ranked, 1) if item.candidate_id == edge.candidate_id),
        len(outgoing) or 1,
    )
    out_degree: dict[str, int] = defaultdict(int)
    in_degree: dict[str, int] = defaultdict(int)
    for item in edges:
        out_degree[normalize_node(item.src)] += 1
        in_degree[normalize_node(item.dst)] += 1
    edge_types = {str(value) for value in edge.edge_types}
    native = float("0" in edge_types)
    token = float("1" in edge_types)
    other = float(bool(edge_types - {"0", "1"}))
    hop = _safe_number(edge.metadata.get("hop_from_seed"))
    return [
        _log1p(edge.amount),
        _log1p(edge.metadata.get("edge_count")),
        _log1p(edge.metadata.get("tx_count")),
        edge.amount / total_out if total_out > 0.0 else 0.0,
        1.0 / amount_rank,
        1.0 / time_rank,
        float(src in active),
        _log1p(out_degree[src]),
        _log1p(in_degree[src]),
        _log1p(out_degree[dst]),
        _log1p(in_degree[dst]),
        native,
        token,
        other,
        hop,
        float(bool(src) and src == dst),
    ]


def candidate_seed_nodes(task: dict[str, Any], edges: list[CandidateEdge]) -> list[str]:
    context = task.get("graph_context") or {}
    declared = [
        normalize_node(value) for value in context.get("candidate_seed_nodes") or [] if value
    ]
    if declared:
        return list(dict.fromkeys(declared))
    return list(
        dict.fromkeys(normalize_node(edge.src) for edge in edges if normalize_node(edge.src))
    )


def node_feature_vector(
    node: str,
    task: dict[str, Any],
    edges: list[CandidateEdge],
) -> list[float]:
    normalized = normalize_node(node)
    outgoing = [edge for edge in edges if normalize_node(edge.src) == normalized]
    incoming = [edge for edge in edges if normalize_node(edge.dst) == normalized]
    out_amount = sum(edge.amount for edge in outgoing)
    in_amount = sum(edge.amount for edge in incoming)
    declared = {
        normalize_node(value)
        for value in (task.get("graph_context") or {}).get("candidate_seed_nodes") or []
        if value
    }
    return [
        _log1p(len(outgoing)),
        _log1p(len(incoming)),
        _log1p(out_amount),
        _log1p(in_amount),
        _log1p(max((edge.amount for edge in outgoing), default=0.0)),
        _log1p(max((edge.amount for edge in incoming), default=0.0)),
        float(len({normalize_node(edge.dst) for edge in outgoing})),
        float(len({normalize_node(edge.src) for edge in incoming})),
        _safe_ratio(out_amount, in_amount + 1.0),
        len(outgoing) / (len(incoming) + 1.0),
        float(normalized in declared),
    ]


def public_active_nodes(task: dict[str, Any]) -> list[str]:
    context = task.get("graph_context") or {}
    values: list[Any] = []
    values.extend([context.get("start_node"), context.get("current_node")])
    values.extend(context.get("candidate_seed_nodes") or [])
    values.extend(
        state.get("current_node")
        for state in context.get("states") or []
        if isinstance(state, dict)
    )
    return list(dict.fromkeys(normalize_node(value) for value in values if value))


def training_active_nodes(task: dict[str, Any]) -> list[str]:
    """Teacher-force training-case SNF seeds; never reads target-case labels."""
    if task.get("task_type") == "subgraph_noise_filtering":
        seeds = sorted(gold_seed_nodes(task))
        if seeds:
            return seeds
    return public_active_nodes(task)


def gold_edge_actions(task: dict[str, Any]) -> dict[str, str]:
    answer = task.get("answer_value") or {}
    if not isinstance(answer, dict):
        return {}
    if isinstance(answer.get("candidate_actions"), dict):
        return {
            str(edge_id): str(action)
            for edge_id, action in answer["candidate_actions"].items()
            if str(action) in ACTIONS
        }
    if isinstance(answer.get("state_action_labels"), dict):
        return {
            str(edge_id): str(action)
            for actions in answer["state_action_labels"].values()
            if isinstance(actions, dict)
            for edge_id, action in actions.items()
            if str(action) in ACTIONS
        }
    if all(str(value) in ACTIONS for value in answer.values()):
        return {str(edge_id): str(action) for edge_id, action in answer.items()}
    return {}


def gold_seed_nodes(task: dict[str, Any]) -> set[str]:
    answer = task.get("answer_value") or {}
    if not isinstance(answer, dict):
        return set()
    return {
        normalize_node(value) for value in answer.get("gold_seed_nodes") or [] if value
    }
