from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Iterable

from .models import normalize_node, numeric_amount


WHOLE_GRAPH_NODE_FEATURE_NAMES = (
    "log_out_edges",
    "log_in_edges",
    "log_out_counterparties",
    "log_in_counterparties",
    "log_native_out_amount",
    "log_native_in_amount",
    "log_max_native_out_amount",
    "log_max_native_in_amount",
    "out_native_share",
    "out_token_share",
    "out_other_share",
    "in_native_share",
    "in_token_share",
    "in_other_share",
    "log_activity_span_seconds",
    "log_self_loops",
)


def _bucket(edge: dict[str, Any]) -> str:
    edge_type = str(edge.get("edge_type") or "")
    if edge_type == "0" and not edge.get("token_address"):
        return "native"
    if edge_type == "1" or edge.get("token_address"):
        return "token"
    return "other"


def _empty_node(node_id: str) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "out_edges": 0,
        "in_edges": 0,
        "out_counterparties": set(),
        "in_counterparties": set(),
        "native_out_amount": 0.0,
        "native_in_amount": 0.0,
        "max_native_out_amount": 0.0,
        "max_native_in_amount": 0.0,
        "out_buckets": defaultdict(int),
        "in_buckets": defaultdict(int),
        "timestamps": [],
        "self_loops": 0,
        "positive_out_edges": 0,
        "unknown_out_edges": 0,
    }


def build_whole_graph_node_rows(
    edges: Iterable[dict[str, Any]],
    *,
    labels_by_edge: dict[str, dict[str, Any]] | None = None,
    include_labels: bool,
) -> list[dict[str, Any]]:
    """Build one public-feature row per node in a complete case subgraph.

    Hidden edge labels are consulted only when ``include_labels`` is true for
    declared training cases. Target-case inference never needs labels.
    """

    nodes: dict[str, dict[str, Any]] = {}
    for edge in edges:
        src = normalize_node(edge.get("src"))
        dst = normalize_node(edge.get("dst"))
        if not src or not dst:
            continue
        src_row = nodes.setdefault(src, _empty_node(src))
        dst_row = nodes.setdefault(dst, _empty_node(dst))
        bucket = _bucket(edge)
        amount = numeric_amount(edge.get("amount"))
        timestamp = numeric_amount(edge.get("timestamp"))
        src_row["out_edges"] += 1
        dst_row["in_edges"] += 1
        src_row["out_counterparties"].add(dst)
        dst_row["in_counterparties"].add(src)
        src_row["out_buckets"][bucket] += 1
        dst_row["in_buckets"][bucket] += 1
        if timestamp > 0.0:
            src_row["timestamps"].append(timestamp)
            dst_row["timestamps"].append(timestamp)
        if src == dst:
            src_row["self_loops"] += 1
        if bucket == "native":
            src_row["native_out_amount"] += amount
            dst_row["native_in_amount"] += amount
            src_row["max_native_out_amount"] = max(
                src_row["max_native_out_amount"], amount
            )
            dst_row["max_native_in_amount"] = max(
                dst_row["max_native_in_amount"], amount
            )
        if include_labels:
            label = (labels_by_edge or {}).get(str(edge.get("edge_id") or ""))
            usable = bool(label and label.get("usable_for_eval", True))
            action = str((label or {}).get("action_label") or "")
            label_name = str((label or {}).get("label") or "")
            if usable and action == "follow":
                src_row["positive_out_edges"] += 1
            elif not usable or not action or label_name == "uncertain":
                src_row["unknown_out_edges"] += 1

    output: list[dict[str, Any]] = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        if (
            include_labels
            and node["unknown_out_edges"]
            and not node["positive_out_edges"]
        ):
            continue
        out_edges = int(node["out_edges"])
        in_edges = int(node["in_edges"])
        timestamps = node["timestamps"]
        span = max(timestamps) - min(timestamps) if timestamps else 0.0
        row = {
            "node_id": node_id,
            "features": [
                math.log1p(out_edges),
                math.log1p(in_edges),
                math.log1p(len(node["out_counterparties"])),
                math.log1p(len(node["in_counterparties"])),
                math.log1p(node["native_out_amount"]),
                math.log1p(node["native_in_amount"]),
                math.log1p(node["max_native_out_amount"]),
                math.log1p(node["max_native_in_amount"]),
                node["out_buckets"]["native"] / out_edges if out_edges else 0.0,
                node["out_buckets"]["token"] / out_edges if out_edges else 0.0,
                node["out_buckets"]["other"] / out_edges if out_edges else 0.0,
                node["in_buckets"]["native"] / in_edges if in_edges else 0.0,
                node["in_buckets"]["token"] / in_edges if in_edges else 0.0,
                node["in_buckets"]["other"] / in_edges if in_edges else 0.0,
                math.log1p(max(0.0, span)),
                math.log1p(node["self_loops"]),
            ],
        }
        if include_labels:
            row["label"] = int(node["positive_out_edges"] > 0)
        output.append(row)
    return output


def sample_positive_rate(
    rows: list[dict[str, Any]],
    *,
    positive_rate: float,
    seed: int,
    fold_identity: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < positive_rate < 1.0:
        raise ValueError("positive_rate must be within (0, 1)")
    positives = [row for row in rows if int(row.get("label") or 0) == 1]
    negatives = [row for row in rows if int(row.get("label") or 0) == 0]
    if not positives or not negatives:
        raise ValueError(
            "whole-subgraph node training needs both classes; "
            f"positive={len(positives)} negative={len(negatives)}"
        )
    wanted_negatives = math.floor(
        len(positives) * (1.0 - positive_rate) / positive_rate
    )
    limited = wanted_negatives >= len(negatives)
    wanted_negatives = min(wanted_negatives, len(negatives))

    def order_key(row: dict[str, Any]) -> str:
        payload = (
            f"{seed}\x1f{fold_identity}\x1f{row.get('case_id')}\x1f"
            f"{row.get('node_id')}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    sampled = positives + sorted(negatives, key=order_key)[:wanted_negatives]
    sampled.sort(key=lambda row: (str(row.get("case_id") or ""), row["node_id"]))
    return sampled, {
        "requested_positive_rate": positive_rate,
        "achieved_positive_rate": len(positives) / len(sampled),
        "positive_count": len(positives),
        "negative_count": len(sampled) - len(positives),
        "available_negative_count": len(negatives),
        "limited_by_available_negatives": limited,
    }
