from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from storage import MongoDocumentStore

from .config import ChainQAConfig
from .io import write_csv, write_json, write_jsonl
from .schema import CandidateSet, ChainQATask, EdgeLabel, ObservedEdge, SeedNode


TX_CSV_FIELDS = [
    "case_id",
    "edge_id",
    "tx_hash",
    "block_number",
    "timestamp",
    "src",
    "dst",
    "asset",
    "amount",
    "edge_type",
    "hop_from_seed",
    "src_is_black",
    "dst_is_black",
    "edge_label",
    "action_label",
    "stage",
    "is_boundary",
    "is_hard_negative",
    "reported_edge_id",
    "label_semantics",
]

MONGO_COLLECTION_SPECS = [
    ("chain_seed_nodes", "node"),
    ("chain_subgraph_edges", "edge_id"),
    ("chain_labels", "edge_id"),
    ("chain_candidate_sets", "state_id"),
    ("chain_qa_tasks", "id"),
]


def rows(value: list[Any]) -> list[dict[str, Any]]:
    out = []
    for item in value:
        out.append(asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item))
    return out


def mongo_safe(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if value < -(2**63) or value > 2**63 - 1:
            return str(value)
        return value
    if isinstance(value, float) or isinstance(value, str):
        return value
    if isinstance(value, list):
        return [mongo_safe(item) for item in value]
    if isinstance(value, tuple):
        return [mongo_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): mongo_safe(item) for key, item in value.items()}
    return value


def write_bundle(
    out_dir: Path,
    *,
    case: dict[str, Any],
    config: ChainQAConfig,
    seeds: list[SeedNode],
    edges: list[ObservedEdge],
    labels: list[EdgeLabel],
    candidates: list[CandidateSet],
    tasks: list[ChainQATask],
) -> dict[str, Any]:
    label_map = {label.edge_id: label for label in labels}
    black_nodes = set()
    for edge in edges:
        label = label_map.get(edge.edge_id)
        if label and label.label in {"positive", "reported_positive"}:
            black_nodes.add(edge.src)
            black_nodes.add(edge.dst)
    edge_rows = rows(edges)
    label_rows = rows(labels)
    candidate_rows = rows(candidates)
    task_rows = rows(tasks)
    write_jsonl(out_dir / "seed_nodes.jsonl", rows(seeds))
    write_jsonl(out_dir / "subgraph_edges.jsonl", edge_rows)
    write_jsonl(out_dir / "labels.jsonl", label_rows)
    write_jsonl(out_dir / "candidate_sets.jsonl", candidate_rows)
    write_jsonl(out_dir / "qa_tasks.jsonl", task_rows)
    tx_rows = []
    for edge in edges:
        label = label_map.get(edge.edge_id)
        tx_rows.append(
            {
                "case_id": edge.case_id,
                "edge_id": edge.edge_id,
                "tx_hash": edge.tx_hash,
                "block_number": edge.block_number,
                "timestamp": edge.timestamp,
                "src": edge.src,
                "dst": edge.dst,
                "asset": edge.asset,
                "amount": edge.amount,
                "edge_type": edge.edge_type,
                "hop_from_seed": edge.hop_from_seed,
                "src_is_black": edge.src in black_nodes,
                "dst_is_black": edge.dst in black_nodes,
                "edge_label": label.label if label else "",
                "action_label": label.action_label if label else "",
                "stage": label.stage if label else "",
                "is_boundary": bool(label and label.action_label == "stop"),
                "is_hard_negative": bool(label and label.label == "hard_negative"),
                "reported_edge_id": label.reported_edge_id if label else "",
                "label_semantics": label.label_semantics if label else "",
            }
        )
    write_csv(out_dir / "txs.csv", tx_rows, TX_CSV_FIELDS)
    manifest = {
        "dataset_version": config.dataset_version,
        "case_dir": case["case_dir"],
        "case_id": case["evidence"].get("case_id"),
        "case_name": case["evidence"].get("case_name"),
        "eth_index_url": config.eth_index_url,
        "k_hop": config.k_hop,
        "neg_ratio": config.neg_ratio,
        "transaction_neg_ratio": config.transaction_neg_ratio,
        "transaction_query_limit": config.transaction_query_limit,
        "skip_dte": config.skip_dte,
        "tracing_agent_profile": config.tracing_agent_profile,
        "counts": {
            "seeds": len(seeds),
            "unindexed_seeds": sum(not seed.usable_for_tracing for seed in seeds),
            "subgraph_edges": len(edges),
            "labels": len(labels),
            "candidate_sets": len(candidates),
            "qa_tasks": len(tasks),
        },
        "label_counts": dict(Counter(label.label for label in labels)),
        "action_counts": dict(Counter(label.action_label for label in labels)),
        "hop_counts": dict(Counter(str(edge.hop_from_seed) for edge in edges)),
        "missing_hop_count": sum(edge.hop_from_seed is None for edge in edges),
        "invalid_hop_count": sum(
            not isinstance(edge.hop_from_seed, int)
            or isinstance(edge.hop_from_seed, bool)
            or edge.hop_from_seed < 1
            or edge.hop_from_seed > config.k_hop
            for edge in edges
            if edge.hop_from_seed is not None
        ),
        "task_type_counts": dict(Counter(task.task_type for task in tasks)),
        "agent_profile_counts": dict(Counter(task.agent_profile for task in tasks)),
        "tool_requirement_counts": dict(
            Counter(task.tool_requirement for task in tasks)
        ),
        "transaction_existence_counts": dict(
            Counter(
                "positive" if task.answer_value.get("exists") else "negative"
                for task in tasks
                if task.task_type == "direct_transaction_existence"
            )
        ),
        "transaction_negative_sample_type_counts": dict(
            Counter(
                task.answer_value.get("negative_sample_type")
                for task in tasks
                if task.task_type == "direct_transaction_existence"
                and not task.answer_value.get("exists")
            )
        ),
        "link_semantics_counts": dict(Counter(task.link_semantics for task in tasks)),
        "files": {
            "seed_nodes": "seed_nodes.jsonl",
            "subgraph_edges": "subgraph_edges.jsonl",
            "labels": "labels.jsonl",
            "candidate_sets": "candidate_sets.jsonl",
            "qa_tasks": "qa_tasks.jsonl",
            "txs_csv": "txs.csv",
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def write_mongo(
    *,
    case: dict[str, Any],
    manifest: dict[str, Any],
    seeds: list[SeedNode],
    edges: list[ObservedEdge],
    labels: list[EdgeLabel],
    candidates: list[CandidateSet],
    tasks: list[ChainQATask],
    strict: bool = False,
) -> str | None:
    cid = str(case["evidence"].get("case_id") or case["case_dir"])
    try:
        purge_query = {"case_id": cid, "dataset_version": manifest["dataset_version"]}
        purge_counts: dict[str, int] = {}
        bundle_store = MongoDocumentStore("chain_subgraph_bundles")
        purge_counts["chain_subgraph_bundles"] = bundle_store.collection.delete_many(
            purge_query
        ).deleted_count
        for collection, _ in MONGO_COLLECTION_SPECS:
            store = MongoDocumentStore(collection)
            purge_counts[collection] = store.collection.delete_many(purge_query).deleted_count
        manifest["mongo_purge_counts"] = purge_counts

        bundle_store.replace_one(
            {"case_id": cid, "dataset_version": manifest["dataset_version"]},
            mongo_safe(
                {
                    "case_id": cid,
                    "dataset_version": manifest["dataset_version"],
                    "manifest": manifest,
                    "task_type_counts": manifest.get("task_type_counts", {}),
                    "counts": manifest.get("counts", {}),
                }
            ),
        )
        row_sets = {
            "chain_seed_nodes": rows(seeds),
            "chain_subgraph_edges": rows(edges),
            "chain_labels": rows(labels),
            "chain_candidate_sets": rows(candidates),
            "chain_qa_tasks": rows(tasks),
        }
        for collection, key_name in MONGO_COLLECTION_SPECS:
            store = MongoDocumentStore(collection)
            store.create_index([("case_id", 1), ("dataset_version", 1), (key_name, 1)], unique=True)
            for item in row_sets[collection]:
                item["dataset_version"] = manifest["dataset_version"]
                item.setdefault("case_id", cid)
                store.replace_one(
                    {
                        "case_id": item["case_id"],
                        "dataset_version": item["dataset_version"],
                        key_name: item[key_name],
                    },
                    mongo_safe(item),
                )
        return None
    except Exception as exc:
        if strict:
            raise
        return f"{type(exc).__name__}: {exc}"
