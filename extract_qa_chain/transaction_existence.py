from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import requests
from tqdm import tqdm

from .config import ChainQAConfig
from .index_client import EthIndexClient, EthIndexError
from .pipeline import block_window, case_id, case_name, normalize_index_edge
from .schema import ChainQATask, ObservedEdge, normalize_address, stable_hash


@dataclass
class VerifiedNeighborQuery:
    edges: list[ObservedEdge]
    complete: bool
    error: str | None = None


class TransactionQueryCache:
    def __init__(
        self,
        *,
        case_id_: str,
        client: EthIndexClient,
        block_min: int,
        block_max: int,
        query_limit: int,
        expected_queries: int = 0,
        show_progress: bool = True,
    ) -> None:
        self.case_id = case_id_
        self.client = client
        self.block_min = block_min
        self.block_max = block_max
        self.query_limit = query_limit
        self.cache: dict[tuple[str, int], VerifiedNeighborQuery] = {}
        self.query_errors: dict[tuple[str, int], str] = {}
        self.query_durations: dict[tuple[str, int], float] = {}
        self.progress = tqdm(
            total=expected_queries,
            desc=f"{case_id_} tx oracle",
            unit="query",
            dynamic_ncols=True,
            disable=True if not show_progress else None,
        )

    def close(self) -> None:
        self.progress.close()

    def get(self, src: str, edge_type: int) -> VerifiedNeighborQuery:
        key = (src, edge_type)
        if key in self.cache:
            return self.cache[key]
        if self.progress.n >= (self.progress.total or 0):
            self.progress.total = self.progress.n + 1
        self.progress.set_postfix_str(
            f"src={src[:10]}...{src[-6:]} type={edge_type}", refresh=True
        )
        started = time.perf_counter()
        try:
            raw_rows, complete = self.client.complete_neighbors(
                src,
                direction="out",
                block_min=self.block_min,
                block_max=self.block_max,
                edge_type=edge_type,
                limit=self.query_limit,
            )
        except (requests.RequestException, EthIndexError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = VerifiedNeighborQuery([], False, error)
            self.cache[key] = result
            self.query_errors[key] = error
            self.query_durations[key] = time.perf_counter() - started
            self.progress.update(1)
            return result
        normalized: dict[str, ObservedEdge] = {}
        for raw_row in raw_rows:
            edge = normalize_index_edge(
                raw_row,
                case_id_=self.case_id,
                seed=src,
                hop=1,
            )
            if edge is None or edge.src != src:
                continue
            if normalize_edge_type(edge.edge_type) != edge_type:
                continue
            block = normalize_int(edge.block_number)
            if block is None or block < self.block_min or block > self.block_max:
                continue
            normalized[edge.edge_id] = edge
        result = VerifiedNeighborQuery(list(normalized.values()), complete)
        self.cache[key] = result
        duration = time.perf_counter() - started
        self.query_durations[key] = duration
        self.progress.set_postfix_str(
            f"last={duration:.1f}s rows={len(raw_rows)}", refresh=False
        )
        self.progress.update(1)
        if duration >= 5.0:
            print(
                f"\n{self.case_id}: slow tx oracle query src={src} "
                f"edge_type={edge_type} duration={duration:.2f}s "
                f"rows={len(raw_rows)} complete={complete} "
                f"blocks={self.block_min}-{self.block_max} "
                f"limit={self.query_limit}",
                file=sys.stderr,
                flush=True,
            )
        return result


def normalize_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_edge_type(value: Any) -> int | None:
    return normalize_int(value)


def transaction_window(
    case: dict[str, Any], edges: list[ObservedEdge], padding: int
) -> tuple[int | None, int | None]:
    block_min, block_max = block_window(case["evidence"], padding)
    if block_min is not None and block_max is not None:
        return block_min, block_max
    blocks = [
        block
        for block in (normalize_int(edge.block_number) for edge in edges)
        if block is not None
    ]
    if not blocks:
        return None, None
    return max(0, min(blocks) - padding), max(blocks) + padding


def matching_edges(
    query: VerifiedNeighborQuery, dst: str
) -> list[ObservedEdge]:
    return [edge for edge in query.edges if edge.dst == dst]


def edge_oracle(edges: list[ObservedEdge]) -> dict[str, Any]:
    edge_ids = sorted(
        {
            str(
                edge.raw_index_edge_id
                if edge.raw_index_edge_id is not None
                else edge.edge_id
            )
            for edge in edges
        }
    )
    tx_hashes = sorted({edge.tx_hash for edge in edges if edge.tx_hash})
    return {
        "exists": bool(edges),
        "transaction_count": len(tx_hashes),
        "matching_edge_count": len(edge_ids),
        "matching_edge_ids": edge_ids,
        "matching_tx_hashes": tx_hashes,
    }


def query_context(
    *, src: str, dst: str, block_min: int, block_max: int, edge_type: int
) -> dict[str, Any]:
    return {
        "src": src,
        "dst": dst,
        "chain": "ethereum",
        "direction": "out",
        "block_min": block_min,
        "block_max": block_max,
        "edge_types": [edge_type],
        "edge_type_mode": "any_of",
        "require_tx_hashes": True,
        "response_schema": {
            "exists": "boolean",
            "transaction_count": "integer",
            "matching_edge_ids": ["string"],
            "matching_tx_hashes": ["0x-prefixed transaction hash"],
        },
    }


def build_task(
    *,
    case: dict[str, Any],
    dataset_version: str,
    src: str,
    dst: str,
    block_min: int,
    block_max: int,
    edge_type: int,
    oracle: dict[str, Any],
    negative_sample_type: str | None,
) -> ChainQATask:
    cid = case_id(case)
    context = query_context(
        src=src,
        dst=dst,
        block_min=block_min,
        block_max=block_max,
        edge_type=edge_type,
    )
    answer_value = {
        **oracle,
        "verified_absent": not oracle["exists"],
        "negative_sample_type": negative_sample_type,
        "verification_query": {
            key: context[key]
            for key in (
                "src",
                "dst",
                "chain",
                "direction",
                "block_min",
                "block_max",
                "edge_types",
            )
        },
        "oracle_source": "local_ethereum_index_complete_neighbors",
    }
    task_hash = stable_hash(
        [src, dst, block_min, block_max, edge_type, negative_sample_type]
    )
    return ChainQATask(
        id=f"{cid}:direct_transaction_existence:{task_hash}",
        dataset_version=dataset_version,
        case_id=cid,
        case_dir=case["case_dir"],
        case_name=case_name(case),
        task_type="direct_transaction_existence",
        difficulty="medium" if negative_sample_type else "easy",
        question=(
            f"在 Ethereum 区块 {block_min} 到 {block_max} 范围内，仅考虑 edge_type={edge_type}，"
            f"地址 {src} 是否直接向地址 {dst} 发生过至少一笔交易或资金关系？"
        ),
        answer=json.dumps(answer_value, ensure_ascii=False, sort_keys=True),
        answer_value=answer_value,
        answer_format="object",
        graph_context=context,
        edge_summaries=[],
        labels={
            "existence_oracle": oracle["exists"],
            "negative_sample_type": negative_sample_type,
        },
        evidence={
            "chain_edges": [],
            "reported_edges": [],
            "verification_status": "query_required",
        },
        evaluation={
            "metric": [
                "existence_accuracy",
                "transaction_count_accuracy",
                "edge_id_precision",
                "edge_id_recall",
                "tx_hash_precision",
                "tx_hash_recall",
            ],
            "primary_metric": "existence_accuracy",
            "scorer": "deterministic_transaction_existence_v1",
        },
        link_semantics="transaction_existence",
        question_type="<s,p,o>",
        agent_profile="tool_grounded",
        required_tools=["eth_neighbors"],
        allowed_tools=["eth_neighbors", "eth_edge"],
        tool_requirement="required",
        public_context_profile="query_only",
    )


def deterministic_order(
    values: list[tuple[str, str, int, str]], cid: str, category: str
) -> list[tuple[str, str, int, str]]:
    return sorted(
        values,
        key=lambda item: stable_hash([cid, "tx-existence-negative", category, item]),
    )


def build_transaction_existence_tasks(
    case: dict[str, Any],
    edges: list[ObservedEdge],
    config: ChainQAConfig,
    client: EthIndexClient,
) -> tuple[list[ChainQATask], dict[str, Any]]:
    cid = case_id(case)
    block_min, block_max = transaction_window(case, edges, config.block_padding)
    if block_min is None or block_max is None:
        return [], {
            "ok": False,
            "errors": ["transaction existence block window is unavailable"],
            "task_count": 0,
            "query_count": 0,
        }
    observed_keys = sorted(
        {
            (edge.src, edge.dst, edge_type)
            for edge in edges
            if (edge_type := normalize_edge_type(edge.edge_type)) is not None
        },
        key=lambda item: stable_hash([cid, "tx-existence-positive", item]),
    )
    cache = TransactionQueryCache(
        case_id_=cid,
        client=client,
        block_min=block_min,
        block_max=block_max,
        query_limit=config.transaction_query_limit,
        expected_queries=len({(src, edge_type) for src, _, edge_type in observed_keys}),
        show_progress=config.show_progress,
    )
    tasks: list[ChainQATask] = []
    positives: list[tuple[str, str, int]] = []
    for src, dst, edge_type in observed_keys:
        query = cache.get(src, edge_type)
        if not query.complete:
            continue
        matches = matching_edges(query, dst)
        oracle = edge_oracle(matches)
        if not oracle["exists"] or not oracle["matching_tx_hashes"]:
            continue
        positives.append((src, dst, edge_type))
        tasks.append(
            build_task(
                case=case,
                dataset_version=config.dataset_version,
                src=src,
                dst=dst,
                block_min=block_min,
                block_max=block_max,
                edge_type=edge_type,
                oracle=oracle,
                negative_sample_type=None,
            )
        )

    if not positives or config.transaction_neg_ratio <= 0:
        validation = validate_transaction_tasks_with_cache(tasks, cache)
        cache.close()
        return tasks, validation
    nodes = sorted({edge.src for edge in edges} | {edge.dst for edge in edges})
    edge_types = sorted({edge_type for _, _, edge_type in positives})
    pools: dict[str, list[tuple[str, str, int, str]]] = {
        "reversed_direction": [],
        "wrong_edge_type": [],
        "same_source_non_neighbor": [],
        "same_case_non_edge": [],
    }
    for src, dst, edge_type in positives:
        pools["reversed_direction"].append(
            (dst, src, edge_type, "reversed_direction")
        )
        for other_type in edge_types:
            if other_type != edge_type:
                pools["wrong_edge_type"].append(
                    (src, dst, other_type, "wrong_edge_type")
                )
    for src, edge_type in sorted({(src, edge_type) for src, _, edge_type in positives}):
        existing = {edge.dst for edge in cache.get(src, edge_type).edges}
        for dst in nodes:
            if dst != src and dst not in existing:
                category = (
                    "same_case_non_edge"
                    if int(stable_hash([cid, src, dst, edge_type], length=2), 16) % 4 == 0
                    else "same_source_non_neighbor"
                )
                pools[category].append((src, dst, edge_type, category))

    for category, values in pools.items():
        deduplicated = list(dict.fromkeys(values))
        pools[category] = deterministic_order(deduplicated, cid, category)
    pools["reversed_direction"] = pools["reversed_direction"][:8]

    target = len(positives) * config.transaction_neg_ratio
    selected_keys: set[tuple[str, str, int]] = set(positives)
    offsets = Counter()
    negative_count = 0

    def add_from_pool(category: str, quota: int) -> int:
        added = 0
        pool = pools[category]
        while offsets[category] < len(pool) and added < quota:
            src, dst, edge_type, negative_type = pool[offsets[category]]
            offsets[category] += 1
            key = (src, dst, edge_type)
            if key in selected_keys:
                continue
            query = cache.get(src, edge_type)
            if not query.complete or matching_edges(query, dst):
                continue
            selected_keys.add(key)
            tasks.append(
                build_task(
                    case=case,
                    dataset_version=config.dataset_version,
                    src=src,
                    dst=dst,
                    block_min=block_min,
                    block_max=block_max,
                    edge_type=edge_type,
                    oracle=edge_oracle([]),
                    negative_sample_type=negative_type,
                )
            )
            added += 1
        return added

    planned = [
        ("wrong_edge_type", max(1, target // 4)),
        ("reversed_direction", min(4, max(1, target // 20))),
        ("same_case_non_edge", max(1, target // 5)),
        ("same_source_non_neighbor", target),
    ]
    for category, quota in planned:
        remaining = target - negative_count
        if remaining <= 0:
            break
        negative_count += add_from_pool(category, min(quota, remaining))
    for category in (
        "same_source_non_neighbor",
        "same_case_non_edge",
        "wrong_edge_type",
        "reversed_direction",
    ):
        remaining = target - negative_count
        if remaining <= 0:
            break
        negative_count += add_from_pool(category, remaining)
    validation = validate_transaction_tasks_with_cache(tasks, cache)
    cache.close()
    return tasks, validation


def generate_transaction_existence_tasks(
    case: dict[str, Any],
    edges: list[ObservedEdge],
    config: ChainQAConfig,
    client: EthIndexClient,
) -> list[ChainQATask]:
    tasks, _ = build_transaction_existence_tasks(case, edges, config, client)
    return tasks


def validate_transaction_tasks_with_cache(
    target_tasks: list[ChainQATask], cache: TransactionQueryCache
) -> dict[str, Any]:
    errors: list[str] = []
    if not target_tasks:
        errors.append("no direct_transaction_existence tasks generated")
    for task in target_tasks:
        context = task.graph_context
        src = normalize_address(context.get("src"))
        dst = normalize_address(context.get("dst"))
        edge_types = context.get("edge_types") or []
        if not src or not dst or len(edge_types) != 1:
            errors.append(f"invalid transaction query context: {task.id}")
            continue
        query = cache.get(src, int(edge_types[0]))
        if not query.complete:
            errors.append(f"transaction verification query incomplete: {task.id}")
            continue
        expected = edge_oracle(matching_edges(query, dst))
        actual = task.answer_value if isinstance(task.answer_value, dict) else {}
        for key in (
            "exists",
            "transaction_count",
            "matching_edge_count",
            "matching_edge_ids",
            "matching_tx_hashes",
        ):
            if actual.get(key) != expected.get(key):
                errors.append(f"transaction oracle mismatch {key}: {task.id}")
        if not expected["exists"] and actual.get("verified_absent") is not True:
            errors.append(f"negative task is not verified absent: {task.id}")
        if expected["exists"] and not expected["matching_tx_hashes"]:
            errors.append(f"positive task missing tx hashes: {task.id}")
    return {
        "ok": not errors,
        "errors": errors,
        "task_count": len(target_tasks),
        "query_count": len(cache.cache),
        "query_error_count": len(cache.query_errors),
        "query_errors": [
            {"src": src, "edge_type": edge_type, "error": error}
            for (src, edge_type), error in sorted(cache.query_errors.items())
        ],
        "query_total_seconds": round(sum(cache.query_durations.values()), 3),
        "query_max_seconds": round(max(cache.query_durations.values(), default=0.0), 3),
        "query_avg_seconds": round(
            sum(cache.query_durations.values()) / len(cache.query_durations), 3
        )
        if cache.query_durations
        else 0.0,
        "slow_query_count": sum(
            duration >= 5.0 for duration in cache.query_durations.values()
        ),
        "slow_queries": [
            {
                "src": src,
                "edge_type": edge_type,
                "duration_seconds": round(duration, 3),
                "error": cache.query_errors.get((src, edge_type)),
            }
            for (src, edge_type), duration in sorted(
                cache.query_durations.items(), key=lambda item: item[1], reverse=True
            )
            if duration >= 5.0
        ],
        "positive_count": sum(bool(task.answer_value.get("exists")) for task in target_tasks),
        "negative_count": sum(not bool(task.answer_value.get("exists")) for task in target_tasks),
        "negative_sample_type_counts": dict(
            Counter(
                task.answer_value.get("negative_sample_type")
                for task in target_tasks
                if not task.answer_value.get("exists")
            )
        ),
    }


def validate_transaction_existence_tasks(
    tasks: list[ChainQATask],
    client: EthIndexClient,
    *,
    query_limit: int,
) -> dict[str, Any]:
    target_tasks = [
        task for task in tasks if task.task_type == "direct_transaction_existence"
    ]
    if not target_tasks:
        return {
            "ok": False,
            "errors": ["no direct_transaction_existence tasks generated"],
            "task_count": 0,
            "query_count": 0,
        }
    first = target_tasks[0]
    first_context = first.graph_context
    cache = TransactionQueryCache(
        case_id_=first.case_id,
        client=client,
        block_min=int(first_context["block_min"]),
        block_max=int(first_context["block_max"]),
        query_limit=query_limit,
        expected_queries=len(
            {
                (
                    normalize_address(task.graph_context.get("src")),
                    int(task.graph_context["edge_types"][0]),
                )
                for task in target_tasks
            }
        ),
        show_progress=False,
    )
    validation = validate_transaction_tasks_with_cache(target_tasks, cache)
    cache.close()
    return validation
