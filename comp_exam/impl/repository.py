from __future__ import annotations

from typing import Any, Protocol

from chain_qa_version import dataset_version_sort_key

from .models import CandidateEdge, numeric_amount
from .node_actionability import build_whole_graph_node_rows


RUNS_COLLECTION = "comparison_experiment_runs"
TASK_RESULTS_COLLECTION = "comparison_experiment_task_results"
STAGES_COLLECTION = "comparison_experiment_stage_results"


def _candidate_ids(task: dict[str, Any]) -> list[str]:
    context = task.get("graph_context") or {}
    ids = [str(value) for value in context.get("candidate_edges") or []]
    if ids:
        return list(dict.fromkeys(ids))
    for summary in task.get("edge_summaries") or []:
        if not isinstance(summary, dict):
            continue
        edge_id = summary.get("candidate_id") or summary.get("edge_id")
        if edge_id:
            ids.append(str(edge_id))
    return list(dict.fromkeys(ids))


def raw_edge_ids_for_task(task: dict[str, Any]) -> list[str]:
    context = task.get("graph_context") or {}
    groups = context.get("candidate_edge_groups") or {}
    ids: list[str] = []
    for candidate_id in _candidate_ids(task):
        group = groups.get(candidate_id) if isinstance(groups, dict) else None
        ids.extend(str(value) for value in group or [candidate_id])
    for summary in task.get("edge_summaries") or []:
        if isinstance(summary, dict):
            ids.extend(str(value) for value in summary.get("sample_edge_ids") or [])
    return list(dict.fromkeys(ids))


def build_candidate_edges(
    task: dict[str, Any], raw_edges: list[dict[str, Any]]
) -> list[CandidateEdge]:
    context = task.get("graph_context") or {}
    groups = context.get("candidate_edge_groups") or {}
    raw_by_id = {
        str(row.get("edge_id")): row
        for row in raw_edges
        if isinstance(row, dict) and row.get("edge_id")
    }
    summaries: dict[str, dict[str, Any]] = {}
    for row in task.get("edge_summaries") or []:
        if not isinstance(row, dict):
            continue
        edge_id = row.get("candidate_id") or row.get("edge_id")
        if edge_id:
            summaries[str(edge_id)] = row

    candidates: list[CandidateEdge] = []
    for candidate_id in _candidate_ids(task):
        summary = summaries.get(candidate_id, {})
        group_ids = list(
            dict.fromkeys(
                str(value)
                for value in (
                    (groups.get(candidate_id) if isinstance(groups, dict) else None)
                    or summary.get("sample_edge_ids")
                    or [candidate_id]
                )
            )
        )
        members = [raw_by_id[edge_id] for edge_id in group_ids if edge_id in raw_by_id]
        first = members[0] if members else raw_by_id.get(candidate_id, {})
        src = str(summary.get("src") or first.get("src") or first.get("src_address") or "")
        dst = str(summary.get("dst") or first.get("dst") or first.get("dst_address") or "")
        explicit_amount = next(
            (
                summary.get(key)
                for key in ("total_amount", "amount_sum", "amount")
                if summary.get(key) is not None
            ),
            None,
        )
        amount = (
            numeric_amount(explicit_amount)
            if explicit_amount is not None
            else sum(numeric_amount(row.get("amount")) for row in members)
        )
        edge_types = summary.get("edge_types") or [
            row.get("edge_type") for row in members if row.get("edge_type") is not None
        ]
        assets = list(
            dict.fromkeys(
                str(value)
                for value in (
                    summary.get("assets")
                    or [row.get("asset") for row in members if row.get("asset")]
                )
                if value
            )
        )
        token_addresses = list(
            dict.fromkeys(
                str(value).lower()
                for value in (
                    summary.get("token_addresses")
                    or [
                        row.get("token_address")
                        for row in members
                        if row.get("token_address")
                    ]
                )
                if value
            )
        )
        amount_comparable = (
            len(token_addresses) <= 1
            and (bool(token_addresses) or len(assets) <= 1)
        )
        if not amount_comparable:
            # Aggregated relation rows can contain different token units. Their
            # raw amounts must not be added or ranked as one economic quantity.
            amount = 0.0
        timestamps = [
            numeric_amount(row.get("timestamp"))
            for row in members
            if numeric_amount(row.get("timestamp")) > 0.0
        ]
        block_numbers = [
            numeric_amount(row.get("block_number"))
            for row in members
            if numeric_amount(row.get("block_number")) > 0.0
        ]
        hop = summary.get("hop_from_seed")
        if hop is None and members:
            hop = members[0].get("hop") or members[0].get("hop_from_seed")
        candidates.append(
            CandidateEdge(
                candidate_id=candidate_id,
                src=src,
                dst=dst,
                amount=amount,
                raw_edge_ids=tuple(group_ids),
                edge_types=tuple(dict.fromkeys(str(value) for value in edge_types)),
                metadata={
                    "aggregation": summary.get("aggregation") or "raw_edge",
                    "edge_count": summary.get("edge_count") or len(members) or len(group_ids),
                    "tx_count": summary.get("tx_count"),
                    "min_timestamp": min(timestamps) if timestamps else summary.get("timestamp"),
                    "max_timestamp": max(timestamps) if timestamps else summary.get("timestamp"),
                    "min_block_number": (
                        min(block_numbers) if block_numbers else summary.get("block_number")
                    ),
                    "max_block_number": (
                        max(block_numbers) if block_numbers else summary.get("block_number")
                    ),
                    "hop_from_seed": hop,
                    "assets": assets,
                    "token_addresses": token_addresses,
                    "amount_comparable": amount_comparable,
                },
            )
        )
    return candidates


class ExperimentRepository(Protocol):
    def load_tasks(
        self,
        *,
        case: str,
        dataset_version: str | None,
        task_types: tuple[str, ...],
        limit: int | None,
    ) -> list[dict[str, Any]]: ...

    def load_candidate_edges(self, task: dict[str, Any]) -> list[CandidateEdge]: ...

    def load_case_node_rows(
        self, *, case: str, dataset_version: str, include_labels: bool
    ) -> list[dict[str, Any]]: ...

    def load_case_graph_edges(self, task: dict[str, Any]) -> list[CandidateEdge]: ...

    def create_run(self, document: dict[str, Any]) -> None: ...

    def update_run(self, run_id: str, document: dict[str, Any]) -> None: ...

    def save_stage(self, document: dict[str, Any]) -> None: ...

    def save_task_result(self, document: dict[str, Any]) -> None: ...


class MongoExperimentRepository:
    def __init__(self) -> None:
        from storage import MongoDocumentStore

        self.tasks = MongoDocumentStore("chain_qa_tasks")
        self.edges = MongoDocumentStore("chain_subgraph_edges")
        self.labels = MongoDocumentStore("chain_labels")
        self.runs = MongoDocumentStore(RUNS_COLLECTION)
        self.task_results = MongoDocumentStore(TASK_RESULTS_COLLECTION)
        self.stages = MongoDocumentStore(STAGES_COLLECTION)
        self.runs.create_index([("run_id", 1)], unique=True)
        self.task_results.create_index([("run_id", 1), ("task_id", 1)], unique=True)
        self.stages.create_index(
            [("run_id", 1), ("task_id", 1), ("stage", 1)], unique=True
        )

    def load_tasks(
        self,
        *,
        case: str,
        dataset_version: str | None,
        task_types: tuple[str, ...],
        limit: int | None,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {
            "$or": [{"case_id": case}, {"case_dir": case}],
            "task_type": {"$in": list(task_types)},
        }
        if dataset_version:
            query["dataset_version"] = dataset_version
        rows = self.tasks.find(query)
        if rows and not dataset_version:
            versions = {str(row.get("dataset_version") or "") for row in rows}
            latest = max(versions, key=dataset_version_sort_key)
            rows = [row for row in rows if str(row.get("dataset_version") or "") == latest]
        rows.sort(key=lambda row: (str(row.get("task_type") or ""), str(row.get("id") or "")))
        return rows[:limit] if limit else rows

    def load_candidate_edges(self, task: dict[str, Any]) -> list[CandidateEdge]:
        edge_ids = raw_edge_ids_for_task(task)
        rows: list[dict[str, Any]] = []
        if edge_ids:
            query: dict[str, Any] = {
                "case_id": task.get("case_id"),
                "edge_id": {"$in": edge_ids},
            }
            if task.get("dataset_version"):
                query["dataset_version"] = task["dataset_version"]
            rows = self.edges.find(query)
        return build_candidate_edges(task, rows)

    def _resolve_case_id(self, case: str, dataset_version: str) -> str:
        rows = self.tasks.find(
            {
                "$or": [{"case_id": case}, {"case_dir": case}],
                "dataset_version": dataset_version,
            },
            limit=1,
        )
        if not rows:
            raise LookupError(
                f"no chain QA case mapping for case={case!r} version={dataset_version!r}"
            )
        return str(rows[0].get("case_id") or "")

    def load_case_node_rows(
        self, *, case: str, dataset_version: str, include_labels: bool
    ) -> list[dict[str, Any]]:
        case_id = self._resolve_case_id(case, dataset_version)
        query = {"case_id": case_id, "dataset_version": dataset_version}
        edge_rows = self.edges.find(query)
        labels = (
            {
                str(row.get("edge_id") or ""): row
                for row in self.labels.find(query)
            }
            if include_labels
            else None
        )
        rows = build_whole_graph_node_rows(
            edge_rows,
            labels_by_edge=labels,
            include_labels=include_labels,
        )
        for row in rows:
            row["case_id"] = case_id
        return rows

    def load_case_graph_edges(self, task: dict[str, Any]) -> list[CandidateEdge]:
        query = {
            "case_id": task.get("case_id"),
            "dataset_version": task.get("dataset_version"),
        }
        output: list[CandidateEdge] = []
        for row in self.edges.find(query):
            src = str(row.get("src") or "")
            dst = str(row.get("dst") or "")
            if not src or not dst:
                continue
            asset = str(row.get("asset") or "")
            token = str(row.get("token_address") or "").lower()
            output.append(
                CandidateEdge(
                    candidate_id=str(row.get("edge_id") or ""),
                    src=src,
                    dst=dst,
                    amount=numeric_amount(row.get("amount")),
                    raw_edge_ids=(str(row.get("edge_id") or ""),),
                    edge_types=(str(row.get("edge_type") or ""),),
                    metadata={
                        "edge_count": 1,
                        "tx_count": 1,
                        "min_timestamp": row.get("timestamp"),
                        "max_timestamp": row.get("timestamp"),
                        "tx_hash": row.get("tx_hash"),
                        "assets": [asset] if asset else [],
                        "token_addresses": [token] if token else [],
                        "amount_comparable": True,
                    },
                )
            )
        return output

    def create_run(self, document: dict[str, Any]) -> None:
        self.runs.replace_one({"run_id": document["run_id"]}, document)

    def update_run(self, run_id: str, document: dict[str, Any]) -> None:
        self.runs.upsert_one({"run_id": run_id}, document)

    def save_stage(self, document: dict[str, Any]) -> None:
        key = {
            "run_id": document["run_id"],
            "task_id": document["task_id"],
            "stage": document["stage"],
        }
        self.stages.upsert_one(key, document)

    def save_task_result(self, document: dict[str, Any]) -> None:
        key = {"run_id": document["run_id"], "task_id": document["task_id"]}
        self.task_results.upsert_one(key, document)
