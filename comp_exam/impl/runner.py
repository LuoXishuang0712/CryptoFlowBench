from __future__ import annotations

import math
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chain_qa_version import chain_qa_versions_compatible

from .algorithms import EdgeClassifier, NodeDetector
from .checkpoints import persist_detector_checkpoint
from .evaluation import evaluate_task
from .models import (
    SUPPORTED_TASK_TYPES,
    AlgorithmOutput,
    CandidateEdge,
    ProbabilityPath,
    normalize_node,
)
from .repository import ExperimentRepository


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ExperimentConfig:
    case: str
    dataset_version: str | None = None
    task_types: tuple[str, ...] = SUPPORTED_TASK_TYPES
    limit: int | None = None
    follow_threshold: float = 0.5
    taint_budget_ratio: float = 0.5
    seed_top_k: int = 1
    seed_probability_threshold: float = 0.5
    training_cases: tuple[str, ...] = ()
    max_depth: int = 3
    beam_width: int = 3
    max_node_expansions: int = 64
    max_inspected_edges: int = 512
    fail_fast: bool = False
    model_output_dir: str | None = None

    def __post_init__(self) -> None:
        unsupported = set(self.task_types) - set(SUPPORTED_TASK_TYPES)
        if unsupported:
            raise ValueError(f"unsupported task types: {sorted(unsupported)}")
        if not 0.0 <= self.follow_threshold <= 1.0:
            raise ValueError("follow_threshold must be within [0, 1]")
        if not 0.0 < self.taint_budget_ratio <= 1.0:
            raise ValueError("taint_budget_ratio must be within (0, 1]")
        if self.seed_top_k < 1:
            raise ValueError("seed_top_k must be positive")
        if not 0.0 <= self.seed_probability_threshold <= 1.0:
            raise ValueError("seed_probability_threshold must be within [0, 1]")
        if self.case in self.training_cases:
            raise ValueError("target case must not appear in training_cases")
        for field in ("max_depth", "beam_width", "max_node_expansions", "max_inspected_edges"):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")


class ComparisonExperimentRunner:
    def __init__(
        self,
        *,
        repository: ExperimentRepository,
        edge_classifier: EdgeClassifier,
        node_detector: NodeDetector,
        config: ExperimentConfig,
    ) -> None:
        self.repository = repository
        self.edge_classifier = edge_classifier
        self.node_detector = node_detector
        self.config = config
        self._node_feature_rows_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._case_graph_edges_cache: dict[tuple[str, str], list[CandidateEdge]] = {}

    def run(self, *, run_id: str | None = None) -> dict[str, Any]:
        run_id = run_id or f"comp-{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:10]}"
        started = utc_now()
        run_document = {
            "run_id": run_id,
            "status": "running",
            "case": self.config.case,
            "dataset_version": self.config.dataset_version,
            "task_types": list(self.config.task_types),
            "edge_classifier": {
                "name": self.edge_classifier.name,
                "version": self.edge_classifier.version,
            },
            "node_detector": {
                "name": self.node_detector.name,
                "version": self.node_detector.version,
            },
            "config": asdict(self.config),
            "started_at": started,
            "progress": {"completed": 0, "failed": 0, "total": 0},
        }
        self.repository.create_run(run_document)
        try:
            tasks = self.repository.load_tasks(
                case=self.config.case,
                dataset_version=self.config.dataset_version,
                task_types=self.config.task_types,
                limit=self.config.limit,
            )
            if not tasks:
                raise LookupError(
                    f"no supported QA tasks found in MongoDB for case={self.config.case!r}"
                )
            resolved_version = str(tasks[0].get("dataset_version") or "")
            target_aliases = {
                self.config.case,
                str(tasks[0].get("case_id") or ""),
                str(tasks[0].get("case_dir") or ""),
            }
            leaked_training_cases = target_aliases & set(self.config.training_cases)
            if leaked_training_cases:
                raise ValueError(
                    "target case aliases must not appear in training_cases: "
                    f"{sorted(leaked_training_cases)}"
                )
            training = self._fit_algorithms(resolved_version)
            checkpoints = self._persist_algorithms(
                case_id=str(
                    tasks[0].get("case_id")
                    or tasks[0].get("case_dir")
                    or self.config.case
                ),
                dataset_version=resolved_version,
                training=training,
            )
            self.repository.update_run(
                run_id,
                {
                    "dataset_version": resolved_version,
                    "training": training,
                    "checkpoints": checkpoints,
                    "progress": {"completed": 0, "failed": 0, "total": len(tasks)},
                },
            )
            task_results: list[dict[str, Any]] = []
            failed = 0
            for index, task in enumerate(tasks, start=1):
                try:
                    result = self._run_task(run_id, task)
                except Exception as exc:
                    failed += 1
                    result = self._failed_task_result(run_id, task, exc)
                    self.repository.save_task_result(result)
                    if self.config.fail_fast:
                        task_results.append(result)
                        raise
                task_results.append(result)
                self.repository.update_run(
                    run_id,
                    {
                        "progress": {
                            "completed": index - failed,
                            "failed": failed,
                            "total": len(tasks),
                        },
                        "last_task_id": task.get("id"),
                    },
                )
            summary = summarize_results(task_results)
            completed = utc_now()
            final = {
                "run_id": run_id,
                "status": "completed_with_errors" if failed else "completed",
                "case": self.config.case,
                "dataset_version": resolved_version,
                "started_at": started,
                "completed_at": completed,
                "duration_seconds": (completed - started).total_seconds(),
                "progress": {
                    "completed": len(tasks) - failed,
                    "failed": failed,
                    "total": len(tasks),
                },
                "training": training,
                "checkpoints": checkpoints,
                "summary": summary,
            }
            self.repository.update_run(run_id, final)
            return final
        except Exception as exc:
            failed_at = utc_now()
            self.repository.update_run(
                run_id,
                {
                    "status": "failed",
                    "completed_at": failed_at,
                    "duration_seconds": (failed_at - started).total_seconds(),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise

    def _persist_algorithms(
        self,
        *,
        case_id: str,
        dataset_version: str,
        training: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.config.model_output_dir:
            return {}
        summaries = training.get("algorithms") or {}
        root = Path(self.config.model_output_dir)
        outputs: dict[str, Any] = {}
        for role, algorithm in (
            ("edge_classifier", self.edge_classifier),
            ("node_detector", self.node_detector),
        ):
            name = str(getattr(algorithm, "name", type(algorithm).__name__))
            outputs[role] = persist_detector_checkpoint(
                algorithm,
                root=root,
                case_id=case_id,
                role=role,
                dataset_version=dataset_version,
                training_cases=list(self.config.training_cases),
                training_summary=summaries.get(name),
                case_disjoint=training.get("leakage_guard") in {None, "case_disjoint"},
            )
        return outputs

    def _run_task(self, run_id: str, task: dict[str, Any]) -> dict[str, Any]:
        task_started = time.perf_counter()
        task_id = str(task.get("id") or "")
        task_type = str(task.get("task_type") or "")
        edges = self.repository.load_candidate_edges(task)
        if bool(getattr(self.edge_classifier, "requires_case_graph", False)):
            cache_key = (
                str(task.get("case_id") or ""),
                str(task.get("dataset_version") or ""),
            )
            if cache_key not in self._case_graph_edges_cache:
                self._case_graph_edges_cache[cache_key] = (
                    self.repository.load_case_graph_edges(task)
                )
            self.edge_classifier.prepare_case(
                case_id=cache_key[0],
                dataset_version=cache_key[1],
                edges=self._case_graph_edges_cache[cache_key],
            )
        self._save_stage(
            run_id,
            task,
            "task_loaded",
            {
                "public_task": public_task_snapshot(task),
                "candidate_edges": [edge.as_dict() for edge in edges],
                "candidate_count": len(edges),
            },
        )

        output = AlgorithmOutput()
        context = task.get("graph_context") or {}
        current_node = normalize_node(context.get("start_node") or context.get("current_node"))
        if task_type == "subgraph_noise_filtering":
            node_feature_rows = None
            if getattr(self.node_detector, "training_scope", None) == (
                "whole_subgraph_actionability"
            ):
                cache_key = (
                    str(task.get("case_id") or ""),
                    str(task.get("dataset_version") or ""),
                )
                if cache_key not in self._node_feature_rows_cache:
                    self._node_feature_rows_cache[cache_key] = (
                        self.repository.load_case_node_rows(
                            case=cache_key[0],
                            dataset_version=cache_key[1],
                            include_labels=False,
                        )
                    )
                node_feature_rows = self._node_feature_rows_cache[cache_key]
            output.node_predictions = self.node_detector.predict(
                task=task,
                edges=edges,
                node_feature_rows=node_feature_rows,
            )
            output.selected_seed_nodes = [
                item.node_id
                for item in output.node_predictions
                if item.probability >= self.config.seed_probability_threshold
            ][: self.config.seed_top_k]
            active_nodes = list(output.selected_seed_nodes)
            self._save_stage(
                run_id,
                task,
                "node_detection",
                {
                    "algorithm": self.node_detector.name,
                    "predictions": [item.as_dict() for item in output.node_predictions],
                    "selected_seed_nodes": output.selected_seed_nodes,
                    "probability_threshold": self.config.seed_probability_threshold,
                },
            )
        else:
            active_nodes = [current_node] if current_node else []
            self._save_stage(
                run_id,
                task,
                "node_detection",
                {"status": "skipped", "reason": f"{task_type} supplies its start state"},
            )

        threshold = _task_follow_threshold(task, self.config.follow_threshold)
        mode_nodes: dict[str, list[str]] = {}
        if task_type == "path_completion" and context.get("sequential_policy"):
            mode_nodes = {
                "teacher_forced": [
                    normalize_node(state.get("current_node"))
                    for state in context.get("states") or []
                    if isinstance(state, dict) and state.get("current_node")
                ],
                "free_rollout": active_nodes,
            }
        elif task_type == "subgraph_noise_filtering" and context.get("sequential_policy"):
            oracle_nodes = [
                normalize_node(value)
                for value in (task.get("answer_value") or {}).get("gold_seed_nodes") or []
            ]
            mode_nodes = {
                "oracle_seed": oracle_nodes,
                "predicted_seed": active_nodes,
            }
        else:
            mode_nodes = {"default": active_nodes}

        for mode, nodes in mode_nodes.items():
            predictions = self.edge_classifier.predict(
                task=task,
                edges=edges,
                active_nodes=nodes,
            )
            selected = [
                item.edge_id
                for item in predictions
                if item.action == "follow" and item.follow_probability >= threshold
            ]
            output.edge_predictions_by_mode[mode] = predictions
            output.selected_follow_edges_by_mode[mode] = selected

        primary_mode = (
            "free_rollout"
            if "free_rollout" in mode_nodes
            else "predicted_seed"
            if "predicted_seed" in mode_nodes
            else "default"
        )
        output.edge_predictions = output.edge_predictions_by_mode.get(primary_mode, [])
        output.selected_follow_edges = output.selected_follow_edges_by_mode.get(primary_mode, [])
        self._save_stage(
            run_id,
            task,
            "edge_classification",
            {
                "algorithm": self.edge_classifier.name,
                "active_nodes_by_mode": mode_nodes,
                "follow_threshold": threshold,
                "predictions_by_mode": {
                    mode: [item.as_dict() for item in predictions]
                    for mode, predictions in output.edge_predictions_by_mode.items()
                },
                "selected_follow_edges_by_mode": output.selected_follow_edges_by_mode,
            },
        )

        if task_type in {"path_completion", "subgraph_noise_filtering"}:
            for mode, nodes in mode_nodes.items():
                paths, usage = probability_paths(
                    task,
                    edges,
                    self.edge_classifier,
                    nodes,
                    follow_threshold=threshold,
                    config=self.config,
                )
                output.paths_by_mode[mode] = paths
                output.budget_usage_by_mode[mode] = usage
            output.paths = output.paths_by_mode.get(primary_mode, [])
            self._save_stage(
                run_id,
                task,
                "path_generation",
                {
                    "mode": "beam_probability_rollout",
                    "paths_by_mode": {
                        mode: [path.as_dict() for path in paths]
                        for mode, paths in output.paths_by_mode.items()
                    },
                    "budget_usage_by_mode": output.budget_usage_by_mode,
                },
            )
        else:
            self._save_stage(
                run_id,
                task,
                "path_generation",
                {"status": "skipped", "reason": "single-step classification task"},
            )

        output.diagnostics = {
            "candidate_count": len(edges),
            "active_node_count": len(active_nodes),
            "path_count": len(output.paths),
            "rollout_modes": list(mode_nodes),
        }
        evaluation = evaluate_task(task, output)
        self._save_stage(run_id, task, "evaluation", evaluation)
        result = {
            "run_id": run_id,
            "task_id": task_id,
            "case_id": task.get("case_id"),
            "case_dir": task.get("case_dir"),
            "dataset_version": task.get("dataset_version"),
            "task_type": task_type,
            "status": "completed",
            "edge_classifier": self.edge_classifier.name,
            "node_detector": self.node_detector.name,
            "algorithm_output": output.as_dict(),
            "evaluation": evaluation,
            "duration_seconds": time.perf_counter() - task_started,
        }
        self.repository.save_task_result(result)
        return result

    def _fit_algorithms(self, target_dataset_version: str) -> dict[str, Any]:
        prefit_summaries = {
            str(algorithm.name): dict(getattr(algorithm, "training_summary"))
            for algorithm in (self.edge_classifier, self.node_detector)
            if getattr(algorithm, "training_summary", None)
            and not bool(getattr(algorithm, "requires_fit", False))
        }
        trainable = [
            algorithm
            for algorithm in (self.edge_classifier, self.node_detector)
            if bool(getattr(algorithm, "requires_fit", False))
        ]
        if not trainable:
            return {
                "required": bool(prefit_summaries),
                "cases": list(self.config.training_cases) if prefit_summaries else [],
                "task_count": 0,
                "algorithms": prefit_summaries,
                "leakage_guard": "case_disjoint" if prefit_summaries else None,
            }
        if not self.config.training_cases:
            names = ", ".join(str(algorithm.name) for algorithm in trainable)
            raise ValueError(
                f"{names} require one or more disjoint --train-case values"
            )
        whole_graph_node_detector = (
            self.node_detector
            if getattr(self.node_detector, "training_scope", None)
            == "whole_subgraph_actionability"
            else None
        )
        task_trainable = [
            algorithm
            for algorithm in trainable
            if algorithm is not whole_graph_node_detector
        ]
        examples: list[tuple[dict[str, Any], list[CandidateEdge]]] = []
        node_training_rows: list[dict[str, Any]] = []
        task_counts: dict[str, int] = {}
        node_row_counts: dict[str, int] = {}
        versions_by_case: dict[str, str] = {}
        for case in self.config.training_cases:
            tasks = self.repository.load_tasks(
                case=case,
                dataset_version=self.config.dataset_version,
                task_types=SUPPORTED_TASK_TYPES,
                limit=None,
            )
            task_counts[case] = len(tasks)
            versions = {str(task.get("dataset_version") or "") for task in tasks}
            if len(versions) > 1:
                raise ValueError(f"training case {case!r} mixed dataset versions: {sorted(versions)}")
            training_version = next(iter(versions), "")
            versions_by_case[case] = training_version
            if not chain_qa_versions_compatible(target_dataset_version, training_version):
                raise ValueError(
                    f"training case {case!r} uses incompatible dataset version "
                    f"{training_version!r}; target uses {target_dataset_version!r}"
                )
            if task_trainable:
                examples.extend(
                    (task, self.repository.load_candidate_edges(task)) for task in tasks
                )
            if whole_graph_node_detector is not None:
                case_rows = self.repository.load_case_node_rows(
                    case=case,
                    dataset_version=training_version,
                    include_labels=True,
                )
                node_row_counts[case] = len(case_rows)
                node_training_rows.extend(case_rows)
        if task_trainable and not examples:
            raise LookupError("no training tasks found for configured training_cases")
        if whole_graph_node_detector is not None and not node_training_rows:
            raise LookupError("no whole-subgraph node rows found for training cases")
        summaries = dict(prefit_summaries) | {
            str(algorithm.name): algorithm.fit(examples)
            for algorithm in task_trainable
        }
        if whole_graph_node_detector is not None:
            summaries[str(whole_graph_node_detector.name)] = (
                whole_graph_node_detector.fit_node_rows(
                    node_training_rows,
                    seed=0,
                    fold_identity=self.config.case,
                )
            )
        return {
            "required": True,
            "cases": list(self.config.training_cases),
            "task_count": sum(task_counts.values()),
            "task_counts_by_case": task_counts,
            "node_row_counts_by_case": node_row_counts,
            "dataset_versions_by_case": versions_by_case,
            "target_dataset_version": target_dataset_version,
            "algorithms": summaries,
            "leakage_guard": "case_disjoint",
        }

    def _save_stage(
        self,
        run_id: str,
        task: dict[str, Any],
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        self.repository.save_stage(
            {
                "run_id": run_id,
                "task_id": str(task.get("id") or ""),
                "task_type": task.get("task_type"),
                "case_id": task.get("case_id"),
                "dataset_version": task.get("dataset_version"),
                "stage": stage,
                "status": "completed",
                "payload": payload,
                "recorded_at": utc_now(),
            }
        )

    def _failed_task_result(
        self, run_id: str, task: dict[str, Any], exc: Exception
    ) -> dict[str, Any]:
        document = {
            "run_id": run_id,
            "task_id": str(task.get("id") or ""),
            "task_type": task.get("task_type"),
            "case_id": task.get("case_id"),
            "dataset_version": task.get("dataset_version"),
            "stage": "error",
            "status": "failed",
            "payload": {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            "recorded_at": utc_now(),
        }
        self.repository.save_stage(document)
        return {
            "run_id": run_id,
            "task_id": document["task_id"],
            "case_id": document["case_id"],
            "dataset_version": document["dataset_version"],
            "task_type": document["task_type"],
            "status": "failed",
            "error": document["payload"],
        }


def _task_follow_threshold(task: dict[str, Any], fallback: float) -> float:
    try:
        value = float((task.get("graph_context") or {}).get("follow_threshold", fallback))
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else fallback


def probability_paths(
    task: dict[str, Any],
    edges: list[CandidateEdge],
    edge_classifier: EdgeClassifier,
    active_nodes: list[str],
    *,
    follow_threshold: float,
    config: ExperimentConfig,
) -> tuple[list[ProbabilityPath], dict[str, int]]:
    context = task.get("graph_context") or {}
    configured_budget = context.get("rollout_budget") or {}
    max_depth = max(1, min(int(configured_budget.get("max_depth") or config.max_depth), config.max_depth))
    beam_width = max(1, min(int(configured_budget.get("beam_width") or config.beam_width), config.beam_width))
    max_node_expansions = max(
        1,
        min(
            int(configured_budget.get("max_node_expansions") or config.max_node_expansions),
            config.max_node_expansions,
        ),
    )
    max_inspected_edges = max(
        1,
        min(
            int(configured_budget.get("max_inspected_edges") or config.max_inspected_edges),
            config.max_inspected_edges,
        ),
    )
    prefix = tuple(
        str(value)
        for value in context.get("path_prefix_edges") or []
    )
    outgoing: dict[str, list[CandidateEdge]] = {}
    for edge in edges:
        outgoing.setdefault(normalize_node(edge.src), []).append(edge)
    beam = [
        (normalize_node(node), prefix, (normalize_node(node),), 1.0, 0.0)
        for node in dict.fromkeys(active_nodes)
        if normalize_node(node)
    ]
    completed: list[ProbabilityPath] = []
    node_expansions = 0
    inspected_edges = 0
    depth_reached = 0
    for depth in range(1, max_depth + 1):
        if not beam:
            break
        next_beam: list[tuple[str, tuple[str, ...], tuple[str, ...], float, float]] = []
        for current_node, path_edges, path_nodes, probability, log_probability in beam:
            if node_expansions >= max_node_expansions or inspected_edges >= max_inspected_edges:
                completed.append(
                    ProbabilityPath(path_edges, path_nodes, probability, log_probability, "budget")
                )
                continue
            candidates = outgoing.get(current_node, [])
            node_expansions += 1
            remaining = max_inspected_edges - inspected_edges
            candidates = candidates[:remaining]
            inspected_edges += len(candidates)
            if not candidates:
                completed.append(
                    ProbabilityPath(path_edges, path_nodes, probability, log_probability, "dead_end")
                )
                continue
            predictions = edge_classifier.predict(
                task=task,
                edges=candidates,
                active_nodes=[current_node],
            )
            by_prediction = {item.edge_id: item for item in predictions}
            followed = []
            for edge in candidates:
                prediction = by_prediction.get(edge.candidate_id)
                if (
                    prediction is None
                    or prediction.action != "follow"
                    or prediction.follow_probability < follow_threshold
                ):
                    continue
                if normalize_node(edge.dst) in path_nodes:
                    continue
                followed.append((edge, prediction))
            if not followed:
                completed.append(
                    ProbabilityPath(path_edges, path_nodes, probability, log_probability, "no_follow")
                )
                continue
            for edge, prediction in followed:
                edge_probability = max(prediction.follow_probability, 1e-12)
                next_probability = probability * edge_probability
                next_beam.append(
                    (
                        normalize_node(edge.dst),
                        path_edges + (prediction.edge_id,),
                        path_nodes + (normalize_node(edge.dst),),
                        next_probability,
                        log_probability + math.log(edge_probability),
                    )
                )
        next_beam.sort(key=lambda item: (-item[3], item[1]))
        beam = next_beam[:beam_width]
        depth_reached = depth
    for _, path_edges, path_nodes, probability, log_probability in beam:
        completed.append(
            ProbabilityPath(path_edges, path_nodes, probability, log_probability, "budget")
        )
    completed.sort(key=lambda item: (-item.probability, item.edge_ids))
    return completed[:beam_width], {
        "node_expansions": node_expansions,
        "inspected_edges": inspected_edges,
        "depth_reached": depth_reached,
        "tool_calls": 0,
    }


def public_task_snapshot(task: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "id",
        "case_id",
        "case_dir",
        "case_name",
        "dataset_version",
        "task_type",
        "difficulty",
        "question",
        "graph_context",
        "edge_summaries",
        "link_semantics",
    )
    return {key: task.get(key) for key in allowed if key in task}


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") != "completed":
            continue
        by_type.setdefault(str(result.get("task_type") or ""), []).append(result)
    summary: dict[str, Any] = {"task_count": len(results), "by_task_type": {}}
    for task_type, rows in sorted(by_type.items()):
        numeric: dict[str, list[float]] = {}
        passes = 0
        for row in rows:
            evaluation = row.get("evaluation") or {}
            passes += int(bool(evaluation.get("passed")))
            for key, value in (evaluation.get("metrics") or {}).items():
                if isinstance(value, bool):
                    numeric.setdefault(key, []).append(float(value))
                elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                    numeric.setdefault(key, []).append(float(value))
        summary["by_task_type"][task_type] = {
            "count": len(rows),
            "pass_rate": passes / len(rows) if rows else 0.0,
            "metric_means": {
                key: sum(values) / len(values) for key, values in sorted(numeric.items())
            },
        }
    return summary
