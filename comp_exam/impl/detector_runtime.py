from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Iterable

from .checkpoints import load_detector_checkpoint
from .models import CandidateEdge, normalize_node, numeric_amount
from .node_actionability import build_whole_graph_node_rows
from .repository import build_candidate_edges


DEFAULT_DETECTOR_TOOL_METHODS = ("LR", "RF", "XGB")

DETECTOR_TOOL_METHODS = {
    "Poison": "poison",
    "Haircut": "haircut",
    "FIFO": "fifo",
    "LIFO": "lifo",
    "TIHO": "tiho",
    "TRacer": "tracer",
    "DenseFlow": "denseflow",
    "DenseFlow+": "denseflow_plus",
    "LR": "logistic_edge",
    "RF": "random_forest_edge",
    "MLP": "mlp_edge",
    "GCN": "gcn_edge",
    "I²BGNN": "i2bgnn_edge",
    "PEAE-GNN": "peae_gnn_edge",
    "TokenScout": "tokenscout_edge",
    "XGB": "xgboost_edge",
}

_METHOD_LOOKUP = {
    key.casefold(): display
    for display, implementation in DETECTOR_TOOL_METHODS.items()
    for key in (display, implementation)
}
_MODEL_CACHE: dict[tuple[str, str, str, str], tuple[Any, dict[str, Any]]] = {}
_MODEL_CACHE_LOCK = threading.RLock()
SHARED_NODE_DETECTOR = "xgboost_node_1pct"
DEFAULT_SEED_PROBABILITY_THRESHOLD = 0.5
DEFAULT_SEED_TOP_K = 1


def normalize_detector_tool_methods(
    values: Iterable[str] | None,
    *,
    default: bool = True,
) -> tuple[str, ...]:
    source = DEFAULT_DETECTOR_TOOL_METHODS if values is None and default else values or ()
    output: list[str] = []
    unknown: list[str] = []
    for value in source:
        text = str(value).strip()
        canonical = _METHOD_LOOKUP.get(text.casefold())
        if canonical is None:
            unknown.append(text)
        elif canonical not in output:
            output.append(canonical)
    if unknown:
        raise ValueError(
            "unknown detector tool methods: "
            f"{sorted(unknown)}; supported={sorted(DETECTOR_TOOL_METHODS)}"
        )
    return tuple(output)


def detector_tool_schema(methods: Iterable[str]) -> dict[str, Any]:
    enabled = list(normalize_detector_tool_methods(methods, default=False))
    return {
        "type": "function",
        "function": {
            "name": "detector_predict",
            "description": (
                "Run selected frozen comparison edge classifiers across public task states "
                "and, for SNF, the shared frozen node detector. Predictions are advisory "
                "and do not replace required eth_edge evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "methods": {
                        "type": "array",
                        "items": {"type": "string", "enum": enabled},
                        # "uniqueItems": True,
                        "description": (
                            "Enabled methods to run. Omit to run all methods enabled for this agent."
                        ),
                    },
                    "state_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Public graph_context.states ids to score. Omit to score every "
                            "public state, or the task-level state for EAC."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    }


def _selected_states(task: dict[str, Any], state_ids: Any) -> list[dict[str, Any]]:
    context = task.get("graph_context") or {}
    states = [state for state in context.get("states") or [] if isinstance(state, dict)]
    if state_ids is None:
        return states or [context]
    if not isinstance(state_ids, list):
        raise ValueError("state_ids must be a list of public state ids")
    requested = list(dict.fromkeys(str(value).strip() for value in state_ids if str(value).strip()))
    if not requested:
        raise ValueError("state_ids must not be empty when provided")
    by_id = {str(item.get("state_id") or ""): item for item in states}
    missing = [value for value in requested if value not in by_id]
    if missing:
        raise ValueError(f"state_ids are outside the public task states: {missing}")
    return [by_id[value] for value in requested]


def _state_task(task: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    context = dict(task.get("graph_context") or {})
    for key in ("candidate_edges", "candidate_edge_groups", "current_node"):
        if key in state:
            context[key] = state[key]
    return {**task, "graph_context": context}


def _raw_edge_ids(task: dict[str, Any]) -> list[str]:
    context = task.get("graph_context") or {}
    groups = context.get("candidate_edge_groups") or {}
    output: list[str] = []
    for candidate_id in context.get("candidate_edges") or []:
        raw_ids = groups.get(candidate_id) if isinstance(groups, dict) else None
        output.extend(str(value) for value in raw_ids or [candidate_id])
    return list(dict.fromkeys(output))


def _mongo_rows(task: dict[str, Any], edge_ids: list[str] | None = None) -> list[dict[str, Any]]:
    store = None
    try:
        from storage import MongoDocumentStore

        query: dict[str, Any] = {
            "case_id": task.get("case_id"),
            "dataset_version": task.get("dataset_version"),
        }
        if edge_ids is not None:
            query["edge_id"] = {"$in": edge_ids}
        store = MongoDocumentStore("chain_subgraph_edges")
        return store.find(query)
    except Exception:
        return []
    finally:
        if store is not None:
            store.client.close()


def _local_rows(
    task: dict[str, Any], local_root: Path, edge_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    path = local_root / str(task.get("case_dir") or "") / "subgraph_edges.jsonl"
    if not path.exists():
        return []
    wanted = set(edge_ids or ())
    import json

    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not wanted or str(row.get("edge_id") or "") in wanted:
                output.append(row)
    return output


def _candidate_edges(task: dict[str, Any], local_root: Path) -> list[CandidateEdge]:
    edge_ids = _raw_edge_ids(task)
    rows = _mongo_rows(task, edge_ids) or _local_rows(task, local_root, edge_ids)
    return build_candidate_edges(task, rows)


def _case_raw_rows(task: dict[str, Any], local_root: Path) -> list[dict[str, Any]]:
    return _mongo_rows(task) or _local_rows(task, local_root)


def _case_graph_edges(task: dict[str, Any], local_root: Path) -> list[CandidateEdge]:
    rows = _case_raw_rows(task, local_root)
    output: list[CandidateEdge] = []
    for row in rows:
        src = str(row.get("src") or "")
        dst = str(row.get("dst") or "")
        if not src or not dst:
            continue
        asset = str(row.get("asset") or "")
        token = str(row.get("token_address") or "").lower()
        edge_id = str(row.get("edge_id") or "")
        output.append(
            CandidateEdge(
                candidate_id=edge_id,
                src=src,
                dst=dst,
                amount=numeric_amount(row.get("amount")),
                raw_edge_ids=(edge_id,),
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


class DetectorToolRuntime:
    def __init__(
        self,
        *,
        task: dict[str, Any],
        methods: Iterable[str] | None = None,
        checkpoint_root: Path | None = None,
        local_root: Path | None = None,
    ) -> None:
        self.task = task
        self.methods = normalize_detector_tool_methods(methods)
        self.checkpoint_root = checkpoint_root or (Path(__file__).resolve().parents[2] / "detector_tool")
        self.local_root = local_root or (Path(__file__).resolve().parents[2] / "chain_qa")

    @property
    def enabled(self) -> bool:
        return bool(self.methods)

    def schema(self) -> dict[str, Any]:
        return detector_tool_schema(self.methods)

    def _load_checkpoint(
        self, *, role: str, implementation: str
    ) -> tuple[Any, dict[str, Any]]:
        case_id = str(self.task.get("case_id") or self.task.get("case_dir") or "")
        directory = (
            self.checkpoint_root
            / case_id
            / f"{role}-{implementation}"
        )
        expected_version = str(self.task.get("dataset_version") or "")
        key = (str(directory.resolve()), case_id, implementation, expected_version)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                return cached
            model, manifest = load_detector_checkpoint(directory)
            if manifest.get("role") != role:
                raise ValueError(f"checkpoint role mismatch: {directory}")
            if str(manifest.get("case_id") or "") != case_id:
                raise ValueError(f"checkpoint case_id mismatch: {directory}")
            if str(manifest.get("dataset_version") or "") != expected_version:
                raise ValueError(
                    f"checkpoint dataset version mismatch for {implementation}: "
                    f"{manifest.get('dataset_version')!r} != {expected_version!r}"
                )
            if str(getattr(model, "name", "")) != implementation:
                raise ValueError(
                    f"checkpoint algorithm mismatch for {implementation}: {directory}"
                )
            if bool(getattr(model, "requires_case_graph", False)):
                graph = _case_graph_edges(self.task, self.local_root)
                model.prepare_case(
                    case_id=case_id,
                    dataset_version=expected_version,
                    edges=graph,
                )
            _MODEL_CACHE[key] = (model, manifest)
            return model, manifest

    def _load_model(self, method: str) -> tuple[Any, dict[str, Any]]:
        return self._load_checkpoint(
            role="edge_classifier",
            implementation=DETECTOR_TOOL_METHODS[method],
        )

    def _node_predictions(self) -> dict[str, Any] | None:
        if self.task.get("task_type") != "subgraph_noise_filtering":
            return None
        try:
            model, manifest = self._load_checkpoint(
                role="node_detector",
                implementation=SHARED_NODE_DETECTOR,
            )
            edges = _candidate_edges(self.task, self.local_root)
            node_rows = build_whole_graph_node_rows(
                _case_raw_rows(self.task, self.local_root),
                include_labels=False,
            )
            predictions = model.predict(
                task=self.task,
                edges=edges,
                node_feature_rows=node_rows,
            )
            rows = [
                {
                    "node_id": item.node_id,
                    "probability": round(float(item.probability), 8),
                }
                for item in predictions
            ]
            selected = [
                item.node_id
                for item in predictions
                if item.probability >= DEFAULT_SEED_PROBABILITY_THRESHOLD
            ][:DEFAULT_SEED_TOP_K]
            return {
                "status": "ok",
                "algorithm": manifest.get("algorithm"),
                "model_sha256": manifest.get("model_sha256"),
                "predictions": rows,
                "selected_seed_nodes": selected,
                "probability_threshold": DEFAULT_SEED_PROBABILITY_THRESHOLD,
                "top_k": DEFAULT_SEED_TOP_K,
            }
        except Exception as exc:
            return {
                "status": "error",
                "algorithm": SHARED_NODE_DETECTOR,
                "error": f"{type(exc).__name__}: {exc}",
                "predictions": [],
                "selected_seed_nodes": [],
            }

    def _predict_state(
        self, state: dict[str, Any], requested: tuple[str, ...]
    ) -> dict[str, Any]:
        state_task = _state_task(self.task, state)
        edges = _candidate_edges(state_task, self.local_root)
        candidate_ids = [
            str(value)
            for value in (state_task.get("graph_context") or {}).get("candidate_edges") or []
        ]
        if not edges or {edge.candidate_id for edge in edges} != set(candidate_ids):
            missing = sorted(set(candidate_ids) - {edge.candidate_id for edge in edges})
            raise LookupError(
                f"public candidate edge materialization incomplete: missing={missing}"
            )
        current_node = normalize_node(
            state.get("current_node")
            or (state_task.get("graph_context") or {}).get("current_node")
            or (state_task.get("graph_context") or {}).get("start_node")
        )
        threshold = float(
            (state_task.get("graph_context") or {}).get("follow_threshold", 0.5)
        )
        outputs: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for method in requested:
            try:
                model, manifest = self._load_model(method)
                predictions = model.predict(
                    task=state_task,
                    edges=edges,
                    active_nodes=[current_node] if current_node else [],
                )
                rows = [
                    {
                        "edge_id": item.edge_id,
                        "action": item.action,
                        "follow_probability": round(item.follow_probability, 8),
                    }
                    for item in predictions
                ]
                outputs[method] = {
                    "algorithm": manifest.get("algorithm"),
                    "model_sha256": manifest.get("model_sha256"),
                    "predictions": rows,
                    "selected_follow_edges": [
                        item.edge_id
                        for item in predictions
                        if item.action == "follow"
                        and item.follow_probability >= threshold
                    ],
                }
            except Exception as exc:
                errors[method] = f"{type(exc).__name__}: {exc}"
        return {
            "state_id": str(state.get("state_id") or "task_state"),
            "current_node": current_node or None,
            "candidate_ids": candidate_ids,
            "methods": outputs,
            "errors": errors,
        }

    def predict(self, args: dict[str, Any]) -> dict[str, Any]:
        requested = normalize_detector_tool_methods(
            args.get("methods") if "methods" in args else self.methods,
            default=False,
        )
        outside = set(requested) - set(self.methods)
        if outside:
            raise ValueError(f"detector methods are not enabled for this run: {sorted(outside)}")
        if not requested:
            raise ValueError("detector_predict requires at least one enabled method")
        states = _selected_states(
            self.task,
            args.get("state_ids", [args["state_id"]] if args.get("state_id") else None),
        )
        state_results: list[dict[str, Any]] = []
        state_errors: dict[str, str] = {}
        for state in states:
            state_id = str(state.get("state_id") or "task_state")
            try:
                state_results.append(self._predict_state(state, requested))
            except Exception as exc:
                state_errors[state_id] = f"{type(exc).__name__}: {exc}"
        node_detector = self._node_predictions()
        successful_method_states = sum(
            len(state.get("methods") or {}) for state in state_results
        )
        has_node_result = bool(node_detector and node_detector.get("status") == "ok")
        has_output = bool(successful_method_states or has_node_result)
        has_errors = bool(
            state_errors
            or any(state.get("errors") for state in state_results)
            or (node_detector and node_detector.get("status") == "error")
        )
        return {
            "status": "ok" if has_output and not has_errors else "partial" if has_output else "error",
            "case_id": self.task.get("case_id"),
            "dataset_version": self.task.get("dataset_version"),
            "requested_methods": list(requested),
            "requested_state_count": len(states),
            "states": state_results,
            "state_errors": state_errors,
            "node_detector": node_detector,
            "advisory_only": True,
            "required_index_evidence_still_required": True,
        }
