from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


ACTIONS = ("follow", "inspect", "stop", "ignore")
SUPPORTED_TASK_TYPES = (
    "edge_action_classification",
    "path_completion",
    "subgraph_noise_filtering",
)


def normalize_node(value: Any) -> str:
    return str(value or "").strip().lower()


def numeric_amount(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0.0 else 0.0


@dataclass(frozen=True)
class CandidateEdge:
    candidate_id: str
    src: str
    dst: str
    amount: float = 0.0
    raw_edge_ids: tuple[str, ...] = ()
    edge_types: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionPrediction:
    edge_id: str
    action: str
    action_probabilities: dict[str, float]
    rationale: str = ""

    @property
    def follow_probability(self) -> float:
        return float(self.action_probabilities.get("follow") or 0.0)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodePrediction:
    node_id: str
    probability: float
    score: float
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbabilityPath:
    edge_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    probability: float
    log_probability: float
    terminal_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlgorithmOutput:
    node_predictions: list[NodePrediction] = field(default_factory=list)
    edge_predictions: list[ActionPrediction] = field(default_factory=list)
    paths: list[ProbabilityPath] = field(default_factory=list)
    selected_seed_nodes: list[str] = field(default_factory=list)
    selected_follow_edges: list[str] = field(default_factory=list)
    edge_predictions_by_mode: dict[str, list[ActionPrediction]] = field(default_factory=dict)
    selected_follow_edges_by_mode: dict[str, list[str]] = field(default_factory=dict)
    paths_by_mode: dict[str, list[ProbabilityPath]] = field(default_factory=dict)
    budget_usage_by_mode: dict[str, dict[str, int]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_predictions": [item.as_dict() for item in self.node_predictions],
            "edge_predictions": [item.as_dict() for item in self.edge_predictions],
            "paths": [item.as_dict() for item in self.paths],
            "selected_seed_nodes": list(self.selected_seed_nodes),
            "selected_follow_edges": list(self.selected_follow_edges),
            "edge_predictions_by_mode": {
                mode: [item.as_dict() for item in predictions]
                for mode, predictions in self.edge_predictions_by_mode.items()
            },
            "selected_follow_edges_by_mode": {
                mode: list(edge_ids)
                for mode, edge_ids in self.selected_follow_edges_by_mode.items()
            },
            "paths_by_mode": {
                mode: [item.as_dict() for item in paths]
                for mode, paths in self.paths_by_mode.items()
            },
            "budget_usage_by_mode": {
                mode: dict(usage) for mode, usage in self.budget_usage_by_mode.items()
            },
            "diagnostics": dict(self.diagnostics),
        }
