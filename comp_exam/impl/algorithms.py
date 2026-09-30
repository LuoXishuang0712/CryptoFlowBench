from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Protocol

from .features import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    candidate_seed_nodes,
    edge_feature_vector,
    gold_edge_actions,
    gold_seed_nodes,
    node_feature_vector,
    training_active_nodes,
)

from .models import (
    ACTIONS,
    ActionPrediction,
    CandidateEdge,
    NodePrediction,
    normalize_node,
)
from .node_actionability import (
    WHOLE_GRAPH_NODE_FEATURE_NAMES,
    sample_positive_rate,
)
from .denseflow import DenseFlowEdgeClassifier, DenseFlowPlusEdgeClassifier
from .gcn import GCNEdgeClassifier
from .paper_gnns import (
    I2BGNNEdgeClassifier,
    PEAEGNNEdgeClassifier,
    TokenScoutEdgeClassifier,
)
from .tracer import TRacerEdgeClassifier


class EdgeClassifier(Protocol):
    name: str
    version: str

    def predict(
        self,
        *,
        task: dict[str, Any],
        edges: list[CandidateEdge],
        active_nodes: list[str],
    ) -> list[ActionPrediction]: ...


class NodeDetector(Protocol):
    name: str
    version: str

    def predict(
        self,
        *,
        task: dict[str, Any],
        edges: list[CandidateEdge],
        node_feature_rows: list[dict[str, Any]] | None = None,
    ) -> list[NodePrediction]: ...


TrainingExample = tuple[dict[str, Any], list[CandidateEdge]]


def _binary_action_prediction(
    edge: CandidateEdge,
    score: float,
    rationale: str,
) -> ActionPrediction:
    follow = min(1.0, max(0.0, float(score)))
    probabilities = {
        "follow": follow,
        "inspect": 0.0,
        "stop": 0.0,
        "ignore": 1.0 - follow,
    }
    return ActionPrediction(
        edge_id=edge.candidate_id,
        action="follow" if follow >= 0.5 else "ignore",
        action_probabilities=probabilities,
        rationale=rationale,
    )


class PoisonEdgeClassifier:
    """Propagate taint through every outgoing relation of an active node."""

    name = "poison"
    version = "0.2"

    def __init__(self, *, follow_probability: float = 1.0) -> None:
        if not 0.5 < follow_probability <= 1.0:
            raise ValueError("follow_probability must be within (0.5, 1]")
        self.follow_probability = follow_probability

    def predict(
        self,
        *,
        task: dict[str, Any],
        edges: list[CandidateEdge],
        active_nodes: list[str],
    ) -> list[ActionPrediction]:
        active = {normalize_node(node) for node in active_nodes if node}
        residual = (1.0 - self.follow_probability) / 3.0
        predictions: list[ActionPrediction] = []
        for edge in edges:
            is_outgoing = bool(edge.src) and normalize_node(edge.src) in active
            if is_outgoing:
                probabilities = {
                    "follow": self.follow_probability,
                    "inspect": residual,
                    "stop": residual,
                    "ignore": residual,
                }
                action = "follow"
                rationale = "poison propagation from an active source node"
            else:
                probabilities = {
                    "follow": residual,
                    "inspect": residual,
                    "stop": residual,
                    "ignore": self.follow_probability,
                }
                action = "ignore"
                rationale = "edge source is not currently tainted"
            predictions.append(
                ActionPrediction(
                    edge_id=edge.candidate_id,
                    action=action,
                    action_probabilities={name: probabilities[name] for name in ACTIONS},
                    rationale=rationale,
                )
            )
        return predictions


class ValueTaintEdgeClassifier:
    """Ethereum account-relation adaptation of value-allocation taint rules.

    ``taint_budget_ratio`` is shared across Haircut/FIFO/LIFO/TIHO. It defines
    the tainted value available at each active source as a fraction of visible
    outgoing value. This is an explicit benchmark adaptation, not a claim of
    UTXO-equivalent semantics.
    """

    version = "0.2-ethereum-adapted"
    method = "haircut"

    def __init__(self, *, taint_budget_ratio: float = 0.5) -> None:
        if not 0.0 < taint_budget_ratio <= 1.0:
            raise ValueError("taint_budget_ratio must be within (0, 1]")
        self.taint_budget_ratio = taint_budget_ratio

    @property
    def name(self) -> str:
        return self.method

    @staticmethod
    def _order_key(edge: CandidateEdge) -> tuple[float, float, str]:
        timestamp = float(edge.metadata.get("min_timestamp") or float("inf"))
        block_number = float(edge.metadata.get("min_block_number") or float("inf"))
        return timestamp, block_number, edge.candidate_id

    def _ordered(self, edges: list[CandidateEdge]) -> list[CandidateEdge]:
        if self.method == "fifo":
            return sorted(edges, key=self._order_key)
        if self.method == "lifo":
            return sorted(edges, key=self._order_key, reverse=True)
        if self.method == "tiho":
            return sorted(edges, key=lambda edge: (-edge.amount, self._order_key(edge)))
        return list(edges)

    def _scores(self, outgoing: list[CandidateEdge]) -> dict[str, float]:
        if not outgoing:
            return {}
        if self.method == "haircut":
            return {
                edge.candidate_id: self.taint_budget_ratio for edge in outgoing
            }
        weights = {
            edge.candidate_id: edge.amount if edge.amount > 0.0 else 1.0
            for edge in outgoing
        }
        remaining = sum(weights.values()) * self.taint_budget_ratio
        scores: dict[str, float] = {edge.candidate_id: 0.0 for edge in outgoing}
        for edge in self._ordered(outgoing):
            value = weights[edge.candidate_id]
            allocated = min(value, remaining)
            scores[edge.candidate_id] = allocated / value if value > 0.0 else 0.0
            remaining -= allocated
            if remaining <= 0.0:
                break
        return scores

    @staticmethod
    def _asset_bucket(edge: CandidateEdge) -> tuple[str, ...]:
        token_addresses = tuple(
            str(value).lower() for value in edge.metadata.get("token_addresses") or []
        )
        if token_addresses:
            return ("token",) + token_addresses
        assets = tuple(str(value).lower() for value in edge.metadata.get("assets") or [])
        if assets:
            return ("asset",) + assets
        return ("edge_type",) + tuple(edge.edge_types or ("unknown",))

    def predict(
        self,
        *,
        task: dict[str, Any],
        edges: list[CandidateEdge],
        active_nodes: list[str],
    ) -> list[ActionPrediction]:
        active = {normalize_node(node) for node in active_nodes if node}
        scores: dict[str, float] = {}
        for node in active:
            outgoing_by_asset: dict[tuple[str, ...], list[CandidateEdge]] = defaultdict(list)
            for edge in edges:
                if normalize_node(edge.src) == node:
                    outgoing_by_asset[self._asset_bucket(edge)].append(edge)
            for outgoing in outgoing_by_asset.values():
                scores.update(self._scores(outgoing))
        return [
            _binary_action_prediction(
                edge,
                scores.get(edge.candidate_id, 0.0),
                (
                    f"{self.method} Ethereum adaptation; shared taint_budget_ratio="
                    f"{self.taint_budget_ratio:.6g}"
                ),
            )
            for edge in edges
        ]


class HaircutEdgeClassifier(ValueTaintEdgeClassifier):
    method = "haircut"


class FIFOEdgeClassifier(ValueTaintEdgeClassifier):
    method = "fifo"


class LIFOEdgeClassifier(ValueTaintEdgeClassifier):
    method = "lifo"


class TIHOEdgeClassifier(ValueTaintEdgeClassifier):
    method = "tiho"


def _require_sklearn() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "feature baselines require scikit-learn; run `uv sync --extra baselines`"
        ) from exc
    return LogisticRegression, RandomForestClassifier, MLPClassifier, make_pipeline, StandardScaler


def _create_feature_model(
    backend: str, *, multiclass: bool, num_classes: int | None = None
) -> Any:
    (
        LogisticRegression,
        RandomForestClassifier,
        MLPClassifier,
        make_pipeline,
        StandardScaler,
    ) = _require_sklearn()
    if backend == "logistic":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=0,
        )
    if backend == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=1,
        )
    if backend == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(50,),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=200,
                alpha=0.0,
                random_state=0,
            ),
        )
    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "xgboost baseline requires xgboost; run `uv sync --extra baselines`"
            ) from exc
        parameters: dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "objective": "multi:softprob" if multiclass else "binary:logistic",
            "eval_metric": "mlogloss" if multiclass else "logloss",
            "random_state": 0,
            "n_jobs": 1,
        }
        if multiclass:
            parameters["num_class"] = num_classes or 2
        return XGBClassifier(
            **parameters,
        )
    raise ValueError(f"unknown feature-model backend: {backend}")


class FeatureEdgeClassifier:
    version = "0.1-public-features"
    backend = "logistic"
    requires_fit = True

    def __init__(self) -> None:
        self.model: Any | None = None
        self.class_to_index = {action: index for index, action in enumerate(ACTIONS)}
        self.index_to_class = dict(enumerate(ACTIONS))
        self.model_index_to_action: dict[int, str] = {}
        self.training_sample_count = 0

    @property
    def name(self) -> str:
        return f"{self.backend}_edge"

    def fit(self, examples: Iterable[TrainingExample]) -> dict[str, Any]:
        features: list[list[float]] = []
        labels: list[int] = []
        task_count = 0
        for task, edges in examples:
            actions = gold_edge_actions(task)
            if not actions:
                continue
            task_count += 1
            active_nodes = training_active_nodes(task)
            for edge in edges:
                action = actions.get(edge.candidate_id)
                if action not in self.class_to_index:
                    continue
                features.append(edge_feature_vector(edge, edges, active_nodes))
                labels.append(self.class_to_index[action])
        if not features or len(set(labels)) < 2:
            raise ValueError("edge feature baseline requires at least two action classes")
        present = sorted(set(labels))
        encoded = {class_index: model_index for model_index, class_index in enumerate(present)}
        self.model_index_to_action = {
            model_index: self.index_to_class[class_index]
            for class_index, model_index in encoded.items()
        }
        self.model = _create_feature_model(
            self.backend, multiclass=True, num_classes=len(present)
        )
        self.model.fit(features, [encoded[label] for label in labels])
        self.training_sample_count = len(features)
        summary = {
            "task_count": task_count,
            "sample_count": len(features),
            "feature_names": list(EDGE_FEATURE_NAMES),
            "label_counts": {
                action: labels.count(index) for action, index in self.class_to_index.items()
            },
        }
        if self.backend == "mlp":
            summary.update(
                {
                    "architecture": "one_hidden_layer",
                    "hidden_dim": 50,
                    "epochs": 200,
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "feature_scaling": "training_standardization",
                }
            )
        return summary

    def predict(
        self,
        *,
        task: dict[str, Any],
        edges: list[CandidateEdge],
        active_nodes: list[str],
    ) -> list[ActionPrediction]:
        if self.model is None:
            raise RuntimeError(f"{self.name} must be fitted on disjoint training cases")
        matrix = [edge_feature_vector(edge, edges, active_nodes) for edge in edges]
        if not matrix:
            return []
        raw_probabilities = self.model.predict_proba(matrix)
        model_classes = [int(value) for value in self.model.classes_]
        predictions: list[ActionPrediction] = []
        for edge, row in zip(edges, raw_probabilities):
            probabilities = {action: 0.0 for action in ACTIONS}
            for class_index, probability in zip(model_classes, row):
                probabilities[self.model_index_to_action[class_index]] = float(probability)
            action = max(ACTIONS, key=lambda value: (probabilities[value], -ACTIONS.index(value)))
            predictions.append(
                ActionPrediction(
                    edge_id=edge.candidate_id,
                    action=action,
                    action_probabilities=probabilities,
                    rationale=(
                        f"{self.backend} public edge features; trained_samples="
                        f"{self.training_sample_count}"
                    ),
                )
            )
        return predictions


class LogisticEdgeClassifier(FeatureEdgeClassifier):
    backend = "logistic"


class RandomForestEdgeClassifier(FeatureEdgeClassifier):
    backend = "random_forest"


class MLPEdgeClassifier(FeatureEdgeClassifier):
    backend = "mlp"
    version = "1.0-weber2019-public-features"


class XGBoostEdgeClassifier(FeatureEdgeClassifier):
    backend = "xgboost"


class MaxOutflowSourceDetector:
    """Rank sources by their largest outgoing edge in the task subgraph."""

    name = "max_outflow_source"
    version = "0.1"

    def predict(
        self,
        *,
        task: dict[str, Any],
        edges: list[CandidateEdge],
        node_feature_rows: list[dict[str, Any]] | None = None,
    ) -> list[NodePrediction]:
        allowed = set(candidate_seed_nodes(task, edges))
        totals: dict[str, float] = defaultdict(float)
        maxima: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for edge in edges:
            src = normalize_node(edge.src)
            if not src or src not in allowed:
                continue
            totals[src] += edge.amount
            maxima[src] = max(maxima[src], edge.amount)
            counts[src] += 1
        if not totals:
            return []
        ranked = sorted(
            totals,
            key=lambda node: (-maxima[node], -totals[node], -counts[node], node),
        )
        largest_edge = max(maxima.values())
        if largest_edge > 0.0:
            weights = [maxima[node] / largest_edge for node in ranked]
        else:
            max_count = max(counts.values())
            weights = [counts[node] / max_count for node in ranked]
        weight_sum = sum(weights) or 1.0
        return [
            NodePrediction(
                node_id=node,
                probability=weight / weight_sum,
                score=maxima[node],
                rationale=(
                    f"largest outgoing edge amount={maxima[node]:.12g}; "
                    f"total outgoing amount={totals[node]:.12g}; "
                    f"outgoing relations={counts[node]}"
                ),
            )
            for node, weight in zip(ranked, weights)
        ]


class FeatureNodeDetector:
    version = "0.1-public-features"
    backend = "logistic"
    requires_fit = True

    def __init__(self) -> None:
        self.model: Any | None = None
        self.training_sample_count = 0

    @property
    def name(self) -> str:
        return f"{self.backend}_node"

    def fit(self, examples: Iterable[TrainingExample]) -> dict[str, Any]:
        features: list[list[float]] = []
        labels: list[int] = []
        task_count = 0
        for task, edges in examples:
            if task.get("task_type") != "subgraph_noise_filtering":
                continue
            candidates = candidate_seed_nodes(task, edges)
            if not candidates:
                continue
            task_count += 1
            gold = gold_seed_nodes(task)
            for node in candidates:
                features.append(node_feature_vector(node, task, edges))
                labels.append(int(normalize_node(node) in gold))
        if not features or len(set(labels)) < 2:
            raise ValueError("node feature baseline requires positive and negative seed samples")
        self.model = _create_feature_model(self.backend, multiclass=False)
        self.model.fit(features, labels)
        self.training_sample_count = len(features)
        return {
            "task_count": task_count,
            "sample_count": len(features),
            "feature_names": list(NODE_FEATURE_NAMES),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
        }

    def predict(
        self,
        *,
        task: dict[str, Any],
        edges: list[CandidateEdge],
        node_feature_rows: list[dict[str, Any]] | None = None,
    ) -> list[NodePrediction]:
        if self.model is None:
            raise RuntimeError(f"{self.name} must be fitted on disjoint training cases")
        candidates = candidate_seed_nodes(task, edges)
        if not candidates:
            return []
        matrix = [node_feature_vector(node, task, edges) for node in candidates]
        raw = self.model.predict_proba(matrix)
        classes = [int(value) for value in self.model.classes_]
        positive_index = classes.index(1)
        predictions = [
            NodePrediction(
                node_id=node,
                probability=float(row[positive_index]),
                score=float(row[positive_index]),
                rationale=(
                    f"{self.backend} public node features; trained_samples="
                    f"{self.training_sample_count}"
                ),
            )
            for node, row in zip(candidates, raw)
        ]
        return sorted(predictions, key=lambda item: (-item.probability, item.node_id))


class LogisticNodeDetector(FeatureNodeDetector):
    backend = "logistic"


class RandomForestNodeDetector(FeatureNodeDetector):
    backend = "random_forest"


class XGBoostNodeDetector(FeatureNodeDetector):
    backend = "xgboost"


class XGBoostWholeGraphOnePercentNodeDetector:
    """Whole-case actionable-node detector selected by the prevalence pilot.

    Training uses one row per node from declared disjoint case subgraphs and
    retains every positive while deterministically sampling negatives to a 1%
    positive rate. Target inference consumes feature-only rows.
    """

    name = "xgboost_node_1pct"
    version = "1.0-whole-subgraph-actionability"
    requires_fit = True
    training_scope = "whole_subgraph_actionability"
    training_positive_rate = 0.01

    def __init__(self) -> None:
        self.model: Any | None = None
        self.training_sample_count = 0
        self.training_summary: dict[str, Any] = {}

    def fit_node_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        seed: int,
        fold_identity: str,
    ) -> dict[str, Any]:
        sampled, sampling = sample_positive_rate(
            rows,
            positive_rate=self.training_positive_rate,
            seed=seed,
            fold_identity=fold_identity,
        )
        labels = [int(row["label"]) for row in sampled]
        self.model = _create_feature_model("xgboost", multiclass=False)
        self.model.fit([row["features"] for row in sampled], labels)
        self.training_sample_count = len(sampled)
        self.training_summary = {
            **sampling,
            "sample_count": len(sampled),
            "feature_names": list(WHOLE_GRAPH_NODE_FEATURE_NAMES),
            "sampling": "retain_all_positives_hash_sample_negatives",
            "label_semantics": "has_evaluable_follow_out_edge",
        }
        # A Phase-A orchestrator may freeze and reuse this exact fitted model
        # across edge methods for one outer target fold.
        self.requires_fit = False
        return dict(self.training_summary)

    def predict(
        self,
        *,
        task: dict[str, Any],
        edges: list[CandidateEdge],
        node_feature_rows: list[dict[str, Any]] | None = None,
    ) -> list[NodePrediction]:
        if self.model is None:
            raise RuntimeError(f"{self.name} must be fitted on disjoint training cases")
        candidates = candidate_seed_nodes(task, edges)
        by_node = {
            normalize_node(row.get("node_id")): row
            for row in node_feature_rows or []
            if row.get("node_id")
        }
        available = [node for node in candidates if normalize_node(node) in by_node]
        if not available:
            return []
        matrix = [by_node[normalize_node(node)]["features"] for node in available]
        raw = self.model.predict_proba(matrix)
        positive_index = [int(value) for value in self.model.classes_].index(1)
        predictions = [
            NodePrediction(
                node_id=normalize_node(node),
                probability=float(row[positive_index]),
                score=float(row[positive_index]),
                rationale=(
                    "xgboost whole-subgraph public node features; "
                    "training_positive_rate=0.01; "
                    f"trained_samples={self.training_sample_count}"
                ),
            )
            for node, row in zip(available, raw)
        ]
        return sorted(predictions, key=lambda item: (-item.probability, item.node_id))


EDGE_CLASSIFIERS = {
    PoisonEdgeClassifier.name: PoisonEdgeClassifier,
    "haircut": HaircutEdgeClassifier,
    "fifo": FIFOEdgeClassifier,
    "lifo": LIFOEdgeClassifier,
    "tiho": TIHOEdgeClassifier,
    TRacerEdgeClassifier.name: TRacerEdgeClassifier,
    "logistic_edge": LogisticEdgeClassifier,
    "random_forest_edge": RandomForestEdgeClassifier,
    "mlp_edge": MLPEdgeClassifier,
    GCNEdgeClassifier.name: GCNEdgeClassifier,
    I2BGNNEdgeClassifier.name: I2BGNNEdgeClassifier,
    PEAEGNNEdgeClassifier.name: PEAEGNNEdgeClassifier,
    TokenScoutEdgeClassifier.name: TokenScoutEdgeClassifier,
    "xgboost_edge": XGBoostEdgeClassifier,
    DenseFlowEdgeClassifier.name: DenseFlowEdgeClassifier,
    DenseFlowPlusEdgeClassifier.name: DenseFlowPlusEdgeClassifier,
}
NODE_DETECTORS = {
    MaxOutflowSourceDetector.name: MaxOutflowSourceDetector,
    "logistic_node": LogisticNodeDetector,
    "random_forest_node": RandomForestNodeDetector,
    "xgboost_node": XGBoostNodeDetector,
    XGBoostWholeGraphOnePercentNodeDetector.name: XGBoostWholeGraphOnePercentNodeDetector,
}


def create_edge_classifier(
    name: str, *, taint_budget_ratio: float = 0.5
) -> EdgeClassifier:
    try:
        classifier = EDGE_CLASSIFIERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown edge classifier: {name}") from exc
    if issubclass(classifier, ValueTaintEdgeClassifier):
        return classifier(taint_budget_ratio=taint_budget_ratio)
    return classifier()


def create_node_detector(name: str) -> NodeDetector:
    try:
        return NODE_DETECTORS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown node detector: {name}") from exc
