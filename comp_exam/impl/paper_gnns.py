from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from .features import EDGE_FEATURE_NAMES, edge_feature_vector, gold_edge_actions, training_active_nodes
from .gcn import _normalized_line_graph
from .models import ACTIONS, ActionPrediction, CandidateEdge, normalize_node


TrainingExample = tuple[dict[str, Any], list[CandidateEdge]]


def _require_numeric() -> tuple[Any, Any]:
    try:
        import numpy as np
        from scipy import sparse
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "paper-derived GNN baselines require numpy/scipy; "
            "run `uv sync --extra baselines`"
        ) from exc
    return np, sparse


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _timestamp(edge: CandidateEdge) -> float:
    return _safe_float(
        edge.metadata.get("min_timestamp")
        or edge.metadata.get("timestamp")
        or edge.metadata.get("min_block_number")
    )


def _aggregate(values: list[float]) -> list[float]:
    np, _ = _require_numeric()
    if not values:
        return [0.0] * 6
    row = np.asarray(values, dtype=np.float64)
    return [
        float(row.max()),
        float(row.min()),
        float(row.mean()),
        float(np.median(row)),
        float(row.std()),
        float(row.sum()),
    ]


def _base_matrix(
    edges: list[CandidateEdge], active_nodes: list[str]
) -> Any:
    np, _ = _require_numeric()
    return np.asarray(
        [edge_feature_vector(edge, edges, active_nodes) for edge in edges],
        dtype=np.float64,
    )


def _append_graph_summary(matrix: Any, summary: Any) -> Any:
    np, _ = _require_numeric()
    return np.hstack([matrix, np.repeat(summary[None, :], len(matrix), axis=0)])


def _i2bgnn_features(edges: list[CandidateEdge], active_nodes: list[str]) -> Any:
    """Transaction-node adaptation of two-layer GCN plus graph max pooling."""
    np, _ = _require_numeric()
    base = _base_matrix(edges, active_nodes)
    adjacency = _normalized_line_graph(edges)
    propagated = adjacency @ (adjacency @ base)
    graph_max = np.max(propagated, axis=0) if len(propagated) else np.zeros(base.shape[1])
    return _append_graph_summary(np.hstack([base, propagated]), graph_max)


def _peae_features(edges: list[CandidateEdge], active_nodes: list[str]) -> Any:
    """PEAE structure/transaction/intensity features with RTM top-3 readout."""
    np, _ = _require_numeric()
    base = _base_matrix(edges, active_nodes)
    outgoing: dict[str, list[CandidateEdge]] = defaultdict(list)
    incoming: dict[str, list[CandidateEdge]] = defaultdict(list)
    for edge in edges:
        outgoing[normalize_node(edge.src)].append(edge)
        incoming[normalize_node(edge.dst)].append(edge)
    augmented: list[list[float]] = []
    for edge, public in zip(edges, base):
        src_rows = outgoing[normalize_node(edge.src)]
        dst_rows = incoming[normalize_node(edge.dst)]
        src_amounts = [item.amount for item in src_rows]
        dst_amounts = [item.amount for item in dst_rows]
        src_times = sorted(value for item in src_rows if (value := _timestamp(item)) > 0.0)
        dst_times = sorted(value for item in dst_rows if (value := _timestamp(item)) > 0.0)
        intervals = [
            right - left
            for values in (src_times, dst_times)
            for left, right in zip(values, values[1:])
        ]
        src_sum = sum(src_amounts)
        dst_sum = sum(dst_amounts)
        src_unique = len({normalize_node(item.dst) for item in src_rows})
        dst_unique = len({normalize_node(item.src) for item in dst_rows})
        augmented.append(
            list(public)
            + _aggregate(src_amounts)
            + _aggregate(dst_amounts)
            + [
                max(src_amounts, default=0.0) / src_sum if src_sum else 0.0,
                max(dst_amounts, default=0.0) / dst_sum if dst_sum else 0.0,
                src_unique / len(src_rows) if src_rows else 0.0,
                dst_unique / len(dst_rows) if dst_rows else 0.0,
            ]
            + _aggregate(intervals)
        )
    matrix = np.asarray(augmented, dtype=np.float64)
    adjacency = _normalized_line_graph(edges)
    propagated = adjacency @ matrix
    top_n = min(3, len(propagated))
    rtm = (
        np.sort(propagated, axis=0)[-top_n:].mean(axis=0)
        if top_n
        else np.zeros(matrix.shape[1])
    )
    return _append_graph_summary(np.hstack([matrix, propagated]), rtm)


def _row_normalized_temporal_adjacency(edges: list[CandidateEdge]) -> Any:
    """Connect an earlier transaction-node to a later one through a shared account."""
    np, sparse = _require_numeric()
    count = len(edges)
    rows: list[int] = []
    columns: list[int] = []
    for target, current in enumerate(edges):
        current_time = _timestamp(current)
        current_nodes = {normalize_node(current.src), normalize_node(current.dst)}
        for source, previous in enumerate(edges):
            previous_time = _timestamp(previous)
            if source == target:
                rows.append(target)
                columns.append(source)
                continue
            if not current_nodes.intersection(
                {normalize_node(previous.src), normalize_node(previous.dst)}
            ):
                continue
            if current_time > 0.0 and previous_time > current_time:
                continue
            rows.append(target)
            columns.append(source)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(count, count), dtype=np.float64
    )
    degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(degree)
    inverse[degree > 0.0] = 1.0 / degree[degree > 0.0]
    return sparse.diags(inverse) @ matrix


def _tokenscout_features(edges: list[CandidateEdge], active_nodes: list[str]) -> Any:
    """TF-GNN-inspired public temporal/node/edge/role representation."""
    np, _ = _require_numeric()
    base = _base_matrix(edges, active_nodes)
    timestamps = np.asarray([_timestamp(edge) for edge in edges], dtype=np.float64)
    positive = timestamps[timestamps > 0.0]
    origin = float(positive.min()) if len(positive) else 0.0
    relative = np.maximum(0.0, timestamps - origin)
    scale = float(relative.max()) or 1.0
    normalized_time = relative / scale
    frequencies = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    temporal_encoding = np.hstack(
        [
            np.sin(normalized_time[:, None] * frequencies[None, :] * math.pi),
            np.cos(normalized_time[:, None] * frequencies[None, :] * math.pi),
        ]
    )
    temporal_rows: list[list[float]] = []
    for index, edge in enumerate(edges):
        src = normalize_node(edge.src)
        dst = normalize_node(edge.dst)
        now = timestamps[index]
        before = [
            item
            for item in edges
            if _timestamp(item) <= now or now <= 0.0
        ]
        src_before = [item for item in before if normalize_node(item.src) == src]
        dst_before = [item for item in before if normalize_node(item.dst) == dst]
        windows: list[float] = []
        for hours in (6.0, 48.0):
            lower = now - hours * 3600.0
            src_window = [item for item in src_before if _timestamp(item) >= lower]
            dst_window = [item for item in dst_before if _timestamp(item) >= lower]
            windows.extend(
                [
                    float(len(src_window)),
                    sum(item.amount for item in src_window),
                    float(len(dst_window)),
                    sum(item.amount for item in dst_window),
                ]
            )
        temporal_rows.append(
            [
                math.log1p(sum(item.amount for item in src_before)),
                math.log1p(sum(item.amount for item in dst_before)),
                math.log1p(len(src_before)),
                math.log1p(len(dst_before)),
                *[math.log1p(max(0.0, value)) for value in windows],
            ]
        )
    matrix = np.hstack(
        [base, temporal_encoding, np.asarray(temporal_rows, dtype=np.float64)]
    )
    temporal_adjacency = _row_normalized_temporal_adjacency(edges)
    messages = temporal_adjacency @ matrix
    global_mean = matrix.mean(axis=0) if len(matrix) else np.zeros(matrix.shape[1])
    source_groups: dict[str, list[int]] = defaultdict(list)
    destination_groups: dict[str, list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        source_groups[normalize_node(edge.src)].append(index)
        destination_groups[normalize_node(edge.dst)].append(index)
    role_rows = []
    for edge in edges:
        creator = matrix[source_groups[normalize_node(edge.src)]].mean(axis=0)
        investor = matrix[destination_groups[normalize_node(edge.dst)]].mean(axis=0)
        role_rows.append(np.concatenate([global_mean, creator, investor]))
    return np.hstack([matrix, messages, np.asarray(role_rows, dtype=np.float64)])


class PaperAdaptedGNNClassifier:
    """Shared trainable two-layer GCN head for paper-derived edge adaptations."""

    requires_fit = True
    hidden_dim = 128
    epochs = 64
    learning_rate = 0.001
    feature_builder = staticmethod(_i2bgnn_features)
    feature_names: tuple[str, ...] = EDGE_FEATURE_NAMES
    paper_title = ""
    adaptation = "candidate relations as transaction line-graph nodes"
    representation_refinement = "none"

    def __init__(self) -> None:
        self.class_to_index = {action: index for index, action in enumerate(ACTIONS)}
        self.model_index_to_action: dict[int, str] = {}
        self.feature_mean: Any | None = None
        self.feature_scale: Any | None = None
        self.weight_input: Any | None = None
        self.weight_output: Any | None = None
        self.prototype_centers: Any | None = None
        self.training_sample_count = 0

    @staticmethod
    def _adam_step(
        parameter: Any,
        gradient: Any,
        first: Any,
        second: Any,
        *,
        step: int,
        learning_rate: float,
    ) -> tuple[Any, Any, Any]:
        beta1, beta2 = 0.9, 0.999
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * (gradient * gradient)
        first_hat = first / (1.0 - beta1**step)
        second_hat = second / (1.0 - beta2**step)
        parameter = parameter - learning_rate * first_hat / (second_hat**0.5 + 1e-8)
        return parameter, first, second

    def _refine_fit(self, blocks: list[Any], targets: Any) -> list[Any]:
        if self.representation_refinement != "training_class_prototype_distances":
            return blocks
        np, _ = _require_numeric()
        stacked = np.vstack(blocks)
        present = sorted(set(int(value) for value in targets))
        self.prototype_centers = np.vstack(
            [stacked[targets == value].mean(axis=0) for value in present]
        )
        return self._append_prototype_distances(blocks)

    def _append_prototype_distances(self, blocks: list[Any]) -> list[Any]:
        np, _ = _require_numeric()
        if self.prototype_centers is None:
            return blocks
        return [
            np.hstack(
                [
                    block,
                    -np.sqrt(
                        ((block[:, None, :] - self.prototype_centers[None, :, :]) ** 2).mean(
                            axis=2
                        )
                    ),
                ]
            )
            for block in blocks
        ]

    def fit(self, examples: Iterable[TrainingExample]) -> dict[str, Any]:
        np, sparse = _require_numeric()
        matrices: list[Any] = []
        feature_blocks: list[Any] = []
        labels: list[int] = []
        task_count = 0
        for task, edges in examples:
            actions = gold_edge_actions(task)
            usable = [
                edge
                for edge in edges
                if actions.get(edge.candidate_id) in self.class_to_index
            ]
            if not usable:
                continue
            task_count += 1
            feature_blocks.append(
                self.feature_builder(usable, training_active_nodes(task))
            )
            matrices.append(_normalized_line_graph(usable))
            labels.extend(
                self.class_to_index[actions[edge.candidate_id]] for edge in usable
            )
        if not feature_blocks or len(set(labels)) < 2:
            raise ValueError(f"{self.name} requires at least two action classes")
        present = sorted(set(labels))
        encoded = {value: index for index, value in enumerate(present)}
        self.model_index_to_action = {
            index: ACTIONS[value] for value, index in encoded.items()
        }
        targets = np.asarray([encoded[label] for label in labels], dtype=np.int64)
        feature_blocks = self._refine_fit(feature_blocks, targets)
        features = np.vstack(feature_blocks)
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = features.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-12] = 1.0
        features = (features - self.feature_mean) / self.feature_scale
        adjacency = sparse.block_diag(matrices, format="csr")
        aggregated = adjacency @ features
        random = np.random.default_rng(0)
        input_limit = (6.0 / (features.shape[1] + self.hidden_dim)) ** 0.5
        output_limit = (6.0 / (self.hidden_dim + len(present))) ** 0.5
        self.weight_input = random.uniform(
            -input_limit, input_limit, (features.shape[1], self.hidden_dim)
        )
        self.weight_output = random.uniform(
            -output_limit, output_limit, (self.hidden_dim, len(present))
        )
        first_input = np.zeros_like(self.weight_input)
        second_input = np.zeros_like(self.weight_input)
        first_output = np.zeros_like(self.weight_output)
        second_output = np.zeros_like(self.weight_output)
        counts = np.bincount(targets, minlength=len(present)).astype(np.float64)
        class_weights = len(targets) / (len(present) * counts)
        sample_weights = class_weights[targets]
        weight_total = sample_weights.sum()
        one_hot = np.eye(len(present), dtype=np.float64)[targets]
        final_loss = 0.0
        for epoch in range(1, self.epochs + 1):
            hidden_linear = aggregated @ self.weight_input
            hidden = np.maximum(hidden_linear, 0.0)
            aggregated_hidden = adjacency @ hidden
            logits = aggregated_hidden @ self.weight_output
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            final_loss = float(
                -np.sum(
                    sample_weights
                    * np.log(probabilities[np.arange(len(targets)), targets] + 1e-12)
                )
                / weight_total
            )
            gradient_logits = (probabilities - one_hot) * (
                sample_weights / weight_total
            )[:, None]
            gradient_output = aggregated_hidden.T @ gradient_logits
            gradient_hidden = adjacency.T @ (gradient_logits @ self.weight_output.T)
            gradient_input = aggregated.T @ (
                gradient_hidden * (hidden_linear > 0.0)
            )
            self.weight_input, first_input, second_input = self._adam_step(
                self.weight_input,
                gradient_input,
                first_input,
                second_input,
                step=epoch,
                learning_rate=self.learning_rate,
            )
            self.weight_output, first_output, second_output = self._adam_step(
                self.weight_output,
                gradient_output,
                first_output,
                second_output,
                step=epoch,
                learning_rate=self.learning_rate,
            )
        self.training_sample_count = len(targets)
        return {
            "task_count": task_count,
            "sample_count": self.training_sample_count,
            "label_counts": {
                action: labels.count(index)
                for action, index in self.class_to_index.items()
            },
            "architecture": "paper_derived_two_layer_transaction_node_gcn",
            "paper_title": self.paper_title,
            "adaptation": self.adaptation,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "optimizer": "adam",
            "learning_rate": self.learning_rate,
            "loss": "class_weighted_cross_entropy",
            "representation_refinement": self.representation_refinement,
            "final_training_loss": final_loss,
            "feature_dimension": int(features.shape[1]),
            "public_features_only": True,
        }

    def predict(
        self,
        *,
        task: dict[str, Any],
        edges: list[CandidateEdge],
        active_nodes: list[str],
    ) -> list[ActionPrediction]:
        del task
        if self.weight_input is None or self.weight_output is None:
            raise RuntimeError(f"{self.name} must be fitted on disjoint training cases")
        if not edges:
            return []
        np, _ = _require_numeric()
        blocks = [self.feature_builder(edges, active_nodes)]
        blocks = self._append_prototype_distances(blocks)
        features = (blocks[0] - self.feature_mean) / self.feature_scale
        adjacency = _normalized_line_graph(edges)
        hidden = np.maximum((adjacency @ features) @ self.weight_input, 0.0)
        logits = (adjacency @ hidden) @ self.weight_output
        logits -= logits.max(axis=1, keepdims=True)
        rows = np.exp(logits)
        rows /= rows.sum(axis=1, keepdims=True)
        predictions: list[ActionPrediction] = []
        for edge, row in zip(edges, rows):
            probabilities = {action: 0.0 for action in ACTIONS}
            for model_index, probability in enumerate(row):
                probabilities[self.model_index_to_action[model_index]] = float(probability)
            action = max(
                ACTIONS,
                key=lambda value: (probabilities[value], -ACTIONS.index(value)),
            )
            predictions.append(
                ActionPrediction(
                    edge_id=edge.candidate_id,
                    action=action,
                    action_probabilities=probabilities,
                    rationale=(
                        f"{self.name} paper-derived transaction-node adaptation; "
                        f"trained_samples={self.training_sample_count}"
                    ),
                )
            )
        return predictions


class I2BGNNEdgeClassifier(PaperAdaptedGNNClassifier):
    name = "i2bgnn_edge"
    version = "1.0-paper-derived-transaction-node-adaptation"
    paper_title = "Identity Inference on Blockchain using Graph Neural Network"
    feature_builder = staticmethod(_i2bgnn_features)
    epochs = 50
    adaptation = (
        "candidate relations are line-graph transaction nodes; local embeddings "
        "are concatenated with the paper's graph max-pooled context"
    )


class PEAEGNNEdgeClassifier(PaperAdaptedGNNClassifier):
    name = "peae_gnn_edge"
    version = "1.0-paper-derived-transaction-node-adaptation"
    paper_title = "PEAE-GNN: Phishing Detection Through Feature Augmentation"
    feature_builder = staticmethod(_peae_features)
    epochs = 64
    adaptation = (
        "candidate relations are line-graph transaction nodes with public "
        "structure, amount, interaction-intensity, interval, and RTM top-3 context"
    )


class TokenScoutEdgeClassifier(PaperAdaptedGNNClassifier):
    name = "tokenscout_edge"
    version = "1.0-paper-derived-transaction-node-adaptation"
    paper_title = "TokenScout: Early Detection of Ethereum Scam Tokens via Temporal Graph Learning"
    feature_builder = staticmethod(_tokenscout_features)
    epochs = 60
    representation_refinement = "training_class_prototype_distances"
    adaptation = (
        "candidate relations are temporal transaction nodes; TF-GNN node/edge/time "
        "messages and graph/source/destination role fusion are mapped to edge actions; "
        "the paper's token-level asymmetric contrastive stage is approximated by "
        "training-class prototype-distance refinement"
    )
