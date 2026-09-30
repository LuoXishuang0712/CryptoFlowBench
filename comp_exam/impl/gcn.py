from __future__ import annotations

from typing import Any, Iterable

from .features import EDGE_FEATURE_NAMES, edge_feature_vector, gold_edge_actions, training_active_nodes
from .models import ACTIONS, ActionPrediction, CandidateEdge, normalize_node


TrainingExample = tuple[dict[str, Any], list[CandidateEdge]]


def _normalized_line_graph(edges: list[CandidateEdge]) -> Any:
    """Return symmetric-normalized adjacency for the candidate-edge line graph."""
    try:
        import numpy as np
        from scipy import sparse
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GCN baseline requires numpy/scipy; run `uv sync --extra baselines`"
        ) from exc
    count = len(edges)
    if not count:
        return sparse.csr_matrix((0, 0), dtype=np.float64)
    node_ids = sorted(
        {
            node
            for edge in edges
            for node in (normalize_node(edge.src), normalize_node(edge.dst))
            if node
        }
    )
    node_index = {node: index for index, node in enumerate(node_ids)}
    rows: list[int] = []
    columns: list[int] = []
    for edge_index, edge in enumerate(edges):
        for node in dict.fromkeys((normalize_node(edge.src), normalize_node(edge.dst))):
            if node:
                rows.append(edge_index)
                columns.append(node_index[node])
    incidence = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(count, len(node_ids)),
        dtype=np.float64,
    )
    adjacency = (incidence @ incidence.T).tocsr()
    adjacency.data[:] = 1.0
    adjacency.setdiag(1.0)
    adjacency.eliminate_zeros()
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse_sqrt = np.zeros_like(degrees)
    nonzero = degrees > 0.0
    inverse_sqrt[nonzero] = degrees[nonzero] ** -0.5
    scale = sparse.diags(inverse_sqrt)
    return (scale @ adjacency @ scale).tocsr()


class GCNEdgeClassifier:
    """Two-layer GCN adaptation of Weber et al. (2019) to candidate edges.

    Candidate relations become line-graph nodes, joined when they share an
    endpoint. Training remains case-disjoint and consumes only the common
    public edge feature vector.
    """

    name = "gcn_edge"
    version = "1.0-weber2019-line-graph-adaptation"
    requires_fit = True
    hidden_dim = 100
    epochs = 1000
    learning_rate = 0.001

    def __init__(self) -> None:
        self.class_to_index = {action: index for index, action in enumerate(ACTIONS)}
        self.model_index_to_action: dict[int, str] = {}
        self.feature_mean: Any | None = None
        self.feature_scale: Any | None = None
        self.weight_input: Any | None = None
        self.weight_output: Any | None = None
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

    def fit(self, examples: Iterable[TrainingExample]) -> dict[str, Any]:
        try:
            import numpy as np
            from scipy import sparse
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "GCN baseline requires numpy/scipy; run `uv sync --extra baselines`"
            ) from exc
        matrices: list[Any] = []
        feature_blocks: list[Any] = []
        labels: list[int] = []
        task_count = 0
        for task, edges in examples:
            actions = gold_edge_actions(task)
            usable = [edge for edge in edges if actions.get(edge.candidate_id) in self.class_to_index]
            if not usable:
                continue
            task_count += 1
            active_nodes = training_active_nodes(task)
            feature_blocks.append(
                np.asarray(
                    [edge_feature_vector(edge, usable, active_nodes) for edge in usable],
                    dtype=np.float64,
                )
            )
            matrices.append(_normalized_line_graph(usable))
            labels.extend(self.class_to_index[actions[edge.candidate_id]] for edge in usable)
        if not feature_blocks or len(set(labels)) < 2:
            raise ValueError("GCN edge baseline requires at least two action classes")
        present = sorted(set(labels))
        encoded = {class_index: model_index for model_index, class_index in enumerate(present)}
        self.model_index_to_action = {
            model_index: ACTIONS[class_index] for class_index, model_index in encoded.items()
        }
        targets = np.asarray([encoded[label] for label in labels], dtype=np.int64)
        features = np.vstack(feature_blocks)
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = features.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-12] = 1.0
        features = (features - self.feature_mean) / self.feature_scale
        adjacency = sparse.block_diag(matrices, format="csr")
        aggregated_features = adjacency @ features

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
            hidden_linear = aggregated_features @ self.weight_input
            hidden = np.maximum(hidden_linear, 0.0)
            aggregated_hidden = adjacency @ hidden
            logits = aggregated_hidden @ self.weight_output
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            final_loss = float(
                -np.sum(sample_weights * np.log(probabilities[np.arange(len(targets)), targets] + 1e-12))
                / weight_total
            )
            gradient_logits = (probabilities - one_hot) * (sample_weights / weight_total)[:, None]
            gradient_output = aggregated_hidden.T @ gradient_logits
            gradient_hidden = adjacency.T @ (gradient_logits @ self.weight_output.T)
            gradient_input = aggregated_features.T @ (
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
            "feature_names": list(EDGE_FEATURE_NAMES),
            "label_counts": {
                action: labels.count(index) for action, index in self.class_to_index.items()
            },
            "architecture": "two_layer_gcn",
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "optimizer": "adam",
            "learning_rate": self.learning_rate,
            "loss": "class_weighted_cross_entropy",
            "final_training_loss": final_loss,
            "graph_adaptation": "candidate_edges_as_line_graph_nodes_shared_endpoint_adjacency",
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
        import numpy as np

        features = np.asarray(
            [edge_feature_vector(edge, edges, active_nodes) for edge in edges],
            dtype=np.float64,
        )
        features = (features - self.feature_mean) / self.feature_scale
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
            action = max(ACTIONS, key=lambda value: (probabilities[value], -ACTIONS.index(value)))
            predictions.append(
                ActionPrediction(
                    edge_id=edge.candidate_id,
                    action=action,
                    action_probabilities=probabilities,
                    rationale=(
                        "two-layer GCN on public-feature candidate line graph; "
                        f"trained_samples={self.training_sample_count}"
                    ),
                )
            )
        return predictions
