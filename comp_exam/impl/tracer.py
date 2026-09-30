from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .models import ACTIONS, ActionPrediction, CandidateEdge, normalize_node


BLOCKCHAIN_SPIDER_ROOT = (
    Path(__file__).resolve().parents[1] / "refs" / "BlockchainSpider"
)


def _reference_ttr_redirect() -> Any:
    """Load the bundled MIT-licensed BlockchainSpider TRacer implementation."""
    if not (BLOCKCHAIN_SPIDER_ROOT / "BlockchainSpider" / "strategies" / "txs" / "ttr.py").is_file():
        raise RuntimeError(
            "TRacer baseline requires comp_exam/refs/BlockchainSpider"
        )
    root = str(BLOCKCHAIN_SPIDER_ROOT)
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        from BlockchainSpider.strategies.txs.ttr import TTRRedirect
    finally:
        if inserted:
            sys.path.remove(root)
    module = sys.modules.get(TTRRedirect.__module__)
    module_path = Path(str(getattr(module, "__file__", ""))).resolve()
    if BLOCKCHAIN_SPIDER_ROOT.resolve() not in module_path.parents:
        raise RuntimeError(
            f"TRacer resolved an unexpected BlockchainSpider module: {module_path}"
        )
    return TTRRedirect


def _edge_symbol(edge: CandidateEdge) -> str:
    tokens = [str(value).lower() for value in edge.metadata.get("token_addresses") or []]
    if tokens:
        return "token:" + "|".join(sorted(tokens))
    assets = [str(value).lower() for value in edge.metadata.get("assets") or []]
    if assets:
        return "asset:" + "|".join(sorted(assets))
    return "edge_type:" + "|".join(sorted(edge.edge_types or ("unknown",)))


class TRacerEdgeClassifier:
    """TRacer/TTRRedirect adapter over the fixed public case graph.

    The bundled BlockchainSpider implementation supplies the original residual
    push and token-redirection logic. The benchmark already materializes the
    candidate graph, so API expansion is replaced by budgeted reads of that
    fixed graph, followed by the paper's rank-guided community extraction.
    """

    name = "tracer"
    version = "1.0-blockchainspider-ttrredirect-adaptation"
    requires_case_graph = True
    alpha = 0.15
    beta = 0.7
    epsilon = 1e-3
    conductance_threshold = 1e-3
    default_max_expansions = 64

    def __init__(self) -> None:
        self._case_key: tuple[str, str] | None = None
        self._edges: list[CandidateEdge] = []
        self._reference_edges_by_node: dict[str, list[dict[str, Any]]] = {}
        self._adjacency: dict[str, set[str]] = {}
        self._rank_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._fallback_amount_count = 0

    def prepare_case(
        self,
        *,
        case_id: str,
        dataset_version: str,
        edges: list[CandidateEdge],
    ) -> None:
        key = (case_id, dataset_version)
        if self._case_key == key:
            return
        self._case_key = key
        self._edges = list(edges)
        self._reference_edges_by_node = {}
        self._adjacency = {}
        self._rank_cache = {}
        self._fallback_amount_count = 0
        for edge in self._edges:
            src = normalize_node(edge.src)
            dst = normalize_node(edge.dst)
            if not src or not dst:
                continue
            value = float(edge.amount)
            if value <= 0.0:
                value = 1.0
                self._fallback_amount_count += 1
            timestamp = int(float(edge.metadata.get("min_timestamp") or 0.0))
            reference_edge = {
                "hash": str(edge.metadata.get("tx_hash") or edge.candidate_id),
                "from": src,
                "to": dst,
                "value": value,
                "timeStamp": timestamp,
                "symbol": _edge_symbol(edge),
            }
            self._reference_edges_by_node.setdefault(src, []).append(reference_edge)
            if dst != src:
                self._reference_edges_by_node.setdefault(dst, []).append(reference_edge)
            self._adjacency.setdefault(src, set()).add(dst)
            self._adjacency.setdefault(dst, set()).add(src)

    def _max_expansions(self, task: dict[str, Any]) -> int:
        budget = (task.get("graph_context") or {}).get("rollout_budget") or {}
        try:
            value = int(budget.get("max_node_expansions") or self.default_max_expansions)
        except (TypeError, ValueError):
            value = self.default_max_expansions
        return max(1, min(value, self.default_max_expansions))

    def _conductance(self, community: set[str], ranks: dict[str, float]) -> float:
        boundary = {
            neighbour
            for node in community
            for neighbour in self._adjacency.get(node, set())
            if neighbour not in community
        }
        internal_rank = sum(max(0.0, ranks.get(node, 0.0)) for node in community)
        if internal_rank <= 0.0:
            return float("inf")
        return sum(max(0.0, ranks.get(node, 0.0)) for node in boundary) / internal_rank

    def _rank(self, source: str, max_expansions: int) -> dict[str, Any]:
        cache_key = (source, max_expansions)
        if cache_key in self._rank_cache:
            return self._rank_cache[cache_key]
        strategy_class = _reference_ttr_redirect()
        strategy = strategy_class(
            source,
            alpha=self.alpha,
            beta=self.beta,
            epsilon=self.epsilon,
        )
        node: str | None = source
        context: dict[str, Any] = {}
        expanded: list[str] = []
        while node is not None and len(expanded) < max_expansions:
            strategy.push(
                node,
                [dict(edge) for edge in self._reference_edges_by_node.get(node, [])],
                **context,
            )
            expanded.append(node)
            node, context = strategy.pop()
        ranks = {
            normalize_node(key): max(0.0, float(value))
            for key, value in strategy.get_node_rank().items()
            if normalize_node(key)
        }
        community = {source}
        remaining = set(ranks) - community
        while (
            remaining
            and len(community) < max_expansions + 1
            and self._conductance(community, ranks) >= self.conductance_threshold
        ):
            selected = max(remaining, key=lambda value: (ranks.get(value, 0.0), value))
            community.add(selected)
            remaining.remove(selected)
        result = {
            "ranks": ranks,
            "community": community,
            "expansions": len(expanded),
            "terminated_by_epsilon": node is None,
        }
        self._rank_cache[cache_key] = result
        return result

    def predict(
        self,
        *,
        task: dict[str, Any],
        edges: list[CandidateEdge],
        active_nodes: list[str],
    ) -> list[ActionPrediction]:
        if not self._edges:
            self.prepare_case(
                case_id=str(task.get("case_id") or "task-local"),
                dataset_version=str(task.get("dataset_version") or ""),
                edges=edges,
            )
        sources = sorted({normalize_node(node) for node in active_nodes if normalize_node(node)})
        max_expansions = self._max_expansions(task)
        ranked = [self._rank(source, max_expansions) for source in sources]
        raw_scores = [
            max((result["ranks"].get(normalize_node(edge.dst), 0.0) for result in ranked), default=0.0)
            for edge in edges
        ]
        maximum = max(raw_scores, default=0.0) or 1.0
        predictions: list[ActionPrediction] = []
        for edge, raw_score in zip(edges, raw_scores):
            src = normalize_node(edge.src)
            dst = normalize_node(edge.dst)
            in_community = any(
                src in result["community"] and dst in result["community"]
                for result in ranked
            )
            relative = min(1.0, max(0.0, raw_score / maximum))
            follow = (0.5 + 0.49 * relative) if in_community else (0.49 * relative)
            probabilities = {
                "follow": follow,
                "inspect": 0.0,
                "stop": 0.0,
                "ignore": 1.0 - follow,
            }
            predictions.append(
                ActionPrediction(
                    edge_id=edge.candidate_id,
                    action="follow" if follow >= 0.5 else "ignore",
                    action_probabilities={name: probabilities[name] for name in ACTIONS},
                    rationale=(
                        "TRacer TTRRedirect via bundled BlockchainSpider; "
                        f"alpha={self.alpha}; beta={self.beta}; epsilon={self.epsilon}; "
                        f"phi={self.conductance_threshold}; fixed_graph_budget={max_expansions}; "
                        f"zero_amount_unit_fallbacks={self._fallback_amount_count}"
                    ),
                )
            )
        return predictions
