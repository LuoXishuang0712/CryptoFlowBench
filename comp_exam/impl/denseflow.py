from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .models import ACTIONS, ActionPrediction, CandidateEdge, normalize_node


SECONDS_PER_MONTH = 30 * 24 * 60 * 60
BURST_RADIUS_SECONDS = 2 * 24 * 60 * 60
HISTOGRAM_BINS = 10
EPSILON = 1e-12


def _asset_bucket(edge: CandidateEdge) -> tuple[str, ...]:
    tokens = tuple(str(value).lower() for value in edge.metadata.get("token_addresses") or [])
    if tokens:
        return ("token",) + tokens
    assets = tuple(str(value).lower() for value in edge.metadata.get("assets") or [])
    if assets:
        return ("asset",) + assets
    return ("edge_type",) + tuple(edge.edge_types or ("unknown",))


def _timestamp(edge: CandidateEdge) -> float:
    try:
        value = float(edge.metadata.get("min_timestamp") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def _gini(values: list[float]) -> float:
    usable = sorted(max(0.0, float(value)) for value in values)
    total = sum(usable)
    if not usable or total <= 0.0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(usable))
    return max(0.0, min(1.0, 2.0 * weighted / (len(usable) * total) - (len(usable) + 1) / len(usable)))


@dataclass(frozen=True)
class _Transaction:
    src: str
    dst: str
    bucket: tuple[str, ...]
    capacity: float
    timestamp: float
    rating: float
    temporal: float
    rating_bin: int


def _transactions(edges: list[CandidateEdge], *, plus: bool) -> list[_Transaction]:
    raw: list[dict[str, Any]] = []
    groups: dict[tuple[str, tuple[str, ...], int], list[int]] = defaultdict(list)
    for edge in edges:
        src = normalize_node(edge.src)
        dst = normalize_node(edge.dst)
        if not src or not dst or src == dst:
            continue
        timestamp = _timestamp(edge)
        capacity = edge.amount if edge.amount > 0.0 else 1.0
        bucket = _asset_bucket(edge)
        month = int(timestamp // SECONDS_PER_MONTH) if timestamp else 0
        index = len(raw)
        raw.append(
            {
                "src": src,
                "dst": dst,
                "bucket": bucket,
                "capacity": capacity,
                "timestamp": timestamp,
            }
        )
        groups[(dst, bucket, month)].append(index)

    ratings = [0.0] * len(raw)
    temporal = [0.0] * len(raw)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: raw[index]["timestamp"])
        total = sum(raw[index]["capacity"] for index in ordered) or 1.0
        left = 0
        right = 0
        window_sum = 0.0
        daily: dict[int, float] = defaultdict(float)
        for index in ordered:
            day = int(raw[index]["timestamp"] // (24 * 60 * 60))
            daily[day] += raw[index]["capacity"]
        daily_values = [daily[day] for day in sorted(daily)]
        gini = _gini(daily_values)
        previous_by_day: dict[int, float] = {}
        previous = 0.0
        for day in sorted(daily):
            previous_by_day[day] = previous
            previous = daily[day]
        for position, index in enumerate(ordered):
            center = raw[index]["timestamp"]
            while right < len(ordered) and raw[ordered[right]]["timestamp"] <= center + BURST_RADIUS_SECONDS:
                window_sum += raw[ordered[right]]["capacity"]
                right += 1
            while left < len(ordered) and raw[ordered[left]]["timestamp"] < center - BURST_RADIUS_SECONDS:
                window_sum -= raw[ordered[left]]["capacity"]
                left += 1
            rating = max(0.0, min(1.0, window_sum / total))
            ratings[index] = rating
            if plus:
                day = int(center // (24 * 60 * 60))
                positive_jump = max(0.0, daily[day] - previous_by_day.get(day, 0.0)) / total
                temporal[index] = rating * (1.0 + gini) * (1.0 + positive_jump)
            else:
                # DenseFlow delegates awakening/peak extraction to MultiBurst,
                # whose full parameterization is absent from the paper. The
                # frozen adapter uses the paper's explicit +/-2-day ratio.
                temporal[index] = rating

    output: list[_Transaction] = []
    for index, row in enumerate(raw):
        value = max(0.0, ratings[index])
        output.append(
            _Transaction(
                src=row["src"],
                dst=row["dst"],
                bucket=row["bucket"],
                capacity=row["capacity"],
                timestamp=row["timestamp"],
                rating=value,
                temporal=max(0.0, temporal[index]),
                rating_bin=min(HISTOGRAM_BINS - 1, int(value * HISTOGRAM_BINS)),
            )
        )
    return output


def _kl_divergence(left: list[int], right: list[int]) -> float:
    left_total = sum(left)
    right_total = sum(right)
    if left_total <= 0 or right_total <= 0:
        return 0.0
    epsilon = 1e-9
    left_denominator = left_total + epsilon * len(left)
    right_denominator = right_total + epsilon * len(right)
    return sum(
        ((l + epsilon) / left_denominator)
        * math.log(
            ((l + epsilon) / left_denominator)
            / ((r + epsilon) / right_denominator)
        )
        for l, r in zip(left, right)
    )


def _dense_subset(
    transactions: list[_Transaction], *, base: float
) -> tuple[set[str], dict[str, float]]:
    nodes = {tx.src for tx in transactions} | {tx.dst for tx in transactions}
    if not nodes:
        return set(), {}
    outgoing: dict[str, list[_Transaction]] = defaultdict(list)
    total_in: dict[str, int] = defaultdict(int)
    total_temporal: dict[str, float] = defaultdict(float)
    total_hist: dict[str, list[int]] = {
        node: [0] * HISTOGRAM_BINS for node in nodes
    }
    for tx in transactions:
        outgoing[tx.src].append(tx)
        total_in[tx.dst] += 1
        total_temporal[tx.dst] += tx.temporal
        total_hist[tx.dst][tx.rating_bin] += 1

    active = set(nodes)
    internal_in = dict(total_in)
    internal_temporal = dict(total_temporal)
    internal_hist = {node: list(total_hist[node]) for node in nodes}
    scores: dict[str, float] = {}

    def node_score(node: str) -> float:
        inside = internal_in.get(node, 0)
        total = total_in.get(node, 0)
        if inside <= 0 or total <= 0:
            return 0.0
        alpha = inside / total
        temporal_total = total_temporal.get(node, 0.0)
        beta = (
            internal_temporal.get(node, 0.0) / temporal_total
            if temporal_total > 0.0
            else 0.0
        )
        complement = [
            total_hist[node][index] - internal_hist[node][index]
            for index in range(HISTOGRAM_BINS)
        ]
        outside = total - inside
        gamma = 0.0
        if outside > 0:
            balance = min(inside / outside, outside / inside)
            kl = _kl_divergence(internal_hist[node], complement)
            gamma = balance * (1.0 - math.exp(-max(0.0, kl)))
        exponent = max(-12.0, min(3.0, alpha + beta + gamma - 3.0))
        return inside * (base**exponent)

    heap: list[tuple[float, int, str]] = []
    versions: dict[str, int] = defaultdict(int)
    total_score = 0.0
    for node in nodes:
        scores[node] = node_score(node)
        total_score += scores[node]
        heapq.heappush(heap, (scores[node], versions[node], node))
    best_score = total_score / len(active)
    best_set = set(active)
    best_node_scores = dict(scores)

    while active:
        while heap:
            _, version, node = heapq.heappop(heap)
            if node in active and version == versions[node]:
                break
        else:
            break
        total_score -= scores.get(node, 0.0)
        active.remove(node)
        for tx in outgoing.get(node, []):
            target = tx.dst
            if target not in active:
                continue
            total_score -= scores.get(target, 0.0)
            internal_in[target] = max(0, internal_in.get(target, 0) - 1)
            internal_temporal[target] = max(
                0.0, internal_temporal.get(target, 0.0) - tx.temporal
            )
            internal_hist[target][tx.rating_bin] = max(
                0, internal_hist[target][tx.rating_bin] - 1
            )
            scores[target] = node_score(target)
            total_score += scores[target]
            versions[target] += 1
            heapq.heappush(heap, (scores[target], versions[target], target))
        if active:
            objective = total_score / len(active)
            if objective > best_score + EPSILON:
                best_score = objective
                best_set = set(active)
                best_node_scores = {node_id: scores[node_id] for node_id in active}
    maximum = max(best_node_scores.values(), default=0.0)
    normalized = {
        node: score / maximum if maximum > 0.0 else 0.0
        for node, score in best_node_scores.items()
    }
    return best_set, normalized


class _Dinic:
    def __init__(self) -> None:
        self.graph: dict[str, list[list[Any]]] = defaultdict(list)

    def add_edge(self, src: str, dst: str, capacity: float, pair: tuple[str, str] | None) -> None:
        forward = [dst, float(capacity), len(self.graph[dst]), pair, float(capacity)]
        reverse = [src, 0.0, len(self.graph[src]), None, 0.0]
        self.graph[src].append(forward)
        self.graph[dst].append(reverse)

    def solve(self, source: str, sink: str) -> set[tuple[str, str]]:
        used_pairs: set[tuple[str, str]] = set()
        while True:
            level = {source: 0}
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge[1] > EPSILON and edge[0] not in level:
                        level[edge[0]] = level[node] + 1
                        queue.append(edge[0])
            if sink not in level:
                break
            positions: dict[str, int] = defaultdict(int)

            def send(node: str, amount: float) -> float:
                if node == sink:
                    return amount
                while positions[node] < len(self.graph[node]):
                    edge = self.graph[node][positions[node]]
                    if edge[1] > EPSILON and level.get(edge[0]) == level[node] + 1:
                        pushed = send(edge[0], min(amount, edge[1]))
                        if pushed > EPSILON:
                            edge[1] -= pushed
                            self.graph[edge[0]][edge[2]][1] += pushed
                            return pushed
                    positions[node] += 1
                return 0.0

            while send(source, float("inf")) > EPSILON:
                pass
        for edges in self.graph.values():
            for edge in edges:
                if edge[3] is not None and edge[4] - edge[1] > EPSILON:
                    used_pairs.add(edge[3])
        return used_pairs


def _maximum_flow_pairs(
    transactions: list[_Transaction],
    *,
    sources: set[str],
    suspicious: set[str],
) -> set[tuple[str, str]]:
    used: set[tuple[str, str]] = set()
    by_bucket: dict[tuple[str, ...], list[_Transaction]] = defaultdict(list)
    for tx in transactions:
        by_bucket[tx.bucket].append(tx)
    for bucket_transactions in by_bucket.values():
        nodes = {tx.src for tx in bucket_transactions} | {tx.dst for tx in bucket_transactions}
        targets = (suspicious & nodes) - sources
        bucket_sources = sources & nodes
        if not targets or not bucket_sources:
            continue
        capacities: dict[tuple[str, str], float] = defaultdict(float)
        for tx in bucket_transactions:
            capacities[(tx.src, tx.dst)] += tx.capacity
        total_capacity = sum(capacities.values()) or 1.0
        source_id = "__denseflow_super_source__"
        sink_id = "__denseflow_super_sink__"
        solver = _Dinic()
        for (src, dst), capacity in capacities.items():
            solver.add_edge(src, dst, capacity, (src, dst))
        for source in bucket_sources:
            solver.add_edge(source_id, source, total_capacity + 1.0, None)
        for target in targets:
            solver.add_edge(target, sink_id, total_capacity + 1.0, None)
        used.update(solver.solve(source_id, sink_id))
    return used


class DenseFlowEdgeClassifier:
    """Paper-derived DenseFlow adapter for the benchmark edge interface.

    The original method outputs suspicious nodes and max-flow paths. This
    adapter maps path relations to ``follow`` and all other candidates to
    ``ignore``. MultiBurst details absent from the paper are frozen to the
    explicit +/-2-day rating in Eq. (4). Amount capacities are isolated per
    asset/token bucket.
    """

    name = "denseflow"
    version = "0.1-paper-derived-adaptation"
    requires_case_graph = True
    plus = False
    fusion_base = 2.0

    def __init__(self) -> None:
        self._case_key: tuple[str, str] | None = None
        self._transactions: list[_Transaction] = []
        self._suspicious: set[str] = set()
        self._node_scores: dict[str, float] = {}
        self._flow_cache: dict[tuple[str, ...], set[tuple[str, str]]] = {}

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
        self._transactions = _transactions(edges, plus=self.plus)
        self._suspicious, self._node_scores = _dense_subset(
            self._transactions, base=self.fusion_base
        )
        self._flow_cache = {}

    def predict(
        self,
        *,
        task: dict[str, Any],
        edges: list[CandidateEdge],
        active_nodes: list[str],
    ) -> list[ActionPrediction]:
        if not self._transactions:
            self.prepare_case(
                case_id=str(task.get("case_id") or "task-local"),
                dataset_version=str(task.get("dataset_version") or ""),
                edges=edges,
            )
        sources = tuple(sorted({normalize_node(node) for node in active_nodes if node}))
        if sources not in self._flow_cache:
            self._flow_cache[sources] = _maximum_flow_pairs(
                self._transactions,
                sources=set(sources),
                suspicious=self._suspicious,
            )
        flow_pairs = self._flow_cache[sources]
        output: list[ActionPrediction] = []
        for edge in edges:
            pair = (normalize_node(edge.src), normalize_node(edge.dst))
            follow = 0.99 if pair in flow_pairs else 0.01
            output.append(
                ActionPrediction(
                    edge_id=edge.candidate_id,
                    action="follow" if follow >= 0.5 else "ignore",
                    action_probabilities={
                        "follow": follow,
                        "inspect": 0.0,
                        "stop": 0.0,
                        "ignore": 1.0 - follow,
                    },
                    rationale=(
                        f"{self.name} paper-derived adapter; dynamic dense peeling; "
                        "per-asset maximum-flow relation mapping"
                    ),
                )
            )
        return output


class DenseFlowPlusEdgeClassifier(DenseFlowEdgeClassifier):
    """DenseFlow+ adapter with the paper's Gini/TxSpike temporal extension."""

    name = "denseflow_plus"
    version = "0.1-paper-derived-adaptation"
    plus = True
