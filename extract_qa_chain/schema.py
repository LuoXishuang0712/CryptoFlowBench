from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{40}")
TX_RE = re.compile(r"0x[a-fA-F0-9]{64}")


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def normalize_address(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = ADDRESS_RE.search(value)
    if not match:
        return None
    return match.group(0).lower()


def normalize_tx_hash(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = TX_RE.search(value)
    if not match:
        return None
    return match.group(0).lower()


def short_id(value: Any) -> str:
    text = str(value or "")
    if len(text) <= 18:
        return text
    if text.startswith("0x"):
        return f"{text[:8]}...{text[-6:]}"
    return text[:32]


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


@dataclass
class SeedNode:
    node: str
    role: str
    source: str
    confidence: str = "U"
    usable_for_tracing: bool = True


@dataclass
class ObservedEdge:
    edge_id: str
    case_id: str
    src: str
    dst: str
    tx_hash: str | None = None
    block_number: int | None = None
    timestamp: int | str | None = None
    edge_type: str | int | None = None
    asset: str | None = None
    amount: str | float | int | None = None
    token_address: str | None = None
    direction_from_seed: str | None = None
    hop_from_seed: int | None = None
    seed: str | None = None
    raw_index_edge_id: str | int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeLabel:
    edge_id: str
    case_id: str
    label: str
    action_label: str
    label_semantics: str
    confidence: str = "U"
    stage: str | None = None
    path_id: str | None = None
    reported_edge_id: str | None = None
    negative_sample_type: str | None = None
    usable_for_eval: bool = True
    usable_for_qa: bool = True


@dataclass
class CandidateSet:
    state_id: str
    case_id: str
    current_node: str
    path_prefix: list[str]
    candidate_edges: list[str]
    candidate_edge_groups: dict[str, list[str]]
    candidate_summaries: list[dict[str, Any]]
    correct_next_edges: list[str]
    boundary_edges: list[str]
    inspect_edges: list[str]
    ignore_edges: list[str]
    action_labels: dict[str, str]


@dataclass
class ChainQATask:
    id: str
    dataset_version: str
    case_id: str
    case_dir: str
    case_name: str
    task_type: str
    difficulty: str
    question: str
    answer: str
    answer_value: Any
    answer_format: str
    graph_context: dict[str, Any]
    edge_summaries: list[dict[str, Any]]
    labels: dict[str, Any]
    evidence: dict[str, Any]
    evaluation: dict[str, Any]
    link_semantics: str
    question_type: str | None = None
    agent_profile: str = "context_only"
    required_tools: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    tool_requirement: str = "disabled"
    public_context_profile: str = "graph_context"
