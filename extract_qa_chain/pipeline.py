from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from case_selection import discover_selected_cases
from chain_qa_version import chain_qa_version_at_least

from .config import ChainQAConfig, SUMMARIZED_DIR
from .index_client import EthIndexClient
from .io import read_json
from .schema import (
    ADDRESS_RE,
    CandidateSet,
    ChainQATask,
    EdgeLabel,
    ObservedEdge,
    SeedNode,
    normalize_address,
    normalize_tx_hash,
    short_id,
    stable_hash,
)


BLACK_ROLES = {
    "attacker_initial",
    "attacker_ethereum_wallet",
    "attacker_contract",
    "known_intermediary",
    "collector",
    "gas_funder",
    "victim",
    "victim_contract",
}
BOUNDARY_HINTS = ("cex", "exchange", "tornado", "mixer", "blender", "bridge")
INSPECT_HINTS = ("router", "swap", "dex", "bridge")


class CaseLoadError(RuntimeError):
    pass


def load_case(case_dir: str) -> dict[str, Any]:
    base = SUMMARIZED_DIR / case_dir
    summary_path = base / "summary.json"
    evidence_path = base / "chain_evidence.json"
    if not summary_path.exists() or not evidence_path.exists():
        raise CaseLoadError(f"missing summary.json or chain_evidence.json for {case_dir}")
    return {
        "case_dir": case_dir,
        "summary": read_json(summary_path),
        "evidence": read_json(evidence_path),
    }


def discover_cases() -> list[str]:
    if not SUMMARIZED_DIR.exists():
        return []
    return discover_selected_cases(
        SUMMARIZED_DIR,
        required_files=("summary.json", "chain_evidence.json"),
    )


def case_name(case: dict[str, Any]) -> str:
    evidence = case["evidence"]
    summary = case["summary"]
    return str(evidence.get("case_name") or summary.get("case_name") or case["case_dir"])


def case_id(case: dict[str, Any]) -> str:
    evidence = case["evidence"]
    return str(evidence.get("case_id") or case["case_dir"])


def extract_seed_nodes(evidence: dict[str, Any]) -> list[SeedNode]:
    seeds: dict[str, SeedNode] = {}
    for entity in evidence.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        address = normalize_address(entity.get("address") or entity.get("id"))
        if not address:
            continue
        role = str(entity.get("role") or "entity_address")
        if role not in BLACK_ROLES and not role.startswith("attacker"):
            continue
        seeds[address] = SeedNode(
            node=address,
            role=role,
            source="chain_evidence.entities",
            confidence=str(entity.get("confidence") or "U"),
            usable_for_tracing=True,
        )
    for edge in evidence.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        for key, role in (("src", "reported_edge_src"), ("dst", "reported_edge_dst")):
            address = normalize_address(edge.get(key))
            if address and address not in seeds:
                seeds[address] = SeedNode(
                    node=address,
                    role=role,
                    source=str(edge.get("id") or "chain_evidence.edges"),
                    confidence=str(edge.get("confidence") or "U"),
                    usable_for_tracing=True,
                )
    return list(seeds.values())


def block_window(evidence: dict[str, Any], padding: int) -> tuple[int | None, int | None]:
    blocks = []
    for tx in evidence.get("transaction_evidence") or []:
        if isinstance(tx, dict) and isinstance(tx.get("block_number"), int):
            blocks.append(tx["block_number"])
    if not blocks:
        return None, None
    return max(0, min(blocks) - padding), max(blocks) + padding


def raw_value(edge: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if edge.get(key) not in (None, "", [], {}):
            return edge.get(key)
    raw = edge.get("raw_parsed") or edge.get("raw") or {}
    for key in keys:
        if isinstance(raw, dict) and raw.get(key) not in (None, "", [], {}):
            return raw.get(key)
    return None


def edge_amount(edge: dict[str, Any]) -> str | float | int | None:
    value_eth = raw_value(edge, "value_eth")
    if value_eth is not None:
        return value_eth
    value = raw_value(edge, "amount", "value", "value_wei", "raw_token_amount")
    token_address = raw_value(edge, "token_address")
    if value is not None and not token_address:
        try:
            numeric = int(value)
            if numeric >= 10**12:
                return numeric / 10**18
        except (TypeError, ValueError):
            pass
    return str(value) if value is not None else None


def normalize_index_edge(
    raw_edge: dict[str, Any],
    *,
    case_id_: str,
    seed: str | None = None,
    hop: int | None = None,
) -> ObservedEdge | None:
    src = normalize_address(raw_value(raw_edge, "src", "src_address", "from_address"))
    dst = normalize_address(raw_value(raw_edge, "dst", "dst_address", "to_address", "neighbor_address"))
    if not src or not dst:
        return None
    tx_hash = normalize_tx_hash(raw_value(raw_edge, "tx_hash", "hash", "transaction_hash"))
    raw_id = raw_value(raw_edge, "edge_id", "id")
    edge_id = f"eth:{raw_id}" if raw_id is not None else f"eth:{stable_hash([tx_hash, src, dst, raw_edge])}"
    return ObservedEdge(
        edge_id=edge_id,
        case_id=case_id_,
        src=src,
        dst=dst,
        tx_hash=tx_hash,
        block_number=raw_value(raw_edge, "block_number"),
        timestamp=raw_value(raw_edge, "timestamp", "block_timestamp"),
        edge_type=raw_value(raw_edge, "edge_type", "raw_kind", "type"),
        asset=raw_value(raw_edge, "asset", "symbol", "token_symbol"),
        amount=edge_amount(raw_edge),
        token_address=normalize_address(raw_value(raw_edge, "token_address")),
        direction_from_seed="out" if seed and src == seed else ("in" if seed and dst == seed else None),
        hop_from_seed=hop,
        seed=seed,
        raw_index_edge_id=raw_id,
        raw=raw_edge,
    )


def extract_expand_edges(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("edges")
    nested = data.get("data")
    if rows is None and isinstance(nested, dict):
        rows = nested.get("edges")
    if rows is None and isinstance(nested, list):
        rows = nested
    if rows is None:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def hydrate_edges(client: EthIndexClient, edges: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    hydrated = []
    for edge in edges[:limit]:
        edge_id = edge.get("edge_id") or edge.get("id")
        if edge_id is not None:
            full = client.edge(edge_id)
            if full:
                traversal = {
                    key: edge[key]
                    for key in ("hop", "hop_from_seed", "seed")
                    if edge.get(key) is not None
                }
                merged = dict(edge)
                merged.update(full)
                merged.update(traversal)
                hydrated.append(merged)
                continue
        hydrated.append(edge)
    return hydrated


def build_subgraph(
    case: dict[str, Any],
    config: ChainQAConfig,
    client: EthIndexClient,
) -> tuple[list[SeedNode], list[ObservedEdge]]:
    evidence = case["evidence"]
    cid = case_id(case)
    seeds = extract_seed_nodes(evidence)
    seed_addresses = [seed.node for seed in seeds if seed.usable_for_tracing]
    block_min, block_max = block_window(evidence, config.block_padding)
    raw_edges: list[dict[str, Any]] = []
    if seed_addresses:
        expanded = client.expand(
            seed_addresses,
            k=config.k_hop,
            direction=config.direction,
            max_nodes=config.max_nodes,
            max_edges=config.max_edges,
            block_min=block_min,
            block_max=block_max,
        )
        unindexed_seeds = set(expanded.get("unindexed_seeds") or [])
        for seed in seeds:
            if seed.node in unindexed_seeds:
                seed.usable_for_tracing = False
        raw_edges.extend(extract_expand_edges(expanded))
    raw_edges = hydrate_edges(client, raw_edges, config.max_edges)
    observed: dict[str, ObservedEdge] = {}
    seed_set = set(seed_addresses)
    for raw_edge in raw_edges:
        seed = normalize_address(raw_edge.get("seed")) or next(
            (
                node
                for node in seed_set
                if node
                in {
                    normalize_address(raw_value(raw_edge, "src", "src_address", "from_address")),
                    normalize_address(raw_value(raw_edge, "dst", "dst_address", "to_address", "neighbor_address")),
                }
            ),
            None,
        )
        normalized = normalize_index_edge(raw_edge, case_id_=cid, seed=seed, hop=raw_edge.get("hop") or raw_edge.get("hop_from_seed"))
        if normalized:
            observed[normalized.edge_id] = normalized
    for tx in evidence.get("transaction_evidence") or []:
        if not isinstance(tx, dict):
            continue
        for raw_edge in tx.get("index_edges") or [tx]:
            if isinstance(raw_edge, dict):
                tx_src = normalize_address(
                    raw_value(raw_edge, "src", "src_address", "from_address")
                )
                tx_dst = normalize_address(
                    raw_value(raw_edge, "dst", "dst_address", "to_address")
                )
                reported_hop = raw_edge.get("hop") or raw_edge.get("hop_from_seed")
                tx_seed = next(
                    (node for node in seed_set if node in {tx_src, tx_dst}), None
                )
                if reported_hop is None and tx_seed:
                    reported_hop = 1
                normalized = normalize_index_edge(
                    raw_edge,
                    case_id_=cid,
                    seed=tx_seed,
                    hop=reported_hop,
                )
                if normalized:
                    observed.setdefault(normalized.edge_id, normalized)
    return seeds, list(observed.values())


def reported_edges_by_address(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    entity_addresses: dict[str, list[str]] = {}
    for entity in evidence.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        addresses = []
        direct = normalize_address(entity.get("address"))
        if direct:
            addresses.append(direct)
        for mapping in entity.get("addresses") or []:
            if not isinstance(mapping, dict) or str(mapping.get("chain") or "").lower() != "ethereum":
                continue
            address = normalize_address(mapping.get("address"))
            if address:
                addresses.append(address)
        entity_addresses[str(entity.get("id"))] = sorted(set(addresses))

    out = []
    for edge in evidence.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        src_ref = str(edge.get("src") or "")
        dst_ref = str(edge.get("dst") or "")
        src_addresses = entity_addresses.get(src_ref) or [normalize_address(src_ref)]
        dst_addresses = entity_addresses.get(dst_ref) or [normalize_address(dst_ref)]
        for src in [value for value in src_addresses if value] or [None]:
            for dst in [value for value in dst_addresses if value] or [None]:
                if not src and not dst:
                    continue
                item = dict(edge)
                item["_src_address"] = src
                item["_dst_address"] = dst
                item["_resolved_from_entity_addresses"] = bool(
                    src_ref in entity_addresses or dst_ref in entity_addresses
                )
                out.append(item)
    return out


def transaction_positive_pairs(evidence: dict[str, Any]) -> set[tuple[str, str]]:
    pairs = set()
    for tx in evidence.get("transaction_evidence") or []:
        if not isinstance(tx, dict):
            continue
        label = (tx.get("rule_labels") or {}).get("primary_label") or ""
        if "attacker" not in str(label) and "outflow" not in str(label) and "transfer_from" not in str(label):
            continue
        src = normalize_address(tx.get("src_address"))
        dst = normalize_address(tx.get("dst_address"))
        if src and dst:
            pairs.add((src, dst))
    return pairs


def classify_action(edge: ObservedEdge, label: str, evidence: dict[str, Any]) -> str:
    if label in {"negative", "hard_negative"}:
        return "ignore"
    address_entities: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in evidence.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        for mapping in entity.get("addresses") or []:
            if not isinstance(mapping, dict) or str(mapping.get("chain") or "").lower() != "ethereum":
                continue
            address = normalize_address(mapping.get("address"))
            if address:
                address_entities[address].append({"entity": entity, "mapping": mapping})

    destination_matches = address_entities.get(edge.dst, [])
    if destination_matches:
        entity_ids = {str(item["entity"].get("id")) for item in destination_matches}
        if len(entity_ids) > 1:
            return "inspect"
        for item in destination_matches:
            mapping = item["mapping"]
            address_role = str(mapping.get("address_role") or "").lower()
            status = str(mapping.get("attribution_status") or "").lower()
            tx_hashes = {normalize_tx_hash(value) for value in mapping.get("matched_tx_hashes") or []}
            index_edge_ids = {str(value) for value in mapping.get("matched_index_edge_ids") or []}
            observed_reported_match = (
                status != "reported"
                or (edge.tx_hash and edge.tx_hash in tx_hashes)
                or edge.edge_id in index_edge_ids
            )
            if address_role in {
                "cex_deposit_address",
                "cex_terminal_address",
                "mixer_deposit_contract",
            } and observed_reported_match:
                return "stop"
            if address_role in {
                "bridge_contract",
                "dex_router",
                "cex_labeled_address",
                "protocol_contract",
                "service_address",
            }:
                return "inspect"

    raw = edge.raw or {}
    raw_parsed = raw.get("raw_parsed") if isinstance(raw, dict) else {}
    text = " ".join(
        str(v or "").lower()
        for v in [
            edge.dst,
            edge.asset,
            edge.edge_type,
            raw.get("entity") if isinstance(raw, dict) else None,
            raw.get("label") if isinstance(raw, dict) else None,
            raw.get("pattern") if isinstance(raw, dict) else None,
            raw_parsed.get("method") if isinstance(raw_parsed, dict) else None,
        ]
    )
    if any(hint in text for hint in ("tornado", "mixer", "cex", "exchange")):
        return "stop"
    if any(hint in text for hint in INSPECT_HINTS):
        return "inspect"
    if label in {"positive", "reported_positive"}:
        return "follow"
    return "ignore"


def label_subgraph(
    case: dict[str, Any],
    seeds: list[SeedNode],
    edges: list[ObservedEdge],
    neg_ratio: int,
) -> list[EdgeLabel]:
    evidence = case["evidence"]
    cid = case_id(case)
    seed_nodes = {seed.node for seed in seeds}
    positive_pairs = transaction_positive_pairs(evidence)
    reported = reported_edges_by_address(evidence)
    reported_pairs = {
        (edge["_src_address"], edge["_dst_address"]): edge
        for edge in reported
        if edge.get("_src_address") and edge.get("_dst_address")
    }
    labels: list[EdgeLabel] = []
    positives = 0
    for edge in edges:
        pair = (edge.src, edge.dst)
        reverse_pair = (edge.dst, edge.src)
        reported_edge = reported_pairs.get(pair) or reported_pairs.get(reverse_pair)
        if pair in positive_pairs or reported_edge:
            label = "positive"
            semantics = "chain_direct"
            confidence = "B"
            stage = (reported_edge or {}).get("stage")
            reported_id = (reported_edge or {}).get("id")
            positives += 1
        elif edge.src in seed_nodes or edge.dst in seed_nodes:
            label = "hard_negative"
            semantics = "chain_direct"
            confidence = "C"
            stage = None
            reported_id = None
        else:
            label = "negative"
            semantics = "chain_direct"
            confidence = "C"
            stage = None
            reported_id = None
        action = classify_action(edge, label, evidence)
        if action in {"stop", "inspect"} and label not in {"positive", "reported_positive"}:
            label = "boundary" if action == "stop" else "hard_negative"
        labels.append(
            EdgeLabel(
                edge_id=edge.edge_id,
                case_id=cid,
                label=label,
                action_label=action,
                label_semantics=semantics,
                confidence=confidence,
                stage=stage,
                path_id=f"{cid}:observed_path",
                reported_edge_id=reported_id,
                negative_sample_type="same_seed_or_subgraph_neighbor" if label in {"negative", "hard_negative"} else None,
                usable_for_eval=label != "uncertain",
                usable_for_qa=True,
            )
        )
    for reported_edge in reported:
        if reported_edge.get("verification_status") == "chain_verified":
            continue
        labels.append(
            EdgeLabel(
                edge_id=f"reported:{reported_edge.get('id') or stable_hash(reported_edge)}",
                case_id=cid,
                label="reported_positive",
                action_label="inspect",
                label_semantics="reported_direct",
                confidence=str(reported_edge.get("confidence") or "U"),
                stage=reported_edge.get("stage"),
                path_id=f"{cid}:reported_path",
                reported_edge_id=reported_edge.get("id"),
                usable_for_eval=False,
                usable_for_qa=True,
            )
        )
    return labels


def enrich_action_outgoing(
    case: dict[str, Any],
    edges: list[ObservedEdge],
    labels: list[EdgeLabel],
    config: ChainQAConfig,
    client: EthIndexClient,
) -> list[ObservedEdge]:
    """Add real outgoing neighbors for nodes that already have an actionable edge."""
    edge_map = {edge.edge_id: edge for edge in edges}
    action_node_hops: dict[str, list[int]] = defaultdict(list)
    for label in labels:
        edge = edge_map.get(label.edge_id)
        if edge is None or label.action_label not in {"follow", "inspect", "stop"}:
            continue
        if isinstance(edge.hop_from_seed, int):
            action_node_hops[edge.src].append(edge.hop_from_seed)
        else:
            action_node_hops.setdefault(edge.src, [])
    block_min, block_max = block_window(case["evidence"], config.block_padding)
    for node in sorted(action_node_hops):
        source_outgoing_hop = min(action_node_hops[node], default=None)
        raw_neighbors = client.neighbors(
            node,
            direction="out",
            limit=config.neighbor_limit,
            block_min=block_min,
            block_max=block_max,
        )
        for raw_edge in raw_neighbors:
            normalized = normalize_index_edge(
                raw_edge,
                case_id_=case_id(case),
                seed=node,
                hop=(
                    raw_edge.get("hop")
                    or raw_edge.get("hop_from_seed")
                    or source_outgoing_hop
                ),
            )
            if normalized and normalized.src == node:
                edge_map.setdefault(normalized.edge_id, normalized)
    return list(edge_map.values())


def build_candidate_sets(
    case: dict[str, Any],
    edges: list[ObservedEdge],
    labels: list[EdgeLabel],
    *,
    max_candidates: int,
    neg_ratio: int,
) -> list[CandidateSet]:
    cid = case_id(case)
    by_label = {label.edge_id: label for label in labels}
    outgoing: dict[str, dict[str, list[ObservedEdge]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for edge in edges:
        outgoing[edge.src][edge.dst].append(edge)
    candidate_sets = []
    for node, destination_groups in sorted(outgoing.items()):
        units: list[dict[str, Any]] = []
        for dst, group_edges in sorted(destination_groups.items()):
            group_edges = sorted(
                group_edges,
                key=lambda edge: stable_hash([cid, "candidate-raw-edge", edge.edge_id]),
            )
            group_labels = [
                by_label[edge.edge_id]
                for edge in group_edges
                if edge.edge_id in by_label and by_label[edge.edge_id].usable_for_qa
            ]
            actions = {label.action_label for label in group_labels}
            if not group_labels or len(actions) != 1:
                continue
            action = next(iter(actions))
            candidate_id = f"pair:{stable_hash([cid, node, dst])}"
            tx_hashes = sorted({edge.tx_hash for edge in group_edges if edge.tx_hash})
            summary = {
                "edge_id": candidate_id,
                "candidate_id": candidate_id,
                "aggregation": "node_pair_over_raw_txs",
                "src": node,
                "dst": dst,
                "edge_count": len(group_edges),
                "tx_count": len(tx_hashes),
                "sample_edge_ids": [edge.edge_id for edge in group_edges[:12]],
                "sample_tx_hashes": tx_hashes[:12],
                "assets": sorted({str(edge.asset) for edge in group_edges if edge.asset}),
                "edge_types": sorted(
                    {str(edge.edge_type) for edge in group_edges if edge.edge_type is not None}
                ),
                "token_addresses": sorted(
                    {edge.token_address for edge in group_edges if edge.token_address}
                )[:12],
                "hop_from_seed": min(
                    (edge.hop_from_seed for edge in group_edges if edge.hop_from_seed is not None),
                    default=None,
                ),
            }
            units.append(
                {
                    "candidate_id": candidate_id,
                    "action": action,
                    "raw_edge_ids": [edge.edge_id for edge in group_edges],
                    "summary": summary,
                }
            )
        by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for unit in units:
            by_action[unit["action"]].append(unit)
        for action in by_action:
            by_action[action].sort(
                key=lambda unit: stable_hash(
                    [cid, "candidate-unit", node, action, unit["candidate_id"]]
                )
            )
        if not any(by_action.get(action) for action in ("follow", "stop", "inspect")):
            continue
        follow_cap = max(
            1,
            min(len(by_action.get("follow", [])), max_candidates // max(2, neg_ratio + 1)),
        )
        if not by_action.get("ignore") and len(by_action.get("follow", [])) > 1:
            follow_cap = max(2, follow_cap)
        chosen: list[dict[str, Any]] = []
        for group in (
            by_action.get("follow", [])[:follow_cap],
            by_action.get("stop", [])[:1],
            by_action.get("inspect", [])[:1],
        ):
            chosen.extend(group)
        negative_target = min(
            max_candidates - len(chosen),
            max(1, neg_ratio * max(1, len(by_action.get("follow", [])[:follow_cap]))),
        )
        chosen.extend(by_action.get("ignore", [])[:negative_target])
        chosen = chosen[:max_candidates]
        if len(chosen) < 2:
            continue
        chosen.sort(
            key=lambda unit: stable_hash(
                [cid, "candidate-order", node, unit["candidate_id"]]
            )
        )
        candidate_ids = [unit["candidate_id"] for unit in chosen]
        action_labels = {
            unit["candidate_id"]: unit["action"]
            for unit in chosen
        }
        candidate_sets.append(
            CandidateSet(
                state_id=f"{cid}:state:{stable_hash([node, candidate_ids])}",
                case_id=cid,
                current_node=node,
                path_prefix=[],
                candidate_edges=candidate_ids,
                candidate_edge_groups={
                    unit["candidate_id"]: unit["raw_edge_ids"] for unit in chosen
                },
                candidate_summaries=[unit["summary"] for unit in chosen],
                correct_next_edges=[
                    candidate_id
                    for candidate_id in candidate_ids
                    if action_labels[candidate_id] == "follow"
                ],
                boundary_edges=[
                    candidate_id
                    for candidate_id in candidate_ids
                    if action_labels[candidate_id] == "stop"
                ],
                inspect_edges=[
                    candidate_id
                    for candidate_id in candidate_ids
                    if action_labels[candidate_id] == "inspect"
                ],
                ignore_edges=[
                    candidate_id
                    for candidate_id in candidate_ids
                    if action_labels[candidate_id] == "ignore"
                ],
                action_labels=action_labels,
            )
        )
    return candidate_sets


def edge_summary(edge: ObservedEdge) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "src": edge.src,
        "dst": edge.dst,
        "tx_hash": edge.tx_hash,
        "block_number": edge.block_number,
        "timestamp": edge.timestamp,
        "asset": edge.asset,
        "amount": edge.amount,
        "edge_type": edge.edge_type,
        "hop_from_seed": edge.hop_from_seed,
    }


def answer_format(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


def generate_tasks(
    case: dict[str, Any],
    edges: list[ObservedEdge],
    labels: list[EdgeLabel],
    candidates: list[CandidateSet],
    dataset_version: str,
    *,
    agent_profile: str = "tool_grounded",
) -> list[ChainQATask]:
    if agent_profile not in {"tool_grounded", "context_only"}:
        raise ValueError(f"unsupported tracing agent profile: {agent_profile}")
    cid = case_id(case)
    cname = case_name(case)
    edge_map = {edge.edge_id: edge for edge in edges}
    label_map = {label.edge_id: label for label in labels}
    tasks: list[ChainQATask] = []
    sequential_policy = chain_qa_version_at_least(dataset_version, 3, 1)

    def candidate_edge_groups(context: dict[str, Any]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}

        def add_groups(value: Any) -> None:
            if not isinstance(value, dict):
                return
            for candidate_id, raw_ids in value.items():
                normalized = [str(raw_id) for raw_id in raw_ids or [] if str(raw_id)]
                if normalized:
                    groups.setdefault(str(candidate_id), [])
                    groups[str(candidate_id)].extend(normalized)

        add_groups(context.get("candidate_edge_groups"))
        for state in context.get("states") or []:
            if isinstance(state, dict):
                add_groups(state.get("candidate_edge_groups"))
        return {
            candidate_id: list(dict.fromkeys(raw_ids))
            for candidate_id, raw_ids in groups.items()
        }

    def configure_profile(
        task_type: str, context: dict[str, Any]
    ) -> tuple[list[str], list[str], str, str]:
        if agent_profile == "context_only":
            return [], [], "disabled", "graph_context"
        groups = candidate_edge_groups(context)
        if task_type in {
            "edge_action_classification",
            "path_completion",
            "subgraph_noise_filtering",
        }:
            if not groups:
                raise ValueError(
                    f"tool-grounded {task_type} task has no candidate raw-edge groups"
                )
            max_tool_calls = len(groups)
            context["tool_query_contract"] = {
                "tool": "eth_edge",
                "with_raw": True,
                "required_evidence_per_candidate": 1,
                "candidate_count": len(groups),
                "max_tool_calls": max_tool_calls,
            }
            if isinstance(context.get("rollout_budget"), dict):
                context["rollout_budget"]["max_tool_calls"] = max_tool_calls
            return ["eth_edge"], ["eth_edge"], "required", "candidate_tool_query"
        return [], [], "disabled", "graph_context"

    def public_state(cand: CandidateSet) -> dict[str, Any]:
        return {
            "state_id": cand.state_id,
            "current_node": cand.current_node,
            "candidate_edges": list(cand.candidate_edges),
            "candidate_edge_groups": dict(cand.candidate_edge_groups),
        }

    sequential_response_schema = {
        "seed_predictions": [
            {"node_id": "candidate seed", "probability": "0..1"}
        ],
        "selected_seed_nodes": ["candidate seed"],
        "teacher_forced": {
            "steps": [
                {
                    "state_id": "public state id",
                    "current_node": "state current node",
                    "predictions": "same four-action prediction schema as edge_action_classification",
                    "selected_follow_edges": ["candidate id"],
                }
            ]
        },
        "free_rollout": {
            "steps": "same step schema, restricted to states reached by prior selected follow edges",
            "paths": [
                {
                    "edge_ids": ["candidate id"],
                    "node_ids": ["address"],
                    "probability": "0..1",
                    "terminal_reason": "stop|no_follow|budget|dead_end",
                }
            ],
        },
        "oracle_seed": {
            "steps": "step schema evaluated from hidden oracle seeds",
            "paths": "path schema",
        },
        "predicted_seed": {
            "steps": "step schema evaluated from selected_seed_nodes",
            "paths": "path schema",
        },
        "no_valid_seed": "boolean",
    }
    snf_state_policy_response_schema = {
        "seed_predictions": [
            {"node_id": "candidate seed", "probability": "0..1"}
        ],
        "selected_seed_nodes": ["candidate seed"],
        "state_policy": {
            "steps": [
                {
                    "state_id": "every public state id exactly once",
                    "current_node": "state current node",
                    "predictions": (
                        "all candidates with the same four-action probability schema "
                        "as edge_action_classification"
                    ),
                    "selected_follow_edges": ["candidate id"],
                }
            ]
        },
        "no_valid_seed": "boolean",
        "evaluator_derives": ["oracle_seed", "predicted_seed"],
    }

    def add_task(
        task_type: str,
        difficulty: str,
        question: str,
        answer: str,
        answer_value: Any,
        graph_context: dict[str, Any],
        edge_ids: list[str],
        metrics: list[str],
        link_semantics: str,
        question_type: str | None = None,
        *,
        summaries_override: list[dict[str, Any]] | None = None,
        evaluation_extra: dict[str, Any] | None = None,
    ) -> None:
        required_tools, allowed_tools, tool_requirement, public_context_profile = (
            configure_profile(task_type, graph_context)
        )
        summaries = summaries_override or [
            edge_summary(edge_map[edge_id]) for edge_id in edge_ids if edge_id in edge_map
        ]
        label_payload = {
            "positive_edges": [edge_id for edge_id in edge_ids if label_map.get(edge_id) and label_map[edge_id].label == "positive"],
            "negative_edges": [edge_id for edge_id in edge_ids if label_map.get(edge_id) and label_map[edge_id].label in {"negative", "hard_negative"}],
            "boundary_edges": [edge_id for edge_id in edge_ids if label_map.get(edge_id) and label_map[edge_id].action_label == "stop"],
            "action_labels": {edge_id: label_map[edge_id].action_label for edge_id in edge_ids if edge_id in label_map},
        }
        tasks.append(
            ChainQATask(
                id=f"{cid}:{task_type}:{len(tasks)+1:05d}",
                dataset_version=dataset_version,
                case_id=cid,
                case_dir=case["case_dir"],
                case_name=cname,
                task_type=task_type,
                difficulty=difficulty,
                question=question,
                answer=answer,
                answer_value=answer_value,
                answer_format=answer_format(answer_value),
                graph_context=graph_context,
                edge_summaries=summaries,
                labels=label_payload,
                evidence={
                    "chain_edges": [edge_id for edge_id in edge_ids if edge_id in edge_map],
                    "reported_edges": sorted({label_map[edge_id].reported_edge_id for edge_id in edge_ids if edge_id in label_map and label_map[edge_id].reported_edge_id}),
                    "verification_status": link_semantics,
                },
                evaluation={
                    "metric": metrics,
                    "primary_metric": metrics[0],
                    **(evaluation_extra or {}),
                },
                link_semantics=link_semantics,
                question_type=question_type,
                agent_profile=agent_profile,
                required_tools=required_tools,
                allowed_tools=allowed_tools,
                tool_requirement=tool_requirement,
                public_context_profile=public_context_profile,
            )
        )

    for cand in candidates:
        candidate_ids = cand.candidate_edges
        summary_map = {
            str(summary["candidate_id"]): summary for summary in cand.candidate_summaries
        }
        sample_raw_edge_ids = list(
            dict.fromkeys(
                edge_id
                for candidate_id in candidate_ids
                for edge_id in summary_map[candidate_id].get("sample_edge_ids") or []
            )
        )
        public_edge_groups = {
            candidate_id: list(summary_map[candidate_id].get("sample_edge_ids") or [])
            for candidate_id in candidate_ids
        }
        answer_value = {
            "candidate_actions": cand.action_labels,
            "correct_follow_edges": cand.correct_next_edges,
            "none_of_above": not cand.correct_next_edges,
            "ranking_key": "action_probabilities.follow",
        }
        public_context = {
            "state_id": cand.state_id,
            "current_node": cand.current_node,
            "path_prefix": cand.path_prefix,
            "candidate_edges": candidate_ids,
            "candidate_edge_groups": public_edge_groups,
            "candidate_aggregation": "node_pair_over_raw_txs",
            "raw_evidence_sampling": "up_to_12_edges_and_12_txs_per_candidate",
            "allowed_actions": ["follow", "inspect", "stop", "ignore"],
            "sample_source": "local_outgoing_candidate_set",
            "top_k": min(3, len(candidate_ids)),
            "follow_threshold": 0.5,
            "response_schema": {
                "predictions": [
                    {
                        "edge_id": "candidate id",
                        "action": "follow|inspect|stop|ignore",
                        "action_probabilities": {
                            "follow": "0..1",
                            "inspect": "0..1",
                            "stop": "0..1",
                            "ignore": "0..1",
                        },
                    }
                ],
                "selected_follow_edges": ["candidate id"],
            },
        }
        add_task(
            "edge_action_classification",
            "hard" if cand.boundary_edges or cand.inspect_edges else "medium",
            (
                f"从 {cname} 的当前节点 {short_id(cand.current_node)} 出发，请对所有候选出边分别给出 "
                "follow、inspect、stop 或 ignore，并给出四类概率。按 follow 概率排序决定下一跳；"
                "如果没有边应 follow，selected_follow_edges 必须为空。仅输出符合 response_schema 的 JSON。"
            ),
            json_dumps_compact(answer_value),
            answer_value,
            public_context,
            sample_raw_edge_ids,
            [
                "hit@k",
                "action_macro_f1",
                "over_expansion_rate",
                "none_of_above_accuracy",
                "mrr",
                "brier_score",
            ],
            "chain_direct",
            "<s,*,*>",
            summaries_override=cand.candidate_summaries,
            evaluation_extra={
                "top_k": min(3, len(candidate_ids)),
                "top_k_values": [
                    value for value in (1, 3, 5) if value <= len(candidate_ids)
                ],
                "follow_threshold": 0.5,
                "scorer": "deterministic_edge_action_v1",
            },
        )

        if not sequential_policy:
            # v3_0 keeps the original single-state stress task for replay.
            add_task(
                "subgraph_noise_filtering",
                "hard",
                f"给定 {cname} 中 {short_id(cand.current_node)} 的候选出边集合，哪些边应 follow、inspect、stop 或 ignore？",
                f"action_labels={cand.action_labels}",
                cand.action_labels,
                {
                    "state_id": cand.state_id,
                    "current_node": cand.current_node,
                    "candidate_edges": candidate_ids,
                    "candidate_edge_groups": public_edge_groups,
                    "raw_evidence_sampling": "up_to_12_edges_and_12_txs_per_candidate",
                    "sample_source": "local_outgoing_candidate_set",
                },
                sample_raw_edge_ids,
                ["macro_f1", "over_expansion_rate"],
                "chain_direct",
                "<s,*,*>",
                summaries_override=cand.candidate_summaries,
            )

    hard_negative_edges = [
        edge_map[label.edge_id]
        for label in labels
        if label.action_label == "ignore" and label.edge_id in edge_map
    ]
    ignored_by_src: dict[str, list[ObservedEdge]] = defaultdict(list)
    for label in labels:
        if label.action_label != "ignore" or label.edge_id not in edge_map:
            continue
        edge = edge_map.get(label.edge_id)
        if edge:
            ignored_by_src[edge.src].append(edge)
    decoy_count = 0
    decoy_used_edges: set[str] = set()
    for node, node_edges in sorted(
        ignored_by_src.items(),
        key=lambda item: stable_hash([cid, "decoy", item[0]]),
    ):
        if decoy_count >= 24:
            break
        selected = sorted(
            node_edges,
            key=lambda edge: stable_hash([cid, "decoy-edge", node, edge.edge_id]),
        )[: min(10, len(node_edges))]
        if len(selected) < 3:
            continue
        edge_ids = [edge.edge_id for edge in selected]
        decoy_used_edges.update(edge_ids)
        action_labels = {edge_id: "ignore" for edge_id in edge_ids}
        graph_context = {
            "candidate_edges": edge_ids,
            "sample_source": "negative_only_khop_ego_subgraph",
        }
        if sequential_policy:
            state_id = f"{cid}:negative-state:{stable_hash([node, edge_ids])}"
            graph_context.update(
                {
                    "sequential_policy": True,
                    "trusted_seed": False,
                    "candidate_seed_nodes": [node],
                    "states": [
                        {
                            "state_id": state_id,
                            "current_node": node,
                            "candidate_edges": edge_ids,
                            "candidate_edge_groups": {
                                edge_id: [edge_id] for edge_id in edge_ids
                            },
                        }
                    ],
                    "evaluation_modes": ["oracle_seed", "predicted_seed"],
                    "agent_output_contract": "snf_state_policy_v2",
                    "rollout_derivation": "evaluator_seed_projection_v1",
                    "rollout_budget": {
                        "max_depth": 1,
                        "beam_width": 1,
                        "max_node_expansions": 1,
                        "max_inspected_edges": len(edge_ids),
                        "max_tool_calls": 0,
                    },
                    "follow_threshold": 0.5,
                    "response_schema": snf_state_policy_response_schema,
                }
            )
            answer_value = {
                "gold_seed_nodes": [],
                "no_valid_seed": True,
                "state_action_labels": {state_id: action_labels},
                "gold_follow_edges": [],
                "gold_paths": [],
            }
        else:
            graph_context["current_node"] = node
            answer_value = action_labels
        add_task(
            "subgraph_noise_filtering",
            "hard",
            f"下面是 {cname} k-hop 子图中随机抽取的非异常候选片段，是否有边应被纳入异常资金链路？若没有，请全部标为 ignore。",
            f"没有应 follow 的边；这些候选边均应 ignore: {', '.join(edge_ids)}。",
            answer_value,
            graph_context,
            edge_ids,
            (
                ["predicted_seed_end_to_end_f1", "seed_mrr", "none_of_above_accuracy", "brier_score", "ece"]
                if sequential_policy
                else ["macro_f1", "over_expansion_rate", "none_of_above_accuracy"]
            ),
            "chain_direct",
            "<s,*,*>",
        )
        decoy_count += 1
    remaining_decoys = [
        edge
        for edge in sorted(
            hard_negative_edges,
            key=lambda item: stable_hash([cid, "global-decoy", item.edge_id]),
        )
        if edge.edge_id not in decoy_used_edges
    ]
    for offset in range(0, len(remaining_decoys), 10):
        if decoy_count >= 12:
            break
        selected = remaining_decoys[offset : offset + 10]
        if len(selected) < 3:
            break
        edge_ids = [edge.edge_id for edge in selected]
        action_labels = {edge_id: "ignore" for edge_id in edge_ids}
        decoy_node = f"decoy:negative_subgraph:{decoy_count + 1}"
        graph_context = {
            "candidate_edges": edge_ids,
            "sample_source": "negative_only_khop_random_subgraph",
        }
        if sequential_policy:
            state_id = f"{cid}:negative-state:{stable_hash([decoy_node, edge_ids])}"
            graph_context.update(
                {
                    "sequential_policy": True,
                    "trusted_seed": False,
                    "candidate_seed_nodes": [decoy_node],
                    "states": [
                        {
                            "state_id": state_id,
                            "current_node": decoy_node,
                            "candidate_edges": edge_ids,
                            "candidate_edge_groups": {
                                edge_id: [edge_id] for edge_id in edge_ids
                            },
                        }
                    ],
                    "evaluation_modes": ["oracle_seed", "predicted_seed"],
                    "agent_output_contract": "snf_state_policy_v2",
                    "rollout_derivation": "evaluator_seed_projection_v1",
                    "rollout_budget": {
                        "max_depth": 1,
                        "beam_width": 1,
                        "max_node_expansions": 1,
                        "max_inspected_edges": len(edge_ids),
                        "max_tool_calls": 0,
                    },
                    "follow_threshold": 0.5,
                    "response_schema": snf_state_policy_response_schema,
                }
            )
            answer_value = {
                "gold_seed_nodes": [],
                "no_valid_seed": True,
                "state_action_labels": {state_id: action_labels},
                "gold_follow_edges": [],
                "gold_paths": [],
            }
        else:
            graph_context["current_node"] = decoy_node
            answer_value = action_labels
        add_task(
            "subgraph_noise_filtering",
            "hard",
            f"下面是 {cname} k-hop 子图中随机抽取的不相干负样本边集合，是否有边应被纳入异常资金链路？若没有，请全部标为 ignore。",
            f"没有应 follow 的边；这些候选边均应 ignore: {', '.join(edge_ids)}。",
            answer_value,
            graph_context,
            edge_ids,
            (
                ["predicted_seed_end_to_end_f1", "seed_mrr", "none_of_above_accuracy", "brier_score", "ece"]
                if sequential_policy
                else ["macro_f1", "over_expansion_rate", "none_of_above_accuracy"]
            ),
            "chain_direct",
            "<s,*,*>",
        )
        decoy_count += 1

    if sequential_policy and candidates:
        candidate_by_node = {cand.current_node: cand for cand in candidates}
        summary_by_candidate = {
            str(summary["candidate_id"]): summary
            for cand in candidates
            for summary in cand.candidate_summaries
        }

        def reachable_rollout(start: CandidateSet, max_depth: int = 3) -> tuple[list[CandidateSet], list[list[str]]]:
            states: dict[str, CandidateSet] = {}
            gold_paths: list[list[str]] = []

            def visit(cand: CandidateSet, path: list[str], depth: int, seen: set[str]) -> None:
                states[cand.state_id] = cand
                follow_edges = sorted(cand.correct_next_edges)
                if not follow_edges:
                    if path:
                        gold_paths.append(path)
                    return
                for candidate_id in follow_edges:
                    next_path = path + [candidate_id]
                    summary = summary_by_candidate.get(candidate_id) or {}
                    dst = str(summary.get("dst") or "")
                    next_state = candidate_by_node.get(dst)
                    if depth >= max_depth or next_state is None or next_state.state_id in seen:
                        gold_paths.append(next_path)
                        continue
                    visit(next_state, next_path, depth + 1, seen | {next_state.state_id})

            visit(start, [], 1, {start.state_id})
            return list(states.values()), gold_paths

        made_sequential_paths = 0
        for start in sorted(candidates, key=lambda item: stable_hash([cid, "sequential-pc", item.state_id])):
            rollout_states, gold_paths = reachable_rollout(start)
            if len(rollout_states) < 2 or not any(len(path) >= 2 for path in gold_paths):
                continue
            state_ids = {state.state_id for state in rollout_states}
            state_actions = {
                state.state_id: dict(state.action_labels) for state in rollout_states
            }
            all_candidate_ids = list(
                dict.fromkeys(
                    candidate_id
                    for state in rollout_states
                    for candidate_id in state.candidate_edges
                )
            )
            all_summaries = [
                summary_by_candidate[candidate_id]
                for candidate_id in all_candidate_ids
                if candidate_id in summary_by_candidate
            ]
            raw_ids = list(
                dict.fromkeys(
                    raw_id
                    for state in rollout_states
                    for raw_ids in state.candidate_edge_groups.values()
                    for raw_id in raw_ids
                )
            )
            answer_value = {
                "state_action_labels": state_actions,
                "gold_follow_edges": sorted(
                    {
                        edge_id
                        for actions in state_actions.values()
                        for edge_id, action in actions.items()
                        if action == "follow"
                    }
                ),
                "gold_paths": gold_paths,
                "teacher_forced_state_ids": sorted(state_ids),
                "terminal_nodes": sorted(
                    state.current_node
                    for state in rollout_states
                    if not state.correct_next_edges
                ),
            }
            add_task(
                "path_completion",
                "hard",
                f"从 {cname} 的 {short_id(start.current_node)} 开始，在统一边策略下完成多步路径；分别返回 teacher-forced 与 free-rollout 结果。",
                json_dumps_compact(answer_value),
                answer_value,
                {
                    "sequential_policy": True,
                    "trusted_seed": True,
                    "start_node": start.current_node,
                    "start_state_id": start.state_id,
                    "states": [public_state(state) for state in rollout_states],
                    "evaluation_modes": ["teacher_forced", "free_rollout"],
                    "rollout_budget": {
                        "max_depth": 3,
                        "beam_width": 3,
                        "max_node_expansions": len(rollout_states),
                        "max_inspected_edges": len(all_candidate_ids),
                        "max_tool_calls": 0,
                    },
                    "follow_threshold": 0.5,
                    "response_schema": sequential_response_schema,
                },
                raw_ids,
                [
                    "free_rollout_complete_path_success",
                    "free_rollout_f1",
                    "teacher_forced_next_step_accuracy",
                    "coverage_at_budget",
                    "over_expansion_rate",
                    "brier_score",
                    "ece",
                ],
                "chain_direct",
                "<s,p,*>",
                summaries_override=all_summaries,
                evaluation_extra={
                    "scorer": "deterministic_sequential_edge_policy_v1",
                    "judge_layer": "context_and_tool_process_v1",
                },
            )
            made_sequential_paths += 1
            if made_sequential_paths >= 24:
                break

        snf_states = sorted(
            candidates,
            key=lambda item: stable_hash([cid, "sequential-snf", item.state_id]),
        )[:4]
        if snf_states:
            state_actions = {state.state_id: dict(state.action_labels) for state in snf_states}
            gold_seed_nodes = sorted(
                state.current_node for state in snf_states if state.correct_next_edges
            )
            candidate_ids = list(
                dict.fromkeys(edge_id for state in snf_states for edge_id in state.candidate_edges)
            )
            summaries = [
                summary_by_candidate[candidate_id]
                for candidate_id in candidate_ids
                if candidate_id in summary_by_candidate
            ]
            raw_ids = list(
                dict.fromkeys(
                    raw_id
                    for state in snf_states
                    for raw_group in state.candidate_edge_groups.values()
                    for raw_id in raw_group
                )
            )
            answer_value = {
                "gold_seed_nodes": gold_seed_nodes,
                "no_valid_seed": not gold_seed_nodes,
                "state_action_labels": state_actions,
                "gold_follow_edges": sorted(
                    {
                        edge_id
                        for actions in state_actions.values()
                        for edge_id, action in actions.items()
                        if action == "follow"
                    }
                ),
                "gold_paths": [],
            }
            add_task(
                "subgraph_noise_filtering",
                "hard",
                f"在未提供可信 seed 的 {cname} 候选子图中，先定位 seed，再对每个公开状态给出一次统一边策略；oracle-seed 与 predicted-seed 结果由评测端派生。",
                json_dumps_compact(answer_value),
                answer_value,
                {
                    "sequential_policy": True,
                    "trusted_seed": False,
                    "candidate_seed_nodes": [state.current_node for state in snf_states],
                    "states": [public_state(state) for state in snf_states],
                    "evaluation_modes": ["oracle_seed", "predicted_seed"],
                    "agent_output_contract": "snf_state_policy_v2",
                    "rollout_derivation": "evaluator_seed_projection_v1",
                    "seed_top_k": min(3, len(snf_states)),
                    "rollout_budget": {
                        "max_depth": 3,
                        "beam_width": 3,
                        "max_node_expansions": len(snf_states),
                        "max_inspected_edges": len(candidate_ids),
                        "max_tool_calls": 0,
                    },
                    "follow_threshold": 0.5,
                    "response_schema": snf_state_policy_response_schema,
                },
                raw_ids,
                [
                    "predicted_seed_end_to_end_f1",
                    "oracle_seed_end_to_end_f1",
                    "seed_mrr",
                    "coverage_at_budget",
                    "over_expansion_rate",
                    "brier_score",
                    "ece",
                ],
                "chain_direct",
                "<s,*,*>",
                summaries_override=summaries,
                evaluation_extra={
                    "scorer": "deterministic_sequential_edge_policy_v1",
                    "judge_layer": "context_and_tool_process_v1",
                },
            )

    positive_edges = [
        edge_map[label.edge_id]
        for label in labels
        if label.label == "positive" and label.edge_id in edge_map
    ]
    positive_by_src: dict[str, list[ObservedEdge]] = defaultdict(list)
    for edge in positive_edges:
        positive_by_src[edge.src].append(edge)
    made_paths = 0
    if sequential_policy:
        return tasks
    for first in sorted(
        positive_edges,
        key=lambda edge: stable_hash([cid, "path", edge.edge_id]),
    ):
        if made_paths >= 48:
            break
        next_edges = positive_by_src.get(first.dst, [])
        if not next_edges:
            continue
        correct = [edge.edge_id for edge in next_edges[:3]]
        distractors = [
            edge.edge_id
            for edge in hard_negative_edges
            if edge.src == first.dst and edge.edge_id not in correct
        ][:6]
        if len(distractors) < 3:
            for edge in hard_negative_edges:
                if edge.edge_id not in correct and edge.edge_id not in distractors:
                    distractors.append(edge.edge_id)
                if len(distractors) >= 6:
                    break
        edge_ids = [first.edge_id] + correct + distractors
        value = {
            "path_prefix_edges": [first.edge_id],
            "missing_next_edges": correct,
            "intermediate_node": first.dst,
            "distractor_edges": distractors,
        }
        add_task(
            "path_completion",
            "hard",
            f"在 {cname} 的异常资金路径中，已知上一跳为 {first.edge_id}（{short_id(first.src)} -> {short_id(first.dst)}），下一跳应补全哪条或哪些 positive 边？",
            json_dumps_compact(value),
            value,
            {
                "path_prefix_edges": [first.edge_id],
                "current_node": first.dst,
                "candidate_edges": correct + distractors,
                "completion_mode": "missing_next_edge_after_prefix",
            },
            edge_ids,
            ["path_recall@k", "mrr"],
            "chain_direct",
            "<s,p,*>",
        )
        made_paths += 1
    return tasks


def json_dumps_compact(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)
