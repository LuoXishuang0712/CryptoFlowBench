"""Convert report summaries into chain-link evidence annotations.

The upstream agents produce:
  cases/<case_dir>/case.json
  summarized/<case_dir>/summary.json

This script turns the narrative summary into a downstream-friendly
`chain_evidence.json` file that separates report-derived facts, address/entity
roles, transaction/flow edges, extraction skills, and negative constraints.

Usage:
    python chain_evidence.py 01-2022-Ronin-2022
    python chain_evidence.py all
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

from case_selection import discover_selected_cases
from storage import get_json_store


PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARIZED_DIR = PROJECT_ROOT / "summarized"
ETH_INDEX_URL = "http://127.0.0.1:8000"

ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{40}")
TX_RE = re.compile(r"0x[a-fA-F0-9]{64}")
SOLANA_ADDRESS_RE = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")


def read_json(path: Path) -> dict[str, Any]:
    data = get_json_store().load_path(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    get_json_store().save_path(
        path,
        data,
        extra_metadata={
            "document_type": "chain_evidence",
            "case_id": data.get("case_id"),
            "case_dir": data.get("case_dir") or path.parent.name,
            "generated_by": "chain_evidence.py",
        },
    )


def normalize_address(value: str) -> str:
    return value.lower()


def unique_preserve_order(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def find_addresses(text: str) -> list[str]:
    return unique_preserve_order([normalize_address(x) for x in ADDRESS_RE.findall(text or "")])


def find_txs(text: str) -> list[str]:
    return unique_preserve_order([x.lower() for x in TX_RE.findall(text or "")])


def is_evm_address(value: str) -> bool:
    return bool(ADDRESS_RE.fullmatch(value))


def is_solana_address(value: str) -> bool:
    return bool(SOLANA_ADDRESS_RE.fullmatch(value)) and not is_evm_address(value)


def normalize_entity_id(value: str) -> str:
    if is_evm_address(value):
        return normalize_address(value)
    return value


def slugify(value: str, *, fallback: str = "entity") -> str:
    parts = [p.lower() for p in re.split(r"[^A-Za-z0-9]+", value) if p]
    return "_".join(parts) or fallback


def entity_id_from_label(label: str, *, prefix: str = "entity") -> str:
    return f"{prefix}:{slugify(label)}"


def parse_date_to_unix_window(date_value: str | None, days: int = 2) -> tuple[int, int] | None:
    if not date_value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(date_value).replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    start = parsed - dt.timedelta(days=days)
    end = parsed + dt.timedelta(days=days + 1)
    return int(start.timestamp()), int(end.timestamp())


def parse_amount_number(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"[\d,.]+", str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_attacker_address_entries(attacker: dict[str, Any]) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    for item in attacker.get("addresses", []) or []:
        text = str(item)
        evm_matches = ADDRESS_RE.findall(text)
        for address in evm_matches:
            lower = normalize_address(address)
            parsed.append(
                {
                    "id": lower,
                    "address": lower,
                    "chain": "Ethereum",
                    "format": "evm",
                    "source_text": text,
                }
            )
        scrubbed = ADDRESS_RE.sub(" ", text)
        for address in SOLANA_ADDRESS_RE.findall(scrubbed):
            parsed.append(
                {
                    "id": address,
                    "address": address,
                    "chain": "Solana",
                    "format": "solana",
                    "source_text": text,
                }
            )
    return unique_preserve_order(parsed)


def compact_case_id(case_dir: str, case_name: str | None = None) -> str:
    """Make a stable id like ronin_2022 from 01-2022-Ronin-2022."""
    match = re.match(r"^\d+-(\d{4})-(.+)$", case_dir)
    if match:
        year, slug = match.groups()
        parts = [p for p in re.split(r"[^A-Za-z0-9]+", slug) if p]
        if parts and parts[-1] == year:
            parts = parts[:-1]
        base = "_".join(p.lower() for p in parts) or "case"
        return f"{base}_{year}"

    source = case_name or case_dir
    year_match = re.search(r"\b(20\d{2})\b", source)
    year = year_match.group(1) if year_match else "unknown"
    name = re.sub(r"\b20\d{2}\b", "", source)
    parts = [p.lower() for p in re.split(r"[^A-Za-z0-9]+", name) if p]
    return f"{'_'.join(parts) or 'case'}_{year}"


def incident_type(summary: dict[str, Any]) -> str:
    protocol = str(summary.get("protocol_type") or "").lower()
    vuln = str(summary.get("vulnerability_type") or "").lower()
    root = str(summary.get("root_cause") or "").lower()
    if "bridge" in protocol:
        return "bridge_exploit"
    if "validator" in vuln or "private key" in root:
        return "key_compromise"
    if "oracle" in vuln or "price" in root:
        return "oracle_or_market_manipulation"
    if "reentr" in vuln:
        return "smart_contract_exploit"
    return "security_incident"


def source_catalog(sources_used: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    sources: list[dict[str, Any]] = []
    entity_to_id: dict[str, str] = {}
    for idx, source in enumerate(sources_used, start=1):
        source_id = f"S{idx:02d}"
        entity = source.get("source_entity") or "unknown"
        grade = source.get("source_grade") or "?"
        sources.append(
            {
                "id": source_id,
                "entity": entity,
                "grade": grade,
                "url": source.get("source"),
                "finding": source.get("findings"),
            }
        )
        entity_to_id[str(entity).lower()] = source_id
    return sources, entity_to_id


def evidence_ids_for_keywords(
    sources: list[dict[str, Any]],
    keywords: list[str],
    grades: set[str] | None = None,
    limit: int = 5,
) -> list[str]:
    ids: list[str] = []
    lowered = [k.lower() for k in keywords if k]
    for source in sources:
        if grades and source.get("grade") not in grades:
            continue
        haystack = f"{source.get('entity','')} {source.get('finding','')}".lower()
        if any(k in haystack for k in lowered):
            ids.append(str(source["id"]))
        if len(ids) >= limit:
            break
    return ids


def confidence_from_evidence(sources: list[dict[str, Any]], evidence_ids: list[str]) -> str:
    grade_by_id = {source["id"]: source.get("grade") for source in sources}
    grades = {grade_by_id.get(eid) for eid in evidence_ids}
    if "A" in grades:
        return "A"
    if "B" in grades:
        return "B"
    if "C" in grades:
        return "C"
    if "D" in grades:
        return "D"
    return "U"


def scanner_urls(address: str, chains: list[str]) -> dict[str, str]:
    urls: dict[str, str] = {}
    if is_evm_address(address):
        lower = normalize_address(address)
        urls["etherscan"] = f"https://etherscan.io/address/{lower}"
        if "Ronin" in chains:
            urls["roninchain"] = f"https://app.roninchain.com/address/{lower}"
        urls["oklink"] = f"https://www.oklink.com/en/eth/address/{lower}"
    elif is_solana_address(address):
        urls["solscan"] = f"https://solscan.io/account/{address}"
        urls["solana_explorer"] = f"https://explorer.solana.com/address/{address}"
        urls["oklink"] = f"https://www.oklink.com/en/sol/address/{address}"
    return urls


def add_entity_once(entities: list[dict[str, Any]], entity: dict[str, Any]) -> None:
    existing_ids = {item.get("id") for item in entities}
    if entity.get("id") not in existing_ids:
        entities.append(entity)


def ensure_reference_entities(evidence: dict[str, Any]) -> list[str]:
    """Add explicit stub entities for every edge endpoint that is not yet defined."""
    entities = evidence.setdefault("entities", [])
    existing_ids = {entity.get("id") for entity in entities}
    added: list[str] = []
    for edge in evidence.get("edges", []):
        for field in ("src", "dst"):
            ref = edge.get(field)
            if not isinstance(ref, str) or not ref or ref in existing_ids:
                continue
            entity: dict[str, Any]
            if is_evm_address(ref):
                entity = {
                    "id": normalize_address(ref),
                    "type": "address",
                    "address": normalize_address(ref),
                    "chain": "Ethereum",
                    "role": "edge_endpoint",
                    "confidence": edge.get("confidence", "U"),
                    "evidence": edge.get("evidence", []),
                    "scanner_urls": scanner_urls(ref, evidence.get("chains", [])),
                    "inferred_from_edge": edge.get("id"),
                }
            elif ref.startswith("entity:"):
                entity = {
                    "id": ref,
                    "type": "entity",
                    "entity": ref.removeprefix("entity:").replace("_", " "),
                    "role": "edge_endpoint",
                    "confidence": edge.get("confidence", "U"),
                    "evidence": edge.get("evidence", []),
                    "inferred_from_edge": edge.get("id"),
                }
            else:
                entity = {
                    "id": ref,
                    "type": "reference",
                    "entity": ref,
                    "role": "edge_endpoint",
                    "confidence": edge.get("confidence", "U"),
                    "evidence": edge.get("evidence", []),
                    "inferred_from_edge": edge.get("id"),
                }
            entities.append(entity)
            existing_ids.add(ref)
            added.append(ref)
    return added


def validate_references(evidence: dict[str, Any]) -> list[str]:
    entity_ids = {entity.get("id") for entity in evidence.get("entities", [])}
    missing: list[str] = []
    for edge in evidence.get("edges", []):
        for field in ("src", "dst"):
            ref = edge.get(field)
            if isinstance(ref, str) and ref and ref not in entity_ids:
                missing.append(f"{edge.get('id')}:{field}:{ref}")
    return missing


def entity_address_role(entity: dict[str, Any], edge: dict[str, Any] | None = None) -> str:
    """Return a conservative address role without upgrading address ownership."""
    role = str(entity.get("role") or "").lower()
    pattern = str((edge or {}).get("pattern") or "").lower()
    if "cex_deposit" in pattern:
        return "cex_deposit_address"
    if "mixer_deposit" in pattern:
        return "mixer_deposit_contract"
    if "bridge" in pattern or "bridge" in role:
        return "bridge_contract"
    if any(value in pattern or value in role for value in ("dex", "swap", "router")):
        return "dex_router"
    if "exchange" in role or "cex" in role:
        return "cex_labeled_address"
    if "mixer" in role:
        return "mixer_deposit_contract"
    if "contract" in role or entity.get("type") == "contract_or_protocol":
        return "protocol_contract"
    return "service_address"


def _normal_mapping(
    mapping: dict[str, Any],
    *,
    default_role: str,
) -> dict[str, Any] | None:
    address = str(mapping.get("address") or "").lower()
    chain = str(mapping.get("chain") or "").lower()
    attribution_status = str(mapping.get("attribution_status") or "reported")
    if (
        chain != "ethereum"
        or not is_evm_address(address)
        or attribution_status not in {"chain_verified", "source_verified", "reported"}
    ):
        return None
    evidence_ids = sorted(
        {str(item) for item in mapping.get("evidence") or [] if str(item).strip()}
    )
    return {
        "chain": chain,
        "address": address,
        "address_role": str(mapping.get("address_role") or default_role),
        "attribution_status": attribution_status,
        "confidence": str(mapping.get("confidence") or "U"),
        "evidence": evidence_ids,
        "valid_from_block": mapping.get("valid_from_block"),
        "valid_to_block": mapping.get("valid_to_block"),
        **(
            {
                "matched_tx_hashes": sorted(
                    {str(value).lower() for value in mapping["matched_tx_hashes"]}
                )
            }
            if mapping.get("matched_tx_hashes")
            else {}
        ),
        **(
            {
                "matched_index_edge_ids": sorted(
                    {str(value) for value in mapping["matched_index_edge_ids"]}
                )
            }
            if mapping.get("matched_index_edge_ids")
            else {}
        ),
    }


def _merge_address_mappings(mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    status_rank = {"reported": 1, "source_verified": 2, "chain_verified": 3}
    confidence_rank = {"U": 0, "D": 1, "C": 2, "B": 3, "A": 4}
    for mapping in mappings:
        key = (
            mapping["chain"],
            mapping["address"],
            mapping["address_role"],
        )
        if key not in merged:
            merged[key] = dict(mapping)
            continue
        current = merged[key]
        current["evidence"] = sorted(set(current["evidence"] + mapping["evidence"]))
        if status_rank[mapping["attribution_status"]] > status_rank[current["attribution_status"]]:
            current["attribution_status"] = mapping["attribution_status"]
        if confidence_rank.get(mapping["confidence"], 0) > confidence_rank.get(current["confidence"], 0):
            current["confidence"] = mapping["confidence"]
        for field in ("matched_tx_hashes", "matched_index_edge_ids"):
            values = list(current.get(field) or []) + list(mapping.get(field) or [])
            if values:
                current[field] = sorted(set(values))
        starts = [value for value in (current.get("valid_from_block"), mapping.get("valid_from_block")) if value is not None]
        ends = [value for value in (current.get("valid_to_block"), mapping.get("valid_to_block")) if value is not None]
        current["valid_from_block"] = min(starts) if starts else None
        current["valid_to_block"] = max(ends) if ends else None
    return list(merged.values())


def _scanner_address_labels(evidence: dict[str, Any]) -> list[dict[str, str]]:
    """Collect explicit Etherscan labels with auditable scanner evidence ids."""
    labels: list[dict[str, str]] = []
    etherscan = evidence.get("scanner_evidence", {}).get("etherscan", {})
    for item in etherscan.get("address_results") or []:
        info = item.get("address_info") or {}
        address = str(info.get("address") or item.get("target", {}).get("address") or "").lower()
        for tag in info.get("tags") or []:
            label = str(tag.get("name") if isinstance(tag, dict) else tag).strip()
            if is_evm_address(address) and label:
                labels.append(
                    {
                        "address": address,
                        "label": label,
                        "evidence": f"etherscan:address:{address}",
                    }
                )
        for tx in (item.get("transactions") or {}).get("transactions") or []:
            tx_hash = str(tx.get("tx_hash") or "").lower()
            for side in ("from", "to"):
                address = str(tx.get(side) or "").lower()
                label = str(tx.get(f"{side}_label") or "").strip()
                if is_evm_address(address) and label:
                    labels.append(
                        {
                            "address": address,
                            "label": label,
                            "evidence": f"etherscan:tx:{tx_hash}",
                        }
                    )
    for item in etherscan.get("transaction_results") or []:
        tx = item.get("transaction") or {}
        tx_hash = str(tx.get("tx_hash") or item.get("target", {}).get("tx_hash") or "").lower()
        for side in ("from", "to"):
            address = str(tx.get(side) or "").lower()
            label = str(tx.get(f"{side}_label") or "").strip()
            if is_evm_address(address) and label:
                labels.append(
                    {
                        "address": address,
                        "label": label,
                        "evidence": f"etherscan:tx:{tx_hash}",
                    }
                )
        for link in tx.get("address_links") or []:
            address = str(link.get("address") or "").lower()
            label = str(link.get("label") or "").strip()
            if is_evm_address(address) and label and label.lower() != "write contract":
                labels.append(
                    {
                        "address": address,
                        "label": label,
                        "evidence": f"etherscan:tx:{tx_hash}",
                    }
                )
    return unique_preserve_order(labels)


def _scanner_label_matches_entity(label: str, entity: dict[str, Any]) -> bool:
    entity_id = str(entity.get("id") or "")
    if entity_id == "entity:dex_route":
        return False
    aliases = {
        "entity:tornado_cash": ("tornado cash", "tornado.cash"),
        "entity:blender_io": ("blender.io", "blender io"),
        "entity:cryptocom": ("crypto.com", "cryptocom"),
        "entity:oneinch": ("1inch", "1 inch"),
    }.get(entity_id)
    if aliases is None:
        name = re.sub(r"\s+", " ", str(entity.get("entity") or "").strip().lower())
        aliases = (name,) if len(name) >= 3 else ()
    normalized_label = re.sub(r"\s+", " ", label.strip().lower())
    return any(alias and alias in normalized_label for alias in aliases)


def attach_entity_address_mappings(evidence: dict[str, Any]) -> dict[str, Any]:
    """Attach exact, auditable service-address mappings and a deterministic audit."""
    entities = [entity for entity in evidence.get("entities") or [] if isinstance(entity, dict)]
    entity_map = {str(entity.get("id")): entity for entity in entities}
    eligible = [
        entity
        for entity in entities
        if entity.get("type") in {"service", "contract_or_protocol"}
        or str(entity.get("id") or "").startswith("entity:")
    ]

    mappings_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in eligible:
        entity_id = str(entity.get("id"))
        for mapping in entity.get("addresses") or []:
            if isinstance(mapping, dict):
                mappings_by_entity[entity_id].append(dict(mapping))

    scanner_labels = _scanner_address_labels(evidence)
    for entity in eligible:
        entity_id = str(entity.get("id"))
        for scanner_match in scanner_labels:
            if not _scanner_label_matches_entity(scanner_match["label"], entity):
                continue
            mappings_by_entity[entity_id].append(
                {
                    "chain": "ethereum",
                    "address": scanner_match["address"],
                    "address_role": entity_address_role(entity),
                    "attribution_status": "source_verified",
                    "confidence": "B",
                    "evidence": [scanner_match["evidence"]],
                    "valid_from_block": None,
                    "valid_to_block": None,
                }
            )

    report_edges = {
        str(edge.get("id")): edge
        for edge in evidence.get("edges") or []
        if isinstance(edge, dict)
    }
    tx_index_edges: dict[str, list[Any]] = defaultdict(list)
    for tx in evidence.get("transaction_evidence") or []:
        tx_hash = str(tx.get("tx_hash") or "").lower()
        tx_index_edges[tx_hash].extend(
            edge.get("edge_id") for edge in tx.get("index_edges") or [] if edge.get("edge_id") is not None
        )
    for match in evidence.get("local_index", {}).get("chain_evidence_candidates") or []:
        if match.get("verification_status") != "chain_candidate_rule_and_amount_match":
            continue
        edge = report_edges.get(str(match.get("report_edge_id")))
        candidate = match.get("candidate") or {}
        if not edge:
            continue
        tx_hash = str(match.get("tx_hash") or candidate.get("tx_hash") or "").lower()
        for endpoint, address_key in (("src", "src_address"), ("dst", "dst_address")):
            entity_id = str(edge.get(endpoint) or "")
            entity = entity_map.get(entity_id)
            address = str(candidate.get(address_key) or "").lower()
            if entity not in eligible or not is_evm_address(address):
                continue
            mappings_by_entity[entity_id].append(
                {
                    "chain": "ethereum",
                    "address": address,
                    "address_role": entity_address_role(entity, edge),
                    "attribution_status": "reported",
                    "confidence": str(edge.get("confidence") or "U"),
                    "evidence": list(edge.get("evidence") or []) + [f"local_index:tx:{tx_hash}"],
                    "valid_from_block": candidate.get("block_number"),
                    "valid_to_block": candidate.get("block_number"),
                    "matched_tx_hashes": [tx_hash] if TX_RE.fullmatch(tx_hash) else [],
                    "matched_index_edge_ids": tx_index_edges.get(tx_hash, []),
                }
            )

    invalid_addresses: list[dict[str, Any]] = []
    missing_evidence: list[dict[str, Any]] = []
    flat_mappings: list[dict[str, Any]] = []
    for entity in sorted(eligible, key=lambda item: str(item.get("id"))):
        entity_id = str(entity.get("id"))
        normalized: list[dict[str, Any]] = []
        for raw_mapping in mappings_by_entity.get(entity_id, []):
            mapping = _normal_mapping(
                raw_mapping,
                default_role=entity_address_role(entity),
            )
            if mapping is None:
                invalid_addresses.append(
                    {
                        "entity_id": entity_id,
                        "chain": raw_mapping.get("chain"),
                        "address": raw_mapping.get("address"),
                    }
                )
                continue
            if not mapping["evidence"]:
                missing_evidence.append(
                    {"entity_id": entity_id, "address": mapping["address"]}
                )
                continue
            normalized.append(mapping)
        entity["addresses"] = sorted(
            _merge_address_mappings(normalized),
            key=lambda item: (
                item["chain"],
                item["address"],
                item["address_role"],
                item["attribution_status"],
            ),
        )
        flat_mappings.extend(
            {"entity_id": entity_id, **mapping} for mapping in entity["addresses"]
        )

    by_address: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for mapping in flat_mappings:
        by_address[(mapping["chain"], mapping["address"])].append(mapping)
    conflicts = [
        {
            "chain": chain,
            "address": address,
            "entity_ids": sorted({item["entity_id"] for item in items}),
            "mappings": sorted(items, key=lambda item: item["entity_id"]),
        }
        for (chain, address), items in sorted(by_address.items())
        if len({item["entity_id"] for item in items}) > 1
    ]

    observed_edges: list[dict[str, Any]] = []
    for tx in evidence.get("transaction_evidence") or []:
        observed_edges.extend(edge for edge in tx.get("index_edges") or [] if isinstance(edge, dict))
    matched_edge_ids: set[str] = set()
    matched_addresses: set[str] = set()
    mapped_addresses = {address for chain, address in by_address if chain == "ethereum"}
    for edge in observed_edges:
        src = str(edge.get("src_address") or "").lower()
        dst = str(edge.get("dst_address") or "").lower()
        if src in mapped_addresses or dst in mapped_addresses:
            matched_addresses.update(address for address in (src, dst) if address in mapped_addresses)
            matched_edge_ids.add(str(edge.get("edge_id")))

    eligible_role_counts = Counter(str(entity.get("role") or "unknown") for entity in eligible)
    mapping_role_counts = Counter(
        str(entity_map[mapping["entity_id"]].get("role") or "unknown")
        for mapping in flat_mappings
    )
    address_role_counts = Counter(mapping["address_role"] for mapping in flat_mappings)
    status_counts = Counter(mapping["attribution_status"] for mapping in flat_mappings)
    mapped_entity_ids = sorted(
        str(entity.get("id")) for entity in eligible if entity.get("addresses")
    )
    audit = {
        "schema_version": "entity_address_matching.v1",
        "entity_count": len(entities),
        "eligible_entity_count": len(eligible),
        "mapped_entity_count": len(mapped_entity_ids),
        "unmapped_entity_count": len(eligible) - len(mapped_entity_ids),
        "address_mapping_count": len(flat_mappings),
        "mapped_entity_ids": mapped_entity_ids,
        "unmapped_entity_ids": sorted(
            str(entity.get("id")) for entity in eligible if not entity.get("addresses")
        ),
        "counts_by_role": dict(sorted(mapping_role_counts.items())),
        "counts_by_eligible_entity_role": dict(sorted(eligible_role_counts.items())),
        "counts_by_address_role": dict(sorted(address_role_counts.items())),
        "counts_by_attribution_status": dict(sorted(status_counts.items())),
        "matched_observed_addresses": sorted(matched_addresses),
        "matched_observed_address_count": len(matched_addresses),
        "matched_observed_edge_count": len(matched_edge_ids),
        "conflicts": conflicts,
        "invalid_addresses": invalid_addresses,
        "missing_evidence": missing_evidence,
        "mappings": sorted(
            flat_mappings,
            key=lambda item: (item["chain"], item["address"], item["entity_id"]),
        ),
        "status": "ok" if not invalid_addresses and not missing_evidence else "invalid",
    }
    evidence["entity_address_matching"] = audit
    return audit


class EthereumIndexClient:
    """Small wrapper around docs/eth_local_indexer.md HTTP API."""

    def __init__(self, base_url: str = ETH_INDEX_URL, timeout: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any] | None:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return None

    def node(self, address: str) -> dict[str, Any] | None:
        try:
            response = requests.get(
                f"{self.base_url}/node/{normalize_address(address)}",
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return None

    def neighbors(
        self,
        address: str,
        *,
        direction: str = "both",
        block_min: int | None = None,
        block_max: int | None = None,
        limit: int = 25,
        with_raw: bool = True,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "direction": direction,
            "limit": limit,
            "with_raw": str(with_raw).lower(),
        }
        if block_min is not None:
            params["block_min"] = block_min
        if block_max is not None:
            params["block_max"] = block_max
        try:
            response = requests.get(
                f"{self.base_url}/neighbors/{normalize_address(address)}",
                params=params,
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return []
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else data.get("edges", [])
        except requests.RequestException:
            return []


def raw_tx_hash(edge: dict[str, Any]) -> str | None:
    raw = edge.get("raw_parsed") or {}
    value = raw.get("hash") or raw.get("transaction_hash")
    return normalize_address(value) if isinstance(value, str) and TX_RE.fullmatch(value) else value


def raw_value_wei(edge: dict[str, Any]) -> int | None:
    raw = edge.get("raw_parsed") or {}
    value = raw.get("value")
    if value is None:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def summarize_index_edge(edge: dict[str, Any]) -> dict[str, Any]:
    raw = edge.get("raw_parsed") or {}
    value_wei = raw_value_wei(edge)
    token_address = edge.get("token_address") or raw.get("token_address")
    is_token_transfer = bool(raw.get("transaction_hash")) or bool(token_address)
    return {
        "edge_id": edge.get("edge_id"),
        "tx_hash": raw_tx_hash(edge),
        "src_address": edge.get("src_address") or raw.get("from_address"),
        "dst_address": edge.get("dst_address") or raw.get("to_address"),
        "neighbor_address": edge.get("neighbor_address"),
        "block_number": edge.get("block_number") or raw.get("block_number"),
        "timestamp": edge.get("timestamp") or raw.get("block_timestamp"),
        "edge_type": edge.get("edge_type"),
        "token_address": token_address,
        "value_wei": str(value_wei) if value_wei is not None and not is_token_transfer else None,
        "value_eth": value_wei / 10**18 if value_wei is not None and not is_token_transfer else None,
        "raw_token_amount": str(value_wei) if value_wei is not None and is_token_transfer else None,
        "raw_kind": "token_transfer" if is_token_transfer else "transaction",
    }


def tx_key(tx_hash: str | None) -> str | None:
    if isinstance(tx_hash, str) and TX_RE.fullmatch(tx_hash):
        return tx_hash.lower()
    return None


def entity_by_id(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(entity.get("id")): entity for entity in evidence.get("entities", [])}


def known_attacker_addresses(evidence: dict[str, Any]) -> set[str]:
    attackers: set[str] = set()
    for entity in evidence.get("entities", []):
        address = entity.get("address")
        role = str(entity.get("role") or "")
        if isinstance(address, str) and is_evm_address(address) and "attacker" in role:
            attackers.add(normalize_address(address))
    return attackers


def known_service_labels(evidence: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for entity in evidence.get("entities", []):
        entity_id = str(entity.get("id") or "")
        name = str(entity.get("entity") or entity_id).lower()
        role = str(entity.get("role") or "").lower()
        if "exchange" in role or any(x in name for x in ("ftx", "crypto.com", "huobi", "binance")):
            labels[name] = entity_id
    return labels


def classify_transaction_rules(tx: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    src = normalize_address(str(tx.get("src_address") or ""))
    dst = normalize_address(str(tx.get("dst_address") or ""))
    value_eth = tx.get("value_eth")
    token = tx.get("token_address")
    raw_kind = tx.get("raw_kind")
    attacker_addresses = known_attacker_addresses(evidence)

    labels: list[dict[str, Any]] = []
    if src in attacker_addresses and raw_kind == "transaction" and isinstance(value_eth, (int, float)):
        if 0 < value_eth <= 5:
            labels.append(
                {
                    "label": "attacker_temporary_wallet_activation",
                    "stage": "fanout_or_activation",
                    "confidence": "B",
                    "reasons": [
                        "native_eth_transfer_from_known_attacker",
                        "small_amount_consistent_with_gas_or_wallet_activation",
                    ],
                }
            )
        elif value_eth >= 100:
            labels.append(
                {
                    "label": "large_native_eth_outflow_from_attacker",
                    "stage": "layering",
                    "confidence": "B",
                    "reasons": ["native_eth_transfer_from_known_attacker", "large_value_transfer"],
                }
            )

    if src in attacker_addresses and token:
        labels.append(
            {
                "label": "token_transfer_from_attacker",
                "stage": "conversion_or_layering",
                "confidence": "C",
                "reasons": ["token_transfer_from_known_attacker"],
            }
        )

    if dst in attacker_addresses and raw_kind == "transaction":
        labels.append(
            {
                "label": "inbound_funding_to_attacker_address",
                "stage": "funding",
                "confidence": "C",
                "reasons": ["native_eth_transfer_to_known_attacker"],
            }
        )

    if not labels:
        labels.append(
            {
                "label": "unclassified_chain_transaction",
                "stage": "unknown",
                "confidence": "U",
                "reasons": ["no_rule_matched"],
            }
        )

    primary = labels[0]
    return {
        "primary_label": primary["label"],
        "stage": primary["stage"],
        "confidence": primary["confidence"],
        "labels": labels,
    }


def build_transaction_evidence(
    evidence: dict[str, Any],
    lookups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_hash: dict[str, dict[str, Any]] = {}
    window = parse_date_to_unix_window(evidence.get("time_window", {}).get("attack_start"), days=2)
    for lookup in lookups:
        lookup_address = lookup.get("address")
        for raw_edge in lookup.get("sample_neighbors", []):
            summarized = summarize_index_edge(raw_edge)
            tx_hash = tx_key(summarized.get("tx_hash"))
            if not tx_hash:
                continue
            if window and summarized.get("timestamp") is not None:
                try:
                    timestamp = int(str(summarized["timestamp"]))
                    summarized["time_window_match"] = window[0] <= timestamp <= window[1]
                except ValueError:
                    summarized["time_window_match"] = False
            entry = by_hash.setdefault(
                tx_hash,
                {
                    "tx_hash": tx_hash,
                    "chain": "Ethereum",
                    "index_edges": [],
                    "seen_from_lookup_addresses": [],
                    "scanner_urls": {
                        "etherscan": f"https://etherscan.io/tx/{tx_hash}",
                        "oklink": f"https://www.oklink.com/en/eth/tx/{tx_hash}",
                    },
                },
            )
            if lookup_address and lookup_address not in entry["seen_from_lookup_addresses"]:
                entry["seen_from_lookup_addresses"].append(lookup_address)
            entry["index_edges"].append(summarized)

    txs: list[dict[str, Any]] = []
    for entry in by_hash.values():
        representative = entry["index_edges"][0]
        entry.update({k: representative.get(k) for k in (
            "src_address",
            "dst_address",
            "block_number",
            "timestamp",
            "value_wei",
            "value_eth",
            "token_address",
            "raw_token_amount",
            "raw_kind",
            "time_window_match",
        )})
        rule_label = classify_transaction_rules(entry, evidence)
        entry["rule_labels"] = rule_label
        entry["verification_status"] = "chain_index_observed_rule_labeled"
        txs.append(entry)
    return sorted(txs, key=lambda x: (str(x.get("timestamp") or ""), x["tx_hash"]))


def edge_match_reasons(report_edge: dict[str, Any], index_edge: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    src = str(report_edge.get("src") or "").lower()
    dst = str(report_edge.get("dst") or "").lower()
    idx_src = str(index_edge.get("src_address") or "").lower()
    idx_dst = str(index_edge.get("dst_address") or "").lower()
    if src and is_evm_address(src) and src == idx_src:
        reasons.append("src_address_match")
    if dst and is_evm_address(dst) and dst == idx_dst:
        reasons.append("dst_address_match")
    if src and is_evm_address(src) and src == idx_dst:
        reasons.append("reverse_src_seen")
    if dst and is_evm_address(dst) and dst == idx_src:
        reasons.append("reverse_dst_seen")

    expected = parse_amount_number(report_edge.get("amount"))
    value_eth = index_edge.get("value_eth")
    if expected is not None and isinstance(value_eth, (int, float)):
        tolerance = max(expected * 0.005, 0.000001)
        if abs(value_eth - expected) <= tolerance:
            reasons.append("amount_match")
        elif expected > 100 and value_eth > expected * 0.2:
            reasons.append("large_value_same_order")
    return reasons


def transaction_supports_report_edge(
    report_edge: dict[str, Any],
    tx: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    label = str(tx.get("rule_labels", {}).get("primary_label") or "")
    stage = str(report_edge.get("stage") or "")
    pattern = str(report_edge.get("pattern") or "")
    value_eth = tx.get("value_eth")
    expected = parse_amount_number(report_edge.get("amount"))

    if pattern == "cex_deposit":
        if label != "large_native_eth_outflow_from_attacker":
            return False, []
        if expected is not None and isinstance(value_eth, (int, float)):
            tolerance = max(expected * 0.01, 0.01)
            if abs(value_eth - expected) <= tolerance:
                reasons.append("cex_amount_match")
            else:
                return False, []
        return bool(reasons), reasons

    if stage == "conversion":
        if label not in {"token_transfer_from_attacker", "large_native_eth_outflow_from_attacker"}:
            return False, []
        if tx.get("token_address"):
            reasons.append("token_transfer_conversion_candidate")
        return bool(reasons), reasons

    if stage in {"theft", "bridge_exit"}:
        if expected is not None and isinstance(value_eth, (int, float)):
            tolerance = max(expected * 0.005, 0.01)
            if abs(value_eth - expected) <= tolerance:
                reasons.append("amount_match")
        if tx.get("time_window_match"):
            reasons.append("attack_time_window_match")
        return "amount_match" in reasons, reasons

    if stage in {"layering", "mixing"} and label in {
        "large_native_eth_outflow_from_attacker",
        "token_transfer_from_attacker",
    }:
        reasons.append("stage_compatible_rule_label")
        return True, reasons

    return False, []


def build_index_candidates(
    evidence: dict[str, Any],
    lookups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    txs = evidence.get("transaction_evidence") or build_transaction_evidence(evidence, lookups)
    for tx in txs:
        for report_edge in evidence.get("edges", []):
            supported, reasons = transaction_supports_report_edge(report_edge, tx)
            if supported:
                candidates.append(
                    {
                        "report_edge_id": report_edge.get("id"),
                        "tx_hash": tx.get("tx_hash"),
                        "candidate": {
                            key: tx.get(key)
                            for key in (
                                "tx_hash",
                                "src_address",
                                "dst_address",
                                "block_number",
                                "timestamp",
                                "value_wei",
                                "value_eth",
                                "token_address",
                                "raw_token_amount",
                                "raw_kind",
                                "time_window_match",
                            )
                        },
                        "transaction_label": tx.get("rule_labels", {}).get("primary_label"),
                        "match_reasons": reasons,
                        "verification_status": (
                            "chain_candidate_rule_and_amount_match"
                            if any("amount_match" in r for r in reasons)
                            else "chain_candidate_partial_match"
                        ),
                    }
                )
    return unique_preserve_order(candidates)


def build_scanner_targets(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for entity in evidence.get("entities", []):
        address = entity.get("address")
        if not isinstance(address, str) or not is_evm_address(address):
            continue
        targets.append(
            {
                "chain": entity.get("chain") or "Ethereum",
                "scanner": "etherscan",
                "address": normalize_address(address),
                "role": entity.get("role"),
                "url": f"https://etherscan.io/address/{normalize_address(address)}",
            }
        )
    for tx in evidence.get("exploit_txs", []):
        tx_hash = tx.get("tx_hash")
        if isinstance(tx_hash, str) and TX_RE.fullmatch(tx_hash):
            targets.append(
                {
                    "chain": "Ethereum",
                    "scanner": "etherscan",
                    "tx_hash": tx_hash.lower(),
                    "role": tx.get("role", "transaction"),
                    "url": f"https://etherscan.io/tx/{tx_hash.lower()}",
                }
            )
    for tx in evidence.get("transaction_evidence", []):
        tx_hash = tx.get("tx_hash")
        if isinstance(tx_hash, str) and TX_RE.fullmatch(tx_hash):
            targets.append(
                {
                    "chain": "Ethereum",
                    "scanner": "etherscan",
                    "tx_hash": tx_hash.lower(),
                    "role": tx.get("rule_labels", {}).get("primary_label", "transaction"),
                    "url": f"https://etherscan.io/tx/{tx_hash.lower()}",
                }
            )
    return unique_preserve_order(targets)


def enrich_with_etherscan(
    evidence: dict[str, Any],
    *,
    max_transactions_pages: int,
    max_transaction_targets: int,
) -> None:
    targets = build_scanner_targets(evidence)
    address_targets = [target for target in targets if target.get("address")]
    tx_targets = [target for target in targets if target.get("tx_hash")][:max_transaction_targets]
    fetch_targets = address_targets + tx_targets
    evidence.setdefault("scanner_evidence", {})["targets"] = targets
    evidence["scanner_evidence"]["etherscan_fetch_targets"] = fetch_targets
    if not targets:
        evidence["scanner_evidence"]["etherscan"] = {
            "status": "skipped",
            "reason": "no_evm_address_or_tx_targets",
        }
        return
    try:
        from etherscan_browser import EtherscanBrowserClient

        client = EtherscanBrowserClient()
        address_results: list[dict[str, Any]] = []
        tx_results: list[dict[str, Any]] = []
        for target in fetch_targets:
            address = target.get("address")
            tx_hash = target.get("tx_hash")
            if address:
                item: dict[str, Any] = {"target": target}
                try:
                    item["address_info"] = client.get_address_info(address)
                except Exception as exc:  # noqa: BLE001
                    item["address_info_error"] = str(exc)
                try:
                    item["transactions"] = client.get_transactions(
                        address=address,
                        max_pages=max_transactions_pages,
                    )
                except Exception as exc:  # noqa: BLE001
                    item["transactions_error"] = str(exc)
                address_results.append(item)
            elif tx_hash:
                item = {"target": target}
                try:
                    item["transaction"] = client.get_transaction(tx_hash)
                except Exception as exc:  # noqa: BLE001
                    item["transaction_error"] = str(exc)
                tx_results.append(item)
        evidence["scanner_evidence"]["etherscan"] = {
            "status": "available",
            "implementation": "etherscan_browser.EtherscanBrowserClient",
            "address_results": address_results,
            "transaction_results": tx_results,
            "transaction_target_limit": max_transaction_targets,
        }
    except Exception as exc:  # noqa: BLE001
        evidence["scanner_evidence"]["etherscan"] = {
            "status": "unavailable",
            "implementation": "etherscan_browser.EtherscanBrowserClient",
            "error": str(exc),
        }


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            import os

            os.environ.setdefault(key, value)


def complete_llm_json(messages: list[dict[str, str]]) -> dict[str, Any]:
    import os

    load_env_file(PROJECT_ROOT / ".env")
    base_url = os.environ.get("LLM_PROVIDER_URL", "http://127.0.0.1:18080/v1").rstrip("/")
    model = os.environ.get("LLM_NAME", "Qwen/Qwen3.6-27B")
    api_key = os.environ.get("LLM_API_KEY", "NO_API_KEY")
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
        },
        timeout=180,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def attach_llm_transaction_labels(evidence: dict[str, Any]) -> None:
    txs = evidence.get("transaction_evidence") or []
    if not txs:
        return

    scanner_tx_by_hash: dict[str, dict[str, Any]] = {}
    for item in evidence.get("scanner_evidence", {}).get("etherscan", {}).get("transaction_results", []):
        tx = item.get("transaction") or {}
        tx_hash = tx_key(tx.get("tx_hash"))
        if tx_hash:
            scanner_tx_by_hash[tx_hash] = tx

    labels: list[dict[str, Any]] = []
    for tx in txs[:50]:
        tx_hash = tx.get("tx_hash")
        scanner_tx = scanner_tx_by_hash.get(tx_hash, {})
        prompt_payload = {
            "case_id": evidence.get("case_id"),
            "incident_type": evidence.get("incident_type"),
            "report_edges": evidence.get("edges"),
            "transaction": {
                k: tx.get(k)
                for k in (
                    "tx_hash",
                    "src_address",
                    "dst_address",
                    "timestamp",
                    "value_eth",
                    "token_address",
                    "raw_token_amount",
                    "raw_kind",
                    "rule_labels",
                )
            },
            "etherscan_transaction": scanner_tx,
            "allowed_labels": [
                "attacker_temporary_wallet_activation",
                "large_native_eth_layering_transfer",
                "cex_deposit",
                "dex_swap_or_conversion",
                "mixer_deposit_or_withdrawal",
                "exploit_or_theft",
                "irrelevant_or_unknown",
            ],
        }
        try:
            llm_result = complete_llm_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You label blockchain transactions for incident chain evidence. "
                            "Return strict JSON with label, confidence A/B/C/U, reasons, and matched_report_edge_ids. "
                            "Do not infer CEX ownership or DEX swap unless the tx details or labels support it."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt_payload, ensure_ascii=False),
                    },
                ]
            )
        except Exception as exc:  # noqa: BLE001
            llm_result = {
                "label": "llm_label_error",
                "confidence": "U",
                "reasons": [str(exc)],
                "matched_report_edge_ids": [],
            }
        labels.append({"tx_hash": tx_hash, "llm_label": llm_result})

    evidence.setdefault("llm_labels", {})["transactions"] = labels


def build_entities(
    summary: dict[str, Any],
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chains = summary.get("chains") or []
    attacker = summary.get("attacker") or {}
    attacker_addresses = parse_attacker_address_entries(attacker)

    entities: list[dict[str, Any]] = []
    for victim in summary.get("victim_contracts") or []:
        victim_label = str(victim)
        evidence = evidence_ids_for_keywords(sources, [victim_label, "bridge", "victim"], limit=5)
        add_entity_once(
            entities,
            {
                "id": entity_id_from_label(victim_label),
                "type": "contract_or_protocol",
                "entity": victim_label,
                "role": "victim",
                "confidence": confidence_from_evidence(sources, evidence),
                "evidence": evidence,
            },
        )

    ofac_evidence = evidence_ids_for_keywords(sources, ["OFAC", "sanction", "SDN"], {"A", "B"})
    attacker_evidence = evidence_ids_for_keywords(
        sources,
        ["attacker address", "hacker address", "Lazarus", "OFAC"],
        {"A", "B", "C", "D"},
    )

    for idx, entry in enumerate(attacker_addresses):
        address = entry["address"]
        role = "attacker_initial" if idx == 0 else "attacker_controlled_wallet"
        if entry["format"] == "evm":
            role = "attacker_ethereum_wallet" if idx else "attacker_initial"
        elif "fake sysvar" in entry["source_text"].lower():
            role = "fake_sysvar_account"
        elif entry["format"] == "solana":
            role = "attacker_solana_wallet" if idx else "attacker_initial"
        address_evidence = evidence_ids_for_keywords(sources, [address], limit=5)
        if address_evidence:
            evidence = address_evidence
        elif idx == 0 and attacker_evidence:
            evidence = attacker_evidence
        elif idx < 4 and ofac_evidence:
            evidence = ofac_evidence
        else:
            evidence = []
        entities.append(
            {
                "id": entry["id"],
                "type": "address",
                "address": address,
                "chain": entry["chain"],
                "role": role,
                "entity": attacker.get("attribution"),
                "confidence": confidence_from_evidence(sources, evidence),
                "evidence": evidence,
                "scanner_urls": scanner_urls(address, chains),
            }
        )

    text_blob = json.dumps(summary, ensure_ascii=False)
    for address in find_addresses(text_blob):
        if address in {entity["address"] for entity in entities if "address" in entity}:
            continue
        evidence = evidence_ids_for_keywords(sources, [address], limit=3)
        entities.append(
            {
                "id": address,
                "type": "address",
                "address": address,
                "chain": "Ethereum",
                "role": "mentioned_address",
                "confidence": confidence_from_evidence(sources, evidence),
                "evidence": evidence,
                "scanner_urls": scanner_urls(address, chains),
            }
        )

    for address in SOLANA_ADDRESS_RE.findall(ADDRESS_RE.sub(" ", text_blob)):
        if address in {entity["address"] for entity in entities if "address" in entity}:
            continue
        evidence = evidence_ids_for_keywords(sources, [address], limit=3)
        entities.append(
            {
                "id": address,
                "type": "address",
                "address": address,
                "chain": "Solana",
                "role": "mentioned_address",
                "confidence": confidence_from_evidence(sources, evidence),
                "evidence": evidence,
                "scanner_urls": scanner_urls(address, chains),
            }
        )

    if "tornado" in text_blob.lower():
        evidence = evidence_ids_for_keywords(sources, ["Tornado Cash", "mixer"], {"A", "B", "C"})
        entities.append(
            {
                "id": "entity:tornado_cash",
                "type": "service",
                "entity": "Tornado Cash",
                "role": "mixer_contract_set",
                "confidence": confidence_from_evidence(sources, evidence),
                "evidence": evidence,
            }
        )

    service_specs = [
        ("entity:dex_route", "DEX route", "dex_or_swap_route", ["DEX", "swap", "1inch", "Uniswap", "Curve"]),
        ("entity:maker", "Maker", "lending_protocol", ["Maker", "borrow", "DAI"]),
        ("entity:curve", "Curve", "dex_or_swap_route", ["Curve", "SynthSwap"]),
        ("entity:oneinch", "1inch", "dex_or_swap_route", ["1inch"]),
        ("entity:ftx", "FTX", "centralized_exchange", ["FTX"]),
        ("entity:cryptocom", "Crypto.com", "centralized_exchange", ["Crypto.com"]),
        ("entity:huobi", "Huobi", "centralized_exchange", ["Huobi"]),
    ]
    for entity_id, entity_name, role, keywords in service_specs:
        if any(keyword.lower() in text_blob.lower() for keyword in keywords):
            evidence = evidence_ids_for_keywords(sources, keywords, {"B", "C", "D"})
            add_entity_once(
                entities,
                {
                    "id": entity_id,
                    "type": "service",
                    "entity": entity_name,
                    "role": role,
                    "confidence": confidence_from_evidence(sources, evidence),
                    "evidence": evidence,
                },
            )

    if "blender" in text_blob.lower():
        evidence = evidence_ids_for_keywords(sources, ["Blender", "mixer"], {"A", "B", "C"})
        entities.append(
            {
                "id": "entity:blender_io",
                "type": "service",
                "entity": "Blender.io",
                "role": "bitcoin_mixer",
                "confidence": confidence_from_evidence(sources, evidence),
                "evidence": evidence,
            }
        )

    return unique_preserve_order(entities)


def parse_loss_assets(loss_assets: list[str]) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    for item in loss_assets:
        match = re.match(r"\s*([\d,.]+)\s+([A-Za-z0-9]+)\s*$", str(item))
        if not match:
            continue
        amount, asset = match.groups()
        parsed.append({"amount": amount, "asset": asset.upper()})
    return parsed


def build_edges(summary: dict[str, Any], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attacker_entries = parse_attacker_address_entries(summary.get("attacker") or {})
    evm_attackers = [entry["id"] for entry in attacker_entries if entry["format"] == "evm"]
    sol_attackers = [
        entry["id"]
        for entry in attacker_entries
        if entry["format"] == "solana" and "fake sysvar" not in entry["source_text"].lower()
    ]
    primary = evm_attackers[0] if evm_attackers else (sol_attackers[0] if sol_attackers else "entity:attacker_unknown")
    primary_solana = sol_attackers[0] if sol_attackers else primary
    victim_contracts = [str(v) for v in summary.get("victim_contracts") or []]
    victim_label = victim_contracts[0] if victim_contracts else str(summary.get("protocol_type") or "Victim protocol")
    victim = entity_id_from_label(victim_label)
    incident_date = summary.get("incident_date")

    theft_evidence = evidence_ids_for_keywords(
        sources,
        ["stolen", "mint", "withdraw", "withdrawal", "bridge", *[str(x) for x in summary.get("loss_assets") or []]],
        {"B", "C", "D"},
    )
    edges: list[dict[str, Any]] = []
    for asset in parse_loss_assets(summary.get("loss_assets") or []):
        text_blob = json.dumps(summary, ensure_ascii=False).lower()
        if "mint" in text_blob and "without collateral" in text_blob:
            pattern = "fraudulent_uncollateralized_mint"
            dst = primary_solana
            notes = "Report-derived Solana-side mint edge; Ethereum index cannot directly verify Solana program execution."
        else:
            pattern = "fraudulent_bridge_withdrawal"
            dst = primary
            notes = "Report-derived bridge withdrawal edge; concrete tx hash must be filled from scanner or local index before tx-level training."
        edges.append(
            {
                "id": f"edge:theft:{asset['asset'].lower()}",
                "src": victim,
                "dst": dst,
                "stage": "theft",
                "pattern": pattern,
                "asset": "WETH" if asset["asset"] == "ETH" else asset["asset"],
                "amount": asset["amount"],
                "timestamp": incident_date,
                "tx_hash": None,
                "confidence": confidence_from_evidence(sources, theft_evidence),
                "evidence": theft_evidence,
                "verification_status": "reported_not_chain_verified",
                "notes": notes,
            }
        )

    fund_flow = str(summary.get("fund_flow") or "")
    if "USDC" in fund_flow and ("swap" in fund_flow.lower() or "DEX" in fund_flow):
        evidence = evidence_ids_for_keywords(sources, ["USDC", "swapped", "DEX"], {"B", "C"})
        edges.append(
            {
                "id": "edge:conversion:usdc_to_eth",
                "src": primary,
                "dst": "entity:dex_route",
                "stage": "conversion",
                "pattern": "stablecoin_to_native_asset_swap",
                "asset_in": "USDC",
                "asset_out": "ETH",
                "amount": "25,500,000 USDC",
                "confidence": confidence_from_evidence(sources, evidence),
                "evidence": evidence,
                "verification_status": "reported_not_chain_verified",
            }
        )

    cex_specs = [
        ("FTX", r"FTX[:：]?\s*([\d,.]+)\s*ETH"),
        ("Crypto.com", r"Crypto\.com[:：]?\s*([\d,.]+)\s*ETH"),
        ("Huobi", r"Huobi[:：]?\s*([\d,.]+)\s*ETH"),
    ]
    for name, pattern in cex_specs:
        match = re.search(pattern, fund_flow, flags=re.IGNORECASE)
        if not match:
            continue
        evidence = evidence_ids_for_keywords(sources, [name, match.group(1)], {"B", "C", "D"})
        edges.append(
            {
                "id": f"edge:cex:{name.lower().replace('.', '').replace(' ', '_')}",
                "src": primary,
                "dst": f"entity:{name.lower().replace('.', '').replace(' ', '_')}",
                "stage": "layering",
                "pattern": "cex_deposit",
                "asset": "ETH",
                "amount": match.group(1),
                "confidence": confidence_from_evidence(sources, evidence),
                "evidence": evidence,
                "verification_status": "reported_not_chain_verified",
            }
        )

    tornado_match = re.search(r"Tornado Cash.*?(\$?[\d,.]+M\+?|\d[\d,.]*\s*ETH)", fund_flow, re.IGNORECASE)
    if "tornado" in fund_flow.lower():
        evidence = evidence_ids_for_keywords(sources, ["Tornado Cash", "mixer"], {"A", "B", "C"})
        edges.append(
            {
                "id": "edge:funding_or_mixing:tornado_cash",
                "src": "entity:tornado_cash" if "from tornado" in fund_flow.lower() or "received" in fund_flow.lower() else primary,
                "dst": "entity:tornado_cash",
                "stage": "funding" if "from tornado" in fund_flow.lower() or "received" in fund_flow.lower() else "mixing",
                "pattern": "mixer_withdrawal_to_attacker" if "from tornado" in fund_flow.lower() or "received" in fund_flow.lower() else "mixer_deposit",
                "amount": tornado_match.group(1) if tornado_match else None,
                "confidence": confidence_from_evidence(sources, evidence),
                "evidence": evidence,
                "verification_status": "reported_not_chain_verified",
            }
        )
        if edges[-1]["stage"] == "funding":
            edges[-1]["dst"] = primary

    if "BNB" in fund_flow and "BitTorrent" in fund_flow:
        evidence = evidence_ids_for_keywords(sources, ["BNB", "USDD", "BitTorrent", "chain-hopping"], {"B", "C"})
        edges.append(
            {
                "id": "edge:chain_hop:eth_bnb_bittorrent",
                "src": "Ethereum",
                "dst": "BitTorrent Chain",
                "stage": "chain_hopping",
                "pattern": "bridge_swap_bridge",
                "route": ["Ethereum", "BNB Chain", "USDD", "BitTorrent Chain"],
                "confidence": confidence_from_evidence(sources, evidence),
                "evidence": evidence,
                "verification_status": "reported_not_chain_verified",
            }
        )

    if "bridged" in fund_flow.lower() and evm_attackers:
        bridge_match = re.search(r"bridged\s+([\d,.]+)\s+(?:W?ETH|ETH)", fund_flow, re.IGNORECASE)
        evidence = evidence_ids_for_keywords(sources, ["93,750", "Ethereum address", "bridged"], {"B", "C"})
        if bridge_match or evidence:
            edges.append(
                {
                    "id": "edge:bridge_exit:to_ethereum",
                    "src": primary_solana,
                    "dst": evm_attackers[0],
                    "stage": "bridge_exit",
                    "pattern": "cross_chain_exit_to_ethereum",
                    "asset": "ETH",
                    "amount": bridge_match.group(1) if bridge_match else "93,750",
                    "confidence": confidence_from_evidence(sources, evidence),
                    "evidence": evidence,
                    "verification_status": "reported_not_chain_verified",
                }
            )

    if "steth" in fund_flow.lower() or "wsteth" in fund_flow.lower():
        evidence = evidence_ids_for_keywords(sources, ["stETH", "wstETH", "Maker", "DAI"], {"B", "C"})
        edges.append(
            {
                "id": "edge:defi_reinvestment:eth_to_steth_maker",
                "src": primary,
                "dst": "entity:maker",
                "stage": "post_dormancy_reinvestment",
                "pattern": "swap_stake_collateralize_borrow",
                "route": ["ETH", "stETH/wstETH", "Maker collateral", "DAI borrow"],
                "confidence": confidence_from_evidence(sources, evidence),
                "evidence": evidence,
                "verification_status": "reported_not_chain_verified",
            }
        )

    if len(evm_attackers) >= 3:
        peck_evidence = evidence_ids_for_keywords(sources, ["PeckShield", "1,528", "10,129"], limit=3)
        if peck_evidence:
            intermediary = normalize_address("0x3Cffd56B47B7b41c56258D9C7731ABaDc360E073")
            dst = normalize_address("0x8fa7b50fc8306ab3de028254df72bf08216742b6")
            if intermediary in evm_attackers:
                edges.append(
                    {
                        "id": "edge:layering:primary_to_intermediary_3cffd",
                        "src": primary,
                        "dst": intermediary,
                        "stage": "layering",
                        "pattern": "reported_intermediary_transfer",
                        "asset": "ETH",
                        "amount": "10,129.9",
                        "confidence": confidence_from_evidence(sources, peck_evidence),
                        "evidence": peck_evidence,
                        "verification_status": "reported_not_chain_verified",
                    }
                )
            edges.append(
                {
                    "id": "edge:layering:intermediary_to_8fa7",
                    "src": intermediary,
                    "dst": dst,
                    "stage": "layering",
                    "pattern": "reported_intermediary_transfer",
                    "asset": "ETH",
                    "amount": "1,528.2",
                    "confidence": confidence_from_evidence(sources, peck_evidence),
                    "evidence": peck_evidence,
                    "verification_status": "reported_not_chain_verified",
                }
            )

    return unique_preserve_order(edges)


def derive_skills(summary: dict[str, Any], edges: list[dict[str, Any]]) -> list[str]:
    text = json.dumps(summary, ensure_ascii=False).lower()
    skills = [
        "extract_seed_addresses_from_multi_source_reports",
        "separate_reported_edges_from_chain_verified_edges",
        "assign_confidence_from_source_grade_and_cross_source_support",
    ]
    if "bridge" in text:
        skills.append("identify_bridge_exit_path")
    if "validator" in text or "private key" in text:
        skills.append("recognize_validator_key_compromise_as_root_cause")
    if "usdc" in text and "swap" in text:
        skills.append("trace_stablecoin_swap_to_native_asset_after_theft")
    if any(edge.get("pattern") == "cex_deposit" for edge in edges):
        skills.append("detect_cex_deposit_layering_from_reported_amounts")
    if "tornado" in text:
        skills.append("stop_at_censored_mixer_boundary_unless_withdrawal_linkage_evidence")
    if "chain-hopping" in text or "bnb" in text:
        skills.append("identify_bridge_swap_bridge_chain_hopping_route")
    if "12,000" in text or "12000" in text:
        skills.append("detect_large_address_fanout_laundering_pattern")
    return unique_preserve_order(skills)


def negative_constraints(summary: dict[str, Any]) -> list[str]:
    constraints = [
        "do_not_treat_reported_edges_as_chain_verified_without_tx_hash_or_index_match",
        "do_not_infer_cex_account_owner_without_external_evidence",
        "do_not_merge_all_attacker_mentioned_addresses_without_source_support",
    ]
    text = json.dumps(summary, ensure_ascii=False).lower()
    if "tornado" in text:
        constraints.append("do_not_label_all_tornado_withdrawals_as_same_actor")
    if "sanction" in text or "ofac" in text:
        constraints.append("do_not_expand_sanctioned_entity_scope_beyond_listed_addresses_without_evidence")
    return constraints


def enrich_with_local_index(
    evidence: dict[str, Any],
    *,
    client: EthereumIndexClient,
    neighbor_limit: int,
) -> None:
    health = client.health()
    evidence["local_index"] = {
        "base_url": client.base_url,
        "health": health,
        "query_strategy": "address_neighbors_with_raw_rows; timestamp window is evaluated after retrieval because API filters by block number",
        "attack_time_window_unix": parse_date_to_unix_window(evidence.get("time_window", {}).get("attack_start"), days=2),
        "lookups": [],
    }
    if not health:
        evidence["local_index"]["status"] = "unavailable"
        return

    evidence["local_index"]["status"] = "available"
    for entity in evidence.get("entities", []):
        address = entity.get("address")
        if not address:
            continue
        node = client.node(address)
        neighbors = client.neighbors(address, limit=neighbor_limit, with_raw=True)
        evidence["local_index"]["lookups"].append(
            {
                "address": address,
                "node": node,
                "sample_neighbors": neighbors[:neighbor_limit],
            }
        )
    evidence["transaction_evidence"] = build_transaction_evidence(
        evidence,
        evidence["local_index"]["lookups"],
    )
    evidence["local_index"]["chain_evidence_candidates"] = build_index_candidates(
        evidence,
        evidence["local_index"]["lookups"],
    )


def build_chain_evidence(
    summary_path: Path,
    *,
    enrich_index: bool = False,
    enrich_etherscan: bool = False,
    label_transactions_llm: bool = False,
    eth_index_url: str = ETH_INDEX_URL,
    neighbor_limit: int = 10,
    etherscan_pages: int = 1,
    etherscan_tx_limit: int = 25,
) -> dict[str, Any]:
    raw = read_json(summary_path)
    summary = raw.get("summary") or {}
    case_dir = raw.get("case_dir") or summary_path.parent.name
    sources, _ = source_catalog(raw.get("sources_used") or [])
    entities = build_entities(summary, sources)
    edges = build_edges(summary, sources)
    txs = find_txs(json.dumps(summary, ensure_ascii=False))

    evidence: dict[str, Any] = {
        "schema_version": "chain_evidence.v1",
        "case_id": compact_case_id(case_dir, raw.get("case_name")),
        "case_name": raw.get("case_name"),
        "case_dir": case_dir,
        "incident_type": incident_type(summary),
        "chains": summary.get("chains") or [],
        "protocol_type": summary.get("protocol_type"),
        "time_window": {
            "attack_start": summary.get("incident_date"),
            "public_disclosure": "2022-03-29" if "ronin" in str(raw.get("case_name", "")).lower() else None,
            "laundering_observed_until": None,
        },
        "loss": {
            "usd": summary.get("loss_amount_usd"),
            "assets": summary.get("loss_assets") or [],
        },
        "root_cause": {
            "category": summary.get("vulnerability_type"),
            "summary": summary.get("root_cause"),
        },
        "entities": entities,
        "edges": edges,
        "exploit_txs": [
            {
                "tx_hash": tx_hash,
                "role": "exploit",
                "confidence": "B",
                "verification_status": "extracted_from_summary_text",
            }
            for tx_hash in txs
        ],
        "attack_flow": summary.get("attack_flow") or [],
        "skills": derive_skills(summary, edges),
        "negative_constraints": negative_constraints(summary),
        "open_questions": [
            "Exploit transaction hashes were not available in the current summary; verify via Ronin/Ethereum scanner or local index before training tx-level labels.",
            "Mixer deposit and withdrawal linkage should remain boundary-limited unless independent linkage evidence is added.",
        ],
        "sources": sources,
        "scanner_evidence": {
            "targets": [],
            "notes": "Scanner fetch is optional. Run with --enrich-etherscan to collect live Etherscan address/transaction tables via etherscan_browser.py and docs/browser.md.",
        },
        "generation": {
            "method": "deterministic_summary_to_chain_evidence",
            "input": str(summary_path.relative_to(PROJECT_ROOT)),
        },
    }
    evidence["scanner_evidence"]["targets"] = build_scanner_targets(evidence)

    if enrich_index:
        enrich_with_local_index(
            evidence,
            client=EthereumIndexClient(eth_index_url),
            neighbor_limit=neighbor_limit,
        )
        evidence["scanner_evidence"]["targets"] = build_scanner_targets(evidence)
    if enrich_etherscan:
        enrich_with_etherscan(
            evidence,
            max_transactions_pages=etherscan_pages,
            max_transaction_targets=etherscan_tx_limit,
        )
    if label_transactions_llm:
        attach_llm_transaction_labels(evidence)

    added_refs = ensure_reference_entities(evidence)
    matching_audit = attach_entity_address_mappings(evidence)
    missing_refs = validate_references(evidence)
    evidence["quality"] = {
        "entity_refs_added": added_refs,
        "missing_entity_refs": missing_refs,
        "entity_address_matching_status": matching_audit["status"],
        "status": (
            "ok"
            if not missing_refs and matching_audit["status"] == "ok"
            else "invalid_evidence"
        ),
    }

    return evidence


def resolve_summary_paths(case: str) -> list[Path]:
    if case == "all":
        return [
            SUMMARIZED_DIR / case_dir / "summary.json"
            for case_dir in discover_selected_cases(
                SUMMARIZED_DIR,
                required_files=("summary.json",),
            )
        ]

    case_path = SUMMARIZED_DIR / case / "summary.json"
    if case_path.exists():
        return [case_path]

    direct = Path(case)
    if direct.is_file():
        return [direct]

    raise FileNotFoundError(f"Cannot find summary for {case!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        nargs="?",
        default="all",
        help="Case directory under summarized/, path to summary.json, or 'all'.",
    )
    parser.add_argument(
        "--enrich-local-index",
        action="store_true",
        help="Query docs/eth_local_indexer.md HTTP API and attach sample lookup evidence.",
    )
    parser.add_argument(
        "--enrich-etherscan",
        action="store_true",
        help="Use the configured Etherscan fetch helper to fetch address evidence.",
    )
    parser.add_argument(
        "--label-transactions-llm",
        action="store_true",
        help="Use the configured OpenAI-compatible LLM endpoint to label transaction_evidence.",
    )
    parser.add_argument(
        "--eth-index-url",
        default=ETH_INDEX_URL,
        help=f"Ethereum local index base URL (default: {ETH_INDEX_URL}).",
    )
    parser.add_argument(
        "--neighbor-limit",
        type=int,
        default=10,
        help="Sample neighbor count per address when --enrich-local-index is enabled.",
    )
    parser.add_argument(
        "--etherscan-pages",
        type=int,
        default=1,
        help="Max Etherscan transaction pages per address when --enrich-etherscan is enabled.",
    )
    parser.add_argument(
        "--etherscan-tx-limit",
        type=int,
        default=25,
        help="Max transaction detail pages to fetch when --enrich-etherscan is enabled.",
    )
    args = parser.parse_args(argv)

    paths = resolve_summary_paths(args.case)
    if not paths:
        print("No summary.json files found.", file=sys.stderr)
        return 1

    for summary_path in paths:
        evidence = build_chain_evidence(
            summary_path,
            enrich_index=args.enrich_local_index,
            enrich_etherscan=args.enrich_etherscan,
            label_transactions_llm=args.label_transactions_llm,
            eth_index_url=args.eth_index_url,
            neighbor_limit=args.neighbor_limit,
            etherscan_pages=args.etherscan_pages,
            etherscan_tx_limit=args.etherscan_tx_limit,
        )
        out_path = summary_path.parent / "chain_evidence.json"
        write_json(out_path, evidence)
        print(out_path)
        audit = evidence["entity_address_matching"]
        print(
            "entity-address matching: "
            f"entities={audit['entity_count']} "
            f"mapped={audit['mapped_entity_count']} "
            f"unmapped={audit['unmapped_entity_count']} "
            f"addresses={audit['address_mapping_count']} "
            f"observed_edges={audit['matched_observed_edge_count']} "
            f"conflicts={len(audit['conflicts'])} "
            f"invalid={len(audit['invalid_addresses'])} "
            f"missing_evidence={len(audit['missing_evidence'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
