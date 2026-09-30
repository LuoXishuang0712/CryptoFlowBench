"""Convert private-chain transaction sheets into chain-link evidence reports.

Pipeline:
  1. Chain Construction  — parse Excel sheet, build transaction chains
  2. Chain Reporting     — enrich with TronScan + StarryGraph, emit JSON
  3. Chain Verification  — independent agent re-calls every node/edge

Usage:
    python tron_chain_evidence.py docs/sheet_example.xlsx
    python tron_chain_evidence.py docs/sheet_example.xlsx --enrich-tronscan
    python tron_chain_evidence.py docs/sheet_example.xlsx --enrich-tronscan --verify
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent

TRON_ADDRESS_RE = re.compile(r"^T[A-Za-z1-9]{33}$")
TRON_TX_HASH_RE = re.compile(r"^[a-f0-9]{64}$")

TRONSCAN_GATEWAY_URL = "http://127.0.0.1:5002"
STARRYGRAPH_URL = "http://127.0.0.1:5100"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
            os.environ.setdefault(key, value)


def is_tron_address(value: str) -> bool:
    return bool(TRON_ADDRESS_RE.fullmatch(str(value)))


def is_tron_tx_hash(value: str) -> bool:
    return bool(TRON_TX_HASH_RE.fullmatch(str(value)))


def tronscan_address_url(address: str) -> str:
    return f"https://tronscan.org/#/address/{address}"


def tronscan_tx_url(tx_hash: str) -> str:
    return f"https://tronscan.org/#/transaction/{tx_hash}"


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


def timestamp_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.isoformat()
    try:
        return dt.datetime.fromisoformat(str(value)).isoformat()
    except (ValueError, TypeError):
        return None


def timestamp_to_unix(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return int(value.timestamp())
    try:
        return int(dt.datetime.fromisoformat(str(value)).timestamp())
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Stage 1: Chain Construction
# ---------------------------------------------------------------------------


def read_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    col_map = {}
    for col in df.columns:
        low = str(col).lower().strip()
        if "transhash" in low or "tx" in low:
            col_map[col] = "tx_hash"
        elif "timestamp" in low or "time" in low:
            col_map[col] = "timestamp"
        elif col == "from":
            col_map[col] = "from_address"
        elif "from_tag" in low:
            col_map[col] = "from_tag"
        elif col == "to":
            col_map[col] = "to_address"
        elif "to_tag" in low:
            col_map[col] = "to_tag"
        elif "value" in low or "amount" in low:
            col_map[col] = "value"
    df = df.rename(columns=col_map)
    df = df.dropna(subset=["tx_hash", "from_address", "to_address"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def infer_address_roles(df: pd.DataFrame) -> dict[str, str]:
    roles: dict[str, str] = {}
    out_counts: dict[str, int] = {}
    in_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        src = str(row.get("from_address", ""))
        dst = str(row.get("to_address", ""))
        out_counts[src] = out_counts.get(src, 0) + 1
        in_counts[dst] = in_counts.get(dst, 0) + 1

    all_addresses = set(out_counts) | set(in_counts)
    source_only = {
        a
        for a in all_addresses
        if in_counts.get(a, 0) == 0 and out_counts.get(a, 0) > 0
    }
    sink_only = {
        a
        for a in all_addresses
        if out_counts.get(a, 0) == 0 and in_counts.get(a, 0) > 0
    }

    for addr in source_only:
        roles[addr] = "source_or_attacker"
    for addr in sink_only:
        roles[addr] = "sink_or_destination"
    for addr in all_addresses - source_only - sink_only:
        roles[addr] = "intermediary"

    for _, row in df.iterrows():
        src = str(row.get("from_address", ""))
        src_tag = str(row.get("from_tag", "") or "").strip()
        dst = str(row.get("to_address", ""))
        dst_tag = str(row.get("to_tag", "") or "").strip()
        if src_tag:
            roles[src] = _refine_role(roles.get(src, ""), src_tag)
        if dst_tag:
            roles[dst] = _refine_role(roles.get(dst, ""), dst_tag)

    return roles


def _refine_role(current: str, tag: str) -> str:
    tag_low = tag.lower()
    if any(k in tag_low for k in ("hack", "attack", "exploit", "phish", "steal")):
        return "attacker"
    if any(k in tag_low for k in ("victim", "target", "exploited")):
        return "victim"
    if any(k in tag_low for k in ("cex", "exchange", "binance", "okex", "huobi")):
        return "cex"
    if any(k in tag_low for k in ("mixer", "tornado")):
        return "mixer"
    if any(k in tag_low for k in ("dex", "swap", "uniswap", "justswap")):
        return "dex"
    if any(k in tag_low for k in ("bridge", "cross")):
        return "bridge"
    return current or "tagged_entity"


def build_chains_from_sheet(df: pd.DataFrame) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    visited_txs: set[str] = set()
    roles = infer_address_roles(df)

    adjacency: dict[str, list[dict[str, Any]]] = {}
    for idx, row in df.iterrows():
        src = str(row.get("from_address", ""))
        tx_hash = str(row.get("tx_hash", ""))
        adjacency.setdefault(src, []).append(row.to_dict())

    for idx, row in df.iterrows():
        src = str(row.get("from_address", ""))
        tx_hash = str(row.get("tx_hash", ""))
        if tx_hash in visited_txs:
            continue
        chain_edges: list[dict[str, Any]] = []
        current_addr = src
        current_tx = tx_hash
        local_visited: set[str] = set()

        while current_tx and current_tx not in local_visited:
            local_visited.add(current_tx)
            visited_txs.add(current_tx)
            row_match = df[df["tx_hash"] == current_tx]
            if row_match.empty:
                break
            edge_row = row_match.iloc[0]
            edge_src = str(edge_row.get("from_address", ""))
            edge_dst = str(edge_row.get("to_address", ""))
            chain_edges.append(
                {
                    "tx_hash": current_tx,
                    "from_address": edge_src,
                    "to_address": edge_dst,
                    "from_tag": str(edge_row.get("from_tag", "") or ""),
                    "to_tag": str(edge_row.get("to_tag", "") or ""),
                    "value": edge_row.get("value"),
                    "timestamp": edge_row.get("timestamp"),
                    "from_role": roles.get(edge_src, "unknown"),
                    "to_role": roles.get(edge_dst, "unknown"),
                }
            )
            next_edges = adjacency.get(edge_dst, [])
            next_edge = None
            for ne in next_edges:
                if str(ne.get("tx_hash", "")) not in local_visited:
                    next_edge = ne
                    break
            if next_edge:
                current_addr = edge_dst
                current_tx = str(next_edge.get("tx_hash", ""))
            else:
                current_tx = None

        if chain_edges:
            chains.append(
                {
                    "chain_id": f"chain_{len(chains):03d}",
                    "start_address": chain_edges[0]["from_address"],
                    "end_address": chain_edges[-1]["to_address"],
                    "length": len(chain_edges),
                    "edges": chain_edges,
                }
            )

    return chains


def build_entities_from_chains(
    chains: list[dict[str, Any]],
    roles: dict[str, str],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    entities: list[dict[str, Any]] = []

    for chain in chains:
        for edge in chain["edges"]:
            for addr_key, tag_key, direction in (
                ("from_address", "from_tag", "source"),
                ("to_address", "to_tag", "destination"),
            ):
                addr = edge.get(addr_key, "")
                if not addr or addr in seen:
                    continue
                seen.add(addr)
                tag = edge.get(tag_key, "")
                role = roles.get(addr, "unknown")
                entities.append(
                    {
                        "id": addr,
                        "type": "address",
                        "address": addr,
                        "chain": "Tron",
                        "role": role,
                        "entity": tag or None,
                        "confidence": "B" if role != "unknown" else "U",
                        "evidence": [],
                        "scanner_urls": {
                            "tronscan": tronscan_address_url(addr),
                        },
                    }
                )

    return entities


def build_edges_from_chains(
    chains: list[dict[str, Any]],
    roles: dict[str, str],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    edge_counter = 0

    for chain in chains:
        for i, chain_edge in enumerate(chain["edges"]):
            src = chain_edge["from_address"]
            dst = chain_edge["to_address"]
            src_role = roles.get(src, "unknown")
            dst_role = roles.get(dst, "unknown")
            stage = _infer_stage(src_role, dst_role, i, len(chain["edges"]))
            pattern = _infer_pattern(src_role, dst_role)
            edge_counter += 1
            edges.append(
                {
                    "id": f"edge:{chain['chain_id']}:{edge_counter:03d}",
                    "src": src,
                    "dst": dst,
                    "stage": stage,
                    "pattern": pattern,
                    "asset": "TRX",
                    "amount": str(chain_edge.get("value", "")),
                    "timestamp": timestamp_to_iso(chain_edge.get("timestamp")),
                    "tx_hash": chain_edge.get("tx_hash"),
                    "confidence": "B",
                    "evidence": [],
                    "verification_status": "sheet_derived_not_chain_verified",
                }
            )

    return edges


def _infer_stage(src_role: str, dst_role: str, position: int, total: int) -> str:
    if src_role in ("attacker", "source_or_attacker") and position == 0:
        return "theft"
    if dst_role in ("cex",):
        return "layering"
    if dst_role in ("mixer",):
        return "mixing"
    if dst_role in ("dex",):
        return "conversion"
    if dst_role in ("bridge",):
        return "chain_hopping"
    if position == total - 1:
        return "destination"
    if position > 0:
        return "layering"
    return "transfer"


def _infer_pattern(src_role: str, dst_role: str) -> str:
    if dst_role == "cex":
        return "cex_deposit"
    if dst_role == "mixer":
        return "mixer_deposit"
    if dst_role == "dex":
        return "dex_swap"
    if dst_role == "bridge":
        return "cross_chain_bridge"
    if src_role in ("attacker", "source_or_attacker"):
        return "attacker_outflow"
    return "transfer"


# ---------------------------------------------------------------------------
# Stage 2: Chain Reporting with TronScan + StarryGraph enrichment
# ---------------------------------------------------------------------------


class TronScanClient:
    def __init__(
        self, gateway_url: str = TRONSCAN_GATEWAY_URL, timeout: int = 300
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout

    def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        try:
            resp = requests.get(
                f"{self.gateway_url}{path}",
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("status") == "ok":
                return data.get("data", {})
            return data
        except Exception:
            return None

    def get_address_info(
        self, address: str, *, tx: bool = False, max_pages: int = 1
    ) -> dict[str, Any] | None:
        params: dict[str, Any] = {}
        if tx:
            params["tx"] = "true"
            params["max_pages"] = max_pages
        return self._get(f"/api/address/{address}", params)

    def get_transaction_info(self, tx_hash: str) -> dict[str, Any] | None:
        return self._get(f"/api/transaction/{tx_hash}")

    def get_transactions(
        self, address: str, max_pages: int = 1
    ) -> dict[str, Any] | None:
        return self._get(
            "/api/transactions", {"address": address, "max_pages": max_pages}
        )


class StarryGraphClient:
    def __init__(self, url: str = STARRYGRAPH_URL, timeout: int = 30) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _post(self, method: str, payload: dict[str, Any]) -> Any:
        try:
            resp = self.session.post(
                f"{self.url}/{method.lstrip('/')}",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, dict) and result.get("ok") is False:
                return None
            if isinstance(result, dict) and "ok" in result and "data" in result:
                return result["data"]
            return result
        except Exception:
            return None

    def exact_node_query(self, address: str) -> Any:
        return self._post("node/exact", {"address": address})

    def one_hop_transactions(
        self,
        address: str,
        direction: str = "OUT",
        limit_edges: int = 50,
    ) -> Any:
        return self._post(
            "transactions/one-hop",
            {
                "address": address,
                "direction": direction,
                "limit_edges": limit_edges,
                "get_loop": False,
                "edge_type": ["transaction"],
            },
        )


def enrich_entity_with_tronscan(
    entity: dict[str, Any],
    client: TronScanClient,
) -> dict[str, Any]:
    address = entity.get("address", "")
    if not address:
        return entity
    info = client.get_address_info(address, tx=False)
    if info:
        entity["tronscan_info"] = {
            "balance": info.get("balance"),
            "balance_str": info.get("balance_str"),
            "total_transaction_count": info.get("total_transaction_count"),
            "transactions_in": info.get("transactions_in"),
            "transactions_out": info.get("transactions_out"),
            "name": info.get("name"),
            "date_created": info.get("date_created"),
            "latest_operation_time": info.get("latest_operation_time"),
            "activated": info.get("activated"),
            "red_tag": info.get("red_tag"),
            "blue_tag": info.get("blue_tag"),
            "grey_tag": info.get("grey_tag"),
            "account_fetch_status": info.get("account_fetch_status"),
            "access_time": info.get("access_time"),
        }
        red_tag = info.get("red_tag", "")
        if red_tag:
            entity["role"] = "flagged_address"
            entity["confidence"] = "A"
            entity.setdefault("evidence", []).append("tronscan_red_tag")
        blue_tag = info.get("blue_tag", "")
        if blue_tag and not red_tag:
            entity["confidence"] = "B"
            entity.setdefault("evidence", []).append("tronscan_blue_tag")
        name = info.get("name", "")
        if name and not entity.get("entity"):
            entity["entity"] = name
    return entity


def enrich_entity_with_starrygraph(
    entity: dict[str, Any],
    client: StarryGraphClient,
) -> dict[str, Any]:
    address = entity.get("address", "")
    if not address:
        return entity
    node = client.exact_node_query(address)
    if node:
        entity["starrygraph_node"] = node
    return entity


def enrich_edge_with_tronscan(
    edge: dict[str, Any],
    client: TronScanClient,
) -> dict[str, Any]:
    tx_hash = edge.get("tx_hash", "")
    if not tx_hash:
        return edge
    info = client.get_transaction_info(tx_hash)
    if info:
        edge["tronscan_tx_info"] = {
            "hash": info.get("hash"),
            "block": info.get("block"),
            "timestamp": info.get("timestamp"),
            "owner_address": info.get("owner_address"),
            "to_address": info.get("to_address"),
            "contract_type": info.get("contract_type"),
            "confirmed": info.get("confirmed"),
            "revert": info.get("revert"),
            "cost": info.get("cost"),
            "fee_limit": info.get("fee_limit"),
            "risk_transaction": info.get("risk_transaction"),
            "trc20_transfers": info.get("trc20_transfers"),
            "token_transfer_info": info.get("token_transfer_info"),
            "transfers_all_list": info.get("transfers_all_list"),
            "trigger_info": info.get("trigger_info"),
            "contract_info": info.get("contract_info"),
            "access_time": info.get("access_time"),
        }
        if info.get("risk_transaction"):
            edge["confidence"] = "A"
            edge.setdefault("evidence", []).append("tronscan_risk_transaction_flag")
        if info.get("revert"):
            edge["verification_status"] = "tronscan_reverted"
        else:
            edge["verification_status"] = "tronscan_confirmed"
        if info.get("trc20_transfers"):
            edge["asset"] = "TRC20"
            edge["trc20_details"] = info["trc20_transfers"]
        if info.get("token_transfer_info"):
            edge["asset"] = "TRC10"
            edge["trc10_details"] = info["token_transfer_info"]
    return edge


def enrich_edge_with_starrygraph(
    edge: dict[str, Any],
    client: StarryGraphClient,
) -> dict[str, Any]:
    src = edge.get("src", "")
    if src:
        one_hop = client.one_hop_transactions(src, direction="OUT", limit_edges=20)
        if one_hop:
            edge["starrygraph_src_outgoing"] = one_hop
    return edge


def build_scanner_targets(
    entities: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    address_targets: list[dict[str, Any]] = []
    tx_targets: list[dict[str, Any]] = []

    for entity in entities:
        addr = entity.get("address", "")
        if addr:
            address_targets.append(
                {
                    "chain": "Tron",
                    "scanner": "tronscan",
                    "address": addr,
                    "role": entity.get("role"),
                    "url": tronscan_address_url(addr),
                }
            )

    for edge in edges:
        tx_hash = edge.get("tx_hash", "")
        if tx_hash:
            tx_targets.append(
                {
                    "chain": "Tron",
                    "scanner": "tronscan",
                    "tx_hash": tx_hash,
                    "role": edge.get("stage"),
                    "url": tronscan_tx_url(tx_hash),
                }
            )

    return {
        "address_targets": address_targets,
        "tx_targets": tx_targets,
    }


def build_tron_chain_evidence(
    sheet_path: Path,
    *,
    enrich_tronscan: bool = False,
    enrich_starrygraph: bool = False,
    tronscan_gateway_url: str = TRONSCAN_GATEWAY_URL,
    starrygraph_url: str = STARRYGRAPH_URL,
) -> dict[str, Any]:
    load_env_file(PROJECT_ROOT / ".env")

    df = read_sheet(sheet_path)
    chains = build_chains_from_sheet(df)
    roles = infer_address_roles(df)
    entities = build_entities_from_chains(chains, roles)
    edges = build_edges_from_chains(chains, roles)

    evidence: dict[str, Any] = {
        "schema_version": "tron_chain_evidence.v1",
        "source_sheet": str(sheet_path),
        "chain_type": "private_tron",
        "chains": ["Tron"],
        "total_transactions": len(df),
        "total_chains": len(chains),
        "time_window": _compute_time_window(df),
        "entities": entities,
        "edges": edges,
        "exploit_txs": [
            {
                "tx_hash": row.get("tx_hash"),
                "role": "exploit",
                "confidence": "B",
                "verification_status": "extracted_from_sheet",
            }
            for _, row in df.iterrows()
        ],
        "scanner_evidence": {
            "targets": build_scanner_targets(entities, edges),
            "tronscan": {"status": "not_fetched"},
            "starrygraph": {"status": "not_fetched"},
        },
        "skills": [
            "extract_transaction_chains_from_sheet",
            "infer_entity_roles_from_flow_topology",
            "separate_sheet_derived_from_chain_verified_edges",
            "assign_confidence_from_tronscan_tags",
        ],
        "negative_constraints": [
            "do_not_treat_sheet_addresses_as_verified_without_tronscan_confirmation",
            "do_not_infer_cex_account_ownership_without_external_evidence",
            "sheet_tags_are_hints_not_verified_labels",
        ],
        "open_questions": [],
        "generation": {
            "method": "sheet_to_tron_chain_evidence",
            "input": str(sheet_path),
        },
    }

    if enrich_tronscan:
        _enrich_all_tronscan(evidence, tronscan_gateway_url)

    if enrich_starrygraph:
        _enrich_all_starrygraph(evidence, starrygraph_url)

    added_refs = ensure_reference_entities(evidence)
    missing_refs = validate_references(evidence)
    evidence["quality"] = {
        "entity_refs_added": added_refs,
        "missing_entity_refs": missing_refs,
        "status": "ok" if not missing_refs else "missing_references",
    }

    return evidence


def _compute_time_window(df: pd.DataFrame) -> dict[str, Any]:
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        return {"attack_start": None, "last_observed": None}
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    return {
        "attack_start": timestamp_to_iso(ts_min),
        "last_observed": timestamp_to_iso(ts_max),
    }


def _enrich_all_tronscan(evidence: dict[str, Any], gateway_url: str) -> None:
    client = TronScanClient(gateway_url)
    tronscan_results: list[dict[str, Any]] = []

    for entity in evidence.get("entities", []):
        addr = entity.get("address", "")
        if not addr:
            continue
        info = client.get_address_info(addr, tx=False)
        result = {"address": addr, "address_info": info}
        if info:
            enrich_entity_with_tronscan(entity, client)
        tronscan_results.append(result)

    for edge in evidence.get("edges", []):
        tx_hash = edge.get("tx_hash", "")
        if not tx_hash:
            continue
        tx_info = client.get_transaction_info(tx_hash)
        result = {"tx_hash": tx_hash, "transaction_info": tx_info}
        if tx_info:
            enrich_edge_with_tronscan(edge, client)
        tronscan_results.append(result)

    evidence["scanner_evidence"]["tronscan"] = {
        "status": "available",
        "gateway_url": gateway_url,
        "results": tronscan_results,
    }


def _enrich_all_starrygraph(evidence: dict[str, Any], sg_url: str) -> None:
    client = StarryGraphClient(sg_url)
    sg_results: list[dict[str, Any]] = []

    for entity in evidence.get("entities", []):
        addr = entity.get("address", "")
        if not addr:
            continue
        node = client.exact_node_query(addr)
        result = {"address": addr, "node": node}
        if node:
            enrich_entity_with_starrygraph(entity, client)
        sg_results.append(result)

    for edge in evidence.get("edges", []):
        src = edge.get("src", "")
        if not src:
            continue
        one_hop = client.one_hop_transactions(src, direction="OUT", limit_edges=20)
        result = {"src_address": src, "outgoing": one_hop}
        if one_hop:
            enrich_edge_with_starrygraph(edge, client)
        sg_results.append(result)

    evidence["scanner_evidence"]["starrygraph"] = {
        "status": "available",
        "url": sg_url,
        "results": sg_results,
    }


def ensure_reference_entities(evidence: dict[str, Any]) -> list[str]:
    entities = evidence.setdefault("entities", [])
    existing_ids = {entity.get("id") for entity in entities}
    added: list[str] = []
    for edge in evidence.get("edges", []):
        for field in ("src", "dst"):
            ref = edge.get(field)
            if not isinstance(ref, str) or not ref or ref in existing_ids:
                continue
            entity: dict[str, Any] = {
                "id": ref,
                "type": "address",
                "address": ref,
                "chain": "Tron",
                "role": "edge_endpoint",
                "confidence": edge.get("confidence", "U"),
                "evidence": edge.get("evidence", []),
                "scanner_urls": {"tronscan": tronscan_address_url(ref)},
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


# ---------------------------------------------------------------------------
# Stage 3: Verification Agent
# ---------------------------------------------------------------------------


def verify_chain_evidence(
    evidence: dict[str, Any],
    *,
    tronscan_gateway_url: str = TRONSCAN_GATEWAY_URL,
    starrygraph_url: str = STARRYGRAPH_URL,
) -> dict[str, Any]:
    """Independent verification: re-call TronScan + StarryGraph for every entity and edge."""
    client_ts = TronScanClient(tronscan_gateway_url)
    client_sg = StarryGraphClient(starrygraph_url)

    verification: dict[str, Any] = {
        "schema_version": "tron_verification.v1",
        "source_evidence_id": evidence.get("schema_version"),
        "entity_verifications": [],
        "edge_verifications": [],
        "discrepancies": [],
        "summary": {
            "entities_checked": 0,
            "entities_confirmed": 0,
            "entities_failed": 0,
            "edges_checked": 0,
            "edges_confirmed": 0,
            "edges_failed": 0,
            "discrepancy_count": 0,
        },
    }

    for entity in evidence.get("entities", []):
        addr = entity.get("address", "")
        if not addr:
            continue
        verification["summary"]["entities_checked"] += 1
        v_result = _verify_entity(entity, client_ts, client_sg)
        verification["entity_verifications"].append(v_result)
        if v_result["status"] == "confirmed":
            verification["summary"]["entities_confirmed"] += 1
        else:
            verification["summary"]["entities_failed"] += 1
        for disc in v_result.get("discrepancies", []):
            verification["discrepancies"].append(disc)
            verification["summary"]["discrepancy_count"] += 1

    for edge in evidence.get("edges", []):
        tx_hash = edge.get("tx_hash", "")
        src = edge.get("src", "")
        if not tx_hash and not src:
            continue
        verification["summary"]["edges_checked"] += 1
        v_result = _verify_edge(edge, client_ts, client_sg)
        verification["edge_verifications"].append(v_result)
        if v_result["status"] == "confirmed":
            verification["summary"]["edges_confirmed"] += 1
        else:
            verification["summary"]["edges_failed"] += 1
        for disc in v_result.get("discrepancies", []):
            verification["discrepancies"].append(disc)
            verification["summary"]["discrepancy_count"] += 1

    return verification


def _verify_entity(
    entity: dict[str, Any],
    client_ts: TronScanClient,
    client_sg: StarryGraphClient,
) -> dict[str, Any]:
    addr = entity.get("address", "")
    discrepancies: list[dict[str, Any]] = []
    ts_info = client_ts.get_address_info(addr, tx=False)
    sg_node = client_sg.exact_node_query(addr)

    ts_ok = ts_info is not None and ts_info.get("account_fetch_status") != "error"
    sg_ok = sg_node is not None

    if ts_info and ts_info.get("account_fetch_status") != "error":
        existing_role = entity.get("role", "")
        red_tag = ts_info.get("red_tag", "")
        if red_tag and existing_role not in ("flagged_address", "attacker"):
            discrepancies.append(
                {
                    "type": "role_mismatch",
                    "entity": addr,
                    "field": "role",
                    "expected": "flagged_address",
                    "actual": existing_role,
                    "source": "tronscan_red_tag",
                    "detail": red_tag,
                }
            )

        existing_balance = entity.get("tronscan_info", {}).get("balance")
        fresh_balance = ts_info.get("balance")
        if (
            existing_balance is not None
            and fresh_balance is not None
            and existing_balance != fresh_balance
        ):
            discrepancies.append(
                {
                    "type": "value_changed",
                    "entity": addr,
                    "field": "balance",
                    "previous": existing_balance,
                    "current": fresh_balance,
                    "source": "tronscan",
                }
            )

    return {
        "address": addr,
        "status": "confirmed"
        if (ts_ok or sg_ok) and not discrepancies
        else ("discrepancy" if discrepancies else "unverifiable"),
        "tronscan_fresh": {
            "fetched": ts_info is not None,
            "account_fetch_status": ts_info.get("account_fetch_status")
            if ts_info
            else None,
            "balance": ts_info.get("balance") if ts_info else None,
            "red_tag": ts_info.get("red_tag") if ts_info else None,
            "blue_tag": ts_info.get("blue_tag") if ts_info else None,
            "name": ts_info.get("name") if ts_info else None,
        }
        if ts_info
        else None,
        "starrygraph_fresh": {
            "fetched": sg_ok,
            "node": sg_node,
        },
        "discrepancies": discrepancies,
    }


def _verify_edge(
    edge: dict[str, Any],
    client_ts: TronScanClient,
    client_sg: StarryGraphClient,
) -> dict[str, Any]:
    tx_hash = edge.get("tx_hash", "")
    src = edge.get("src", "")
    discrepancies: list[dict[str, Any]] = []

    ts_tx = None
    if tx_hash:
        ts_tx = client_ts.get_transaction_info(tx_hash)

    sg_out = None
    if src:
        sg_out = client_sg.one_hop_transactions(src, direction="OUT", limit_edges=50)

    confirmed = False

    if ts_tx:
        ts_owner = str(ts_tx.get("owner_address", ""))
        ts_to = str(ts_tx.get("to_address", ""))
        if ts_owner and src and ts_owner != src:
            discrepancies.append(
                {
                    "type": "src_mismatch",
                    "edge": edge.get("id"),
                    "field": "src",
                    "expected": src,
                    "actual": ts_owner,
                    "source": "tronscan_transaction",
                }
            )
        edge_dst = edge.get("dst", "")
        if ts_to and edge_dst and ts_to != edge_dst:
            discrepancies.append(
                {
                    "type": "dst_mismatch",
                    "edge": edge.get("id"),
                    "field": "dst",
                    "expected": edge_dst,
                    "actual": ts_to,
                    "source": "tronscan_transaction",
                }
            )

        if ts_tx.get("revert"):
            discrepancies.append(
                {
                    "type": "reverted_transaction",
                    "edge": edge.get("id"),
                    "tx_hash": tx_hash,
                    "source": "tronscan_transaction",
                }
            )
        elif not discrepancies:
            confirmed = True

        ts_value = ts_tx.get("cost", {})
        if isinstance(ts_value, dict):
            edge_amount = edge.get("amount")
            if edge_amount and ts_value:
                pass

    if sg_out and not confirmed:
        if isinstance(sg_out, dict):
            edges_data = sg_out.get("edges", [])
            if isinstance(edges_data, list):
                found = any(
                    str(e.get("tx_hash", "")) == tx_hash
                    for e in edges_data
                    if isinstance(e, dict)
                )
                if found and not discrepancies:
                    confirmed = True
                elif tx_hash and not found:
                    discrepancies.append(
                        {
                            "type": "tx_not_found_in_starrygraph_outgoing",
                            "edge": edge.get("id"),
                            "tx_hash": tx_hash,
                            "src": src,
                            "source": "starrygraph",
                        }
                    )

    return {
        "edge_id": edge.get("id"),
        "tx_hash": tx_hash,
        "status": "confirmed"
        if confirmed and not discrepancies
        else ("discrepancy" if discrepancies else "unverifiable"),
        "tronscan_fresh": {
            "fetched": ts_tx is not None,
            "confirmed": ts_tx.get("confirmed") if ts_tx else None,
            "revert": ts_tx.get("revert") if ts_tx else None,
            "risk_transaction": ts_tx.get("risk_transaction") if ts_tx else None,
            "owner_address": ts_tx.get("owner_address") if ts_tx else None,
            "to_address": ts_tx.get("to_address") if ts_tx else None,
        }
        if ts_tx
        else None,
        "starrygraph_fresh": {
            "fetched": sg_out is not None,
            "edge_count": len(sg_out.get("edges", []))
            if isinstance(sg_out, dict)
            else None,
        }
        if sg_out
        else None,
        "discrepancies": discrepancies,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sheet",
        type=Path,
        help="Path to the Excel sheet containing transaction data.",
    )
    parser.add_argument(
        "--enrich-tronscan",
        action="store_true",
        help="Fetch TronScan address/transaction info via the gateway API.",
    )
    parser.add_argument(
        "--enrich-starrygraph",
        action="store_true",
        help="Fetch StarryGraph node/edge info for each entity.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run independent verification agent that re-calls every node/edge.",
    )
    parser.add_argument(
        "--tronscan-gateway-url",
        default=TRONSCAN_GATEWAY_URL,
        help=f"TronScan gateway base URL (default: {TRONSCAN_GATEWAY_URL}).",
    )
    parser.add_argument(
        "--starrygraph-url",
        default=STARRYGRAPH_URL,
        help=f"StarryGraph base URL (default: {STARRYGRAPH_URL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for chain_evidence.json (default: same dir as sheet).",
    )
    args = parser.parse_args(argv)

    if not args.sheet.exists():
        print(f"Sheet not found: {args.sheet}", file=sys.stderr)
        return 1

    print(f"[Stage 1] Building chains from {args.sheet} ...")
    evidence = build_tron_chain_evidence(
        args.sheet,
        enrich_tronscan=args.enrich_tronscan,
        enrich_starrygraph=args.enrich_starrygraph,
        tronscan_gateway_url=args.tronscan_gateway_url,
        starrygraph_url=args.starrygraph_url,
    )
    print(
        f"  -> {evidence['total_chains']} chains, {evidence['total_transactions']} transactions, "
        f"{len(evidence['entities'])} entities, {len(evidence['edges'])} edges"
    )

    if args.enrich_tronscan:
        print("[Stage 2] TronScan enrichment complete.")
    if args.enrich_starrygraph:
        print("[Stage 2] StarryGraph enrichment complete.")

    if args.verify:
        print("[Stage 3] Running verification agent ...")
        verification = verify_chain_evidence(
            evidence,
            tronscan_gateway_url=args.tronscan_gateway_url,
            starrygraph_url=args.starrygraph_url,
        )
        evidence["verification"] = verification
        s = verification["summary"]
        print(
            f"  -> Entities: {s['entities_confirmed']}/{s['entities_checked']} confirmed, "
            f"{s['entities_failed']} failed"
        )
        print(
            f"  -> Edges: {s['edges_confirmed']}/{s['edges_checked']} confirmed, "
            f"{s['edges_failed']} failed"
        )
        print(f"  -> Discrepancies: {s['discrepancy_count']}")

    out_path = args.output or args.sheet.parent / "tron_chain_evidence.json"
    write_json(out_path, evidence)
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
