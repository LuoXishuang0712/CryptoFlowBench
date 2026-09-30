from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARIZED_DIR = PROJECT_ROOT / "summarized"
OUTPUT_DIR = PROJECT_ROOT / "chain_qa"
ENV_FILE = PROJECT_ROOT / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        out: list[str] = []
        in_quote = False
        quote_ch = ""
        for ch in line:
            if ch in "\"'":
                if not in_quote:
                    in_quote = True
                    quote_ch = ch
                elif quote_ch == ch:
                    in_quote = False
            if ch == "#" and not in_quote:
                break
            out.append(ch)
        key, value = "".join(out).split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


load_env()


@dataclass(frozen=True)
class ChainQAConfig:
    eth_index_url: str
    output_dir: Path = OUTPUT_DIR
    k_hop: int = 2
    direction: str = "both"
    max_nodes: int = 800
    max_edges: int = 2500
    neighbor_limit: int = 500
    neg_ratio: int = 8
    transaction_neg_ratio: int = 1
    transaction_query_limit: int = 100_000
    skip_dte: bool = False
    tracing_agent_profile: str = "tool_grounded"
    max_candidates: int = 12
    block_padding: int = 120_000
    dataset_version: str = "chain_qa.v3_1"
    request_timeout: int = 60
    connect_timeout: int = 5
    request_retries: int = 1
    retry_backoff: float = 1.0
    show_progress: bool = True

    @classmethod
    def from_env(cls, **overrides: Any) -> "ChainQAConfig":
        values = {
            "eth_index_url": os.environ.get("ETH_INDEX_URL", "http://127.0.0.1:8000"),
            "output_dir": OUTPUT_DIR,
            "k_hop": int(os.environ.get("CHAIN_QA_K_HOP", "2")),
            "direction": os.environ.get("CHAIN_QA_DIRECTION", "both"),
            "max_nodes": int(os.environ.get("CHAIN_QA_MAX_NODES", "800")),
            "max_edges": int(os.environ.get("CHAIN_QA_MAX_EDGES", "2500")),
            "neighbor_limit": int(os.environ.get("CHAIN_QA_NEIGHBOR_LIMIT", "500")),
            "neg_ratio": int(os.environ.get("CHAIN_QA_NEG_RATIO", "8")),
            "transaction_neg_ratio": int(
                os.environ.get("CHAIN_QA_TRANSACTION_NEG_RATIO", "1")
            ),
            "transaction_query_limit": int(
                os.environ.get("CHAIN_QA_TRANSACTION_QUERY_LIMIT", "100000")
            ),
            "tracing_agent_profile": os.environ.get(
                "CHAIN_QA_TRACING_AGENT_PROFILE", "tool_grounded"
            ),
            "max_candidates": int(os.environ.get("CHAIN_QA_MAX_CANDIDATES", "12")),
            "block_padding": int(os.environ.get("CHAIN_QA_BLOCK_PADDING", "120000")),
            "dataset_version": os.environ.get("CHAIN_QA_DATASET_VERSION", "chain_qa.v3_1"),
            "request_timeout": int(os.environ.get("CHAIN_QA_TIMEOUT", "60")),
            "connect_timeout": int(
                os.environ.get("CHAIN_QA_CONNECT_TIMEOUT", "5")
            ),
            "request_retries": int(
                os.environ.get("CHAIN_QA_REQUEST_RETRIES", "1")
            ),
            "retry_backoff": float(
                os.environ.get("CHAIN_QA_RETRY_BACKOFF", "1.0")
            ),
            "show_progress": os.environ.get(
                "CHAIN_QA_SHOW_PROGRESS", "true"
            ).lower()
            not in {"0", "false", "no"},
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)
