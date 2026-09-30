from __future__ import annotations

import re
import sys
import time
from typing import Any

import requests

from .schema import normalize_address


class EthIndexError(RuntimeError):
    pass


class EthIndexAddressNotIndexed(EthIndexError):
    """The index explicitly has no node record for the requested address."""


class EthIndexContractError(EthIndexError):
    """The index response violates a required client-side data contract."""


class EthIndexSeedNotIndexed(EthIndexError):
    def __init__(self, message: str, address: str) -> None:
        super().__init__(message)
        self.address = address


class EthIndexClient:
    """HTTP adapter for the local Ethereum index.

    Edge hydration always uses with_raw=true, because chain QA needs transaction
    fields for reproducible edge-table exports and label debugging.
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 60,
        *,
        connect_timeout: int = 5,
        max_retries: int = 1,
        retry_backoff: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.max_retries = max(0, max_retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.session = requests.Session()
        self.session.trust_env = False

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method,
                    f"{self.base_url}{path}",
                    timeout=(self.connect_timeout, self.timeout),
                    **kwargs,
                )
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.max_retries:
                    raise
                delay = self.retry_backoff * (2**attempt)
                print(
                    f"eth_index retry {attempt + 1}/{self.max_retries}: "
                    f"{method} {path} after {type(exc).__name__}; "
                    f"params={kwargs.get('params')} sleep={delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                if delay:
                    time.sleep(delay)
        if response is None:
            raise EthIndexError(f"{method} {path} returned no response")
        try:
            data = response.json()
        except ValueError:
            data = {"text": response.text[:2000]}
        if response.status_code == 404 and isinstance(data, dict):
            detail = str(data.get("detail") or "").strip().lower()
            if detail == "address not indexed":
                raise EthIndexAddressNotIndexed(
                    f"{method} {path} failed: 404 {data}"
                )
            if detail.startswith("seed address not indexed:"):
                match = re.search(r"0x[0-9a-f]{40}", detail)
                if match:
                    raise EthIndexSeedNotIndexed(
                        f"{method} {path} failed: 404 {data}", match.group(0)
                    )
        if response.status_code >= 400:
            raise EthIndexError(f"{method} {path} failed: {response.status_code} {data}")
        return data

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "EthIndexClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def health(self) -> Any:
        return self.request("GET", "/health")

    def neighbors(
        self,
        address: str,
        *,
        direction: str = "both",
        limit: int = 500,
        block_min: int | None = None,
        block_max: int | None = None,
        edge_type: int | None = None,
        allow_unindexed: bool = True,
    ) -> list[dict[str, Any]]:
        norm = normalize_address(address)
        if not norm:
            return []
        params: dict[str, Any] = {
            "direction": direction,
            "limit": max(1, min(limit, 500_001)),
            "with_raw": "true",
        }
        if block_min is not None:
            params["block_min"] = block_min
        if block_max is not None:
            params["block_max"] = block_max
        if edge_type is not None:
            params["edge_type"] = edge_type
        try:
            data = self.request("GET", f"/neighbors/{norm}", params=params)
        except EthIndexAddressNotIndexed:
            if allow_unindexed:
                return []
            raise
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            rows = data.get("edges") or data.get("neighbors") or data.get("data") or []
            return [row for row in rows if isinstance(row, dict)]
        return []

    def complete_neighbors(
        self,
        address: str,
        *,
        direction: str,
        block_min: int,
        block_max: int,
        edge_type: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return neighbors and whether the result is provably below the cap."""
        rows = self.neighbors(
            address,
            direction=direction,
            limit=limit + 1,
            block_min=block_min,
            block_max=block_max,
            edge_type=edge_type,
            allow_unindexed=False,
        )
        if len(rows) > limit:
            return rows[:limit], False
        return rows, True

    def expand(
        self,
        seeds: list[str],
        *,
        k: int = 2,
        direction: str = "both",
        max_nodes: int = 800,
        max_edges: int = 2500,
        block_min: int | None = None,
        block_max: int | None = None,
        edge_types: list[int] | None = None,
    ) -> dict[str, Any]:
        body = {
            "seeds": [node for node in (normalize_address(seed) for seed in seeds) if node],
            "k": max(1, min(k, 4)),
            "direction": direction,
            "max_nodes": max(1, min(max_nodes, 100000)),
            "max_edges": max(1, min(max_edges, 500000)),
            "block_min": block_min,
            "block_max": block_max,
            "edge_types": edge_types,
        }
        skipped_seeds: list[str] = []
        while body["seeds"]:
            try:
                data = self.request("POST", "/expand", json=body)
                break
            except EthIndexSeedNotIndexed as exc:
                if exc.address not in body["seeds"]:
                    raise
                skipped_seeds.append(exc.address)
                body["seeds"] = [
                    seed for seed in body["seeds"] if seed != exc.address
                ]
                print(
                    f"eth_index expand skipped unindexed seed: {exc.address}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            data = {"nodes": [], "edges": []}
        result = data if isinstance(data, dict) else {"edges": data}
        result["unindexed_seeds"] = skipped_seeds
        rows = result.get("edges")
        nested = result.get("data")
        if rows is None and isinstance(nested, dict):
            rows = nested.get("edges")
        rows = rows if isinstance(rows, list) else []
        invalid = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                invalid.append(f"row:{index}")
                continue
            hop = row.get("hop")
            if (
                not isinstance(hop, int)
                or isinstance(hop, bool)
                or hop < 1
                or hop > body["k"]
            ):
                invalid.append(row.get("edge_id") or row.get("id") or f"row:{index}")
        if invalid:
            raise EthIndexContractError(
                "POST /expand returned edges with missing or invalid hop: "
                f"count={len(invalid)} k={body['k']} sample={invalid[:10]}"
            )
        return result

    def edge(self, edge_id: int | str) -> dict[str, Any] | None:
        try:
            data = self.request(
                "GET", f"/edge/{edge_id}", params={"with_raw": "true"}
            )
        except EthIndexError:
            return None
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
            return data["data"]
        return data if isinstance(data, dict) else None
