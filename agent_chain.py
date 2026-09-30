from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlencode, urlparse

import requests

from chain_profile import resolve_tasks_for_agent_profiles, task_agent_profile
from comp_exam.impl.detector_runtime import (
    DEFAULT_DETECTOR_TOOL_METHODS,
    DetectorToolRuntime,
    normalize_detector_tool_methods,
)

try:
    import litellm
    # litellm._turn_on_debug()
except ModuleNotFoundError:  # Allows --retrieve-only without the LLM dependency installed.
    litellm = None


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
SUMMARIZED_DIR = PROJECT_ROOT / "summarized"

BROWSER_BASE = os.environ.get("BROWSER_BASE", "http://127.0.0.1:5010")
ETH_INDEX_URL = os.environ.get("ETH_INDEX_URL", "http://127.0.0.1:8000")

LLM_TIMEOUT = 600
MAX_ITERATIONS = 12
CHAIN_MAX_TOOL_ITERATIONS = int(os.environ.get("CHAIN_MAX_TOOL_ITERATIONS", "4"))
CHAIN_MAX_TOOL_COLLECTION_TURNS = int(
    os.environ.get("CHAIN_MAX_TOOL_COLLECTION_TURNS", "64")
)
CHAIN_TRANSACTION_QUERY_LIMIT = int(
    os.environ.get("CHAIN_QA_TRANSACTION_QUERY_LIMIT", "100000")
)
CHAIN_DETECTOR_TOOL_RESULT_CHAR_LIMIT = int(
    os.environ.get("CHAIN_DETECTOR_TOOL_RESULT_CHAR_LIMIT", "60000")
)

ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{40}")
TX_RE = re.compile(r"0x[a-fA-F0-9]{64}")
BROWSER_TOOL_LOCK = threading.RLock()


def load_env(path: Path = ENV_PATH) -> None:
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

LLM_PROVIDER_URL = os.environ.get("LLM_PROVIDER_URL", "http://127.0.0.1:18080/v1")
LLM_NAME = os.environ.get("LLM_NAME", "Qwen/Qwen3.6-27B")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "NO_API_KEY")
JUDGE_LLM_NAME = os.environ.get("JUDGE_LLM_NAME", LLM_NAME)
LLM_THINK = os.environ.get("LLM_THINK", "False").strip().lower() in {"1", "true", "yes"}
LLM_OUTPUT_LENGTH = int(os.environ.get("LLM_OUTPUT_LENGTH", "20000"))
CHAIN_LLM_OUTPUT_LENGTH = int(
    os.environ.get("CHAIN_LLM_OUTPUT_LENGTH", str(min(LLM_OUTPUT_LENGTH, 20000)))
)
CHAIN_TOOL_RESULT_CHAR_LIMIT = int(
    os.environ.get("CHAIN_TOOL_RESULT_CHAR_LIMIT", "4000")
)
CHAIN_CONTEXT_CHAR_LIMIT = int(os.environ.get("CHAIN_CONTEXT_CHAR_LIMIT", "60000"))


def truncate(value: Any, limit: int = 4000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def compact_json(value: Any, limit: int = 6000) -> str:
    return truncate(json.dumps(value, ensure_ascii=False, indent=2, default=str), limit)


def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = re.sub(r"^www\.", "", parsed.netloc.lower())
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme.lower()}://{host}{path}"
    except Exception:
        return url.lower().strip()


def short_id(value: Any) -> str:
    text = str(value)
    if ADDRESS_RE.fullmatch(text):
        return f"{text[:8]}...{text[-6:]}"
    if TX_RE.fullmatch(text):
        return f"{text[:10]}...{text[-8:]}"
    return text


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"0x[a-f0-9]{40,64}|[a-z0-9_:-]{2,}|[\u4e00-\u9fff]", lowered)
    cjk = re.findall(r"[\u4e00-\u9fff]+", lowered)
    bigrams: list[str] = []
    for chunk in cjk:
        bigrams.extend(chunk[i : i + 2] for i in range(max(0, len(chunk) - 1)))
    return words + bigrams


@dataclass
class EvidenceDoc:
    doc: dict[str, Any]
    text: str
    tokens: Counter[str]


@dataclass
class AgentResult:
    question: str
    answer: str
    evidence: list[dict[str, Any]]
    case: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence": self.evidence,
            "case": self.case,
            "error": self.error,
        }


class EvidenceStore:
    """Retrieval over source evidence, never over generated test-set artifacts."""

    def __init__(self, summarized_dir: Path = SUMMARIZED_DIR) -> None:
        self.summarized_dir = summarized_dir
        self.docs: list[EvidenceDoc] = []
        self.doc_freq: Counter[str] = Counter()
        self._load()

    def _load(self) -> None:
        if not self.summarized_dir.exists():
            return
        for case_path in sorted(self.summarized_dir.iterdir()):
            if not case_path.is_dir() or case_path.name.startswith("_") or case_path.name.startswith("xx-"):
                continue
            summary_path = case_path / "summary.json"
            evidence_path = case_path / "chain_evidence.json"
            if not summary_path.exists() and not evidence_path.exists():
                continue
            summary = self._read_json(summary_path) if summary_path.exists() else {}
            evidence = self._read_json(evidence_path) if evidence_path.exists() else {}
            self._add_case_docs(case_path.name, summary, evidence, summary_path, evidence_path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _source_lookup(summary: dict[str, Any], evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for source in evidence.get("sources") or []:
            if isinstance(source, dict) and source.get("id"):
                out[str(source["id"])] = source
        for idx, source in enumerate(summary.get("sources_used") or [], start=1):
            if isinstance(source, dict):
                source_id = str(source.get("id") or f"summary:S{idx:02d}")
                out.setdefault(source_id, source | {"id": source_id})
        return out

    @staticmethod
    def _refs_for(item: dict[str, Any], sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        refs = item.get("evidence") or item.get("sources") or []
        if isinstance(refs, str):
            refs = [refs]
        out = []
        for ref in refs if isinstance(refs, list) else []:
            source = sources.get(str(ref))
            if source:
                out.append(
                    {
                        "id": str(ref),
                        "entity": source.get("entity") or source.get("source_entity"),
                        "grade": source.get("grade") or source.get("source_grade"),
                        "url": source.get("url") or source.get("source"),
                    }
                )
            else:
                out.append({"id": str(ref)})
        return out

    def _add_doc(
        self,
        *,
        case_dir: str,
        case_name: str,
        kind: str,
        title: str,
        content: Any,
        data: Any,
        source_path: Path,
        supporting_sources: list[dict[str, Any]] | None = None,
    ) -> None:
        doc_id = f"{case_dir}:{kind}:{len(self.docs) + 1:05d}"
        doc = {
            "id": doc_id,
            "case_dir": case_dir,
            "case_name": case_name,
            "kind": kind,
            "title": title,
            "content": content,
            "data": data,
            "supporting_sources": supporting_sources or [],
            "source_file": str(source_path.relative_to(PROJECT_ROOT)),
        }
        text = self._doc_text(doc)
        tokens = Counter(tokenize(text))
        self.docs.append(EvidenceDoc(doc=doc, text=text, tokens=tokens))
        self.doc_freq.update(tokens.keys())

    def _add_case_docs(
        self,
        case_dir: str,
        summary: dict[str, Any],
        evidence: dict[str, Any],
        summary_path: Path,
        evidence_path: Path,
    ) -> None:
        case_name = str(evidence.get("case_name") or summary.get("case_name") or case_dir)
        summary_body = summary.get("summary") if isinstance(summary.get("summary"), dict) else summary
        sources = self._source_lookup(summary, evidence)

        if summary_body:
            self._add_doc(
                case_dir=case_dir,
                case_name=case_name,
                kind="summary",
                title=f"{case_name} summary",
                content=summary_body,
                data=summary_body,
                source_path=summary_path,
            )
            for field, value in summary_body.items():
                if value in (None, "", [], {}):
                    continue
                self._add_doc(
                    case_dir=case_dir,
                    case_name=case_name,
                    kind=f"summary.{field}",
                    title=f"{case_name} {field}",
                    content={field: value},
                    data={field: value},
                    source_path=summary_path,
                )

        if evidence:
            profile_keys = [
                "schema_version",
                "case_id",
                "case_name",
                "incident_type",
                "chains",
                "protocol_type",
                "time_window",
                "loss",
                "root_cause",
                "negative_constraints",
                "open_questions",
            ]
            profile = {key: evidence.get(key) for key in profile_keys if evidence.get(key) not in (None, "", [], {})}
            self._add_doc(
                case_dir=case_dir,
                case_name=case_name,
                kind="chain_evidence.profile",
                title=f"{case_name} chain evidence profile",
                content=profile,
                data=profile,
                source_path=evidence_path,
            )

        for source in evidence.get("sources") or []:
            if isinstance(source, dict):
                self._add_doc(
                    case_dir=case_dir,
                    case_name=case_name,
                    kind="source",
                    title=f"{case_name} source {source.get('id') or source.get('entity')}",
                    content=source,
                    data=source,
                    source_path=evidence_path,
                    supporting_sources=[{
                        "id": source.get("id"),
                        "entity": source.get("entity"),
                        "grade": source.get("grade"),
                        "url": source.get("url"),
                    }],
                )

        for idx, source in enumerate(summary.get("sources_used") or [], start=1):
            if isinstance(source, dict):
                self._add_doc(
                    case_dir=case_dir,
                    case_name=case_name,
                    kind="summary_source",
                    title=f"{case_name} summary source {idx}",
                    content=source,
                    data=source,
                    source_path=summary_path,
                    supporting_sources=[{
                        "id": source.get("id") or f"summary:S{idx:02d}",
                        "entity": source.get("source_entity"),
                        "grade": source.get("source_grade"),
                        "url": source.get("source"),
                    }],
                )

        for section in ("entities", "edges", "transaction_evidence", "scanner_evidence"):
            items = evidence.get(section) or []
            if isinstance(items, dict):
                items = [items]
            for item in items:
                if not isinstance(item, dict):
                    continue
                label = item.get("id") or item.get("tx_hash") or item.get("address") or item.get("entity") or section
                self._add_doc(
                    case_dir=case_dir,
                    case_name=case_name,
                    kind=f"chain_evidence.{section}",
                    title=f"{case_name} {section} {label}",
                    content=item,
                    data=item,
                    source_path=evidence_path,
                    supporting_sources=self._refs_for(item, sources),
                )

    @staticmethod
    def _doc_text(doc: dict[str, Any]) -> str:
        fields = [
            doc.get("case_dir"),
            doc.get("case_name"),
            doc.get("kind"),
            doc.get("title"),
            doc.get("content"),
            doc.get("supporting_sources"),
        ]
        return "\n".join(str(item) for item in fields if item not in (None, "", [], {}))

    def search(self, query: str, *, case: str | None = None, top_k: int = 8) -> list[dict[str, Any]]:
        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            return []
        total = max(1, len(self.docs))
        scored: list[tuple[float, EvidenceDoc]] = []
        case_l = case.lower() if case else ""
        for item in self.docs:
            doc_case = str(item.doc.get("case_dir") or "").lower()
            doc_name = str(item.doc.get("case_name") or "").lower()
            if case_l and case_l not in doc_case and case_l not in doc_name:
                continue
            score = 0.0
            for token, q_tf in query_tokens.items():
                tf = item.tokens.get(token, 0)
                if not tf:
                    continue
                idf = math.log((total + 1) / (self.doc_freq[token] + 0.5))
                score += (1 + math.log(tf)) * (1 + math.log(q_tf)) * max(idf, 0.1)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        out = []
        for score, item in scored[: max(1, min(top_k, 50))]:
            doc = item.doc
            out.append(
                {
                    "score": round(score, 4),
                    "id": doc.get("id"),
                    "case_dir": doc.get("case_dir"),
                    "case_name": doc.get("case_name"),
                    "kind": doc.get("kind"),
                    "title": doc.get("title"),
                    "content": doc.get("content"),
                    "data": doc.get("data"),
                    "supporting_sources": doc.get("supporting_sources") or [],
                    "source_file": doc.get("source_file"),
                }
            )
        return out

    def case_summary(self, case: str, limit: int = 20) -> dict[str, Any]:
        case_l = case.lower()
        docs = [
            item.doc
            for item in self.docs
            if case_l in str(item.doc.get("case_dir") or "").lower()
            or case_l in str(item.doc.get("case_name") or "").lower()
        ]
        by_kind: dict[str, int] = defaultdict(int)
        for doc in docs:
            by_kind[str(doc.get("kind"))] += 1
        return {
            "case": case,
            "count": len(docs),
            "counts_by_kind": dict(sorted(by_kind.items())),
            "sample": [
                {
                    "id": doc.get("id"),
                    "kind": doc.get("kind"),
                    "title": doc.get("title"),
                    "content": doc.get("content"),
                    "supporting_sources": doc.get("supporting_sources") or [],
                }
                for doc in docs[: max(1, min(limit, 100))]
            ],
        }


class BrowserClient:
    def __init__(self, base: str = BROWSER_BASE, timeout: int = 300) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        with BROWSER_TOOL_LOCK:
            url = f"{self.base}{path}"
            resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            if resp.status_code in (502, 503, 504):
                time.sleep(3)
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            try:
                data = resp.json()
            except ValueError:
                return {"status": "error", "message": f"non-JSON response (HTTP {resp.status_code})"}
            if resp.status_code >= 400 and data.get("status") != "error":
                data["status"] = "error"
                data.setdefault("message", f"HTTP {resp.status_code}")
            return data

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def goto(self, url: str, **kwargs: Any) -> dict[str, Any]:
        body = {"url": url, "wait_cf": True, "wait_until": "domcontentloaded", "timeout": 60000}
        body.update(kwargs)
        return self.request("POST", "/api/goto", json=body)

    def text(self, selector: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if selector:
            body["selector"] = selector
        return self.request("POST", "/api/text", json=body)

    def evaluate(self, script: str, arg: Any = None) -> dict[str, Any]:
        return self.request("POST", "/api/evaluate", json={"script": script, "arg": arg})


SEARCH_SKIP_HOSTS = (
    "google.",
    "gstatic.",
    "googleapis.",
    "googleusercontent.",
    "duckduckgo.com",
    "bing.com",
    "microsoft.com",
    "youtube.",
    "youtu.be",
)


def build_search_url(engine: str, query: str, num: int) -> str:
    q = quote(query)
    if engine == "ddg":
        return f"https://html.duckduckgo.com/html/?q={q}"
    if engine == "bing":
        return f"https://www.bing.com/search?q={q}&count={max(num, 10)}"
    return f"https://www.google.com/search?q={q}&num={max(num, 10)}&hl=en"


def decode_search_redirect(url: str) -> str:
    if not url:
        return ""
    if "uddg=" in url.lower():
        match = re.search(r"[?&]uddg=([^&]+)", url)
        if match:
            return unquote(match.group(1))
    if "/url?" in url and "google." in urlparse(url).netloc:
        match = re.search(r"[?&]q=([^&]+)", url)
        if match:
            return unquote(match.group(1))
    return url


def clean_search_links(raw_links: list[dict[str, Any]], limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_links:
        url = decode_search_redirect(str(item.get("url") or ""))
        if not url.startswith("http"):
            continue
        host = urlparse(url).netloc.lower()
        if any(skip in host for skip in SEARCH_SKIP_HOSTS):
            continue
        key = normalize_url(url)
        if key in seen:
            continue
        seen.add(key)
        title = str(item.get("title") or item.get("text") or "").strip()
        out.append({"url": url, "title": title[:200]})
        if len(out) >= limit:
            break
    return out


class WebTools:
    def __init__(self, browser: BrowserClient) -> None:
        self.browser = browser

    def search(self, query: str, engine: str = "google", num: int = 10) -> dict[str, Any]:
        with BROWSER_TOOL_LOCK:
            got = self.browser.goto(build_search_url(engine, query, num))
            if got.get("status") != "ok":
                return got
            snap = got.get("data") or {}
            if snap.get("cloudflare_challenge"):
                return {
                    "status": "ok",
                    "engine": engine,
                    "banned": True,
                    "challenge_type": snap.get("challenge_type"),
                    "results": [],
                }
            script = (
                "() => Array.from(document.querySelectorAll('a[href]')).map(a => "
                "({url: a.href, title: (a.innerText || a.textContent || '').trim().slice(0, 200)}))"
            )
            res = self.browser.evaluate(script)
            if res.get("status") != "ok":
                return res
            links = res.get("data", {}).get("result") or []
            return {"status": "ok", "engine": engine, "results": clean_search_links(links, num)}

    def visit(self, url: str, limit: int = 2500) -> dict[str, Any]:
        with BROWSER_TOOL_LOCK:
            got = self.browser.goto(url)
            if got.get("status") != "ok":
                return got
            data = got.get("data") or {}
            return {
                "status": "ok",
                "url": data.get("url", url),
                "title": data.get("title"),
                "cloudflare_challenge": data.get("cloudflare_challenge"),
                "challenge_type": data.get("challenge_type"),
                "text": truncate(data.get("text") or "", max(500, min(limit, 8000))),
            }

    def text(self, selector: str = "", limit: int = 4000) -> dict[str, Any]:
        with BROWSER_TOOL_LOCK:
            res = self.browser.text(selector or None)
            if res.get("status") != "ok":
                return res
            data = res.get("data") or {}
            return {
                "status": "ok",
                "url": data.get("url"),
                "text": truncate(data.get("text") or "", max(500, min(limit, 10000))),
            }

    def links(self, pattern: str = "", limit: int = 30) -> dict[str, Any]:
        with BROWSER_TOOL_LOCK:
            script = (
                "() => Array.from(document.querySelectorAll('a[href]')).map(a => "
                "({url: a.href, text: (a.innerText || a.textContent || '').trim().slice(0, 150)}))"
            )
            res = self.browser.evaluate(script)
            if res.get("status") != "ok":
                return res
            links = res.get("data", {}).get("result") or []
            pat = pattern.lower()
            cleaned = []
            seen = set()
            for item in links:
                url = str(item.get("url") or "")
                text = str(item.get("text") or "")
                if not url.startswith("http"):
                    continue
                if pat and pat not in url.lower() and pat not in text.lower():
                    continue
                key = normalize_url(url)
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append({"url": url, "text": text})
                if len(cleaned) >= limit:
                    break
            return {"status": "ok", "count": len(cleaned), "links": cleaned}


class EthIndexClient:
    def __init__(self, base: str = ETH_INDEX_URL, timeout: int = 120) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base}{path}"
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.ConnectionError:
                if attempt == max_retries:
                    return {"status": "error", "http_status": 0, "data": {"text": "connection error after retries"}}
                time.sleep(1.0 * attempt)
                continue
            try:
                data = resp.json()
            except ValueError:
                data = {"text": resp.text[:2000]}
            if resp.status_code == 404 and isinstance(data, dict) and data.get("detail") == "address not indexed":
                return {"status": "error", "http_status": resp.status_code, "data": data}
            if resp.status_code >= 500 or resp.status_code == 429:
                if attempt < max_retries:
                    time.sleep(1.0 * attempt)
                    continue
            if resp.status_code >= 400:
                return {"status": "error", "http_status": resp.status_code, "data": data}
            return {"status": "ok", "data": data}
        return {"status": "error", "http_status": resp.status_code, "data": data}

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/health")

    def node(self, address: str) -> dict[str, Any]:
        return self.request("GET", f"/node/{address.lower()}")

    def neighbors(
        self,
        address: str,
        direction: str = "both",
        limit: int = 100,
        block_min: int | None = None,
        block_max: int | None = None,
        edge_type: int | None = None,
        with_raw: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "direction": direction,
            "limit": max(1, min(limit, 500001)),
            "with_raw": str(with_raw).lower(),
        }
        if block_min is not None:
            params["block_min"] = block_min
        if block_max is not None:
            params["block_max"] = block_max
        if edge_type is not None:
            params["edge_type"] = edge_type
        return self.request("GET", f"/neighbors/{address.lower()}", params=params)

    def expand(
        self,
        seeds: list[str],
        k: int = 1,
        direction: str = "both",
        max_nodes: int = 200,
        max_edges: int = 500,
        block_min: int | None = None,
        block_max: int | None = None,
        edge_types: list[int] | None = None,
    ) -> dict[str, Any]:
        body = {
            "seeds": [seed.lower() for seed in seeds],
            "k": max(1, min(k, 3)),
            "direction": direction,
            "max_nodes": max(1, min(max_nodes, 10000)),
            "max_edges": max(1, min(max_edges, 100000)),
            "block_min": block_min,
            "block_max": block_max,
            "edge_types": edge_types,
        }
        return self.request("POST", "/expand", json=body)

    def edge(self, edge_id: int, with_raw: bool = True) -> dict[str, Any]:
        return self.request("GET", f"/edge/{edge_id}", params={"with_raw": str(with_raw).lower()})


class LLMClient:
    def __init__(self) -> None:
        self.model = "openai/" + LLM_NAME
        self.api_base = LLM_PROVIDER_URL
        self.api_key = LLM_API_KEY

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.2,
        tool_choice: Any | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        if litellm is None:
            raise RuntimeError("litellm is not installed; run through `uv run` or use --retrieve-only")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or LLM_OUTPUT_LENGTH,
            "timeout": LLM_TIMEOUT,
            "num_retries": 2,
            "drop_params": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if not LLM_THINK:
            if "qwen" in self.model.lower():
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            elif "deepseek" in self.model.lower():
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                kwargs["response_format"] = {"type": "json_object"}
            else:
                print("Unknown model")
        return litellm.completion(**kwargs)


SYSTEM_PROMPT = """You are an on-chain incident QA RAG agent.

Answer questions using the supplied source evidence first. Treat retrieved
summary.json and chain_evidence.json snippets, supporting source ids, and source
URLs as the primary ground truth.
You may call tools for:
- Evidence retrieval from local summarized/ case artifacts.
- Public web/browser checks when the summarized case evidence has an obvious gap.
- The local Ethereum transaction index for address/edge neighborhood facts.
- Etherscan page wrappers for address, transaction, block, and tx-list page facts.

Rules:
- Cite retrieved evidence ids and source_file when using local case evidence.
- Say when a fact comes from local Ethereum index, Etherscan, or public web rather than summarized case evidence.
- Do not call report-derived fund-flow edges chain-verified unless the evidence says verification_status=chain_verified.
- Do not relabel CEX, mixer, DEX, bridge, or router entities as attacker-controlled wallets without evidence.
- If evidence is insufficient, say so and list what was checked.
- Keep the final answer concise and in the user's language.
"""


class OnchainQARagAgent:
    def __init__(
        self,
        *,
        evidence_store: EvidenceStore,
        browser_base: str = BROWSER_BASE,
        eth_index_url: str = ETH_INDEX_URL,
        verbose: bool = True,
    ) -> None:
        self.evidence_store = evidence_store
        self.browser = BrowserClient(browser_base)
        self.web = WebTools(self.browser)
        self.eth = EthIndexClient(eth_index_url)
        self.etherscan: Any | None = None
        self.llm = LLMClient()
        self.verbose = verbose

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_evidence",
                    "description": "Retrieve grounded evidence snippets from local summarized/ artifacts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "case": {"type": "string", "description": "Optional case_dir or case_name substring."},
                            "top_k": {"type": "integer", "default": 8},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "case_summary",
                    "description": "Return evidence counts and sample snippets for a case.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "case": {"type": "string"},
                            "limit": {"type": "integer", "default": 20},
                        },
                        "required": ["case"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search public internet through the persistent browser. Try google, ddg, then bing if challenged.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "engine": {"type": "string", "enum": ["google", "ddg", "bing"], "default": "google"},
                            "num": {"type": "integer", "default": 10},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "visit_page",
                    "description": "Open a web page and return visible text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "limit": {"type": "integer", "default": 2500},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_page_text",
                    "description": "Get more text from the current browser page, optionally by selector.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "limit": {"type": "integer", "default": 4000},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_page_links",
                    "description": "List links from the current browser page, optionally filtered by substring.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "limit": {"type": "integer", "default": 30},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "eth_node",
                    "description": "Look up an Ethereum address in the local index.",
                    "parameters": {
                        "type": "object",
                        "properties": {"address": {"type": "string"}},
                        "required": ["address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "eth_neighbors",
                    "description": "Query local Ethereum index neighbors for an address.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "address": {"type": "string"},
                            "direction": {"type": "string", "enum": ["in", "out", "both"], "default": "both"},
                            "limit": {"type": "integer", "default": 100},
                            "block_min": {"type": "integer"},
                            "block_max": {"type": "integer"},
                            "edge_type": {"type": "integer"},
                            "with_raw": {"type": "boolean", "default": False},
                        },
                        "required": ["address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "eth_expand",
                    "description": "Expand a small k-hop subgraph from Ethereum address seeds.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "seeds": {"type": "array", "items": {"type": "string"}},
                            "k": {"type": "integer", "default": 1},
                            "direction": {"type": "string", "enum": ["in", "out", "both"], "default": "both"},
                            "max_nodes": {"type": "integer", "default": 200},
                            "max_edges": {"type": "integer", "default": 500},
                            "block_min": {"type": "integer"},
                            "block_max": {"type": "integer"},
                            "edge_types": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["seeds"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "eth_edge",
                    "description": "Read one edge from the local Ethereum index by global edge id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "edge_id": {"type": "integer"},
                            "with_raw": {"type": "boolean", "default": True},
                        },
                        "required": ["edge_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "etherscan_address",
                    "description": "Fetch and parse an Etherscan address page through the browser.",
                    "parameters": {
                        "type": "object",
                        "properties": {"address": {"type": "string"}},
                        "required": ["address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "etherscan_tx",
                    "description": "Fetch and parse an Etherscan transaction page through the browser.",
                    "parameters": {
                        "type": "object",
                        "properties": {"tx_hash": {"type": "string"}},
                        "required": ["tx_hash"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "etherscan_block",
                    "description": "Fetch and parse an Etherscan block page through the browser.",
                    "parameters": {
                        "type": "object",
                        "properties": {"block": {"type": "integer"}},
                        "required": ["block"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "etherscan_transactions",
                    "description": "Fetch Etherscan transaction-list pages for an address or block.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "address": {"type": "string"},
                            "block": {"type": "integer"},
                            "failed": {"type": "boolean", "default": False},
                            "outgoing": {"type": "boolean", "default": False},
                            "income": {"type": "boolean", "default": False},
                            "self_tx": {"type": "boolean", "default": False},
                            "contract_create": {"type": "boolean", "default": False},
                            "max_pages": {"type": "integer", "default": 1},
                        },
                    },
                },
            },
        ]

    def dispatch(self, name: str, args: dict[str, Any]) -> Any:
        if name == "retrieve_evidence":
            return {"status": "ok", "evidence": self.evidence_store.search(args["query"], case=args.get("case"), top_k=args.get("top_k", 8))}
        if name == "case_summary":
            return {"status": "ok", "summary": self.evidence_store.case_summary(args["case"], limit=args.get("limit", 20))}
        if name == "web_search":
            return self.web.search(args["query"], engine=args.get("engine", "google"), num=args.get("num", 10))
        if name == "visit_page":
            return self.web.visit(args["url"], limit=args.get("limit", 2500))
        if name == "get_page_text":
            return self.web.text(args.get("selector", ""), limit=args.get("limit", 4000))
        if name == "get_page_links":
            return self.web.links(args.get("pattern", ""), limit=args.get("limit", 30))
        if name == "eth_node":
            return self.eth.node(args["address"])
        if name == "eth_neighbors":
            return self.eth.neighbors(
                args["address"],
                direction=args.get("direction", "both"),
                limit=args.get("limit", 100),
                block_min=args.get("block_min"),
                block_max=args.get("block_max"),
                edge_type=args.get("edge_type"),
                with_raw=args.get("with_raw", False),
            )
        if name == "eth_expand":
            return self.eth.expand(
                args["seeds"],
                k=args.get("k", 1),
                direction=args.get("direction", "both"),
                max_nodes=args.get("max_nodes", 200),
                max_edges=args.get("max_edges", 500),
                block_min=args.get("block_min"),
                block_max=args.get("block_max"),
                edge_types=args.get("edge_types"),
            )
        if name == "eth_edge":
            return self.eth.edge(args["edge_id"], with_raw=args.get("with_raw", True))
        if name == "etherscan_address":
            if self.etherscan is None:
                from etherscan_browser import EtherscanBrowserClient

                self.etherscan = EtherscanBrowserClient(browser_base=self.browser.base)
            with BROWSER_TOOL_LOCK:
                return {"status": "ok", "data": self.etherscan.get_address_info(args["address"])}
        if name == "etherscan_tx":
            if self.etherscan is None:
                from etherscan_browser import EtherscanBrowserClient

                self.etherscan = EtherscanBrowserClient(browser_base=self.browser.base)
            with BROWSER_TOOL_LOCK:
                return {"status": "ok", "data": self.etherscan.get_transaction(args["tx_hash"])}
        if name == "etherscan_block":
            if self.etherscan is None:
                from etherscan_browser import EtherscanBrowserClient

                self.etherscan = EtherscanBrowserClient(browser_base=self.browser.base)
            with BROWSER_TOOL_LOCK:
                return {"status": "ok", "data": self.etherscan.get_block(int(args["block"]))}
        if name == "etherscan_transactions":
            if self.etherscan is None:
                from etherscan_browser import EtherscanBrowserClient

                self.etherscan = EtherscanBrowserClient(browser_base=self.browser.base)
            with BROWSER_TOOL_LOCK:
                return {
                    "status": "ok",
                    "data": self.etherscan.get_transactions(
                        address=args.get("address"),
                        block=args.get("block"),
                        failed=args.get("failed", False),
                        outgoing=args.get("outgoing", False),
                        income=args.get("income", False),
                        self_tx=args.get("self_tx", False),
                        contract_create=args.get("contract_create", False),
                        max_pages=args.get("max_pages", 1),
                    ),
                }
        return {"status": "error", "message": f"unknown tool: {name}"}

    def answer(self, question: str, *, case: str | None = None, top_k: int = 8) -> str:
        initial = self.evidence_store.search(question, case=case, top_k=top_k)
        return self.answer_from_evidence(question, initial, case=case)

    def answer_from_evidence(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        *,
        case: str | None = None,
        allow_tools: bool = True,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Case filter: {case or '(none)'}\n\n"
                    f"Initial local summarized/ evidence retrieval:\n{compact_json(evidence, 14000)}\n\n"
                    "Use tools if needed, then answer with evidence."
                ),
            },
        ]

        for iteration in range(1, MAX_ITERATIONS + 1):
            if self.verbose:
                print(f"[agent] iteration {iteration}", file=sys.stderr, flush=True)
            response = self.llm.complete(messages, tools=self.tools if allow_tools else None)
            msg = response.choices[0].message
            assistant_msg = msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else dict(msg)
            messages.append(assistant_msg)

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                return str(msg.content or "").strip()

            for tool_call in tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if self.verbose:
                    print(f"[tool] {name}({truncate(args, 300)})", file=sys.stderr, flush=True)
                try:
                    result = self.dispatch(name, args)
                except Exception as exc:  # noqa: BLE001
                    result = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": truncate(result, 18000),
                    }
                )
        return self.fallback_answer(question, evidence)

    @staticmethod
    def fallback_answer(question: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return f"没有在本地 summarized/ 证据中检索到足够证据回答：{question}"
        lines = ["LLM 未完成回答，以下是本地 summarized/ 检索到的最高相关证据："]
        for idx, row in enumerate(rows[:5], start=1):
            lines.append(
                f"{idx}. [{row.get('id')}] {row.get('title')}\n"
                f"   内容: {truncate(row.get('content'), 500)}\n"
                f"   来源: {row.get('source_file')}"
            )
        return "\n".join(lines)


def retrieve_evidence(
    question: str,
    *,
    case: str | None = None,
    top_k: int = 8,
    summarized_dir: Path = SUMMARIZED_DIR,
    evidence_store: EvidenceStore | None = None,
) -> list[dict[str, Any]]:
    """Convenience API: retrieve source evidence without touching qa/."""
    store = evidence_store or EvidenceStore(summarized_dir)
    return store.search(question, case=case, top_k=top_k)


def answer_question(
    question: str,
    *,
    case: str | None = None,
    top_k: int = 8,
    browser_base: str = BROWSER_BASE,
    eth_index_url: str = ETH_INDEX_URL,
    summarized_dir: Path = SUMMARIZED_DIR,
    evidence_store: EvidenceStore | None = None,
    verbose: bool = False,
    fallback_on_error: bool = True,
    allow_tools: bool = False,
) -> AgentResult:
    """Convenience API for evaluators and other scripts.

    Returns both the agent answer and the initial summarized/ retrieval used to
    ground the answer. The test-set qa/ folder is intentionally not read here.
    """
    store = evidence_store or EvidenceStore(summarized_dir)
    evidence = store.search(question, case=case, top_k=top_k)
    agent = OnchainQARagAgent(
        evidence_store=store,
        browser_base=browser_base,
        eth_index_url=eth_index_url,
        verbose=verbose,
    )
    try:
        answer = agent.answer_from_evidence(question, evidence, case=case, allow_tools=allow_tools)
        return AgentResult(question=question, answer=answer, evidence=evidence, case=case)
    except Exception as exc:
        if not fallback_on_error:
            raise
        return AgentResult(
            question=question,
            answer=agent.fallback_answer(question, evidence),
            evidence=evidence,
            case=case,
            error=f"{type(exc).__name__}: {exc}",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="On-chain incident QA RAG agent over summarized/ evidence, public browser, local ETH index, and Etherscan.")
    parser.add_argument("question", nargs="*", help="Question to answer.")
    parser.add_argument("--case", default=None, help="Optional case_dir or case_name substring filter.")
    parser.add_argument("--top-k", type=int, default=8, help="Initial local evidence retrieval count.")
    parser.add_argument("--browser-base", default=BROWSER_BASE, help="Agent browser service base URL.")
    parser.add_argument("--eth-index-url", default=ETH_INDEX_URL, help="Local Ethereum index HTTP base URL.")
    parser.add_argument("--retrieve-only", action="store_true", help="Only print local summarized/ evidence retrieval, without LLM/tool loop.")
    parser.add_argument("--quiet", action="store_true", help="Suppress agent progress logs.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.command in {"answer", "eval"}
        and args.no_detector_tools
        and args.detector_tool_methods
    ):
        parser.error("--no-detector-tools cannot be combined with --detector-tool-method")
    question = " ".join(args.question).strip()
    if not question:
        raise SystemExit("Please provide a question.")

    store = EvidenceStore()
    if args.retrieve_only:
        print(compact_json(store.search(question, case=args.case, top_k=args.top_k), 30000))
        return 0

    agent = OnchainQARagAgent(
        evidence_store=store,
        browser_base=args.browser_base,
        eth_index_url=args.eth_index_url,
        verbose=not args.quiet,
    )
    try:
        print(agent.answer(question, case=args.case, top_k=args.top_k))
    except Exception as exc:  # Keep the CLI useful if the configured LLM endpoint is down.
        rows = store.search(question, case=args.case, top_k=args.top_k)
        print(f"[warn] LLM/tool agent failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(agent.fallback_answer(question, rows))
    return 0


CHAIN_ACTIONS = ("follow", "inspect", "stop", "ignore")


def empty_token_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def normalize_token_usage(value: Any) -> dict[str, int]:
    if value is None:
        return empty_token_usage()
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    elif not isinstance(value, dict):
        value = {
            key: getattr(value, key, None)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            )
        }
    input_tokens = value.get("input_tokens")
    if input_tokens is None:
        input_tokens = value.get("prompt_tokens", 0)
    output_tokens = value.get("output_tokens")
    if output_tokens is None:
        output_tokens = value.get("completion_tokens", 0)
    try:
        normalized_input = max(0, int(input_tokens or 0))
    except (TypeError, ValueError):
        normalized_input = 0
    try:
        normalized_output = max(0, int(output_tokens or 0))
    except (TypeError, ValueError):
        normalized_output = 0
    try:
        normalized_total = max(0, int(value.get("total_tokens") or 0))
    except (TypeError, ValueError):
        normalized_total = 0
    if normalized_total == 0:
        normalized_total = normalized_input + normalized_output
    return {
        "input_tokens": normalized_input,
        "output_tokens": normalized_output,
        "total_tokens": normalized_total,
    }


def response_token_usage(response: Any) -> dict[str, int]:
    if isinstance(response, dict):
        return normalize_token_usage(response.get("usage"))
    return normalize_token_usage(getattr(response, "usage", None))


def add_token_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    total["input_tokens"] += int(usage.get("input_tokens") or 0)
    total["output_tokens"] += int(usage.get("output_tokens") or 0)
    total["total_tokens"] += int(usage.get("total_tokens") or 0)

CHAIN_PUBLIC_TASK_FIELDS = {
    "id",
    "dataset_version",
    "case_id",
    "case_dir",
    "case_name",
    "task_type",
    "difficulty",
    "question",
    "graph_context",
    "edge_summaries",
    "evidence",
    "link_semantics",
    "question_type",
    "agent_profile",
    "required_tools",
    "allowed_tools",
    "tool_requirement",
    "public_context_profile",
}
CHAIN_QUERY_TASK_FIELDS = {
    "id",
    "dataset_version",
    "case_id",
    "case_dir",
    "case_name",
    "task_type",
    "difficulty",
    "question",
    "graph_context",
    "link_semantics",
    "question_type",
    "agent_profile",
    "required_tools",
    "allowed_tools",
    "tool_requirement",
    "public_context_profile",
}
CHAIN_ORACLE_KEYS = {
    "answer",
    "answer_value",
    "labels",
    "evaluation",
    "action_labels",
    "correct_next_edges",
    "correct_follow_edges",
    "none_of_above",
    "candidate_actions",
    "state_action_labels",
    "gold_seed_nodes",
    "gold_follow_edges",
    "gold_paths",
    "teacher_forced_state_ids",
    "terminal_nodes",
}
CHAIN_TOOL_ORACLE_KEYS = CHAIN_ORACLE_KEYS | {
    "tx_label",
    "edge_label",
    "action_label",
    "src_is_black",
    "dst_is_black",
    "is_black",
    "is_path_relevant",
    "path_id",
    "hop_index",
    "gold_action",
    "gold_label",
    "case_role",
    "role_label",
    "rule_labels",
    "node_is_black",
    "reported_edge_id",
    "negative_sample_type",
    "usable_for_eval",
    "usable_for_qa",
}

CHAIN_AGENT_PROFILES = {
    "context_only": {
        "allow_tools": False,
        "include_edge_docs": True,
    },
    "tool_grounded": {
        "allow_tools": True,
        "include_edge_docs": False,
    },
}

SNF_STATE_POLICY_RESPONSE_SCHEMA = {
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


def project_snf_state_policy_public_task(public_task: dict[str, Any]) -> dict[str, Any]:
    """Project legacy SNF rows onto the non-oracle-leaking agent output contract."""
    if public_task.get("task_type") != "subgraph_noise_filtering":
        return public_task
    context = public_task.get("graph_context") or {}
    if not context.get("sequential_policy"):
        return public_task
    projected = dict(public_task)
    projected_context = dict(context)
    projected_context["response_schema"] = SNF_STATE_POLICY_RESPONSE_SCHEMA
    projected_context["agent_output_contract"] = "snf_state_policy_v2"
    projected_context["rollout_derivation"] = "evaluator_seed_projection_v1"
    projected["graph_context"] = projected_context
    projected["question"] = (
        f"{str(projected.get('question') or '').strip()}\n"
        "Runtime output contract: rank/select candidate seeds and return one "
        "state_policy step for every public state exactly once. Do not return "
        "oracle_seed or predicted_seed; the evaluator derives both rollouts."
    ).strip()
    return projected

CHAIN_TOOL_SCHEMAS = {
    "eth_neighbors": {
        "type": "function",
        "function": {
            "name": "eth_neighbors",
            "description": (
                "Query the complete local Ethereum index outgoing neighbor set. "
                "The runner filters it to the task's exact destination."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string"},
                    "direction": {"type": "string", "enum": ["in", "out", "both"]},
                    "block_min": {"type": "integer"},
                    "block_max": {"type": "integer"},
                    "edge_type": {"type": "integer"},
                    "with_raw": {"type": "boolean"},
                },
                "required": [
                    "address",
                    "direction",
                    "block_min",
                    "block_max",
                    "edge_type",
                    "with_raw",
                ],
                "additionalProperties": False,
            },
        },
    },
    "eth_edge": {
        "type": "function",
        "function": {
            "name": "eth_edge",
            "description": (
                "Read one public Ethereum index edge by numeric edge id. "
                "When candidate_edge_groups shows eth:N, pass the numeric N suffix."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "edge_id": {"type": "integer"},
                    "with_raw": {"type": "boolean"},
                },
                "required": ["edge_id", "with_raw"],
                "additionalProperties": False,
            },
        },
    },
}


CHAIN_AGENT_SYSTEM_PROMPT = """You are a path-oriented on-chain tracing QA agent.

Answer only from the supplied chain QA task context and retrieved graph evidence.
Do not use hidden labels or external knowledge.

For edge_action_classification, return JSON only. Include every candidate exactly
once in predictions. Each prediction must contain edge_id, one action from follow,
inspect, stop, ignore, and action_probabilities for all four actions summing to 1.
Rank candidates by action_probabilities.follow. selected_follow_edges may be empty
and must equal exactly all candidates classified follow whose follow probability
meets graph_context.follow_threshold. Candidate order is not a relevance signal.

For sequential path_completion and subgraph_noise_filtering, return JSON only
and follow graph_context.response_schema. Evaluate every candidate in each
visited state with the same four-action probability contract. For path completion
return both teacher_forced and free_rollout. For seed-hidden noise filtering rank
every candidate seed and return one state_policy step for every public state
exactly once. Do not guess or return hidden oracle seeds. The evaluator derives
oracle_seed and predicted_seed rollouts from that single state policy.
Respect rollout_budget and do not use candidate order as a relevance signal.

For tool-grounded edge_action_classification, path_completion, and
subgraph_noise_filtering, call eth_edge with with_raw=true for at least one raw
edge id from every public candidate_edge_groups entry before answering. Use only
the public candidate universe, avoid duplicate queries, and stay within the
tool_query_contract and rollout_budget. Persisted raw ids may be written as
eth:N; the eth_edge argument is the numeric N suffix.

When detector assistance is enabled, the runtime supplies one advisory result
covering every public state after mandatory eth_edge evidence collection. For SNF
it also supplies the frozen shared node-detector seed ranking. Consult those
signals during final synthesis, but do not copy them blindly: they never replace
the mandatory index evidence or change the public candidate universe.

For other task types, keep the answer concise and grounded in the supplied evidence.
"""

CHAIN_PROCESS_JUDGE_SYSTEM_PROMPT = """You evaluate the evidence process of an on-chain chain-QA agent.

You have no tools and receive no hidden labels. Evaluate only whether the public
context supplied to the agent was relevant, sufficient, and grounded for the
question, and whether the sanitized tool trace was appropriate, economical, and
useful. Do not repair or reinterpret malformed answers. Do not infer the hidden
correct trajectory and do not override deterministic task correctness.

Return JSON only with retrieval_relevance, answer_correctness, groundedness, pass,
rationale, missing_evidence, and unsupported_claims. Use 0-5 scores. Here
answer_correctness means adherence to the requested evidence process and output
contract, not hidden-label correctness.
"""

TRANSACTION_EXISTENCE_SYSTEM_PROMPT = """You are an Ethereum transaction-existence agent.

The local Ethereum index is the sole authority. You must call eth_neighbors with
the exact public src, direction=out, block_min, block_max, edge_type, and
with_raw=true. Do not use reports, case labels, address roles, or semantic
similarity. Return exists=true only when the tool result contains an exact
normalized src/dst match. Do not invent edge ids or transaction hashes.

Return JSON only with exactly these fields:
{
  "exists": true,
  "transaction_count": 1,
  "matching_edge_ids": ["123"],
  "matching_tx_hashes": ["0x..."]
}
"""


class ChainTaskStore:
    def __init__(
        self,
        *,
        dataset_version: str | None = None,
        prefer_mongo: bool = True,
        local_root: Path | None = None,
    ) -> None:
        self.dataset_version = dataset_version
        self.prefer_mongo = prefer_mongo
        self.local_root = local_root or (PROJECT_ROOT / "chain_qa")

    def tasks(self, case: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self._mongo_tasks(case) if self.prefer_mongo else []
        if not rows:
            rows = self._local_tasks(case)
        rows.sort(key=lambda row: str(row.get("id") or ""))
        return rows[:limit] if limit else rows

    def _mongo_tasks(self, case: str) -> list[dict[str, Any]]:
        try:
            from storage import MongoDocumentStore

            query: dict[str, Any] = {"$or": [{"case_id": case}, {"case_dir": case}]}
            if self.dataset_version:
                query["dataset_version"] = self.dataset_version
            rows = MongoDocumentStore("chain_qa_tasks").find(query)
            if rows and not self.dataset_version:
                versions = {str(row.get("dataset_version") or "") for row in rows}
                latest = max(versions, key=dataset_version_sort_key)
                rows = [row for row in rows if str(row.get("dataset_version") or "") == latest]
            return rows
        except Exception:
            return []

    def _local_tasks(self, case: str) -> list[dict[str, Any]]:
        candidates = [self.local_root / case / "qa_tasks.jsonl"]
        if not candidates[0].exists():
            candidates.extend(sorted(self.local_root.glob("*/qa_tasks.jsonl")))
        rows: list[dict[str, Any]] = []
        for path in candidates:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for raw in f:
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    if case in {row.get("case_id"), row.get("case_dir")} or path.parent.name == case:
                        if not self.dataset_version or row.get("dataset_version") == self.dataset_version:
                            rows.append(row)
            if rows:
                break
        if rows and not self.dataset_version:
            versions = {str(row.get("dataset_version") or "") for row in rows}
            latest = max(versions, key=dataset_version_sort_key)
            rows = [
                row
                for row in rows
                if str(row.get("dataset_version") or "") == latest
            ]
        return rows

    def edge_docs(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        edge_ids = self._task_edge_ids(task)
        rows = self._mongo_edges(task, edge_ids) if self.prefer_mongo else []
        if not rows:
            rows = self._local_edges(task, edge_ids)
        by_id = {str(row.get("edge_id")): row for row in rows if row.get("edge_id")}
        out = []
        for edge_id in edge_ids:
            out.append(by_id.get(edge_id) or self._edge_from_task_summary(task, edge_id) or {"edge_id": edge_id})
        return out

    @staticmethod
    def _task_edge_ids(task: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        for edge in task.get("edge_summaries") or []:
            if isinstance(edge, dict) and edge.get("edge_id"):
                ids.append(str(edge["edge_id"]))
        context = task.get("graph_context") or {}
        if isinstance(context, dict):
            for key in ("candidate_edges", "edge_ids"):
                for edge_id in context.get(key) or []:
                    ids.append(str(edge_id))
        evidence = task.get("evidence") or {}
        if isinstance(evidence, dict):
            for edge_id in evidence.get("chain_edges") or []:
                ids.append(str(edge_id))
        return list(dict.fromkeys(ids))

    @staticmethod
    def _edge_from_task_summary(task: dict[str, Any], edge_id: str) -> dict[str, Any] | None:
        for edge in task.get("edge_summaries") or []:
            if isinstance(edge, dict) and edge.get("edge_id") == edge_id:
                return edge
        return None

    def _mongo_edges(self, task: dict[str, Any], edge_ids: list[str]) -> list[dict[str, Any]]:
        if not edge_ids:
            return []
        try:
            from storage import MongoDocumentStore

            query: dict[str, Any] = {
                "case_id": task.get("case_id"),
                "edge_id": {"$in": edge_ids},
            }
            if task.get("dataset_version"):
                query["dataset_version"] = task["dataset_version"]
            return MongoDocumentStore("chain_subgraph_edges").find(query)
        except Exception:
            return []

    def _local_edges(self, task: dict[str, Any], edge_ids: list[str]) -> list[dict[str, Any]]:
        path = self.local_root / str(task.get("case_dir")) / "subgraph_edges.jsonl"
        if not path.exists() or not edge_ids:
            return []
        wanted = set(edge_ids)
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("edge_id") in wanted:
                    rows.append(row)
        return rows

    def context_for_task(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        profile = chain_agent_profile(task)
        public_fields = (
            CHAIN_QUERY_TASK_FIELDS if profile == "tool_grounded" else CHAIN_PUBLIC_TASK_FIELDS
        )
        public_task = sanitize_chain_public_value(
            {
                key: value
                for key, value in task.items()
                if key in public_fields
            }
        )
        public_task = project_snf_state_policy_public_task(public_task)
        if profile == "tool_grounded":
            return [
                {
                    "kind": "chain_query_context",
                    "task": public_task,
                }
            ]
        return [
            {
                "kind": "chain_qa_task_public_context",
                "task": public_task,
            },
            {
                "kind": "chain_subgraph_edges",
                "edges": sanitize_chain_public_value(self.edge_docs(task)),
            },
        ]


def sanitize_chain_public_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_chain_public_value(item)
            for key, item in value.items()
            if str(key) not in CHAIN_ORACLE_KEYS and str(key) != "_id"
        }
    if isinstance(value, list):
        return [sanitize_chain_public_value(item) for item in value]
    return value


def sanitize_chain_tool_result(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_chain_tool_result(item)
            for key, item in value.items()
            if str(key) not in CHAIN_TOOL_ORACLE_KEYS and str(key) != "_id"
        }
    if isinstance(value, list):
        return [sanitize_chain_tool_result(item) for item in value]
    return value


def chain_agent_profile(task: dict[str, Any]) -> str:
    return task_agent_profile(task)


@dataclass
class ChainAgentResult:
    task_id: str
    question: str
    answer: str
    context: list[dict[str, Any]]
    session_id: str
    case_id: str | None = None
    dataset_version: str | None = None
    agent_llm: str = LLM_NAME
    agent_token_usage: dict[str, int] | None = None
    agent_profile: str = "context_only"
    tool_trace: list[dict[str, Any]] | None = None
    protocol_errors: list[str] | None = None
    structured_answer: dict[str, Any] | None = None
    answer_repair: dict[str, Any] | None = None
    parse_error: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "session_id": self.session_id,
            "case_id": self.case_id,
            "dataset_version": self.dataset_version,
            "agent_llm": self.agent_llm,
            "agent_token_usage": self.agent_token_usage or empty_token_usage(),
            "agent_profile": self.agent_profile,
            "tool_trace": self.tool_trace or [],
            "protocol_errors": self.protocol_errors or [],
            "structured_answer": self.structured_answer,
            "answer_repair": self.answer_repair or {},
            "parse_error": self.parse_error,
            "error": self.error,
        }


def _message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return message
    return {"content": str(message)}


def save_chain_session(session: dict[str, Any], *, strict: bool = False) -> str | None:
    try:
        from storage import save_chain_agent_session

        save_chain_agent_session(session)
        return None
    except Exception as exc:
        if strict:
            raise
        return f"{type(exc).__name__}: {exc}"


def chain_tools_for_task(
    task: dict[str, Any],
    detector_tool_methods: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    if chain_agent_profile(task) != "tool_grounded":
        return []
    allowed = [str(name) for name in task.get("allowed_tools") or []]
    required = {str(name) for name in task.get("required_tools") or []}
    if not allowed:
        raise ValueError("tool_grounded task must declare allowed_tools")
    unknown = set(allowed) - set(CHAIN_TOOL_SCHEMAS)
    if unknown:
        raise ValueError(f"unsupported allowed chain tools: {sorted(unknown)}")
    if not required <= set(allowed):
        raise ValueError("required_tools must be a subset of allowed_tools")
    schemas = [CHAIN_TOOL_SCHEMAS[name] for name in allowed]
    methods = normalize_detector_tool_methods(detector_tool_methods)
    if methods and task.get("task_type") in {
        "edge_action_classification",
        "path_completion",
        "subgraph_noise_filtering",
    }:
        runtime = DetectorToolRuntime(task=task, methods=methods)
        schemas.append(runtime.schema())
    return schemas


def _tool_call_parts(tool_call: Any) -> tuple[str, str, dict[str, Any]]:
    if isinstance(tool_call, dict):
        function = tool_call.get("function") or {}
        call_id = str(tool_call.get("id") or "")
        name = str(function.get("name") or "")
        arguments = function.get("arguments") or "{}"
    else:
        function = tool_call.function
        call_id = str(getattr(tool_call, "id", "") or "")
        name = str(getattr(function, "name", "") or "")
        arguments = getattr(function, "arguments", "{}") or "{}"
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except (json.JSONDecodeError, TypeError, ValueError):
        args = {}
    return call_id, name, args


def _neighbor_rows(result: Any) -> list[dict[str, Any]]:
    value = result
    if isinstance(value, dict) and value.get("status") == "error":
        raise RuntimeError(f"Ethereum index query failed: {value}")
    for _ in range(3):
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if not isinstance(value, dict):
            return []
        next_value = (
            value.get("edges")
            or value.get("neighbors")
            or value.get("data")
            or value.get("result")
        )
        if next_value is value or next_value is None:
            return []
        value = next_value
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _transaction_tool_summary(
    task: dict[str, Any], args: dict[str, Any], result: Any
) -> dict[str, Any]:
    from extract_qa_chain.pipeline import normalize_index_edge
    from extract_qa_chain.transaction_existence import normalize_edge_type, normalize_int

    context = task.get("graph_context") or {}
    src = str(context.get("src") or "").lower()
    dst = str(context.get("dst") or "").lower()
    block_min = normalize_int(context.get("block_min"))
    block_max = normalize_int(context.get("block_max"))
    edge_type = normalize_edge_type(args.get("edge_type"))
    rows = _neighbor_rows(result)
    complete = len(rows) <= CHAIN_TRANSACTION_QUERY_LIMIT
    usable_rows = rows[:CHAIN_TRANSACTION_QUERY_LIMIT]
    matches: dict[str, dict[str, Any]] = {}
    for raw in usable_rows:
        edge = normalize_index_edge(raw, case_id_="chain_agent", seed=src, hop=1)
        if edge is None or edge.src != src or edge.dst != dst:
            continue
        block = normalize_int(edge.block_number)
        if block_min is None or block_max is None or block is None:
            continue
        if block < block_min or block > block_max:
            continue
        if normalize_edge_type(edge.edge_type) != edge_type:
            continue
        raw_edge_id = edge.raw_index_edge_id
        edge_id = str(raw_edge_id if raw_edge_id is not None else edge.edge_id)
        matches[edge_id] = {
            "edge_id": edge_id,
            "src": edge.src,
            "dst": edge.dst,
            "tx_hash": edge.tx_hash,
            "block_number": edge.block_number,
            "timestamp": edge.timestamp,
            "asset": edge.asset,
            "amount": edge.amount,
            "edge_type": edge.edge_type,
            "token_address": edge.token_address,
        }
    ordered = [sanitize_chain_tool_result(matches[key]) for key in sorted(matches)]
    tx_hashes = sorted(
        {str(edge["tx_hash"]).lower() for edge in ordered if edge.get("tx_hash")}
    )
    return {
        "status": "ok" if complete else "error",
        "query_complete": complete,
        "returned_neighbor_count": min(len(rows), CHAIN_TRANSACTION_QUERY_LIMIT),
        "matching_edge_count": len(ordered),
        "transaction_count": len(tx_hashes),
        "matching_edge_ids": [str(edge["edge_id"]) for edge in ordered],
        "matching_tx_hashes": tx_hashes,
        "matching_edges": ordered,
        "error": None if complete else "neighbor query reached the configured completeness cap",
    }


def dispatch_chain_tool(
    task: dict[str, Any],
    name: str,
    args: dict[str, Any],
    eth: EthIndexClient,
    detector_runtime: DetectorToolRuntime | None = None,
) -> dict[str, Any]:
    if name == "detector_predict":
        if detector_runtime is None or not detector_runtime.enabled:
            raise ValueError("detector_predict is not enabled for this run")
        return detector_runtime.predict(args)
    allowed = {str(value) for value in task.get("allowed_tools") or []}
    if name not in allowed:
        raise ValueError(f"tool is not allowed for this task: {name}")
    if name == "eth_neighbors":
        result = eth.neighbors(
            str(args.get("address") or ""),
            direction=str(args.get("direction") or "both"),
            limit=CHAIN_TRANSACTION_QUERY_LIMIT + 1,
            block_min=args.get("block_min"),
            block_max=args.get("block_max"),
            edge_type=args.get("edge_type"),
            with_raw=bool(args.get("with_raw")),
        )
        if task.get("task_type") == "direct_transaction_existence":
            return _transaction_tool_summary(task, args, result)
        return sanitize_chain_tool_result(result)
    if name == "eth_edge":
        edge_id = _canonical_eth_edge_id(args.get("edge_id"))
        if edge_id is None:
            raise ValueError(f"invalid Ethereum edge id: {args.get('edge_id')!r}")
        return sanitize_chain_tool_result(
            eth.edge(int(edge_id), with_raw=bool(args.get("with_raw", True)))
        )
    raise ValueError(f"unknown chain tool: {name}")


def _canonical_eth_edge_id(value: Any) -> str | None:
    """Normalize persisted ``eth:N`` ids and provider integer forms to ``N``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value) if value >= 0 else None
    text = str(value).strip()
    match = re.fullmatch(r"(?:eth:)?([0-9]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return str(int(match.group(1)))


def _required_true(value: Any) -> bool | None:
    """Accept provider-equivalent true encodings while rejecting false/missing."""
    if value is True:
        return True
    if isinstance(value, str) and value.strip().lower() == "true":
        return True
    return None


def _normalized_eth_edge_args(args: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(args)
    edge_id = _canonical_eth_edge_id(args.get("edge_id"))
    if edge_id is not None:
        normalized["edge_id"] = int(edge_id)
    if _required_true(args.get("with_raw")) is True:
        normalized["with_raw"] = True
    return normalized


def _compact_tool_evidence(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 4,
    max_dict_items: int = 24,
    max_list_items: int = 3,
    max_string_chars: int = 512,
) -> Any:
    """Keep verifiable samples while bounding tool history and judge context."""
    value = sanitize_chain_tool_result(value)
    if isinstance(value, str):
        return value if len(value) <= max_string_chars else value[:max_string_chars] + "...[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= max_depth:
        return {"_summary": f"{type(value).__name__} truncated at depth {max_depth}"}
    if isinstance(value, list):
        output = [
            _compact_tool_evidence(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_dict_items=max_dict_items,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            output.append({"_truncated_items": len(value) - max_list_items})
        return output
    if isinstance(value, dict):
        output = {
            str(key): _compact_tool_evidence(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_dict_items=max_dict_items,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
            for key, item in list(value.items())[:max_dict_items]
        }
        if len(value) > max_dict_items:
            output["_truncated_fields"] = len(value) - max_dict_items
        return output
    return str(value)[:max_string_chars]


def _tracing_runtime_contract(task: dict[str, Any]) -> dict[str, Any] | None:
    if task.get("task_type") not in {
        "edge_action_classification",
        "path_completion",
        "subgraph_noise_filtering",
    } or chain_agent_profile(task) != "tool_grounded":
        return None
    groups = _candidate_edge_groups(task)
    raw_to_candidates: dict[str, set[str]] = {}
    for candidate_id, raw_ids in groups.items():
        for raw_id in raw_ids:
            canonical_id = _canonical_eth_edge_id(raw_id)
            if canonical_id is not None:
                raw_to_candidates.setdefault(canonical_id, set()).add(candidate_id)
    context = task.get("graph_context") or {}
    contract = context.get("tool_query_contract") or {}
    try:
        configured_budget = int(contract.get("max_tool_calls"))
    except (TypeError, ValueError):
        configured_budget = len(groups)
    max_tool_calls = max(1, configured_budget or len(groups) or 1)
    return {
        "candidate_groups": groups,
        "raw_to_candidates": raw_to_candidates,
        "max_tool_calls": max_tool_calls,
        "collection_turn_limit": min(
            CHAIN_MAX_TOOL_COLLECTION_TURNS,
            max(CHAIN_MAX_TOOL_ITERATIONS, max_tool_calls),
        ),
    }


def _runtime_rejected_tool_result(error: str) -> dict[str, Any]:
    return {"status": "error", "error": error, "runtime_rejected": True}


def _parse_chain_structured_answer(
    task: dict[str, Any], answer: str
) -> tuple[dict[str, Any] | None, list[str]]:
    task_type = task.get("task_type")
    if task_type == "edge_action_classification":
        return parse_edge_action_answer(task, answer)
    if task_type in {"path_completion", "subgraph_noise_filtering"} and (
        task.get("graph_context") or {}
    ).get("sequential_policy"):
        return parse_sequential_policy_answer(task, answer)
    if task_type == "direct_transaction_existence":
        return parse_transaction_existence_answer(task, answer)
    return None, []


def _decision_signature(task: dict[str, Any], parsed: dict[str, Any]) -> Any:
    """Freeze substantive decisions while allowing only structural list repair."""
    if task.get("task_type") == "edge_action_classification":
        return parsed.get("predictions")
    if task.get("task_type") not in {"path_completion", "subgraph_noise_filtering"}:
        return None
    signature: dict[str, Any] = {}
    for key in ("seed_predictions", "selected_seed_nodes", "no_valid_seed"):
        if key in parsed:
            signature[key] = parsed.get(key)
    for mode in (
        "teacher_forced",
        "free_rollout",
        "state_policy",
        "oracle_seed",
        "predicted_seed",
    ):
        value = parsed.get(mode)
        if not isinstance(value, dict):
            continue
        signature[mode] = {
            "seed_node": value.get("seed_node"),
            "steps": [
                {
                    "state_id": step.get("state_id"),
                    "current_node": step.get("current_node"),
                    "predictions": step.get("predictions"),
                }
                for step in value.get("steps") or []
                if isinstance(step, dict)
            ],
        }
    return signature


def _selected_follow_only_errors(errors: list[str]) -> bool:
    return bool(errors) and all(
        "selected_follow_edges must equal follow predictions meeting the threshold"
        in error
        for error in errors
    )


def answer_chain_task(
    task: dict[str, Any],
    *,
    task_store: ChainTaskStore | None = None,
    run_id: str | None = None,
    save_session: bool = True,
    strict_mongo: bool = False,
    llm_client: LLMClient | None = None,
    eth_client: EthIndexClient | None = None,
    detector_tool_methods: list[str] | tuple[str, ...] | None = None,
    detector_tool_root: Path | None = None,
) -> ChainAgentResult:
    import uuid
    from datetime import datetime, timezone

    store = task_store or ChainTaskStore(dataset_version=task.get("dataset_version"))
    profile = chain_agent_profile(task)
    context = store.context_for_task(task)
    prompt_question = str(task.get("question") or "")
    for context_item in context:
        public_context_task = context_item.get("task") if isinstance(context_item, dict) else None
        if isinstance(public_context_task, dict) and public_context_task.get("question"):
            prompt_question = str(public_context_task["question"])
            break
    resolved_detector_methods = normalize_detector_tool_methods(detector_tool_methods)
    detector_runtime = (
        DetectorToolRuntime(
            task=task,
            methods=resolved_detector_methods,
            checkpoint_root=detector_tool_root,
            local_root=store.local_root,
        )
        if profile == "tool_grounded"
        and task.get("task_type")
        in {"edge_action_classification", "path_completion", "subgraph_noise_filtering"}
        and resolved_detector_methods
        else None
    )
    advertised_tools = chain_tools_for_task(task, resolved_detector_methods)
    # Detector assistance is injected once by the runtime after evidence
    # collection. Keeping it out of the collection tool menu avoids repeated
    # calls consuming turns needed for mandatory eth_edge coverage.
    tools = [
        schema
        for schema in advertised_tools
        if (schema.get("function") or {}).get("name") != "detector_predict"
    ]
    system_prompt = (
        TRANSACTION_EXISTENCE_SYSTEM_PROMPT
        if task.get("task_type") == "direct_transaction_existence"
        else CHAIN_AGENT_SYSTEM_PROMPT
    )
    session_id = f"{run_id or 'adhoc'}:{task.get('id') or uuid.uuid4().hex}:{uuid.uuid4().hex[:8]}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Question: {prompt_question}\n\n"
                f"Chain QA public context:\n{compact_json(context, CHAIN_CONTEXT_CHAR_LIMIT)}\n\n"
                "Answer now. Do not mention hidden labels."
            ),
        },
    ]
    tool_trace: list[dict[str, Any]] = []
    agent_token_usage = empty_token_usage()
    session = {
        "session_id": session_id,
        "run_id": run_id,
        "task_id": task.get("id"),
        "case_id": task.get("case_id"),
        "case_dir": task.get("case_dir"),
        "dataset_version": task.get("dataset_version"),
        "question": task.get("question"),
        "agent_llm": LLM_NAME,
        "agent_token_usage": agent_token_usage,
        "agent_profile": profile,
        "profile_projection": task.get("profile_projection"),
        "required_tools": task.get("required_tools") or [],
        "allowed_tools": task.get("allowed_tools") or [],
        "detector_tool_methods": list(resolved_detector_methods),
        "detector_tool_root": str(
            detector_tool_root or (PROJECT_ROOT / "detector_tool")
        ),
        "context": context,
        "messages": messages,
        "tool_trace": tool_trace,
        "protocol_errors": [],
        "answer": None,
        "structured_answer": None,
        "answer_repair": {},
        "parse_error": None,
        "error": None,
        "started_at": datetime.now(timezone.utc),
        "finished_at": None,
    }
    try:
        llm = llm_client or LLMClient()
        eth = eth_client or EthIndexClient(ETH_INDEX_URL)
        answer = ""
        tracing_runtime = _tracing_runtime_contract(task)
        queried_edge_ids: set[str] = set()
        covered_candidate_ids: set[str] = set()
        successful_tool_calls = 0
        force_synthesis = False
        collection_turn_limit = (
            int(tracing_runtime["collection_turn_limit"])
            if tracing_runtime
            else CHAIN_MAX_TOOL_ITERATIONS
        )
        for iteration in range(1, collection_turn_limit + 1):
            tool_choice = (
                "required"
                if tools and iteration == 1 and task.get("tool_requirement") == "required"
                else None
            )
            response = llm.complete(
                messages,
                tools=tools or None,
                temperature=0.0,
                tool_choice=tool_choice,
                max_tokens=CHAIN_LLM_OUTPUT_LENGTH,
            )
            add_token_usage(agent_token_usage, response_token_usage(response))
            msg = response.choices[0].message
            assistant_msg = _message_to_dict(msg)
            messages.append(assistant_msg)
            tool_calls = getattr(msg, "tool_calls", None) or assistant_msg.get("tool_calls") or []
            if not tool_calls:
                proposed_answer = str(
                    getattr(msg, "content", None) or assistant_msg.get("content") or ""
                ).strip()
                if tracing_runtime:
                    # Tracing tasks always finish in a distinct tools-disabled turn.
                    # A premature answer simply closes evidence collection.
                    force_synthesis = True
                    break
                answer = proposed_answer
                break
            progress_before = (
                len(covered_candidate_ids),
                successful_tool_calls,
            )
            for tool_call in tool_calls:
                try:
                    call_id, name, args = _tool_call_parts(tool_call)
                except Exception as exc:
                    call_id = str(getattr(tool_call, "id", None) or uuid.uuid4().hex)
                    name = "invalid_tool_call"
                    args = {}
                    tool_result = _runtime_rejected_tool_result(
                        f"invalid tool call: {type(exc).__name__}: {exc}"
                    )
                    trace_entry = {
                        "iteration": iteration,
                        "tool_call_id": call_id,
                        "tool": name,
                        "arguments": {},
                        "status": "error",
                        "result_summary": tool_result,
                        "error": tool_result["error"],
                    }
                    tool_trace.append(trace_entry)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": compact_json(
                                tool_result, CHAIN_TOOL_RESULT_CHAR_LIMIT
                            ),
                        }
                    )
                    continue
                trace_entry: dict[str, Any] = {
                    "iteration": iteration,
                    "tool_call_id": call_id,
                    "tool": name,
                    "arguments": sanitize_chain_public_value(args),
                    "status": "ok",
                    "result_summary": None,
                    "error": None,
                }
                try:
                    runtime_error = None
                    raw_id = _canonical_eth_edge_id(args.get("edge_id"))
                    if tracing_runtime and name == "eth_edge":
                        submitted_args = dict(args)
                        args = _normalized_eth_edge_args(args)
                        trace_entry["arguments"] = sanitize_chain_public_value(args)
                        if args != submitted_args:
                            trace_entry["submitted_arguments"] = (
                                sanitize_chain_public_value(submitted_args)
                            )
                        if _required_true(submitted_args.get("with_raw")) is not True:
                            runtime_error = "with_raw=true is required"
                        elif raw_id is None:
                            runtime_error = "edge_id must be an integer or eth:<integer>"
                        elif raw_id not in tracing_runtime["raw_to_candidates"]:
                            runtime_error = "edge_id is outside public candidate groups"
                        elif raw_id in queried_edge_ids:
                            runtime_error = "duplicate edge query rejected"
                        elif successful_tool_calls >= tracing_runtime["max_tool_calls"]:
                            runtime_error = "tool call budget exhausted"
                        if raw_id is not None:
                            queried_edge_ids.add(raw_id)
                    if runtime_error:
                        tool_result = _runtime_rejected_tool_result(runtime_error)
                    else:
                        tool_result = dispatch_chain_tool(
                            task,
                            name,
                            args,
                            eth,
                            detector_runtime=detector_runtime,
                        )
                        tool_result = sanitize_chain_tool_result(tool_result)
                        if (
                            tracing_runtime
                            and name == "eth_edge"
                            and tool_result.get("status") != "error"
                        ):
                            successful_tool_calls += 1
                            covered_candidate_ids.update(
                                tracing_runtime["raw_to_candidates"].get(raw_id, set())
                            )
                    if tool_result.get("status") == "error":
                        trace_entry["status"] = "error"
                        trace_entry["error"] = str(tool_result.get("error") or "tool error")
                except Exception as exc:
                    tool_result = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    trace_entry["status"] = "error"
                    trace_entry["error"] = tool_result["error"]
                if tracing_runtime and name == "eth_edge":
                    tool_result = _compact_tool_evidence(tool_result)
                trace_entry["result_summary"] = tool_result
                tool_trace.append(trace_entry)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": compact_json(
                            tool_result,
                            CHAIN_DETECTOR_TOOL_RESULT_CHAR_LIMIT
                            if name == "detector_predict"
                            else CHAIN_TOOL_RESULT_CHAR_LIMIT,
                        ),
                    }
                )
            if tracing_runtime:
                all_candidates = set(tracing_runtime["candidate_groups"])
                no_progress = progress_before == (
                    len(covered_candidate_ids),
                    successful_tool_calls,
                )
                if (
                    all_candidates <= covered_candidate_ids
                    or successful_tool_calls >= tracing_runtime["max_tool_calls"]
                    or no_progress
                    or iteration >= collection_turn_limit
                ):
                    force_synthesis = True
                    break
        if tracing_runtime and detector_runtime is not None:
            detector_call_id = f"runtime-detector-{uuid.uuid4().hex[:12]}"
            detector_args = {"methods": list(resolved_detector_methods)}
            detector_assistant = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": detector_call_id,
                        "type": "function",
                        "function": {
                            "name": "detector_predict",
                            "arguments": json.dumps(detector_args, ensure_ascii=False),
                        },
                    }
                ],
            }
            messages.append(detector_assistant)
            detector_trace: dict[str, Any] = {
                "iteration": collection_turn_limit + 1,
                "phase": "detector_consultation",
                "invocation_mode": "runtime_required",
                "tool_call_id": detector_call_id,
                "tool": "detector_predict",
                "arguments": detector_args,
                "status": "ok",
                "result_summary": None,
                "error": None,
            }
            try:
                detector_result = sanitize_chain_tool_result(
                    detector_runtime.predict(detector_args)
                )
                if detector_result.get("status") == "error":
                    detector_trace["status"] = "error"
                    detector_trace["error"] = "detector consultation produced no usable output"
            except Exception as exc:
                detector_result = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                detector_trace["status"] = "error"
                detector_trace["error"] = detector_result["error"]
            detector_trace["result_summary"] = detector_result
            tool_trace.append(detector_trace)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": detector_call_id,
                    "name": "detector_predict",
                    "content": compact_json(
                        detector_result, CHAIN_DETECTOR_TOOL_RESULT_CHAR_LIMIT
                    ),
                }
            )
        if not answer and (tools or detector_runtime is not None) and (
            force_synthesis or tool_trace
        ):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Evidence collection is now closed. Do not call any more tools. "
                        "Return the final answer now as JSON matching graph_context.response_schema, "
                        "using only the public task context and collected tool evidence."
                    ),
                }
            )
            response = llm.complete(
                messages,
                tools=None,
                temperature=0.0 if profile == "tool_grounded" else 0.2,
                tool_choice=None,
                max_tokens=CHAIN_LLM_OUTPUT_LENGTH,
            )
            add_token_usage(agent_token_usage, response_token_usage(response))
            msg = response.choices[0].message
            assistant_msg = _message_to_dict(msg)
            messages.append(assistant_msg)
            answer = str(
                getattr(msg, "content", None) or assistant_msg.get("content") or ""
            ).strip()
        if not answer:
            raise RuntimeError("chain agent did not produce a final answer")
        structured_answer, parse_errors = _parse_chain_structured_answer(task, answer)
        answer_repair: dict[str, Any] = {"attempted": False, "accepted": False}
        if (
            isinstance(structured_answer, dict)
            and _selected_follow_only_errors(parse_errors)
        ):
            answer_repair = {
                "attempted": True,
                "accepted": False,
                "original_parse_errors": list(parse_errors),
            }
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Perform one structure-only JSON correction. Preserve every edge id, "
                        "action, action probability, seed prediction, no-valid-seed decision, "
                        "and visited state exactly. Change only selected_follow_edges and "
                        "derived path/list fields so each selected_follow_edges equals exactly "
                        "all predictions whose action is follow and whose follow probability "
                        "meets that state's threshold. Return the complete corrected JSON only."
                    ),
                }
            )
            repair_response = llm.complete(
                messages,
                tools=None,
                temperature=0.0,
                tool_choice=None,
                max_tokens=CHAIN_LLM_OUTPUT_LENGTH,
            )
            add_token_usage(agent_token_usage, response_token_usage(repair_response))
            repair_msg = repair_response.choices[0].message
            repair_message = _message_to_dict(repair_msg)
            messages.append(repair_message)
            repaired_answer = str(
                getattr(repair_msg, "content", None)
                or repair_message.get("content")
                or ""
            ).strip()
            repaired, repair_errors = _parse_chain_structured_answer(
                task, repaired_answer
            )
            signature_preserved = bool(
                isinstance(repaired, dict)
                and _decision_signature(task, repaired)
                == _decision_signature(task, structured_answer)
            )
            answer_repair.update(
                {
                    "repair_parse_errors": list(repair_errors),
                    "decision_signature_preserved": signature_preserved,
                }
            )
            if isinstance(repaired, dict) and not repair_errors and signature_preserved:
                answer = repaired_answer
                structured_answer = repaired
                parse_errors = []
                answer_repair["accepted"] = True
        parse_error = "; ".join(parse_errors) if parse_errors else None
        protocol_errors = validate_chain_tool_protocol(
            task, tool_trace, structured_answer
        )
        session["answer"] = answer
        session["structured_answer"] = structured_answer
        session["parse_error"] = parse_error
        session["answer_repair"] = answer_repair
        session["protocol_errors"] = protocol_errors
        return ChainAgentResult(
            task_id=str(task.get("id")),
            question=str(task.get("question") or ""),
            answer=answer,
            context=context,
            session_id=session_id,
            case_id=task.get("case_id"),
            dataset_version=task.get("dataset_version"),
            agent_llm=LLM_NAME,
            agent_token_usage=dict(agent_token_usage),
            agent_profile=profile,
            tool_trace=tool_trace,
            protocol_errors=protocol_errors,
            structured_answer=structured_answer,
            answer_repair=answer_repair,
            parse_error=parse_error,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        fallback = "Agent LLM failed; no grounded answer was produced."
        session["answer"] = fallback
        session["error"] = err
        return ChainAgentResult(
            task_id=str(task.get("id")),
            question=str(task.get("question") or ""),
            answer=fallback,
            context=context,
            session_id=session_id,
            case_id=task.get("case_id"),
            dataset_version=task.get("dataset_version"),
            agent_llm=LLM_NAME,
            agent_token_usage=dict(agent_token_usage),
            agent_profile=profile,
            tool_trace=tool_trace,
            error=err,
        )
    finally:
        session["messages"] = messages
        session["finished_at"] = datetime.now(timezone.utc)
        if save_session:
            mongo_error = save_chain_session(session, strict=strict_mongo)
            if mongo_error:
                session["mongo_error"] = mongo_error


def answer_question(
    question: str,
    *,
    case: str,
    dataset_version: str | None = None,
    save_session: bool = True,
    detector_tool_methods: list[str] | tuple[str, ...] | None = None,
    detector_tool_root: Path | None = None,
) -> ChainAgentResult:
    store = ChainTaskStore(dataset_version=dataset_version)
    tasks = store.tasks(case)
    if not tasks:
        raise ValueError(f"no chain QA tasks found for {case}")
    query_tokens = set(tokenize(question))
    best = max(
        tasks,
        key=lambda task: len(query_tokens & set(tokenize(str(task.get("question") or "")))),
    )
    return answer_chain_task(
        best,
        task_store=store,
        save_session=save_session,
        detector_tool_methods=detector_tool_methods,
        detector_tool_root=detector_tool_root,
    )


def sample_chain_tasks(tasks: list[dict[str, Any]], k: int | None, seed: int) -> list[dict[str, Any]]:
    if not k or k >= len(tasks):
        return tasks
    import random

    rng = random.Random(seed)
    return [tasks[index] for index in sorted(rng.sample(range(len(tasks)), k))]


def dataset_version_sort_key(value: str) -> tuple[Any, ...]:
    from chain_qa_version import dataset_version_sort_key as shared_sort_key

    return shared_sort_key(value)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("answer does not contain a JSON object")


def parse_transaction_existence_answer(
    task: dict[str, Any], answer: str
) -> tuple[dict[str, Any] | None, list[str]]:
    del task
    try:
        data = extract_json_object(answer)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, [str(exc)]
    errors: list[str] = []
    required_fields = {
        "exists",
        "transaction_count",
        "matching_edge_ids",
        "matching_tx_hashes",
    }
    missing = required_fields - set(data)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    unexpected = set(data) - required_fields
    if unexpected:
        errors.append(f"unexpected fields: {sorted(unexpected)}")
    exists = data.get("exists")
    if not isinstance(exists, bool):
        errors.append("exists must be boolean")
    count = data.get("transaction_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        errors.append("transaction_count must be a non-negative integer")
        count = 0

    normalized_lists: dict[str, list[str]] = {}
    for field in ("matching_edge_ids", "matching_tx_hashes"):
        values = data.get(field)
        if not isinstance(values, list) or not all(
            isinstance(value, (str, int)) and not isinstance(value, bool)
            for value in values or []
        ):
            errors.append(f"{field} must be a list of ids")
            normalized_lists[field] = []
            continue
        normalized = [str(value).lower() for value in values]
        if len(normalized) != len(set(normalized)):
            errors.append(f"{field} contains duplicates")
        normalized_lists[field] = normalized
    tx_hashes = normalized_lists["matching_tx_hashes"]
    invalid_hashes = [value for value in tx_hashes if not TX_RE.fullmatch(value)]
    if invalid_hashes:
        errors.append("matching_tx_hashes contains invalid transaction hashes")
    if isinstance(exists, bool) and exists != (count > 0):
        errors.append("exists must match transaction_count")
    if count != len(set(tx_hashes)):
        errors.append("transaction_count must equal unique matching_tx_hash count")
    if exists and not normalized_lists["matching_edge_ids"]:
        errors.append("exists=true requires matching_edge_ids")
    if exists is False and (
        normalized_lists["matching_edge_ids"] or normalized_lists["matching_tx_hashes"]
    ):
        errors.append("exists=false requires empty matching id lists")
    return {
        "exists": exists,
        "transaction_count": count,
        **normalized_lists,
    }, errors


def _transaction_protocol_details(
    task: dict[str, Any],
    tool_trace: list[dict[str, Any]],
    parsed_answer: dict[str, Any] | None,
) -> dict[str, Any]:
    context = task.get("graph_context") or {}
    required = {str(value) for value in task.get("required_tools") or []}
    allowed = {str(value) for value in task.get("allowed_tools") or []}
    called = {str(entry.get("tool") or "") for entry in tool_trace}
    successful = {
        str(entry.get("tool") or "")
        for entry in tool_trace
        if entry.get("status") == "ok"
    }
    errors: list[str] = []
    missing_required = required - successful
    if missing_required:
        errors.append(f"required tools were not called successfully: {sorted(missing_required)}")
    disallowed = called - allowed
    if disallowed:
        errors.append(f"disallowed tools were called: {sorted(disallowed)}")

    expected_src = str(context.get("src") or "").lower()
    expected_direction = str(context.get("direction") or "out")
    expected_types = {int(value) for value in context.get("edge_types") or []}
    valid_types: set[int] = set()
    neighbor_calls = [entry for entry in tool_trace if entry.get("tool") == "eth_neighbors"]
    valid_neighbor_calls = 0
    observed_edges: set[str] = set()
    observed_hashes: set[str] = set()
    observed_count = 0
    for entry in neighbor_calls:
        args = entry.get("arguments") or {}
        call_errors = []
        if str(args.get("address") or "").lower() != expected_src:
            call_errors.append("wrong source address")
        if args.get("direction") != expected_direction:
            call_errors.append("wrong direction")
        for field in ("block_min", "block_max"):
            try:
                actual = int(args.get(field))
                expected = int(context.get(field))
            except (TypeError, ValueError):
                actual, expected = None, None
            if actual != expected:
                call_errors.append(f"{field} not preserved")
        try:
            edge_type = int(args.get("edge_type"))
        except (TypeError, ValueError):
            edge_type = None
        if edge_type not in expected_types:
            call_errors.append("wrong edge_type")
        if context.get("require_tx_hashes") and args.get("with_raw") is not True:
            call_errors.append("with_raw=true is required")
        summary = entry.get("result_summary") or {}
        if entry.get("status") != "ok" or summary.get("query_complete") is not True:
            call_errors.append("neighbor query was not complete")
        if call_errors:
            errors.append(f"eth_neighbors protocol error: {', '.join(call_errors)}")
            continue
        valid_neighbor_calls += 1
        if edge_type is not None:
            valid_types.add(edge_type)
        observed_edges.update(str(value).lower() for value in summary.get("matching_edge_ids") or [])
        observed_hashes.update(str(value).lower() for value in summary.get("matching_tx_hashes") or [])
        observed_count += int(summary.get("transaction_count") or 0)
    if expected_types - valid_types:
        errors.append(f"edge types were not queried successfully: {sorted(expected_types - valid_types)}")

    tool_groundedness = 0.0
    if parsed_answer is None:
        errors.append("invalid structured answer")
    else:
        answer_edges = set(parsed_answer.get("matching_edge_ids") or [])
        answer_hashes = set(parsed_answer.get("matching_tx_hashes") or [])
        answer_count = int(parsed_answer.get("transaction_count") or 0)
        grounded = (
            answer_edges == observed_edges
            and answer_hashes == observed_hashes
            and answer_count == len(observed_hashes)
            and bool(parsed_answer.get("exists")) == bool(observed_hashes)
        )
        tool_groundedness = 1.0 if grounded else 0.0
        if not grounded:
            errors.append("answer does not exactly match successful tool results")
    tool_errors = sum(1 for entry in tool_trace if entry.get("status") != "ok")
    unnecessary = sum(1 for entry in tool_trace if entry.get("tool") not in required)
    return {
        "errors": errors,
        "tool_use_rate": 1.0 if tool_trace else 0.0,
        "required_tool_success_rate": (
            len(required & successful) / len(required) if required else 1.0
        ),
        "query_parameter_accuracy": (
            valid_neighbor_calls / len(neighbor_calls) if neighbor_calls else 0.0
        ),
        "unnecessary_tool_rate": unnecessary / len(tool_trace) if tool_trace else 0.0,
        "tool_error_rate": tool_errors / len(tool_trace) if tool_trace else 0.0,
        "tool_result_groundedness": tool_groundedness,
        "average_tool_calls": float(len(tool_trace)),
        "observed_matching_edge_ids": sorted(observed_edges),
        "observed_matching_tx_hashes": sorted(observed_hashes),
        "observed_transaction_count": observed_count,
    }


def _candidate_edge_groups(task: dict[str, Any]) -> dict[str, list[str]]:
    context = task.get("graph_context") or {}
    groups: dict[str, list[str]] = {}

    def add_groups(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for candidate_id, raw_ids in value.items():
            normalized = [str(raw_id) for raw_id in raw_ids or [] if str(raw_id)]
            if normalized:
                groups.setdefault(str(candidate_id), []).extend(normalized)

    add_groups(context.get("candidate_edge_groups"))
    for state in context.get("states") or []:
        if isinstance(state, dict):
            add_groups(state.get("candidate_edge_groups"))
    return {
        candidate_id: list(dict.fromkeys(raw_ids))
        for candidate_id, raw_ids in groups.items()
    }


def _tracing_protocol_details(
    task: dict[str, Any],
    tool_trace: list[dict[str, Any]],
    parsed_answer: dict[str, Any] | None,
) -> dict[str, Any]:
    context = task.get("graph_context") or {}
    contract = context.get("tool_query_contract") or {}
    required = {str(value) for value in task.get("required_tools") or []}
    allowed = {str(value) for value in task.get("allowed_tools") or []}
    evidence_trace = [
        entry for entry in tool_trace if entry.get("tool") != "detector_predict"
    ]
    called = {str(entry.get("tool") or "") for entry in evidence_trace}
    successful = {
        str(entry.get("tool") or "")
        for entry in evidence_trace
        if entry.get("status") == "ok"
    }
    groups = _candidate_edge_groups(task)
    raw_to_candidates: dict[str, set[str]] = {}
    for candidate_id, raw_ids in groups.items():
        for raw_id in raw_ids:
            canonical_id = _canonical_eth_edge_id(raw_id)
            if canonical_id is not None:
                raw_to_candidates.setdefault(canonical_id, set()).add(candidate_id)

    errors: list[str] = []
    missing_required = required - successful
    if missing_required:
        errors.append(f"required tools were not called successfully: {sorted(missing_required)}")
    disallowed = called - allowed
    if disallowed:
        errors.append(f"disallowed tools were called: {sorted(disallowed)}")

    covered_candidates: set[str] = set()
    valid_calls = 0
    unnecessary_calls = 0
    seen_raw_ids: set[str] = set()
    edge_calls = [entry for entry in evidence_trace if entry.get("tool") == "eth_edge"]
    for entry in edge_calls:
        args = entry.get("arguments") or {}
        raw_id = _canonical_eth_edge_id(args.get("edge_id"))
        call_errors: list[str] = []
        if _required_true(args.get("with_raw")) is not True:
            call_errors.append("with_raw=true is required")
        if raw_id is None:
            call_errors.append("invalid edge_id")
        elif raw_id not in raw_to_candidates:
            call_errors.append("edge_id is outside public candidate groups")
        if entry.get("status") != "ok":
            call_errors.append("edge query failed")
        if raw_id is not None and raw_id in seen_raw_ids:
            call_errors.append("duplicate edge query")
        if call_errors:
            errors.append(f"eth_edge protocol error: {', '.join(call_errors)}")
            unnecessary_calls += 1
        else:
            valid_calls += 1
            covered_candidates.update(raw_to_candidates[raw_id])
        if raw_id is not None:
            seen_raw_ids.add(raw_id)

    uncovered = sorted(set(groups) - covered_candidates)
    if uncovered:
        errors.append(f"candidate groups lack successful raw-edge evidence: {uncovered}")
    try:
        max_tool_calls = int(contract.get("max_tool_calls"))
    except (TypeError, ValueError):
        max_tool_calls = -1
    if max_tool_calls >= 0 and len(evidence_trace) > max_tool_calls:
        errors.append(
            f"tool call budget exceeded: {len(evidence_trace)} > {max_tool_calls}"
        )
    if parsed_answer is None:
        errors.append("invalid structured answer")

    tool_errors = sum(1 for entry in evidence_trace if entry.get("status") != "ok")
    coverage = len(covered_candidates) / len(groups) if groups else 0.0
    return {
        "errors": errors,
        "tool_use_rate": 1.0 if evidence_trace else 0.0,
        "required_tool_success_rate": (
            len(required & successful) / len(required) if required else 1.0
        ),
        "query_parameter_accuracy": (
            valid_calls / len(edge_calls) if edge_calls else 0.0
        ),
        "unnecessary_tool_rate": (
            unnecessary_calls / len(evidence_trace) if evidence_trace else 0.0
        ),
        "tool_error_rate": tool_errors / len(evidence_trace) if evidence_trace else 0.0,
        "tool_result_groundedness": coverage,
        "candidate_evidence_coverage": coverage,
        "average_tool_calls": float(len(evidence_trace)),
        "auxiliary_detector_calls": float(
            sum(1 for entry in tool_trace if entry.get("tool") == "detector_predict")
        ),
        "covered_candidate_ids": sorted(covered_candidates),
        "uncovered_candidate_ids": uncovered,
    }


def _protocol_details(
    task: dict[str, Any],
    tool_trace: list[dict[str, Any]],
    parsed_answer: dict[str, Any] | None,
) -> dict[str, Any]:
    if task.get("task_type") == "direct_transaction_existence":
        return _transaction_protocol_details(task, tool_trace, parsed_answer)
    if task.get("task_type") in {
        "edge_action_classification",
        "path_completion",
        "subgraph_noise_filtering",
    }:
        return _tracing_protocol_details(task, tool_trace, parsed_answer)
    return {
        "errors": [],
        "tool_use_rate": 1.0 if tool_trace else 0.0,
        "required_tool_success_rate": 1.0,
        "query_parameter_accuracy": 1.0,
        "unnecessary_tool_rate": 0.0,
        "tool_error_rate": 0.0,
        "tool_result_groundedness": 1.0,
        "average_tool_calls": float(len(tool_trace)),
    }


def validate_chain_tool_protocol(
    task: dict[str, Any],
    tool_trace: list[dict[str, Any]],
    parsed_answer: dict[str, Any] | None,
) -> list[str]:
    if chain_agent_profile(task) != "tool_grounded":
        return []
    return _protocol_details(task, tool_trace, parsed_answer)["errors"]


def _set_precision_recall(predicted: set[str], gold: set[str]) -> tuple[float, float]:
    if not predicted:
        precision = 1.0 if not gold else 0.0
    else:
        precision = len(predicted & gold) / len(predicted)
    recall = len(predicted & gold) / len(gold) if gold else (1.0 if not predicted else 0.0)
    return precision, recall


def score_transaction_existence_answer(
    task: dict[str, Any],
    answer: str,
    tool_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    parsed, parse_errors = parse_transaction_existence_answer(task, answer)
    valid_parsed = parsed if not parse_errors else None
    protocol = _protocol_details(task, tool_trace, valid_parsed)
    oracle = task.get("answer_value") or {}
    gold_edges = {str(value).lower() for value in oracle.get("matching_edge_ids") or []}
    gold_hashes = {str(value).lower() for value in oracle.get("matching_tx_hashes") or []}
    predicted_edges = set((valid_parsed or {}).get("matching_edge_ids") or [])
    predicted_hashes = set((valid_parsed or {}).get("matching_tx_hashes") or [])
    edge_precision, edge_recall = _set_precision_recall(predicted_edges, gold_edges)
    tx_precision, tx_recall = _set_precision_recall(predicted_hashes, gold_hashes)
    existence_correct = bool(valid_parsed) and valid_parsed.get("exists") == bool(oracle.get("exists"))
    count_correct = bool(valid_parsed) and int(valid_parsed.get("transaction_count") or 0) == int(
        oracle.get("transaction_count") or 0
    )
    task_pass = bool(
        valid_parsed
        and not parse_errors
        and existence_correct
        and count_correct
        and predicted_edges == gold_edges
        and predicted_hashes == gold_hashes
    )
    protocol_pass = not protocol["errors"]
    overall_pass = task_pass and protocol_pass
    negative_type = oracle.get("negative_sample_type")
    task_metrics = {
        "existence_accuracy": float(existence_correct),
        "transaction_count_accuracy": float(count_correct),
        "edge_id_precision": edge_precision,
        "edge_id_recall": edge_recall,
        "tx_hash_precision": tx_precision,
        "tx_hash_recall": tx_recall,
        "positive_accuracy": float(existence_correct) if oracle.get("exists") else None,
        "negative_accuracy": float(existence_correct) if not oracle.get("exists") else None,
        "direction_reversal_accuracy": (
            float(existence_correct) if negative_type == "reversed_direction" else None
        ),
    }
    tool_metrics = {
        key: protocol[key]
        for key in (
            "tool_use_rate",
            "required_tool_success_rate",
            "query_parameter_accuracy",
            "unnecessary_tool_rate",
            "tool_error_rate",
            "tool_result_groundedness",
            "average_tool_calls",
        )
    }
    errors = parse_errors + protocol["errors"]
    return {
        "scorer": "deterministic_transaction_existence_v1",
        "retrieval_relevance": 5.0 * tool_metrics["tool_result_groundedness"],
        "answer_correctness": 5.0 if task_pass else 0.0,
        "groundedness": 5.0 if protocol_pass else 0.0,
        "pass": overall_pass,
        "task_pass": task_pass,
        "tool_protocol_pass": protocol_pass,
        "overall_pass": overall_pass,
        "rationale": "passed deterministic task and tool protocol" if overall_pass else "; ".join(errors),
        "missing_evidence": [],
        "unsupported_claims": [],
        "parse_errors": parse_errors,
        "protocol_errors": protocol["errors"],
        "parsed_answer": valid_parsed,
        "deterministic_metrics": {**task_metrics, **tool_metrics},
        "task_metrics": task_metrics,
        "tool_metrics": tool_metrics,
    }


def parse_edge_action_answer(
    task: dict[str, Any], answer: str
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        data = extract_json_object(answer)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, [str(exc)]
    context = task.get("graph_context") or {}
    candidate_ids = [str(value) for value in context.get("candidate_edges") or []]
    candidate_set = set(candidate_ids)
    if not candidate_ids:
        errors.append("graph_context.candidate_edges must not be empty")
    if len(candidate_ids) != len(candidate_set):
        errors.append("graph_context.candidate_edges contains duplicates")
    predictions = data.get("predictions")
    if not isinstance(predictions, list):
        return data, ["predictions must be a list"]
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            errors.append(f"predictions[{index}] must be an object")
            continue
        edge_id = str(prediction.get("edge_id") or "")
        if edge_id not in candidate_set:
            errors.append(f"unknown candidate edge_id: {edge_id}")
            continue
        if edge_id in seen:
            errors.append(f"duplicate candidate edge_id: {edge_id}")
            continue
        seen.add(edge_id)
        action = str(prediction.get("action") or "").lower()
        if action not in CHAIN_ACTIONS:
            errors.append(f"invalid action for {edge_id}: {action}")
        raw_probabilities = prediction.get("action_probabilities")
        if not isinstance(raw_probabilities, dict):
            errors.append(f"missing action_probabilities for {edge_id}")
            continue
        probabilities: dict[str, float] = {}
        for name in CHAIN_ACTIONS:
            try:
                value = float(raw_probabilities[name])
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid probability {name} for {edge_id}")
                value = 0.0
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                errors.append(f"probability out of range {name} for {edge_id}")
            probabilities[name] = value
        if abs(sum(probabilities.values()) - 1.0) > 0.02:
            errors.append(f"probabilities do not sum to 1 for {edge_id}")
        max_probability = max(probabilities.values(), default=0.0)
        if action in CHAIN_ACTIONS and probabilities[action] + 1e-9 < max_probability:
            errors.append(f"action is not argmax probability for {edge_id}")
        normalized.append(
            {
                "edge_id": edge_id,
                "action": action,
                "action_probabilities": probabilities,
            }
        )
    missing = candidate_set - seen
    if missing:
        errors.append(f"missing candidate predictions: {sorted(missing)}")
    selected = data.get("selected_follow_edges")
    if not isinstance(selected, list):
        errors.append("selected_follow_edges must be a list")
        selected_ids: list[str] = []
    else:
        selected_ids = [str(value) for value in selected]
        if len(selected_ids) != len(set(selected_ids)):
            errors.append("selected_follow_edges contains duplicates")
        unknown_selected = set(selected_ids) - candidate_set
        if unknown_selected:
            errors.append(f"selected_follow_edges contains unknown candidates: {sorted(unknown_selected)}")
    try:
        threshold = float(context.get("follow_threshold", 0.5))
    except (TypeError, ValueError):
        errors.append("graph_context.follow_threshold must be numeric")
        threshold = 0.5
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        errors.append("graph_context.follow_threshold must be within [0, 1]")
        threshold = 0.5
    expected_selected = {
        prediction["edge_id"]
        for prediction in normalized
        if prediction["action"] == "follow"
        and prediction["action_probabilities"]["follow"] >= threshold
    }
    if set(selected_ids) != expected_selected:
        errors.append(
            "selected_follow_edges must equal follow predictions meeting the threshold"
        )
    return {
        "predictions": normalized,
        "selected_follow_edges": selected_ids,
    }, errors


def _macro_f1(gold: dict[str, str], predicted: dict[str, str]) -> float:
    scores = []
    active_actions = {
        action for action in CHAIN_ACTIONS if action in set(gold.values()) | set(predicted.values())
    }
    for action in active_actions:
        true_positive = sum(
            1 for edge_id, value in gold.items() if value == action and predicted.get(edge_id) == action
        )
        false_positive = sum(
            1 for edge_id, value in predicted.items() if value == action and gold.get(edge_id) != action
        )
        false_negative = sum(
            1 for edge_id, value in gold.items() if value == action and predicted.get(edge_id) != action
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def score_edge_action_answer(
    task: dict[str, Any], answer: str
) -> dict[str, Any]:
    parsed, errors = parse_edge_action_answer(task, answer)
    if parsed is None or errors:
        return {
            "scorer": "deterministic_edge_action_v1",
            "retrieval_relevance": 0.0,
            "answer_correctness": 0.0,
            "groundedness": 0.0,
            "pass": False,
            "rationale": "; ".join(errors) or "invalid structured answer",
            "missing_evidence": [],
            "unsupported_claims": [],
            "parse_errors": errors,
            "deterministic_metrics": {},
        }
    gold_value = task.get("answer_value") or {}
    gold_actions = {
        str(edge_id): str(action)
        for edge_id, action in (gold_value.get("candidate_actions") or {}).items()
    }
    gold_follow = {
        str(edge_id) for edge_id in gold_value.get("correct_follow_edges") or []
    }
    candidate_set = {
        str(edge_id)
        for edge_id in (task.get("graph_context") or {}).get("candidate_edges") or []
    }
    dataset_errors = []
    if set(gold_actions) != candidate_set:
        dataset_errors.append("gold candidate_actions do not match public candidates")
    if not gold_follow <= candidate_set:
        dataset_errors.append("gold correct_follow_edges contain unknown candidates")
    if dataset_errors:
        return {
            "scorer": "deterministic_edge_action_v1",
            "retrieval_relevance": 0.0,
            "answer_correctness": 0.0,
            "groundedness": 0.0,
            "pass": False,
            "rationale": "; ".join(dataset_errors),
            "missing_evidence": [],
            "unsupported_claims": [],
            "parse_errors": [],
            "dataset_errors": dataset_errors,
            "deterministic_metrics": {},
        }
    predictions = parsed["predictions"]
    predicted_actions = {
        prediction["edge_id"]: prediction["action"] for prediction in predictions
    }
    selected = set(parsed["selected_follow_edges"])
    ranked = sorted(
        predictions,
        key=lambda prediction: (
            -prediction["action_probabilities"]["follow"],
            prediction["edge_id"],
        ),
    )
    context = task.get("graph_context") or {}
    try:
        requested_top_k = int(context.get("top_k") or 1)
    except (TypeError, ValueError):
        requested_top_k = 1
    top_k = max(1, min(requested_top_k, len(ranked)))
    top_k_values = [value for value in (1, 3, 5) if value <= len(ranked)]
    hits = {
        f"hit@{value}": (
            bool({item["edge_id"] for item in ranked[:value]} & gold_follow)
            if gold_follow
            else None
        )
        for value in top_k_values
    }
    if gold_follow:
        binary_pass = bool(
            {item["edge_id"] for item in ranked[:top_k]} & gold_follow
        )
        reciprocal_rank = next(
            (
                1.0 / rank
                for rank, item in enumerate(ranked, start=1)
                if item["edge_id"] in gold_follow
            ),
            0.0,
        )
        none_of_above_accuracy = None
    else:
        binary_pass = not selected
        reciprocal_rank = None
        none_of_above_accuracy = 1.0 if not selected else 0.0
    action_accuracy = (
        sum(
            1 for edge_id, action in gold_actions.items() if predicted_actions.get(edge_id) == action
        )
        / len(gold_actions)
        if gold_actions
        else 0.0
    )
    false_follow = selected - gold_follow
    over_expansion_rate = len(false_follow) / len(selected) if selected else 0.0
    brier = 0.0
    if predictions:
        brier = sum(
            sum(
                (
                    prediction["action_probabilities"][action]
                    - (1.0 if gold_actions.get(prediction["edge_id"]) == action else 0.0)
                )
                ** 2
                for action in CHAIN_ACTIONS
            )
            for prediction in predictions
        ) / len(predictions)
    metrics = {
        "binary_top_k_pass": binary_pass,
        "top_k": top_k,
        **hits,
        "action_accuracy": action_accuracy,
        "action_macro_f1": _macro_f1(gold_actions, predicted_actions),
        "over_expansion_rate": over_expansion_rate,
        "none_of_above_accuracy": none_of_above_accuracy,
        "mrr": reciprocal_rank,
        "brier_score": brier,
    }
    return {
        "scorer": "deterministic_edge_action_v1",
        "retrieval_relevance": 5.0 if binary_pass else 0.0,
        "answer_correctness": round(5.0 * action_accuracy, 6),
        "groundedness": 5.0,
        "pass": binary_pass,
        "rationale": (
            f"deterministic top-{top_k} decision {'passed' if binary_pass else 'failed'}; "
            f"action_accuracy={action_accuracy:.3f}"
        ),
        "missing_evidence": [],
        "unsupported_claims": [],
        "parse_errors": [],
        "parsed_answer": parsed,
        "deterministic_metrics": metrics,
    }


def parse_sequential_policy_answer(
    task: dict[str, Any], answer: str
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        data = extract_json_object(answer)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, [str(exc)]
    context = task.get("graph_context") or {}
    states = {
        str(state.get("state_id") or ""): state
        for state in context.get("states") or []
        if isinstance(state, dict) and state.get("state_id")
    }
    errors: list[str] = []
    if not states:
        errors.append("graph_context.states must not be empty")
    normalized_modes: dict[str, dict[str, Any]] = {}
    if task.get("task_type") == "subgraph_noise_filtering":
        required_modes = ["state_policy"]
    else:
        required_modes = [str(value) for value in context.get("evaluation_modes") or []]
        if not required_modes:
            required_modes = ["teacher_forced", "free_rollout"]
    for mode in required_modes:
        raw_mode = data.get(mode)
        if not isinstance(raw_mode, dict):
            errors.append(f"{mode} must be an object")
            continue
        raw_steps = raw_mode.get("steps")
        if not isinstance(raw_steps, list):
            errors.append(f"{mode}.steps must be a list")
            continue
        normalized_steps: list[dict[str, Any]] = []
        seen_states: set[str] = set()
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, dict):
                errors.append(f"{mode}.steps[{index}] must be an object")
                continue
            state_id = str(raw_step.get("state_id") or "")
            state = states.get(state_id)
            if state is None:
                errors.append(f"{mode}.steps[{index}] has unknown state_id: {state_id}")
                continue
            if state_id in seen_states:
                errors.append(f"{mode} contains duplicate state_id: {state_id}")
                continue
            seen_states.add(state_id)
            current_node = str(raw_step.get("current_node") or "").lower()
            expected_node = str(state.get("current_node") or "").lower()
            if current_node != expected_node:
                errors.append(f"{mode}.{state_id} current_node does not match public state")
            step_task = {
                "graph_context": {
                    "candidate_edges": state.get("candidate_edges") or [],
                    "follow_threshold": context.get("follow_threshold", 0.5),
                }
            }
            parsed_step, step_errors = parse_edge_action_answer(
                step_task, json.dumps(raw_step, ensure_ascii=False)
            )
            errors.extend(f"{mode}.{state_id}: {error}" for error in step_errors)
            if parsed_step is not None:
                normalized_steps.append(
                    {
                        "state_id": state_id,
                        "current_node": expected_node,
                        **parsed_step,
                    }
                )
        raw_paths = raw_mode.get("paths") or []
        if not isinstance(raw_paths, list):
            errors.append(f"{mode}.paths must be a list")
            raw_paths = []
        all_candidates = {
            str(edge_id)
            for state in states.values()
            for edge_id in state.get("candidate_edges") or []
        }
        normalized_paths: list[dict[str, Any]] = []
        for index, raw_path in enumerate(raw_paths):
            if not isinstance(raw_path, dict):
                errors.append(f"{mode}.paths[{index}] must be an object")
                continue
            edge_ids = [str(value) for value in raw_path.get("edge_ids") or []]
            unknown = set(edge_ids) - all_candidates
            if unknown:
                errors.append(f"{mode}.paths[{index}] contains unknown edges: {sorted(unknown)}")
            try:
                probability = float(raw_path.get("probability", 0.0))
            except (TypeError, ValueError):
                probability = -1.0
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                errors.append(f"{mode}.paths[{index}] probability must be within [0, 1]")
            normalized_paths.append(
                {
                    "edge_ids": edge_ids,
                    "node_ids": [str(value).lower() for value in raw_path.get("node_ids") or []],
                    "probability": probability,
                    "terminal_reason": str(raw_path.get("terminal_reason") or ""),
                }
            )
        normalized_modes[mode] = {"steps": normalized_steps, "paths": normalized_paths}

    seed_candidates = [
        str(value).lower() for value in context.get("candidate_seed_nodes") or []
    ]
    seed_predictions: list[dict[str, Any]] = []
    selected_seeds: list[str] = []
    no_valid_seed = bool(data.get("no_valid_seed", False))
    if task.get("task_type") == "subgraph_noise_filtering":
        raw_seed_predictions = data.get("seed_predictions")
        if not isinstance(raw_seed_predictions, list):
            errors.append("seed_predictions must be a list")
            raw_seed_predictions = []
        seen_seeds: set[str] = set()
        for index, prediction in enumerate(raw_seed_predictions):
            if not isinstance(prediction, dict):
                errors.append(f"seed_predictions[{index}] must be an object")
                continue
            node_id = str(prediction.get("node_id") or "").lower()
            if node_id not in seed_candidates:
                errors.append(f"unknown candidate seed: {node_id}")
                continue
            if node_id in seen_seeds:
                errors.append(f"duplicate candidate seed: {node_id}")
                continue
            seen_seeds.add(node_id)
            try:
                probability = float(prediction.get("probability"))
            except (TypeError, ValueError):
                probability = -1.0
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                errors.append(f"seed probability out of range: {node_id}")
            seed_predictions.append({"node_id": node_id, "probability": probability})
        if set(seed_candidates) != seen_seeds:
            errors.append("seed_predictions must include every candidate seed exactly once")
        raw_selected = data.get("selected_seed_nodes")
        if not isinstance(raw_selected, list):
            errors.append("selected_seed_nodes must be a list")
            raw_selected = []
        selected_seeds = [str(value).lower() for value in raw_selected]
        if len(selected_seeds) != len(set(selected_seeds)):
            errors.append("selected_seed_nodes contains duplicates")
        if set(selected_seeds) - set(seed_candidates):
            errors.append("selected_seed_nodes contains unknown seeds")
        seed_top_k = max(1, int(context.get("seed_top_k") or 1))
        if len(selected_seeds) > seed_top_k:
            errors.append("selected_seed_nodes exceeds seed_top_k")
        if no_valid_seed and selected_seeds:
            errors.append("no_valid_seed=true requires empty selected_seed_nodes")

    state_by_node = {
        str(state.get("current_node") or "").lower(): state_id
        for state_id, state in states.items()
    }
    candidate_dst = {
        str(summary.get("candidate_id") or summary.get("edge_id") or ""): str(
            summary.get("dst") or ""
        ).lower()
        for summary in task.get("edge_summaries") or []
        if isinstance(summary, dict)
    }

    def validate_reachable(mode: str, seed_nodes: list[str]) -> None:
        mode_value = normalized_modes.get(mode) or {}
        steps = mode_value.get("steps") or []
        reachable = {
            state_by_node[node]
            for node in seed_nodes
            if node in state_by_node
        }
        pending = list(steps)
        changed = True
        while changed:
            changed = False
            for step in list(pending):
                if step["state_id"] not in reachable:
                    continue
                pending.remove(step)
                changed = True
                for edge_id in step.get("selected_follow_edges") or []:
                    next_state = state_by_node.get(candidate_dst.get(edge_id, ""))
                    if next_state:
                        reachable.add(next_state)
        if pending:
            errors.append(
                f"{mode} contains states not reached from its seed decisions: "
                f"{sorted(step['state_id'] for step in pending)}"
            )

    if task.get("task_type") == "path_completion":
        teacher_states = {
            step["state_id"]
            for step in (normalized_modes.get("teacher_forced") or {}).get("steps") or []
        }
        if teacher_states != set(states):
            errors.append("teacher_forced must evaluate every public state exactly once")
        start_node = str(context.get("start_node") or "").lower()
        validate_reachable("free_rollout", [start_node])
    else:
        state_policy_states = {
            step["state_id"]
            for step in (normalized_modes.get("state_policy") or {}).get("steps") or []
        }
        if state_policy_states != set(states):
            errors.append("state_policy must evaluate every public state exactly once")

    budget = context.get("rollout_budget") or {}
    for mode, mode_value in normalized_modes.items():
        step_count = len(mode_value.get("steps") or [])
        inspected_edges = sum(
            len(step.get("predictions") or []) for step in mode_value.get("steps") or []
        )
        max_path_depth = max(
            (len(path.get("edge_ids") or []) for path in mode_value.get("paths") or []),
            default=0,
        )
        if step_count > int(budget.get("max_node_expansions") or step_count):
            errors.append(f"{mode} exceeds max_node_expansions")
        if inspected_edges > int(budget.get("max_inspected_edges") or inspected_edges):
            errors.append(f"{mode} exceeds max_inspected_edges")
        if max_path_depth > int(budget.get("max_depth") or max_path_depth):
            errors.append(f"{mode} exceeds max_depth")

    return {
        **normalized_modes,
        "seed_predictions": seed_predictions,
        "selected_seed_nodes": selected_seeds,
        "no_valid_seed": no_valid_seed,
    }, errors


def _binary_set_metrics(predicted: set[str], gold: set[str]) -> dict[str, float]:
    precision, recall = _set_precision_recall(predicted, gold)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "over_expansion_rate": (
            len(predicted - gold) / len(predicted) if predicted else 0.0
        ),
    }


def _follow_ece(samples: list[tuple[float, float]], bins: int = 10) -> float:
    if not samples:
        return 0.0
    total = len(samples)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        bucket = [
            (confidence, label)
            for confidence, label in samples
            if lower <= confidence <= upper
            and (bin_index == bins - 1 or confidence < upper)
        ]
        if not bucket:
            continue
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        mean_label = sum(item[1] for item in bucket) / len(bucket)
        error += len(bucket) / total * abs(mean_confidence - mean_label)
    return error


def _derive_snf_seed_modes(
    task: dict[str, Any], parsed: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Apply one public state policy to hidden-oracle and predicted seed sets."""
    context = task.get("graph_context") or {}
    states = [state for state in context.get("states") or [] if isinstance(state, dict)]
    state_by_id = {str(state.get("state_id") or ""): state for state in states}
    state_by_node = {
        str(state.get("current_node") or "").lower(): str(state.get("state_id") or "")
        for state in states
        if state.get("state_id") and state.get("current_node")
    }
    policy_steps = {
        str(step.get("state_id") or ""): step
        for step in (parsed.get("state_policy") or {}).get("steps") or []
        if isinstance(step, dict) and step.get("state_id")
    }
    candidate_dst = {
        str(summary.get("candidate_id") or summary.get("edge_id") or ""): str(
            summary.get("dst") or ""
        ).lower()
        for summary in task.get("edge_summaries") or []
        if isinstance(summary, dict)
    }
    budget = context.get("rollout_budget") or {}
    max_expansions = max(0, int(budget.get("max_node_expansions") or len(states)))
    max_depth = max(0, int(budget.get("max_depth") or len(states)))

    def rollout(seed_nodes: list[str]) -> dict[str, Any]:
        queue: list[tuple[str, int]] = []
        for node in seed_nodes:
            state_id = state_by_node.get(str(node).lower())
            if state_id and all(existing != state_id for existing, _ in queue):
                queue.append((state_id, 0))
        visited: set[str] = set()
        steps: list[dict[str, Any]] = []
        while queue and len(steps) < max_expansions:
            state_id, depth = queue.pop(0)
            if state_id in visited or state_id not in state_by_id:
                continue
            visited.add(state_id)
            step = policy_steps.get(state_id)
            if step is None:
                continue
            steps.append(step)
            if depth >= max_depth:
                continue
            for edge_id in step.get("selected_follow_edges") or []:
                next_state = state_by_node.get(candidate_dst.get(str(edge_id), ""))
                if next_state and next_state not in visited:
                    queue.append((next_state, depth + 1))
        return {"steps": steps, "paths": []}

    oracle_seeds = [
        str(value).lower()
        for value in (task.get("answer_value") or {}).get("gold_seed_nodes") or []
    ]
    predicted_seeds = [str(value).lower() for value in parsed.get("selected_seed_nodes") or []]
    return {
        "oracle_seed": rollout(oracle_seeds),
        "predicted_seed": rollout(predicted_seeds),
    }


def score_sequential_policy_answer(
    task: dict[str, Any],
    answer: str,
    tool_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    parsed, errors = parse_sequential_policy_answer(task, answer)
    if parsed is None or errors:
        return {
            "scorer": "deterministic_sequential_edge_policy_v1",
            "retrieval_relevance": 0.0,
            "answer_correctness": 0.0,
            "groundedness": 0.0,
            "pass": False,
            "task_pass": False,
            "rationale": "; ".join(errors) or "invalid structured answer",
            "missing_evidence": [],
            "unsupported_claims": [],
            "parse_errors": errors,
            "deterministic_metrics": {},
        }
    oracle = task.get("answer_value") or {}
    state_actions = {
        str(state_id): {str(edge_id): str(action) for edge_id, action in actions.items()}
        for state_id, actions in (oracle.get("state_action_labels") or {}).items()
        if isinstance(actions, dict)
    }
    gold_follow = {
        str(value) for value in oracle.get("gold_follow_edges") or []
    }
    context = task.get("graph_context") or {}
    required_modes = [str(value) for value in context.get("evaluation_modes") or []]
    if not required_modes:
        required_modes = (
            ["teacher_forced", "free_rollout"]
            if task.get("task_type") == "path_completion"
            else ["oracle_seed", "predicted_seed"]
        )
    scored_modes = (
        _derive_snf_seed_modes(task, parsed)
        if task.get("task_type") == "subgraph_noise_filtering"
        else {mode: parsed.get(mode) or {"steps": [], "paths": []} for mode in required_modes}
    )
    mode_metrics: dict[str, dict[str, Any]] = {}
    all_calibration_samples: list[tuple[float, float]] = []
    all_brier_terms: list[float] = []
    for mode in required_modes:
        mode_value = scored_modes.get(mode) or {"steps": [], "paths": []}
        predictions: dict[str, dict[str, Any]] = {}
        predicted_actions: dict[str, str] = {}
        selected: set[str] = set()
        gold_actions_for_steps: dict[str, str] = {}
        for step in mode_value.get("steps") or []:
            gold_for_state = state_actions.get(step["state_id"], {})
            for prediction in step.get("predictions") or []:
                edge_id = prediction["edge_id"]
                predictions[edge_id] = prediction
                predicted_actions[edge_id] = prediction["action"]
                if edge_id in gold_for_state:
                    gold_actions_for_steps[edge_id] = gold_for_state[edge_id]
                    follow_label = 1.0 if gold_for_state[edge_id] == "follow" else 0.0
                    follow_probability = prediction["action_probabilities"]["follow"]
                    all_calibration_samples.append((follow_probability, follow_label))
                    all_brier_terms.append(
                        sum(
                            (
                                prediction["action_probabilities"][action]
                                - (1.0 if gold_for_state[edge_id] == action else 0.0)
                            )
                            ** 2
                            for action in CHAIN_ACTIONS
                        )
                    )
            selected.update(step.get("selected_follow_edges") or [])
        relevant_gold = {
            edge_id for edge_id, action in gold_actions_for_steps.items() if action == "follow"
        }
        set_metrics = _binary_set_metrics(
            selected,
            gold_follow
            if task.get("task_type") == "subgraph_noise_filtering"
            else relevant_gold,
        )
        predicted_paths = {
            tuple(path.get("edge_ids") or []) for path in mode_value.get("paths") or []
        }
        gold_paths = {tuple(path) for path in oracle.get("gold_paths") or []}
        complete_path_success = bool(gold_paths) and bool(predicted_paths & gold_paths)
        action_accuracy = (
            sum(
                predicted_actions.get(edge_id) == action
                for edge_id, action in gold_actions_for_steps.items()
            )
            / len(gold_actions_for_steps)
            if gold_actions_for_steps
            else 0.0
        )
        mode_metrics[mode] = {
            **set_metrics,
            "action_accuracy": action_accuracy,
            "action_macro_f1": _macro_f1(gold_actions_for_steps, predicted_actions),
            "complete_path_success": complete_path_success,
            "node_expansions": len(mode_value.get("steps") or []),
            "inspected_edges": sum(
                len(step.get("predictions") or []) for step in mode_value.get("steps") or []
            ),
            "coverage_at_budget": (
                len(selected & gold_follow) / len(gold_follow) if gold_follow else float(not selected)
            ),
        }

    metrics: dict[str, Any] = {
        "brier_score": sum(all_brier_terms) / len(all_brier_terms) if all_brier_terms else 0.0,
        "ece": _follow_ece(all_calibration_samples),
        "tool_calls": len(tool_trace or []),
    }
    for mode, values in mode_metrics.items():
        metrics.update({f"{mode}_{key}": value for key, value in values.items()})

    if task.get("task_type") == "path_completion":
        free_metrics = mode_metrics.get("free_rollout", {})
        teacher_metrics = mode_metrics.get("teacher_forced", {})
        metrics["teacher_forced_next_step_accuracy"] = teacher_metrics.get(
            "action_accuracy", 0.0
        )
        task_pass = bool(free_metrics.get("complete_path_success"))
    else:
        gold_seeds = {str(value).lower() for value in oracle.get("gold_seed_nodes") or []}
        selected_seeds = set(parsed.get("selected_seed_nodes") or [])
        ranked_seeds = sorted(
            parsed.get("seed_predictions") or [],
            key=lambda item: (-item["probability"], item["node_id"]),
        )
        seed_mrr = next(
            (
                1.0 / rank
                for rank, item in enumerate(ranked_seeds, 1)
                if item["node_id"] in gold_seeds
            ),
            0.0,
        )
        seed_hit = bool(selected_seeds & gold_seeds) if gold_seeds else not selected_seeds
        seed_metrics = _binary_set_metrics(selected_seeds, gold_seeds)
        metrics.update(
            {
                "seed_precision": seed_metrics["precision"],
                "seed_recall": seed_metrics["recall"],
                "seed_f1": seed_metrics["f1"],
                "seed_mrr": seed_mrr,
                "no_valid_seed_accuracy": (
                    float(not selected_seeds and parsed.get("no_valid_seed"))
                    if oracle.get("no_valid_seed")
                    else None
                ),
            }
        )
        predicted_metrics = mode_metrics.get("predicted_seed", {})
        oracle_metrics = mode_metrics.get("oracle_seed", {})
        metrics["predicted_seed_end_to_end_f1"] = predicted_metrics.get("f1", 0.0)
        metrics["oracle_seed_end_to_end_f1"] = oracle_metrics.get("f1", 0.0)
        if oracle.get("no_valid_seed"):
            task_pass = bool(seed_hit and not any(
                step.get("selected_follow_edges")
                for step in (scored_modes.get("predicted_seed") or {}).get("steps") or []
            ))
        else:
            task_pass = bool(seed_hit and predicted_metrics.get("f1", 0.0) > 0.0)

    return {
        "scorer": "deterministic_sequential_edge_policy_v1",
        "retrieval_relevance": 5.0,
        "answer_correctness": 5.0 if task_pass else 0.0,
        "groundedness": 5.0,
        "pass": task_pass,
        "task_pass": task_pass,
        "rationale": "passed deterministic sequential policy scorer" if task_pass else "failed deterministic sequential policy scorer",
        "missing_evidence": [],
        "unsupported_claims": [],
        "parse_errors": [],
        "parsed_answer": {
            **parsed,
            **(scored_modes if task.get("task_type") == "subgraph_noise_filtering" else {}),
            **(
                {"rollout_derivation": "evaluator_seed_projection_v1"}
                if task.get("task_type") == "subgraph_noise_filtering"
                else {}
            ),
        },
        "deterministic_metrics": metrics,
        "mode_metrics": mode_metrics,
    }


def _attach_tool_protocol(
    task: dict[str, Any],
    judgement: dict[str, Any],
    tool_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    if chain_agent_profile(task) != "tool_grounded":
        return judgement
    protocol = _protocol_details(
        task,
        tool_trace,
        judgement.get("parsed_answer")
        if isinstance(judgement.get("parsed_answer"), dict)
        else None,
    )
    task_pass = bool(judgement.get("task_pass", judgement.get("pass")))
    tool_protocol_pass = not protocol["errors"]
    overall_pass = task_pass and tool_protocol_pass
    metrics = judgement.setdefault("deterministic_metrics", {})
    metrics.update(
        {
            key: value
            for key, value in protocol.items()
            if key != "errors"
        }
    )
    judgement["task_pass"] = task_pass
    judgement["tool_protocol_pass"] = tool_protocol_pass
    judgement["overall_pass"] = overall_pass
    judgement["pass"] = overall_pass
    judgement["protocol_errors"] = list(protocol["errors"])
    judgement["groundedness"] = round(
        5.0 * float(protocol.get("tool_result_groundedness") or 0.0), 6
    )
    if protocol["errors"]:
        detail = "; ".join(protocol["errors"])
        rationale = str(judgement.get("rationale") or "").strip()
        judgement["rationale"] = f"{rationale}; tool protocol: {detail}".strip("; ")
    return judgement


def _normalize_judgement(data: dict[str, Any]) -> dict[str, Any]:
    def score(key: str) -> float:
        try:
            return max(0.0, min(5.0, float(data.get(key) or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    rel = score("retrieval_relevance")
    corr = score("answer_correctness")
    grd = score("groundedness")
    passed = data.get("pass")
    if not isinstance(passed, bool):
        passed = rel >= 3 and corr >= 3 and grd >= 3
    return {
        "retrieval_relevance": rel,
        "answer_correctness": corr,
        "groundedness": grd,
        "pass": passed,
        "rationale": str(data.get("rationale") or ""),
        "missing_evidence": data.get("missing_evidence") if isinstance(data.get("missing_evidence"), list) else [],
        "unsupported_claims": data.get("unsupported_claims") if isinstance(data.get("unsupported_claims"), list) else [],
    }


def judge_chain_answer_with_usage(
    task: dict[str, Any], agent_result: ChainAgentResult
) -> tuple[dict[str, Any], dict[str, int]]:
    if task.get("task_type") == "direct_transaction_existence":
        return (
            score_transaction_existence_answer(
                task, agent_result.answer, agent_result.tool_trace or []
            ),
            empty_token_usage(),
        )
    if task.get("task_type") == "edge_action_classification":
        judgement = score_edge_action_answer(task, agent_result.answer)
        return (
            _attach_tool_protocol(task, judgement, agent_result.tool_trace or []),
            empty_token_usage(),
        )
    from eval_agent import call_judge, judge_settings

    if (
        task.get("task_type") in {"path_completion", "subgraph_noise_filtering"}
        and (task.get("graph_context") or {}).get("sequential_policy")
    ):
        deterministic = score_sequential_policy_answer(
            task, agent_result.answer, agent_result.tool_trace or []
        )
        public_task = project_snf_state_policy_public_task(
            {
                key: sanitize_chain_public_value(task.get(key))
                for key in CHAIN_PUBLIC_TASK_FIELDS
                if key in task
            }
        )
        raw_judgement = call_judge(
            test_item=public_task,
            agent_answer=agent_result.answer,
            retrieved_evidence=[
                {
                    "public_context": sanitize_chain_public_value(agent_result.context),
                    "sanitized_tool_trace": sanitize_chain_public_value(
                        agent_result.tool_trace or []
                    ),
                }
            ],
            settings=judge_settings(),
            system_prompt=CHAIN_PROCESS_JUDGE_SYSTEM_PROMPT,
            test_item_limit=30000,
            evidence_limit=60000,
        )
        usage = normalize_token_usage(raw_judgement.pop("llm_usage", None))
        raw_judgement.pop("llm_model", None)
        process_judgement = _normalize_judgement(raw_judgement)
        deterministic["retrieval_relevance"] = process_judgement[
            "retrieval_relevance"
        ]
        deterministic["groundedness"] = process_judgement["groundedness"]
        deterministic["judge_pass"] = process_judgement["pass"]
        deterministic["judge_assessment"] = process_judgement
        deterministic["evaluation_layers"] = {
            "primary": deterministic["scorer"],
            "supplementary": "llm_context_and_tool_process_v1",
        }
        deterministic = _attach_tool_protocol(
            task, deterministic, agent_result.tool_trace or []
        )
        return deterministic, usage

    raw_judgement = call_judge(
        test_item=task,
        agent_answer=agent_result.answer,
        retrieved_evidence=agent_result.context,
        settings=judge_settings(),
    )
    usage = normalize_token_usage(raw_judgement.pop("llm_usage", None))
    raw_judgement.pop("llm_model", None)
    return _normalize_judgement(raw_judgement), usage


def judge_chain_answer(task: dict[str, Any], agent_result: ChainAgentResult) -> dict[str, Any]:
    judgement, _ = judge_chain_answer_with_usage(task, agent_result)
    return judgement


def save_chain_eval_result(result: dict[str, Any], *, strict: bool = False) -> str | None:
    try:
        from storage import save_chain_agent_eval_result

        save_chain_agent_eval_result(result)
        return None
    except Exception as exc:
        if strict:
            raise
        return f"{type(exc).__name__}: {exc}"


def save_chain_eval_summary(summary: dict[str, Any], *, strict: bool = False) -> str | None:
    try:
        from storage import save_chain_agent_eval_summary

        save_chain_agent_eval_summary(summary)
        return None
    except Exception as exc:
        if strict:
            raise
        return f"{type(exc).__name__}: {exc}"


def evaluate_chain_task(
    *,
    index: int,
    total: int,
    task: dict[str, Any],
    task_store: ChainTaskStore,
    run_id: str,
    strict_mongo: bool,
    quiet: bool,
    detector_tool_methods: tuple[str, ...] = DEFAULT_DETECTOR_TOOL_METHODS,
    detector_tool_root: Path | None = None,
) -> dict[str, Any]:
    import time as _time
    from datetime import datetime, timezone

    if not quiet:
        print(f"[{index}/{total}] {task.get('id')}: {truncate(task.get('question'), 120)}", flush=True)
    started = _time.time()
    result = {
        "run_id": run_id,
        "index": index,
        "task_id": task.get("id"),
        "case_id": task.get("case_id"),
        "case_dir": task.get("case_dir"),
        "dataset_version": task.get("dataset_version"),
        "agent_llm": LLM_NAME,
        "judge_llm": JUDGE_LLM_NAME,
        "agent_token_usage": empty_token_usage(),
        "judge_token_usage": empty_token_usage(),
        "token_usage_complete": True,
        "task": task,
        "detector_tool_methods": list(detector_tool_methods),
        "agent": None,
        "judgement": None,
        "agent_session_id": None,
        "error": None,
        "created_at": datetime.now(timezone.utc),
    }
    try:
        agent_result = answer_chain_task(
            task,
            task_store=task_store,
            run_id=run_id,
            save_session=True,
            strict_mongo=strict_mongo,
            detector_tool_methods=detector_tool_methods,
            detector_tool_root=detector_tool_root,
        )
        result["agent"] = agent_result.as_dict()
        result["agent_session_id"] = agent_result.session_id
        result["agent_token_usage"] = normalize_token_usage(
            agent_result.agent_token_usage
        )
        if agent_result.error:
            result["error"] = agent_result.error
        else:
            judgement, judge_usage = judge_chain_answer_with_usage(task, agent_result)
            result["judgement"] = judgement
            result["judge_token_usage"] = judge_usage
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if not quiet:
            print(f"[error] {task.get('id')}: {result['error']}", file=sys.stderr, flush=True)
    result["elapsed_seconds"] = round(_time.time() - started, 3)
    mongo_error = save_chain_eval_result(result, strict=strict_mongo)
    if mongo_error:
        result["mongo_error"] = mongo_error
    return result


def summarize_result_token_usage(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    agent_usage = empty_token_usage()
    judge_usage = empty_token_usage()
    for item in results:
        add_token_usage(
            agent_usage, normalize_token_usage(item.get("agent_token_usage"))
        )
        add_token_usage(
            judge_usage, normalize_token_usage(item.get("judge_token_usage"))
        )
    all_usage = empty_token_usage()
    add_token_usage(all_usage, agent_usage)
    add_token_usage(all_usage, judge_usage)
    return {
        "agent_llm": agent_usage,
        "judge_llm": judge_usage,
        "all_llms": all_usage,
    }


def summarize_chain_eval(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if (item.get("judgement") or {}).get("pass"))
    errors = sum(1 for item in results if item.get("error"))
    keys = ["retrieval_relevance", "answer_correctness", "groundedness"]
    means = {
        key: (sum(float((item.get("judgement") or {}).get(key) or 0.0) for item in results) / total if total else 0.0)
        for key in keys
    }
    by_type: dict[str, dict[str, Any]] = {}
    by_profile: dict[str, dict[str, Any]] = {}
    deterministic_by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    judge_by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    deterministic_values: dict[str, list[float]] = defaultdict(list)
    for item in results:
        task_type = str((item.get("task") or {}).get("task_type") or "unknown")
        profile = str((item.get("task") or {}).get("agent_profile") or "context_only")
        bucket = by_type.setdefault(
            task_type, {"total": 0, "pass": 0, "judge_total": 0, "judge_pass": 0}
        )
        profile_bucket = by_profile.setdefault(
            profile,
            {
                "total": 0,
                "pass": 0,
                "task_pass": 0,
                "tool_protocol_pass": 0,
            },
        )
        bucket["total"] += 1
        profile_bucket["total"] += 1
        if (item.get("judgement") or {}).get("pass"):
            bucket["pass"] += 1
            profile_bucket["pass"] += 1
        judge_assessment = (item.get("judgement") or {}).get("judge_assessment")
        if isinstance(judge_assessment, dict):
            bucket["judge_total"] += 1
            bucket["judge_pass"] += int(bool(judge_assessment.get("pass")))
            for key in ("retrieval_relevance", "answer_correctness", "groundedness"):
                value = judge_assessment.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    judge_by_type[task_type][key].append(float(value))
        if profile == "tool_grounded" and (item.get("judgement") or {}).get("task_pass"):
            profile_bucket["task_pass"] += 1
        if profile == "tool_grounded" and (item.get("judgement") or {}).get("tool_protocol_pass"):
            profile_bucket["tool_protocol_pass"] += 1
        for key, value in (
            (item.get("judgement") or {}).get("deterministic_metrics") or {}
        ).items():
            if isinstance(value, bool):
                deterministic_values[key].append(float(value))
                deterministic_by_type[task_type][key].append(float(value))
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                deterministic_values[key].append(float(value))
                deterministic_by_type[task_type][key].append(float(value))
    for task_type, bucket in by_type.items():
        bucket["pass_rate"] = bucket["pass"] / bucket["total"] if bucket["total"] else 0.0
        bucket["deterministic_metric_means"] = {
            key: sum(values) / len(values)
            for key, values in sorted(deterministic_by_type[task_type].items())
            if values
        }
        bucket["judge_pass_rate"] = (
            bucket["judge_pass"] / bucket["judge_total"]
            if bucket["judge_total"]
            else None
        )
        bucket["judge_metric_means"] = {
            key: sum(values) / len(values)
            for key, values in sorted(judge_by_type[task_type].items())
            if values
        }
    for profile, bucket in by_profile.items():
        denominator = bucket["total"] or 1
        bucket["pass_rate"] = bucket["pass"] / denominator
        if profile == "tool_grounded":
            bucket["task_pass_rate"] = bucket["task_pass"] / denominator
            bucket["tool_protocol_pass_rate"] = bucket["tool_protocol_pass"] / denominator
        else:
            bucket.pop("task_pass", None)
            bucket.pop("tool_protocol_pass", None)
    return {
        "total": total,
        "pass": passed,
        "pass_rate": passed / total if total else 0.0,
        "errors": errors,
        "mean_scores": means,
        "by_task_type": by_type,
        "by_agent_profile": by_profile,
        "token_usage": summarize_result_token_usage(results),
        "deterministic_metric_means": {
            key: sum(values) / len(values)
            for key, values in sorted(deterministic_values.items())
            if values
        },
    }


def run_chain_eval(
    *,
    case: str,
    dataset_version: str | None,
    sample_size: int | None,
    seed: int,
    parallelism: int,
    strict_mongo: bool,
    quiet: bool,
    task_types: list[str] | None = None,
    agent_profiles: list[str] | None = None,
    run_id: str | None = None,
    detector_tool_methods: list[str] | tuple[str, ...] | None = None,
    detector_tool_root: Path | None = None,
) -> dict[str, Any]:
    import uuid
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime, timezone

    run_id = run_id or f"chain-agent-eval:{case}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}:{uuid.uuid4().hex[:8]}"
    store = ChainTaskStore(dataset_version=dataset_version)
    tasks = store.tasks(case)
    if task_types:
        wanted_types = set(task_types)
        tasks = [task for task in tasks if task.get("task_type") in wanted_types]
    tasks, profile_resolution = resolve_tasks_for_agent_profiles(
        tasks, agent_profiles
    )
    tasks = sample_chain_tasks(tasks, sample_size, seed)
    if not tasks:
        raise ValueError(f"no chain QA tasks found for {case}")
    results: list[dict[str, Any]] = []
    resolved_detector_methods = normalize_detector_tool_methods(
        detector_tool_methods
    )
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_map = {
            executor.submit(
                evaluate_chain_task,
                index=idx,
                total=len(tasks),
                task=task,
                task_store=store,
                run_id=run_id,
                strict_mongo=strict_mongo,
                quiet=quiet,
                detector_tool_methods=resolved_detector_methods,
                detector_tool_root=detector_tool_root,
            ): idx
            for idx, task in enumerate(tasks, start=1)
        }
        for future in as_completed(future_map):
            results.append(future.result())
    results.sort(key=lambda item: item["index"])
    metrics_summary = summarize_chain_eval(results)
    summary = {
        "run_id": run_id,
        "case": case,
        "case_id": tasks[0].get("case_id"),
        "dataset_version": dataset_version or tasks[0].get("dataset_version"),
        "sample_size": len(tasks),
        "seed": seed,
        "parallelism": parallelism,
        "agent_llm": LLM_NAME,
        "judge_llm": JUDGE_LLM_NAME,
        "token_usage": metrics_summary["token_usage"],
        "token_usage_complete": all(
            bool(result.get("token_usage_complete")) for result in results
        ),
        "token_usage_result_count": len(results),
        "task_type_filter": task_types or [],
        "agent_profile_filter": agent_profiles or [],
        "agent_profile_resolution": profile_resolution,
        "detector_tool_methods": list(resolved_detector_methods),
        "detector_tool_root": str(
            detector_tool_root or (PROJECT_ROOT / "detector_tool")
        ),
        "summary": metrics_summary,
        "result_collection": "chain_agent_eval_results",
        "session_collection": "chain_agent_sessions",
        "created_at": datetime.now(timezone.utc),
    }
    mongo_error = save_chain_eval_summary(summary, strict=strict_mongo)
    if mongo_error:
        summary["mongo_error"] = mongo_error
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chain-QA agent backed by Mongo chain_qa_tasks and session/result storage.")
    sub = parser.add_subparsers(dest="command")

    answer_p = sub.add_parser("answer", help="Answer one chain QA task or nearest question.")
    answer_p.add_argument("case")
    answer_p.add_argument("question", nargs="*")
    answer_p.add_argument("--task-id")
    answer_p.add_argument("--dataset-version")
    answer_p.add_argument("--no-session", action="store_true")
    answer_p.add_argument(
        "--detector-tool-method",
        action="append",
        dest="detector_tool_methods",
        help="enable one frozen detector method; repeatable; defaults to LR, RF, XGB",
    )
    answer_p.add_argument("--no-detector-tools", action="store_true")
    answer_p.add_argument(
        "--detector-tool-root",
        type=Path,
        default=PROJECT_ROOT / "detector_tool",
    )

    eval_p = sub.add_parser("eval", help="Evaluate chain QA tasks and write sessions/results to Mongo.")
    eval_p.add_argument("case")
    eval_p.add_argument("-k", "--sample-size", type=int, default=None)
    eval_p.add_argument("--seed", type=int, default=0)
    eval_p.add_argument("-p", "--parallelism", type=int, default=1)
    eval_p.add_argument("--dataset-version", default=None)
    eval_p.add_argument(
        "--task-type",
        action="append",
        dest="task_types",
        help="Evaluate only this task type; repeat to include multiple types.",
    )
    eval_p.add_argument(
        "--agent-profile",
        action="append",
        choices=sorted(CHAIN_AGENT_PROFILES),
        dest="agent_profiles",
        help="Evaluate only this agent profile; repeat to include multiple profiles.",
    )
    eval_p.add_argument("--strict-mongo", action="store_true")
    eval_p.add_argument("--quiet", action="store_true")
    eval_p.add_argument(
        "--detector-tool-method",
        action="append",
        dest="detector_tool_methods",
        help="enable one frozen detector method; repeatable; defaults to LR, RF, XGB",
    )
    eval_p.add_argument("--no-detector-tools", action="store_true")
    eval_p.add_argument(
        "--detector-tool-root",
        type=Path,
        default=PROJECT_ROOT / "detector_tool",
    )
    eval_p.add_argument(
        "--run-id",
        help="optional externally assigned run id for resumable experiment orchestration",
    )

    list_p = sub.add_parser("list", help="List task ids for a case.")
    list_p.add_argument("case")
    list_p.add_argument("--dataset-version")
    list_p.add_argument("--limit", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "list":
        store = ChainTaskStore(dataset_version=args.dataset_version)
        for task in store.tasks(args.case, limit=args.limit):
            print(f"{task.get('id')}\t{task.get('task_type')}\t{task.get('question')}")
        return 0
    if args.command == "answer":
        if args.no_detector_tools and args.detector_tool_methods:
            raise SystemExit(
                "--no-detector-tools cannot be combined with --detector-tool-method"
            )
        detector_methods = (
            ()
            if args.no_detector_tools
            else normalize_detector_tool_methods(args.detector_tool_methods)
        )
        store = ChainTaskStore(dataset_version=args.dataset_version)
        tasks = store.tasks(args.case)
        if not tasks:
            raise SystemExit(f"no chain QA tasks found for {args.case}")
        if args.task_id:
            task = next((item for item in tasks if item.get("id") == args.task_id), None)
            if not task:
                raise SystemExit(f"task not found: {args.task_id}")
            result = answer_chain_task(
                task,
                task_store=store,
                save_session=not args.no_session,
                detector_tool_methods=detector_methods,
                detector_tool_root=args.detector_tool_root,
            )
        else:
            question = " ".join(args.question).strip()
            if not question:
                raise SystemExit("provide --task-id or a question")
            result = answer_question(
                question,
                case=args.case,
                dataset_version=args.dataset_version,
                save_session=not args.no_session,
                detector_tool_methods=detector_methods,
                detector_tool_root=args.detector_tool_root,
            )
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str))
        return 0
    if args.command == "eval":
        if args.parallelism < 1:
            raise SystemExit("-p/--parallelism must be >= 1")
        if args.no_detector_tools and args.detector_tool_methods:
            raise SystemExit(
                "--no-detector-tools cannot be combined with --detector-tool-method"
            )
        detector_methods = (
            ()
            if args.no_detector_tools
            else normalize_detector_tool_methods(args.detector_tool_methods)
        )
        summary = run_chain_eval(
            case=args.case,
            dataset_version=args.dataset_version,
            sample_size=args.sample_size,
            seed=args.seed,
            parallelism=args.parallelism,
            strict_mongo=args.strict_mongo,
            quiet=args.quiet,
            task_types=args.task_types,
            agent_profiles=args.agent_profiles,
            run_id=args.run_id,
            detector_tool_methods=detector_methods,
            detector_tool_root=args.detector_tool_root,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    raise SystemExit("choose a command: list, answer, or eval")


if __name__ == "__main__":
    raise SystemExit(main())
