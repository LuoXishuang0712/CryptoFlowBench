"""Case summarization LLM Agent.

For each case under `cases/<NN>-YYYY-<slug>/` (whose `case.json` lists the
source documents/URLs collected by `collect_report.py`), this agent drives the
long-lived browser (see docs/browser.md) to visit every source URL, read the
content, use vision (screenshots / image URLs) to understand charts & diagrams
that text cannot convey, cross-reference the facts across sources, and finally
write a structured case summary into `summarized/<NN>-YYYY-<slug>/summary.json`.

Local PDF files are intentionally ignored -- everything is read through the
agent browser from the recorded `source` URLs.

Usage:
    python extract_report_info.py                 # summarize every case
    python extract_report_info.py 01-2022-Ronin   # summarize one case dir
    python extract_report_info.py all             # summarize every case
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import litellm
import requests

# Reuse the shared infra (env loading, browser client, llm client, helpers)
# that collect_report.py already provides, to stay consistent with the repo.
from collect_report import (
    LLM_CONTEXT_LENGTH,
    LLM_OUTPUT_LENGTH,
    LLM_TIMEOUT,
    BrowserClient,
    LLMClient,
    load_env,
    truncate,
)
from storage import get_json_store

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
CASES_DIR = PROJECT_ROOT / "cases"
BROWSER_BASE = os.environ.get("BROWSER_BASE", "http://127.0.0.1:5010")

load_env(ENV_PATH)

SUMMARIZED_DIR = PROJECT_ROOT / "summarized"

MAX_ITERATIONS = 70  # enough to visit ~10-13 sources + extract + finish
LLM_TIMEOUT = 600


# --------------------------------------------------------------------------- #
# Vision helpers
# --------------------------------------------------------------------------- #


_IMAGE_MARK = "_image_base64"  # internal marker popped off tool results

# Cap how much raw image bytes we are willing to pull in via view_image_url.
MAX_IMAGE_BYTES = 4 * 1024 * 1024


def _http_image_to_data_url(url: str, timeout: int = 30) -> tuple[str, str] | None:
    """Download an image URL and return (data_url, mime). None if not feasible."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (onchain-case-agent)"}
        resp = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        if resp.status_code != 200:
            return None
        mime = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not mime.startswith("image/"):
            # Some servers lie; try to infer from extension.
            ext = urlparse(url).path.lower().rsplit(".", 1)[-1]
            mime = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
                "svg": "image/svg+xml",
            }.get(ext, "")
            if not mime:
                return None
        if mime == "image/svg+xml":
            # Vision models generally can't rasterize SVG; skip.
            return None
        data = resp.content
        if len(data) > MAX_IMAGE_BYTES:
            return None
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}", mime
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = """You are an on-chain security case summarization agent.

# Goal
You are given a case directory whose `case.json` lists already-collected report
sources (each with a `source` URL, `source_entity`, `source_type`,
`source_grade`). Your job is to visit those URLs through the provided browser,
read the content, EXTRACT structured forensic facts, cross-reference them across
sources, and finally produce a single consolidated case summary by calling
`finish`.

# What you must NOT do
- Do NOT read or rely on local PDF files. Everything must come from the live
  `source` URLs via the browser. The `filename` field is irrelevant.
- Do NOT invent facts. Every concrete address / tx hash / date / amount must
  come from a page you actually visited. If a value is uncertain, omit it or
  qualify it in narrative fields.
- Do NOT collect new sources -- the source list is fixed. Only visit URLs from
  the provided document list.

# Workflow
1. Start with the highest-credibility sources (grade A: official / law
   enforcement / project postmortem; then B: security/chain-analysis firms).
   Use `list_documents` to recall the list with indices, then `visit_page` to
   open a URL. Use `get_page_text` when the snapshot is too short.
2. For each visited source, call `record_findings` with:
   - source_url / source_entity / source_grade (copy from the document list)
   - findings: a short narrative of what THIS source contributed
   - key_fields: any structured facts you could confirm (addresses, txs, dates,
     amounts, root cause, etc.). Fill only what this source actually states.
3. Use vision when a page contains charts / fund-flow diagrams / tables that the
   text does not capture:
   - `take_screenshot` (full_page=false) to grab the viewport as an image, OR
   - `view_image_url` with the URL of a specific <img> on the page.
   The image is fed back to you automatically. Use it to extract addresses,
   transaction hashes, dollar amounts and flow paths that only appear visually.
   Use vision sparingly -- only when text is insufficient.
4. Cross-reference: prefer A-grade facts as ground truth. Note disagreements
   between sources in the narrative fields.
5. When you have covered the key sources (aim for all A and B sources, plus a
   couple of C/D for context), call `finish` with the complete `summary` object.
   Compose the narrative fields (overview, root_cause, fund_flow, recovery,
   sanctions, aftermath) by synthesizing across everything you read.

# Field guidance for the final `summary`
- overview: 2-4 sentence executive summary of what happened.
- incident_date: YYYY-MM-DD if known, else best ISO date or null.
- chains: affected chains (e.g. ["Ethereum","Ronin"]).
- protocol_type: e.g. "Cross-chain Bridge", "DEX", "Lending".
- loss_amount_usd: numeric USD loss if confidently known, else null.
- loss_assets: list of stolen assets with amounts (e.g. ["173,600 ETH"]).
- root_cause: the technical / operational root cause.
- vulnerability_type: e.g. "Compromised validator keys", "Signature replay".
- attack_flow: ordered list of attack steps (strings).
- attacker: {attribution, addresses[], initial_funding}.
- victim_contracts: list of victim contract addresses.
- exploit_txs: list of exploit transaction hashes.
- fund_flow: narrative of how stolen funds were moved / laundered.
- recovery: narrative of any recovered / returned funds.
- sanctions: narrative of any sanctions / enforcement actions.
- aftermath: narrative of protocol response, forks, lawsuits, etc.
- key_takeaways: 2-5 bullet lessons.

# Cloudflare / challenges
If a snapshot reports `cloudflare_challenge: true` after `visit_page`:
- Optionally try `click_relative` near the checkbox (x_ratio~0.5, y_ratio~0.6),
  then re-check with `get_page_text`.
- If it stays blocked, call `call_user` (urgent=true) with the URL, then skip
  that source and continue with the rest. Do not get stuck.

Be efficient: prefer reading the snapshot text over excessive browsing. Stop as
soon as you have solid coverage of the key facts and call `finish`."""


# Known scalar/list keys we accumulate from record_findings.
KEY_FIELD_KEYS = [
    "incident_date",
    "chains",
    "protocol_type",
    "loss_amount_usd",
    "loss_assets",
    "root_cause",
    "vulnerability_type",
    "attacker_attribution",
    "attacker_addresses",
    "victim_contracts",
    "exploit_txs",
    "fund_flow",
    "recovery",
    "sanctions",
    "aftermath",
]


class ReportSummarizerAgent:
    def __init__(
        self,
        case_dir: Path,
        browser: BrowserClient,
        llm: LLMClient,
        max_iterations: int = MAX_ITERATIONS,
        verbose: bool = True,
    ) -> None:
        self.case_dir = case_dir
        self.case_dir_name = case_dir.name
        manifest_path = case_dir / "case.json"
        manifest = get_json_store().load_path(manifest_path)
        self.case_name: str = manifest.get("case_name", case_dir.name)
        self.documents: list[dict] = manifest.get("documents", [])
        self.browser = browser
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.findings: list[dict] = []  # per-source findings records
        self.key_fields: dict[str, Any] = {}  # accumulated structured facts
        self.finished = False
        self.output_path: Path | None = None
        self.final_summary: dict | None = None

    # -- logging ----------------------------------------------------------- #
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -- tool implementations ---------------------------------------------- #
    def tool_list_documents(self) -> Any:
        return {
            "status": "ok",
            "case_name": self.case_name,
            "count": len(self.documents),
            "documents": [
                {
                    "index": i,
                    "source": d.get("source"),
                    "source_entity": d.get("source_entity"),
                    "source_type": d.get("source_type"),
                    "source_grade": d.get("source_grade"),
                }
                for i, d in enumerate(self.documents)
            ],
        }

    def tool_visit_page(self, url: str) -> Any:
        got = self.browser.goto(url, wait_until="domcontentloaded", timeout=60000)
        if got.get("status") == "error":
            return got
        snap = got.get("data", {})
        return {
            "status": "ok",
            "url": snap.get("url", url),
            "title": snap.get("title", ""),
            "cloudflare_challenge": bool(snap.get("cloudflare_challenge")),
            "text": truncate(snap.get("text", ""), 1800),
        }

    def tool_get_page_text(self, selector: str = "", limit: int = 4000) -> Any:
        res = self.browser.text(selector or None)
        if res.get("status") == "error":
            return res
        data = res.get("data", {})
        return {
            "status": "ok",
            "url": data.get("url", ""),
            "text": truncate(data.get("text", ""), max(500, min(limit, 6000))),
        }

    def tool_click_element(
        self, selector: str, wait_after: str = "domcontentloaded"
    ) -> Any:
        res = self.browser.click(selector, wait_after=wait_after)
        return self._snapshot(res)

    def tool_click_relative(
        self, x_ratio: float, y_ratio: float, wait_after: str | None = None
    ) -> Any:
        body = {"x_ratio": x_ratio, "y_ratio": y_ratio}
        if wait_after:
            body["wait_after"] = wait_after
        res = self.browser.click_relative(**body)
        return self._snapshot(res)

    def tool_fill_input(
        self, selector: str, value: str, press_enter: bool = False
    ) -> Any:
        res = self.browser.fill(selector, value, press_enter=press_enter)
        return self._snapshot(res)

    def tool_evaluate_js(self, script: str, arg: Any = None) -> Any:
        res = self.browser.evaluate(script, arg)
        if res.get("status") == "error":
            return res
        data = res.get("data", {})
        result = data.get("result")
        if isinstance(result, str):
            return {
                "status": "ok",
                "url": data.get("url", ""),
                "result": truncate(result, 4000),
            }
        return {
            "status": "ok",
            "url": data.get("url", ""),
            "result": truncate(json.dumps(result, ensure_ascii=False), 4000),
        }

    def tool_take_screenshot(self, full_page: bool = False) -> Any:
        res = self.browser.screenshot(full_page=full_page, return_base64=True)
        if res.get("status") == "error":
            return res
        data = res.get("data", {})
        img = data.get("image_base64") or ""
        out: dict[str, Any] = {
            "status": "ok",
            "url": data.get("url", ""),
            "viewport": data.get("viewport"),
            "full_page": full_page,
            "has_image": bool(img),
            "note": (
                "Viewport screenshot captured; the image is attached to your "
                "next input. Analyze charts/diagrams/tables and extract any "
                "case-relevant facts (addresses, txs, amounts, flow)."
                if img
                else "No image returned by the browser."
            ),
        }
        if img:
            out[_IMAGE_MARK] = img
        return out

    def tool_view_image_url(self, url: str) -> Any:
        if not url or not url.startswith(("http://", "https://", "data:image/")):
            return {
                "status": "error",
                "message": "url must be an http(s)/data image URL",
            }
        if url.startswith("data:image/"):
            out = {
                "status": "ok",
                "url": url,
                "has_image": True,
                "note": "Image attached to your next input.",
            }
            out[_IMAGE_MARK] = url.split(",", 1)[1] if "," in url else ""
            return out
        got = _http_image_to_data_url(url)
        if not got:
            return {
                "status": "ok",
                "url": url,
                "has_image": False,
                "note": "Could not fetch a usable image (non-image, too large, or SVG).",
            }
        data_url, mime = got
        b64 = data_url.split(",", 1)[1]
        out = {
            "status": "ok",
            "url": url,
            "mime": mime,
            "has_image": True,
            "note": "Image attached to your next input. Analyze and extract facts.",
        }
        out[_IMAGE_MARK] = b64
        return out

    def tool_call_user(self, content: str, urgent: bool = False) -> Any:
        try:
            from call_user import send_msg

            title = "[Urgent] From LLM Agent" if urgent else "From LLM Agent"
            send_msg(title=title, desp=content)
            return {"status": "ok", "message": "User notified."}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": f"Failed to notify user: {e}"}

    def tool_record_findings(
        self,
        source_url: str,
        source_entity: str,
        source_grade: str,
        findings: str,
        key_fields: dict | None = None,
    ) -> Any:
        rec = {
            "source": source_url,
            "source_entity": source_entity,
            "source_grade": source_grade,
            "findings": findings,
        }
        self.findings.append(rec)
        merged: list[str] = []
        if key_fields and isinstance(key_fields, dict):
            for k, v in key_fields.items():
                if v in (None, "", [], {}):
                    continue
                if k not in self.key_fields or self.key_fields[k] in (None, "", [], {}):
                    # list fields: union with existing
                    if isinstance(v, list) and isinstance(self.key_fields.get(k), list):
                        for item in v:
                            if item not in self.key_fields[k]:
                                self.key_fields[k].append(item)
                    else:
                        self.key_fields[k] = v
                    if k not in merged:
                        merged.append(k)
                elif isinstance(v, list) and isinstance(self.key_fields.get(k), list):
                    for item in v:
                        if item not in self.key_fields[k]:
                            self.key_fields[k].append(item)
        self.log(
            f"  * recorded [{source_grade}] {source_entity}: "
            f"{truncate(findings, 120)}{' +' + ','.join(merged) if merged else ''}"
        )
        return {
            "status": "ok",
            "recorded_count": len(self.findings),
            "accumulated_key_fields": sorted(self.key_fields.keys()),
        }

    def tool_finish(self, summary: dict) -> Any:
        if not isinstance(summary, dict):
            return {"status": "error", "message": "summary must be an object"}
        self.final_summary = summary
        out_dir = SUMMARIZED_DIR / self.case_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # sources_used: prefer recorded findings; fall back to case.json docs
        # that were never recorded (mark as "not visited").
        recorded_urls = {f["source"] for f in self.findings}
        sources_used = list(self.findings)
        for d in self.documents:
            if d.get("source") not in recorded_urls:
                sources_used.append(
                    {
                        "source": d.get("source"),
                        "source_entity": d.get("source_entity"),
                        "source_grade": d.get("source_grade"),
                        "findings": "(not visited / not recorded)",
                    }
                )

        output = {
            "case_name": self.case_name,
            "case_dir": self.case_dir_name,
            "summary": summary,
            "sources_used": sources_used,
        }
        out_path = out_dir / "summary.json"
        get_json_store().save_path(
            out_path,
            output,
            extra_metadata={
                "document_type": "case_report_summary",
                "case_dir": self.case_dir_name,
                "generated_by": "extract_report_info.py",
            },
        )
        self.output_path = out_path
        self.finished = True
        self.log(f"\n[done] wrote {out_path}")
        return {"status": "ok", "path": str(out_path)}

    # -- helpers ----------------------------------------------------------- #
    def _snapshot(self, res: dict) -> Any:
        if res.get("status") == "error":
            return res
        snap = res.get("data", {})
        return {
            "status": "ok",
            "url": snap.get("url", ""),
            "title": snap.get("title", ""),
            "cloudflare_challenge": bool(snap.get("cloudflare_challenge")),
            "text": truncate(snap.get("text", ""), 1200),
        }

    # -- tool schemas ------------------------------------------------------ #
    @property
    def tools(self) -> list[dict]:
        key_fields_schema = {
            "type": "object",
            "description": "Structured facts confirmed by THIS source. Fill only what it states.",
            "properties": {
                "incident_date": {"type": "string"},
                "chains": {"type": "array", "items": {"type": "string"}},
                "protocol_type": {"type": "string"},
                "loss_amount_usd": {"type": "number"},
                "loss_assets": {"type": "array", "items": {"type": "string"}},
                "root_cause": {"type": "string"},
                "vulnerability_type": {"type": "string"},
                "attacker_attribution": {"type": "string"},
                "attacker_addresses": {"type": "array", "items": {"type": "string"}},
                "victim_contracts": {"type": "array", "items": {"type": "string"}},
                "exploit_txs": {"type": "array", "items": {"type": "string"}},
                "fund_flow": {"type": "string"},
                "recovery": {"type": "string"},
                "sanctions": {"type": "string"},
                "aftermath": {"type": "string"},
            },
            "additionalProperties": True,
        }
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_documents",
                    "description": "List the case's source documents (index, source URL, entity, grade).",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "visit_page",
                    "description": "Navigate the browser to a URL and return a snapshot (url,title,text,cloudflare_challenge).",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_page_text",
                    "description": "Return (more) text of the current page, optionally of a CSS selector.",
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
                    "name": "take_screenshot",
                    "description": "Capture the viewport (or full page) as an image and feed it back to you for visual analysis. Use when a page has charts/diagrams/tables not captured by text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "full_page": {"type": "boolean", "default": False},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "view_image_url",
                    "description": "Fetch a specific image URL (e.g. an <img> on the page) and feed it back for visual analysis. Useful for fund-flow diagrams.",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "click_element",
                    "description": "Click an element by Playwright selector.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "wait_after": {
                                "type": "string",
                                "default": "domcontentloaded",
                            },
                        },
                        "required": ["selector"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "click_relative",
                    "description": "Click at relative viewport coords (0..1). Useful for Cloudflare checkboxes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x_ratio": {"type": "number"},
                            "y_ratio": {"type": "number"},
                            "wait_after": {"type": "string"},
                        },
                        "required": ["x_ratio", "y_ratio"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fill_input",
                    "description": "Fill an input/textarea (replaces content).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "value": {"type": "string"},
                            "press_enter": {"type": "boolean", "default": False},
                        },
                        "required": ["selector", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_js",
                    "description": 'Run a JS function expression in the page, e.g. "(arg)=>document.title". Returns its result.',
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script": {"type": "string"},
                            "arg": {"description": "Optional serializable argument"},
                        },
                        "required": ["script"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "call_user",
                    "description": "Notify the user (e.g. when blocked by Cloudflare). Use urgent=true for blockers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "urgent": {"type": "boolean", "default": False},
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "record_findings",
                    "description": "Record what one source contributed and any structured facts you could confirm from it. Call this after reading each source.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_url": {"type": "string"},
                            "source_entity": {"type": "string"},
                            "source_grade": {
                                "type": "string",
                                "enum": ["A", "B", "C", "D", "E"],
                            },
                            "findings": {
                                "type": "string",
                                "description": "Short narrative of what this source contributed.",
                            },
                            "key_fields": key_fields_schema,
                        },
                        "required": [
                            "source_url",
                            "source_entity",
                            "source_grade",
                            "findings",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Finalize: write summarized/<case_dir>/summary.json from the consolidated summary and end.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "object",
                                "description": "The consolidated case summary.",
                                "properties": {
                                    "overview": {"type": "string"},
                                    "incident_date": {"type": "string"},
                                    "chains": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "protocol_type": {"type": "string"},
                                    "loss_amount_usd": {"type": "number"},
                                    "loss_assets": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "root_cause": {"type": "string"},
                                    "vulnerability_type": {"type": "string"},
                                    "attack_flow": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "attacker": {
                                        "type": "object",
                                        "properties": {
                                            "attribution": {"type": "string"},
                                            "addresses": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "initial_funding": {"type": "string"},
                                        },
                                    },
                                    "victim_contracts": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "exploit_txs": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "fund_flow": {"type": "string"},
                                    "recovery": {"type": "string"},
                                    "sanctions": {"type": "string"},
                                    "aftermath": {"type": "string"},
                                    "key_takeaways": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["overview", "root_cause", "attack_flow"],
                            }
                        },
                        "required": ["summary"],
                    },
                },
            },
        ]

    # -- dispatch ---------------------------------------------------------- #
    def dispatch(self, name: str, args: dict) -> Any:
        handlers = {
            "list_documents": lambda a: self.tool_list_documents(),
            "visit_page": lambda a: self.tool_visit_page(a["url"]),
            "get_page_text": lambda a: self.tool_get_page_text(
                a.get("selector", ""), a.get("limit", 4000)
            ),
            "take_screenshot": lambda a: self.tool_take_screenshot(
                a.get("full_page", False)
            ),
            "view_image_url": lambda a: self.tool_view_image_url(a["url"]),
            "click_element": lambda a: self.tool_click_element(
                a["selector"], a.get("wait_after", "domcontentloaded")
            ),
            "click_relative": lambda a: self.tool_click_relative(
                a["x_ratio"], a["y_ratio"], a.get("wait_after")
            ),
            "fill_input": lambda a: self.tool_fill_input(
                a["selector"], a["value"], a.get("press_enter", False)
            ),
            "evaluate_js": lambda a: self.tool_evaluate_js(a["script"], a.get("arg")),
            "call_user": lambda a: self.tool_call_user(
                a["content"], a.get("urgent", False)
            ),
            "record_findings": lambda a: self.tool_record_findings(
                a["source_url"],
                a["source_entity"],
                a["source_grade"],
                a["findings"],
                a.get("key_fields"),
            ),
            "finish": lambda a: self.tool_finish(a["summary"]),
        }
        handler = handlers.get(name)
        if not handler:
            return {"status": "error", "message": f"Unknown tool: {name}"}
        try:
            return handler(args)
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": f"{type(e).__name__}: {e}"}

    # -- message assembly -------------------------------------------------- #
    def _initial_messages(self) -> list[dict]:
        doc_lines = []
        for i, d in enumerate(self.documents):
            doc_lines.append(
                f"[{i}] grade={d.get('source_grade')} type={d.get('source_type')} "
                f"entity={d.get('source_entity')} -> {d.get('source')}"
            )
        docs_block = "\n".join(doc_lines) if doc_lines else "(no documents)"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Case name: {self.case_name}\n"
                    f"Case directory: {self.case_dir_name}\n\n"
                    f"Source documents (index | grade | type | entity -> URL):\n{docs_block}\n\n"
                    "Visit the sources (prioritize A then B), extract the facts, "
                    "record findings for each, then call `finish` with the "
                    "consolidated summary."
                ),
            },
        ]

    def _append_tool_result(
        self, messages: list[dict], tc_id: str, name: str, result: Any
    ) -> str | None:
        """Append a tool message. Returns base64 image if the tool produced one."""
        img_b64: str | None = None
        if isinstance(result, dict) and _IMAGE_MARK in result:
            img_b64 = result.pop(_IMAGE_MARK)
        # Keep tool result JSON-serializable & compact.
        try:
            content = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            content = str(result)
        messages.append(
            {"role": "tool", "tool_call_id": tc_id, "name": name, "content": content}
        )
        return img_b64

    @staticmethod
    def _append_images(messages: list[dict], images: list[tuple[str, str]]) -> None:
        if not images:
            return
        parts: list[dict] = []
        for url, b64 in images:
            parts.append(
                {"type": "text", "text": f"[vision] screenshot/image of {url}"}
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        parts.append(
            {
                "type": "text",
                "text": "Extract any case-relevant facts visible here (addresses, tx "
                "hashes, amounts, flow paths, dates). If nothing relevant, continue.",
            }
        )
        messages.append({"role": "user", "content": parts})

    # -- main loop --------------------------------------------------------- #
    def run(self) -> Path | None:
        messages = self._initial_messages()
        no_tool_turns = 0

        for i in range(1, self.max_iterations + 1):
            self.log(f"\n--- [{self.case_dir_name}] iteration {i} ---")
            try:
                response = self.llm.complete(
                    messages, tools=self.tools, tool_choice="auto"
                )
            except Exception as e:  # noqa: BLE001
                self.log(f"[llm error] {e}; retrying in 5s")
                time.sleep(5)
                continue

            choice = response.choices[0]
            msg = choice.message
            assistant_msg = (
                msg.model_dump(exclude_none=True)
                if hasattr(msg, "model_dump")
                else dict(msg)
            )
            messages.append(assistant_msg)

            if msg.content:
                self.log(f"[assistant] {truncate(str(msg.content), 600)}")

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                no_tool_turns += 1
                if no_tool_turns >= 2:
                    if self.findings:
                        self.log("[agent idle] forcing finish from recorded findings.")
                        if self._force_finish(messages):
                            return self.output_path
                        self.log("[force-finish failed] writing best-effort summary.")
                        self._best_effort_finish()
                        return self.output_path
                    else:
                        self.log("[agent idle] no findings and no tool use; stopping.")
                        break
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Use a tool to continue visiting sources / "
                            "recording findings, or call `finish` when done.",
                        }
                    )
                continue
            no_tool_turns = 0

            images: list[tuple[str, str]] = []
            for tc in tool_calls:
                fn = tc.function
                name = fn.name
                try:
                    args = json.loads(fn.arguments) if fn.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                self.log(
                    f"[tool] {name}({truncate(json.dumps(args, ensure_ascii=False), 200)})"
                )
                result = self.dispatch(name, args)
                img = self._append_tool_result(messages, tc.id, name, result)
                if img:
                    url = result.get("url", "") if isinstance(result, dict) else ""
                    images.append((url or name, img))
                if name == "finish" and self.finished:
                    return self.output_path

            # Feed any images back as a single follow-up user message (kept after
            # all tool messages so the tool-call/response ordering stays valid).
            self._append_images(messages, images)

        # Exhausted iterations.
        if not self.finished:
            self.log("[max iterations] attempting force-finish.")
            if not self._force_finish(messages):
                self.log("[force-finish failed] writing best-effort summary.")
                self._best_effort_finish()
        return self.output_path

    def _force_finish(self, messages: list[dict]) -> bool:
        """One more LLM call forced to emit `finish`. Returns True on success."""
        if self.finished:
            return True
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have run out of steps. Call `finish` NOW with the complete "
                    "`summary` object based on everything you have gathered so far. "
                    "Do not call any other tool."
                ),
            }
        )
        try:
            response = self.llm.complete(
                messages,
                tools=self.tools,
                tool_choice={"type": "function", "function": {"name": "finish"}},
            )
        except Exception as e:  # noqa: BLE001
            self.log(f"[force-finish llm error] {e}")
            return False
        choice = response.choices[0]
        msg = choice.message
        messages.append(
            msg.model_dump(exclude_none=True)
            if hasattr(msg, "model_dump")
            else dict(msg)
        )
        for tc in msg.tool_calls or []:
            if tc.function.name == "finish":
                try:
                    args = (
                        json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {}
                    )
                except json.JSONDecodeError:
                    args = {}
                self.dispatch("finish", args)
                return self.finished
        return False

    def _best_effort_finish(self) -> None:
        """If the LLM never produced a summary, write one from recorded data."""
        if self.finished:
            return
        kf = self.key_fields
        summary = {
            "overview": (
                f"Best-effort summary for {self.case_name}. The agent did not emit a "
                "final consolidated summary; this is assembled from per-source "
                "recorded findings."
            ),
            "incident_date": kf.get("incident_date"),
            "chains": kf.get("chains", []),
            "protocol_type": kf.get("protocol_type"),
            "loss_amount_usd": kf.get("loss_amount_usd"),
            "loss_assets": kf.get("loss_assets", []),
            "root_cause": kf.get("root_cause", "(unknown)"),
            "vulnerability_type": kf.get("vulnerability_type"),
            "attack_flow": [],
            "attacker": {
                "attribution": kf.get("attacker_attribution"),
                "addresses": kf.get("attacker_addresses", []),
                "initial_funding": None,
            },
            "victim_contracts": kf.get("victim_contracts", []),
            "exploit_txs": kf.get("exploit_txs", []),
            "fund_flow": kf.get("fund_flow", ""),
            "recovery": kf.get("recovery", ""),
            "sanctions": kf.get("sanctions", ""),
            "aftermath": kf.get("aftermath", ""),
            "key_takeaways": [],
            "note": "Auto-generated from record_findings; LLM did not call finish.",
        }
        self.tool_finish(summary)


# --------------------------------------------------------------------------- #
# Case discovery / CLI
# --------------------------------------------------------------------------- #


def discover_cases() -> list[Path]:
    if not CASES_DIR.exists():
        return []
    cases = []
    for entry in sorted(CASES_DIR.iterdir()):
        if entry.is_dir() and (entry / "case.json").exists():
            cases.append(entry)
    return cases


def already_summarized(case_dir: Path) -> bool:
    return (SUMMARIZED_DIR / case_dir.name / "summary.json").exists()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize on-chain case reports into structured summaries via an LLM browser agent."
    )
    parser.add_argument(
        "case",
        nargs="?",
        default="all",
        help="Case directory name under cases/ (e.g. '01-2022-Ronin') or 'all' (default).",
    )
    parser.add_argument(
        "--browser-base", default=BROWSER_BASE, help="Browser service base URL"
    )
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument(
        "--force", action="store_true", help="Re-summarize even if summary.json exists."
    )
    parser.add_argument("--quiet", action="store_true", help="Less logging")
    args = parser.parse_args()

    litellm.drop_params = True
    litellm.set_verbose = False

    # Select cases to process.
    if args.case.lower() == "all":
        cases = discover_cases()
        if not cases:
            print("No cases found under cases/.", file=sys.stderr)
            return 1
    else:
        case_path = CASES_DIR / args.case
        if not (case_path.is_dir() and (case_path / "case.json").exists()):
            print(
                f"Case not found: {args.case} (looked in {case_path})", file=sys.stderr
            )
            return 1
        cases = [case_path]

    browser = BrowserClient(args.browser_base)
    llm = LLMClient()

    # Probe the browser service.
    try:
        h = browser.health()
        print(f"[health] {h}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] browser health check failed: {e}", file=sys.stderr)
        print(
            "[warn] is the browser service running? continuing anyway...",
            file=sys.stderr,
        )

    failures = 0
    for case_dir in cases:
        if not args.force and already_summarized(case_dir):
            print(f"[skip] {case_dir.name} already summarized (use --force to redo).")
            continue
        print(
            f"\n==================== summarizing {case_dir.name} ===================="
        )
        agent = ReportSummarizerAgent(
            case_dir=case_dir,
            browser=browser,
            llm=llm,
            max_iterations=args.max_iterations,
            verbose=not args.quiet,
        )
        try:
            out = agent.run()
            if out:
                print(f"Result: {out}")
            else:
                print(
                    f"[warn] no summary produced for {case_dir.name}", file=sys.stderr
                )
                failures += 1
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"[error] {case_dir.name} failed: {e}", file=sys.stderr)

    print(f"\nDone. {len(cases) - failures}/{len(cases)} cases summarized.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
