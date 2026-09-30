"""Report collection LLM Agent.

Given a case name, this agent drives a long-lived browser (see docs/browser.md)
to Google for relevant reports, classifies each source according to
docs/data_source_class.md, and writes a case.json manifest into a managed
cases/<NN>-YYYY-<slug>/ directory. PDFs are NOT downloaded here.

Usage:
    python collect_report.py "Ronin Bridge Hack 2022"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import litellm
import requests

from storage import get_json_store

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
CASES_DIR = PROJECT_ROOT / "cases"
BROWSER_BASE = os.environ.get("BROWSER_BASE", "http://127.0.0.1:5010")

MAX_ITERATIONS = 50
LLM_TIMEOUT = 600  # seconds per completion call


def load_env(path: Path) -> None:
    """Load .env into os.environ (real env vars take precedence)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # strip inline comments that are outside quotes
        out: list[str] = []
        in_q = False
        quote_ch = ""
        for ch in line:
            if ch in "\"'":
                if not in_q:
                    in_q = True
                    quote_ch = ch
                elif quote_ch == ch:
                    in_q = False
            if ch == "#" and not in_q:
                break
            out.append(ch)
        line = "".join(out).strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if val and val[0] in "\"'" and val[-1] == val[0]:
            val = val[1:-1]
        os.environ.setdefault(key, val)


load_env(ENV_PATH)

LLM_PROVIDER_URL = os.environ.get("LLM_PROVIDER_URL", "http://127.0.0.1:18080/v1")
LLM_NAME = os.environ.get("LLM_NAME", "Qwen/Qwen3.6-27B")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "NO_API_KEY")
LLM_THINK = os.environ.get("LLM_THINK", "False").strip().lower() == "true"
LLM_CONTEXT_LENGTH = int(os.environ.get("LLM_CONTEXT_LENGTH", "262144"))
LLM_OUTPUT_LENGTH = int(os.environ.get("LLM_OUTPUT_LENGTH", "20000"))


def save_report_json(path: Path, payload: dict[str, Any], *, document_type: str) -> None:
    get_json_store().save_path(
        path,
        payload,
        extra_metadata={
            "document_type": document_type,
            "case_dir": path.parent.name,
            "generated_by": "collect_report.py",
        },
    )


# --------------------------------------------------------------------------- #
# Browser client (thin wrapper over the HTTP API in docs/browser.md)
# --------------------------------------------------------------------------- #


class BrowserClient:
    def __init__(self, base: str = BROWSER_BASE, timeout: int = 300) -> None:
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kw) -> dict:
        url = f"{self.base}{path}"
        resp = self.session.request(method, url, timeout=self.timeout, **kw)
        # Retry once on gateway-level failures (worker likely rotated).
        if resp.status_code in (502, 503, 504):
            time.sleep(3)
            resp = self.session.request(method, url, timeout=self.timeout, **kw)
        try:
            data = resp.json()
        except ValueError:
            return {
                "status": "error",
                "message": f"non-JSON response (HTTP {resp.status_code})",
            }
        if resp.status_code >= 400 and data.get("status") != "error":
            data["status"] = "error"
            data.setdefault("message", f"HTTP {resp.status_code}")
        return data

    def health(self) -> dict:
        return self._request("GET", "/health")

    def goto(self, url: str, **kw) -> dict:
        body = {"url": url, "wait_cf": True}
        body.update(kw)
        return self._request("POST", "/api/goto", json=body)

    def post_form(self, url: str, fields: dict, **kw) -> dict:
        body = {"url": url, "fields": fields, "wait_cf": True}
        body.update(kw)
        return self._request("POST", "/api/post_form", json=body)

    def state(self, text_limit: int = 2000) -> dict:
        return self._request("GET", "/api/state", params={"text_limit": text_limit})

    def text(self, selector: str | None = None) -> dict:
        body: dict = {}
        if selector:
            body["selector"] = selector
        return self._request("POST", "/api/text", json=body)

    def click(self, selector: str, **kw) -> dict:
        body = {"selector": selector}
        body.update(kw)
        return self._request("POST", "/api/click", json=body)

    def click_relative(self, x_ratio: float, y_ratio: float, **kw) -> dict:
        body = {"x_ratio": x_ratio, "y_ratio": y_ratio}
        body.update(kw)
        return self._request("POST", "/api/click_relative", json=body)

    def fill(self, selector: str, value: str, **kw) -> dict:
        body = {"selector": selector, "value": value}
        body.update(kw)
        return self._request("POST", "/api/fill", json=body)

    def press(self, key: str, selector: str | None = None, **kw) -> dict:
        body: dict = {"key": key}
        if selector:
            body["selector"] = selector
        body.update(kw)
        return self._request("POST", "/api/press", json=body)

    def screenshot(
        self,
        full_page: bool = True,
        return_base64: bool = False,
        path: str | None = None,
    ) -> dict:
        body = {"full_page": full_page, "return_base64": return_base64}
        if path:
            body["path"] = path
        return self._request("POST", "/api/screenshot", json=body)

    def evaluate(self, script: str, arg: Any = None) -> dict:
        return self._request(
            "POST", "/api/evaluate", json={"script": script, "arg": arg}
        )

    def rotate(self) -> dict:
        return self._request("POST", "/rotate")


# --------------------------------------------------------------------------- #
# LLM client
# --------------------------------------------------------------------------- #


class LLMClient:
    def __init__(self) -> None:
        # Route any OpenAI-compatible endpoint through litellm's openai provider.
        # LLM_NAME (e.g. "hosted_vllm/Qwen/Qwen3.6-27B") is forwarded verbatim as
        # the model id to the server at LLM_PROVIDER_URL.
        self.model = "openai/" + LLM_NAME
        self.api_base = LLM_PROVIDER_URL
        self.api_key = LLM_API_KEY
        self.think_enabled = LLM_THINK

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "messages": messages,
            "max_tokens": LLM_OUTPUT_LENGTH,
            "temperature": 0.3,
            "drop_params": True,
            "timeout": LLM_TIMEOUT,
            "num_retries": 2,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if not self.think_enabled:
            # Qwen3: disable thinking via chat_template_kwargs.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return litellm.completion(**kwargs)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "document"
    if len(name) > 180:
        name = name[:180].strip()
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def next_case_index() -> int:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    for entry in CASES_DIR.iterdir():
        if entry.is_dir():
            m = re.match(r"^(\d+)-", entry.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


# Per-run focus hints, cycled across runs to encourage source diversity.
RUN_HINTS = [
    "Focus this run on OFFICIAL / law-enforcement / project-side sources: OFAC SDN, "
    "U.S. Treasury press releases, FBI/DOJ/Europol/CISA, project postmortems, governance "
    "announcements. Try queries like '<case> OFAC', '<case> Treasury sanctions', "
    "'<case> FBI', '<case> postmortem', '<case> official statement'.",
    "Focus this run on SECURITY / chain-analysis companies: Chainalysis, Elliptic, TRM Labs, "
    "SlowMist/MistTrack, BlockSec/Phalcon, PeckShield, CertiK, Halborn, Merkle Science, "
    "Cyvers, Beosin, Hacken. Try queries like '<case> Chainalysis', '<case> SlowMist "
    "analysis', '<case> Elliptic', '<case> Halborn', '<case> PeckShield'.",
    "Focus this run on EVENT DATABASES, NEWS, and ACADEMIC sources: Rekt.news, DefiLlama "
    "Hacks, SlowMist Hacked, The Block, CoinDesk, DLNews, Reuters, arXiv. Try queries like "
    "'<case> rekt.news', '<case> DefiLlama hacks', '<case> arxiv', '<case> CoinDesk'.",
    "Focus this run on COMMUNITY alerts and deep layer-2 digging: X/Twitter threads by "
    "ZachXBT, Scam Sniffer, PeckShieldAlert, CyversAlerts, SlowMist_Team, Lookonchain, "
    "SpotOnChain. Try queries like '<case> ZachXBT', '<case> site:x.com'. On every page "
    "you visit, also call get_page_links and follow linked primary sources (original "
    "postmortem, referenced report, OFAC entry, etc.).",
]


def normalize_url(url: str) -> str:
    """Normalize a URL for dedup: lowercase scheme+host, drop fragment, strip
    tracking params, drop trailing slash. Keeps identity-bearing path/query."""
    if not url:
        return ""
    u = url.strip()
    if "://" not in u:
        u = "https://" + u
    try:
        from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

        parts = urlsplit(u)
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()
        if "@" in netloc:
            netloc = netloc.rsplit("@", 1)[1]
        if netloc.endswith(":80"):
            netloc = netloc[:-3]
        elif netloc.endswith(":443"):
            netloc = netloc[:-4]
        netloc = re.sub(r"^www\.", "", netloc)
        path = parts.path.rstrip("/") or "/"
        tracking = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "gclid", "fbclid", "ref", "source", "_ga", "mc_cid", "mc_eid",
        }
        q = [(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in tracking]
        query = urlencode(q)
        return urlunsplit((scheme, netloc, path, query, ""))
    except Exception:
        return u.lower().rstrip("/")


def merge_runs(run_docs: dict[int, list[dict]]) -> list[dict]:
    """Merge per-run document lists into unique candidates keyed by normalized URL.

    Returns candidates with extra `provenance` (sorted run indices) and `run_count`
    fields; metadata taken from the highest-grade variant. Sorted by run_count desc
    then grade (A first).
    """
    grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    merged: dict[str, dict] = {}
    for run_idx, docs in run_docs.items():
        for d in docs:
            key = normalize_url(d.get("source", ""))
            if not key:
                continue
            entry = merged.setdefault(
                key,
                {
                    "filename": d.get("filename", ""),
                    "source": d.get("source", ""),
                    "source_entity": d.get("source_entity", ""),
                    "source_type": d.get("source_type", ""),
                    "source_grade": d.get("source_grade", ""),
                    "runs": set(),
                    "variants": [],
                },
            )
            entry["runs"].add(run_idx)
            if d.get("source") and d.get("source") not in entry["variants"]:
                entry["variants"].append(d.get("source"))
            # prefer the better (lower) grade variant as canonical metadata
            if grade_rank.get(d.get("source_grade", "E"), 9) < grade_rank.get(
                entry["source_grade"], "E"
            ):
                entry.update(
                    filename=d.get("filename", entry["filename"]),
                    source=d.get("source", entry["source"]),
                    source_entity=d.get("source_entity", entry["source_entity"]),
                    source_type=d.get("source_type", entry["source_type"]),
                    source_grade=d.get("source_grade", entry["source_grade"]),
                )
            # prefer a longer filename when grades tie
            elif d.get("source_grade") == entry.get("source_grade") and len(
                d.get("filename", "")
            ) > len(entry["filename"]):
                entry["filename"] = d.get("filename", entry["filename"])
    candidates: list[dict] = []
    for entry in merged.values():
        cand = {
            "filename": entry["filename"],
            "source": entry["source"],
            "source_entity": entry["source_entity"],
            "source_type": entry["source_type"],
            "source_grade": entry["source_grade"],
            "provenance": sorted(entry["runs"]),
            "run_count": len(entry["runs"]),
        }
        if len(entry["variants"]) > 1:
            cand["source_variants"] = entry["variants"]
        candidates.append(cand)
    candidates.sort(
        key=lambda c: (-c["run_count"], grade_rank.get(c.get("source_grade", "E"), 9))
    )
    return candidates


# --------------------------------------------------------------------------- #
# Search engine helpers (Google / DuckDuckGo / Bing)
# --------------------------------------------------------------------------- #

SEARCH_ENGINES = ("google", "ddg", "bing")
SEARCH_ENGINE_PRIORITY = "google > ddg > bing"

# Hosts that are never real results (the search engine's own chrome).
_SEARCH_SKIP_HOSTS = (
    "google.", "gstatic.", "googleapis.", "googleusercontent.",
    "youtube.", "youtu.be", "blogger.", "webcache.",
    "duckduckgo.com", "ddg.gg",
    "bing.com", "msn.com", "microsoft.com", "live.com",
    "microsofttranslator.com", "linkedin.com",
)


def build_search_url(engine: str, query: str, num: int) -> str:
    q = quote(query)
    n = max(num, 10)
    if engine == "google":
        return f"https://www.google.com/search?q={q}&num={n}&hl=en"
    if engine == "ddg":
        # html.duckduckgo.com is the lightweight, scraper-friendly endpoint.
        return f"https://html.duckduckgo.com/html/?q={q}"
    if engine == "bing":
        return f"https://www.bing.com/search?q={q}&count={n}"
    # default to google
    return f"https://www.google.com/search?q={q}&num={n}&hl=en"


def _decode_search_redirect(u: str) -> str:
    """Strip search-engine redirect wrappers and return the real URL."""
    if not u:
        return ""
    # Google: /url?q=<real>
    if "/url?q=" in u or ("?" in u and "q=" in u and "google." in urlparse(u).netloc):
        m = re.search(r"[?&]q=([^&]+)", u)
        if m:
            return unquote(m.group(1))
    # DuckDuckGo: //duckduckgo.com/l/?uddg=<real>  (also /l/?uddg=)
    low = u.lower()
    if "uddg=" in low:
        m = re.search(r"[?&]uddg=([^&]+)", u)
        if m:
            return unquote(m.group(1))
    # Bing: usually direct, but some go through bing.com/ck/a?...&u=a1<base64>
    if "/ck/a?" in u and "bing." in urlparse(u).netloc:
        # leave bing redirect encoded links alone; rarely worth decoding
        pass
    return u


def extract_search_results(raw_links: list[dict], num: int) -> list[dict]:
    """Clean a raw {url,title} list from a search results page into real
    result URLs, decoding redirects and dropping search-engine chrome."""
    cleaned: list[dict] = []
    seen: set[str] = set()
    for item in raw_links:
        u = _decode_search_redirect(item.get("url", ""))
        if not u or not u.startswith("http"):
            continue
        host = urlparse(u).netloc.lower()
        if not host or any(s in host for s in _SEARCH_SKIP_HOSTS):
            continue
        title = (item.get("title") or item.get("text") or "").strip()
        if len(title) < 3:
            continue
        if u in seen:
            continue
        seen.add(u)
        cleaned.append({"url": u, "title": title[:200]})
        if len(cleaned) >= num:
            break
    return cleaned


# JS to grab all anchors with href + visible text, used for all engines.
_SEARCH_LINKS_JS = (
    "() => {"
    "  const out = [];"
    "  document.querySelectorAll('a[href]').forEach(a => {"
    "    const h = a.href; const t = (a.innerText||a.textContent||'').trim();"
    "    if (h) out.push({url: h, title: t.slice(0,200)});"
    "  });"
    "  return out;"
    "}"
)


def do_search(browser: BrowserClient, engine: str, query: str,
              num: int = 15) -> dict:
    """Run a search on the given engine via the browser and return cleaned
    result links. Detects Cloudflare/Google challenge pages."""
    url = build_search_url(engine, query, num)
    got = browser.goto(url, wait_until="domcontentloaded", timeout=60000)
    if got.get("status") == "error":
        return got
    snap = got.get("data", {})
    ctype = snap.get("challenge_type") or (
        "challenge" if snap.get("cloudflare_challenge") else None
    )
    if snap.get("cloudflare_challenge") or ctype:
        return {
            "status": "ok",
            "engine": engine,
            "banned": True,
            "challenge_type": ctype,
            "note": (f"{engine} is showing a challenge/ban page "
                     f"({ctype}). Switch to another engine (priority google>ddg>bing)."),
        }
    res = browser.evaluate(_SEARCH_LINKS_JS)
    if res.get("status") == "error":
        return {
            "status": "ok",
            "engine": engine,
            "results": [],
            "page_text": truncate(snap.get("text", ""), 2000),
        }
    links = res.get("data", {}).get("result", []) or []
    cleaned = extract_search_results(links, num)
    return {"status": "ok", "engine": engine, "results": cleaned}


# SlowMist Hacked search-box selector (from example.html).
_SLOWMIST_SEARCH_INPUT = ".search-box input[name='q']"
_SLOWMIST_HOME = "https://hacked.slowmist.io/"


def do_slowmist_hacked(browser: BrowserClient, keyword: str) -> dict:
    """Search the SlowMist Hacked database by driving the real page's form.

    The form on https://hacked.slowmist.io/ contains a hidden csrfmiddlewaretoken;
    a raw POST (post_form) would lack it. Instead we: open the home page, fill the
    search input, and press Enter so the browser submits the form natively (CSRF
    token included automatically). Returns the results-page snapshot + links.
    """
    # 1. Open the home page (has the search form with CSRF token).
    got = browser.goto(_SLOWMIST_HOME, wait_until="domcontentloaded", timeout=60000)
    if got.get("status") == "error":
        return got
    snap = got.get("data", {})
    ctype = snap.get("challenge_type") or (
        "challenge" if snap.get("cloudflare_challenge") else None
    )
    if snap.get("cloudflare_challenge") or ctype:
        return {
            "status": "ok",
            "banned": True,
            "challenge_type": ctype,
            "note": f"SlowMist Hacked home is showing a challenge page ({ctype}).",
        }
    # 2. Fill the search input and press Enter to submit the form natively.
    filled = browser.fill(_SLOWMIST_SEARCH_INPUT, keyword, press_enter=True)
    if filled.get("status") == "error":
        return filled
    snap = filled.get("data", {})
    cur_url = snap.get("url", "")
    # 3. If fill+Enter didn't navigate to /search/, try an explicit press.
    if "/search/" not in cur_url:
        pressed = browser.press("Enter", _SLOWMIST_SEARCH_INPUT)
        if pressed.get("status") == "ok":
            snap = pressed.get("data", {})
            cur_url = snap.get("url", "")
    # 4. Extract links from whatever page we're on now.
    res = browser.evaluate(_SEARCH_LINKS_JS)
    links: list[dict] = []
    if res.get("status") != "error":
        raw = res.get("data", {}).get("result", []) or []
        for item in raw:
            u = item.get("url", "")
            if not u or not u.startswith("http"):
                continue
            links.append({"url": u, "title": (item.get("title") or "")[:200]})
            if len(links) >= 30:
                break
    return {
        "status": "ok",
        "keyword": keyword,
        "url": cur_url,
        "title": snap.get("title", ""),
        "results": links,
        "page_text": truncate(snap.get("text", ""), 2000),
    }


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = """You are an on-chain security case report collection agent.

# Goal
Given a case name (an on-chain hack / exploit / sanctions event), use the provided browser tools to search the web and collect the most relevant, authoritative report URLs about that case. For each report you decide to keep, record its metadata. Finally write a case.json manifest via the `finish` tool.

# What you must NOT do
- Do NOT download PDFs or fetch/save file bytes. The user handles PDF fetching later. You only record URLs and metadata.
- Do NOT invent URLs. Every recorded `source` URL must be a real page you reached via the browser.

# Search engines - priority google > ddg > bing
You have a `search(query, engine)` tool with three engines: `google` (default), `ddg` (DuckDuckGo), `bing`.
- Prefer Google first. If Google returns a challenge/ban page (the result has `"banned": true`), FALL BACK to `ddg`, then `bing` - do NOT keep hammering a banned engine.
- There is also a dedicated `slowmist_hacked(keyword)` tool that searches the SlowMist Hacked incident database by driving the real page (open home -> fill search box -> submit form, CSRF-safe). It is a C-grade event database. Use it for the case name / project name to find incident pages and linked analysis.

# Workflow - INTERLEAVE search & visit (never batch searches on one engine)
Running several searches back-to-back on the SAME engine looks like a bot and gets you rate-limited/banned.
After every `search` you MUST immediately `visit_page` 1-3 of its results and read them before the next
search. The script enforces this: a warning is appended after two same-engine searches without a visit.
Rotating the engine (google->ddg->bing) also resets the streak. Repeat this cycle until coverage is sufficient:

  A. `search` - one query, rotated across the themes below (vary phrasing each time). Start with google;
     fall back to ddg/bing if banned.
  B. `visit_page` 1-3 of the most authoritative/relevant results from that search. Read the snapshot;
     `get_page_text` for more body. Confirm the page is genuinely about THIS case.
  C. LAYER 2 (dig deeper) - if a visited page is promising but partial (a summary, a leaderboard entry,
     a tweet, a news index, a press-release listing), call `get_page_links` to list links ON that page,
     then `visit_page` the ones pointing to fuller/primary material: the original postmortem, the
     OFAC/Treasury/FBI entry, a referenced arXiv paper, a linked SlowMist / Chainalysis / PeckShield /
     Elliptic / TRM Labs report, "read full report" buttons, related X/Twitter threads, etc. This second
     layer recovers detail the first-layer hit only summarizes. Do 1-3 layer-2 visits per promising page;
     skip when a page is already the primary source.
  D. For SlowMist specifically, prefer `slowmist_hacked(keyword)` (drives the real page
     form, CSRF-safe) over googling "slowmist hacked" - googling that landing page tends
     to trap the browser in the app's JS. Use the tool, read the returned results/page_text,
     then `visit_page` specific incident/analysis links.
  E. For each confirmed authoritative/relevant page (layer-1 or layer-2), call `add_document`.
  F. Move to the next query theme and repeat.

Query themes to rotate (one per cycle; don't reuse the exact phrasing verbatim):
- "<case> hack report" / "<case> exploit analysis" / "<case> postmortem"
- "<case> OFAC sanctions" / "<case> FBI" / "<case> Treasury" / "<case> DOJ"
- "<case> Chainalysis" / "<case> PeckShield" / "<case> Elliptic" / "<case> TRM Labs" / "<case> Halborn"
- "<case> rekt.news" / "<case> DefiLlama hacks" / "<case> arxiv"
- "<case> <year> Lazarus" / "<case> stolen funds" / "<case> address tx" (if relevant)
Plus: `slowmist_hacked("<project>")` and `slowmist_hacked("<case keyword>")` for the SlowMist DB.

For each `add_document`:
- filename: a clean PDF filename derived from the page title (Windows-safe). Auto-sanitized.
- source: the exact URL of the page you actually reached.
- source_entity: the specific publisher/site (e.g. "OFAC", "U.S. Treasury press release",
  "SlowMist", "Chainalysis", "Rekt.News", "CoinDesk", "X", "ArXiv").
- source_type: one of official | security_company | event_news/database | community_alert | scholar_paper
- source_grade: one of A | B | C | D | E

Aim for good coverage: ~5-10 documents across multiple grades. Prioritize A (official /
sanctions / law-enforcement / project postmortem) and B (security / chain-analysis companies),
then add C (event databases/news), D (X/community alerts), E (academic papers) as available.
Avoid near-duplicate URLs. When coverage is sufficient, call `finish` with the display
`case_name`; the script writes this run's `run<N>.json` into the pre-created case directory.

# Source classification guide (from docs/data_source_class.md)
- A / official: OFAC SDN, U.S. Treasury press releases, FBI/DOJ/Europol/CISA, project-side postmortems, governance/Snapshot/Discord announcements, exchange announcements.
- B / security_company: Chainalysis, Elliptic, TRM Labs, SlowMist/MistTrack, BlockSec/Phalcon, PeckShield, CertiK Skynet, Halborn, Merkle Science, Cyvers, Beosin, Hacken.
- C / event_news/database: DefiLlama Hacks, Rekt.news, SlowMist Hacked, CryptoSec/DeFiHackLabs/SunSec, The Block/CoinDesk/DLNews/Reuters, Dune/Flipside dashboards.
- D / community_alert: X/Twitter posts by ZachXBT, Scam Sniffer, PeckShieldAlert, CyversAlerts, SlowMist_Team, Lookonchain, SpotOnChain, Tayvano, etc. Any x.com/twitter.com post.
- E / scholar_paper: arXiv, academic papers, datasets.

# Cloudflare / Google challenges
If a snapshot reports `cloudflare_challenge: true` (or `challenge_type` is set) after `visit_page`:
- For a Cloudflare checkbox, try `click_relative` near it (e.g. x_ratio~0.5, y_ratio~0.6), then re-check with `get_page_text`.
- For a Google reCAPTCHA (`challenge_type: "google_recaptcha"`) on a SEARCH result page: do NOT try to solve it - switch engine (ddg/bing) or use `slowmist_hacked`.
- If a CONTENT page stays blocked, call `call_user` (urgent=true) describing the URL so the user can solve it manually, then skip that URL and continue.

# Output schema (case.json)
{
  "case_name": "<display name>",
  "documents": [
    {"filename": "<title>.pdf", "source": "<url>", "source_entity": "<entity>",
     "source_type": "<type>", "source_grade": "<grade>"}
  ]
}

Be efficient but thorough: do one `search`, then targeted `visit_page` (plus a little layer-2 `get_page_links` digging on promising pages) rather than many shallow searches. NEVER run two same-engine searches back-to-back; rotate engines or visit a result in between. Stop as soon as you have solid multi-grade coverage and call `finish`."""


class ReportCollectorAgent:
    def __init__(
        self,
        case_name: str,
        browser: BrowserClient,
        llm: LLMClient,
        max_iterations: int = MAX_ITERATIONS,
        verbose: bool = True,
        run_index: int = 1,
        total_runs: int = 1,
        case_dir: Path | None = None,
        run_hint: str = "",
    ) -> None:
        self.case_name = case_name
        self.browser = browser
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.run_index = run_index
        self.total_runs = total_runs
        self.case_dir = case_dir
        self.run_hint = run_hint
        self.documents: list[dict] = []
        self.finished = False
        self.output_path: Path | None = None
        self.consecutive_searches = 0
        self.last_search_engine: str = ""

    # -- logging ----------------------------------------------------------- #
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -- tool implementations ---------------------------------------------- #
    def tool_search(self, query: str, engine: str = "google", num: int = 15) -> Any:
        engine = engine if engine in SEARCH_ENGINES else "google"
        # Per-engine streak: reset when the engine changes so rotating
        # google->ddg->bing is fine, but google->google warns.
        if engine != self.last_search_engine:
            self.consecutive_searches = 0
        self.last_search_engine = engine
        self.consecutive_searches += 1
        time.sleep(random.uniform(1.5, 3.5))
        out = do_search(self.browser, engine, query, num)
        if out.get("status") == "error":
            return out
        if out.get("banned"):
            out["note"] = (
                f"{engine} returned a challenge/ban page. Switch engine: try "
                f"the next in priority ({SEARCH_ENGINE_PRIORITY}), and interleave "
                "with visit_page. You may also use slowmist_hacked for this case."
            )
            return out
        if self.consecutive_searches >= 2:
            out["warning"] = (
                f"{self.consecutive_searches} '{engine}' searches in a row without "
                "a visit_page in between - this triggers bot detection on a single "
                "engine. Visit a result, or switch engine before searching again."
            )
        return out

    def tool_slowmist_hacked(self, keyword: str) -> Any:
        """Search the SlowMist Hacked database by driving the real page form
        (open home -> fill search box -> Enter). Avoids CSRF issues."""
        self.consecutive_searches = 0
        self.last_search_engine = ""
        time.sleep(random.uniform(1.0, 2.5))
        return do_slowmist_hacked(self.browser, keyword)

    def tool_visit_page(self, url: str) -> Any:
        self.consecutive_searches = 0
        self.last_search_engine = ""
        got = self.browser.goto(url, wait_until="domcontentloaded", timeout=60000)
        if got.get("status") == "error":
            return got
        snap = got.get("data", {})
        return {
            "status": "ok",
            "url": snap.get("url", url),
            "title": snap.get("title", ""),
            "cloudflare_challenge": bool(snap.get("cloudflare_challenge")),
            "text": truncate(snap.get("text", ""), 1500),
        }

    def tool_get_page_text(self, selector: str = "", limit: int = 3000) -> Any:
        body: dict = {}
        if selector:
            body["selector"] = selector
        res = self.browser.text(selector or None)
        if res.get("status") == "error":
            return res
        data = res.get("data", {})
        return {
            "status": "ok",
            "url": data.get("url", ""),
            "text": truncate(data.get("text", ""), max(500, min(limit, 6000))),
        }

    def tool_get_page_links(self, pattern: str = "", limit: int = 30) -> Any:
        # Layer-2 discovery: list links on the current page so the agent can dig
        # deeper into referenced primary sources (postmortem, OFAC entry, arxiv, etc.).
        js = (
            "() => {"
            "  const out = [];"
            "  document.querySelectorAll('a[href]').forEach(a => {"
            "    const h = a.href; const t = (a.innerText||a.textContent||'').trim();"
            "    if (h) out.push({url: h, text: t.slice(0,150)});"
            "  });"
            "  return out;"
            "}"
        )
        res = self.browser.evaluate(js)
        if res.get("status") == "error":
            return res
        links = res.get("data", {}).get("result", []) or []
        cleaned: list[dict] = []
        seen: set[str] = set()
        pat = (pattern or "").lower()
        skip_hosts = (
            "google.",
            "gstatic.",
            "googleapis.",
            "googleusercontent.",
            "youtube.",
            "youtu.be",
            "blogger.",
            "webcache.",
        )
        for item in links:
            u = item.get("url", "")
            if not u or u.startswith("javascript:") or u.startswith("#"):
                continue
            host = urlparse(u).netloc.lower()
            if not host or any(s in host for s in skip_hosts):
                continue
            txt = item.get("text", "")
            if pat and pat not in u.lower() and pat not in txt.lower():
                continue
            if u in seen:
                continue
            seen.add(u)
            cleaned.append({"url": u, "text": txt})
            if len(cleaned) >= limit:
                break
        return {"status": "ok", "count": len(cleaned), "links": cleaned}

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
        return {
            "status": "ok",
            "url": data.get("url", ""),
            "result": truncate(json.dumps(result, ensure_ascii=False), 4000)
            if not isinstance(result, str)
            else truncate(result, 4000),
        }

    def tool_take_screenshot(
        self, full_page: bool = False, return_base64: bool = False
    ) -> Any:
        res = self.browser.screenshot(full_page=full_page, return_base64=return_base64)
        if res.get("status") == "error":
            return res
        data = res.get("data", {})
        out = {
            "status": "ok",
            "url": data.get("url", ""),
            "viewport": data.get("viewport"),
            "path": data.get("path"),
        }
        if return_base64 and data.get("image_base64"):
            out["image_base64"] = data["image_base64"][:200] + "...[truncated]"
            out["note"] = (
                "base64 truncated in tool output; call with return_base64=false normally."
            )
        return out

    def tool_call_user(self, content: str, urgent: bool = False) -> Any:
        try:
            from call_user import send_msg

            title = "[Urgent] From LLM Agent" if urgent else "From LLM Agent"
            send_msg(title=title, desp=content)
            return {"status": "ok", "message": "User notified."}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": f"Failed to notify user: {e}"}

    def tool_list_documents(self) -> Any:
        return {
            "status": "ok",
            "count": len(self.documents),
            "documents": self.documents,
        }

    def tool_add_document(
        self,
        filename: str,
        source: str,
        source_entity: str,
        source_type: str,
        source_grade: str,
    ) -> Any:
        valid_types = {
            "official",
            "security_company",
            "event_news/database",
            "community_alert",
            "scholar_paper",
        }
        valid_grades = {"A", "B", "C", "D", "E"}
        if source_type not in valid_types:
            return {
                "status": "error",
                "message": f"source_type must be one of {sorted(valid_types)}",
            }
        if source_grade not in valid_grades:
            return {
                "status": "error",
                "message": f"source_grade must be one of {sorted(valid_grades)}",
            }
        if not source.startswith("http"):
            return {"status": "error", "message": "source must be an http(s) URL"}
        for d in self.documents:
            if d["source"].rstrip("/") == source.rstrip("/"):
                return {
                    "status": "ok",
                    "message": "duplicate, already collected",
                    "count": len(self.documents),
                }
        doc = {
            "filename": sanitize_filename(filename),
            "source": source,
            "source_entity": source_entity,
            "source_type": source_type,
            "source_grade": source_grade,
        }
        self.documents.append(doc)
        self.log(
            f"  + [{source_grade}/{source_type}] {source_entity}: {doc['filename']}"
        )
        return {"status": "ok", "count": len(self.documents), "document": doc}

    def tool_finish(self, case_name: str) -> Any:
        if self.case_dir is None:
            return {"status": "error", "message": "case_dir not set for this run"}
        manifest = {
            "case_name": case_name or self.case_name,
            "documents": self.documents,
        }
        out_path = self.case_dir / f"run{self.run_index}.json"
        save_report_json(out_path, manifest, document_type="report_collection_run")
        self.output_path = out_path
        self.finished = True
        self.log(
            f"\n[run {self.run_index} done] wrote {out_path} "
            f"({len(self.documents)} documents)"
        )
        return {
            "status": "ok",
            "path": str(out_path),
            "run": self.run_index,
            "document_count": len(self.documents),
        }

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
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": (
                        "Search the web via the browser and return a result {url,title} list. "
                        "Engines: google (default, best), ddg (DuckDuckGo), bing. Priority "
                        "google>ddg>bing - fall back to the next engine if a search returns "
                        "banned:true (challenge/rate-limit page). Vary keywords across calls."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "engine": {
                                "type": "string",
                                "enum": ["google", "ddg", "bing"],
                                "default": "google",
                            },
                            "num": {
                                "type": "integer",
                                "description": "Approx number of results",
                                "default": 15,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "slowmist_hacked",
                    "description": (
                        "Search the SlowMist Hacked incident database by opening the real "
                        "page (hacked.slowmist.io), filling its search box, and submitting "
                        "the form (CSRF-safe). Returns the results page text + incident "
                        "links. Use the case/project name as keyword. Prefer this over "
                        "googling 'slowmist hacked' (which traps the browser in the app's JS). "
                        "SlowMist Hacked is a C-grade event database."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {
                                "type": "string",
                                "description": "Case or project name to look up",
                            },
                        },
                        "required": ["keyword"],
                    },
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
                            "selector": {
                                "type": "string",
                                "description": "Optional CSS selector",
                            },
                            "limit": {"type": "integer", "default": 3000},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_page_links",
                    "description": "List links on the current page as {url,text}, optionally filtered by a case-insensitive substring matched against url or link text. Use for LAYER-2 digging: on a promising but partial page, find links to fuller/primary material (e.g. 'full report', 'original post', OFAC/Treasury entry, arxiv, related X thread, referenced SlowMist/Chainalysis report) and visit_page them.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Optional substring filter (case-insensitive) on url or link text",
                            },
                            "limit": {"type": "integer", "default": 30},
                        },
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
                    "name": "take_screenshot",
                    "description": "Take a screenshot. Returns viewport/path (base64 is truncated, so prefer return_base64=false).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "full_page": {"type": "boolean", "default": False},
                            "return_base64": {"type": "boolean", "default": False},
                        },
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
                    "name": "list_documents",
                    "description": "List documents collected so far.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "add_document",
                    "description": "Record a confirmed relevant report. filename is derived from page title; source must be a real visited URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "source": {"type": "string"},
                            "source_entity": {"type": "string"},
                            "source_type": {
                                "type": "string",
                                "enum": [
                                    "official",
                                    "security_company",
                                    "event_news/database",
                                    "community_alert",
                                    "scholar_paper",
                                ],
                            },
                            "source_grade": {
                                "type": "string",
                                "enum": ["A", "B", "C", "D", "E"],
                            },
                        },
                        "required": [
                            "filename",
                            "source",
                            "source_entity",
                            "source_type",
                            "source_grade",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": "Finalize this run: write run<N>.json from collected documents into the case directory and end. The directory is already created for you.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "case_name": {
                                "type": "string",
                                "description": "Display case name",
                            },
                        },
                        "required": ["case_name"],
                    },
                },
            },
        ]

    # -- dispatch ---------------------------------------------------------- #
    def dispatch(self, name: str, args: dict) -> Any:
        handlers = {
            "search": lambda a: self.tool_search(
                a["query"], a.get("engine", "google"), a.get("num", 15)
            ),
            "slowmist_hacked": lambda a: self.tool_slowmist_hacked(a["keyword"]),
            "visit_page": lambda a: self.tool_visit_page(a["url"]),
            "get_page_text": lambda a: self.tool_get_page_text(
                a.get("selector", ""), a.get("limit", 3000)
            ),
            "get_page_links": lambda a: self.tool_get_page_links(
                a.get("pattern", ""), a.get("limit", 30)
            ),
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
            "take_screenshot": lambda a: self.tool_take_screenshot(
                a.get("full_page", False), a.get("return_base64", False)
            ),
            "call_user": lambda a: self.tool_call_user(
                a["content"], a.get("urgent", False)
            ),
            "list_documents": lambda a: self.tool_list_documents(),
            "add_document": lambda a: self.tool_add_document(
                a["filename"],
                a["source"],
                a["source_entity"],
                a["source_type"],
                a["source_grade"],
            ),
            "finish": lambda a: self.tool_finish(a["case_name"]),
        }
        handler = handlers.get(name)
        if not handler:
            return {"status": "error", "message": f"Unknown tool: {name}"}
        try:
            return handler(args)
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": f"{type(e).__name__}: {e}"}

    # -- main loop --------------------------------------------------------- #
    def run(self) -> Path | None:
        user_content = (
            f"Case name: {self.case_name}\n\n"
            f"This is collection run {self.run_index} of {self.total_runs}. "
            "Multiple independent runs are executed and later merged, so you do NOT need "
            "to cover everything - but you should cover YOUR focus well. Collect relevant "
            "report sources for this case and call `finish` with the display case_name."
        )
        if self.run_hint:
            user_content += f"\n\n# This run's focus (for diversity across runs)\n{self.run_hint}"
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        no_tool_turns = 0
        for i in range(1, self.max_iterations + 1):
            self.log(f"\n--- iteration {i} ---")
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
                    if self.documents:
                        self.log(
                            "[agent idle] forcing finish with collected documents."
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": "You have not used a tool. Call `finish` now with the display case_name.",
                            }
                        )
                    else:
                        self.log("[agent idle] no documents and no tool use; stopping.")
                        break
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Use a tool to continue searching, or call `finish` when you have enough documents.",
                        }
                    )
                continue
            no_tool_turns = 0

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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )
                if name == "finish" and self.finished:
                    return self.output_path

        if not self.finished and self.documents:
            self.log("[max iterations] auto-finalizing with collected documents.")
            self.tool_finish(self.case_name)
            return self.output_path
        if not self.finished:
            self.log("[stopped] no documents collected; nothing written.")
        return self.output_path

    def _guess_year(self) -> int:
        m = re.search(r"(20\d{2})", self.case_name)
        return int(m.group(1)) if m else 2024

    def _guess_slug(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", self.case_name).strip("-")
        return slug or "case"


# --------------------------------------------------------------------------- #
# Aggregation agent - merges multiple runs into a final case.json
# --------------------------------------------------------------------------- #


AGGREGATE_PROMPT = """You are the aggregation agent for an on-chain security case report collection.

# Goal
Several independent collection runs each produced a `run<N>.json` of candidate report sources for the same case. Their documents have been merged and deduplicated by URL, and you are given the resulting candidate list (each entry has a `provenance` field = which runs found it, and `run_count` = how many runs). Your job is to produce the final, curated `case.json` document list.

# What you must do
1. Review the candidate list. `run_count` is a confidence signal: documents found by multiple runs are very likely real and relevant; documents found by only one run need a closer look.
2. For LOW-confidence candidates (run_count == 1) or any that look possibly off-topic / noise / tangentially related, use `visit_page` to open the URL and confirm it is genuinely about THIS case and is a substantive report (not a generic landing page, an unrelated sanctions entry, a tangential policy PDF, a broken page, etc.). Drop noise.
3. If two candidates are clearly the same underlying document at different URLs (e.g. arxiv abstract vs html vs pdf, or a blog with/without trailing params), keep the canonical/best one and drop the other.
4. Ensure good coverage across grades (A official, B security-company, C event-database/news, D community, E academic) when available, but do NOT pad with weak/irrelevant items. Quality over quantity. Around 5-15 documents is typical.
5. You may use `search` (engines google>ddg>bing, fall back if banned) or `slowmist_hacked` to fill an obvious gap (e.g. if no official postmortem was found), but this is optional - prefer working from the candidates.
6. When the final list is decided, call `finish` with the display `case_name` and the full `documents` array. Each document must have: filename, source, source_entity, source_type (official|security_company|event_news/database|community_alert|scholar_paper), source_grade (A|B|C|D|E).

# Rules
- Do NOT invent URLs. Only include sources from the candidate list or pages you actually reached via the browser.
- Do NOT download/save PDFs. You only record metadata.
- filename should be a clean PDF name derived from the page title (Windows-safe); it is sanitized automatically.

# Source classification (from docs/data_source_class.md)
- A / official: OFAC SDN, U.S. Treasury press releases, FBI/DOJ/Europol/CISA, project postmortems, governance/exchange announcements.
- B / security_company: Chainalysis, Elliptic, TRM Labs, SlowMist/MistTrack, BlockSec/Phalcon, PeckShield, CertiK, Halborn, Merkle Science, Cyvers, Beosin, Hacken.
- C / event_news/database: DefiLlama Hacks, Rekt.news, SlowMist Hacked, The Block/CoinDesk/DLNews/Reuters, Dune/Flipside.
- D / community_alert: x.com/twitter.com posts (ZachXBT, Scam Sniffer, PeckShieldAlert, Lookonchain, etc.).
- E / scholar_paper: arXiv, academic papers, datasets.

Be efficient: you do not need to visit every candidate - trust high run_count entries and only verify the dubious ones. Call `finish` once done."""


class AggregateAgent:
    def __init__(
        self,
        case_name: str,
        case_dir: Path,
        candidates: list[dict],
        browser: BrowserClient,
        llm: LLMClient,
        max_iterations: int = 30,
        verbose: bool = True,
    ) -> None:
        self.case_name = case_name
        self.case_dir = case_dir
        self.candidates = candidates
        self.browser = browser
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.finished = False
        self.output_path: Path | None = None
        self.consecutive_searches = 0
        self.last_search_engine: str = ""

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -- tool implementations ---------------------------------------------- #
    def tool_search(self, query: str, engine: str = "google", num: int = 15) -> Any:
        engine = engine if engine in SEARCH_ENGINES else "google"
        if engine != self.last_search_engine:
            self.consecutive_searches = 0
        self.last_search_engine = engine
        self.consecutive_searches += 1
        time.sleep(random.uniform(1.5, 3.5))
        out = do_search(self.browser, engine, query, num)
        if out.get("status") == "error":
            return out
        if out.get("banned"):
            out["note"] = (
                f"{engine} returned a challenge/ban page. Switch engine "
                f"(priority {SEARCH_ENGINE_PRIORITY})."
            )
            return out
        if self.consecutive_searches >= 2:
            out["warning"] = (
                f"{self.consecutive_searches} '{engine}' searches in a row - "
                "visit a result or switch engine before searching again."
            )
        return out

    def tool_slowmist_hacked(self, keyword: str) -> Any:
        self.consecutive_searches = 0
        self.last_search_engine = ""
        time.sleep(random.uniform(1.0, 2.5))
        return do_slowmist_hacked(self.browser, keyword)

    def tool_visit_page(self, url: str) -> Any:
        self.consecutive_searches = 0
        self.last_search_engine = ""
        got = self.browser.goto(url, wait_until="domcontentloaded", timeout=60000)
        if got.get("status") == "error":
            return got
        snap = got.get("data", {})
        return {
            "status": "ok",
            "url": snap.get("url", url),
            "title": snap.get("title", ""),
            "cloudflare_challenge": bool(snap.get("cloudflare_challenge")),
            "text": truncate(snap.get("text", ""), 1500),
        }

    def tool_get_page_text(self, selector: str = "", limit: int = 3000) -> Any:
        res = self.browser.text(selector or None)
        if res.get("status") == "error":
            return res
        data = res.get("data", {})
        return {"status": "ok", "url": data.get("url", ""),
                "text": truncate(data.get("text", ""), max(500, min(limit, 6000)))}

    def tool_get_page_links(self, pattern: str = "", limit: int = 30) -> Any:
        js = (
            "() => {"
            "  const out = [];"
            "  document.querySelectorAll('a[href]').forEach(a => {"
            "    const h = a.href; const t = (a.innerText||a.textContent||'').trim();"
            "    if (h) out.push({url: h, text: t.slice(0,150)});"
            "  });"
            "  return out;"
            "}"
        )
        res = self.browser.evaluate(js)
        if res.get("status") == "error":
            return res
        links = res.get("data", {}).get("result", []) or []
        cleaned: list[dict] = []
        seen: set[str] = set()
        pat = (pattern or "").lower()
        skip_hosts = ("google.", "gstatic.", "googleapis.", "googleusercontent.",
                      "youtube.", "youtu.be", "blogger.", "webcache.")
        for item in links:
            u = item.get("url", "")
            if not u or u.startswith("javascript:") or u.startswith("#"):
                continue
            host = urlparse(u).netloc.lower()
            if not host or any(s in host for s in skip_hosts):
                continue
            txt = item.get("text", "")
            if pat and pat not in u.lower() and pat not in txt.lower():
                continue
            if u in seen:
                continue
            seen.add(u)
            cleaned.append({"url": u, "text": txt})
            if len(cleaned) >= limit:
                break
        return {"status": "ok", "count": len(cleaned), "links": cleaned}

    def tool_call_user(self, content: str, urgent: bool = False) -> Any:
        try:
            from call_user import send_msg

            title = "[Urgent] From LLM Agent" if urgent else "From LLM Agent"
            send_msg(title=title, desp=content)
            return {"status": "ok", "message": "User notified."}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": f"Failed to notify user: {e}"}

    def tool_finish(self, case_name: str, documents: list[dict]) -> Any:
        valid_types = {"official", "security_company", "event_news/database",
                       "community_alert", "scholar_paper"}
        valid_grades = {"A", "B", "C", "D", "E"}
        cleaned: list[dict] = []
        seen_keys: set[str] = set()
        for d in documents or []:
            source = d.get("source", "")
            if not source.startswith("http"):
                return {"status": "error",
                        "message": f"invalid source url: {source!r}"}
            stype = d.get("source_type", "")
            sgrade = d.get("source_grade", "")
            if stype not in valid_types:
                return {"status": "error",
                        "message": f"source_type must be one of {sorted(valid_types)}; got {stype!r}"}
            if sgrade not in valid_grades:
                return {"status": "error",
                        "message": f"source_grade must be one of {sorted(valid_grades)}; got {sgrade!r}"}
            key = normalize_url(source)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cleaned.append({
                "filename": sanitize_filename(d.get("filename", "")),
                "source": source,
                "source_entity": d.get("source_entity", ""),
                "source_type": stype,
                "source_grade": sgrade,
            })
        manifest = {
            "case_name": case_name or self.case_name,
            "documents": cleaned,
        }
        out_path = self.case_dir / "case.json"
        save_report_json(out_path, manifest, document_type="case_report_manifest")
        self.output_path = out_path
        self.finished = True
        self.log(f"\n[aggregate done] wrote {out_path} ({len(cleaned)} documents)")
        return {"status": "ok", "path": str(out_path),
                "document_count": len(cleaned)}

    # -- tool schemas ------------------------------------------------------ #
    @property
    def tools(self) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": "search",
                "description": ("Search the web (optional, only to fill an obvious gap). Engines "
                                "google>ddg>bing; fall back if a search returns banned:true. "
                                "Returns {url,title} list."),
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"},
                    "engine": {"type": "string", "enum": ["google", "ddg", "bing"], "default": "google"},
                    "num": {"type": "integer", "default": 15}},
                    "required": ["query"]}}},
            {"type": "function", "function": {
                "name": "slowmist_hacked",
                "description": "Search the SlowMist Hacked incident database by opening the real page and submitting its search form (CSRF-safe). Returns results + page_text.",
                "parameters": {"type": "object", "properties": {
                    "keyword": {"type": "string"}},
                    "required": ["keyword"]}}},
            {"type": "function", "function": {
                "name": "visit_page",
                "description": "Open a URL and return a snapshot (url,title,text,cloudflare_challenge). Use to verify dubious candidates.",
                "parameters": {"type": "object", "properties": {
                    "url": {"type": "string"}},
                    "required": ["url"]}}},
            {"type": "function", "function": {
                "name": "get_page_text",
                "description": "Return (more) text of the current page.",
                "parameters": {"type": "object", "properties": {
                    "selector": {"type": "string"},
                    "limit": {"type": "integer", "default": 3000}}}}},
            {"type": "function", "function": {
                "name": "get_page_links",
                "description": "List links on the current page, optionally filtered by substring.",
                "parameters": {"type": "object", "properties": {
                    "pattern": {"type": "string"},
                    "limit": {"type": "integer", "default": 30}}}}},
            {"type": "function", "function": {
                "name": "call_user",
                "description": "Notify the user (e.g. blocked by Cloudflare).",
                "parameters": {"type": "object", "properties": {
                    "content": {"type": "string"},
                    "urgent": {"type": "boolean", "default": False}},
                    "required": ["content"]}}},
            {"type": "function", "function": {
                "name": "finish",
                "description": "Finalize: write case.json from the curated documents array and end. Each document needs filename, source, source_entity, source_type, source_grade.",
                "parameters": {"type": "object", "properties": {
                    "case_name": {"type": "string"},
                    "documents": {"type": "array", "items": {"type": "object", "properties": {
                        "filename": {"type": "string"},
                        "source": {"type": "string"},
                        "source_entity": {"type": "string"},
                        "source_type": {"type": "string"},
                        "source_grade": {"type": "string"}},
                        "required": ["filename", "source", "source_entity", "source_type", "source_grade"]}}},
                    "required": ["case_name", "documents"]}}},
        ]

    # -- dispatch ---------------------------------------------------------- #
    def dispatch(self, name: str, args: dict) -> Any:
        handlers = {
            "search": lambda a: self.tool_search(a["query"], a.get("engine", "google"), a.get("num", 15)),
            "slowmist_hacked": lambda a: self.tool_slowmist_hacked(a["keyword"]),
            "visit_page": lambda a: self.tool_visit_page(a["url"]),
            "get_page_text": lambda a: self.tool_get_page_text(a.get("selector", ""), a.get("limit", 3000)),
            "get_page_links": lambda a: self.tool_get_page_links(a.get("pattern", ""), a.get("limit", 30)),
            "call_user": lambda a: self.tool_call_user(a["content"], a.get("urgent", False)),
            "finish": lambda a: self.tool_finish(a["case_name"], a.get("documents", [])),
        }
        handler = handlers.get(name)
        if not handler:
            return {"status": "error", "message": f"Unknown tool: {name}"}
        try:
            return handler(args)
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": f"{type(e).__name__}: {e}"}

    # -- main loop --------------------------------------------------------- #
    def run(self) -> Path | None:
        cand_str = json.dumps(self.candidates, ensure_ascii=False, indent=2)
        messages: list[dict] = [
            {"role": "system", "content": AGGREGATE_PROMPT},
            {"role": "user", "content": (
                f"Case name: {self.case_name}\n\n"
                f"# Merged candidate documents from {len(self.candidates)} unique sources "
                f"(provenance = which runs found each):\n\n{cand_str}\n\n"
                "Curate this into the final case.json: verify dubious (run_count==1 or "
                "off-topic) entries via visit_page, drop noise/duplicates, then call "
                "`finish` with the final documents list."
            )},
        ]
        no_tool_turns = 0
        for i in range(1, self.max_iterations + 1):
            self.log(f"\n--- aggregate iteration {i} ---")
            try:
                response = self.llm.complete(messages, tools=self.tools, tool_choice="auto")
            except Exception as e:  # noqa: BLE001
                self.log(f"[llm error] {e}; retrying in 5s")
                time.sleep(5)
                continue

            choice = response.choices[0]
            msg = choice.message
            assistant_msg = (
                msg.model_dump(exclude_none=True)
                if hasattr(msg, "model_dump") else dict(msg)
            )
            messages.append(assistant_msg)
            if msg.content:
                self.log(f"[assistant] {truncate(str(msg.content), 600)}")

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                no_tool_turns += 1
                if no_tool_turns >= 2:
                    self.log("[aggregate idle] auto-finalizing from candidates.")
                    self.tool_finish(self.case_name, self._fallback_documents())
                    return self.output_path
                messages.append({"role": "user",
                                 "content": "Use a tool (visit_page to verify, or finish) to continue."})
                continue
            no_tool_turns = 0

            for tc in tool_calls:
                fn = tc.function
                name = fn.name
                try:
                    args = json.loads(fn.arguments) if fn.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                self.log(f"[tool] {name}({truncate(json.dumps(args, ensure_ascii=False), 200)})")
                result = self.dispatch(name, args)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "name": name,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
                if name == "finish" and self.finished:
                    return self.output_path

        if not self.finished:
            self.log("[aggregate max iterations] auto-finalizing from candidates.")
            self.tool_finish(self.case_name, self._fallback_documents())
        return self.output_path

    def _fallback_documents(self) -> list[dict]:
        """Best-effort final list: drop provenance-only fields, keep all candidates
        sorted as-is (merge_runs already sorted by confidence)."""
        out: list[dict] = []
        for c in self.candidates:
            out.append({
                "filename": c.get("filename", ""),
                "source": c.get("source", ""),
                "source_entity": c.get("source_entity", ""),
                "source_type": c.get("source_type", ""),
                "source_grade": c.get("source_grade", ""),
            })
        return out


# --------------------------------------------------------------------------- #
# Orchestration: multi-run collection + aggregation
# --------------------------------------------------------------------------- #


def resolve_case_dir(year: int, slug: str) -> Path:
    idx = next_case_index()
    dir_name = f"{idx:02d}-{year}-{slug}"
    case_dir = CASES_DIR / dir_name
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def collect_multi(
    case_name: str,
    runs: int,
    browser: BrowserClient,
    llm: LLMClient,
    year: int | None = None,
    slug: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
    verbose: bool = True,
) -> Path | None:
    if year is None:
        m = re.search(r"(20\d{2})", case_name)
        year = int(m.group(1)) if m else 2024
    if not slug:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", case_name).strip("-") or "case"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", slug).strip("-") or "case"

    case_dir = resolve_case_dir(year, slug)
    print(f"[orchestrator] case directory: {case_dir}", flush=True)

    run_docs: dict[int, list[dict]] = {}
    for i in range(1, runs + 1):
        hint = RUN_HINTS[(i - 1) % len(RUN_HINTS)]
        print(f"\n========================= RUN {i}/{runs} =========================",
              flush=True)
        agent = ReportCollectorAgent(
            case_name=case_name,
            browser=browser,
            llm=llm,
            max_iterations=max_iterations,
            verbose=verbose,
            run_index=i,
            total_runs=runs,
            case_dir=case_dir,
            run_hint=hint,
        )
        agent.run()
        # ensure runN.json exists even if the agent never called finish
        run_file = case_dir / f"run{i}.json"
        if not run_file.exists():
            manifest = {"case_name": case_name, "documents": agent.documents}
            save_report_json(
                run_file,
                manifest,
                document_type="report_collection_run_fallback",
            )
            if verbose:
                print(f"[orchestrator] wrote fallback {run_file} "
                      f"({len(agent.documents)} documents)", flush=True)
        try:
            data = json.loads(run_file.read_text(encoding="utf-8"))
            run_docs[i] = data.get("documents", [])
        except Exception as e:  # noqa: BLE001
            print(f"[orchestrator] failed to read {run_file}: {e}", file=sys.stderr)
            run_docs[i] = agent.documents
        print(f"[orchestrator] run {i} collected {len(run_docs[i])} documents",
              flush=True)
        # pause between runs to reduce bot-detection correlation
        if i < runs:
            time.sleep(random.uniform(3, 6))

    print("\n========================= AGGREGATION =========================",
          flush=True)
    candidates = merge_runs(run_docs)
    print(f"[orchestrator] {len(candidates)} unique candidate sources across {runs} runs",
          flush=True)
    agg = AggregateAgent(
        case_name=case_name,
        case_dir=case_dir,
        candidates=candidates,
        browser=browser,
        llm=llm,
        verbose=verbose,
    )
    out = agg.run()
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect report sources for an on-chain case via an LLM browser agent "
                    "(multi-run collection + aggregation)."
    )
    parser.add_argument("case", help="Case name, e.g. 'Ronin Bridge Hack 2022'")
    parser.add_argument(
        "--browser-base", default=BROWSER_BASE, help="Browser service base URL"
    )
    parser.add_argument("--runs", type=int, default=4,
                        help="Number of independent collection runs (default 4)")
    parser.add_argument("--year", type=int, default=None,
                        help="Event year for the case directory (default: inferred from case name)")
    parser.add_argument("--slug", default=None,
                        help="Short ASCII slug for the case directory (default: derived from case name)")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS,
                        help="Max LLM iterations per collection run")
    parser.add_argument("--quiet", action="store_true", help="Less logging")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")

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

    litellm.drop_params = True
    litellm.set_verbose = False

    out = collect_multi(
        case_name=args.case,
        runs=args.runs,
        browser=browser,
        llm=llm,
        year=args.year,
        slug=args.slug,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )
    if out:
        print(f"\nResult: {out}")
        return 0
    print("\nNo result produced.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
