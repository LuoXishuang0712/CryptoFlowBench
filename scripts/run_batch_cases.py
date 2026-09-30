from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_selection import read_case_list

CASES_DIR = ROOT / "cases"
SUMMARIZED_DIR = ROOT / "summarized"
REPORT_PATH = ROOT / "batch_run_report.md"
LOG_DIR = ROOT / "batch_logs"

MAX_ATTEMPTS = 3
TIMEOUTS = {
    "collect": 90 * 60,
    "extract": 90 * 60,
    "chain": 45 * 60,
}


@dataclass
class CaseJob:
    name: str
    year: int
    slug: str
    case_dir: str | None = None
    statuses: dict[str, str] = field(
        default_factory=lambda: {"collect": "pending", "extract": "pending", "chain": "pending"}
    )
    notes: list[str] = field(default_factory=list)


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def slugify(name: str) -> str:
    compact = re.sub(r"\b20\d{2}\b", "", name)
    compact = compact.replace("/", " ")
    compact = re.sub(r"[^A-Za-z0-9_-]+", "-", compact).strip("-")
    return compact or "case"


def name_tokens(value: str) -> set[str]:
    generic = {
        "2022",
        "2023",
        "2024",
        "2025",
        "bridge",
        "bridges",
        "finance",
        "protocol",
        "capital",
        "markets",
        "pools",
        "elastic",
        "v2",
    }
    tokens = set(re.findall(r"[a-z0-9]+", value.lower()))
    return {token for token in tokens if token not in generic}


def compatible_case_name(job: CaseJob, path: Path) -> bool:
    target = name_tokens(job.name)
    sources = [path.name]
    manifest = read_json(path / "case.json")
    if manifest.get("case_name"):
        sources.append(str(manifest["case_name"]))
    for source in sources:
        candidate = name_tokens(source)
        if not candidate or not target:
            continue
        if candidate <= target or target <= candidate:
            return True
        if len(candidate & target) >= min(2, len(target), len(candidate)):
            return True
    return False


def parse_cases() -> list[CaseJob]:
    jobs: list[CaseJob] = []
    for case_dir in read_case_list():
        manifest = read_json(CASES_DIR / case_dir / "case.json")
        year_match = re.search(r"(?:^|-)(20\d{2})(?:-|$)", case_dir)
        year = int(year_match.group(1)) if year_match else 2024
        fallback_name = re.sub(r"^\d+-20\d{2}-", "", case_dir).replace("-", " ")
        name = str(manifest.get("case_name") or fallback_name)
        jobs.append(
            CaseJob(
                name=name,
                year=year,
                slug=slugify(name),
                case_dir=case_dir,
            )
        )
    return jobs


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_existing_case(job: CaseJob) -> str | None:
    candidates = []
    for path in CASES_DIR.glob(f"*-{job.year}-{job.slug}"):
        if (path / "case.json").exists():
            candidates.append(path)
    if not candidates:
        for path in CASES_DIR.glob(f"*-{job.year}-*"):
            if (path / "case.json").exists() and compatible_case_name(job, path):
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[0].name


def has_case_json(case_dir: str | None) -> bool:
    return bool(case_dir) and (CASES_DIR / case_dir / "case.json").exists()


def has_summary(case_dir: str | None) -> bool:
    return bool(case_dir) and (SUMMARIZED_DIR / case_dir / "summary.json").exists()


def has_chain(case_dir: str | None) -> bool:
    return bool(case_dir) and (SUMMARIZED_DIR / case_dir / "chain_evidence.json").exists()


def snapshot_case_dirs() -> set[str]:
    if not CASES_DIR.exists():
        return set()
    return {p.name for p in CASES_DIR.iterdir() if p.is_dir()}


def remove_failed_new_dirs(before: set[str], keep: str | None, job: CaseJob) -> None:
    after = snapshot_case_dirs()
    for name in sorted(after - before):
        if name == keep:
            continue
        path = CASES_DIR / name
        if (path / "case.json").exists():
            job.notes.append(f"left unexpected generated directory with case.json: {name}")
            continue
        shutil.rmtree(path)
        job.notes.append(f"removed failed generated directory: {name}")


def run_command(job: CaseJob, step: str, command: str, timeout: int, attempt: int) -> tuple[int, str]:
    LOG_DIR.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{job.slug}_{step}_attempt{attempt}")
    log_path = LOG_DIR / f"{safe}.log"
    started = datetime.now().isoformat(timespec="seconds")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {started}\n# {command}\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            return proc.returncode, str(log_path.relative_to(ROOT))
        except subprocess.TimeoutExpired:
            log.write(f"\n[TIMEOUT] exceeded {timeout} seconds\n")
            return 124, str(log_path.relative_to(ROOT))


def write_report(jobs: list[CaseJob]) -> None:
    lines = [
        "# Batch Case Run Report",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Case | Directory | Collect | Extract | Chain Evidence | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for job in jobs:
        notes = "<br>".join(job.notes) if job.notes else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    job.name.replace("|", "\\|"),
                    job.case_dir or "",
                    job.statuses["collect"],
                    job.statuses["extract"],
                    job.statuses["chain"],
                    notes.replace("|", "\\|"),
                ]
            )
            + " |"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mark_existing(job: CaseJob) -> None:
    if not job.case_dir or not (CASES_DIR / job.case_dir).is_dir():
        job.case_dir = find_existing_case(job)
    if has_case_json(job.case_dir):
        job.statuses["collect"] = "skipped: case.json exists"
    if has_summary(job.case_dir):
        job.statuses["extract"] = "skipped: summary.json exists"
    if has_chain(job.case_dir):
        job.statuses["chain"] = "skipped: chain_evidence.json exists"


def attempt_step(job: CaseJob, step: str, command: str, timeout: int) -> bool:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        before = snapshot_case_dirs()
        code, log = run_command(job, step, command, timeout, attempt)
        if step == "collect":
            found = find_existing_case(job)
            if found:
                job.case_dir = found
            remove_failed_new_dirs(before, job.case_dir, job)
        if code == 0:
            job.statuses[step] = f"ok attempt {attempt}"
            job.notes.append(f"{step} log: {log}")
            return True
        job.notes.append(f"{step} attempt {attempt} failed with code {code}; log: {log}")
        job.statuses[step] = f"failed attempt {attempt}"
        write_report(all_jobs)
        time.sleep(5)
    return False


def process(job: CaseJob) -> None:
    mark_existing(job)
    if not has_case_json(job.case_dir):
        command = (
            "uv run collect_report.py "
            f"{powershell_quote(job.name)} --year {job.year} --slug {powershell_quote(job.slug)}"
        )
        if not attempt_step(job, "collect", command, TIMEOUTS["collect"]):
            job.statuses["extract"] = "skipped: collect failed"
            job.statuses["chain"] = "skipped: collect failed"
            return
    if not has_summary(job.case_dir):
        command = f"uv run extract_report_info.py {powershell_quote(job.case_dir or '')}"
        if not attempt_step(job, "extract", command, TIMEOUTS["extract"]):
            job.statuses["chain"] = "skipped: extract failed"
            return
    if not has_chain(job.case_dir):
        command = (
            f"uv run chain_evidence.py {powershell_quote(job.case_dir or '')} "
            "--enrich-local-index --enrich-etherscan"
        )
        attempt_step(job, "chain", command, TIMEOUTS["chain"])


all_jobs = parse_cases()


def main() -> int:
    for job in all_jobs:
        mark_existing(job)
    write_report(all_jobs)
    for job in all_jobs:
        process(job)
        write_report(all_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
