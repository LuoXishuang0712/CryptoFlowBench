from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from storage import default_namespace, get_json_store, read_json  # noqa: E402


DOCUMENT_NAMES = {
    "case.json": "case_report_manifest",
    "summary.json": "case_report_summary",
    "chain_evidence.json": "chain_evidence",
}


def discover_documents(root: Path, case: str = "all", *, include_runs: bool = False) -> list[Path]:
    paths: list[Path] = []
    for base_name, filenames in (
        ("cases", ("case.json",)),
        ("summarized", ("summary.json", "chain_evidence.json")),
    ):
        base = root / base_name
        if not base.exists():
            continue
        case_dirs = [base / case] if case != "all" else sorted(base.iterdir())
        for case_dir in case_dirs:
            if not case_dir.is_dir() or case_dir.name.startswith(("_", "xx-")):
                continue
            for filename in filenames:
                path = case_dir / filename
                if path.is_file():
                    paths.append(path)
            if include_runs and base_name == "cases":
                paths.extend(sorted(case_dir.glob("run*.json")))
    return sorted(set(paths))


def metadata_for(path: Path, payload: Any) -> dict[str, Any]:
    document_type = DOCUMENT_NAMES.get(path.name, "report_collection_run")
    metadata: dict[str, Any] = {
        "document_type": document_type,
        "case_dir": path.parent.name,
        "imported_by": "scripts/import_report_chain_documents.py",
    }
    if isinstance(payload, dict):
        if payload.get("case_id"):
            metadata["case_id"] = payload["case_id"]
        if payload.get("case_name"):
            metadata["case_name"] = payload["case_name"]
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import report manifests, summaries, and chain evidence into JsonDocumentStore."
    )
    parser.add_argument(
        "case",
        nargs="?",
        default="all",
        help="Case directory name or 'all'.",
    )
    parser.add_argument(
        "--include-runs",
        action="store_true",
        help="Also import cases/<case>/runN.json collection intermediates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List documents without writing MongoDB.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = discover_documents(ROOT, args.case, include_runs=args.include_runs)
    if not paths:
        print("No matching report or chain documents found.", file=sys.stderr)
        return 1

    counts = Counter(default_namespace(path.relative_to(ROOT)) for path in paths)
    if args.dry_run:
        for path in paths:
            print(path.relative_to(ROOT).as_posix())
    else:
        store = get_json_store()
        for path in paths:
            payload = read_json(path)
            store.save_path(
                path,
                payload,
                local_backup=False,
                extra_metadata=metadata_for(path, payload),
            )

    count_text = " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    action = "would import" if args.dry_run else "imported"
    print(f"{action} {len(paths)} documents: {count_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
