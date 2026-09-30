from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bson import json_util
from pymongo import MongoClient, ReplaceOne

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from storage import MongoSettings  # noqa: E402


COLLECTION_KEYS = {
    "chain_seed_nodes": ("case_id", "dataset_version", "node"),
    "chain_subgraph_edges": ("case_id", "dataset_version", "edge_id"),
    "chain_labels": ("case_id", "dataset_version", "edge_id"),
    "chain_candidate_sets": ("case_id", "dataset_version", "state_id"),
    "chain_subgraph_bundles": ("case_id", "dataset_version"),
    "chain_qa_tasks": ("case_id", "dataset_version", "id"),
}


def read_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise RuntimeError(f"Unsupported publish dataset format: {manifest_path}")
    return manifest


def validate_manifest_files(data_dir: Path, manifest: dict[str, Any]) -> None:
    for item in manifest.get("files") or []:
        relative = str(item.get("path") or "")
        path = data_dir / relative
        if not path.is_file():
            raise RuntimeError(f"Published data file is missing: {path}")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != item.get("sha256"):
            raise RuntimeError(f"SHA-256 mismatch: {path}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            row = json_util.loads(raw)
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_no} is not a JSON object")
            yield row


def materialize_local_artifacts(data_dir: Path, *, force: bool = False) -> int:
    destinations = {
        "case_reports.jsonl": ROOT / "cases",
        "case_summaries.jsonl": ROOT / "summarized",
        "case_graphs.jsonl": ROOT / "summarized",
    }
    count = 0
    for filename, root in destinations.items():
        for row in iter_jsonl(data_dir / filename):
            case_dir = str(row.get("case_dir") or "")
            artifact = str(row.get("artifact") or "")
            if not case_dir or Path(case_dir).name != case_dir:
                raise RuntimeError(f"Unsafe case_dir in {filename}: {case_dir!r}")
            if filename == "case_reports.jsonl":
                valid_artifact = artifact == "case.json" or (
                    artifact.startswith("run")
                    and artifact.endswith(".json")
                    and artifact[3:-5].isdigit()
                )
            else:
                expected = (
                    "summary.json"
                    if filename == "case_summaries.jsonl"
                    else "chain_evidence.json"
                )
                valid_artifact = artifact == expected
            if not valid_artifact:
                raise RuntimeError(f"Unsafe artifact in {filename}: {artifact!r}")
            target = root / case_dir / artifact
            if target.exists() and not force:
                raise RuntimeError(
                    f"Local artifact already exists: {target}; pass --force-local to replace"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(row["payload"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            count += 1
    return count


def import_dataset(
    data_dir: Path,
    *,
    apply: bool = False,
    materialize_local: bool = False,
    force_local: bool = False,
) -> dict[str, Any]:
    manifest = read_manifest(data_dir)
    validate_manifest_files(data_dir, manifest)
    expected = manifest.get("mongo_jsonl_counts") or {}
    settings = MongoSettings.from_env()
    summary = {
        "dataset_version": manifest.get("dataset_version"),
        "database": settings.database,
        "apply": apply,
        "collections": expected,
    }
    if not apply:
        local_count = (
            materialize_local_artifacts(data_dir, force=force_local)
            if materialize_local
            else 0
        )
        return summary | {"materialized_local_artifacts": local_count}

    client = MongoClient(settings.uri())
    try:
        db = client[settings.database]
        actual: dict[str, int] = {}
        for collection_name, keys in COLLECTION_KEYS.items():
            path = data_dir / "mongo" / f"{collection_name}.jsonl"
            operations = []
            count = 0
            for row in iter_jsonl(path):
                if row.get("dataset_version") != manifest.get("dataset_version"):
                    raise RuntimeError(
                        f"Unexpected dataset version in {path}: "
                        f"{row.get('dataset_version')!r}"
                    )
                key = {name: row[name] for name in keys}
                operations.append(ReplaceOne(key, row, upsert=True))
                count += 1
                if len(operations) >= 1000:
                    db[collection_name].bulk_write(operations, ordered=False)
                    operations.clear()
            if operations:
                db[collection_name].bulk_write(operations, ordered=False)
            if count != int(expected.get(collection_name, -1)):
                raise RuntimeError(
                    f"Row count mismatch for {collection_name}: {count} != "
                    f"{expected.get(collection_name)}"
                )
            actual[collection_name] = count
        local_count = (
            materialize_local_artifacts(data_dir, force=force_local)
            if materialize_local
            else 0
        )
        return summary | {"imported": actual, "materialized_local_artifacts": local_count}
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upsert the published latest-version chain-QA JSONL files into MongoDB."
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform idempotent upserts; without this flag only validate the manifest.",
    )
    parser.add_argument(
        "--materialize-local",
        action="store_true",
        help="Also restore cases/ and summarized/ files from the three local JSONL files.",
    )
    parser.add_argument(
        "--force-local",
        action="store_true",
        help="Allow --materialize-local to replace existing local JSON artifacts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.force_local and not args.materialize_local:
        raise SystemExit("--force-local requires --materialize-local")
    print(
        json.dumps(
            import_dataset(
                args.data_dir,
                apply=args.apply,
                materialize_local=args.materialize_local,
                force_local=args.force_local,
            ),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
