from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(value: str) -> str:
    component = _SAFE_COMPONENT.sub("_", value.strip()).strip("._")
    if not component or component in {".", ".."}:
        raise ValueError(f"unsafe empty checkpoint path component: {value!r}")
    return component


def _atomic_replace(source: Path, target: Path, *, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.025 * (2**attempt))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def persist_detector_checkpoint(
    algorithm: Any,
    *,
    root: Path,
    case_id: str,
    role: str,
    dataset_version: str,
    training_cases: list[str],
    training_summary: dict[str, Any] | None,
    case_disjoint: bool,
) -> dict[str, Any]:
    """Atomically persist one loadable comparison-method checkpoint and manifest."""
    if role not in {"edge_classifier", "node_detector"}:
        raise ValueError(f"unsupported checkpoint role: {role!r}")
    name = str(getattr(algorithm, "name", type(algorithm).__name__))
    directory = (
        root
        / _safe_component(case_id)
        / f"{_safe_component(role)}-{_safe_component(name)}"
    )
    model_payload = pickle.dumps(algorithm, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(model_payload).hexdigest()
    model_path = directory / "model.pkl"
    manifest_path = directory / "manifest.json"
    _atomic_write(model_path, model_payload)
    manifest = {
        "schema_version": 1,
        "case_id": case_id,
        "role": role,
        "algorithm": {
            "name": name,
            "version": str(getattr(algorithm, "version", "")),
            "class": f"{type(algorithm).__module__}.{type(algorithm).__qualname__}",
        },
        "dataset_version": dataset_version,
        "training_cases": list(training_cases),
        "case_disjoint": bool(case_disjoint),
        "training_summary": dict(training_summary or {}),
        "model_file": model_path.name,
        "model_sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n").encode(
            "utf-8"
        ),
    )
    return {
        "directory": str(directory),
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "model_sha256": digest,
    }


def load_detector_checkpoint(directory: Path) -> tuple[Any, dict[str, Any]]:
    """Load a local trusted checkpoint after verifying its manifest digest."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    payload = (directory / str(manifest["model_file"])).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    expected = str(manifest["model_sha256"])
    if actual != expected:
        raise ValueError(
            f"checkpoint sha256 mismatch for {directory}: expected={expected}, actual={actual}"
        )
    return pickle.loads(payload), manifest
