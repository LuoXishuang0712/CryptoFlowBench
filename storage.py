from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote_plus


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_JSON_COLLECTION = "json_documents"
CHAIN_AGENT_SESSIONS_COLLECTION = "chain_agent_sessions"
CHAIN_AGENT_EVAL_RESULTS_COLLECTION = "chain_agent_eval_results"
CHAIN_AGENT_EVAL_SUMMARIES_COLLECTION = "chain_agent_eval_summaries"


def load_env(path: Path = ENV_PATH) -> None:
    """Load .env values without overwriting real environment variables."""
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
class MongoSettings:
    url: str
    database: str
    username: str = ""
    password: str = ""
    auth_source: str = "admin"

    @classmethod
    def from_env(cls) -> "MongoSettings":
        return cls(
            url=os.environ.get("MONGO_URL", "localhost:27017").strip(),
            database=os.environ.get("MONGO_DB", "onchain_case").strip(),
            username=os.environ.get("MONGO_USER", "").strip(),
            password=os.environ.get("MONGO_PASSWD", "").strip(),
            auth_source=os.environ.get("MONGO_AUTH_SOURCE", "admin").strip() or "admin",
        )

    def uri(self, *, include_database: bool = False) -> str:
        url = self.url
        if url.startswith(("mongodb://", "mongodb+srv://")):
            return url
        auth = ""
        if self.username:
            auth = quote_plus(self.username)
            if self.password:
                auth += f":{quote_plus(self.password)}"
            auth += "@"
        db_part = f"/{self.database}" if include_database else "/"
        query = f"?authSource={quote_plus(self.auth_source)}" if self.username else ""
        return f"mongodb://{auth}{url}{db_part}{query}"


def _require_pymongo() -> Any:
    try:
        from pymongo import MongoClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Mongo storage requires pymongo. Run `uv sync` after installing project dependencies."
        ) from exc
    return MongoClient


def json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mongo_safe(value: Any) -> Any:
    """Convert Python values into Mongo-friendly recursive structures."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        if value < -(2**63) or value > 2**63 - 1:
            return str(value)
        return value
    if isinstance(value, (float, str, datetime)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [mongo_safe(item) for item in value]
    if isinstance(value, tuple):
        return [mongo_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): mongo_safe(item) for key, item in value.items()}
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl_file(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str))
            f.write("\n")


def logical_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def default_namespace(path: Path) -> str:
    parts = path.parts
    if not parts:
        return "documents"
    if path.suffix == ".jsonl":
        return f"{parts[0]}.{path.stem}"
    if len(parts) >= 3:
        return f"{parts[0]}.{path.name.removesuffix(path.suffix)}"
    return parts[0]


class MongoDocumentStore:
    """Small document-store wrapper for direct collection access."""

    def __init__(
        self,
        collection: str,
        *,
        settings: MongoSettings | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or MongoSettings.from_env()
        self.client = client or _require_pymongo()(self.settings.uri())
        self.db = self.client[self.settings.database]
        self.collection = self.db[collection]

    def upsert_one(self, key: Mapping[str, Any], document: Mapping[str, Any]) -> Any:
        now = datetime.now(timezone.utc)
        body = dict(document)
        body["updated_at"] = now
        insert_fields = {k: v for k, v in dict(key).items() if k not in body}
        insert_fields["created_at"] = now
        return self.collection.update_one(dict(key), {"$set": body, "$setOnInsert": insert_fields}, upsert=True)

    def replace_one(self, key: Mapping[str, Any], document: Mapping[str, Any]) -> Any:
        now = datetime.now(timezone.utc)
        body = dict(key) | dict(document)
        body.setdefault("created_at", now)
        body["updated_at"] = now
        return self.collection.replace_one(dict(key), body, upsert=True)

    def find_one(self, key: Mapping[str, Any]) -> dict[str, Any] | None:
        return self.collection.find_one(dict(key), {"_id": False})

    def find(self, query: Mapping[str, Any] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self.collection.find(dict(query or {}), {"_id": False}, **kwargs))

    def delete_one(self, key: Mapping[str, Any]) -> Any:
        return self.collection.delete_one(dict(key))

    def create_index(self, keys: list[tuple[str, int]], **kwargs: Any) -> Any:
        return self.collection.create_index(keys, **kwargs)


class JsonDocumentStore:
    """Dual-write helper: Mongo is primary, local JSON remains the restorable backup."""

    def __init__(
        self,
        *,
        collection: str = DEFAULT_JSON_COLLECTION,
        root: Path = PROJECT_ROOT,
        settings: MongoSettings | None = None,
        client: Any | None = None,
    ) -> None:
        self.root = root
        self.store = MongoDocumentStore(collection, settings=settings, client=client)
        self.store.create_index([("namespace", 1), ("path", 1)], unique=True)

    def save_path(
        self,
        path: Path,
        data: Any,
        *,
        namespace: str | None = None,
        local_backup: bool = True,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        rel_path = logical_path(path, self.root)
        ns = namespace or default_namespace(Path(rel_path))
        doc = mongo_safe({
            "namespace": ns,
            "path": rel_path,
            "payload": data,
            "payload_sha256": json_hash(data),
            "metadata": dict(extra_metadata or {}),
        })
        self.store.replace_one({"namespace": ns, "path": rel_path}, doc)
        if local_backup:
            write_json_file(path, data)

    def load_path(self, path: Path, *, namespace: str | None = None, fallback_local: bool = True) -> Any:
        rel_path = logical_path(path, self.root)
        ns = namespace or default_namespace(Path(rel_path))
        doc = self.store.find_one({"namespace": ns, "path": rel_path})
        if doc is not None:
            return doc.get("payload")
        if fallback_local and path.exists():
            return read_json(path)
        raise FileNotFoundError(f"No Mongo document or local JSON backup found for {rel_path}")

    def import_path(self, path: Path, *, namespace: str | None = None) -> None:
        self.save_path(path, read_json(path), namespace=namespace, local_backup=False)

    def export_path(self, path: Path, *, namespace: str | None = None) -> None:
        write_json_file(path, self.load_path(path, namespace=namespace, fallback_local=False))


def get_json_store(**kwargs: Any) -> JsonDocumentStore:
    return JsonDocumentStore(**kwargs)


def get_mongo_store(collection: str, **kwargs: Any) -> MongoDocumentStore:
    return MongoDocumentStore(collection, **kwargs)


def ensure_chain_agent_indexes() -> None:
    sessions = MongoDocumentStore(CHAIN_AGENT_SESSIONS_COLLECTION)
    sessions.create_index([("session_id", 1)], unique=True)
    sessions.create_index([("run_id", 1), ("task_id", 1)])
    sessions.create_index([("case_id", 1), ("dataset_version", 1)])

    results = MongoDocumentStore(CHAIN_AGENT_EVAL_RESULTS_COLLECTION)
    results.create_index([("run_id", 1), ("task_id", 1)], unique=True)
    results.create_index([("case_id", 1), ("dataset_version", 1)])
    results.create_index([("agent_session_id", 1)])
    results.create_index([("agent_llm", 1), ("judge_llm", 1)])

    summaries = MongoDocumentStore(CHAIN_AGENT_EVAL_SUMMARIES_COLLECTION)
    summaries.create_index([("run_id", 1)], unique=True)
    summaries.create_index([("case_id", 1), ("dataset_version", 1)])
    summaries.create_index([("agent_llm", 1), ("judge_llm", 1)])


def save_chain_agent_session(session: Mapping[str, Any]) -> Any:
    store = MongoDocumentStore(CHAIN_AGENT_SESSIONS_COLLECTION)
    store.create_index([("session_id", 1)], unique=True)
    return store.replace_one({"session_id": session["session_id"]}, mongo_safe(dict(session)))


def save_chain_agent_eval_result(result: Mapping[str, Any]) -> Any:
    store = MongoDocumentStore(CHAIN_AGENT_EVAL_RESULTS_COLLECTION)
    store.create_index([("run_id", 1), ("task_id", 1)], unique=True)
    return store.replace_one(
        {"run_id": result["run_id"], "task_id": result["task_id"]},
        mongo_safe(dict(result)),
    )


def save_chain_agent_eval_summary(summary: Mapping[str, Any]) -> Any:
    store = MongoDocumentStore(CHAIN_AGENT_EVAL_SUMMARIES_COLLECTION)
    store.create_index([("run_id", 1)], unique=True)
    return store.replace_one({"run_id": summary["run_id"]}, mongo_safe(dict(summary)))
