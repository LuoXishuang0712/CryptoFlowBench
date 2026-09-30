from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from case_selection import discover_selected_cases

PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARIZED_DIR = PROJECT_ROOT / "summarized"
QA_DIR = PROJECT_ROOT / "qa"
ENV_FILE = PROJECT_ROOT / ".env"

QUESTION_TYPES = {"<s,*,*>", "<s,p,*>", "<s,*,o>", "<s,p,o>"}
DEFAULT_MAX_LLM_EXTRA = 12
GRADE_ORDER = {"A": 4, "B": 3, "C": 2, "D": 1, "U": 0}
MAX_LLM_REPAIR_ROUNDS = 1
QA_DOMAINS = {"general", "chain"}
CHAIN_REASONS = {
    "edge_profile_lookup",
    "edge_metadata_lookup",
    "transaction_profile_lookup",
    "transaction_metadata_lookup",
    "relation_inference",
    "constraint_check",
    "constraint_lookup",
    "source_reliability_check",
    "agent_action_policy",
    "ground_truth_usability_check",
    "boundary_decision",
    "infrastructure_boundary_check",
    "chain_report_alignment",
    "false_positive_check",
    "multi_hop_stage_reasoning",
    "direct_link_verification",
    "negative_link_verification",
    "single_hop_prediction",
    "path_completion",
    "multi_hop_path_prediction",
}

QUESTION_TYPE_RULES = """question_type is a KG triple query pattern, not a reasoning label.
It must be exactly one of these four strings:
- "<s,*,*>": subject is known, predicate="*", object="*".
- "<s,p,*>": subject and predicate are known, object="*".
- "<s,*,o>": subject and object are known, predicate="*".
- "<s,p,o>": subject, predicate, and object are all known, usually for true/false verification.
Never put reasoning labels such as multi_hop_stage_reasoning, boundary_decision, source_reliability_check,
agent_action_policy, or ground_truth_usability_check in question_type. Put those in reasoning_type only."""


@dataclass(frozen=True)
class CaseInput:
    case_dir: str
    summary: dict[str, Any]
    evidence: dict[str, Any]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split("#", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def normalize_case_name(
    case_dir: str, summary: dict[str, Any], evidence: dict[str, Any]
) -> str:
    return str(evidence.get("case_name") or summary.get("case_name") or case_dir)


def short_id(value: Any) -> str:
    text = str(value)
    if re.fullmatch(r"0x[a-fA-F0-9]{40}", text):
        return f"{text[:8]}...{text[-6:]}"
    if re.fullmatch(r"0x[a-fA-F0-9]{64}", text):
        return f"{text[:10]}...{text[-8:]}"
    return text


def compact_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(compact_value(item) for item in value)
    if isinstance(value, dict):
        pairs = []
        for key, item in value.items():
            if item not in (None, "", [], {}):
                pairs.append(f"{key}: {compact_value(item)}")
        return "; ".join(pairs)
    return str(value)


def infer_answer_format(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def evidence_ids(obj: dict[str, Any]) -> list[str]:
    evidence = obj.get("evidence") or obj.get("sources") or []
    if isinstance(evidence, str):
        return [evidence]
    if isinstance(evidence, list):
        return [str(item) for item in evidence]
    return []


def source_map(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for source in evidence.get("sources") or []:
        if isinstance(source, dict) and source.get("id"):
            out[str(source["id"])] = source
    return out


def supporting_source_objects(
    ids: list[str], sources: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for source_id in ids:
        source = sources.get(source_id)
        if not source:
            rows.append({"id": source_id})
            continue
        rows.append(
            {
                "id": source_id,
                "entity": source.get("entity"),
                "grade": source.get("grade"),
                "url": source.get("url"),
            }
        )
    return rows


def source_grade(source: dict[str, Any] | None) -> str:
    if not source:
        return "U"
    return str(source.get("grade") or source.get("source_grade") or "U")


def highest_source_grade(ids: list[str], sources: dict[str, dict[str, Any]]) -> str:
    grades = [source_grade(sources.get(source_id)) for source_id in ids]
    if not grades:
        return "U"
    return max(grades, key=lambda grade: GRADE_ORDER.get(grade, -1))


def source_grade_summary(ids: list[str], sources: dict[str, dict[str, Any]]) -> str:
    if not ids:
        return "无显式 source id"
    parts = []
    for source_id in ids:
        source = sources.get(source_id)
        if source:
            parts.append(
                f"{source_id}({source.get('entity')}, grade={source_grade(source)})"
            )
        else:
            parts.append(f"{source_id}(grade=U)")
    return ", ".join(parts)


def triple(subject: str, predicate: str, obj: Any, origin: str) -> dict[str, Any]:
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "origin": origin,
    }


def clean_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "qa"


def add_qa(
    rows: list[dict[str, Any]],
    *,
    case_input: CaseInput,
    question_type: str,
    subject: str,
    predicate: str,
    obj: Any,
    question: str,
    answer: str,
    supporting_triples: list[dict[str, Any]],
    reasoning_type: str,
    difficulty: str,
    evidence_ids_: list[str] | None = None,
    sources: dict[str, dict[str, Any]] | None = None,
    generator: str = "deterministic_template",
    answer_value: Any = None,
    answer_format: str | None = None,
    answer_aliases: list[str] | None = None,
    is_correct: bool | None = None,
    correction: dict[str, Any] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"unsupported question_type: {question_type}")
    if not question.strip() or not answer.strip():
        return

    evidence_ids_ = evidence_ids_ or []
    source_objects = supporting_source_objects(evidence_ids_, sources or {})
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    structured_answer = answer if answer_value is None else answer_value
    row = {
        "id": "",
        "case_dir": case_input.case_dir,
        "case_id": case_input.evidence.get("case_id"),
        "case_name": case_name,
        "question_type": question_type,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "question": question.strip(),
        "answer": answer.strip(),
        "answer_value": structured_answer,
        "answer_aliases": answer_aliases or [],
        "answer_format": answer_format or infer_answer_format(structured_answer),
        "supporting_triples": supporting_triples,
        "supporting_sources": source_objects,
        "reasoning_type": reasoning_type,
        "difficulty": difficulty,
        "generator": generator,
    }
    if is_correct is not None:
        row["is_correct"] = is_correct
    if correction is not None:
        row["correction"] = correction
    if extra_fields:
        row.update(extra_fields)
    row["qa_domain"] = classify_qa_domain(row)
    key = f"{case_input.case_dir}:{len(rows) + 1:04d}:{clean_slug(subject)[:40]}:{clean_slug(predicate)[:30]}"
    row["id"] = key
    rows.append(row)


def entity_label(entity: dict[str, Any]) -> str:
    return str(
        entity.get("id")
        or entity.get("address")
        or entity.get("entity")
        or "entity:unknown"
    )


def edge_label(edge: dict[str, Any]) -> str:
    return str(edge.get("id") or f"{edge.get('src')}->{edge.get('dst')}")


def link_semantics_for_edge(edge: dict[str, Any]) -> str:
    if is_chain_verified(edge.get("verification_status")):
        return "chain_direct"
    return "reported_direct"


def classify_qa_domain(row: dict[str, Any]) -> str:
    if row.get("qa_domain") in QA_DOMAINS:
        return str(row["qa_domain"])
    if row.get("link_semantics"):
        return "chain"
    subject = str(row.get("subject") or "")
    if subject.startswith("edge:") or re.fullmatch(r"0x[a-fA-F0-9]{64}", subject):
        return "chain"
    if row.get("supporting_edge_id"):
        return "chain"
    if row.get("reasoning_type") in CHAIN_REASONS:
        triples = row.get("supporting_triples") or []
        if row.get("reasoning_type") not in {
            "source_reliability_check",
            "false_positive_check",
        }:
            return "chain"
        if any(
            str(triple_obj.get("origin", "")).startswith("chain_evidence.edges")
            or str(triple_obj.get("origin", "")).startswith(
                "chain_evidence.transaction_evidence"
            )
            for triple_obj in triples
            if isinstance(triple_obj, dict)
        ):
            return "chain"
    return "general"


def edge_relation_answer(edge: dict[str, Any]) -> str:
    bits = []
    if edge.get("pattern"):
        bits.append(str(edge["pattern"]))
    if edge.get("stage"):
        bits.append(f"stage={edge['stage']}")
    if edge.get("amount"):
        bits.append(f"amount={edge['amount']}")
    if edge.get("asset"):
        bits.append(f"asset={edge['asset']}")
    if edge.get("asset_in") or edge.get("asset_out"):
        bits.append(f"asset_flow={edge.get('asset_in')} -> {edge.get('asset_out')}")
    if edge.get("verification_status"):
        bits.append(f"verification_status={edge['verification_status']}")
    return "; ".join(bits) if bits else "存在报告链路关系。"


def edge_endpoint_nodes(edges: list[dict[str, Any]]) -> list[str]:
    nodes: list[str] = []
    for edge in edges:
        for key in ("src", "dst"):
            value = edge.get(key)
            if value not in (None, "", [], {}) and str(value) not in nodes:
                nodes.append(str(value))
    return nodes


def entity_nodes(evidence: dict[str, Any]) -> list[str]:
    nodes: list[str] = []
    for entity in evidence.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        value = entity_label(entity)
        if value not in nodes:
            nodes.append(value)
    return nodes


def compact_candidates(
    candidates: list[str], answer_values: list[str], limit: int = 6
) -> list[str]:
    out: list[str] = []
    for value in answer_values + candidates:
        if value and value not in out:
            out.append(value)
        if len(out) >= limit:
            break
    return out


def format_candidate_set(candidates: list[str]) -> str:
    labels = []
    for idx, candidate in enumerate(candidates):
        letter = chr(ord("A") + idx)
        labels.append(f"{letter}. {short_id(candidate)}")
    return "; ".join(labels)


def edge_support_triples(edge: dict[str, Any]) -> list[dict[str, Any]]:
    subject = edge_label(edge)
    support = []
    for key in [
        "src",
        "dst",
        "stage",
        "pattern",
        "amount",
        "asset",
        "asset_in",
        "asset_out",
        "route",
        "verification_status",
        "confidence",
    ]:
        if edge.get(key) not in (None, "", [], {}):
            support.append(triple(subject, key, edge[key], "chain_evidence.edges"))
    return support


def is_chain_verified(status: Any) -> bool:
    return str(status or "").lower() == "chain_verified"


def recommended_action(edge: dict[str, Any]) -> tuple[str, str]:
    pattern = str(edge.get("pattern") or "").lower()
    stage = str(edge.get("stage") or "").lower()
    dst = str(edge.get("dst") or "").lower()

    if "mixer" in pattern or "tornado" in dst or "mix" in stage:
        return (
            "stop_at_mixer_boundary",
            "把 mixer deposit 作为边界点；除非有独立 linkage evidence，不要把 withdrawal 直接接成同一攻击者路径。",
        )
    if (
        "cex" in pattern
        or "exchange" in dst
        or stage in {"cashout", "cex", "layering"}
        and "cex" in dst
    ):
        return (
            "stop_at_service_terminal",
            "把 CEX/service endpoint 作为终端或待外部证据确认的节点；不要推断交易所账户所有者或把 CEX 标为攻击者钱包。",
        )
    if (
        "swap" in pattern
        or "conversion" in stage
        or edge.get("asset_in")
        or edge.get("asset_out")
    ):
        return (
            "inspect_swap_logs",
            "继续解析 DEX/router、Transfer logs 和真实 recipient；不要把 router 简化为最终资金持有人。",
        )
    if "fraudulent" in pattern or "theft" in stage:
        return (
            "verify_tx_hash_and_mark_theft_edge",
            "标记为 theft/exploit 边，并优先补齐 tx_hash 或本地索引匹配后再作为 tx-level ground truth。",
        )
    if (
        "bridge" in pattern
        or "chain_hop" in str(edge.get("id") or "")
        or "chain_hopping" in stage
    ):
        return (
            "inspect_bridge_route",
            "检查 bridge event、跨链 route 和目标链接收方；跨链跳转需要保留 route 而不是压成单跳地址转账。",
        )
    if "intermediary" in pattern or "layering" in stage:
        return (
            "follow_if_source_supported_and_verify_tx",
            "可以作为候选下一跳继续检查，但需要 source support 或 tx/index 证据，不能只因地址被提及就合并为攻击者路径。",
        )
    return (
        "inspect_edge_evidence",
        "先检查该边的 src/dst、stage、pattern、evidence 和 verification_status，再决定是否继续追踪。",
    )


def ground_truth_usability(edge: dict[str, Any]) -> tuple[bool, str]:
    status = edge.get("verification_status")
    pattern = str(edge.get("pattern") or "").lower()
    dst = str(edge.get("dst") or "").lower()
    if not is_chain_verified(status):
        return (
            False,
            f"该边 verification_status={status}，不是 chain_verified，不能直接作为 tx-level next-hop ground truth。",
        )
    if "mixer" in pattern or "tornado" in dst:
        return (
            False,
            "即使有 deposit 证据，mixer withdrawal linkage 仍需要独立证据，不能直接跨 mixer 连接下一跳。",
        )
    return (
        True,
        "该边已标记为 chain_verified，且没有明显 mixer/CEX 边界限制，可作为候选链上 ground truth。",
    )


def case_profile_answer(evidence: dict[str, Any]) -> str:
    parts = [
        f"incident_type={evidence.get('incident_type')}",
        f"chains={compact_value(evidence.get('chains'))}",
        f"protocol_type={evidence.get('protocol_type')}",
        f"loss={compact_value(evidence.get('loss'))}",
    ]
    root_cause = evidence.get("root_cause") or {}
    if isinstance(root_cause, dict):
        parts.append(
            f"root_cause={root_cause.get('category') or root_cause.get('summary')}"
        )
    return "; ".join(part for part in parts if part and not part.endswith("=None"))


def generate_case_level_qa(case_input: CaseInput, rows: list[dict[str, Any]]) -> None:
    evidence = case_input.evidence
    case_name = normalize_case_name(case_input.case_dir, case_input.summary, evidence)
    subject = f"case:{evidence.get('case_id') or clean_slug(case_name)}"

    add_qa(
        rows,
        case_input=case_input,
        question_type="<s,*,*>",
        subject=subject,
        predicate="*",
        obj="*",
        question=f"{case_name} 有哪些核心链上事件属性？",
        answer=case_profile_answer(evidence),
        supporting_triples=[
            triple(
                subject,
                "incident_type",
                evidence.get("incident_type"),
                "chain_evidence",
            ),
            triple(subject, "chains", evidence.get("chains"), "chain_evidence"),
            triple(
                subject,
                "protocol_type",
                evidence.get("protocol_type"),
                "chain_evidence",
            ),
            triple(subject, "loss", evidence.get("loss"), "chain_evidence"),
            triple(subject, "root_cause", evidence.get("root_cause"), "chain_evidence"),
        ],
        reasoning_type="case_profile_lookup",
        difficulty="easy",
    )

    simple_fields = [
        ("incident_type", f"{case_name} 的事件类型是什么？"),
        ("chains", f"{case_name} 涉及哪些链？"),
        ("protocol_type", f"{case_name} 的协议类型是什么？"),
        ("loss", f"{case_name} 的损失规模是什么？"),
        ("root_cause", f"{case_name} 的根因是什么？"),
    ]
    for field, question in simple_fields:
        value = evidence.get(field)
        if value in (None, "", [], {}):
            continue
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,p,*>",
            subject=subject,
            predicate=field,
            obj="*",
            question=question,
            answer=compact_value(value),
            supporting_triples=[triple(subject, field, value, "chain_evidence")],
            reasoning_type="case_metadata_lookup",
            difficulty="easy",
        )


def generate_entity_qa(
    case_input: CaseInput,
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    max_entities: int,
) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    entities = [
        e for e in case_input.evidence.get("entities") or [] if isinstance(e, dict)
    ]
    rank = {
        "attacker_initial": 0,
        "victim": 1,
        "attacker_ethereum_wallet": 2,
        "centralized_exchange": 3,
    }
    entities.sort(
        key=lambda e: (rank.get(str(e.get("role")), 9), str(e.get("id") or ""))
    )

    for entity in entities[:max_entities]:
        subject = entity_label(entity)
        display = short_id(subject)
        ev_ids = evidence_ids(entity)
        support = []
        for key in ["role", "type", "entity", "chain", "confidence"]:
            if entity.get(key) not in (None, "", [], {}):
                support.append(
                    triple(subject, key, entity[key], "chain_evidence.entities")
                )

        if support:
            answer_parts = []
            for key in ["type", "role", "entity", "chain", "confidence"]:
                if entity.get(key) not in (None, "", [], {}):
                    answer_parts.append(f"{key}={compact_value(entity[key])}")
            if ev_ids:
                answer_parts.append(f"evidence={', '.join(ev_ids)}")
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,*,*>",
                subject=subject,
                predicate="*",
                obj="*",
                question=f"在 {case_name} 中，{display} 有哪些已知实体属性？",
                answer="; ".join(answer_parts),
                supporting_triples=support,
                reasoning_type="entity_profile_lookup",
                difficulty="easy",
                evidence_ids_=ev_ids,
                sources=sources,
            )

        for predicate, zh_name in [
            ("role", "角色"),
            ("entity", "归因或实体名称"),
            ("chain", "所在链"),
            ("confidence", "置信度"),
        ]:
            if entity.get(predicate) in (None, "", [], {}):
                continue
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,p,*>",
                subject=subject,
                predicate=predicate,
                obj="*",
                question=f"在 {case_name} 中，{display} 的{zh_name}是什么？",
                answer=compact_value(entity[predicate]),
                supporting_triples=[
                    triple(
                        subject, predicate, entity[predicate], "chain_evidence.entities"
                    )
                ],
                reasoning_type="entity_metadata_lookup",
                difficulty="easy",
                evidence_ids_=ev_ids,
                sources=sources,
            )

        if entity.get("role"):
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,p,o>",
                subject=subject,
                predicate="role",
                obj=entity["role"],
                question=f"判断三元组 ⟨{display}, role, {entity['role']}⟩ 在 {case_name} 中是否正确。",
                answer="正确。",
                supporting_triples=[
                    triple(subject, "role", entity["role"], "chain_evidence.entities")
                ],
                reasoning_type="triple_verification",
                difficulty="easy",
                evidence_ids_=ev_ids,
                sources=sources,
            )

        if entity.get("entity") and entity.get("role"):
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,*,o>",
                subject=subject,
                predicate="*",
                obj=entity["entity"],
                question=f"在 {case_name} 中，{display} 与 {short_id(entity['entity'])} 之间是什么关系？",
                answer=f"该实体的 role={entity['role']}，entity/attribution={entity['entity']}。",
                supporting_triples=[
                    triple(subject, "role", entity["role"], "chain_evidence.entities"),
                    triple(
                        subject, "entity", entity["entity"], "chain_evidence.entities"
                    ),
                ],
                reasoning_type="relation_inference",
                difficulty="medium",
                evidence_ids_=ev_ids,
                sources=sources,
            )


def generate_edge_qa(
    case_input: CaseInput,
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    edges = [e for e in case_input.evidence.get("edges") or [] if isinstance(e, dict)]

    for edge in edges:
        subject = edge_label(edge)
        ev_ids = evidence_ids(edge)
        support = edge_support_triples(edge)

        if support:
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,*,*>",
                subject=subject,
                predicate="*",
                obj="*",
                question=f"{case_name} 中 {subject} 这条链路边包含哪些信息？",
                answer=edge_relation_answer(edge),
                supporting_triples=support,
                reasoning_type="edge_profile_lookup",
                difficulty="easy",
                evidence_ids_=ev_ids,
                sources=sources,
                answer_value={item["predicate"]: item["object"] for item in support},
            )

        for predicate, zh_name in [
            ("stage", "链路阶段"),
            ("pattern", "资金流模式"),
            ("verification_status", "链上验证状态"),
            ("evidence", "证据来源"),
            ("confidence", "置信度"),
        ]:
            if predicate == "evidence":
                value = ev_ids
            else:
                value = edge.get(predicate)
            if value in (None, "", [], {}):
                continue
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,p,*>",
                subject=subject,
                predicate=predicate,
                obj="*",
                question=f"{case_name} 中 {subject} 的{zh_name}是什么？",
                answer=compact_value(value),
                supporting_triples=[
                    triple(subject, predicate, value, "chain_evidence.edges")
                ],
                reasoning_type="edge_metadata_lookup",
                difficulty="easy",
                evidence_ids_=ev_ids,
                sources=sources,
                answer_value=value,
            )

        if edge.get("src") and edge.get("dst"):
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,*,o>",
                subject=str(edge["src"]),
                predicate="*",
                obj=edge["dst"],
                question=f"在 {case_name} 中，通过 {subject} 表示的 {short_id(edge['src'])} 与 {short_id(edge['dst'])} 之间是什么链路关系？",
                answer=edge_relation_answer(edge),
                supporting_triples=[
                    triple(subject, "src", edge["src"], "chain_evidence.edges"),
                    triple(subject, "dst", edge["dst"], "chain_evidence.edges"),
                    triple(subject, "stage", edge.get("stage"), "chain_evidence.edges"),
                    triple(
                        subject, "pattern", edge.get("pattern"), "chain_evidence.edges"
                    ),
                    triple(
                        subject,
                        "verification_status",
                        edge.get("verification_status"),
                        "chain_evidence.edges",
                    ),
                ],
                reasoning_type="relation_inference",
                difficulty="medium",
                evidence_ids_=ev_ids,
                sources=sources,
                answer_value={
                    "edge_id": subject,
                    "stage": edge.get("stage"),
                    "pattern": edge.get("pattern"),
                    "verification_status": edge.get("verification_status"),
                },
                extra_fields={"supporting_edge_id": subject},
            )

        for predicate in ["stage", "pattern", "verification_status"]:
            if edge.get(predicate) in (None, "", [], {}):
                continue
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,p,o>",
                subject=subject,
                predicate=predicate,
                obj=edge[predicate],
                question=f"判断三元组 ⟨{subject}, {predicate}, {edge[predicate]}⟩ 在 {case_name} 中是否正确。",
                answer="正确。",
                supporting_triples=[
                    triple(subject, predicate, edge[predicate], "chain_evidence.edges")
                ],
                reasoning_type="triple_verification",
                difficulty="easy",
                evidence_ids_=ev_ids,
                sources=sources,
                answer_value=True,
                is_correct=True,
            )

        verification_status = edge.get("verification_status")
        if verification_status and verification_status != "chain_verified":
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,p,o>",
                subject=subject,
                predicate="verification_status",
                obj="chain_verified",
                question=f"判断三元组 ⟨{subject}, verification_status, chain_verified⟩ 在 {case_name} 中是否正确。",
                answer=f"错误。当前 verification_status 是 {verification_status}，不能把报告事实描述成已链上验证事实。",
                supporting_triples=[
                    triple(
                        subject,
                        "verification_status",
                        verification_status,
                        "chain_evidence.edges",
                    )
                ],
                reasoning_type="constraint_check",
                difficulty="medium",
                evidence_ids_=ev_ids,
                sources=sources,
                answer_value=False,
                is_correct=False,
                correction={
                    "predicate": "verification_status",
                    "object": verification_status,
                },
            )


def generate_source_reliability_qa(
    case_input: CaseInput,
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    objects: list[tuple[str, str, list[str], list[dict[str, Any]]]] = []
    for edge in case_input.evidence.get("edges") or []:
        if isinstance(edge, dict):
            subject = edge_label(edge)
            objects.append(
                (
                    subject,
                    "chain_evidence.edges",
                    evidence_ids(edge),
                    edge_support_triples(edge),
                )
            )
    for entity in case_input.evidence.get("entities") or []:
        if isinstance(entity, dict):
            subject = entity_label(entity)
            support = [
                triple(subject, key, entity[key], "chain_evidence.entities")
                for key in ["role", "type", "entity", "confidence"]
                if entity.get(key) not in (None, "", [], {})
            ]
            objects.append(
                (subject, "chain_evidence.entities", evidence_ids(entity), support)
            )

    for subject, origin, ev_ids, base_support in objects:
        if not ev_ids:
            continue
        grade = highest_source_grade(ev_ids, sources)
        source_support = [
            triple(
                source_id,
                "grade",
                source_grade(sources.get(source_id)),
                "chain_evidence.sources",
            )
            for source_id in ev_ids
        ]
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,p,*>",
            subject=subject,
            predicate="highest_source_grade",
            obj="*",
            question=f"{case_name} 中 {short_id(subject)} 的支持证据里最高 source grade 是什么？",
            answer=f"{grade}。支持来源包括：{source_grade_summary(ev_ids, sources)}。",
            supporting_triples=base_support[:4] + source_support,
            reasoning_type="source_reliability_check",
            difficulty="medium",
            evidence_ids_=ev_ids,
            sources=sources,
            answer_value=grade,
            extra_fields={"source_ids": ev_ids, "source_origin": origin},
        )


def generate_agent_action_qa(
    case_input: CaseInput,
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    constraints = [
        str(item) for item in case_input.evidence.get("negative_constraints") or []
    ]
    constraint_support = [
        triple(
            f"constraint:{constraint}",
            "policy",
            constraint,
            "chain_evidence.negative_constraints",
        )
        for constraint in constraints
    ]

    for edge in case_input.evidence.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        subject = edge_label(edge)
        ev_ids = evidence_ids(edge)
        support = edge_support_triples(edge)
        action, explanation = recommended_action(edge)
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,p,*>",
            subject=subject,
            predicate="recommended_agent_action",
            obj="*",
            question=f"Agent 在 {case_name} 中遇到 {subject} 这条链路边时，下一步更应该执行什么动作？",
            answer=f"{action}。{explanation}",
            supporting_triples=support + constraint_support[:3],
            reasoning_type="agent_action_policy",
            difficulty="hard",
            evidence_ids_=ev_ids,
            sources=sources,
            answer_value={"action": action, "explanation": explanation},
        )

        usable, reason = ground_truth_usability(edge)
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,p,*>",
            subject=subject,
            predicate="usable_as_tx_level_next_hop_ground_truth",
            obj="*",
            question=f"{case_name} 中 {subject} 是否可以直接作为 tx-level next-hop ground truth？",
            answer=("可以。 " if usable else "不可以。 ") + reason,
            supporting_triples=support + constraint_support[:3],
            reasoning_type="ground_truth_usability_check",
            difficulty="hard",
            evidence_ids_=ev_ids,
            sources=sources,
            answer_value=usable,
        )


def generate_constraint_application_qa(
    case_input: CaseInput,
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    constraints = [
        str(item) for item in case_input.evidence.get("negative_constraints") or []
    ]
    constraint_triples = {
        constraint: triple(
            f"constraint:{constraint}",
            "policy",
            constraint,
            "chain_evidence.negative_constraints",
        )
        for constraint in constraints
    }
    has_cex_constraint = (
        "do_not_infer_cex_account_owner_without_external_evidence" in constraints
    )
    has_mixer_constraint = (
        "do_not_label_all_tornado_withdrawals_as_same_actor" in constraints
    )
    has_verified_constraint = (
        "do_not_treat_reported_edges_as_chain_verified_without_tx_hash_or_index_match"
        in constraints
    )

    for edge in case_input.evidence.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        subject = edge_label(edge)
        pattern = str(edge.get("pattern") or "").lower()
        dst = str(edge.get("dst") or "").lower()
        support = edge_support_triples(edge)
        ev_ids = evidence_ids(edge)

        if has_mixer_constraint and ("mixer" in pattern or "tornado" in dst):
            constraint = "do_not_label_all_tornado_withdrawals_as_same_actor"
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,p,*>",
                subject=subject,
                predicate="boundary_decision",
                obj="*",
                question=f"Agent 在 {case_name} 中追踪到 {subject} 后，能否直接把后续 mixer withdrawal 标成同一攻击者路径？",
                answer=f"不能。{subject} 是 mixer 边界，约束 {constraint} 要求除非有独立 linkage evidence，否则不要把所有 withdrawal 标成同一 actor。",
                supporting_triples=support + [constraint_triples[constraint]],
                reasoning_type="boundary_decision",
                difficulty="hard",
                evidence_ids_=ev_ids,
                sources=sources,
                answer_value=False,
            )

        if has_cex_constraint and ("cex" in pattern or "exchange" in dst):
            constraint = "do_not_infer_cex_account_owner_without_external_evidence"
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,p,*>",
                subject=subject,
                predicate="boundary_decision",
                obj="*",
                question=f"如果 {case_name} 的资金流进入 {short_id(edge.get('dst'))}，Agent 是否应把该 CEX endpoint 标为 attacker_controlled_wallet？",
                answer=f"不应该。{constraint} 要求没有外部证据时不要推断 CEX account owner；该边最多表示 reported CEX deposit/service terminal。",
                supporting_triples=support + [constraint_triples[constraint]],
                reasoning_type="infrastructure_boundary_check",
                difficulty="hard",
                evidence_ids_=ev_ids,
                sources=sources,
                answer_value=False,
            )

        if (
            has_verified_constraint
            and edge.get("verification_status")
            and not is_chain_verified(edge.get("verification_status"))
        ):
            constraint = "do_not_treat_reported_edges_as_chain_verified_without_tx_hash_or_index_match"
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,p,*>",
                subject=subject,
                predicate="chain_report_alignment",
                obj="*",
                question=f"{case_name} 中 {subject} 能否被表述为已经由链上索引验证的事实？",
                answer=f"不能。该边当前 verification_status={edge.get('verification_status')}，且约束 {constraint} 要求没有 tx_hash 或 index match 时不要升级为 chain_verified。",
                supporting_triples=support + [constraint_triples[constraint]],
                reasoning_type="chain_report_alignment",
                difficulty="hard",
                evidence_ids_=ev_ids,
                sources=sources,
                answer_value=False,
            )


def choose_corruption(actual: Any, candidates: list[str]) -> str | None:
    actual_text = str(actual or "")
    for candidate in candidates:
        if candidate and candidate != actual_text:
            return candidate
    return None


def generate_negative_triple_qa(
    case_input: CaseInput,
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    stage_candidates = [
        "theft",
        "conversion",
        "layering",
        "mixing",
        "chain_hopping",
        "cashout",
    ]
    pattern_candidates = [
        "fraudulent_bridge_withdrawal",
        "stablecoin_to_native_asset_swap",
        "cex_deposit",
        "mixer_deposit",
        "bridge_swap_bridge",
        "reported_intermediary_transfer",
    ]
    role_candidates = [
        "attacker_initial",
        "attacker_controlled_wallet",
        "centralized_exchange",
        "mixer_contract_set",
        "victim",
    ]

    for edge in case_input.evidence.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        subject = edge_label(edge)
        ev_ids = evidence_ids(edge)
        support = edge_support_triples(edge)
        for predicate, candidates in [
            ("stage", stage_candidates),
            ("pattern", pattern_candidates),
        ]:
            actual = edge.get(predicate)
            wrong = choose_corruption(actual, candidates)
            if actual in (None, "", [], {}) or not wrong:
                continue
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,p,o>",
                subject=subject,
                predicate=predicate,
                obj=wrong,
                question=f"判断三元组 ⟨{subject}, {predicate}, {wrong}⟩ 在 {case_name} 中是否正确。",
                answer=f"错误。{subject} 的 {predicate} 是 {actual}，不是 {wrong}。",
                supporting_triples=[
                    triple(subject, predicate, actual, "chain_evidence.edges")
                ],
                reasoning_type="false_positive_check",
                difficulty="medium",
                evidence_ids_=ev_ids,
                sources=sources,
                answer_value=False,
                is_correct=False,
                correction={"predicate": predicate, "object": actual},
            )

    for entity in case_input.evidence.get("entities") or []:
        if not isinstance(entity, dict) or entity.get("role") in (None, "", [], {}):
            continue
        subject = entity_label(entity)
        actual = entity.get("role")
        wrong = choose_corruption(actual, role_candidates)
        if not wrong:
            continue
        role_text = str(actual).lower()
        if (
            "exchange" in role_text
            or "mixer" in role_text
            or "router" in role_text
            or "victim" in role_text
        ):
            difficulty = "medium"
        else:
            difficulty = "hard"
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,p,o>",
            subject=subject,
            predicate="role",
            obj=wrong,
            question=f"判断三元组 ⟨{short_id(subject)}, role, {wrong}⟩ 在 {case_name} 中是否正确。",
            answer=f"错误。{short_id(subject)} 的已知 role 是 {actual}，不是 {wrong}。",
            supporting_triples=[
                triple(subject, "role", actual, "chain_evidence.entities")
            ],
            reasoning_type="false_positive_check",
            difficulty=difficulty,
            evidence_ids_=evidence_ids(entity),
            sources=sources,
            answer_value=False,
            is_correct=False,
            correction={"predicate": "role", "object": actual},
        )


def generate_direct_link_verification_qa(
    case_input: CaseInput,
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    edges = [
        edge
        for edge in case_input.evidence.get("edges") or []
        if isinstance(edge, dict) and edge.get("src") and edge.get("dst")
    ]
    nodes = edge_endpoint_nodes(edges) + [
        node
        for node in entity_nodes(case_input.evidence)
        if node not in edge_endpoint_nodes(edges)
    ]
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_src: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        by_pair.setdefault((str(edge["src"]), str(edge["dst"])), []).append(edge)
        by_src.setdefault(str(edge["src"]), []).append(edge)
    direct_pairs = set(by_pair)

    for (subject, obj), pair_edges in by_pair.items():
        edge_ids = [edge_label(edge) for edge in pair_edges]
        semantics = (
            "reported_direct"
            if any(
                not is_chain_verified(edge.get("verification_status"))
                for edge in pair_edges
            )
            else "chain_direct"
        )
        ev_ids = sorted(
            {source_id for edge in pair_edges for source_id in evidence_ids(edge)}
        )
        support = []
        stages = []
        patterns = []
        statuses = []
        for edge in pair_edges:
            support.extend(edge_support_triples(edge))
            if edge.get("stage") not in stages:
                stages.append(edge.get("stage"))
            if edge.get("pattern") not in patterns:
                patterns.append(edge.get("pattern"))
            if edge.get("verification_status") not in statuses:
                statuses.append(edge.get("verification_status"))
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,p,o>",
            subject=subject,
            predicate="case_path_directly_connected_to",
            obj=obj,
            question=f"在 {case_name} 的 case graph 中，{short_id(subject)} 是否直接连接到 {short_id(obj)}？",
            answer=(
                f"正确。存在 {compact_value(edge_ids)}；stage={compact_value(stages)}，"
                f"pattern={compact_value(patterns)}；verification_status={compact_value(statuses)}。"
            ),
            supporting_triples=support,
            reasoning_type="direct_link_verification",
            difficulty="easy",
            evidence_ids_=ev_ids,
            sources=sources,
            answer_value=True,
            is_correct=True,
            extra_fields={
                "supporting_edge_ids": edge_ids,
                "link_semantics": semantics,
                "edge_stages": stages,
                "edge_patterns": patterns,
            },
        )

        wrong_dst = None
        for candidate in nodes:
            if candidate != subject and (subject, candidate) not in direct_pairs:
                wrong_dst = candidate
                break
        if not wrong_dst:
            continue
        actual_dsts = []
        for out_edge in by_src.get(subject, []):
            dst = str(out_edge["dst"])
            if dst not in actual_dsts:
                actual_dsts.append(dst)
        support = []
        for out_edge in by_src.get(subject, [])[:5]:
            support.extend(edge_support_triples(out_edge))
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,p,o>",
            subject=subject,
            predicate="case_path_directly_connected_to",
            obj=wrong_dst,
            question=f"在 {case_name} 的 case graph 中，{short_id(subject)} 是否直接连接到 {short_id(wrong_dst)}？",
            answer=(
                f"错误。当前 case evidence 中没有 {short_id(subject)} -> {short_id(wrong_dst)} 的直接边；"
                f"已知直接下一跳包括 {compact_value(actual_dsts)}。"
            ),
            supporting_triples=support,
            reasoning_type="negative_link_verification",
            difficulty="easy",
            evidence_ids_=sorted(
                {
                    source_id
                    for out_edge in by_src.get(subject, [])
                    for source_id in evidence_ids(out_edge)
                }
            ),
            sources=sources,
            answer_value=False,
            is_correct=False,
            correction={
                "predicate": "case_path_directly_connected_to",
                "valid_objects": actual_dsts,
            },
            extra_fields={
                "link_semantics": "reported_direct",
                "negative_sample_type": "same_case_non_edge",
            },
        )


def generate_single_hop_prediction_qa(
    case_input: CaseInput,
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    max_questions_per_src: int = 4,
) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    edges = [
        edge
        for edge in case_input.evidence.get("edges") or []
        if isinstance(edge, dict) and edge.get("src") and edge.get("dst")
    ]
    by_src: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        by_src.setdefault(str(edge["src"]), []).append(edge)
    all_nodes = edge_endpoint_nodes(edges) + [
        node
        for node in entity_nodes(case_input.evidence)
        if node not in edge_endpoint_nodes(edges)
    ]

    for src, src_edges in by_src.items():
        made_for_src = 0
        for field_name, predicate_prefix in [
            ("stage", "next_hop_for_stage"),
            ("pattern", "next_hop_for_pattern"),
        ]:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for edge in src_edges:
                value = edge.get(field_name)
                if value not in (None, "", [], {}):
                    grouped.setdefault(str(value), []).append(edge)
            for value, group_edges in grouped.items():
                if made_for_src >= max_questions_per_src:
                    break
                correct = []
                for edge in group_edges:
                    dst = str(edge["dst"])
                    if dst not in correct:
                        correct.append(dst)
                candidate_set = compact_candidates(all_nodes, correct)
                support = []
                ev_ids = []
                for edge in group_edges:
                    support.extend(edge_support_triples(edge))
                    ev_ids.extend(evidence_ids(edge))
                answer_value: Any = correct[0] if len(correct) == 1 else correct
                add_qa(
                    rows,
                    case_input=case_input,
                    question_type="<s,p,*>",
                    subject=src,
                    predicate=f"{predicate_prefix}:{value}",
                    obj="*",
                    question=(
                        f"给定候选集 [{format_candidate_set(candidate_set)}]，"
                        f"在 {case_name} 中从 {short_id(src)} 出发，{field_name}={value} 的下一跳是谁？"
                    ),
                    answer=f"{compact_value(correct)}。对应边为 {compact_value([edge_label(edge) for edge in group_edges])}。",
                    supporting_triples=support,
                    reasoning_type="single_hop_prediction",
                    difficulty="medium",
                    evidence_ids_=sorted(set(ev_ids)),
                    sources=sources,
                    answer_value=answer_value,
                    extra_fields={
                        "candidate_set": candidate_set,
                        "correct_candidates": correct,
                        "distractors": [
                            candidate
                            for candidate in candidate_set
                            if candidate not in correct
                        ],
                        "ranking_expected": True,
                        "link_semantics": "reported_direct"
                        if any(
                            not is_chain_verified(edge.get("verification_status"))
                            for edge in group_edges
                        )
                        else "chain_direct",
                    },
                )
                made_for_src += 1


def generate_path_completion_qa(
    case_input: CaseInput,
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    max_stage_sequences: int = 24,
) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    case_subject = f"case:{case_input.evidence.get('case_id') or clean_slug(case_name)}"
    edges = [
        edge
        for edge in case_input.evidence.get("edges") or []
        if isinstance(edge, dict) and edge.get("src") and edge.get("dst")
    ]
    by_src: dict[str, list[dict[str, Any]]] = {}
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for edge in edges:
        by_src.setdefault(str(edge["src"]), []).append(edge)
        by_pair.setdefault((str(edge["src"]), str(edge["dst"])), []).append(edge)

    for (src, dst), pair_edges in by_pair.items():
        edge_ids = [edge_label(edge) for edge in pair_edges]
        ev_ids = sorted(
            {source_id for edge in pair_edges for source_id in evidence_ids(edge)}
        )
        support = []
        stages = []
        patterns = []
        statuses = []
        for edge in pair_edges:
            support.extend(edge_support_triples(edge))
            if edge.get("stage") not in stages:
                stages.append(edge.get("stage"))
            if edge.get("pattern") not in patterns:
                patterns.append(edge.get("pattern"))
            if edge.get("verification_status") not in statuses:
                statuses.append(edge.get("verification_status"))
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,*,o>",
            subject=src,
            predicate="*",
            obj=dst,
            question=f"在 {case_name} 中，若要从 {short_id(src)} 追踪到 {short_id(dst)}，应该补全哪条 case graph edge？",
            answer=(
                f"应补全 {compact_value(edge_ids)}；pattern={compact_value(patterns)}，"
                f"stage={compact_value(stages)}，verification_status={compact_value(statuses)}。"
            ),
            supporting_triples=support,
            reasoning_type="path_completion",
            difficulty="hard",
            evidence_ids_=ev_ids,
            sources=sources,
            answer_value={
                "path_edges": edge_ids,
                "stages": stages,
                "patterns": patterns,
                "verification_status": statuses,
            },
            extra_fields={
                "supporting_edge_ids": edge_ids,
                "link_semantics": "reported_direct"
                if any(
                    not is_chain_verified(edge.get("verification_status"))
                    for edge in pair_edges
                )
                else "chain_direct",
            },
        )

    made = 0
    for src, src_edges in by_src.items():
        stage_edges = [edge for edge in src_edges if edge.get("stage")]
        if len(stage_edges) < 2:
            continue
        for idx, first in enumerate(stage_edges):
            for second in stage_edges[idx + 1 :]:
                if made >= max_stage_sequences:
                    return
                if first.get("stage") == second.get("stage"):
                    continue
                selected = [first, second]
                edge_ids = [edge_label(edge) for edge in selected]
                ev_ids = sorted(
                    {source_id for edge in selected for source_id in evidence_ids(edge)}
                )
                add_qa(
                    rows,
                    case_input=case_input,
                    question_type="<s,p,*>",
                    subject=case_subject,
                    predicate="path_by_stage_sequence",
                    obj="*",
                    question=(
                        f"在 {case_name} 中，哪组 reported case graph edge 可以体现 "
                        f"{first.get('stage')} -> {second.get('stage')} 的阶段序列，"
                        f"并从 {short_id(src)} 指向 {short_id(second.get('dst'))}？"
                    ),
                    answer=(
                        f"可由 {edge_ids[0]} 和 {edge_ids[1]} 组成；"
                        f"patterns={compact_value([first.get('pattern'), second.get('pattern')])}。"
                        "注意这是 stage-level reported sequence，不等价于 tx-level 连续路径。"
                    ),
                    supporting_triples=edge_support_triples(first)
                    + edge_support_triples(second),
                    reasoning_type="multi_hop_path_prediction",
                    difficulty="hard",
                    evidence_ids_=ev_ids,
                    sources=sources,
                    answer_value={
                        "path_edges": edge_ids,
                        "stages": [first.get("stage"), second.get("stage")],
                        "patterns": [first.get("pattern"), second.get("pattern")],
                    },
                    extra_fields={
                        "link_semantics": "stage_sequence",
                        "stage_sequence_is_tx_continuous": False,
                    },
                )
                made += 1


def generate_multi_hop_stage_qa(
    case_input: CaseInput,
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    max_questions: int = 24,
) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    edges = [
        edge
        for edge in case_input.evidence.get("edges") or []
        if isinstance(edge, dict)
    ]
    by_src: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        if edge.get("src"):
            by_src.setdefault(str(edge["src"]), []).append(edge)

    made = 0
    for src, src_edges in by_src.items():
        if len(src_edges) < 2 or made >= max_questions:
            continue
        edge_ids = [edge_label(edge) for edge in src_edges[:5]]
        stages = [edge.get("stage") for edge in src_edges[:5] if edge.get("stage")]
        patterns = [
            edge.get("pattern") for edge in src_edges[:5] if edge.get("pattern")
        ]
        support = []
        ev_ids = []
        for edge in src_edges[:5]:
            support.extend(edge_support_triples(edge))
            ev_ids.extend(evidence_ids(edge))
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,p,*>",
            subject=src,
            predicate="outgoing_link_stages",
            obj="*",
            question=f"在 {case_name} 中，从 {short_id(src)} 出发的多条资金链路分别体现了哪些 stage/pattern？",
            answer=f"相关边包括 {', '.join(edge_ids)}；stages={compact_value(stages)}；patterns={compact_value(patterns)}。",
            supporting_triples=support,
            reasoning_type="multi_hop_stage_reasoning",
            difficulty="hard",
            evidence_ids_=sorted(set(ev_ids)),
            sources=sources,
            answer_value={"edge_ids": edge_ids, "stages": stages, "patterns": patterns},
            extra_fields={"link_semantics": "stage_sequence"},
        )
        made += 1

    for edge in edges:
        if made >= max_questions:
            break
        next_edges = by_src.get(str(edge.get("dst") or ""), [])
        if not next_edges:
            continue
        next_edge = next_edges[0]
        first_id = edge_label(edge)
        second_id = edge_label(next_edge)
        ev_ids = evidence_ids(edge) + evidence_ids(next_edge)
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,*,o>",
            subject=str(edge.get("src")),
            predicate="*",
            obj=next_edge.get("dst"),
            question=f"在 {case_name} 中，若第一跳为 {first_id}，{short_id(edge.get('src'))} 到 {short_id(next_edge.get('dst'))} 的两跳链路经过哪些 edge 和阶段？",
            answer=(
                f"两跳路径是 {first_id} -> {second_id}；"
                f"stage 序列为 {edge.get('stage')} -> {next_edge.get('stage')}；"
                f"pattern 序列为 {edge.get('pattern')} -> {next_edge.get('pattern')}。"
            ),
            supporting_triples=edge_support_triples(edge)
            + edge_support_triples(next_edge),
            reasoning_type="multi_hop_stage_reasoning",
            difficulty="hard",
            evidence_ids_=sorted(set(ev_ids)),
            sources=sources,
            answer_value={
                "edge_ids": [first_id, second_id],
                "stages": [edge.get("stage"), next_edge.get("stage")],
                "patterns": [edge.get("pattern"), next_edge.get("pattern")],
            },
            extra_fields={
                "path_relation": "two_hop_path",
                "link_semantics": "reported_direct"
                if not is_chain_verified(edge.get("verification_status"))
                or not is_chain_verified(next_edge.get("verification_status"))
                else "chain_direct",
            },
        )
        made += 1


def deduplicate_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    out = []
    for row in rows:
        key = (
            str(row.get("case_dir")),
            str(row.get("question")),
            str(row.get("answer")),
            str(row.get("reasoning_type")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    for idx, row in enumerate(out, start=1):
        subject = clean_slug(str(row.get("subject") or ""))[:40]
        predicate = clean_slug(str(row.get("predicate") or ""))[:30]
        row["id"] = f"{row['case_dir']}:{idx:04d}:{subject}:{predicate}"
    return out


def generate_constraint_qa(case_input: CaseInput, rows: list[dict[str, Any]]) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    constraints = [
        str(item) for item in case_input.evidence.get("negative_constraints") or []
    ]
    for constraint in constraints:
        subject = f"constraint:{constraint}"
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,p,*>",
            subject=subject,
            predicate="policy",
            obj="*",
            question=f"在 {case_name} 中，边界规则 {constraint} 对 Agent 的要求是什么？",
            answer=f"Agent 必须遵守该负向约束：{constraint}。",
            supporting_triples=[
                triple(
                    subject, "policy", constraint, "chain_evidence.negative_constraints"
                )
            ],
            reasoning_type="constraint_lookup",
            difficulty="medium",
        )

    entity_roles = {
        entity_label(entity): entity.get("role")
        for entity in case_input.evidence.get("entities") or []
        if isinstance(entity, dict) and entity.get("role")
    }
    for subject, role in entity_roles.items():
        role_text = str(role).lower()
        if (
            "exchange" not in role_text
            and "mixer" not in role_text
            and "router" not in role_text
        ):
            continue
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,p,o>",
            subject=subject,
            predicate="role",
            obj="attacker_controlled_wallet",
            question=f"判断三元组 ⟨{short_id(subject)}, role, attacker_controlled_wallet⟩ 在 {case_name} 中是否正确。",
            answer=f"错误。该实体的已知 role 是 {role}，不能在没有外部证据时改写为攻击者控制钱包。",
            supporting_triples=[
                triple(subject, "role", role, "chain_evidence.entities")
            ],
            reasoning_type="constraint_check",
            difficulty="medium",
        )


def generate_transaction_qa(
    case_input: CaseInput,
    rows: list[dict[str, Any]],
    max_transactions: int,
) -> None:
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    txs = [
        tx
        for tx in case_input.evidence.get("transaction_evidence") or []
        if isinstance(tx, dict)
    ]
    for tx in txs[:max_transactions]:
        subject = str(tx.get("tx_hash") or "")
        if not subject:
            continue
        support = []
        for key in [
            "chain",
            "src_address",
            "dst_address",
            "value_eth",
            "block_number",
            "timestamp",
            "verification_status",
        ]:
            if tx.get(key) not in (None, "", [], {}):
                support.append(
                    triple(subject, key, tx[key], "chain_evidence.transaction_evidence")
                )
        labels = tx.get("rule_labels") or {}
        if isinstance(labels, dict):
            for key in ["primary_label", "stage", "confidence"]:
                if labels.get(key) not in (None, "", [], {}):
                    support.append(
                        triple(
                            subject,
                            f"rule_labels.{key}",
                            labels[key],
                            "chain_evidence.transaction_evidence",
                        )
                    )

        if not support:
            continue
        answer_parts = [
            f"{t['predicate']}={compact_value(t['object'])}" for t in support
        ]
        add_qa(
            rows,
            case_input=case_input,
            question_type="<s,*,*>",
            subject=subject,
            predicate="*",
            obj="*",
            question=f"在 {case_name} 中，交易 {short_id(subject)} 有哪些链上证据信息？",
            answer="; ".join(answer_parts),
            supporting_triples=support,
            reasoning_type="transaction_profile_lookup",
            difficulty="medium",
        )
        for predicate in ["verification_status", "src_address", "dst_address"]:
            if tx.get(predicate) in (None, "", [], {}):
                continue
            add_qa(
                rows,
                case_input=case_input,
                question_type="<s,p,*>",
                subject=subject,
                predicate=predicate,
                obj="*",
                question=f"{case_name} 中交易 {short_id(subject)} 的 {predicate} 是什么？",
                answer=compact_value(tx[predicate]),
                supporting_triples=[
                    triple(
                        subject,
                        predicate,
                        tx[predicate],
                        "chain_evidence.transaction_evidence",
                    )
                ],
                reasoning_type="transaction_metadata_lookup",
                difficulty="easy",
            )


def generate_deterministic_qa(
    case_input: CaseInput,
    *,
    max_entities: int,
    max_transactions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = source_map(case_input.evidence)
    generate_case_level_qa(case_input, rows)
    generate_entity_qa(case_input, rows, sources, max_entities=max_entities)
    generate_edge_qa(case_input, rows, sources)
    generate_source_reliability_qa(case_input, rows, sources)
    generate_agent_action_qa(case_input, rows, sources)
    generate_constraint_application_qa(case_input, rows, sources)
    generate_negative_triple_qa(case_input, rows, sources)
    generate_direct_link_verification_qa(case_input, rows, sources)
    generate_single_hop_prediction_qa(case_input, rows, sources)
    generate_path_completion_qa(case_input, rows, sources)
    generate_multi_hop_stage_qa(case_input, rows, sources)
    generate_constraint_qa(case_input, rows)
    generate_transaction_qa(case_input, rows, max_transactions=max_transactions)
    return deduplicate_questions(rows)


def compact_facts_for_llm(case_input: CaseInput, limit: int = 120) -> dict[str, Any]:
    evidence = case_input.evidence
    sources = []
    for source in (evidence.get("sources") or [])[:limit]:
        if not isinstance(source, dict):
            continue
        finding = str(source.get("finding") or source.get("findings") or "")
        sources.append(
            {
                "id": source.get("id"),
                "entity": source.get("entity") or source.get("source_entity"),
                "grade": source.get("grade") or source.get("source_grade"),
                "url": source.get("url") or source.get("source"),
                "finding": finding[:700],
            }
        )
    entities = []
    for entity in (evidence.get("entities") or [])[:limit]:
        if not isinstance(entity, dict):
            continue
        entities.append(
            {
                key: entity.get(key)
                for key in [
                    "id",
                    "type",
                    "role",
                    "entity",
                    "chain",
                    "confidence",
                    "evidence",
                ]
            }
        )
    edges = []
    for edge in (evidence.get("edges") or [])[:limit]:
        if not isinstance(edge, dict):
            continue
        edges.append(
            {
                key: edge.get(key)
                for key in [
                    "id",
                    "src",
                    "dst",
                    "stage",
                    "pattern",
                    "amount",
                    "asset",
                    "asset_in",
                    "asset_out",
                    "route",
                    "confidence",
                    "evidence",
                    "verification_status",
                ]
            }
        )
    return {
        "case_dir": case_input.case_dir,
        "case_name": normalize_case_name(
            case_input.case_dir, case_input.summary, evidence
        ),
        "case_id": evidence.get("case_id"),
        "incident_type": evidence.get("incident_type"),
        "chains": evidence.get("chains"),
        "protocol_type": evidence.get("protocol_type"),
        "loss": evidence.get("loss"),
        "root_cause": evidence.get("root_cause"),
        "entities": entities,
        "edges": edges,
        "sources": sources,
        "negative_constraints": evidence.get("negative_constraints") or [],
        "open_questions": evidence.get("open_questions") or [],
    }


def extract_json_array(text: str) -> list[Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", stripped)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, list):
        raise ValueError("LLM response is not a JSON array")
    return parsed


def is_wildcard(value: Any) -> bool:
    return value in ("*", None, "")


def infer_question_type(subject: Any, predicate: Any, obj: Any) -> str:
    if is_wildcard(predicate) and is_wildcard(obj):
        return "<s,*,*>"
    if not is_wildcard(predicate) and is_wildcard(obj):
        return "<s,p,*>"
    if is_wildcard(predicate) and not is_wildcard(obj):
        return "<s,*,o>"
    return "<s,p,o>"


def normalize_question_type(
    value: Any, subject: Any, predicate: Any, obj: Any
) -> tuple[str, str | None]:
    raw = str(value or "").strip()
    normalized = raw.replace(" ", "")
    aliases = {
        "s**": "<s,*,*>",
        "sp*": "<s,p,*>",
        "s*o": "<s,*,o>",
        "spo": "<s,p,o>",
        "<s,*,*>": "<s,*,*>",
        "<s,p,*>": "<s,p,*>",
        "<s,*,o>": "<s,*,o>",
        "<s,p,o>": "<s,p,o>",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in QUESTION_TYPES:
        return normalized, None
    inferred = infer_question_type(subject, predicate, obj)
    return inferred, raw or None


def normalize_supporting_triples(triples: Any) -> list[dict[str, Any]]:
    normalized = []
    if not isinstance(triples, list):
        return normalized
    for item in triples:
        if isinstance(item, dict):
            if {"subject", "predicate", "object"}.issubset(item):
                normalized.append(item)
            continue
        if isinstance(item, list) and len(item) >= 3:
            normalized.append(
                {
                    "subject": item[0],
                    "predicate": item[1],
                    "object": item[2],
                    "origin": "llm_extra",
                }
            )
    return normalized


def normalize_llm_item(
    item: dict[str, Any],
    *,
    case_input: CaseInput,
    case_name: str,
    idx: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = []
    subject = item.get("subject", "")
    predicate = item.get("predicate", "")
    obj = item.get("object", "")
    question_type, original_question_type = normalize_question_type(
        item.get("question_type"), subject, predicate, obj
    )
    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()
    supporting_triples = normalize_supporting_triples(
        item.get("supporting_triples") or []
    )

    for field_name, field_value in [
        ("subject", subject),
        ("predicate", predicate),
        ("object", obj),
        ("question", question),
        ("answer", answer),
    ]:
        if field_value in ("", None):
            errors.append(f"missing {field_name}")
    if not supporting_triples:
        errors.append("missing supporting_triples")

    if errors:
        return None, errors

    answer_value = item.get("answer_value", answer)
    row = {
        "id": f"{case_input.case_dir}:llm:{idx:03d}",
        "case_dir": case_input.case_dir,
        "case_id": case_input.evidence.get("case_id"),
        "case_name": case_name,
        "question_type": question_type,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "question": question,
        "answer": answer,
        "answer_value": answer_value,
        "answer_aliases": item.get("answer_aliases") or [],
        "answer_format": item.get("answer_format") or infer_answer_format(answer_value),
        "supporting_triples": supporting_triples,
        "supporting_sources": item.get("supporting_sources") or [],
        "reasoning_type": item.get("reasoning_type", "llm_generated"),
        "difficulty": item.get("difficulty", "medium"),
        "generator": "llm_extra",
    }
    if original_question_type:
        row["original_question_type"] = original_question_type
        row["question_type_repaired"] = True
    if "is_correct" in item:
        row["is_correct"] = item["is_correct"]
    if item.get("correction"):
        row["correction"] = item["correction"]
    row["qa_domain"] = classify_qa_domain(row)
    return row, []


def normalize_llm_items(
    raw_items: list[Any],
    *,
    case_input: CaseInput,
    case_name: str,
    max_items: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    failures = []
    for idx, item in enumerate(raw_items, start=1):
        if len(rows) >= max_items:
            break
        if not isinstance(item, dict):
            failures.append(
                {"index": idx, "errors": ["item is not an object"], "item": item}
            )
            continue
        row, errors = normalize_llm_item(
            item,
            case_input=case_input,
            case_name=case_name,
            idx=idx,
        )
        if errors or row is None:
            failures.append({"index": idx, "errors": errors, "item": item})
            continue
        rows.append(row)
    return rows, failures


def build_llm_messages(facts: dict[str, Any], max_items: int) -> list[dict[str, str]]:
    system_prompt = (
        "You generate Chinese KGQA-style QA pairs for on-chain incident RAG evaluation. "
        "Use only the provided JSON facts. Do not add external knowledge. "
        "If verification_status is not chain_verified, do not describe the fact as chain verified. "
        "Do not convert CEX, mixer, DEX, bridge, or router entities into attacker-controlled wallets unless the input says so.\n\n"
        f"{QUESTION_TYPE_RULES}\n\n"
        "Every item must include question_type, subject, predicate, object, question, answer, answer_value, answer_format, "
        "supporting_triples, reasoning_type, difficulty. supporting_triples must be an array of objects with "
        "subject/predicate/object/origin, not bare arrays. Prefer source_reliability_check, boundary_decision, "
        "agent_action_policy, ground_truth_usability_check, and multi_hop_stage_reasoning when facts support them."
    )
    user_prompt = (
        f"Generate up to {max_items} additional medium/hard QA pairs. Return a JSON array only.\n\n"
        "Schema reminder:\n"
        "{\n"
        '  "question_type": "<s,p,*>" | "<s,*,*>" | "<s,*,o>" | "<s,p,o>",\n'
        '  "subject": "known subject or edge id",\n'
        '  "predicate": "known predicate or *",\n'
        '  "object": "known object or *",\n'
        '  "question": "...",\n'
        '  "answer": "...",\n'
        '  "answer_value": "...",\n'
        '  "answer_format": "string|list|boolean|object|number",\n'
        '  "supporting_triples": [{"subject":"...","predicate":"...","object":"...","origin":"chain_evidence.edges"}],\n'
        '  "reasoning_type": "multi_hop_stage_reasoning|boundary_decision|agent_action_policy|source_reliability_check|ground_truth_usability_check|false_positive_check",\n'
        '  "difficulty": "medium|hard"\n'
        "}\n\n"
        'Bad example: question_type="multi_hop_stage_reasoning".\n'
        'Good example: question_type="<s,p,*>" and reasoning_type="multi_hop_stage_reasoning".\n\n'
        f"Facts:\n{json.dumps(facts, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_llm_repair_messages(
    raw_content: str,
    failures: list[dict[str, Any]],
    max_items: int,
) -> list[dict[str, str]]:
    repair_prompt = (
        "Repair the JSON array so every item conforms to the schema. Return a JSON array only.\n\n"
        f"{QUESTION_TYPE_RULES}\n\n"
        "Common repair: if question_type is a reasoning label, move it to reasoning_type and infer question_type from "
        "subject/predicate/object. supporting_triples must be objects with subject/predicate/object/origin.\n\n"
        f"Need up to {max_items} valid items.\n"
        f"Validation failures:\n{json.dumps(failures[:20], ensure_ascii=False)}\n\n"
        f"Original response:\n{raw_content}"
    )
    return [
        {
            "role": "system",
            "content": "You repair malformed KGQA JSON. Return JSON only.",
        },
        {"role": "user", "content": repair_prompt},
    ]


def post_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    enable_thinking: bool,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if "qwen" in model.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_llm_extra(case_input: CaseInput, max_items: int) -> list[dict[str, Any]]:
    load_env_file()
    base_url = os.environ.get("LLM_PROVIDER_URL", "http://127.0.0.1:18080/v1").rstrip(
        "/"
    )
    model = os.environ.get("LLM_NAME", "Qwen/Qwen3.6-27B")
    api_key = os.environ.get("LLM_API_KEY", "NO_API_KEY")
    enable_thinking = os.environ.get("LLM_THINK", "False").lower() in {
        "1",
        "true",
        "yes",
    }
    max_tokens = int(os.environ.get("LLM_OUTPUT_LENGTH", "200000"))

    facts = compact_facts_for_llm(case_input)
    content = post_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=build_llm_messages(facts, max_items),
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
    raw_items = extract_json_array(content)
    case_name = normalize_case_name(
        case_input.case_dir, case_input.summary, case_input.evidence
    )
    rows, failures = normalize_llm_items(
        raw_items,
        case_input=case_input,
        case_name=case_name,
        max_items=max_items,
    )
    if len(rows) >= max_items or not failures:
        return rows

    for _ in range(MAX_LLM_REPAIR_ROUNDS):
        repair_content = post_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=build_llm_repair_messages(
                content, failures, max_items - len(rows)
            ),
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        repaired_items = extract_json_array(repair_content)
        repaired_rows, failures = normalize_llm_items(
            repaired_items,
            case_input=case_input,
            case_name=case_name,
            max_items=max_items - len(rows),
        )
        rows.extend(repaired_rows)
        if len(rows) >= max_items or not failures:
            break
    return rows[:max_items]


def load_case(case_dir: str) -> CaseInput:
    root = SUMMARIZED_DIR / case_dir
    summary_path = root / "summary.json"
    evidence_path = root / "chain_evidence.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path}")
    if not evidence_path.exists():
        raise FileNotFoundError(f"missing {evidence_path}")
    return CaseInput(
        case_dir=case_dir,
        summary=read_json(summary_path),
        evidence=read_json(evidence_path),
    )


def discover_cases() -> list[str]:
    return discover_selected_cases(
        SUMMARIZED_DIR,
        required_files=("summary.json", "chain_evidence.json"),
    )


def select_cases(case_arg: str) -> list[str]:
    if case_arg.lower() == "all":
        return discover_cases()
    return [case_arg]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts_by_type: dict[str, int] = {}
    counts_by_generator: dict[str, int] = {}
    counts_by_difficulty: dict[str, int] = {}
    counts_by_domain: dict[str, int] = {}
    counts_by_reasoning_type: dict[str, int] = {}
    for row in rows:
        counts_by_type[row["question_type"]] = (
            counts_by_type.get(row["question_type"], 0) + 1
        )
        counts_by_generator[row["generator"]] = (
            counts_by_generator.get(row["generator"], 0) + 1
        )
        counts_by_difficulty[row["difficulty"]] = (
            counts_by_difficulty.get(row["difficulty"], 0) + 1
        )
        domain = str(row.get("qa_domain") or classify_qa_domain(row))
        counts_by_domain[domain] = counts_by_domain.get(domain, 0) + 1
        reasoning_type = str(row.get("reasoning_type") or "")
        counts_by_reasoning_type[reasoning_type] = (
            counts_by_reasoning_type.get(reasoning_type, 0) + 1
        )
    return {
        "total_questions": len(rows),
        "counts_by_type": counts_by_type,
        "counts_by_generator": counts_by_generator,
        "counts_by_difficulty": counts_by_difficulty,
        "counts_by_domain": counts_by_domain,
        "counts_by_reasoning_type": counts_by_reasoning_type,
    }


def write_partition_outputs(
    out_dir: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[str]:
    write_jsonl(out_dir / "qa.jsonl", rows)
    write_json(out_dir / "qa.json", rows)
    write_json(out_dir / "manifest.json", manifest)
    return [
        str((out_dir / "qa.jsonl").relative_to(PROJECT_ROOT)),
        str((out_dir / "qa.json").relative_to(PROJECT_ROOT)),
        str((out_dir / "manifest.json").relative_to(PROJECT_ROOT)),
    ]


def write_case_outputs(
    case_input: CaseInput, rows: list[dict[str, Any]], llm_error: str | None = None
) -> None:
    out_dir = QA_DIR / case_input.case_dir
    for row in rows:
        row["qa_domain"] = classify_qa_domain(row)
    general_rows = [row for row in rows if row["qa_domain"] == "general"]
    chain_rows = [row for row in rows if row["qa_domain"] == "chain"]
    write_jsonl(out_dir / "qa.jsonl", rows)
    write_json(out_dir / "qa.json", rows)

    manifest = {
        "case_dir": case_input.case_dir,
        "case_id": case_input.evidence.get("case_id"),
        "case_name": normalize_case_name(
            case_input.case_dir, case_input.summary, case_input.evidence
        ),
        **summarize_rows(rows),
        "input_files": [
            str(
                (SUMMARIZED_DIR / case_input.case_dir / "summary.json").relative_to(
                    PROJECT_ROOT
                )
            ),
            str(
                (
                    SUMMARIZED_DIR / case_input.case_dir / "chain_evidence.json"
                ).relative_to(PROJECT_ROOT)
            ),
        ],
        "output_files": [
            str((out_dir / "qa.jsonl").relative_to(PROJECT_ROOT)),
            str((out_dir / "qa.json").relative_to(PROJECT_ROOT)),
            str((out_dir / "manifest.json").relative_to(PROJECT_ROOT)),
        ],
        "partition_files": {
            "general": [
                str((out_dir / "general" / "qa.jsonl").relative_to(PROJECT_ROOT)),
                str((out_dir / "general" / "qa.json").relative_to(PROJECT_ROOT)),
                str((out_dir / "general" / "manifest.json").relative_to(PROJECT_ROOT)),
            ],
            "chain": [
                str((out_dir / "chain" / "qa.jsonl").relative_to(PROJECT_ROOT)),
                str((out_dir / "chain" / "qa.json").relative_to(PROJECT_ROOT)),
                str((out_dir / "chain" / "manifest.json").relative_to(PROJECT_ROOT)),
            ],
        },
        "partition_policy": {
            "general": "case/entity metadata and non-link profile questions",
            "chain": "edge, transaction, path, next-hop, boundary, verification, and link-prediction questions",
            "all_compatibility_files": [
                str((out_dir / "qa.jsonl").relative_to(PROJECT_ROOT)),
                str((out_dir / "qa.json").relative_to(PROJECT_ROOT)),
            ],
        },
    }
    if llm_error:
        manifest["llm_error"] = llm_error
    write_json(out_dir / "manifest.json", manifest)

    base_manifest = {
        "case_dir": manifest["case_dir"],
        "case_id": manifest["case_id"],
        "case_name": manifest["case_name"],
        "input_files": manifest["input_files"],
    }
    general_manifest = {
        **base_manifest,
        "qa_domain": "general",
        **summarize_rows(general_rows),
    }
    chain_manifest = {
        **base_manifest,
        "qa_domain": "chain",
        **summarize_rows(chain_rows),
        "link_semantics_counts": {},
    }
    for row in chain_rows:
        semantics = str(row.get("link_semantics") or "unspecified")
        chain_manifest["link_semantics_counts"][semantics] = (
            chain_manifest["link_semantics_counts"].get(semantics, 0) + 1
        )
    write_partition_outputs(out_dir / "general", general_rows, general_manifest)
    write_partition_outputs(out_dir / "chain", chain_rows, chain_manifest)


def process_case(
    case_dir: str,
    *,
    max_entities: int,
    max_transactions: int,
    llm_extra: bool,
    max_llm_extra: int,
) -> tuple[str, int, str | None]:
    case_input = load_case(case_dir)
    rows = generate_deterministic_qa(
        case_input,
        max_entities=max_entities,
        max_transactions=max_transactions,
    )
    llm_error = None
    if llm_extra:
        try:
            rows.extend(call_llm_extra(case_input, max_items=max_llm_extra))
        except (
            Exception
        ) as exc:  # Keep deterministic output even if the optional LLM path fails.
            llm_error = f"{type(exc).__name__}: {exc}"
    write_case_outputs(case_input, rows, llm_error=llm_error)
    return case_dir, len(rows), llm_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract KGQA-style QA pairs from summarized on-chain case evidence."
    )
    parser.add_argument(
        "case",
        nargs="?",
        default="all",
        help="Case directory under summarized/, or 'all'.",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=80,
        help="Maximum entities per case to template into QA.",
    )
    parser.add_argument(
        "--max-transactions",
        type=int,
        default=40,
        help="Maximum transaction_evidence records per case to template into QA.",
    )
    parser.add_argument(
        "--llm-extra",
        action="store_true",
        help="Use the .env LLM endpoint to generate extra QA pairs.",
    )
    parser.add_argument("--max-llm-extra", type=int, default=DEFAULT_MAX_LLM_EXTRA)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selected_cases = select_cases(args.case)
    if not selected_cases:
        raise SystemExit(
            "No summarized cases with summary.json and chain_evidence.json were found."
        )

    results = []
    for case_dir in selected_cases:
        results.append(
            process_case(
                case_dir,
                max_entities=args.max_entities,
                max_transactions=args.max_transactions,
                llm_extra=args.llm_extra,
                max_llm_extra=args.max_llm_extra,
            )
        )

    for case_dir, count, llm_error in results:
        suffix = f" (LLM error: {llm_error})" if llm_error else ""
        print(f"{case_dir}: wrote {count} QA pairs to qa/{case_dir}/{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
