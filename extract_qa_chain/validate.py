from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from .schema import CandidateSet, ChainQATask, EdgeLabel, ObservedEdge


def validate_bundle(
    edges: list[ObservedEdge],
    labels: list[EdgeLabel],
    candidates: list[CandidateSet],
    tasks: list[ChainQATask],
    *,
    k_hop: int | None = None,
) -> dict[str, Any]:
    errors = []
    edge_ids = {edge.edge_id for edge in edges}
    label_ids = {label.edge_id for label in labels}
    hop_counts = Counter(edge.hop_from_seed for edge in edges)
    missing_hop_edges = [edge.edge_id for edge in edges if edge.hop_from_seed is None]
    invalid_hop_edges = [
        edge.edge_id
        for edge in edges
        if edge.hop_from_seed is not None
        and (
            not isinstance(edge.hop_from_seed, int)
            or isinstance(edge.hop_from_seed, bool)
            or edge.hop_from_seed < 1
            or (k_hop is not None and edge.hop_from_seed > k_hop)
        )
    ]
    if missing_hop_edges:
        errors.append(
            "observed edges missing hop_from_seed: "
            f"count={len(missing_hop_edges)} sample={missing_hop_edges[:10]}"
        )
    if invalid_hop_edges:
        errors.append(
            "observed edges have invalid hop_from_seed: "
            f"count={len(invalid_hop_edges)} k_hop={k_hop} "
            f"sample={invalid_hop_edges[:10]}"
        )

    def public_candidate_groups(context: dict[str, Any]) -> dict[str, list[str]]:
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
    for label in labels:
        if label.label_semantics == "chain_direct" and label.edge_id not in edge_ids:
            errors.append(f"chain_direct label missing observed edge: {label.edge_id}")
    for cand in candidates:
        summary_ids = {
            str(summary.get("candidate_id") or summary.get("edge_id"))
            for summary in cand.candidate_summaries
        }
        if len(cand.candidate_edges) < 2:
            errors.append(f"candidate set has fewer than two choices: {cand.state_id}")
        for candidate_id in cand.candidate_edges:
            if candidate_id not in cand.candidate_edge_groups:
                errors.append(
                    f"candidate missing raw edge group: {cand.state_id} {candidate_id}"
                )
            if candidate_id not in summary_ids:
                errors.append(
                    f"candidate missing aggregate summary: {cand.state_id} {candidate_id}"
                )
            for edge_id in cand.candidate_edge_groups.get(candidate_id, []):
                if edge_id not in edge_ids:
                    errors.append(
                        f"candidate raw edge missing observed edge: {cand.state_id} {edge_id}"
                    )
                if edge_id not in label_ids:
                    errors.append(
                        f"candidate raw edge missing label: {cand.state_id} {edge_id}"
                    )
    for task in tasks:
        ctx = task.graph_context or {}
        summary_ids = {
            str(summary.get("candidate_id") or summary.get("edge_id"))
            for summary in task.edge_summaries
        }
        for candidate_id in ctx.get("candidate_edges") or []:
            if candidate_id not in edge_ids and candidate_id not in summary_ids:
                errors.append(
                    f"task candidate missing edge or aggregate summary: {task.id} {candidate_id}"
                )
        leaked_keys = {
            "action_labels",
            "correct_next_edges",
            "correct_follow_edges",
            "none_of_above",
            "exists",
            "transaction_count",
            "matching_edge_ids",
            "matching_tx_hashes",
            "verified_absent",
            "negative_sample_type",
        } & set(ctx)
        if leaked_keys:
            errors.append(
                f"task graph_context leaks oracle keys: {task.id} {sorted(leaked_keys)}"
            )
        if task.answer_value in (None, "", [], {}):
            errors.append(f"task missing answer_value: {task.id}")
        if task.task_type == "single_hop_next_hop_prediction":
            errors.append(f"deprecated task type remains: {task.id}")
        if task.task_type == "direct_link_verification":
            errors.append(f"deprecated direct-link task remains: {task.id}")
        if task.task_type == "direct_transaction_existence":
            required_context = {
                "src",
                "dst",
                "chain",
                "direction",
                "block_min",
                "block_max",
                "edge_types",
                "require_tx_hashes",
            }
            if not required_context.issubset(ctx):
                errors.append(f"transaction task missing query constraints: {task.id}")
            if ctx.get("chain") != "ethereum" or ctx.get("direction") != "out":
                errors.append(f"transaction task has invalid chain or direction: {task.id}")
            if str(ctx.get("src") or "") != str(ctx.get("src") or "").lower():
                errors.append(f"transaction task src is not lowercase: {task.id}")
            if str(ctx.get("dst") or "") != str(ctx.get("dst") or "").lower():
                errors.append(f"transaction task dst is not lowercase: {task.id}")
            if not isinstance(ctx.get("block_min"), int) or not isinstance(
                ctx.get("block_max"), int
            ):
                errors.append(f"transaction task block window is not explicit: {task.id}")
            if not isinstance(ctx.get("edge_types"), list) or not ctx.get("edge_types"):
                errors.append(f"transaction task edge types are not explicit: {task.id}")
            if task.edge_summaries:
                errors.append(f"query-only transaction task exposes edge summaries: {task.id}")
            if (task.evidence or {}).get("chain_edges"):
                errors.append(f"query-only transaction task exposes matching edges: {task.id}")
            if task.agent_profile != "tool_grounded":
                errors.append(f"transaction task is not tool_grounded: {task.id}")
            if task.tool_requirement != "required" or "eth_neighbors" not in task.required_tools:
                errors.append(f"transaction task does not require eth_neighbors: {task.id}")
            if task.public_context_profile != "query_only":
                errors.append(f"transaction task is not query_only: {task.id}")
            value = task.answer_value if isinstance(task.answer_value, dict) else {}
            if not isinstance(value.get("exists"), bool):
                errors.append(f"transaction task exists is not boolean: {task.id}")
            if value.get("exists"):
                if not value.get("matching_edge_ids") or not value.get("matching_tx_hashes"):
                    errors.append(f"positive transaction task lacks index oracle: {task.id}")
            elif value.get("verified_absent") is not True:
                errors.append(f"negative transaction task is not verified absent: {task.id}")
        if task.task_type in {
            "edge_action_classification",
            "path_completion",
            "subgraph_noise_filtering",
        }:
            groups = public_candidate_groups(ctx)
            public_candidates = {
                str(candidate_id) for candidate_id in ctx.get("candidate_edges") or []
            }
            for state in ctx.get("states") or []:
                if isinstance(state, dict):
                    public_candidates.update(
                        str(candidate_id)
                        for candidate_id in state.get("candidate_edges") or []
                    )
            if task.agent_profile == "tool_grounded":
                if task.required_tools != ["eth_edge"] or task.allowed_tools != ["eth_edge"]:
                    errors.append(f"tool-grounded tracing task must use only eth_edge: {task.id}")
                if task.tool_requirement != "required":
                    errors.append(f"tool-grounded tracing task does not require tools: {task.id}")
                if task.public_context_profile != "candidate_tool_query":
                    errors.append(f"tool-grounded tracing task has wrong public profile: {task.id}")
                if not groups or set(groups) != public_candidates:
                    errors.append(f"tool-grounded tracing groups do not match candidates: {task.id}")
                contract = ctx.get("tool_query_contract") or {}
                if (
                    contract.get("tool") != "eth_edge"
                    or contract.get("with_raw") is not True
                    or contract.get("required_evidence_per_candidate") != 1
                    or contract.get("candidate_count") != len(groups)
                ):
                    errors.append(f"tool-grounded tracing contract is invalid: {task.id}")
                if contract.get("max_tool_calls") != len(groups):
                    errors.append(f"tool-grounded tracing tool budget is invalid: {task.id}")
            elif task.agent_profile == "context_only":
                if task.required_tools or task.allowed_tools:
                    errors.append(f"context-only tracing task exposes tools: {task.id}")
                if task.tool_requirement != "disabled":
                    errors.append(f"context-only tracing task enables tools: {task.id}")
                if task.public_context_profile != "graph_context":
                    errors.append(f"context-only tracing task has wrong public profile: {task.id}")
            else:
                errors.append(f"tracing task has unsupported agent profile: {task.id}")
        if task.task_type == "edge_action_classification":
            candidate_ids = ctx.get("candidate_edges") or []
            if len(candidate_ids) < 2:
                errors.append(f"edge action task is not multi-candidate: {task.id}")
            value = task.answer_value if isinstance(task.answer_value, dict) else {}
            if set(value.get("candidate_actions") or {}) != set(candidate_ids):
                errors.append(f"edge action gold does not match candidates: {task.id}")
            if ctx.get("response_schema") is None:
                errors.append(f"edge action task missing response schema: {task.id}")
        if (
            task.task_type in {"path_completion", "subgraph_noise_filtering"}
            and ctx.get("sequential_policy")
        ):
            states = ctx.get("states") or []
            state_ids = [
                str(state.get("state_id") or "")
                for state in states
                if isinstance(state, dict)
            ]
            if not states or any(not state_id for state_id in state_ids):
                errors.append(f"sequential task has invalid states: {task.id}")
            if len(state_ids) != len(set(state_ids)):
                errors.append(f"sequential task has duplicate state ids: {task.id}")
            state_candidates = {
                str(state.get("state_id")): {
                    str(value) for value in state.get("candidate_edges") or []
                }
                for state in states
                if isinstance(state, dict) and state.get("state_id")
            }
            value = task.answer_value if isinstance(task.answer_value, dict) else {}
            gold_states = value.get("state_action_labels") or {}
            if set(gold_states) != set(state_candidates):
                errors.append(f"sequential gold states do not match public states: {task.id}")
            for state_id, actions in gold_states.items():
                if not isinstance(actions, dict) or set(actions) != state_candidates.get(state_id, set()):
                    errors.append(
                        f"sequential gold candidates do not match state {state_id}: {task.id}"
                    )
            budget = ctx.get("rollout_budget") or {}
            for field in (
                "max_depth",
                "beam_width",
                "max_node_expansions",
                "max_inspected_edges",
                "max_tool_calls",
            ):
                if not isinstance(budget.get(field), int) or budget[field] < 0:
                    errors.append(f"sequential task has invalid {field}: {task.id}")
            if task.agent_profile == "tool_grounded":
                contract = ctx.get("tool_query_contract") or {}
                if budget.get("max_tool_calls") != contract.get("max_tool_calls"):
                    errors.append(f"sequential task tool budgets disagree: {task.id}")
            if ctx.get("response_schema") is None:
                errors.append(f"sequential task missing response schema: {task.id}")
            if task.task_type == "path_completion":
                if ctx.get("evaluation_modes") != ["teacher_forced", "free_rollout"]:
                    errors.append(f"path task has invalid evaluation modes: {task.id}")
                if len(states) < 2:
                    errors.append(f"sequential path task is not multi-step: {task.id}")
            else:
                if ctx.get("trusted_seed") is not False:
                    errors.append(f"sequential noise task must hide trusted seed: {task.id}")
                if "current_node" in ctx:
                    errors.append(f"sequential noise task exposes current_node: {task.id}")
                candidate_seeds = ctx.get("candidate_seed_nodes") or []
                if not candidate_seeds:
                    errors.append(f"sequential noise task has no seed candidates: {task.id}")
                if ctx.get("evaluation_modes") != ["oracle_seed", "predicted_seed"]:
                    errors.append(f"noise task has invalid evaluation modes: {task.id}")
                if ctx.get("agent_output_contract") != "snf_state_policy_v2":
                    errors.append(f"noise task has invalid agent output contract: {task.id}")
                if ctx.get("rollout_derivation") != "evaluator_seed_projection_v1":
                    errors.append(f"noise task has invalid rollout derivation: {task.id}")
                response_schema = ctx.get("response_schema") or {}
                if "state_policy" not in response_schema:
                    errors.append(f"noise task response schema lacks state_policy: {task.id}")
                if "oracle_seed" in response_schema or "predicted_seed" in response_schema:
                    errors.append(f"noise task exposes evaluator-derived rollout modes: {task.id}")
    return {
        "ok": not errors,
        "errors": errors,
        "counts": {
            "edges": len(edges),
            "labels": len(labels),
            "candidates": len(candidates),
            "tasks": len(tasks),
        },
        "label_counts": dict(Counter(label.label for label in labels)),
        "task_counts": dict(Counter(task.task_type for task in tasks)),
        "hop_counts": {
            str(hop): count
            for hop, count in sorted(
                hop_counts.items(), key=lambda item: (item[0] is None, str(item[0]))
            )
        },
        "missing_hop_count": len(missing_hop_edges),
        "invalid_hop_count": len(invalid_hop_edges),
    }
