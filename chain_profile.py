from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


CHAIN_AGENT_PROFILE_NAMES = {"context_only", "tool_grounded"}
CONTEXT_ONLY_FALLBACK_TASK_TYPES = {
    "edge_action_classification",
    "path_completion",
    "subgraph_noise_filtering",
}
CONTEXT_ONLY_PROJECTION_KIND = "tool_grounded_to_context_only_v1"


def task_agent_profile(task: dict[str, Any]) -> str:
    profile = str(task.get("agent_profile") or "context_only")
    if task.get("task_type") == "direct_transaction_existence":
        profile = "tool_grounded"
    if profile not in CHAIN_AGENT_PROFILE_NAMES:
        raise ValueError(f"unsupported chain agent profile: {profile}")
    return profile


def project_tool_grounded_to_context_only(
    task: dict[str, Any],
) -> dict[str, Any]:
    if task_agent_profile(task) != "tool_grounded":
        raise ValueError("context-only fallback source must be tool_grounded")
    if task.get("task_type") not in CONTEXT_ONLY_FALLBACK_TASK_TYPES:
        raise ValueError(
            "context-only fallback supports only EAC, PC, and SNF tasks"
        )

    projected = deepcopy(task)
    context = projected.get("graph_context")
    if not isinstance(context, dict):
        context = {}
        projected["graph_context"] = context
    context.pop("tool_query_contract", None)
    rollout_budget = context.get("rollout_budget")
    if isinstance(rollout_budget, dict):
        rollout_budget["max_tool_calls"] = 0

    projected["agent_profile"] = "context_only"
    projected["required_tools"] = []
    projected["allowed_tools"] = []
    projected["tool_requirement"] = "disabled"
    projected["public_context_profile"] = "graph_context"
    projected["profile_projection"] = {
        "kind": CONTEXT_ONLY_PROJECTION_KIND,
        "source_agent_profile": "tool_grounded",
        "effective_agent_profile": "context_only",
        "source_dataset_version": task.get("dataset_version"),
        "persisted_source": True,
    }
    return projected


def resolve_tasks_for_agent_profiles(
    tasks: Iterable[dict[str, Any]],
    requested_profiles: Iterable[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(tasks)
    requested = {str(value) for value in requested_profiles or []}
    unknown = requested - CHAIN_AGENT_PROFILE_NAMES
    if unknown:
        raise ValueError(f"unsupported chain agent profiles: {sorted(unknown)}")
    base = {
        "requested_profiles": sorted(requested),
        "fallback_applied": False,
        "projection_kind": None,
        "source_agent_profile": None,
        "effective_agent_profile": None,
        "projected_task_count": 0,
    }
    if not requested:
        return rows, {**base, "resolution": "unfiltered"}

    native = [row for row in rows if task_agent_profile(row) in requested]
    if native:
        return native, {**base, "resolution": "native"}

    if requested == {"context_only"}:
        sources = [
            row
            for row in rows
            if task_agent_profile(row) == "tool_grounded"
            and row.get("task_type") in CONTEXT_ONLY_FALLBACK_TASK_TYPES
        ]
        if sources:
            projected = [
                project_tool_grounded_to_context_only(row) for row in sources
            ]
            return projected, {
                **base,
                "resolution": "runtime_projection",
                "fallback_applied": True,
                "projection_kind": CONTEXT_ONLY_PROJECTION_KIND,
                "source_agent_profile": "tool_grounded",
                "effective_agent_profile": "context_only",
                "projected_task_count": len(projected),
            }

    return [], {**base, "resolution": "no_match"}
