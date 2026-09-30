from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from .config import ChainQAConfig
from .export import write_bundle, write_mongo
from .index_client import EthIndexClient
from .io import write_json
from .pipeline import (
    build_candidate_sets,
    build_subgraph,
    discover_cases,
    enrich_action_outgoing,
    generate_tasks,
    label_subgraph,
    load_case,
)
from .transaction_existence import (
    build_transaction_existence_tasks,
)
from .validate import validate_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build path-oriented chain QA tasks from local Ethereum index subgraphs."
    )
    parser.add_argument("case", nargs="?", default="all", help="Case dir under summarized/, or all.")
    parser.add_argument("--eth-index-url", default=None)
    parser.add_argument("--k-hop", type=int, default=None)
    parser.add_argument("--direction", choices=["in", "out", "both"], default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--neighbor-limit", type=int, default=None)
    parser.add_argument("--neg-ratio", type=int, default=None)
    parser.add_argument("--transaction-neg-ratio", type=int, default=None)
    parser.add_argument("--transaction-query-limit", type=int, default=None)
    parser.add_argument(
        "--skip-dte",
        action="store_true",
        help=(
            "Skip direct_transaction_existence generation and oracle queries. "
            "Use a new dataset version if an existing version must retain DTE."
        ),
    )
    parser.add_argument(
        "--tracing-agent-profile",
        choices=["tool_grounded", "context_only"],
        default=None,
        help=(
            "Profile for EAC/PC/SNF tasks. Defaults to tool_grounded; use "
            "context_only only for a separately versioned ablation snapshot."
        ),
    )
    parser.add_argument("--request-timeout", type=int, default=None)
    parser.add_argument("--connect-timeout", type=int, default=None)
    parser.add_argument("--request-retries", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-mongo", action="store_true", help="Skip Mongo writes.")
    parser.add_argument("--strict-mongo", action="store_true", help="Fail if Mongo write fails.")
    parser.add_argument("--no-health-check", action="store_true")
    return parser


def build_optional_transaction_existence_tasks(
    case: dict[str, Any],
    edges: list[Any],
    config: ChainQAConfig,
    client: EthIndexClient,
) -> tuple[list[Any], dict[str, Any]]:
    if config.skip_dte:
        return [], {
            "ok": True,
            "skipped": True,
            "reason": "disabled_by_skip_dte",
            "task_count": 0,
            "query_count": 0,
            "errors": [],
        }
    return build_transaction_existence_tasks(case, edges, config, client)


def process_case(case_dir: str, config: ChainQAConfig, *, write_db: bool, strict_mongo: bool) -> tuple[str, dict]:
    case_started = time.perf_counter()
    timings: dict[str, float] = {}
    phase_started = case_started
    case = load_case(case_dir)
    timings["load_case"] = time.perf_counter() - phase_started
    client = EthIndexClient(
        config.eth_index_url,
        timeout=config.request_timeout,
        connect_timeout=config.connect_timeout,
        max_retries=config.request_retries,
        retry_backoff=config.retry_backoff,
    )
    try:
        phase_started = time.perf_counter()
        seeds, edges = build_subgraph(case, config, client)
        timings["build_subgraph"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        labels = label_subgraph(case, seeds, edges, config.neg_ratio)
        edges = enrich_action_outgoing(case, edges, labels, config, client)
        labels = label_subgraph(case, seeds, edges, config.neg_ratio)
        timings["label_and_enrich"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        candidates = build_candidate_sets(
            case,
            edges,
            labels,
            max_candidates=config.max_candidates,
            neg_ratio=config.neg_ratio,
        )
        timings["build_candidates"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        tasks, transaction_validation = build_optional_transaction_existence_tasks(
            case, edges, config, client
        )
        timings["transaction_existence"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        tasks.extend(
            generate_tasks(
                case,
                edges,
                labels,
                candidates,
                config.dataset_version,
                agent_profile=config.tracing_agent_profile,
            )
        )
        timings["generate_context_tasks"] = time.perf_counter() - phase_started
    finally:
        client.close()
    phase_started = time.perf_counter()
    validation = validate_bundle(
        edges, labels, candidates, tasks, k_hop=config.k_hop
    )
    validation["transaction_existence"] = transaction_validation
    if not transaction_validation["ok"]:
        validation["ok"] = False
        validation["errors"].extend(transaction_validation["errors"])
    timings["validate"] = time.perf_counter() - phase_started
    out_dir = config.output_dir / case_dir
    phase_started = time.perf_counter()
    manifest = write_bundle(
        out_dir,
        case=case,
        config=config,
        seeds=seeds,
        edges=edges,
        labels=labels,
        candidates=candidates,
        tasks=tasks,
    )
    timings["write_local"] = time.perf_counter() - phase_started
    manifest["validation"] = validation
    manifest["timings_seconds"] = {
        name: round(duration, 3) for name, duration in timings.items()
    }
    mongo_error = None
    if write_db:
        phase_started = time.perf_counter()
        mongo_error = write_mongo(
            case=case,
            manifest=manifest,
            seeds=seeds,
            edges=edges,
            labels=labels,
            candidates=candidates,
            tasks=tasks,
            strict=strict_mongo,
        )
        timings["write_mongo"] = time.perf_counter() - phase_started
        if mongo_error:
            manifest["mongo_error"] = mongo_error
    timings["total"] = time.perf_counter() - case_started
    manifest["timings_seconds"] = {
        name: round(duration, 3) for name, duration in timings.items()
    }
    write_json(out_dir / "manifest.json", manifest)
    return case_dir, {"manifest": manifest, "mongo_error": mongo_error}


def main() -> int:
    args = build_parser().parse_args()
    config = ChainQAConfig.from_env(
        eth_index_url=args.eth_index_url,
        k_hop=args.k_hop,
        direction=args.direction,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        neighbor_limit=args.neighbor_limit,
        neg_ratio=args.neg_ratio,
        transaction_neg_ratio=args.transaction_neg_ratio,
        transaction_query_limit=args.transaction_query_limit,
        skip_dte=args.skip_dte,
        tracing_agent_profile=args.tracing_agent_profile,
        request_timeout=args.request_timeout,
        connect_timeout=args.connect_timeout,
        request_retries=args.request_retries,
        show_progress=False if args.no_progress else None,
        max_candidates=args.max_candidates,
    )
    if args.output_dir:
        from pathlib import Path

        config = ChainQAConfig(**{**config.__dict__, "output_dir": Path(args.output_dir)})
    with EthIndexClient(
        config.eth_index_url,
        timeout=config.request_timeout,
        connect_timeout=config.connect_timeout,
        max_retries=config.request_retries,
        retry_backoff=config.retry_backoff,
    ) as client:
        if not args.no_health_check:
            health = client.health()
            print(f"eth_index health: {health}")
    cases = discover_cases() if args.case == "all" else [args.case]
    if not cases:
        raise SystemExit("no cases found")
    ok = True
    run_started = time.perf_counter()
    case_timings: list[tuple[str, float, str]] = []
    for case_dir in cases:
        case_started = time.perf_counter()
        try:
            _, result = process_case(
                case_dir,
                config,
                write_db=not args.no_mongo,
                strict_mongo=args.strict_mongo,
            )
            manifest = result["manifest"]
            counts = manifest["counts"]
            task_counts = ", ".join(
                f"{task_type}={count}"
                for task_type, count in sorted(manifest.get("task_type_counts", {}).items())
            )
            print(
                f"{case_dir}: edges={counts['subgraph_edges']} labels={counts['labels']} "
                f"candidates={counts['candidate_sets']} tasks={counts['qa_tasks']} "
                f"validation_ok={manifest['validation']['ok']} "
                f"elapsed={manifest['timings_seconds']['total']:.2f}s"
            )
            print(f"{case_dir}: task_type_counts: {task_counts}")
            phase_timings = ", ".join(
                f"{name}={duration:.2f}s"
                for name, duration in manifest["timings_seconds"].items()
                if name != "total"
            )
            print(f"{case_dir}: timings: {phase_timings}")
            case_timings.append(
                (case_dir, manifest["timings_seconds"]["total"], "ok")
            )
            if result.get("mongo_error"):
                print(f"{case_dir}: mongo write warning: {result['mongo_error']}")
            if not manifest["validation"]["ok"]:
                ok = False
        except KeyboardInterrupt:
            ok = False
            elapsed = time.perf_counter() - case_started
            case_timings.append((case_dir, elapsed, "interrupted"))
            print(
                f"{case_dir}: interrupted after {elapsed:.2f}s",
                file=sys.stderr,
            )
            break
        except Exception as exc:
            ok = False
            elapsed = time.perf_counter() - case_started
            case_timings.append((case_dir, elapsed, "failed"))
            print(
                f"{case_dir}: failed after {elapsed:.2f}s: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    total_elapsed = time.perf_counter() - run_started
    print("case timing summary:")
    for case_dir, elapsed, status in case_timings:
        print(f"  {case_dir}: {elapsed:.2f}s ({status})")
    print(f"total elapsed: {total_elapsed:.2f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
