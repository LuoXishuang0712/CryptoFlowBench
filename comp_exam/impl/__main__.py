from __future__ import annotations

import argparse
import json
from typing import Sequence

from .algorithms import (
    EDGE_CLASSIFIERS,
    NODE_DETECTORS,
    create_edge_classifier,
    create_node_detector,
)
from .models import SUPPORTED_TASK_TYPES
from .repository import MongoExperimentRepository
from .runner import ComparisonExperimentRunner, ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Mongo-backed chain-QA comparison experiments."
    )
    parser.add_argument("case", help="case_id or case_dir in chain_qa_tasks")
    parser.add_argument("--dataset-version")
    parser.add_argument(
        "--task-type",
        action="append",
        choices=SUPPORTED_TASK_TYPES,
        dest="task_types",
        help="repeat to select task types; defaults to all supported types",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--edge-classifier", choices=sorted(EDGE_CLASSIFIERS), default="poison"
    )
    parser.add_argument(
        "--node-detector", choices=sorted(NODE_DETECTORS), default="max_outflow_source"
    )
    parser.add_argument(
        "--train-case",
        action="append",
        default=[],
        help="disjoint training case for learned edge/node classifiers; repeatable",
    )
    parser.add_argument("--follow-threshold", type=float, default=0.5)
    parser.add_argument("--seed-top-k", type=int, default=1)
    parser.add_argument("--seed-probability-threshold", type=float, default=0.5)
    parser.add_argument(
        "--taint-budget-ratio",
        type=float,
        default=0.5,
        help="shared visible-outflow taint budget for Haircut/FIFO/LIFO/TIHO",
    )
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-node-expansions", type=int, default=64)
    parser.add_argument("--max-inspected-edges", type=int, default=512)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--model-output-dir",
        default="detector_tool",
        help=(
            "persist loadable method checkpoints under "
            "<dir>/<case_id>/<role>-<method>/; use an empty value to disable"
        ),
    )
    parser.add_argument(
        "--run-id",
        help="optional externally assigned run id for resumable experiment orchestration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        case=args.case,
        dataset_version=args.dataset_version,
        task_types=tuple(args.task_types or SUPPORTED_TASK_TYPES),
        limit=args.limit,
        follow_threshold=args.follow_threshold,
        taint_budget_ratio=args.taint_budget_ratio,
        seed_top_k=args.seed_top_k,
        seed_probability_threshold=args.seed_probability_threshold,
        training_cases=tuple(args.train_case),
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        max_node_expansions=args.max_node_expansions,
        max_inspected_edges=args.max_inspected_edges,
        fail_fast=args.fail_fast,
        model_output_dir=args.model_output_dir or None,
    )
    runner = ComparisonExperimentRunner(
        repository=MongoExperimentRepository(),
        edge_classifier=create_edge_classifier(
            args.edge_classifier,
            taint_budget_ratio=args.taint_budget_ratio,
        ),
        node_detector=create_node_detector(args.node_detector),
        config=config,
    )
    print(json.dumps(runner.run(run_id=args.run_id), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
