from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from comp_exam.impl.algorithms import (
    FIFOEdgeClassifier,
    HaircutEdgeClassifier,
    LIFOEdgeClassifier,
    LogisticEdgeClassifier,
    LogisticNodeDetector,
    MLPEdgeClassifier,
    GCNEdgeClassifier,
    RandomForestEdgeClassifier,
    RandomForestNodeDetector,
    MaxOutflowSourceDetector,
    PoisonEdgeClassifier,
    TIHOEdgeClassifier,
    TRacerEdgeClassifier,
    XGBoostEdgeClassifier,
    XGBoostNodeDetector,
)
from comp_exam.impl.features import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, edge_feature_vector, node_feature_vector
from comp_exam.impl.models import CandidateEdge
from comp_exam.impl.repository import build_candidate_edges
from comp_exam.impl.runner import (
    ComparisonExperimentRunner,
    ExperimentConfig,
    probability_paths,
)


NODE_A = "0x" + "a" * 40
NODE_B = "0x" + "b" * 40
NODE_C = "0x" + "c" * 40
NODE_D = "0x" + "d" * 40


class FakeRepository:
    def __init__(
        self, tasks: list[dict[str, Any]], edges: dict[str, list[CandidateEdge]]
    ) -> None:
        self.tasks = tasks
        self.edges = edges
        self.runs: list[dict[str, Any]] = []
        self.run_updates: list[dict[str, Any]] = []
        self.stages: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []

    def load_tasks(self, **_: Any) -> list[dict[str, Any]]:
        return list(self.tasks)

    def load_candidate_edges(self, task: dict[str, Any]) -> list[CandidateEdge]:
        return list(self.edges[str(task["id"])])

    def create_run(self, document: dict[str, Any]) -> None:
        self.runs.append(document)

    def update_run(self, run_id: str, document: dict[str, Any]) -> None:
        self.run_updates.append({"run_id": run_id, **document})

    def save_stage(self, document: dict[str, Any]) -> None:
        self.stages.append(document)

    def save_task_result(self, document: dict[str, Any]) -> None:
        self.results.append(document)


def base_task(task_id: str, task_type: str) -> dict[str, Any]:
    return {
        "id": task_id,
        "case_id": "case_1",
        "case_dir": "01-case",
        "case_name": "Case 1",
        "dataset_version": "chain_qa.test",
        "task_type": task_type,
        "question": "test",
        "edge_summaries": [],
        "link_semantics": "chain_direct",
    }


class CandidateNormalizationTests(unittest.TestCase):
    def test_pair_candidate_aggregates_raw_edge_amounts(self) -> None:
        task = {
            "graph_context": {
                "candidate_edges": ["pair:1"],
                "candidate_edge_groups": {"pair:1": ["eth:1", "eth:2"]},
            },
            "edge_summaries": [
                {"candidate_id": "pair:1", "src": NODE_A, "dst": NODE_B}
            ],
        }
        edges = build_candidate_edges(
            task,
            [
                {"edge_id": "eth:1", "src": NODE_A, "dst": NODE_B, "amount": 2.5},
                {"edge_id": "eth:2", "src": NODE_A, "dst": NODE_B, "amount": "3.5"},
            ],
        )
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].candidate_id, "pair:1")
        self.assertEqual(edges[0].amount, 6.0)
        self.assertEqual(edges[0].raw_edge_ids, ("eth:1", "eth:2"))

    def test_candidate_metadata_retains_order_for_taint_rules(self) -> None:
        task = {
            "graph_context": {"candidate_edges": ["pair:1"]},
            "edge_summaries": [
                {"candidate_id": "pair:1", "src": NODE_A, "dst": NODE_B}
            ],
        }
        edge = build_candidate_edges(
            task,
            [
                {
                    "edge_id": "pair:1",
                    "src": NODE_A,
                    "dst": NODE_B,
                    "amount": 5,
                    "timestamp": 20,
                    "block_number": 200,
                }
            ],
        )[0]
        self.assertEqual(edge.metadata["min_timestamp"], 20.0)
        self.assertEqual(edge.metadata["min_block_number"], 200.0)

    def test_mixed_token_amounts_are_not_summed_across_units(self) -> None:
        task = {
            "graph_context": {
                "candidate_edges": ["pair:1"],
                "candidate_edge_groups": {"pair:1": ["token:1", "token:2"]},
            },
            "edge_summaries": [
                {"candidate_id": "pair:1", "src": NODE_A, "dst": NODE_B}
            ],
        }
        edge = build_candidate_edges(
            task,
            [
                {
                    "edge_id": "token:1",
                    "src": NODE_A,
                    "dst": NODE_B,
                    "amount": 100,
                    "token_address": NODE_C,
                },
                {
                    "edge_id": "token:2",
                    "src": NODE_A,
                    "dst": NODE_B,
                    "amount": 2,
                    "token_address": NODE_D,
                },
            ],
        )[0]
        self.assertEqual(edge.amount, 0.0)
        self.assertFalse(edge.metadata["amount_comparable"])


class TraditionalTaintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.early = CandidateEdge(
            "edge:early",
            NODE_A,
            NODE_B,
            6.0,
            metadata={"min_timestamp": 10},
        )
        self.late = CandidateEdge(
            "edge:late",
            NODE_A,
            NODE_C,
            4.0,
            metadata={"min_timestamp": 20},
        )
        self.edges = [self.early, self.late]

    def actions(self, classifier: Any) -> dict[str, str]:
        return {
            prediction.edge_id: prediction.action
            for prediction in classifier.predict(
                task={}, edges=self.edges, active_nodes=[NODE_A]
            )
        }

    def test_shared_budget_produces_distinct_fifo_lifo_tiho_behaviour(self) -> None:
        self.assertEqual(
            self.actions(FIFOEdgeClassifier(taint_budget_ratio=0.5)),
            {"edge:early": "follow", "edge:late": "ignore"},
        )
        self.assertEqual(
            self.actions(LIFOEdgeClassifier(taint_budget_ratio=0.5)),
            {"edge:early": "ignore", "edge:late": "follow"},
        )
        self.assertEqual(
            self.actions(TIHOEdgeClassifier(taint_budget_ratio=0.5)),
            {"edge:early": "follow", "edge:late": "ignore"},
        )

    def test_haircut_applies_same_taint_fraction_to_every_output(self) -> None:
        predictions = HaircutEdgeClassifier(taint_budget_ratio=0.5).predict(
            task={}, edges=self.edges, active_nodes=[NODE_A]
        )
        self.assertEqual([item.follow_probability for item in predictions], [0.5, 0.5])
        self.assertTrue(all(item.action == "follow" for item in predictions))

    def test_poison_is_a_deterministic_all_outputs_rule(self) -> None:
        predictions = PoisonEdgeClassifier().predict(
            task={}, edges=self.edges, active_nodes=[NODE_A]
        )
        self.assertTrue(all(item.follow_probability == 1.0 for item in predictions))

    def test_value_taint_budget_isolated_by_asset(self) -> None:
        edges = [
            CandidateEdge(
                "edge:usdc",
                NODE_A,
                NODE_B,
                1000.0,
                metadata={"assets": ["USDC"], "min_timestamp": 10},
            ),
            CandidateEdge(
                "edge:eth",
                NODE_A,
                NODE_C,
                1.0,
                metadata={"assets": ["ETH"], "min_timestamp": 20},
            ),
        ]
        predictions = FIFOEdgeClassifier(taint_budget_ratio=0.5).predict(
            task={}, edges=edges, active_nodes=[NODE_A]
        )
        self.assertEqual(
            [item.follow_probability for item in predictions], [0.5, 0.5]
        )

    def test_public_feature_shapes_are_stable(self) -> None:
        task = {"graph_context": {"candidate_seed_nodes": [NODE_A]}}
        self.assertEqual(
            len(edge_feature_vector(self.early, self.edges, [NODE_A])),
            len(EDGE_FEATURE_NAMES),
        )
        self.assertEqual(
            len(node_feature_vector(NODE_A, task, self.edges)),
            len(NODE_FEATURE_NAMES),
        )

    def test_node_ratio_is_finite_and_float32_safe(self) -> None:
        task = {"graph_context": {"candidate_seed_nodes": [NODE_A]}}
        edges = [
            CandidateEdge(NODE_A, NODE_C, "large", amount=1e100),
            CandidateEdge(NODE_B, NODE_A, "incoming", amount=0.0),
        ]
        ratio = node_feature_vector(NODE_A, task, edges)[8]
        self.assertTrue(math.isfinite(ratio))
        self.assertLessEqual(ratio, 3.4028235e38)

    def test_logistic_edge_and_node_models_fit_public_features(self) -> None:
        task = {
            "task_type": "subgraph_noise_filtering",
            "graph_context": {"candidate_seed_nodes": [NODE_A, NODE_B]},
            "answer_value": {
                "gold_seed_nodes": [NODE_A],
                "candidate_actions": {
                    "edge:a": "follow",
                    "edge:b": "ignore",
                },
            },
        }
        edges = [
            CandidateEdge("edge:a", NODE_A, NODE_C, 100.0),
            CandidateEdge("edge:b", NODE_B, NODE_D, 1.0),
        ]
        for edge_class, node_class in (
            (LogisticEdgeClassifier, LogisticNodeDetector),
            (RandomForestEdgeClassifier, RandomForestNodeDetector),
            (XGBoostEdgeClassifier, XGBoostNodeDetector),
        ):
            with self.subTest(backend=edge_class.backend):
                edge_model = edge_class()
                node_model = node_class()
                edge_training = edge_model.fit([(task, edges)])
                node_training = node_model.fit([(task, edges)])
                edge_predictions = edge_model.predict(
                    task=task, edges=edges, active_nodes=[NODE_A]
                )
                node_predictions = node_model.predict(task=task, edges=edges)
                self.assertEqual(edge_training["sample_count"], 2)
                self.assertEqual(node_training["sample_count"], 2)
                self.assertEqual(len(edge_predictions), 2)
                self.assertAlmostEqual(
                    sum(edge_predictions[0].action_probabilities.values()), 1.0
                )
                self.assertEqual(
                    {item.node_id for item in node_predictions}, {NODE_A, NODE_B}
                )

    def test_weber_mlp_and_gcn_edge_adapters_fit_public_features(self) -> None:
        task = {
            "task_type": "edge_action_classification",
            "graph_context": {"current_node": NODE_A},
            "answer_value": {
                "candidate_actions": {
                    "edge:a": "follow",
                    "edge:b": "ignore",
                }
            },
        }
        edges = [
            CandidateEdge("edge:a", NODE_A, NODE_C, 100.0),
            CandidateEdge("edge:b", NODE_A, NODE_D, 1.0),
        ]
        for edge_class in (MLPEdgeClassifier, GCNEdgeClassifier):
            with self.subTest(classifier=edge_class.__name__):
                model = edge_class()
                training = model.fit([(task, edges)])
                predictions = model.predict(
                    task=task, edges=edges, active_nodes=[NODE_A]
                )
                self.assertEqual(training["sample_count"], 2)
                self.assertEqual(len(predictions), 2)
                for prediction in predictions:
                    self.assertAlmostEqual(
                        sum(prediction.action_probabilities.values()), 1.0
                    )

    def test_tracer_reference_adapter_ranks_weighted_temporal_path(self) -> None:
        def metadata(timestamp: int, tx_hash: str) -> dict[str, Any]:
            return {
                "min_timestamp": timestamp,
                "tx_hash": tx_hash,
                "assets": ["ETH"],
                "token_addresses": [],
            }

        graph = [
            CandidateEdge("edge:ab", NODE_A, NODE_B, 100.0, metadata=metadata(1, "h1")),
            CandidateEdge("edge:bc", NODE_B, NODE_C, 90.0, metadata=metadata(2, "h2")),
            CandidateEdge("edge:ad", NODE_A, NODE_D, 1.0, metadata=metadata(1, "h3")),
        ]
        model = TRacerEdgeClassifier()
        model.prepare_case(case_id="case", dataset_version="chain_qa.v3_2", edges=graph)
        predictions = model.predict(
            task={"graph_context": {"rollout_budget": {"max_node_expansions": 16}}},
            edges=[graph[0], graph[2]],
            active_nodes=[NODE_A],
        )
        by_id = {prediction.edge_id: prediction for prediction in predictions}
        self.assertGreater(
            by_id["edge:ab"].follow_probability,
            by_id["edge:ad"].follow_probability,
        )
        self.assertEqual(by_id["edge:ab"].action, "follow")
        self.assertEqual(model.alpha, 0.15)
        self.assertEqual(model.beta, 0.7)
        self.assertEqual(model.epsilon, 1e-3)
        self.assertIn("BlockchainSpider", by_id["edge:ab"].rationale)


class RunnerTests(unittest.TestCase):
    def test_training_and_target_case_must_be_disjoint(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig(case="case-a", training_cases=("case-a",))

    def test_runner_persists_edge_and_node_checkpoints_by_canonical_case_id(self) -> None:
        task = base_task("task:eac-checkpoint", "edge_action_classification")
        task["graph_context"] = {
            "current_node": NODE_A,
            "candidate_edges": ["edge:ab"],
        }
        task["answer_value"] = {
            "candidate_actions": {"edge:ab": "follow"},
            "correct_follow_edges": ["edge:ab"],
        }
        repository = FakeRepository(
            [task],
            {"task:eac-checkpoint": [CandidateEdge("edge:ab", NODE_A, NODE_B, 1.0)]},
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            result = ComparisonExperimentRunner(
                repository=repository,
                edge_classifier=PoisonEdgeClassifier(),
                node_detector=MaxOutflowSourceDetector(),
                config=ExperimentConfig(
                    case="01-case",
                    model_output_dir=temporary,
                ),
            ).run(run_id="checkpoint-test")
            self.assertEqual(
                set(result["checkpoints"]), {"edge_classifier", "node_detector"}
            )
            case_directory = Path(temporary) / "case_1"
            self.assertTrue(
                (case_directory / "edge_classifier-poison" / "manifest.json").is_file()
            )
            self.assertTrue(
                (
                    case_directory
                    / "node_detector-max_outflow_source"
                    / "model.pkl"
                ).is_file()
            )

    def test_probability_paths_rolls_out_multiple_steps_with_budgets(self) -> None:
        paths, usage = probability_paths(
            {"graph_context": {"rollout_budget": {"max_depth": 3, "beam_width": 2}}},
            [
                CandidateEdge("edge:ab", NODE_A, NODE_B, 100.0),
                CandidateEdge("edge:bc", NODE_B, NODE_C, 50.0),
            ],
            PoisonEdgeClassifier(),
            [NODE_A],
            follow_threshold=0.5,
            config=ExperimentConfig(case="case"),
        )
        self.assertEqual(paths[0].edge_ids, ("edge:ab", "edge:bc"))
        self.assertEqual(paths[0].terminal_reason, "dead_end")
        self.assertEqual(usage["node_expansions"], 3)

    def test_detector_uses_largest_single_edge_not_aggregate_outflow(self) -> None:
        detector = MaxOutflowSourceDetector()
        predictions = detector.predict(
            task={},
            edges=[
                CandidateEdge("edge:a1", NODE_A, NODE_C, 60.0),
                CandidateEdge("edge:a2", NODE_A, NODE_D, 60.0),
                CandidateEdge("edge:b1", NODE_B, NODE_C, 100.0),
            ],
        )
        self.assertEqual(predictions[0].node_id, NODE_B)
        self.assertEqual(predictions[0].score, 100.0)

    def test_detector_only_ranks_declared_candidate_seed_nodes(self) -> None:
        predictions = MaxOutflowSourceDetector().predict(
            task={"graph_context": {"candidate_seed_nodes": [NODE_B]}},
            edges=[
                CandidateEdge("edge:a", NODE_A, NODE_C, 1000.0),
                CandidateEdge("edge:b", NODE_B, NODE_C, 1.0),
            ],
        )
        self.assertEqual([item.node_id for item in predictions], [NODE_B])

    def test_dispatches_three_tasks_and_persists_every_stage(self) -> None:
        eac = base_task("task:eac", "edge_action_classification")
        eac["graph_context"] = {
            "current_node": NODE_A,
            "candidate_edges": ["pair:ab", "pair:ac"],
            "follow_threshold": 0.5,
        }
        eac["answer_value"] = {
            "candidate_actions": {"pair:ab": "follow", "pair:ac": "ignore"},
            "correct_follow_edges": ["pair:ab"],
        }

        pc = base_task("task:pc", "path_completion")
        pc["graph_context"] = {
            "current_node": NODE_B,
            "path_prefix_edges": ["edge:prefix"],
            "candidate_edges": ["edge:bc", "edge:bd"],
        }
        pc["answer_value"] = {
            "path_prefix_edges": ["edge:prefix"],
            "missing_next_edges": ["edge:bc"],
        }

        snf = base_task("task:snf", "subgraph_noise_filtering")
        snf["graph_context"] = {
            "current_node": NODE_A,
            "candidate_edges": ["edge:ab", "edge:cd"],
        }
        snf["answer_value"] = {"edge:ab": "follow", "edge:cd": "ignore"}

        edges = {
            "task:eac": [
                CandidateEdge("pair:ab", NODE_A, NODE_B, 100.0),
                CandidateEdge("pair:ac", NODE_A, NODE_C, 10.0),
            ],
            "task:pc": [
                CandidateEdge("edge:bc", NODE_B, NODE_C, 50.0),
                CandidateEdge("edge:bd", NODE_B, NODE_D, 5.0),
            ],
            "task:snf": [
                CandidateEdge("edge:ab", NODE_A, NODE_B, 100.0),
                CandidateEdge("edge:cd", NODE_C, NODE_D, 10.0),
            ],
        }
        repository = FakeRepository([eac, pc, snf], edges)
        runner = ComparisonExperimentRunner(
            repository=repository,
            edge_classifier=PoisonEdgeClassifier(),
            node_detector=MaxOutflowSourceDetector(),
            config=ExperimentConfig(case="01-case"),
        )
        final = runner.run()

        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["progress"], {"completed": 3, "failed": 0, "total": 3})
        self.assertEqual(len(repository.results), 3)
        self.assertEqual(len(repository.stages), 15)
        self.assertEqual(
            {stage["stage"] for stage in repository.stages},
            {
                "task_loaded",
                "node_detection",
                "edge_classification",
                "path_generation",
                "evaluation",
            },
        )

        snf_result = next(row for row in repository.results if row["task_type"] == "subgraph_noise_filtering")
        self.assertEqual(snf_result["algorithm_output"]["selected_seed_nodes"], [NODE_A])
        self.assertEqual(snf_result["algorithm_output"]["selected_follow_edges"], ["edge:ab"])
        self.assertTrue(snf_result["evaluation"]["metrics"]["seed_hit_at_1"])
        self.assertEqual(snf_result["evaluation"]["metrics"]["path_recall"], 1.0)

        pc_result = next(row for row in repository.results if row["task_type"] == "path_completion")
        self.assertEqual(pc_result["algorithm_output"]["paths"][0]["edge_ids"][0], "edge:prefix")
        self.assertEqual(pc_result["evaluation"]["metrics"]["path_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
