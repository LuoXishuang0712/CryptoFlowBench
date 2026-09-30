from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from comp_exam.impl.algorithms import EDGE_CLASSIFIERS, PoisonEdgeClassifier
from comp_exam.impl.checkpoints import (
    load_detector_checkpoint,
    persist_detector_checkpoint,
)
from comp_exam.impl.models import CandidateEdge
from comp_exam.impl.paper_gnns import (
    I2BGNNEdgeClassifier,
    PEAEGNNEdgeClassifier,
    TokenScoutEdgeClassifier,
)


NODE_A = "0x" + "a" * 40
NODE_B = "0x" + "b" * 40
NODE_C = "0x" + "c" * 40


def training_fixture() -> tuple[dict, list[CandidateEdge]]:
    task = {
        "task_type": "edge_action_classification",
        "graph_context": {"current_node": NODE_A},
        "answer_value": {
            "candidate_actions": {
                "edge:follow": "follow",
                "edge:ignore": "ignore",
            }
        },
    }
    edges = [
        CandidateEdge(
            "edge:follow",
            NODE_A,
            NODE_B,
            100.0,
            metadata={
                "min_timestamp": 100,
                "edge_count": 2,
                "tx_count": 2,
            },
        ),
        CandidateEdge(
            "edge:ignore",
            NODE_A,
            NODE_C,
            1.0,
            metadata={
                "min_timestamp": 200,
                "edge_count": 1,
                "tx_count": 1,
            },
        ),
    ]
    return task, edges


class PaperGNNAdapterTests(unittest.TestCase):
    def test_all_paper_adapters_are_registered_and_return_probabilities(self) -> None:
        task, edges = training_fixture()
        expected = {
            "i2bgnn_edge": I2BGNNEdgeClassifier,
            "peae_gnn_edge": PEAEGNNEdgeClassifier,
            "tokenscout_edge": TokenScoutEdgeClassifier,
        }
        for name, classifier_type in expected.items():
            with self.subTest(name=name):
                self.assertIs(EDGE_CLASSIFIERS[name], classifier_type)
                model = classifier_type()
                summary = model.fit([(task, edges)])
                predictions = model.predict(
                    task=task,
                    edges=edges,
                    active_nodes=[NODE_A],
                )
                self.assertEqual(summary["sample_count"], 2)
                self.assertTrue(summary["public_features_only"])
                self.assertIn("adaptation", summary)
                self.assertEqual(len(predictions), 2)
                for prediction in predictions:
                    self.assertAlmostEqual(
                        sum(prediction.action_probabilities.values()), 1.0
                    )


class DetectorCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_and_digest_guard(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            output = persist_detector_checkpoint(
                PoisonEdgeClassifier(),
                root=Path(temporary),
                case_id="ronin_2022",
                role="edge_classifier",
                dataset_version="chain_qa.v3_2",
                training_cases=["wormhole_2022"],
                training_summary={"sample_count": 2},
                case_disjoint=True,
            )
            directory = Path(output["directory"])
            restored, manifest = load_detector_checkpoint(directory)
            self.assertIsInstance(restored, PoisonEdgeClassifier)
            self.assertTrue(manifest["case_disjoint"])
            self.assertEqual(manifest["training_summary"]["sample_count"], 2)
            self.assertEqual(len(manifest["model_sha256"]), 64)
            model_path = directory / "model.pkl"
            model_path.write_bytes(model_path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                load_detector_checkpoint(directory)

    def test_manifest_is_utf8_json_and_role_scoped(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            output = persist_detector_checkpoint(
                PoisonEdgeClassifier(),
                root=Path(temporary),
                case_id="case/a",
                role="edge_classifier",
                dataset_version="chain_qa.v3_2",
                training_cases=[],
                training_summary=None,
                case_disjoint=True,
            )
            manifest = json.loads(
                Path(output["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["role"], "edge_classifier")
            self.assertNotIn("/", Path(output["directory"]).parent.name)


if __name__ == "__main__":
    unittest.main()
