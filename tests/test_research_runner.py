from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from benchmark import ModelSpec
from benchmark.runner import _load_submission_file
from research.runner import (
    _calculated_workload,
    materialize_submission,
    run_experiment,
    verify_run_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submissions" / "recurrent" / "submission.py"
SMOKE_MANIFEST = ROOT / "benchmark" / "manifests" / "smoke_cpu.json"


class ResearchRunnerTests(unittest.TestCase):
    def test_materialized_variant_executes_through_official_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.py"
            materialize_submission(SUBMISSION, "state_k4", path)
            submission = _load_submission_file(path)
            model = submission.build_model(
                type(
                    "Spec",
                    (),
                    {
                        "vocab_size": 17,
                        "max_seq_len": 12,
                        "maximum_model_state_elements": 500_000_000,
                    },
                )()
            )
            self.assertEqual(model.num_loops, 4)
            self.assertEqual(model.state_token_count, 4)

    def test_causal_state_materialization_is_numerically_identical(self) -> None:
        module_spec = importlib.util.spec_from_file_location(
            "test_causal_state_research_source",
            SUBMISSION,
        )
        assert module_spec is not None and module_spec.loader is not None
        source_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(source_module)
        model_spec = ModelSpec(
            vocab_size=17,
            max_seq_len=12,
            maximum_model_state_elements=500_000_000,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.py"
            materialize_submission(SUBMISSION, "causal_state_contract", path)
            official_submission = _load_submission_file(path)
            torch.manual_seed(47)
            research_model = source_module.build_model_from_config(
                model_spec,
                source_module.EXPERIMENTS["causal_state_contract"].model,
            ).eval()
            torch.manual_seed(47)
            official_model = official_submission.build_model(model_spec).eval()
            self.assertEqual(
                research_model.state_dict().keys(),
                official_model.state_dict().keys(),
            )
            for name, expected in research_model.state_dict().items():
                torch.testing.assert_close(
                    official_model.state_dict()[name],
                    expected,
                )

            input_ids = torch.tensor([[2, 10, 9, 10, 3, 11, 9, 4, 10]])
            mask = torch.ones_like(input_ids, dtype=torch.bool)
            with torch.no_grad():
                expected, _ = research_model(input_ids, attention_mask=mask)
                actual, _ = official_model(input_ids, attention_mask=mask)
            torch.testing.assert_close(actual, expected)

    def test_causal_state_workload_records_its_real_cell_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.py"
            materialize_submission(SUBMISSION, "causal_state_contract", path)
            workload = _calculated_workload(path, SMOKE_MANIFEST)
        self.assertEqual(workload["inputs"]["architecture"], "causal_state")
        self.assertEqual(workload["inputs"]["scratch_tokens"], 2)
        self.assertEqual(workload["inputs"]["digit_slots"], 16)
        self.assertEqual(workload["inputs"]["training_recurrent_loops"], 4)
        self.assertEqual(workload["flops"]["attention_forward"], 0)
        self.assertGreater(workload["flops"]["forward"], 0)

    def test_causal_state_no_scratch_is_a_single_axis_profile_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scratch_path = Path(directory) / "scratch.py"
            no_scratch_path = Path(directory) / "no_scratch.py"
            materialize_submission(
                SUBMISSION,
                "causal_state_contract",
                scratch_path,
            )
            materialize_submission(
                SUBMISSION,
                "causal_state_no_scratch_profile",
                no_scratch_path,
            )
            scratch = _calculated_workload(scratch_path, SMOKE_MANIFEST)
            no_scratch = _calculated_workload(no_scratch_path, SMOKE_MANIFEST)
        self.assertEqual(scratch["inputs"]["scratch_tokens"], 2)
        self.assertEqual(no_scratch["inputs"]["scratch_tokens"], 0)
        self.assertEqual(
            scratch["inputs"] | {"scratch_tokens": 0},
            no_scratch["inputs"],
        )
        self.assertGreater(
            scratch["flops"]["forward"],
            no_scratch["flops"]["forward"],
        )

    def test_causal_grid_materializes_and_records_cross_digit_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.py"
            materialize_submission(SUBMISSION, "causal_grid_contract", path)
            workload = _calculated_workload(path, SMOKE_MANIFEST)
            submission = _load_submission_file(path)
            model = submission.build_model(
                ModelSpec(
                    vocab_size=17,
                    max_seq_len=12,
                    maximum_model_state_elements=500_000_000,
                )
            )
        self.assertEqual(workload["inputs"]["architecture"], "causal_grid")
        self.assertEqual(workload["inputs"]["scratch_tokens"], 0)
        self.assertEqual(workload["inputs"]["layers_per_loop"], 2)
        self.assertEqual(workload["flops"]["attention_forward"], 0)
        self.assertEqual(len(model.step.layers), 2)
        self.assertGreater(workload["flops"]["forward"], 0)

    def test_causal_dcgru_materializes_with_directional_workload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.py"
            materialize_submission(SUBMISSION, "causal_dcgru_contract", path)
            workload = _calculated_workload(path, SMOKE_MANIFEST)
            submission = _load_submission_file(path)
            model = submission.build_model(
                ModelSpec(
                    vocab_size=17,
                    max_seq_len=12,
                    maximum_model_state_elements=500_000_000,
                )
            )
        self.assertEqual(workload["inputs"]["architecture"], "causal_dcgru")
        self.assertEqual(workload["inputs"]["work_width"], 16)
        self.assertEqual(workload["inputs"]["training_recurrent_loops"], 3)
        self.assertEqual(workload["inputs"]["layers_per_loop"], 4)
        self.assertEqual(workload["flops"]["attention_forward"], 0)
        self.assertEqual(model.step.microsteps, 4)
        self.assertGreater(workload["flops"]["forward"], 0)

    def test_pair_dcgru_adds_only_the_global_pair_workload_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dcgru_path = Path(directory) / "dcgru.py"
            pair_path = Path(directory) / "pair.py"
            materialize_submission(
                SUBMISSION,
                "causal_dcgru_contract",
                dcgru_path,
            )
            materialize_submission(
                SUBMISSION,
                "causal_pair_dcgru_contract",
                pair_path,
            )
            dcgru = _calculated_workload(dcgru_path, SMOKE_MANIFEST)
            pair = _calculated_workload(pair_path, SMOKE_MANIFEST)
            submission = _load_submission_file(pair_path)
            model = submission.build_model(
                ModelSpec(
                    vocab_size=17,
                    max_seq_len=12,
                    maximum_model_state_elements=500_000_000,
                )
            )

        self.assertEqual(pair["inputs"]["architecture"], "causal_pair_dcgru")
        self.assertEqual(
            dcgru["inputs"]
            | {
                "architecture": "causal_pair_dcgru",
                "pair_routing": "learned",
            },
            pair["inputs"],
        )
        self.assertGreater(
            pair["flops"]["forward"],
            dcgru["flops"]["forward"],
        )
        self.assertEqual(tuple(model.step.pair_interaction.route_logits.shape), (16, 16, 16))

    def test_uniform_pair_control_removes_only_route_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            learned_path = Path(directory) / "learned.py"
            uniform_path = Path(directory) / "uniform.py"
            materialize_submission(
                SUBMISSION,
                "causal_pair_dcgru_contract",
                learned_path,
            )
            materialize_submission(
                SUBMISSION,
                "causal_uniform_pair_dcgru_contract",
                uniform_path,
            )
            learned = _calculated_workload(learned_path, SMOKE_MANIFEST)
            uniform = _calculated_workload(uniform_path, SMOKE_MANIFEST)

        self.assertEqual(learned["flops"], uniform["flops"])
        self.assertEqual(
            learned["inputs"] | {"pair_routing": "uniform"},
            uniform["inputs"],
        )
        self.assertEqual(
            learned["state"]["model_elements"]
            - uniform["state"]["model_elements"],
            16**3,
        )

    def test_smoke_run_writes_immutable_receipt_and_exact_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(
                submission_path=SUBMISSION,
                experiment="baseline",
                manifest_path=SMOKE_MANIFEST,
                artifact_root=Path(directory),
            )
            receipt = json.loads((run_dir / "receipt.json").read_text())
            result = json.loads((run_dir / "result.json").read_text())
            source = (run_dir / "submission.py").read_bytes()
            self.assertEqual(receipt["status"], "succeeded")
            self.assertEqual(receipt["experiment"], "baseline")
            self.assertEqual(
                receipt["artifacts"]["submission.py"]["sha256"],
                hashlib.sha256(source).hexdigest(),
            )
            self.assertEqual(result["manifest"], "squaring-mod-cpu-smoke")
            self.assertTrue(verify_run_artifacts(run_dir))
            self.assertNotIn("environment_variables", receipt)
            self.assertEqual(receipt["gpu_sampling"]["status"], "disabled")
            self.assertTrue((run_dir / "gpu_samples.csv").is_file())

    def test_failed_run_cannot_be_verified_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "research.runner.run_submission_file",
                    side_effect=RuntimeError("injected runner failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected runner failure"),
            ):
                run_experiment(
                    submission_path=SUBMISSION,
                    experiment="baseline",
                    manifest_path=SMOKE_MANIFEST,
                    artifact_root=Path(directory),
                )
            run_dirs = list(Path(directory).iterdir())
            self.assertEqual(len(run_dirs), 1)
            receipt = json.loads((run_dirs[0] / "receipt.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertFalse(verify_run_artifacts(run_dirs[0]))

    def test_checkpoint_capture_preserves_a_verifiable_final_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(
                submission_path=SUBMISSION,
                experiment="baseline",
                manifest_path=SMOKE_MANIFEST,
                artifact_root=Path(directory),
                save_checkpoint=True,
            )
            checkpoint_path = run_dir / "checkpoint-seed-74.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(checkpoint["seed"], 74)
            self.assertTrue(checkpoint["state_dict"])
            self.assertTrue(verify_run_artifacts(run_dir))

    def test_preflight_failure_writes_failed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "research.runner._calculated_workload",
                    side_effect=RuntimeError("injected preflight failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected preflight failure"),
            ):
                run_experiment(
                    submission_path=SUBMISSION,
                    experiment="baseline",
                    manifest_path=SMOKE_MANIFEST,
                    artifact_root=Path(directory),
                )
            run_dir = next(Path(directory).iterdir())
            receipt = json.loads((run_dir / "receipt.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["failure"]["type"], "RuntimeError")
            self.assertFalse(verify_run_artifacts(run_dir))

    def test_checksum_verification_discriminates_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = run_experiment(
                submission_path=SUBMISSION,
                experiment="baseline",
                manifest_path=SMOKE_MANIFEST,
                artifact_root=Path(directory),
            )
            self.assertTrue(verify_run_artifacts(run_dir))
            with (run_dir / "submission.py").open("a", encoding="utf-8") as file:
                file.write("\n# mutation\n")
            self.assertFalse(verify_run_artifacts(run_dir))


if __name__ == "__main__":
    unittest.main()
