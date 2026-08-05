from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
    count_model_state_elements,
)
from benchmark.manifest import load_manifest
from benchmark.runner import _load_submission_file, _make_model_spec
from benchmark.validation import validate_model_state, validate_optimizer, validate_submission
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    "baseline_adamw": 1,
    "recurrent": 1,
}


def sample_path(name: str) -> Path:
    return ROOT / "submissions" / name / "submission.py"


def load_sample(name: str) -> Submission:
    path = sample_path(name)
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SUBMISSION


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(ROOT / "benchmark/manifests/smoke_cpu.json")
        cls.model_spec = _make_model_spec(cls.manifest)
        cls.optimizer_spec = OptimizerSpec(training_time_seconds=1.0, device_type="cpu")

    def test_samples_are_exactly_one_standalone_python_file(self) -> None:
        for name in SAMPLES:
            directory = sample_path(name).parent
            self.assertEqual([path.name for path in directory.glob("*.py")], ["submission.py"])
            source = sample_path(name).read_text(encoding="utf-8")
            self.assertEqual(
                validate_submission_source("submission.py", source, 256 * 1024),
                "submission.py",
            )
            self.assertNotIn("from model", source)
            self.assertNotIn("from optim", source)

    def test_official_loader_accepts_each_file_in_isolation(self) -> None:
        for name in SAMPLES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                isolated = Path(directory) / "submission.py"
                shutil.copyfile(sample_path(name), isolated)
                submission = _load_submission_file(isolated)
                validate_submission(submission)

    def test_source_policy_rejects_wrong_names_and_private_imports(self) -> None:
        valid = "SUBMISSION = object()\n"
        with self.assertRaisesRegex(ValueError, "must be named submission.py"):
            validate_submission_source("architecture.py", valid, 256 * 1024)
        for source in (
            "import data.squaring_mod\n",
            "import model\n",
            "from optim.muon import Muon\n",
        ):
            with self.assertRaisesRegex(ValueError, "self-contained"):
                validate_submission_source("submission.py", source, 256 * 1024)

    def test_samples_use_expected_compute_and_deterministic_state(self) -> None:
        for name, loops in SAMPLES.items():
            with self.subTest(name=name):
                model = load_sample(name).build_model(self.model_spec)
                self.assertEqual(model.num_loops, loops)
                self.assertEqual(
                    validate_model_state(model, self.manifest.model_state, torch.device("cpu")),
                    201_600,
                )

    def test_basic_download_model_is_non_looping_native_adamw(self) -> None:
        submission = load_sample("baseline_adamw")
        model = submission.build_model(self.model_spec)
        bundle = submission.build_optimizer(model, self.optimizer_spec)
        self.assertEqual(model.num_loops, 1)
        self.assertIsInstance(bundle.optimizer, torch.optim.AdamW)

    def test_every_optimizer_covers_parameters_and_completes_update(self) -> None:
        for name in SAMPLES:
            with self.subTest(name=name):
                submission = load_sample(name)
                model = submission.build_model(self.model_spec)
                bundle = submission.build_optimizer(model, self.optimizer_spec)
                validate_optimizer(bundle, model, torch.device("cpu"))
                input_ids = torch.randint(0, self.model_spec.vocab_size, (1, 6))
                logits, _ = model(input_ids, attention_mask=torch.ones_like(input_ids).bool())
                logits.square().mean().backward()
                bundle.optimizer.step()
                validate_optimizer(bundle, model, torch.device("cpu"))

    def test_samples_support_language_model_contract(self) -> None:
        spec = ModelSpec(24, 12, maximum_model_state_elements=500_000_000)
        input_ids = torch.randint(0, spec.vocab_size, (2, 8))
        mask = torch.ones_like(input_ids).bool()
        for name in SAMPLES:
            with self.subTest(name=name):
                model = load_sample(name).build_model(spec)
                logits, auxiliary = model(input_ids, attention_mask=mask)
                self.assertEqual(tuple(logits.shape), (2, 8, spec.vocab_size))
                self.assertIsNone(auxiliary)
                self.assertLessEqual(count_model_state_elements(model), spec.maximum_model_state_elements)

    def test_optional_training_loss_is_differentiable(self) -> None:
        training_loss = lambda logits, labels, auxiliary: torch.nn.functional.cross_entropy(logits, labels.long())
        logits = torch.zeros(4, self.model_spec.vocab_size, requires_grad=True)
        loss = training_loss(logits, torch.tensor([0, 1, 0, 1]), None)
        loss.backward()
        self.assertEqual(loss.ndim, 0)
        self.assertIsNotNone(logits.grad)

    def test_public_state_assertion_counts_distinct_persistent_state(self) -> None:
        model = torch.nn.Module()
        shared = torch.nn.Parameter(torch.zeros(3))
        model.register_parameter("shared", shared)
        model.register_parameter("frozen", torch.nn.Parameter(torch.zeros(2), requires_grad=False))
        model.register_buffer("persistent", torch.zeros(4), persistent=True)
        model.register_buffer("scratch", torch.zeros(50), persistent=False)
        self.assertEqual(count_model_state_elements(model), 9)
        self.assertEqual(assert_model_state(model, ModelSpec(17, 10, 9)), 9)
        with self.assertRaisesRegex(AssertionError, "exceeds maximum"):
            assert_model_state(model, ModelSpec(17, 10, 8))

    def test_optional_loss_must_be_callable(self) -> None:
        submission = Submission(
            build_model=lambda spec: None,
            build_optimizer=lambda model, spec: None,
            training_loss=object(),
        )
        with self.assertRaisesRegex(TypeError, "training_loss must be callable"):
            validate_submission(submission)

    def test_optional_token_training_loss_validation(self) -> None:
        callback = lambda batch: batch.logits.sum()
        submission = Submission(
            build_model=lambda spec: None,
            build_optimizer=lambda model, spec: None,
            token_training_loss=callback,
        )
        validate_submission(submission)
        self.assertIs(submission.token_training_loss, callback)
        self.assertIn("valid_mask", TokenLossBatch.__dataclass_fields__)

        invalid = Submission(
            build_model=lambda spec: None,
            build_optimizer=lambda model, spec: None,
            token_training_loss=object(),
        )
        with self.assertRaisesRegex(
            TypeError, "token_training_loss must be callable"
        ):
            validate_submission(invalid)

        with self.assertRaisesRegex(ValueError, "cannot define both"):
            validate_submission(
                Submission(
                    build_model=lambda spec: None,
                    build_optimizer=lambda model, spec: None,
                    training_loss=callback,
                    token_training_loss=callback,
                )
            )

    def test_optional_training_controls_must_be_positive_integers(self) -> None:
        valid = Submission(
            build_model=lambda spec: None,
            build_optimizer=lambda model, spec: None,
            batch_size=128,
            max_steps=500,
            eval_batch_size=256,
        )
        validate_submission(valid)
        for field, value in (
            ("batch_size", 0),
            ("batch_size", True),
            ("eval_batch_size", 0),
            ("eval_batch_size", True),
            ("eval_batch_size", 1.5),
            ("max_steps", -1),
            ("max_steps", 1.5),
        ):
            submission = Submission(
                build_model=lambda spec: None,
                build_optimizer=lambda model, spec: None,
                **{field: value},
            )
            with self.subTest(field=field, value=value), self.assertRaisesRegex(
                ValueError, f"{field} must be a positive integer"
            ):
                validate_submission(submission)

    def test_eval_batch_size_does_not_reorder_existing_positional_fields(self) -> None:
        submission = Submission(
            lambda spec: None,
            lambda model, spec: None,
            None,
            128,
            500,
        )
        self.assertEqual(submission.batch_size, 128)
        self.assertEqual(submission.max_steps, 500)
        self.assertIsNone(submission.eval_batch_size)
        self.assertIsNone(submission.token_training_loss)

    def test_optimizer_bundle_multi_pass_controls_validate(self) -> None:
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        bundle = OptimizerBundle(
            optimizer,
            scheduler,
            backward_passes_per_step=2,
            between_backward_passes=lambda context: None,
            should_reuse_batch=lambda context: False,
        )

        validate_optimizer(bundle, model, torch.device("cpu"))
        self.assertIs(bundle.scheduler, scheduler)

        for value in (0, True, 1.5):
            with self.subTest(backward_passes_per_step=value), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                validate_optimizer(
                    OptimizerBundle(
                        optimizer,
                        backward_passes_per_step=value,
                    ),
                    model,
                    torch.device("cpu"),
                )

        for field in ("between_backward_passes", "should_reuse_batch"):
            with self.subTest(field=field), self.assertRaisesRegex(
                TypeError, "must be callable"
            ):
                validate_optimizer(
                    OptimizerBundle(optimizer, **{field: object()}),
                    model,
                    torch.device("cpu"),
                )

    def test_removed_packages_and_harness_are_absent(self) -> None:
        self.assertFalse((ROOT / "model").exists())
        self.assertFalse((ROOT / "optim").exists())
        self.assertFalse((ROOT / "train.py").exists())
        self.assertFalse((ROOT / "wandb_helpers.py").exists())
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn('"model"', pyproject)
        self.assertNotIn('"optim"', pyproject)


if __name__ == "__main__":
    unittest.main()
