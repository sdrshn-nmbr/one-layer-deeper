from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from benchmark.runner import _load_submission_file
from benchmark.validation import validate_optimizer, validate_submission
from research.interventions import (
    PredictionMeasurement,
    _causal_state_interventions,
    _compare_measurement,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_PATH = ROOT / "submissions" / "recurrent" / "submission.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "test_recurrent_submission_module", SUBMISSION_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecurrentSubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.spec = ModelSpec(
            vocab_size=17,
            max_seq_len=12,
            maximum_model_state_elements=500_000_000,
        )

    def build(self, name: str):
        experiment = self.module.EXPERIMENTS[name]
        return self.module.build_model_from_config(self.spec, experiment.model)

    def test_submission_loads_in_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory) / "submission.py"
            shutil.copyfile(SUBMISSION_PATH, isolated)
            submission = _load_submission_file(isolated)
            validate_submission(submission)

    def test_selected_baseline_is_numerically_identical_to_reference(self) -> None:
        reference_path = (
            ROOT / "submissions" / "baseline_adamw" / "submission.py"
        )
        reference_spec = importlib.util.spec_from_file_location(
            "test_reference_baseline", reference_path
        )
        assert reference_spec is not None and reference_spec.loader is not None
        reference_module = importlib.util.module_from_spec(reference_spec)
        reference_spec.loader.exec_module(reference_module)

        torch.manual_seed(11)
        reference = reference_module.Model(self.spec)
        torch.manual_seed(11)
        candidate = self.build("baseline")
        self.assertEqual(reference.state_dict().keys(), candidate.state_dict().keys())
        for name, expected in reference.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[name], expected)

        input_ids = torch.randint(0, self.spec.vocab_size, (2, 10))
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        with torch.no_grad():
            expected, _ = reference(input_ids, attention_mask=mask)
            actual, _ = candidate(input_ids, attention_mask=mask)
        torch.testing.assert_close(actual, expected)

    def test_experiment_configs_are_distinct_single_axis_steps(self) -> None:
        configs = {
            name: asdict(experiment.model)
            for name, experiment in self.module.EXPERIMENTS.items()
        }
        self.assertEqual(configs["baseline"]["num_loops"], 1)
        self.assertEqual(configs["scaled_init"]["initialization_std"], 0.02)
        self.assertEqual(configs["scaled_init"]["d_model"], 128)
        self.assertEqual(configs["wide_scaled"]["d_model"], 512)
        self.assertEqual(configs["wide_scaled"]["initialization_std"], 0.02)
        self.assertEqual(
            self.module.EXPERIMENTS["wide_scaled_foreach"].optimizer.implementation,
            "foreach",
        )
        self.assertEqual(
            self.module.EXPERIMENTS["wide_scaled_fused"].optimizer.implementation,
            "fused",
        )
        self.assertTrue(
            self.module.EXPERIMENTS["wide_scaled_fused_compile"].compile_model
        )
        self.assertEqual(
            self.module.EXPERIMENTS[
                "wide_scaled_reuse2"
            ].optimizer.max_batch_uses,
            2,
        )
        self.assertEqual(
            self.module.EXPERIMENTS[
                "wide_scaled_reuse4"
            ].optimizer.max_batch_uses,
            4,
        )
        self.assertEqual(
            self.module.EXPERIMENTS[
                "wide_scaled_reuse8"
            ].optimizer.max_batch_uses,
            8,
        )
        self.assertEqual(
            self.module.EXPERIMENTS[
                "wide_scaled_fused_reuse8"
            ].optimizer.implementation,
            "fused",
        )
        self.assertEqual(
            self.module.EXPERIMENTS[
                "wide_scaled_fused_reuse8"
            ].optimizer.max_batch_uses,
            8,
        )
        self.assertEqual(configs["wide_scaled_tied_k2"]["d_model"], 512)
        self.assertEqual(configs["wide_scaled_tied_k2"]["num_loops"], 2)
        self.assertEqual(configs["tied_k4"]["num_loops"], 4)
        self.assertFalse(configs["tied_k4"]["gated_update"])
        self.assertTrue(configs["gated_k4"]["gated_update"])
        self.assertEqual(configs["gated_k4"]["state_tokens"], 0)
        self.assertEqual(configs["state_k4"]["state_tokens"], 4)
        continuous = self.module.EXPERIMENTS["register_continuous"]
        discrete = self.module.EXPERIMENTS["register_discrete_structured"]
        self.assertEqual(continuous.model.architecture, "register")
        self.assertEqual(continuous.model.state_feedback, "continuous")
        self.assertFalse(continuous.model.structured_features)
        self.assertEqual(discrete.model.state_feedback, "straight_through")
        self.assertTrue(discrete.model.structured_features)
        self.assertEqual(continuous.model.d_model, discrete.model.d_model)
        self.assertEqual(continuous.model.step_layers, discrete.model.step_layers)
        self.assertEqual(continuous.model.num_loops, discrete.model.num_loops)
        self.assertTrue(continuous.optimizer.wall_clock_schedule)
        self.assertTrue(discrete.optimizer.wall_clock_schedule)
        reproduction = self.module.EXPERIMENTS[
            "register_frontier_reproduction"
        ]
        self.assertEqual(
            reproduction.model.state_feedback,
            "straight_through",
        )
        self.assertEqual(reproduction.model.final_hidden_mode, "batch_max")
        self.assertFalse(reproduction.model.entropy_active_only)
        self.assertFalse(reproduction.model.entropy_mask_padding)
        self.assertEqual(reproduction.model.num_loops, 64)
        self.assertEqual(reproduction.model.training_loop_cap, 16)
        self.assertEqual(reproduction.model.linear_initialization_scale, 0.4)
        self.assertIsNone(reproduction.model.initialization_std)
        self.assertTrue(reproduction.model.random_detach_prefix)
        per_example = self.module.EXPERIMENTS["register_frontier_per_example"]
        self.assertEqual(per_example.model.final_hidden_mode, "per_example")
        self.assertEqual(
            per_example.model,
            self.module.replace(reproduction.model, final_hidden_mode="per_example"),
        )
        init_002 = self.module.EXPERIMENTS["register_frontier_init_002"]
        self.assertEqual(init_002.model.initialization_std, 0.02)
        self.assertIsNone(init_002.model.linear_initialization_scale)
        masked_entropy = self.module.EXPERIMENTS[
            "register_frontier_masked_entropy"
        ]
        self.assertTrue(masked_entropy.model.entropy_active_only)
        self.assertTrue(masked_entropy.model.entropy_mask_padding)
        full_bptt = self.module.EXPERIMENTS["register_frontier_full_bptt"]
        self.assertFalse(full_bptt.model.random_detach_prefix)
        causal = self.module.EXPERIMENTS["causal_state_contract"]
        self.assertEqual(causal.model.architecture, "causal_state")
        self.assertEqual(causal.model.scratch_tokens, 2)
        self.assertEqual(causal.model.digit_slots, 16)
        self.assertEqual(causal.model.num_loops, 4)
        self.assertEqual(causal.model.training_loop_cap, 4)
        self.assertEqual(causal.model.state_feedback, "none")
        self.assertEqual(self.module.SELECTED_EXPERIMENT, "baseline")

    def test_every_variant_satisfies_model_and_optimizer_contracts(self) -> None:
        for name, experiment in self.module.EXPERIMENTS.items():
            with self.subTest(name=name):
                model = self.build(name)
                self.assertLessEqual(
                    count_model_state_elements(model),
                    self.spec.maximum_model_state_elements,
                )
                optimizer = self.module.build_optimizer_from_config(
                    model,
                    OptimizerSpec(60.0, "cpu"),
                    experiment.optimizer,
                )
                validate_optimizer(optimizer, model, torch.device("cpu"))

    def test_optimizer_factory_selects_requested_execution_path(self) -> None:
        expectations = {
            "wide_scaled": (None, None),
            "wide_scaled_foreach": (True, None),
            "wide_scaled_fused": (None, True),
        }
        for name, (foreach, fused) in expectations.items():
            with self.subTest(name=name):
                experiment = self.module.EXPERIMENTS[name]
                model = self.build(name)
                bundle = self.module.build_optimizer_from_config(
                    model,
                    OptimizerSpec(60.0, "cpu"),
                    experiment.optimizer,
                )
                self.assertEqual(bundle.optimizer.defaults["foreach"], foreach)
                self.assertEqual(bundle.optimizer.defaults["fused"], fused)

    def test_batch_reuse_is_enabled_only_for_requested_variants(self) -> None:
        for name, expected in {
            "wide_scaled": 1,
            "wide_scaled_reuse2": 2,
            "wide_scaled_reuse4": 4,
            "wide_scaled_reuse8": 8,
            "wide_scaled_fused_reuse8": 8,
        }.items():
            with self.subTest(name=name):
                experiment = self.module.EXPERIMENTS[name]
                model = self.build(name)
                bundle = self.module.build_optimizer_from_config(
                    model,
                    OptimizerSpec(60.0, "cpu"),
                    experiment.optimizer,
                )
                if expected > 1:
                    self.assertIsNotNone(bundle.should_reuse_batch)
                    for current_batch_uses in range(1, expected + 1):
                        context = self.module.BatchReuseContext(
                            current_batch_uses,
                            current_batch_uses,
                            2.0,
                        )
                        self.assertEqual(
                            bundle.should_reuse_batch(context),
                            current_batch_uses < expected,
                        )
                else:
                    self.assertIsNone(bundle.should_reuse_batch)

    def test_foreach_update_matches_default_update(self) -> None:
        torch.manual_seed(17)
        default_model = self.build("wide_scaled")
        foreach_model = self.build("wide_scaled_foreach")
        foreach_model.load_state_dict(default_model.state_dict())
        default_config = self.module.EXPERIMENTS["wide_scaled"].optimizer
        foreach_config = self.module.EXPERIMENTS["wide_scaled_foreach"].optimizer
        default_optimizer = self.module.build_optimizer_from_config(
            default_model,
            OptimizerSpec(60.0, "cpu"),
            default_config,
        ).optimizer
        foreach_optimizer = self.module.build_optimizer_from_config(
            foreach_model,
            OptimizerSpec(60.0, "cpu"),
            foreach_config,
        ).optimizer
        for default, foreach in zip(
            default_model.parameters(),
            foreach_model.parameters(),
            strict=True,
        ):
            gradient = torch.randn_like(default)
            default.grad = gradient.clone()
            foreach.grad = gradient.clone()
        default_optimizer.step()
        foreach_optimizer.step()
        for default, foreach in zip(
            default_model.parameters(),
            foreach_model.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(default, foreach)

    def test_output_shape_excludes_internal_state_tokens(self) -> None:
        model = self.build("state_k4")
        input_ids = torch.randint(0, self.spec.vocab_size, (3, 9))
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        logits, auxiliary = model(input_ids, attention_mask=mask)
        self.assertEqual(tuple(logits.shape), (3, 9, self.spec.vocab_size))
        self.assertIsNone(auxiliary)

    def test_numeric_prompt_features_are_exact(self) -> None:
        input_ids = torch.tensor(
            [[2, 10, 9, 10, 3, 11, 9, 4, 10]],
            dtype=torch.long,
        )
        field, place, time_steps = self.module.derived_features(input_ids)
        torch.testing.assert_close(
            field,
            torch.tensor([[1, 1, 1, 1, 2, 2, 2, 3, 3]]),
        )
        torch.testing.assert_close(
            place,
            torch.tensor([[0, 2, 1, 0, 0, 1, 0, 0, 0]]),
        )
        torch.testing.assert_close(time_steps, torch.tensor([3]))

    def test_causal_state_forward_trace_and_decode_are_identical(self) -> None:
        torch.manual_seed(31)
        model = self.build("causal_state_contract").eval()
        state_before = {
            name: value.clone()
            for name, value in model.state_dict().items()
        }
        input_ids = torch.tensor(
            [
                [2, 10, 9, 10, 3, 11, 9, 4, 8],
                [2, 10, 9, 10, 3, 11, 9, 4, 10],
            ],
            dtype=torch.long,
        )
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        with torch.no_grad():
            logits, auxiliary = model(input_ids, attention_mask=mask)
            trace = model.forward_with_trace(input_ids, attention_mask=mask)
            decoded = model.decode_residue(
                trace.residue_states[-1],
                mask,
                input_ids.shape[1],
            )
        self.assertIsNone(auxiliary)
        self.assertEqual(tuple(logits.shape), (2, 9, self.spec.vocab_size))
        self.assertEqual(logits.dtype, model.token_embedding.weight.dtype)
        self.assertTrue(torch.isfinite(logits).all().item())
        torch.testing.assert_close(trace.logits, logits)
        torch.testing.assert_close(decoded, logits)
        self.assertEqual(len(trace.residue_states), 5)
        self.assertEqual(len(trace.scratch_states), 5)
        self.assertEqual(len(trace.active_masks), 4)
        self.assertEqual(tuple(trace.static_memory.shape), (2, 16, 64))
        for name, expected in state_before.items():
            torch.testing.assert_close(model.state_dict()[name], expected)

        prompt_bypass = logits + torch.nn.functional.linear(
            model.token_embedding(input_ids),
            model.head.weight,
        )
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(prompt_bypass, decoded)

    def test_causal_state_rejects_bypass_prone_configurations(self) -> None:
        with self.assertRaisesRegex(ValueError, "continuous hidden state"):
            self.module.ModelConfig(
                architecture="causal_state",
                state_feedback="continuous",
            )
        with self.assertRaisesRegex(ValueError, "per_example"):
            self.module.ModelConfig(
                architecture="causal_state",
                final_hidden_mode="batch_max",
            )
        with self.assertRaisesRegex(ValueError, "scratch_tokens"):
            self.module.ModelConfig(
                architecture="causal_state",
                state_tokens=1,
            )
        with self.assertRaisesRegex(ValueError, "structured_features"):
            self.module.ModelConfig(architecture="causal_state")

    def test_causal_state_masks_each_row_at_its_own_depth(self) -> None:
        torch.manual_seed(37)
        model = self.build("causal_state_contract").eval()
        input_ids = torch.tensor(
            [
                [2, 10, 9, 10, 3, 11, 9, 4, 8],
                [2, 10, 9, 10, 3, 11, 9, 4, 10],
            ],
            dtype=torch.long,
        )
        mask = torch.ones_like(input_ids, dtype=torch.bool)

        def assert_expected_depth(trace) -> None:
            expected = torch.tensor(
                [
                    [True, True],
                    [False, True],
                    [False, True],
                    [False, False],
                ]
            )
            torch.testing.assert_close(torch.stack(trace.active_masks), expected)
            for step_index, active in enumerate(trace.active_masks):
                inactive = ~active
                if inactive.any().item():
                    torch.testing.assert_close(
                        trace.residue_states[step_index + 1][inactive],
                        trace.residue_states[step_index][inactive],
                    )
            active_changes = []
            for step_index, active in enumerate(trace.active_masks):
                if active.any().item():
                    delta = (
                        trace.residue_states[step_index + 1][active]
                        - trace.residue_states[step_index][active]
                    )
                    active_changes.append(delta.abs().max())
            self.assertGreater(torch.stack(active_changes).max().item(), 0)

        with torch.no_grad():
            trace = model.forward_with_trace(input_ids, attention_mask=mask)
        assert_expected_depth(trace)

        with patch.object(
            model,
            "_active_rows",
            side_effect=lambda _step, time_steps: torch.ones_like(
                time_steps,
                dtype=torch.bool,
            ),
        ):
            with torch.no_grad():
                broken_mask_trace = model.forward_with_trace(
                    input_ids,
                    attention_mask=mask,
                )
        with self.assertRaises(AssertionError):
            assert_expected_depth(broken_mask_trace)

        with patch.object(
            model.step,
            "forward",
            side_effect=lambda residue, _modulus, scratch: (residue, scratch),
        ):
            with torch.no_grad():
                frozen_trace = model.forward_with_trace(
                    input_ids,
                    attention_mask=mask,
                )
        with self.assertRaises(AssertionError):
            assert_expected_depth(frozen_trace)

    def test_causal_state_has_an_unbroken_gradient_path(self) -> None:
        torch.manual_seed(41)
        model = self.build("causal_state_contract").train()
        input_ids = torch.tensor([[2, 10, 9, 10, 3, 11, 9, 4, 10]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        trace = model.forward_with_trace(input_ids, attention_mask=mask)
        trace.residue_states[0].retain_grad()
        trace.logits.square().mean().backward()
        self.assertIsNotNone(trace.residue_states[0].grad)
        self.assertGreater(
            torch.count_nonzero(trace.residue_states[0].grad).item(),
            0,
        )
        self.assertIsNotNone(model.step.residue_cell.weight_hh.grad)
        self.assertGreater(
            torch.count_nonzero(model.step.residue_cell.weight_hh.grad).item(),
            0,
        )
        self.assertIsNotNone(model.step.scratch_cell.weight_hh.grad)
        self.assertGreater(
            torch.count_nonzero(model.step.scratch_cell.weight_hh.grad).item(),
            0,
        )

    def test_causal_state_interventions_target_recurrent_channels(self) -> None:
        torch.manual_seed(59)
        model = self.build("causal_state_contract").eval()
        input_ids = torch.tensor(
            [
                [2, 10, 9, 10, 3, 11, 9, 4, 11],
                [2, 10, 9, 10, 3, 12, 9, 4, 11],
            ]
        )
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        interventions = _causal_state_interventions(model)

        with torch.no_grad():
            baseline = model.forward_with_trace(input_ids, attention_mask=mask)
        with interventions["freeze_residue_all"]():
            with torch.no_grad():
                frozen = model.forward_with_trace(input_ids, attention_mask=mask)
        with interventions["freeze_residue_after_step_1"]():
            with torch.no_grad():
                one_step = model.forward_with_trace(input_ids, attention_mask=mask)
        with interventions["zero_scratch_transition"]():
            with torch.no_grad():
                zero_scratch = model.forward_with_trace(
                    input_ids,
                    attention_mask=mask,
                )

        for state in frozen.residue_states[1:]:
            torch.testing.assert_close(state, frozen.residue_states[0])
        torch.testing.assert_close(
            one_step.residue_states[1],
            baseline.residue_states[1],
        )
        for state in one_step.residue_states[2:]:
            torch.testing.assert_close(state, one_step.residue_states[1])
        for state in zero_scratch.scratch_states[1:]:
            torch.testing.assert_close(state, torch.zeros_like(state))

    def test_prediction_comparison_detects_causal_changes_without_accuracy_change(
        self,
    ) -> None:
        baseline = PredictionMeasurement(
            metrics={"loss": 2.0, "exact_accuracy": 0.0},
            predictions=((1, 2), (3, 4)),
            correct=(False, False),
        )
        intervention = PredictionMeasurement(
            metrics={"loss": 2.5, "exact_accuracy": 0.0},
            predictions=((1, 5), (3, 6)),
            correct=(False, False),
        )
        comparison = _compare_measurement(intervention, baseline)
        self.assertEqual(comparison["accuracy_delta"], 0.0)
        self.assertEqual(comparison["prediction_change_rate"], 1.0)
        self.assertEqual(comparison["token_change_rate"], 0.5)

    def test_causal_state_preserves_padding_and_batch_order(self) -> None:
        torch.manual_seed(43)
        model = self.build("causal_state_contract").eval()
        input_ids = torch.tensor(
            [
                [2, 10, 9, 10, 3, 11, 9, 4, 8],
                [2, 10, 9, 10, 3, 11, 10, 4, 9],
            ],
            dtype=torch.long,
        )
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        padded_ids = torch.nn.functional.pad(input_ids, (0, 3), value=0)
        padded_mask = torch.nn.functional.pad(mask, (0, 3), value=False)
        permutation = torch.tensor([1, 0])
        with torch.no_grad():
            expected, _ = model(input_ids, attention_mask=mask)
            padded, _ = model(padded_ids, attention_mask=padded_mask)
            permuted, _ = model(
                input_ids[permutation],
                attention_mask=mask[permutation],
            )
        torch.testing.assert_close(padded[:, : input_ids.shape[1]], expected)
        torch.testing.assert_close(permuted, expected[permutation])

    def test_register_feedback_controls_are_behaviorally_distinct(self) -> None:
        input_ids = torch.tensor(
            [
                [2, 10, 9, 10, 3, 11, 4, 8],
                [2, 10, 9, 10, 3, 12, 4, 10],
            ],
            dtype=torch.long,
        )
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        traces = {}
        for name in ("register_continuous", "register_discrete_structured"):
            model = self.build(name).eval()
            with torch.no_grad():
                traces[name] = model.forward_with_trace(
                    input_ids,
                    attention_mask=mask,
                )
            self.assertEqual(len(traces[name].register_states), 4)
            self.assertEqual(len(traces[name].prompt_states), 4)
            torch.testing.assert_close(
                traces[name].register_states[1][0],
                traces[name].register_states[2][0],
            )
            torch.testing.assert_close(
                traces[name].register_states[2][0],
                traces[name].register_states[3][0],
            )

        continuous = traces["register_continuous"].register_states[-1]
        discrete = traces["register_discrete_structured"].register_states[-1]
        torch.testing.assert_close(
            continuous.sum(dim=-1),
            torch.ones_like(continuous[..., 0]),
        )
        torch.testing.assert_close(
            discrete.sum(dim=-1),
            torch.ones_like(discrete[..., 0]),
        )
        self.assertTrue(((discrete == 0) | (discrete == 1)).all().item())
        self.assertFalse(((continuous == 0) | (continuous == 1)).all().item())

    def test_discrete_register_keeps_gradient_path(self) -> None:
        torch.manual_seed(23)
        model = self.build("register_discrete_structured").train()
        input_ids = torch.tensor([[2, 10, 9, 10, 3, 11, 4, 10]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        logits, auxiliary = model(input_ids, attention_mask=mask)
        loss = logits.square().mean()
        if auxiliary is not None:
            loss = loss + auxiliary
        loss.backward()
        gradient = model.state_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(torch.count_nonzero(gradient).item(), 0)

    def test_register_padding_does_not_change_valid_logits(self) -> None:
        model = self.build("register_discrete_structured").eval()
        input_ids = torch.tensor([[2, 10, 9, 10, 3, 11, 4, 10]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        padded_ids = torch.nn.functional.pad(input_ids, (0, 3), value=0)
        padded_mask = torch.nn.functional.pad(mask, (0, 3), value=False)
        with torch.no_grad():
            expected, _ = model(input_ids, attention_mask=mask)
            padded, _ = model(padded_ids, attention_mask=padded_mask)
        torch.testing.assert_close(padded[:, : input_ids.shape[1]], expected)

    def test_register_optimizer_uses_wall_clock_schedule(self) -> None:
        experiment = self.module.EXPERIMENTS["register_continuous"]
        model = self.build("register_continuous")
        with patch.object(self.module.time, "monotonic", return_value=0.0):
            bundle = self.module.build_optimizer_from_config(
                model,
                OptimizerSpec(60.0, "cpu"),
                experiment.optimizer,
            )
        self.assertIsNotNone(bundle.scheduler)
        initial_learning_rate = bundle.optimizer.param_groups[0]["lr"]
        with patch.object(self.module.time, "monotonic", return_value=30.0):
            bundle.scheduler.step()
        self.assertLess(
            bundle.optimizer.param_groups[0]["lr"],
            initial_learning_rate,
        )

    def test_frontier_reproduction_preserves_embedding_scale(self) -> None:
        torch.manual_seed(29)
        controlled = self.build("register_discrete_structured")
        torch.manual_seed(29)
        reproduction = self.build("register_frontier_reproduction")
        self.assertLess(controlled.token_embedding.weight.std().item(), 0.03)
        self.assertLess(reproduction.token_embedding.weight.std().item(), 0.04)
        self.assertGreater(reproduction.position_embedding.weight.std().item(), 0.5)
        self.assertGreater(reproduction.field_embedding.weight.std().item(), 0.5)
        self.assertGreater(reproduction.place_embedding.weight.std().item(), 0.5)
        expected_state_projection_std = 0.4 * self.spec.vocab_size ** -0.5
        self.assertAlmostEqual(
            reproduction.state_projection.weight.std().item(),
            expected_state_projection_std,
            delta=0.015,
        )

    def test_batch_permutation_equivariance(self) -> None:
        torch.manual_seed(3)
        model = self.build("state_k4").eval()
        input_ids = torch.randint(0, self.spec.vocab_size, (4, 10))
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            expected, _ = model(input_ids, attention_mask=mask)
            permuted, _ = model(
                input_ids[permutation], attention_mask=mask[permutation]
            )
        torch.testing.assert_close(permuted, expected[permutation])

    def test_padding_does_not_change_valid_position_logits(self) -> None:
        torch.manual_seed(5)
        model = self.build("state_k4").eval()
        input_ids = torch.randint(1, self.spec.vocab_size, (2, 7))
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        padded_ids = torch.nn.functional.pad(input_ids, (0, 3), value=0)
        padded_mask = torch.nn.functional.pad(mask, (0, 3), value=False)
        with torch.no_grad():
            expected, _ = model(input_ids, attention_mask=mask)
            padded, _ = model(padded_ids, attention_mask=padded_mask)
        torch.testing.assert_close(padded[:, : input_ids.shape[1]], expected)

    def test_trace_records_recurrent_occurrences_not_module_identity(self) -> None:
        model = self.build("state_k4").eval()
        input_ids = torch.randint(0, self.spec.vocab_size, (2, 8))
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        with torch.no_grad():
            trace = model.forward_with_trace(input_ids, attention_mask=mask)
        self.assertEqual(len(trace.prompt_states), 5)
        self.assertEqual(len(trace.memory_states), 5)
        self.assertEqual(tuple(trace.prompt_states[-1].shape), (2, 8, 128))
        self.assertEqual(tuple(trace.memory_states[-1].shape), (2, 4, 128))

    def test_padding_invariant_discriminates_a_broken_mask(self) -> None:
        torch.manual_seed(7)
        model = self.build("baseline").eval()
        input_ids = torch.randint(1, self.spec.vocab_size, (1, 6))
        padded_ids = torch.nn.functional.pad(input_ids, (0, 2), value=0)
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        broken_mask = torch.ones_like(padded_ids, dtype=torch.bool)
        with torch.no_grad():
            expected, _ = model(input_ids, attention_mask=valid_mask)
            broken, _ = model(padded_ids, attention_mask=broken_mask)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(broken[:, :6], expected)


if __name__ == "__main__":
    unittest.main()
