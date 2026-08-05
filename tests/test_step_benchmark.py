from __future__ import annotations

import math
import unittest

from research.step_benchmark import compile_break_even, summarize_samples


class StepBenchmarkTests(unittest.TestCase):
    def test_sample_summary_preserves_raw_distribution_statistics(self) -> None:
        summary = summarize_samples([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["median_ms"], 2.5)
        self.assertEqual(summary["minimum_ms"], 1.0)
        self.assertEqual(summary["maximum_ms"], 4.0)
        self.assertGreater(summary["coefficient_of_variation"], 0)

    def test_sample_summary_rejects_invalid_timing_evidence(self) -> None:
        invalid = ([1.0, 2.0], [1.0, math.nan, 2.0], [1.0, 0.0, 2.0])
        for samples in invalid:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                summarize_samples(samples)

    def test_compile_break_even_discriminates_setup_overhead(self) -> None:
        promoted = compile_break_even(
            budget_seconds=60.0,
            eager_step_ms=500.0,
            compiled_step_ms=250.0,
            compile_overhead_ms=10_000.0,
        )
        rejected = compile_break_even(
            budget_seconds=60.0,
            eager_step_ms=500.0,
            compiled_step_ms=250.0,
            compile_overhead_ms=40_000.0,
        )
        self.assertTrue(promoted["promoted"])
        self.assertFalse(rejected["promoted"])
        self.assertEqual(promoted["eager_updates"], 120.0)
        self.assertEqual(promoted["candidate_updates"], 200.0)


if __name__ == "__main__":
    unittest.main()
