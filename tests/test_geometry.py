from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from research.geometry import (
    _pair_routing_statistics,
    linear_cka,
    orthogonal_procrustes_similarity,
)


class GeometryMetricTests(unittest.TestCase):
    def test_identical_representations_have_unit_similarity(self) -> None:
        torch.manual_seed(11)
        representation = torch.randn(64, 8)
        self.assertAlmostEqual(linear_cka(representation, representation), 1.0, places=5)
        self.assertAlmostEqual(
            orthogonal_procrustes_similarity(representation, representation),
            1.0,
            places=5,
        )

    def test_orthogonal_rotation_preserves_both_similarities(self) -> None:
        torch.manual_seed(17)
        representation = torch.randn(64, 8)
        rotation, _ = torch.linalg.qr(torch.randn(8, 8))
        rotated = representation @ rotation
        self.assertAlmostEqual(linear_cka(representation, rotated), 1.0, places=5)
        self.assertAlmostEqual(
            orthogonal_procrustes_similarity(representation, rotated),
            1.0,
            places=5,
        )

    def test_shape_errors_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "row-aligned"):
            linear_cka(torch.randn(3, 2), torch.randn(4, 2))
        with self.assertRaisesRegex(ValueError, "same matrix shape"):
            orthogonal_procrustes_similarity(torch.randn(3, 2), torch.randn(3, 4))

    def test_pair_routing_statistics_expose_learned_route_structure(self) -> None:
        class PairInteraction(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.route_logits = nn.Parameter(torch.zeros(4, 4, 4))

            def routing_weights(self) -> torch.Tensor:
                return self.route_logits.flatten(1).softmax(dim=-1).view_as(
                    self.route_logits
                )

        model = SimpleNamespace(
            step=SimpleNamespace(pair_interaction=PairInteraction())
        )
        statistics = _pair_routing_statistics(model)
        assert statistics is not None

        self.assertEqual(statistics["route_logit_std"], 0.0)
        self.assertEqual(statistics["mean_symmetry_error"], 0.0)
        self.assertAlmostEqual(
            statistics["outputs"][0]["normalized_entropy"],
            1.0,
        )
        self.assertAlmostEqual(
            statistics["outputs"][0]["effective_pair_count"],
            16.0,
        )
        self.assertAlmostEqual(statistics["outputs"][0]["sum_route_mass"], 1 / 16)
        self.assertAlmostEqual(statistics["outputs"][3]["sum_route_mass"], 4 / 16)


if __name__ == "__main__":
    unittest.main()
