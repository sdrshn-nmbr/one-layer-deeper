from __future__ import annotations

import unittest

import torch

from research.geometry import linear_cka, orthogonal_procrustes_similarity


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


if __name__ == "__main__":
    unittest.main()
