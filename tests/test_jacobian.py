from __future__ import annotations

import unittest

import torch

from research.jacobian import average_jacobians


class JacobianEstimatorTests(unittest.TestCase):
    def test_average_jacobian_recovers_a_shared_linear_map(self) -> None:
        torch.manual_seed(7)
        batch = 3
        length = 4
        width = 5
        dim_batch = 2
        weight = torch.randn(width, width)
        source = torch.randn(
            dim_batch * batch,
            length,
            width,
            requires_grad=True,
        )
        target = source @ weight.T
        mask = torch.ones(batch, length, dtype=torch.bool)
        actual = average_jacobians(
            target=target,
            sources=[source],
            attention_mask=mask,
            original_batch_size=batch,
            dim_batch=dim_batch,
        )[0]
        torch.testing.assert_close(actual, weight)


if __name__ == "__main__":
    unittest.main()
