from __future__ import annotations

import unittest

import torch

from model.modules.residual_baseline import compute_persistence_baseline
from model.modules.multihead_decoder import MultiHeadHorizonDecoder


class TestResidualBaseline(unittest.TestCase):
    def test_persistence_baseline_shape_and_invariants(self):
        B, L, N, F = 2, 168, 12, 17
        H = 24
        D = 6
        target_idx = list(range(D))  # pretend first 6 features are pollutants

        # Use identity scaler so "raw" == "scaled" for all features.
        center = torch.zeros(F)
        scale = torch.ones(F)

        X = torch.randn(B, L, N, F)
        y_base = compute_persistence_baseline(
            X,
            horizon=H,
            target_feature_indices=target_idx,
            input_center=center,
            input_scale=scale,
            assert_checks=True,
        )

        self.assertEqual(tuple(y_base.shape), (B, H, N, D))
        self.assertTrue(torch.allclose(y_base[:, 0], y_base[:, -1]))
        self.assertTrue(torch.allclose(y_base[:, 0], X[:, -1, :, :D]))


class TestMultiHeadDecoder(unittest.TestCase):
    def test_multihead_decoder_shape(self):
        B, N, C = 2, 12, 64
        H = 24
        D = 6
        dec = MultiHeadHorizonDecoder(d_in=C, horizon=H, d_h=16, num_targets=D, hidden_dim=32, dropout=0.0)
        rep = torch.randn(B, N, C)
        out = dec(rep)
        self.assertEqual(tuple(out.shape), (B, H, N, D))


if __name__ == "__main__":
    unittest.main()

