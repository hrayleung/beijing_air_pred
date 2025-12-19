from __future__ import annotations

import unittest

import torch

from extreme_model.models.stformer import STFormer, STFormerConfig


class TestSTFormer(unittest.TestCase):
    def test_forward_shape(self):
        B, L, N, F = 2, 168, 12, 17
        H, D = 24, 6

        cfg = STFormerConfig(
            num_nodes=N,
            in_features=F,
            lookback=L,
            horizon=H,
            num_targets=D,
            d_model=32,
            n_heads=4,
            enc_layers=1,
            dec_layers=1,
            ff_dim=64,
            dropout=0.0,
            use_future_time_features=True,
            assert_shapes=True,
        )

        time_idx = {"hour_sin": 13, "hour_cos": 14, "month_sin": 15, "month_cos": 16}
        center = torch.zeros(F)
        scale = torch.ones(F)
        model = STFormer(cfg, time_feature_indices=time_idx, input_center=center, input_scale=scale)

        X = torch.randn(B, L, N, F)
        out = model(X)
        self.assertEqual(tuple(out.shape), (B, H, N, D))


if __name__ == "__main__":
    unittest.main()

