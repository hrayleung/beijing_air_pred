from __future__ import annotations

import unittest

import torch

from model.modules.temporal_attn_decoder import TemporalAttentionHorizonDecoder


class TestTemporalAttentionHorizonDecoder(unittest.TestCase):
    def test_temporal_attn_decoder_shape(self):
        B, N, C, L = 2, 12, 64, 168
        H, D = 24, 6
        dec = TemporalAttentionHorizonDecoder(d_in=C, horizon=H, d_h=16, out_dim=D, hidden_dim=32, dropout=0.0)
        rep_seq = torch.randn(B, N, C, L)
        out = dec(rep_seq)
        self.assertEqual(tuple(out.shape), (B, H, N, D))


if __name__ == "__main__":
    unittest.main()

