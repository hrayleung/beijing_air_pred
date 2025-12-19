from __future__ import annotations

import unittest

import torch

from model.modules.dynamic_graph import WindGatedDynamicGraphBuilder


class TestWindGatedDynamicGraphBuilder(unittest.TestCase):
    def test_wind_gate_changes_attention_temperature(self):
        torch.manual_seed(0)

        B, L, N = 2, 3, 4
        d_model = 8

        builder = WindGatedDynamicGraphBuilder(
            num_nodes=N,
            d_model=d_model,
            d_qk=d_model,
            d_node_emb=4,
            wind_gate_hidden=8,
            lambda_gate=5.0,
            add_loops=False,
        )

        h = torch.randn(B, L, N, d_model)
        gate_zeros = torch.zeros(B, L, N, 1)
        gate_ones = torch.ones(B, L, N, 1)

        A0 = builder.build_A_dyn(h, gate_zeros)
        A1 = builder.build_A_dyn(h, gate_ones)

        delta = float((A0 - A1).abs().mean().item())
        self.assertGreater(delta, 1e-4)


if __name__ == "__main__":
    unittest.main()
