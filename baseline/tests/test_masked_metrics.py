import unittest
import numpy as np

from baseline.evaluation.masked_metrics import masked_mae


class TestMaskedMetrics(unittest.TestCase):
    def test_masked_mae_zero_when_equal_on_observed(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=(3, 24, 12, 6)).astype(np.float32)
        mask = (rng.random(size=y.shape) > 0.3).astype(np.float32)
        pred = y.copy()
        self.assertAlmostEqual(float(masked_mae(pred, y, mask)), 0.0, places=7)

    def test_masked_mae_ignores_missing_positions(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=(2, 24, 12, 6)).astype(np.float32)
        mask = np.zeros_like(y, dtype=np.float32)
        mask[:, :, :, :] = 0.0
        # Mark a few observed entries
        mask[0, 0, 0, 0] = 1.0
        mask[1, 23, 11, 5] = 1.0

        pred = y.copy()
        pred[mask == 0] = 1e9  # should be ignored
        self.assertAlmostEqual(float(masked_mae(pred, y, mask)), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()

