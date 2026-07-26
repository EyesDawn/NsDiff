import unittest

import torch

from src.experiments.prob_forecast import ProbForecastExp


class _SingleBatchLoader(list):
    def __init__(self, batch):
        super().__init__([batch])
        self.dataset = [0]


class ProbForecastEvaluateTest(unittest.TestCase):
    def test_evaluate_returns_metrics_without_debug_assertion(self):
        experiment = object.__new__(ProbForecastExp)
        experiment.model = torch.nn.Identity()
        experiment.device = torch.device("cpu")
        experiment.invtrans_loss = False
        experiment._init_metrics()

        samples = torch.arange(1, 21, dtype=torch.float32).reshape(1, 1, 1, 20)
        targets = torch.tensor([[[10.0]]])
        experiment._process_val_batch = lambda *_: (samples, targets)
        batch = tuple(torch.zeros(1, 1, 1) for _ in range(6))

        result = experiment._evaluate(_SingleBatchLoader(batch))

        self.assertEqual(
            set(result), {"crps", "crps_sum", "qice", "picp", "mse", "mae", "rmse"}
        )
