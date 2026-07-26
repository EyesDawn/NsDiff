import numpy as np
import torch
from torchmetrics import Metric


def ensemble_crps_sum(
    truth: np.ndarray, predictions: np.ndarray, observation_chunk_size: int = 4096
) -> float:
    """Sum equal-weight ensemble CRPS with bounded temporary memory.

    For sorted ensemble members x_(i), half the expected pairwise distance is
    sum_i (2i - S - 1) x_(i) / S**2.  Processing observations in chunks avoids
    the O(N * S**2) memory used by properscoring's non-Numba fallback.
    """
    truth = np.asarray(truth)
    predictions = np.asarray(predictions)
    if truth.ndim != 1 or predictions.ndim != 2 or predictions.shape[0] != truth.shape[0]:
        raise ValueError("Expected truth [N] and predictions [N, S].")
    if predictions.shape[1] < 1:
        raise ValueError("predictions must contain at least one ensemble member.")
    if observation_chunk_size < 1:
        raise ValueError("observation_chunk_size must be at least 1.")

    sample_count = predictions.shape[1]
    coefficients = 2.0 * np.arange(sample_count, dtype=np.float64) - sample_count + 1.0
    total = 0.0
    for start in range(0, truth.shape[0], observation_chunk_size):
        stop = start + observation_chunk_size
        sorted_predictions = np.sort(predictions[start:stop], axis=1)
        first = np.abs(sorted_predictions - truth[start:stop, None]).mean(axis=1)
        second = sorted_predictions @ coefficients / (sample_count ** 2)
        total += float(np.sum(first - second, dtype=np.float64))
    return total


class CRPS(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        """
        Args:
            pred: (B, O, N, S) - Samples from the predicted distribution
            true: (B, O, N)    - True observations
        """
        # 1. Flatten dimensions: (Total_Points, Samples) vs (Total_Points,)
        # Assuming S (Samples) is the last dimension
        pred = pred.reshape(-1, pred.shape[-1])
        true = true.reshape(-1)
        
        # 2. Convert to NumPy for the chunked sorted-ensemble calculation.
        pred_np = pred.detach().cpu().numpy()
        true_np = true.detach().cpu().numpy()

        # 3. Accumulate results.
        batch_sum = ensemble_crps_sum(true_np, pred_np)
        
        self.total_crps += torch.as_tensor(batch_sum, device=self.device)
        self.total_samples += torch.tensor(len(true_np), device=self.device)

    def compute(self):
        return self.total_crps / self.total_samples
