import torch
from torchmetrics import Metric
import properscoring as ps  # Import standard library
import numpy as np

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
        
        # 2. Convert to Numpy (properscoring uses CPU/Numpy operations)
        pred_np = pred.detach().cpu().numpy()
        true_np = true.detach().cpu().numpy()

        # 3. Vectorized calculation
        # crps_ensemble accepts array inputs and calculates CRPS for all points at once
        # Much faster than for loops by several orders of magnitude
        crps_values = ps.crps_ensemble(true_np, pred_np)

        # 4. Accumulate results
        batch_sum = crps_values.sum()
        
        self.total_crps += torch.tensor(batch_sum, device=self.device)
        self.total_samples += torch.tensor(len(true_np), device=self.device)

    def compute(self):
        return self.total_crps / self.total_samples