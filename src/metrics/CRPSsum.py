import torch
from torchmetrics import Metric
import properscoring as ps

class CRPSSum(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        """
        Args:
            pred: Tensor of predicted distributions, shape (B, O, N, S).
            true: Tensor of true values, shape (B, O, N).
        """
        # sum over N dimension: (B, O, N, S) -> (B, O, S), (B, O, N) -> (B, O)
        pred_sum = pred.sum(dim=2)  # Shape: (B, O, S)
        true_sum = true.sum(dim=2)   # Shape: (B, O)
        
        # flatten: (B, O, S) -> (B*O, S), (B, O) -> (B*O,)
        pred_flat = pred_sum.reshape(-1, pred_sum.shape[-1])  # (B*O, S)
        true_flat = true_sum.reshape(-1)  # (B*O,)
        
        # convert to numpy
        pred_np = pred_flat.detach().cpu().numpy()
        true_np = true_flat.detach().cpu().numpy()
        
        # vectorized calculation of CRPS
        crps_values = ps.crps_ensemble(true_np, pred_np)
        
        # accumulate results
        self.total_crps += torch.tensor(crps_values.sum(), device=self.device)
        self.total_samples += torch.tensor(len(true_np), device=self.device)

    def compute(self):
        if self.total_samples == 0:
            return torch.tensor(0.0)
        return self.total_crps / self.total_samples