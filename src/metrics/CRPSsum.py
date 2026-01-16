import torch
from torchmetrics import Metric
import properscoring as ps
import numpy as np

class CRPSSum(Metric):
    def __init__(self, normalize=False, dist_sync_on_step=False):
        """
        Args:
            normalize: If True, normalize CRPS by the L1 norm of true values (sum of absolute values).
                      This makes the metric scale-invariant and comparable across different datasets.
            dist_sync_on_step: Whether to synchronize metric state across processes at each step.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.normalize = normalize
        self.add_state("total_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        if self.normalize:
            self.add_state("total_denom", default=torch.tensor(0.0), dist_reduce_fx="sum")

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
        
        # vectorized calculation of CRPS using properscoring
        crps_values = ps.crps_ensemble(true_np, pred_np)
        
        # Accumulate CRPS values
        batch_crps_sum = crps_values.sum()
        self.total_crps += torch.tensor(batch_crps_sum, device=self.device)
        
        if self.normalize:
            # Accumulate denominator (L1 norm of true values)
            # This matches calc_quantile_CRPS_sum: denom = np.sum(np.abs(target))
            batch_denom = np.sum(np.abs(true_np))
            self.total_denom += torch.tensor(batch_denom, device=self.device)
        
        # accumulate results
        self.total_samples += torch.tensor(len(true_np), device=self.device)

    def compute(self):
        if self.total_samples == 0:
            return torch.tensor(0.0)
        
        if self.normalize:
            # Normalize by total denominator (matching calc_quantile_CRPS_sum)
            # This matches: CRPS = (Σ q_loss) / denom / num_quantiles
            # where denom = np.sum(np.abs(target)) for the entire dataset
            if self.total_denom > 0:
                return self.total_crps / self.total_denom
            else:
                return torch.tensor(0.0)
        else:
            # Original implementation: simple average of CRPS values
            return self.total_crps / self.total_samples

    # ============================================================================
    # ORIGINAL IMPLEMENTATION (COMMENTED FOR REFERENCE)
    # ============================================================================
    # The original implementation without normalization is preserved below for reference.
    # To restore it, uncomment this section and remove the normalize parameter.
    #
    # def __init__(self, dist_sync_on_step=False):
    #     super().__init__(dist_sync_on_step=dist_sync_on_step)
    #     self.add_state("total_crps", default=torch.tensor(0.0), dist_reduce_fx="sum")
    #     self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
    #
    # def update(self, pred: torch.Tensor, true: torch.Tensor):
    #     """
    #     Args:
    #         pred: Tensor of predicted distributions, shape (B, O, N, S).
    #         true: Tensor of true values, shape (B, O, N).
    #     """
    #     # sum over N dimension: (B, O, N, S) -> (B, O, S), (B, O, N) -> (B, O)
    #     pred_sum = pred.sum(dim=2)  # Shape: (B, O, S)
    #     true_sum = true.sum(dim=2)   # Shape: (B, O)
    #     
    #     # flatten: (B, O, S) -> (B*O, S), (B, O) -> (B*O,)
    #     pred_flat = pred_sum.reshape(-1, pred_sum.shape[-1])  # (B*O, S)
    #     true_flat = true_sum.reshape(-1)  # (B*O,)
    #     
    #     # convert to numpy
    #     pred_np = pred_flat.detach().cpu().numpy()
    #     true_np = true_flat.detach().cpu().numpy()
    #     
    #     # vectorized calculation of CRPS
    #     crps_values = ps.crps_ensemble(true_np, pred_np)
    #     
    #     # accumulate results
    #     self.total_crps += torch.tensor(crps_values.sum(), device=self.device)
    #     self.total_samples += torch.tensor(len(true_np), device=self.device)
    #
    # def compute(self):
    #     if self.total_samples == 0:
    #         return torch.tensor(0.0)
    #     return self.total_crps / self.total_samples