import torch
from torchmetrics import Metric


class EnergyScore(Metric):
    """Ensemble Energy Score over each complete forecast window."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_windows", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, true: torch.Tensor):
        """Accept pred [B, horizon, variates, samples], true [B, horizon, variates]."""
        if pred.ndim != 4 or true.ndim != 3 or pred.shape[:3] != true.shape:
            raise ValueError("Expected pred [B, P, D, S] aligned with true [B, P, D].")

        samples = pred.permute(0, 3, 1, 2).flatten(2).double()
        targets = true.flatten(1).double()
        sample_count = samples.shape[1]
        first_term = torch.linalg.vector_norm(samples - targets.unsqueeze(1), dim=2).mean(dim=1)
        # cdist keeps only [B, S, S] distances, avoiding a [B, S, P*D]
        # broadcast temporary for every ensemble member.
        pair_term = torch.cdist(samples, samples, p=2).sum(dim=(1, 2)) / (2.0 * sample_count ** 2)
        score = (first_term - pair_term).sum()
        self.total_score += score.to(self.total_score.device)
        self.total_windows += pred.shape[0]

    def compute(self):
        if self.total_windows == 0:
            return torch.tensor(0.0, device=self.total_score.device)
        return self.total_score / self.total_windows
