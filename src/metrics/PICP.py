import torch
from torchmetrics import Metric

class PICP(Metric):
    def __init__(self, low_percentile: int = 5, high_percentile: int = 95, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.add_state("coverage", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, all_gen_y: torch.Tensor, y_true: torch.Tensor):
        all_gen_y = all_gen_y.view(-1, all_gen_y.shape[3]).detach().cpu()
        y_true = y_true.view(-1).detach().cpu()

        low, high = self.low_percentile, self.high_percentile
        CI_y_pred = torch.quantile(all_gen_y, torch.tensor([low / 100.0, high / 100.0]).float(), dim=1)
        
        y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
        
        # accumulate the number of samples covered, not the average
        self.coverage += y_in_range.float().sum().to(self.device)
        self.total_samples += y_true.size(0)

    def compute(self):
        if self.total_samples == 0:
            return torch.tensor(0.0)
        return self.coverage / self.total_samples
