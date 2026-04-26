import torch

from src.models.DLinear import Model as DLinear
from src.models.iReflow import iReflow


class iReflowDLinear(iReflow):
    """
    iReflow variant that swaps the iTransformer conditioner for DLinear.
    """

    def __init__(self, configs):
        super().__init__(configs)
        self.itransformer = DLinear(configs)

    def _get_encoder_outputs(self, x_enc, x_mark_enc):
        enc_features, y_hat, norm_stats = self.itransformer.encode(x_enc, x_mark_enc)
        sigma = self.uncertainty_estimator(enc_features).permute(0, 2, 1)

        if self.itransformer.use_norm and norm_stats["stdev"] is not None:
            scale = norm_stats["stdev"][:, 0, :].unsqueeze(1).expand(
                -1, self.pred_len, -1
            )
            sigma = sigma * scale
            min_sigma = sigma.new_full((), 0.001)
            sigma = torch.clamp(sigma, min=min_sigma, max=2.0 * scale)
        else:
            sigma = torch.clamp(sigma, min=0.001)

        var = sigma.pow(2)
        s = torch.log(var)
        return enc_features, y_hat, s, var, sigma
