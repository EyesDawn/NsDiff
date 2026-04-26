import torch
import torch.nn as nn


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class moving_avg(nn.Module):
    """
    Moving average block used by DLinear.
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block from DLinear.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class Model(nn.Module):
    """
    DLinear point-forecast model with an additional conditioning feature head
    for iReflow-style downstream usage.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = getattr(configs, "enc_in", 1)
        self.d_model = getattr(configs, "d_model", 128)
        self.output_attention = getattr(configs, "output_attention", False)
        self.use_norm = _as_bool(getattr(configs, "use_norm", True))
        self.individual = _as_bool(getattr(configs, "individual", False))
        self.moving_avg = int(getattr(configs, "moving_avg", 25))

        self.decompsition = series_decomp(self.moving_avg)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)]
            )
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)]
            )
            self.feature_projection = nn.ModuleList(
                [nn.Linear(self.seq_len * 2, self.d_model) for _ in range(self.enc_in)]
            )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.feature_projection = nn.Linear(self.seq_len * 2, self.d_model)

    def _normalize(self, x_enc):
        if not self.use_norm:
            return x_enc, None, None

        means = x_enc.mean(1, keepdim=True).detach()
        x_centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_norm = x_centered / stdev
        return x_norm, means, stdev

    def _denormalize(self, y_hat, means, stdev):
        if not self.use_norm:
            return y_hat
        y_hat = y_hat * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        y_hat = y_hat + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return y_hat

    def encode(self, x_enc, x_mark_enc=None):
        x_norm, means, stdev = self._normalize(x_enc)
        seasonal_init, trend_init = self.decompsition(x_norm)

        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        feature_input = torch.cat([seasonal_init, trend_init], dim=-1)

        if self.individual:
            seasonal_output = []
            trend_output = []
            enc_features = []
            for idx in range(x_enc.shape[-1]):
                seasonal_output.append(
                    self.Linear_Seasonal[idx](seasonal_init[:, idx, :])
                )
                trend_output.append(self.Linear_Trend[idx](trend_init[:, idx, :]))
                enc_features.append(
                    self.feature_projection[idx](feature_input[:, idx, :])
                )
            seasonal_output = torch.stack(seasonal_output, dim=1)
            trend_output = torch.stack(trend_output, dim=1)
            enc_features = torch.stack(enc_features, dim=1)
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            enc_features = self.feature_projection(feature_input)

        dec_out = seasonal_output + trend_output
        y_hat = dec_out.permute(0, 2, 1)
        y_hat = self._denormalize(y_hat, means, stdev)

        norm_stats = {
            "means": means,
            "stdev": stdev,
        }
        return enc_features, y_hat, norm_stats

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        _, y_hat, _ = self.encode(x_enc, x_mark_enc)
        return y_hat, None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        return dec_out[:, -self.pred_len :, :]
