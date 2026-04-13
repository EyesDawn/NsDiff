from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.TimeGrad import TimeGrad
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
# import multiprocessing
import torch.multiprocessing as mp

import numpy as np
import torch.distributed as dist
import torch
from tqdm import tqdm
import concurrent.futures



@dataclass
class TimeGradParameters:
    num_layers : int = 2
    rnn_hidden_size : int = 40
    rnn_type : str = 'LSTM'
    dropout_rate : float = 0.1
    lags_seq : List[int] = field(default_factory= lambda : [1, 24, 144])
    diff_steps : int = 100
    beta_end : float = 0.1
    beta_schedule : str = 'linear'
    residual_layers : int = 8
    residual_channels : int = 8
    context_length : int = 24
    covariate_hidden_size : int = 128
    conditioning_length : int = 100
    dilation_cycle_length : int = 2
    num_samples : int = 100
    scale:bool = True

@dataclass
class TimeGradForecast(ProbForecastExp, TimeGradParameters):
    model_type: str = "TimeGrad"
    def _init_model(self):
        max_supported_lag = self.windows - self.context_length
        valid_lags = [lag for lag in self.lags_seq if lag <= max_supported_lag]
        if not valid_lags:
            raise ValueError(
                "TimeGrad requires at least one lag compatible with the "
                f"current window/context_length setup, but got lags={self.lags_seq}, "
                f"windows={self.windows}, context_length={self.context_length}."
            )
        if valid_lags != self.lags_seq:
            print(
                "Adjusting TimeGrad lags_seq from "
                f"{self.lags_seq} to {valid_lags} for windows={self.windows} "
                f"and context_length={self.context_length}."
            )
        self.lags_seq = valid_lags
        self.minisample = min(100, self.num_samples)
        self.model = TimeGrad(
            pred_len=self.pred_len,
            sequence_length=self.windows,
            enc_in=self.dataset.num_features,
            lags=self.lags_seq,
            context_length=self.context_length,
            TimeF = freq_map[self.dataset.freq],
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_type = self.rnn_type,
            num_samples=self.minisample,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            scale=self.scale,            
            conditioning_length=self.conditioning_length,
            covariate_hidden_size=self.covariate_hidden_size,
            diff_steps=self.diff_steps,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            residual_layers=self.residual_layers,
            residual_channels=self.residual_channels,
            dilation_cycle_length=self.dilation_cycle_length,
            )
        self.model = self.model.to(self.device)

    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()

        # no decoder input
        # label_len = 1
        noise, pred_noise = self.model(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, train=True)
        return noise, pred_noise

    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()

        # no decoder input
        # label_len = 1
        
        samples = []
        sample_batches = max(1, (self.num_samples + self.minisample - 1) // self.minisample)
        for _ in range(sample_batches):
            # B, S, O, N
            output = self.model(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, train=False)
            samples.append(output)
            
        output = torch.concat(samples, dim=1) # B, S, O, N
        output = output[:, : self.num_samples]
        output = output.permute(0, 2, 3, 1) # B, O, N, S
        return output, batch_y


if __name__ == "__main__":
    import fire
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    fire.Fire(TimeGradForecast)
