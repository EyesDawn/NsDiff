from pathlib import Path

import numpy as np
import properscoring as ps
import torch
import yaml

from src.analysis import third_order_shape_pilot
from src.analysis.third_order_shape_pilot import (
    experiment_kwargs_from_config,
    iter_microbatches,
)
from src.metrics.CRPS import ensemble_crps_sum


def test_e2e_dataset_field_is_mapped_to_ireflow_data_field():
    config_path = Path("configs/iReflow_e2e/etth1.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    kwargs = experiment_kwargs_from_config(config)

    assert kwargs["data"] == "ETTh1"


def test_process_title_identifies_dataset_gpu_and_run_count(monkeypatch):
    titles = []
    monkeypatch.setattr(third_order_shape_pilot, "setproctitle", titles.append)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")

    title = third_order_shape_pilot.set_process_title(["Traffic"], 5)

    assert title == "iReflow-L3:Traffic:gpu=5:runs=5"
    assert titles == [title]


def test_chunked_crps_matches_properscoring_for_equal_weight_ensembles():
    rng = np.random.default_rng(42)
    truth = rng.normal(size=17)
    predictions = rng.normal(size=(17, 5))

    result = ensemble_crps_sum(truth, predictions, observation_chunk_size=3)

    assert np.isclose(result, ps.crps_ensemble(truth, predictions).sum())


def test_iter_microbatches_preserves_order_and_handles_last_short_batch():
    tensors = tuple(torch.arange(5).unsqueeze(1) + offset for offset in range(4))

    batches = list(iter_microbatches(*tensors, micro_batch_size=2))

    assert [batch[0].flatten().tolist() for batch in batches] == [[0, 1], [2, 3], [4]]
    assert [batch[3].flatten().tolist() for batch in batches] == [[3, 4], [5, 6], [7]]


def test_iter_microbatches_with_size_one_is_one_window_at_a_time():
    tensors = tuple(torch.arange(3).unsqueeze(1) + offset for offset in range(4))

    batches = list(iter_microbatches(*tensors, micro_batch_size=1))

    assert len(batches) == 3
    assert [batch[2].item() for batch in batches] == [2, 3, 4]
