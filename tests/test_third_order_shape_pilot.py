from pathlib import Path

import yaml

from src.analysis import third_order_shape_pilot
from src.analysis.third_order_shape_pilot import experiment_kwargs_from_config


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
