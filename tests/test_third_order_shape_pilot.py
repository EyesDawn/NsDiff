from pathlib import Path

import yaml

from src.analysis.third_order_shape_pilot import experiment_kwargs_from_config


def test_e2e_dataset_field_is_mapped_to_ireflow_data_field():
    config_path = Path("configs/iReflow_e2e/etth1.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    kwargs = experiment_kwargs_from_config(config)

    assert kwargs["data"] == "ETTh1"
