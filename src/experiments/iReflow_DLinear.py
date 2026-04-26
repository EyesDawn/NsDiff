from dataclasses import dataclass
import os

import setproctitle
import torch

from src.experiments.iReflow import iReflowExp
from src.models.iReflow_DLinear import iReflowDLinear


def _strip_module_prefix(state_dict):
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


class DLinearBackboneExperimentMixin:
    estimator_run_group = "estimator_dlinear"

    def _setup_estimator_run_paths(self, setting: str, seed: int):
        dataset = getattr(self, "dataset_type", getattr(self, "data", "custom"))
        run_dir = os.path.join(
            "./results", "runs", self.estimator_run_group, dataset, setting, f"seed_{seed}"
        )
        self.run_save_dir = run_dir
        os.makedirs(self.run_save_dir, exist_ok=True)
        self.run_checkpoint_filepath = os.path.join(self.run_save_dir, "run_checkpoint.pth")
        self.best_checkpoint_filepath = os.path.join(self.run_save_dir, "best_model.pth")

        if hasattr(self, "early_stopping") and hasattr(self.early_stopping, "path"):
            self.early_stopping.path = self.best_checkpoint_filepath

    def _load_uncertainty_estimator(self, setting):
        dataset = getattr(self, "dataset_type", getattr(self, "data", "custom"))
        seed = getattr(self, "current_seed", 42)
        stage2_run_dir = os.path.join(
            "./results",
            "runs",
            self.estimator_run_group,
            dataset,
            setting,
            f"seed_{seed}",
        )
        best_model_path = os.path.join(stage2_run_dir, "best_model.pth")

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(
                f"Stage 2 DLinear checkpoint not found at {best_model_path}. "
                "Please run pretrain_uncertainty_estimator_DLinear.py first."
            )

        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
        checkpoint = _strip_module_prefix(checkpoint)

        uncertainty_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("uncertainty_estimator."):
                uncertainty_state_dict[key[len("uncertainty_estimator.") :]] = value

        if not uncertainty_state_dict:
            raise ValueError(
                "Could not find uncertainty_estimator weights in the Stage 2 DLinear checkpoint."
            )

        self.model.uncertainty_estimator.load_state_dict(uncertainty_state_dict, strict=True)
        print(f"Loaded pretrained Uncertainty Estimator from {best_model_path}")

    def _load_itransformer_only(self, setting):
        path = os.path.join(self.checkpoints, setting)
        best_model_path = os.path.join(path, "checkpoint.pth")

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(
                f"DLinear checkpoint not found at {best_model_path}. "
                "Please ensure the DLinear Stage 1 model has been trained and saved."
            )

        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        checkpoint = _strip_module_prefix(checkpoint)

        direct_dlinear_keys = (
            "decompsition.moving_avg.avg.weight",
            "Linear_Seasonal.weight",
            "Linear_Trend.weight",
            "feature_projection.weight",
        )

        if any(key.startswith("itransformer.") for key in checkpoint.keys()):
            dlinear_state_dict = {
                key[len("itransformer.") :]: value
                for key, value in checkpoint.items()
                if key.startswith("itransformer.")
            }
        elif any(key in checkpoint for key in direct_dlinear_keys) or any(
            key.startswith("Linear_Seasonal")
            or key.startswith("Linear_Trend")
            or key.startswith("feature_projection")
            or key.startswith("decompsition.")
            for key in checkpoint.keys()
        ):
            dlinear_state_dict = checkpoint
        else:
            raise ValueError(
                "Could not extract DLinear weights from checkpoint. "
                f"Available keys: {list(checkpoint.keys())[:10]}"
            )

        missing_keys, unexpected_keys = self.model.itransformer.load_state_dict(
            dlinear_state_dict, strict=False
        )
        if missing_keys:
            print(f"Warning: Missing keys in DLinear backbone: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in DLinear backbone: {unexpected_keys}")
        print(f"Loaded DLinear backbone from {best_model_path}")

    def _freeze_itransformer(self):
        if self.is_training == 1:
            for param in self.model.itransformer.parameters():
                param.requires_grad = False
            print("DLinear backbone parameters frozen after loading pretrained weights")


@dataclass
class iReflowDLinearExp(DLinearBackboneExperimentMixin, iReflowExp):
    model: str = "iReflow_DLinear"
    model_type: str = "iReflow_DLinear"
    checkpoints: str = "./results/runs/DLinear/"
    moving_avg: int = 25
    individual: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.model_configs.enc_in = self.enc_in
        self.model_configs.moving_avg = self.moving_avg
        self.model_configs.individual = self.individual

    def _init_model(self):
        self.model = iReflowDLinear(self.model_configs).to(self.device)


if __name__ == "__main__":
    setproctitle.setproctitle("iReflow_DLinear_main")

    import fire

    fire.Fire(iReflowDLinearExp)
