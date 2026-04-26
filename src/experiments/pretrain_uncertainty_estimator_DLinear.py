from dataclasses import dataclass

import setproctitle

from src.experiments.iReflow_DLinear import DLinearBackboneExperimentMixin
from src.experiments.pretrain_uncertainty_estimator import (
    UncertaintyEstimatorPretrainExp,
)
from src.models.iReflow_DLinear import iReflowDLinear


@dataclass
class UncertaintyEstimatorPretrainDLinearExp(
    DLinearBackboneExperimentMixin, UncertaintyEstimatorPretrainExp
):
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

    def _freeze_itransformer(self):
        for param in self.model.itransformer.parameters():
            param.requires_grad = False
        print("DLinear backbone parameters frozen for Stage 2")


if __name__ == "__main__":
    setproctitle.setproctitle("UncertaintyEstimator_Pretrain_DLinear")

    import fire

    fire.Fire(UncertaintyEstimatorPretrainDLinearExp)
