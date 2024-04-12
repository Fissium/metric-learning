import logging

import hydra
import torch
from omegaconf import DictConfig
from oml.const import HYDRA_BEHAVIOUR
from oml.lightning.pipelines.train import extractor_training_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


@hydra.main(
    config_path="./configs/",
    config_name="train_vit.yaml",
    version_base=HYDRA_BEHAVIOUR,
)
def main_hydra(cfg: DictConfig) -> None:
    logger.info("Traning model...")
    extractor_training_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
