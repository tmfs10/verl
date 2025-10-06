import os

import hydra
from omegaconf import OmegaConf

from .config import ESConfig
from .fsdp_trainer import EsFsdpTrainer


@hydra.main(config_path="../config", config_name="es_trainer", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    config = ESConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))

    # Ensure torch distributed env is ready (handled in trainer)
    trainer = EsFsdpTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

