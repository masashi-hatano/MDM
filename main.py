import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config_mdm.yaml")
def main(cfg):
    # initialize random seeds
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # data module
    data_module = instantiate(cfg.data.data_module, cfg=cfg.data.dataset)

    # model
    model = instantiate(cfg.trainer)

    if torch.cuda.is_available() and len(cfg.devices):
        print(f"Using {len(cfg.devices)} GPUs !")

    train_logger = loggers.TensorBoardLogger("tensor_board", default_hp_metric=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="model_{step:09d}",
        every_n_train_steps=cfg.save_interval,
        save_top_k=-1,
    )

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        max_steps=cfg.max_steps,
        logger=train_logger,
        callbacks=[checkpoint_callback],
        detect_anomaly=False,
        use_distributed_sampler=False,
    )

    if cfg.train:
        trainer.fit(model, data_module)
        print(trainer.callback_metrics)

    if cfg.test:
        logging.basicConfig(level=logging.DEBUG)
        trainer.test(model, data_module)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
