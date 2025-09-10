from typing import Optional

import pytorch_lightning as pl
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

from datamodules.datasets.adt_dataset import ADTDataset
from datamodules.utils.collate_fn import (post_train_collate, t2m_collate,
                                          t2m_eval_collate, t2m_prefix_collate)


def get_collate_fn(split: str = "train", pred_len: int = 0, batch_size: int = 32):
    if split in ["post-train"]:
        return lambda x: post_train_collate(x, batch_size)
    if split in ["train", "predict"]:
        if pred_len > 0:
            return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
        return lambda x: t2m_collate(x, batch_size)
    if split in ["eval", "test"]:
        return t2m_eval_collate


class ADTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        pred_len: int = 0,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super(ADTDataModule, self).__init__()
        self.cfg = cfg
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        return NotImplementedError

    def setup(self, stage: Optional[str] = None):
        # Args:
        #     stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        if stage == "fit" or stage is None:
            self.train_dataset = ADTDataset(split="train", **self.cfg)
            self.collate_fn_train = get_collate_fn(
                split="post-train", pred_len=self.pred_len, batch_size=self.batch_size
            )

        elif stage == "predict":
            self.predict_dataset = ADTDataset(split="test", **self.cfg)
            self.collate_fn_predict = get_collate_fn(split="predict", pred_len=self.pred_len, batch_size=self.batch_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_train,
            shuffle=True,
            pin_memory=True,
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_predict,
            shuffle=False,
            pin_memory=True,
        )
