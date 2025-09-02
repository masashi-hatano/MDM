from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim import AdamW

from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from models.cfg_sampler import ClassifierFreeSampleModel
from models.mdm import MDM
from utils.logger import Logger


class MDMTrainer(pl.LightningModule):
    def __init__(
        self,
        model: MDM,
        diffusion: GaussianDiffusion,
        optimizer: DictConfig,
        use_ema: bool = True,
        avg_model_beta: float = 0.9999,
        gen_guidance_param: float = 1.0,
        schedule_sampler_type: str = "uniform",
    ):
        super(MDMTrainer, self).__init__()
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer

        # EMA model
        self.use_ema = use_ema
        self.model_avg = None
        self.avg_model_beta = avg_model_beta
        if self.use_ema:
            self.model_avg = deepcopy(self.model)
        self.model_for_eval = self.model_avg if self.use_ema else self.model
        if gen_guidance_param != 1:
            self.model_for_eval = ClassifierFreeSampleModel(
                self.model_for_eval
            )  # wrapping model with the classifier-free sampler

        # model parameters
        self.cond_mode = model.cond_mode

        # optimizer parameters
        self.lr = optimizer.lr
        self.weight_decay = optimizer.weight_decay
        self.betas = optimizer.adam_betas

        # schedule sampler
        self.schedule_sampler = create_named_schedule_sampler(
            schedule_sampler_type, diffusion
        )
        
        # initialization
        self.training_logger = Logger()

    def training_step(self, batch, batch_idx):
        motion, cond = batch

        cond["y"] = {
            key: val.to(self.device) if torch.is_tensor(val) else val
            for key, val in cond["y"].items()
        }
        t, weights = self.schedule_sampler.sample(motion.shape[0], motion.device)
        
        losses = self.diffusion.training_losses(
            self.model,
            motion,  # [bs, njoints, nfeats, nframes]
            t,  # [bs](int) sampled denoising timesteps
            cond,
        )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

        loss = (losses["loss"] * weights).mean()
        
        # update training outputs
        self.training_logger.update({"loss": loss.detach().cpu()})
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # logging
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        for k, v in self.training_logger.logs.items():
            self.log(f"{k}", v, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        # update the average model using exponential moving average
        if self.use_ema:
            params = self.model.parameters()
            for param, avg_param in zip(params, self.model_avg.parameters()):
                # avg = avg + (param - avg) * (1 - alpha)
                # avg = avg + param * (1 - alpha) - (avg - alpha * avg)
                # avg = alpha * avg + param * (1 - alpha)
                avg_param.data.mul_(self.avg_model_beta).add_(
                    param.data, alpha=1 - self.avg_model_beta
                )
                
    def test_step(self, batch, batch_idx):
        motion, cond = batch
        cond["y"] = {
            key: val.to(self.device) if torch.is_tensor(val) else val
            for key, val in cond["y"].items()
        }
        
        # replicate motion and cond for multiple generations
        motion = motion.repeat_interleave(self.num_replicate, dim=0)
        cond = {key: val.repeat_interleave(self.num_replicate, dim=0) if torch.is_tensor(val) else val for key, val in cond.items()}

        samples = self.diffusion.p_sample_loop(
            self.model_for_eval,
            (motion.shape[0], motion.shape[1], motion.shape[2], motion.shape[3]),
            cond,
            progress=True,
        )
                
    def on_save_checkpoint(self, checkpoint):
        # remove the model_for_eval from checkpoint to save space
        filtered_state_dict = {
            k: v for k, v in checkpoint["state_dict"].items() if "model_for_eval" not in k
        }
        checkpoint["state_dict"] = filtered_state_dict

    def configure_optimizers(self):
        if self.optimizer.name == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
        else:
            raise NotImplementedError(f"{self.optimizer.name} is not supported.")
        return [optimizer]
