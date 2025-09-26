from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.optim import AdamW

from datamodules.utils.recover import recover_from_ric
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from models.cfg_sampler import ClassifierFreeSampleModel
from models.mdm import MDM
from models.utils.load_model import load_saved_model
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
        ckpt_path: Union[str, None] = None,
    ):
        super(MDMTrainer, self).__init__()
        self.root = get_original_cwd()
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer

        if ckpt_path is not None:
            if ckpt_path.endswith(".pt"):
                self.model = load_saved_model(
                    self.model, Path(self.root, ckpt_path), use_avg=use_ema
                )
                print(f"Model is initialized from {ckpt_path}")
            elif ckpt_path.endswith(".ckpt"):
                p = torch.load(
                    Path(self.root, ckpt_path), map_location="cpu", weights_only=False
                )
                state_dict = p["state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model_avg."):
                        new_k = k.replace("model_avg.", "")
                        new_state_dict[new_k] = v
                self.model.load_state_dict(new_state_dict, strict=False)
                print(f"Model is initialized from {ckpt_path}")

        # EMA model
        self.use_ema = use_ema
        self.model_avg = None
        self.avg_model_beta = avg_model_beta
        if self.use_ema:
            self.model_avg = deepcopy(self.model)
        self.model_for_eval = self.model_avg if self.use_ema else self.model
        self.gen_guidance_param = gen_guidance_param
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

        # loss
        self.mse_loss = nn.MSELoss()

        # initialization
        self.training_logger = Logger()
        self.losses = []
        self.save = True

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
            self.log(
                f"{k}", v, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

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

    def predict_step(self, batch, batch_idx):
        motions, cond = batch

        # Get text prompts and lengths
        keys = cond["y"]["db_key"]
        lengths = cond["y"]["lengths"].cpu().numpy()

        # Apply guidance scale if using classifier-free guidance
        if self.gen_guidance_param != 1:
            cond["y"]["scale"] = (
                torch.ones(motions.shape[0], device=motions.device)
                * self.gen_guidance_param
            )

        # cond["y"]['uncond'] = True # ensure uncond is False for all samples

        # Generate motions
        sample = self.diffusion.p_sample_loop(
            self.model_for_eval,
            motions.shape,
            clip_denoised=False,
            model_kwargs=cond,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        sample_np = sample.squeeze(-2).cpu().numpy()
        motions_np = motions.squeeze(-2).cpu().numpy()

        mse_loss = self.mse_loss(sample, motions)

        mpjpe_list, mpjpe_start_list, mpjpe_goal_list = self.calc_mpjpe(
            motions_np, sample_np, lengths
        )

        outputs = {
            "mse_loss": mse_loss.item(),
            "mpjpe": mpjpe_list,
            "mpjpe_start": mpjpe_start_list,
            "mpjpe_goal": mpjpe_goal_list,
        }

        self.losses.append(outputs)

        # Save each motion individually
        if self.save == True:
            for i in range(len(sample_np)):
                # Get motion data: [n_frames, 263]
                motion = sample_np[i]  # [263, n_frames]
                length = lengths[i]
                key = keys[i]

                motion = motion.transpose(1, 0)  # [n_frames, 263]

                # Trim motion to actual length (remove padding)
                motion_trimmed = motion[:, :]

                motion_trimmed_denorm = (
                    self.trainer.predict_dataloaders.dataset.inv_transform(
                        motion_trimmed
                    )
                )

                # Save motion
                save_path = Path(
                    self.root, "predictions", "humanml_trans_dec_512_bert_50steps_600K_with_text", f"{key}.npy"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(save_path, motion_trimmed_denorm)

    def on_predict_end(self):
        # Aggregate and log final metrics
        if len(self.losses) == 0:
            print("No predictions were made.")
            return

        avg_mse_loss = np.mean([loss["mse_loss"] for loss in self.losses])
        avg_mpjpe = np.mean([item for loss in self.losses for item in loss["mpjpe"]])
        avg_mpjpe_start = np.mean(
            [item for loss in self.losses for item in loss["mpjpe_start"]]
        )
        avg_mpjpe_goal = np.mean(
            [item for loss in self.losses for item in loss["mpjpe_goal"]]
        )

        print(f"Average MSE Loss over all predictions: {avg_mse_loss}")
        print(f"Average MPJPE over all predictions: {avg_mpjpe}")
        print(f"Average MPJPE at start frame: {avg_mpjpe_start}")
        print(f"Average MPJPE at goal frame: {avg_mpjpe_goal}")

        # Clear losses for next prediction phase
        self.losses = []

    def test_step(self, batch, batch_idx):
        motion, cond = batch
        cond["y"] = {
            key: val.to(self.device) if torch.is_tensor(val) else val
            for key, val in cond["y"].items()
        }

        # replicate motion and cond for multiple generations
        motion = motion.repeat_interleave(self.num_replicate, dim=0)
        cond = {
            key: (
                val.repeat_interleave(self.num_replicate, dim=0)
                if torch.is_tensor(val)
                else val
            )
            for key, val in cond.items()
        }

        samples = self.diffusion.p_sample_loop(
            self.model_for_eval,
            (motion.shape[0], motion.shape[1], motion.shape[2], motion.shape[3]),
            cond,
            progress=True,
        )

    def on_save_checkpoint(self, checkpoint):
        # remove the model_for_eval from checkpoint to save space
        filtered_state_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if "model_for_eval" not in k and "clip_model" not in k
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

    def calc_mpjpe(self, gt, pred, lengths):
        """Calculate MPJPE (Mean Per Joint Position Error) between ground truth and predicted motions.
        Args:
            gt: Ground truth motion, shape (B, C, T)
            pred: Predicted motion, shape (B, C, T)
            lengths: Lengths of each motion in the batch, shape (B,)
        Returns:
            mpjpe: Mean MPJPE over the entire sequence
            mpjpe_start: MPJPE at the start frame
            mpjpe_goal: MPJPE at the goal frame
        """
        # [bs, 263, n_frames] -> [bs, n_frames, 263] -> denormalize
        b, c, f = gt.shape
        gt = gt.transpose(0, 2, 1)  # [bs, n_frames, 263]
        pred = pred.transpose(0, 2, 1)  # [bs, n_frames, 263]
        gt = self.trainer.predict_dataloaders.dataset.inv_transform(gt)
        pred = self.trainer.predict_dataloaders.dataset.inv_transform(pred)

        mpjpe_list = []
        mpjpe_start_list = []
        mpjpe_goal_list = []

        for i in range(b):
            joints_gt = recover_from_ric(
                torch.tensor(gt[i], dtype=torch.float32)
            ).numpy()  # [n_frames, 22, 3]
            joints_pred = recover_from_ric(
                torch.tensor(pred[i], dtype=torch.float32)
            ).numpy()  # [n_frames, 22, 3]

            joints_gt_i = joints_gt[: lengths[i]]
            joints_pred_i = joints_pred[: lengths[i]]
            mpjpe = np.linalg.norm(joints_gt_i - joints_pred_i, axis=-1).mean()
            mpjpe_start = np.linalg.norm(
                joints_gt_i[0:1] - joints_pred_i[0:1], axis=-1
            ).mean()
            mpjpe_goal = np.linalg.norm(
                joints_gt_i[-2:-1] - joints_pred_i[-2:-1], axis=-1
            ).mean()
            mpjpe_list.append(mpjpe)
            mpjpe_start_list.append(mpjpe_start)
            mpjpe_goal_list.append(mpjpe_goal)
        return mpjpe_list, mpjpe_start_list, mpjpe_goal_list

    def calc_jpe(self, gt, pred, mask):
        # pred: B X T X 22 X 3 or T X 22 X 3
        # gt: B X T X 22 X 3 or T X 22 X 3
        # mask: B X T or T
        diff = pred - gt
        diff = diff**2
        diff = diff.sum(axis=-1)
        diff = diff**0.5
        jpe = diff * mask
        jpe = jpe[jpe != 0]
        return jpe
