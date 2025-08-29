from typing import Optional

import clip
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.utils.rotation2xyz import Rotation2xyz


class MDM(nn.Module):
    def __init__(
        self,
        njoints: int = 263,
        nfeats: int = 1,
        num_actions: int = 0,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        dataset: str = "amass",
        clip_version: str = None,
        cond: Optional[DictConfig] = None,
        embed: Optional[DictConfig] = None,
    ):
        super(MDM, self).__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.input_feats = self.njoints * self.nfeats

        # Conditioning settings
        self.cond_mode = cond.cond_mode
        self.cond_mask_prob = cond.cond_mask_prob
        self.mask_frames = cond.mask_frames

        self.input_process = InputProcess(self.input_feats, self.latent_dim)

        # Embedding for denoising timestep
        self.emb_policy = embed.emb_policy
        self.sequence_pos_encoder = PositionalEncoding(
            self.latent_dim, max_len=embed.pos_embed_max_len
        )
        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        # Conditioning embedding
        if self.cond_mode != "no_cond":
            if "text" in self.cond_mode:
                # We support CLIP encoder and DistilBERT
                print("EMBED TEXT")
                print("Loading CLIP...")
                self.clip_model = self.load_and_freeze_clip(clip_version)
                self.encode_text = self.clip_encode_text
                self.clip_dim = 512
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)

            if "action" in self.cond_mode:
                print("EMBED ACTION")
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)

        # Transformer-based denoiser
        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=num_layers
        )

        self.output_process = OutputProcess(
            self.input_feats, self.latent_dim, self.njoints, self.nfeats
        )

        self.rot2xyz = Rotation2xyz(device="cpu", dataset=dataset)

    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device="cpu", jit=False
        )  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                1, bs, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20
        default_context_length = 77
        context_length = max_text_len + 2  # start_token + 20 + end_token
        assert context_length < default_context_length
        texts = clip.tokenize(
            raw_text, context_length=context_length, truncate=True
        ).to(
            device
        )  # [bs, context_length] # if n_tokens > context_length -> will truncate
        # print('texts', texts.shape)
        zero_pad = torch.zeros(
            [texts.shape[0], default_context_length - context_length],
            dtype=texts.dtype,
            device=texts.device,
        )
        texts = torch.cat([texts, zero_pad], dim=1)
        # print('texts after pad', texts.shape, texts)
        return self.clip_model.encode_text(texts).float().unsqueeze(0)

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get("uncond", False)
        if "text" in self.cond_mode:
            if "text_embed" in y.keys():  # caching option
                enc_text = y["text_embed"]
            else:
                enc_text = self.encode_text(y["text"])
            if type(enc_text) == tuple:
                enc_text, text_mask = enc_text
                if (
                    text_mask.shape[0] == 1 and bs > 1
                ):  # casting mask for the single-prompt-for-all case
                    text_mask = torch.repeat_interleave(text_mask, bs, dim=0)
            text_emb = self.embed_text(
                self.mask_cond(enc_text, force_mask=force_mask)
            )  # casting mask for the single-prompt-for-all case
            if self.emb_policy == "add":
                emb = text_emb + time_emb
            else:
                emb = torch.cat([time_emb, text_emb], dim=0)
                text_mask = torch.cat(
                    [torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1
                )
        if "action" in self.cond_mode:
            action_emb = self.embed_action(y["action"])
            emb = time_emb + self.mask_cond(action_emb, force_mask=force_mask)
        if self.cond_mode == "no_cond":
            # unconstrained
            emb = time_emb

        x = self.input_process(x)

        # TODO - move to collate
        frames_mask = None
        is_valid_mask = (
            y["mask"].shape[-1] > 1
        )  # Don't use mask with the generate script
        if self.mask_frames and is_valid_mask:
            frames_mask = torch.logical_not(
                y["mask"][..., : x.shape[0]].squeeze(1).squeeze(1)
            ).to(device=x.device)
            step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)
            frames_mask = torch.cat([step_mask, frames_mask], dim=1)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, d]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
