from typing import Optional

import clip
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.BERT.BERT_encoder import load_bert
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
        clip_dim: int = 512,
        text_encoder_type: str = "bert",  # clip, bert
        arch: str = "trans_dec",  # trans_enc, trans_dec
        emb_trans_dec: bool = False,
        cond: Optional[DictConfig] = None,
        embed: Optional[DictConfig] = None,
    ):
        super(MDM, self).__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.input_feats = self.njoints * self.nfeats
        self.clip_dim = clip_dim
        self.text_encoder_type = text_encoder_type
        self.arch = arch
        self.emb_trans_dec = emb_trans_dec

        # Conditioning settings
        self.cond_mode = cond.cond_mode
        self.cond_mask_prob = cond.cond_mask_prob
        self.mask_frames = cond.mask_frames

        self.input_process = InputProcess(self.input_feats, self.latent_dim)

        # Embedding for denoising timestep
        self.emb_policy = embed.emb_policy
        self.sequence_pos_encoder = PositionalEncoding(
            self.latent_dim, dropout, max_len=embed.pos_embed_max_len
        )
        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        # Conditioning embedding
        if self.cond_mode != "no_cond":
            if "text" in self.cond_mode:
                # We support CLIP encoder and DistilBERT
                print("EMBED TEXT")

                if self.text_encoder_type == "clip":
                    print("Loading CLIP...")
                    self.clip_version = clip_version
                    self.clip_model = self.load_and_freeze_clip(clip_version)
                    self.encode_text = self.clip_encode_text
                elif self.text_encoder_type == "bert":
                    # assert self.arch == 'trans_dec'
                    # assert self.emb_trans_dec == False # passing just the time embed so it's fine
                    print("Loading BERT...")
                    # bert_model_path = 'model/BERT/distilbert-base-uncased'
                    bert_model_path = "distilbert/distilbert-base-uncased"
                    self.clip_model = load_bert(
                        bert_model_path
                    )  # Sorry for that, the naming is for backward compatibility
                    self.encode_text = self.bert_encode_text
                    self.clip_dim = 768
                else:
                    raise ValueError("We only support [CLIP, BERT] text encoders")

                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)

            if "action" in self.cond_mode:
                print("EMBED ACTION")
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)

            if "start" and "goal" in self.cond_mode:
                print("EMBED START and GOAL")
                self.embed_start = nn.Linear(self.input_feats, self.latent_dim)
                self.embed_goal = nn.Linear(self.input_feats, self.latent_dim)
                self.embed_start_goal = nn.Linear(self.input_feats * 2, self.latent_dim)

            if self.emb_policy == "ca":
                assert (
                    "start" in self.cond_mode and "goal" in self.cond_mode
                ), "CA policy requires both start and goal conditions"
                print("EMBED Cross-Attention")
                self.embed_ca = nn.TransformerDecoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                )

        # Transformer-based denoiser
        if self.arch == "trans_enc":
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
        elif self.arch == "trans_dec":
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
            )
            self.seqTransDecoder = nn.TransformerDecoder(
                seqTransDecoderLayer, num_layers=num_layers
            )
        else:
            raise ValueError(
                "Please choose correct architecture [trans_enc, trans_dec]"
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

    def get_mask(self, cond, force_mask=False):
        bs = cond.shape[-2]
        if force_mask:
            return torch.ones(bs, dtype=torch.bool, device=cond.device)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).to(
                torch.bool
            )  # 1-> use null_cond, 0-> use real cond
            return mask
        else:
            return torch.zeros(bs, dtype=torch.bool, device=cond.device)

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

    def bert_encode_text(self, raw_text):
        # enc_text = self.clip_model(raw_text)
        # enc_text = enc_text.permute(1, 0, 2)
        # return enc_text
        enc_text, mask = self.clip_model(
            raw_text
        )  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        mask = (
            ~mask
        )  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text, mask

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

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
            if self.emb_policy == "add" or self.emb_policy == "ca":
                emb = emb + text_emb
            else:
                emb = torch.cat([emb, text_emb], dim=0)
                text_mask = torch.cat(
                    [torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1
                )
        if "action" in self.cond_mode:
            action_emb = self.embed_action(y["action"])
            emb = emb + self.mask_cond(action_emb, force_mask=force_mask)

        if "start" in self.cond_mode and "goal" in self.cond_mode:
            start_emb = self.embed_start(y["start"]).unsqueeze(0)
            goal_emb = self.embed_goal(y["goal"]).unsqueeze(0)
            start_goal = torch.cat(
                [y["start"], y["goal"]], dim=-1
            )  # [bs, input_feats*2]
            start_goal_emb = self.embed_start_goal(start_goal).unsqueeze(0)  # [1, bs, d]

            if self.emb_policy == "add":
                emb = emb + start_emb + goal_emb
            # elif self.emb_policy == "cat":
            #     emb = torch.cat([time_emb, self.mask_cond(start_emb, force_mask=force_mask), self.mask_cond(goal_emb, force_mask=force_mask)], dim=0)
            elif self.emb_policy == "ca":
                memory = start_goal_emb
                emb = emb + self.embed_ca(emb, memory)

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
            if self.emb_trans_dec or self.arch == "trans_enc":
                step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device)
                frames_mask = torch.cat([step_mask, frames_mask], dim=1)

        # adding the timestep embed
        if self.arch == "trans_enc":
            # adding the timestep embed
            num_tokens_cond = emb.shape[0]
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)[
                num_tokens_cond:
            ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == "trans_dec":
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

            if self.text_encoder_type == "clip":
                output = self.seqTransDecoder(
                    tgt=xseq, memory=emb, tgt_key_padding_mask=frames_mask
                )
            elif self.text_encoder_type == "bert":
                output = self.seqTransDecoder(
                    tgt=xseq,
                    memory=emb,
                    memory_key_padding_mask=text_mask,
                    tgt_key_padding_mask=frames_mask,
                )  # Rotem's bug fix
            else:
                raise ValueError()
            
            if self.emb_trans_dec:
                output = output[1:]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        # not used in the final model
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


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
