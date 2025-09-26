import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

from datamodules.utils.word_vectorizer import WordVectorizer


class HumanMLDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str = "train",
        data_dir: str = "dataset/humanml",
        motion_dir: str = "dataset/humanml/HumanML3D/new_joint_vecs",
        text_dir: str = "dataset/humanml/HumanML3D/texts",
        mean: str = "dataset/humanml/HumanML3D/Mean.npy",
        std: str = "dataset/humanml/HumanML3D/Std.npy",
        w_vectorizer: Optional[DictConfig] = None,
        max_len: int = 20,
        fixed_len: int = 0,
        max_motion_len: int = 196,
        min_motion_len: int = 40,
        max_text_len: int = 20,
        unit_len: int = 4,
        disable_offset_aug: bool = False,
        dataset_name: str = "t2m",
        use_cache: bool = True,
    ):
        super(HumanMLDataset, self).__init__()
        # Initialize dataset paths
        root = get_original_cwd()
        self.split = split
        self.data_dir = Path(root, data_dir)
        motion_dir = Path(root, motion_dir)
        text_dir = Path(root, text_dir)
        mean_path = Path(root, mean)
        std_path = Path(root, std)

        # instantiated word vectorizer
        self.w_vectorizer = WordVectorizer(**w_vectorizer)

        # other settings
        self.max_len = max_len
        self.fixed_len = fixed_len
        if fixed_len > 0:
            self.max_len = fixed_len
        self.pointer = 0
        self.max_motion_len = max_motion_len
        self.min_motion_len = min_motion_len
        self.max_text_len = max_text_len
        self.unit_len = unit_len
        self.disable_offset_aug = disable_offset_aug

        cache_path = self.data_dir / Path(dataset_name + "_" + split + ".npy")

        if use_cache and cache_path.exists():
            print(f"Loading motions from cache file [{cache_path}]...")
            _cache = np.load(cache_path, allow_pickle=True).item()
            name_list, length_list, data_dict = (
                _cache["name_list"],
                _cache["length_list"],
                _cache["data_dict"],
            )
        else:
            name_list, length_list, data_dict = self.create_cache(
                motion_dir=motion_dir,
                text_dir=text_dir,
                cache_path=cache_path,
                min_motion_len=min_motion_len,
            )

        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_len)
        print(f"Dataset {split} size: {len(self.data_dict)}")

    def create_cache(self, motion_dir, text_dir, cache_path, min_motion_len):
        data_dict = {}
        new_name_list = []
        length_list = []

        with open(self.data_dir / Path("HumanML3D", f"{self.split}.txt"), "r") as f:
            id_list = [line.strip() for line in f.readlines()]

        for name in tqdm(id_list):
            try:
                motion = np.load(Path(motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with open(Path(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20) : int(to_tag * 20)]
                                if (len(n_motion)) < min_motion_len or (
                                    len(n_motion) >= 200
                                ):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                    + "_"
                                    + name
                                )
                                while new_name in data_dict:
                                    new_name = (
                                        random.choice("ABCDEFGHIJKLMNOPQRSTUVW")
                                        + "_"
                                        + name
                                    )
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(
                                    line_split[2],
                                    line_split[3],
                                    f_tag,
                                    to_tag,
                                    name,
                                )

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1])
        )
        print(f"Saving motions to cache file [{cache_path}]...")
        np.save(
            cache_path,
            {
                "name_list": name_list,
                "length_list": length_list,
                "data_dict": data_dict,
            },
        )
        return name_list, length_list, data_dict

    def reset_max_len(self, length):
        # if length is larger than max_motion_len, discard the larger motion_len samples
        assert length <= self.max_motion_len
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_len = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        key = self.name_list[idx]
        data = self.data_dict[key]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        
        # caption = "C walks towards the hallway then bends her right arm and gestures her right hand as she continues to talk with her peer." # DEBUG
        # caption = "C moves to pick object." # DEBUG

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_len < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_len - 1) * self.unit_len
        elif coin2 == "single":
            m_length = (m_length // self.unit_len) * self.unit_len

        original_length = None
        if self.fixed_len > 0:
            # Crop fixed_len
            original_length = m_length
            m_length = self.fixed_len

        idx = random.randint(0, len(motion) - m_length)
        if self.disable_offset_aug:
            idx = random.randint(0, self.unit_len)
        motion = motion[idx : idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_len:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.max_motion_len - m_length, motion.shape[1])),
                ],
                axis=0,
            )

        length = (original_length, m_length) if self.fixed_len > 0 else m_length

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            length,
            "_".join(tokens),
            key,
        )
