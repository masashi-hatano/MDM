import json
import random
from pathlib import Path

import numpy as np
import torch
from hydra.utils import get_original_cwd
from tqdm import tqdm

from datamodules.utils.recover import recover_from_ric


class ADTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str = "train",
        data_dir: str = "dataset/ADT",
        motion_dir: str = "dataset/ADT/new_joint_vecs",
        text_dir: str = "dataset/ADT/texts",
        mean: str = "dataset/humanml/HumanML3D/Mean.npy",
        std: str = "dataset/humanml/HumanML3D/Std.npy",
        max_len: int = 20,
        fixed_len: int = 0,
        max_motion_len: int = 256,
        min_motion_len: int = 60,
        unit_len: int = 4,
        disable_offset_aug: bool = False,
        dataset_name: str = "adt",
        use_cache: bool = True,
    ):
        super(ADTDataset, self).__init__()
        # Initialize dataset paths
        root = get_original_cwd()
        self.split = split
        self.data_dir = Path(root, data_dir)
        motion_dir = Path(root, motion_dir)
        self.split_json_path = self.data_dir / Path("split.json")
        text_dir = Path(root, text_dir)
        mean_path = Path(root, mean)
        std_path = Path(root, std)

        # other settings
        self.max_len = max_len
        self.fixed_len = fixed_len
        if fixed_len > 0:
            self.max_len = fixed_len
        self.pointer = 0
        self.max_motion_len = max_motion_len
        self.min_motion_len = min_motion_len
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

        split_json = json.load(open(self.split_json_path))[self.split]

        for motion_path in tqdm(motion_dir.glob("*.npy")):
            name = motion_path.stem
            video_uid = motion_path.stem.split("_")[:-2]
            video_uid = "_".join(video_uid)
            if video_uid not in split_json:
                continue
            text_data = []
            with open(Path(text_dir, name + ".txt")) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip()
                    caption = line_split

                    text_dict["caption"] = caption
                    text_data.append(text_dict)
            try:
                motion = np.load(motion_path)
                if (len(motion)) < min_motion_len:
                    continue

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

        # only one caption per motion
        text_dict = text_list[0]
        caption = text_dict["caption"]

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
        
        joints = recover_from_ric(
            torch.tensor(motion, dtype=torch.float32),
        ).numpy().reshape(-1, 22 * 3)

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
            None,  # word embeddings
            None,  # pos one hots
            caption,  # caption
            None,  # sentence length
            motion,
            length,
            None,  # tokens
            key,  # name
            joints,  # joints
        )
