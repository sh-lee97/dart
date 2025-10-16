from os.path import join

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_complement_indices(indices, n_data):
    """Given a list of indices and number of total datapoints, computes complement indices"""
    comp_indices = []
    for i in range(n_data):
        if i not in indices:
            comp_indices.append(i)

    return comp_indices


class HAADataset(Dataset):
    def __init__(
        self,
        base_dir="/data1/diffrir",
        room="",
        split_mode="random",
        split="train",
        num_train_data=12,
        echogram_len_sec=0.32,
        radiance_sampling_rate=1000,
        rir_sampling_rate=16000,
        data_per_epoch=1000,
        normalize=False,
        **kwargs,
    ):
        super().__init__()

        self.room = room
        self.room_dir = join(base_dir, room)

        self.split_mode = split_mode
        self.split = split

        self.num_train_data = num_train_data

        self.echogram_len_sec = echogram_len_sec
        self.radiance_sampling_rate = radiance_sampling_rate
        self.rir_sampling_rate = rir_sampling_rate
        self.data_per_epoch = data_per_epoch
        self.normalize = normalize

        self.load_haa()

    def load_haa(self):
        print("Loading HAA data ...")
        receiver_pos = np.load(join(self.room_dir, "xyzs.npy"))
        rirs_full = np.load(join(self.room_dir, "RIRs._short.npy"))
        rirs_full = rirs_full[:, : int(self.echogram_len_sec * 48000)]
        print("Done, now post-processing")

        if self.rir_sampling_rate != 48000:
            rirs_full = librosa.resample(
                rirs_full,
                orig_sr=48000,
                target_sr=self.rir_sampling_rate,
                res_type="polyphase",
            )

        if self.normalize:
            rirs_full = rirs_full / np.max(np.abs(rirs_full), axis=-1, keepdims=True)

        match self.room:
            case "classroomBase":
                train_idx = [0, 57, 114, 171, 228, 285, 342, 399, 456, 513, 570, 627]
                valid_idx = compute_complement_indices(
                    list(train_idx) + list(np.arange(315) * 2), 630
                )[::2]
                test_idx = compute_complement_indices(train_idx + valid_idx, 630)
                source_pos = [3.5838, 5.7230, 1.2294]
                source_orientation = [0, -1.0, 0]

            case "complexBase":
                train_idx = [5, 47, 82, 117, 145, 187, 220, 255, 290, 342, 372, 404]
                valid_idx = compute_complement_indices(train_idx, 408)[::2]
                test_idx = compute_complement_indices(train_idx + valid_idx, 408)
                source_pos = [2.8377, 10.1228, 1.1539]
                source_orientation = [0, -1.0, 0]

            case "hallwayBase":
                train_idx = [5, 58, 99, 148, 203, 241, 296, 342, 384, 441, 482, 535]
                valid_idx = compute_complement_indices(
                    train_idx + list(np.arange(288) * 2), 576
                )[::2]
                test_idx = compute_complement_indices(train_idx + valid_idx, 576)
                source_pos = [0.6870, 10.2452, 0.5367]
                source_orientation = [0, -1.0, 0]

            case "dampenedBase":
                train_idx = [0, 23, 46, 69, 92 + 12, 115, 138, 161, 184, 207, 230, 253]
                valid_idx = compute_complement_indices(
                    train_idx + list(np.arange(138) * 2), 276
                )[::2]
                test_idx = compute_complement_indices(train_idx + valid_idx, 276)
                source_pos = [2.4542, 2.4981, 1.2654]
                source_orientation = [0, -1.0, 0]

        if self.num_train_data != 12:
            rng = np.random.default_rng(0)
            rng.shuffle(train_idx)
            rng.shuffle(valid_idx)
            if self.num_train_data < 12:
                train_idx = train_idx[: self.num_train_data]
                valid_idx = train_idx[self.num_train_data :] + valid_idx
            else:
                train_idx = train_idx + valid_idx[: self.num_train_data - 12]
                valid_idx = valid_idx[self.num_train_data - 12 :]

        self.source_pos = np.array(source_pos)
        self.source_orientation = np.array(source_orientation)

        self.full_receiver_pos = receiver_pos

        match self.split:
            case "train":
                self.receiver_pos = receiver_pos[train_idx]
                self.idxs = train_idx
                rirs = rirs_full[train_idx]

            case "valid":
                self.receiver_pos = receiver_pos[valid_idx]
                self.idxs = valid_idx
                rirs = rirs_full[valid_idx]

            case "test":
                self.receiver_pos = receiver_pos[test_idx]
                self.idxs = test_idx
                rirs = rirs_full[test_idx]

        num_data = len(rirs)

        ds_factor = self.rir_sampling_rate // self.radiance_sampling_rate

        echograms = rirs**2
        echograms = np.reshape(echograms, (num_data, -1, ds_factor))
        echograms = np.sum(echograms, -1)

        self.rirs = rirs
        self.echograms = echograms
        print("Done")

    def __len__(self):
        if self.split == "train":
            return self.data_per_epoch
        else:
            return len(self.rirs)

    def __getitem__(self, idx):
        idx = idx % len(self.rirs)
        rir = self.rirs[idx]
        echogram = self.echograms[idx]
        source_pos = self.source_pos
        source_direction = self.source_orientation
        receiver_pos = self.receiver_pos[idx]

        if self.split == "train":
            source_pos = source_pos + np.random.uniform(
                -0.1, 0.1, size=source_pos.shape
            )
            source_direction = source_direction + np.random.uniform(
                -0.1, 0.1, size=source_direction.shape
            )
            source_direction = source_direction / np.linalg.norm(source_direction)
            receiver_pos = receiver_pos + np.random.uniform(
                -0.1, 0.1, size=receiver_pos.shape
            )
        return {
            "rir": torch.from_numpy(rir).float(),
            "source_pos": torch.from_numpy(source_pos).float(),
            "source_orientation": torch.from_numpy(source_direction).float(),
            "receiver_pos": torch.from_numpy(receiver_pos).float(),
            "echogram": torch.from_numpy(echogram).float(),
            "idx": self.idxs[idx],
        }


def make_haa_short(base_dir="/data1/diffrir"):
    for room in ["classroomBase", "complexBase", "hallwayBase", "dampenedBase"]:
        room_dir = join(base_dir, room)
        rirs = np.load(join(room_dir, "RIRs.npy"))
        rirs = rirs[:, :48000]
        np.save(join(room_dir, "RIRs._short.npy"), rirs)
        print(f"Saved {room} RIRs._short.npy")
