import pickle

import librosa
import numpy as np
import sofa
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from tqdm import tqdm


def view_up_to_quaternion(view, up):
    y_axis = np.cross(up, view)
    rotation = R.from_matrix(np.moveaxis(np.asarray([view, y_axis, up]), 0, -1))
    return R.as_quat(rotation)


def preprocess_coupled_room(base_dir="/data1/coupled_room"):
    rooms = [
        "meeting_room_to_hallway",
        "office_to_anechoic_chamber",
        "office_to_kitchen",
        "office_to_stairwell",
    ]
    for room in tqdm(rooms):
        match room:
            case "meeting_room_to_hallway":
                prefix = "roomToHallway"
                room_a, room_b = "Room", "Hallway"
            case "office_to_anechoic_chamber":
                prefix = "officeToChamber"
                room_a, room_b = "Office", "Chamber"
            case "office_to_kitchen":
                prefix = "officeToKitchen"
                room_a, room_b = "Office", "Kitchen"
            case "office_to_stairwell":
                prefix = "officeToStairwell"
                room_a, room_b = "Office", "Stairwell"

        data_dirs = {
            "los_a": f"{base_dir}/{prefix}_src{room_a}_LOS.sofa",
            "los_b": f"{base_dir}/{prefix}_src{room_b}_LOS.sofa",
            "nlos_a": f"{base_dir}/{prefix}_src{room_a}_noLOS.sofa",
            "nlos_b": f"{base_dir}/{prefix}_src{room_b}_noLOS.sofa",
        }
        databases = {k: sofa.Database.open(v) for k, v in data_dirs.items()}
        full_data = {}

        for k, database in tqdm(databases.items()):
            source_position, source_view, source_up = database.Emitter.get_pose()
            source_position = source_position[0, :, 0]
            source_view = source_view[0, :, 0]
            source_up = source_up[0, :, 0]
            source_orientation = view_up_to_quaternion(source_view, source_up)

            receiver_position, _, _ = database.Receiver.get_pose()
            receiver_position = receiver_position[0, :, :].T
            sr = int(database.Data.SamplingRate.get_values())

            rirs = database.Data.IR.get_values()
            rirs = rirs[:, 0, :sr]  # omnidirectional mic, at most 1 sec
            print(
                source_position.shape,
                source_orientation.shape,
                receiver_position.shape,
                rirs.shape,
            )

            full_data[k] = {
                "source_position": source_position,
                "source_orientation": source_orientation,
                "receiver_position": receiver_position,
                "rirs": rirs,
            }

        pickle.dump(
            full_data,
            open(f"{base_dir}/{room}_data.pickle", "wb"),
        )


def postprocess_coupled_room(base_dir="/data1/coupled_room", delay=True):
    import pickle

    from scipy.spatial.transform import Rotation as R


    for room in [
        "meeting_room_to_hallway",
        "office_to_anechoic_chamber",
        "office_to_kitchen",
        "office_to_stairwell",
    ]:
        data = pickle.load(open(f"{base_dir}/{room}_data.pickle", "rb"))
        for k in data:
            if room == "office_to_anechoic_chamber" and k == "nlos_b":
                data[k]["source_position"] = np.array([-1.22 + 3.8 + 0.7, 2.62, 1.5])
                data[k]["source_direction"] = np.array([1.0, 0, 0])
            else:
                source_orientation = data[k]["source_orientation"]
                r = R.from_quat(source_orientation)
                data[k]["source_direction"] = r.apply([1, 0, 0])

            source_position = data[k]["source_position"]
            source_direction = data[k]["source_direction"]
            receiver_position = data[k]["receiver_position"]
            for i in range(len(receiver_position)):
                distance = np.linalg.norm(receiver_position[i] - source_position)
                distance_delay = int(distance / 343 * 48000)
                if delay:
                    data[k]["rirs"][i] = np.roll(data[k]["rirs"][i], distance_delay)
                    data[k]["rirs"][i, :distance_delay] = 0
            data[k].pop("source_orientation")

        if delay:
            pickle.dump(
                data,
                open(f"{base_dir}/{room}_data_pp.pickle", "wb"),
            )
        else:
            pickle.dump(
                data,
                open(f"{base_dir}/{room}_data_pp_no_delay.pickle", "wb"),
            )


class CoupledRoomDataset(Dataset):
    def __init__(
        self,
        base_dir="/data1/coupled_room",
        room="office_to_kitchen",
        split_mode="random",
        split="train",
        num_train_data=12,
        echogram_len_sec=0.32,
        radiance_sampling_rate=1000,
        rir_sampling_rate=16000,
        data_per_epoch=1000,
        delay=True,
        **kwargs,
    ):
        self.room = room

        self.base_dir = base_dir
        self.split_mode = split_mode
        self.split = split

        self.num_train_data = num_train_data

        self.echogram_len_sec = echogram_len_sec
        self.radiance_sampling_rate = radiance_sampling_rate
        self.rir_sampling_rate = rir_sampling_rate
        self.data_per_epoch = data_per_epoch

        self.delay = delay

        self.load_coupled_room()

    def load_coupled_room(self):
        print("Loading Coupled Room data ...")
        if self.delay:
            data = pickle.load(
                open(f"{self.base_dir}/{self.room}_data_pp.pickle", "rb")
            )
        else:
            data = pickle.load(
                open(f"{self.base_dir}/{self.room}_data_pp_no_delay.pickle", "rb")
            )

        match self.split_mode:
            case "random":
                data = self.get_random_split(data)

            case "unseen":
                data = self.get_unseen_split(data)

            case "unseen_2":
                data = self.get_unseen_split_2(data)

        for k in data:
            data[k]["rirs"] = data[k]["rirs"][:, : int(self.echogram_len_sec * 48000)]
        if self.rir_sampling_rate != 48000:
            for k in data:
                data[k]["rirs"] = librosa.resample(
                    data[k]["rirs"],
                    orig_sr=48000,
                    target_sr=self.rir_sampling_rate,
                    res_type="polyphase",
                )

        ds_factor = self.rir_sampling_rate // self.radiance_sampling_rate

        for k in data:
            rirs = data[k]["rirs"]
            num_data = len(rirs)
            echograms = rirs**2
            echograms = np.reshape(echograms, (num_data, -1, ds_factor))
            echograms = np.sum(echograms, -1)
            data[k]["echograms"] = echograms

        self.data = data

    def get_random_split(self, data):
        assert self.num_train_data % 4 == 0
        self.n_train_per_source = self.num_train_data // 4
        self.n_valid_per_source = 24 - self.n_train_per_source
        self.n_test_per_source = 101 - self.n_valid_per_source - self.n_train_per_source
        rng = np.random.default_rng(0)
        ids = {}
        for k in data:
            id_list = np.arange(101)
            rng.shuffle(id_list)
            match self.split:
                case "train":
                    ids[k] = id_list[: self.n_train_per_source]
                    self.data_per_source = self.n_train_per_source
                case "valid":
                    ids[k] = id_list[
                        self.n_train_per_source : self.n_train_per_source
                        + self.n_valid_per_source
                    ]
                    self.data_per_source = self.n_valid_per_source
                    self.num_valid_data = 4 * self.n_valid_per_source
                case "test":
                    ids[k] = id_list[
                        self.n_train_per_source + self.n_valid_per_source :
                    ]
                    self.data_per_source = self.n_test_per_source
                    self.num_test_data = 4 * self.n_test_per_source

        self.ids = ids
        return data

    def get_unseen_split(self, data):
        self.num_valid_data = 6
        self.num_test_data = 25
        return data

    def get_unseen_split_2(self, data):
        self.num_valid_data = 6
        self.num_test_data = 32
        return data

    def __getitem__(self, idx):
        match self.split_mode:
            case "random":
                idx = idx % (4 * self.data_per_source)
                key_ids = idx // self.data_per_source
                data_id = idx % self.data_per_source
                key = list(self.data.keys())[key_ids]
                data_id = self.ids[key][data_id]

            case "unseen":
                match self.split:
                    case "train":
                        keys = ["los_a", "los_b"]
                        idx = idx % 12
                        key_idx = idx // 6
                        key = keys[key_idx]
                        data_id = 10 * (idx % 6)

                    case "valid":
                        key = "nlos_a"
                        data_id = 10 * idx

                    case "test":
                        key = "nlos_b"
                        data_id = 76 + idx

            case "unseen_2":
                match self.split:
                    case "train":
                        keys = ["los_a", "los_b"]
                        idx = idx % 12
                        key_idx = idx // 6
                        key = keys[key_idx]
                        data_id = 24 + 4 * (idx % 6)

                    case "valid":
                        key = "nlos_a"
                        data_id = 4 * idx

                    case "test":
                        key = "nlos_b"
                        data_id = 69 + idx

        rir = self.data[key]["rirs"][data_id]
        echogram = self.data[key]["echograms"][data_id]
        source_pos = self.data[key]["source_position"]
        source_direction = self.data[key]["source_direction"]
        receiver_pos = self.data[key]["receiver_position"][data_id]

        return {
            "rir": torch.from_numpy(rir).float(),
            "source_pos": torch.from_numpy(source_pos).float(),
            "source_orientation": torch.from_numpy(source_direction).float(),
            "receiver_pos": torch.from_numpy(receiver_pos).float(),
            "echogram": torch.from_numpy(echogram).float(),
            "key": key,
            "idx": data_id,
        }

    def __len__(self):
        match self.split:
            case "train":
                return self.data_per_epoch
            case "valid":
                return self.num_valid_data
            case "test":
                return self.num_test_data
