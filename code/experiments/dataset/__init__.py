from experiments.dataset.coupled_room import CoupledRoomDataset
from experiments.dataset.haa import HAADataset
from torch_geometric_acoustics import zoo
from torch_geometric_acoustics.mesh import (
    perturb_mesh,
    split_parallelogram_patches,
    triangulate_mesh,
)


def get_nonsplitted_mesh(dataset, room):
    match dataset:
        case "HAA":
            match room:
                case "classroomBase":
                    mesh = zoo.diffrir_classroom_base()
                case "complexBase":
                    mesh = zoo.diffrir_complex_base()
                case "hallwayBase":
                    mesh = zoo.diffrir_hallway_base()
                case "dampenedBase":
                    mesh = zoo.diffrir_dampened_base()
        case "CR":
            match room:
                case "meeting_room_to_hallway":
                    mesh = zoo.coupled_rooms_meeting_room_to_hallway()
                case "office_to_anechoic_chamber":
                    mesh = zoo.coupled_rooms_office_to_anechoic_chamber()
                case "office_to_kitchen":
                    mesh = zoo.coupled_rooms_office_to_kitchen()
                case "office_to_stairwell":
                    mesh = zoo.coupled_rooms_office_to_stairwell()

    return mesh


def get_datasets_and_mesh(
    dataset,
    room,
    radiance_sampling_rate=1600,
    echogram_len_sec=0.32,
    rir_sampling_rate=16000,
    geometry_distortion=0,
    num_train_data=12,
    patch_split=None,
    **dataset_kwargs,
):
    common_kwargs = {
        "radiance_sampling_rate": radiance_sampling_rate,
        "echogram_len_sec": echogram_len_sec,
        "rir_sampling_rate": rir_sampling_rate,
        "num_train_data": num_train_data,
        **dataset_kwargs,
    }
    match dataset:
        case "HAA":
            match room:
                case "classroomBase":
                    mesh = zoo.diffrir_classroom_base()
                    mesh = split_parallelogram_patches(
                        mesh, 3.2 if patch_split is None else patch_split
                    )
                    mesh = triangulate_mesh(mesh)
                case "complexBase":
                    mesh = zoo.diffrir_complex_base()
                    mesh = split_parallelogram_patches(
                        mesh, 4.8 if patch_split is None else patch_split
                    )
                    mesh = triangulate_mesh(mesh)
                case "hallwayBase":
                    mesh = zoo.diffrir_hallway_base()
                    mesh = split_parallelogram_patches(
                        mesh, 3 if patch_split is None else patch_split
                    )
                    mesh = triangulate_mesh(mesh)
                case "dampenedBase":
                    mesh = zoo.diffrir_dampened_base()
                    mesh = split_parallelogram_patches(
                        mesh, 2.4 if patch_split is None else patch_split
                    )
                    mesh = triangulate_mesh(mesh)
            train_set = HAADataset(
                base_dir="/data1/diffrir", room=room, split="train", **common_kwargs
            )
            valid_set = [
                HAADataset(
                    base_dir="/data1/diffrir", room=room, split="test", **common_kwargs
                ),
                HAADataset(
                    base_dir="/data1/diffrir", room=room, split="valid", **common_kwargs
                ),
            ]

        case "CR":
            match room:
                case "meeting_room_to_hallway":
                    mesh = zoo.coupled_rooms_meeting_room_to_hallway()
                case "office_to_anechoic_chamber":
                    mesh = zoo.coupled_rooms_office_to_anechoic_chamber()
                case "office_to_kitchen":
                    mesh = zoo.coupled_rooms_office_to_kitchen()
                case "office_to_stairwell":
                    mesh = zoo.coupled_rooms_office_to_stairwell()

            mesh = split_parallelogram_patches(
                mesh, 4 if patch_split is None else patch_split
            )
            mesh = triangulate_mesh(mesh)

            train_set = CoupledRoomDataset(
                split="train",
                room=room,
                **common_kwargs,
            )
            if common_kwargs["split_mode"] == "unseen":
                valid_set = [
                    CoupledRoomDataset(split="test", room=room, **common_kwargs),
                    CoupledRoomDataset(split="valid", room=room, **common_kwargs),
                ]
            else:
                valid_set = [
                    CoupledRoomDataset(split="test", room=room, **common_kwargs),
                    CoupledRoomDataset(split="valid", room=room, **common_kwargs),
                ]

    if geometry_distortion > 0:
        mesh = perturb_mesh(mesh, scale=geometry_distortion)
    return mesh, train_set, valid_set
