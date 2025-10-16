r"""
Main DART solver.
"""
import argparse
import pickle

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from loss import LossHandler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from experiments.dataset import get_datasets_and_mesh
from experiments.plot import (
    compare_echogram,
    plot_material_coefficients,
)
from experiments.utils import pretty_print_dict
from torch_geometric_acoustics.art.pdf_art import (
    PDFART_Learnable,
    PDFART_Learnable_Parametric,
)
from torch_geometric_acoustics.draw import draw_mesh_with_source_receiver
from torch_geometric_acoustics.parameters import compare_echogram_parameters


class DARTAcousticFieldSolver(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.__dict__.update(args.__dict__)
        self.save_hyperparameters()
        self.args = args

        torch.set_default_device("cuda")
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)

        print("Loading mesh and dataset...")
        mesh, train_set, valid_set = get_datasets_and_mesh(
            dataset=self.dataset,
            room=self.room,
            radiance_sampling_rate=self.radiance_sampling_rate,
            echogram_len_sec=self.echogram_len_sec,
            rir_sampling_rate=self.rir_sampling_rate,
            geometry_distortion=self.geometry_distortion,
            split_mode=self.split_mode,
            patch_split=self.patch_split,
            num_train_data=self.num_train_data,
        )
        print("Mesh and dataset loaded.")

        self.mesh = mesh
        self.train_set = train_set
        self.valid_set = valid_set

        self.loss_fn = LossHandler()

        if self.parametric:
            if self.radiosity:
                brdfs = ["diffuse"]
            elif self.specular_only:
                brdfs = ["specular"]
            elif self.reflection_only:
                brdfs = [
                    "diffuse",
                    "specular",
                ]
            else:
                brdfs = [
                    "diffuse",
                    "specular",
                    "diffuse_transmission",
                    "specular_transmission",
                ]
            self.art = PDFART_Learnable_Parametric(
                mesh=self.mesh,
                radiance_sampling_rate=self.radiance_sampling_rate,
                echogram_len_sec=self.echogram_len_sec,
                num_bounces=self.num_bounces,
                num_injection_rays=self.num_injection_rays,
                num_detection_rays=self.num_detection_rays,
                direction_sampling_method="stratified_grid",
                directional_source=True,
                fsm_gamma=1e-3,
                brdfs=brdfs,
                learnable_envelope=self.auxiliary,
                learnable_direct_gain=self.auxiliary,
                N_ele=self.N_ele,
                N_azi=self.N_azi,
                shared_param=self.shared_param,
            )
        else:
            self.art = PDFART_Learnable(
                mesh=self.mesh,
                radiance_sampling_rate=self.radiance_sampling_rate,
                echogram_len_sec=self.echogram_len_sec,
                num_bounces=self.num_bounces,
                num_injection_rays=self.num_injection_rays,
                num_detection_rays=self.num_detection_rays,
                direction_sampling_method="stratified_grid",
                directional_source=True,
                fsm_gamma=1e-3,
                learnable_envelope=self.auxiliary,
                learnable_direct_gain=self.auxiliary,
                N_ele=self.N_ele,
                N_azi=self.N_azi,
                inject_ism=self.inject_ism,
                shared_param=self.shared_param,
                reflection_only=self.reflection_only,
            )
        self.art.precompute()
        torch.set_default_device("cpu")

    def training_step(self, data, step_i):
        source_pos, source_orientation, receiver_pos, gt_echogram = (
            data["source_pos"],
            data["source_orientation"],
            data["receiver_pos"],
            data["echogram"],
        )
        source_orientation = (
            None if source_orientation[0] == "none" else source_orientation
        )
        gt_echogram = gt_echogram[0]
        _, _, pred_echogram, reg_loss_dict = self.art(
            source_pos, receiver_pos, source_orientation=source_orientation
        )
        loss_dict = self.loss_fn(
            gt_echogram=gt_echogram,
            pred_echogram=pred_echogram,
            echogram_loss_type=["mse", "edc"],
        )

        if step_i % 10 == 0:
            parameter_distance = compare_echogram_parameters(
                gt_echogram, pred_echogram, self.radiance_sampling_rate, post="lhs"
            )
            results = {**loss_dict, **parameter_distance, **reg_loss_dict}
            self.log_dict(**results)

        loss = loss_dict["echogram_loss"]
        if self.kernel_reg:
            loss += reg_loss_dict["kernel_reg_loss"]
        if self.envelope_reg:
            loss += reg_loss_dict["envelope_reg_loss"]
        return loss

    def on_validation_start(self):
        self.data = {
            "source_pos": [],
            "receiver_pos": [],
            "gt_echogram": [],
            "pred_echogram": [],
        }

    def on_validation_end(self):
        pickle.dump(
            self.data, open(f"{self.valid_dir}/data.{self.current_epoch}.pickle", "wb")
        )

    def validation_step(self, data, step_i, dataloader_idx=None):
        source_pos, source_orientation, receiver_pos, gt_echogram = (
            data["source_pos"],
            data["source_orientation"],
            data["receiver_pos"],
            data["echogram"],
        )
        source_orientation = (
            None if source_orientation[0] == "none" else source_orientation
        )
        gt_echogram = gt_echogram[0]
        absorption_coefficient, scattering_coefficient, pred_echogram, _ = self.art(
            source_pos, receiver_pos, source_orientation=source_orientation
        )
        loss_dict = self.loss_fn(
            gt_echogram=gt_echogram,
            pred_echogram=pred_echogram,
            echogram_loss_type=["mse", "edc"],
        )

        self.data["source_pos"].append(source_pos)
        self.data["receiver_pos"].append(receiver_pos)
        self.data["gt_echogram"].append(gt_echogram)
        self.data["pred_echogram"].append(pred_echogram)

        if step_i == 0:
            plot_material_coefficients(
                mesh=self.mesh,
                absorption_coefficient=absorption_coefficient,
                path=f"{self.valid_dir}/abs.{self.current_epoch}.pdf",
            )
            np.save(
                f"{self.valid_dir}/abs.{self.current_epoch}.npy",
                absorption_coefficient.detach().cpu().numpy(),
            )

        if step_i < 10:
            compare_echogram(
                gt_echogram,
                pred_echogram,
                path=f"{self.valid_dir}/echogram.{step_i}.pdf",
            )
            fig, ax = draw_mesh_with_source_receiver(
                source_pos=source_pos, receiver_pos=receiver_pos, mesh=self.mesh
            )
            fig.savefig(
                f"{self.valid_dir}/echogram.{step_i}.mesh.pdf",
            )
        parameter_distance = compare_echogram_parameters(
            gt_echogram, pred_echogram, self.radiance_sampling_rate, post="lhs"
        )
        results = {**loss_dict, **parameter_distance}
        self.log_dict(**results)
        return loss_dict["echogram_loss"]

    def on_validation_epoch_end(self):
        metrics = {k: v.item() for k, v in self.trainer.callback_metrics.items()}
        pretty_print_dict({"epoch": self.current_epoch, **metrics})

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        sch = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=self.trainer.max_steps,
        )
        return [opt], [{"scheduler": sch, "interval": "step"}]

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        if not isinstance(self.valid_set, list):
            valid_set = [self.valid_set]
        else:
            valid_set = self.valid_set
        return [
            DataLoader(
                valid_set,
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                shuffle=False,
                persistent_workers=True,
            )
            for valid_set in self.valid_set
        ]

    def log_dict(self, **kwargs):
        super().log_dict(
            kwargs, prog_bar=True, logger=True, on_epoch=True, batch_size=1
        )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--room", type=str)
    parser.add_argument("--project", type=str, default="dart-acoustic-field")

    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--rir_sampling_rate", type=int, default=16000)
    parser.add_argument("--radiance_sampling_rate", type=int, default=1000)
    parser.add_argument("--echogram_len_sec", type=float, default=0.32)

    parser.add_argument("--num_bounces", type=int, default=40)
    parser.add_argument("--num_injection_rays", type=int, default=10000)
    parser.add_argument("--num_detection_rays", type=int, default=10000)

    parser.add_argument("--total_opt_steps", type=int, default=25000)
    parser.add_argument("--num_train_data", type=int, default=12)

    parser.add_argument("--geometry_distortion", type=float, default=0)

    parser.add_argument("--split_mode", type=str, default="random")

    parser.add_argument("--delta_kernel", type=bool, default=False)

    parser.add_argument("--lr", type=float, default=1e-2)

    parser.add_argument("--debug", dest="debug", action="store_true", default=False)
    parser.add_argument(
        "--parametric",
        dest="parametric",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--reflection_only",
        dest="reflection_only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--specular_only",
        dest="specular_only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--radiosity",
        dest="radiosity",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--shared_param",
        dest="shared_param",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no_auxiliary",
        dest="auxiliary",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--no_kernel_reg",
        dest="kernel_reg",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--no_envelope_reg",
        dest="envelope_reg",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--N_ele",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--N_azi",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--patch_split",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--inject_ism",
        dest="inject_ism",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    args.name = f"{args.dataset}_{args.room}_{args.name}"

    pretty_print_dict(args.__dict__)
    return args


def setup_logger(args):
    from lightning.pytorch.loggers import WandbLogger

    logger = WandbLogger(
        project=args.project,
        dir="experiments",
        name=f"{args.name}_{args.now}",
        reinit=True,
    )
    logger.experiment.config.update(args.__dict__)
    return logger


def setup_log_dir(args):
    import os
    from datetime import datetime
    from os.path import join

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if args.debug:
        save_dir = "experiments/logs/debug"
    else:
        save_dir = f"experiments/logs/{args.name}_{now}"
    os.makedirs(save_dir, exist_ok=True)
    for subdir in ["train", "valid", "ckpt"]:
        os.makedirs(join(save_dir, subdir), exist_ok=True)

    args.now = now
    args.save_dir = save_dir
    args.train_dir = join(save_dir, "train")
    args.valid_dir = join(save_dir, "valid")

    return args


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_start_method("spawn")
    args = get_args()
    args = setup_log_dir(args)
    logger = None if args.debug else setup_logger(args)
    lr_logger = LearningRateMonitor(logging_interval="step")
    print("Loading solver ... ")
    solver = DARTAcousticFieldSolver(args)
    trainer = L.Trainer(
        callbacks=[lr_logger],
        logger=logger,
        max_steps=args.total_opt_steps,
        devices=1,
        accelerator="gpu",
        strategy="auto",
        default_root_dir="experiments",
        enable_checkpointing=False,
        check_val_every_n_epoch=1,
    )
    trainer.fit(solver)
