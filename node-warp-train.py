#!/usr/bin/env python
# coding: utf-8

import argparse
from copy import deepcopy
from enum import Enum
from multiprocessing import Pool
import os
import os.path as osp
import random
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from ifmorph.dataset import WarpingDataset, Warping3DDataset
from ifmorph.loss_functions import NeuralODELoss
from ifmorph.model import NearIdSIREN
from ifmorph.neural_odes import NeuralODE
from ifmorph.util import (
    create_node_morphing,
    get_landmark_correspondences,
    get_default_argumentparser,
    get_device,
)
from tqdm import tqdm


class LoggerType(Enum):
    TENSORBOARD = "tensorboard"
    NONE = "none"


def train_warping(experiment_config_path, output_path, args):
    """Runs a single warp training.

    Parameters
    ----------
    experiment_config_path: str, PathLike
        Path to the experiment configuration file.

    output_path: str, PathLike
        Path to the output folder.

    args: argparse.Namespace
        Other relevant arguments.
    """
    if not osp.exists(experiment_config_path):
        raise FileNotFoundError(
            f'Experiment configuration file "{experiment_config_path}" not found.'
        )

    with open(experiment_config_path, "r") as fin:
        config = yaml.safe_load(fin)

    experiment_name = osp.splitext(osp.split(experiment_config_path)[-1])[0]
    config["experiment_name"] = experiment_name

    os.makedirs(output_path, exist_ok=True)

    logger_type = LoggerType(args.logging)
    if logger_type == LoggerType.TENSORBOARD:
        summary_path = osp.join(output_path, "summaries")
        if not osp.exists(summary_path):
            os.makedirs(summary_path)
        writer = SummaryWriter(summary_path)

    with open(osp.join(output_path, "config.yaml"), "w") as fout:
        yaml.dump(config, fout)

    device = get_device(args.device, config.get("device", "cuda:0"))

    # ------- processing the input files ------- #
    initial_conditions = [(v, k) for k, v in config["initial_conditions"].items()]
    if len(initial_conditions) < 2:
        raise ValueError(
            "The provided config file should contain at least 2 images as"
            " initial conditions"
        )

    data = WarpingDataset(
        initial_conditions, num_samples=config["training"]["n_samples"], device=device
    )

    loss_config = config["loss"]
    training_config = config["training"]
    training_steps = training_config["n_steps"]
    reconstruction_steps = training_config.get("reconstruction_steps", None)

    warmup_steps = training_config["n_steps"]
    checkpoint_steps = training_config.get("checkpoint_steps", None)
    reconstruct_config = config["reconstruct"]
    n_frames = reconstruct_config.get("n_frames", 100)
    fps = reconstruct_config.get("fps", 10)
    grid_dims = reconstruct_config.get("frame_dims", (320, 320))

    # ------- creating the correspondences ------- #
    src, tgt = None, None
    for i in range(len(data.initial_conditions) - 1):
        if f"points_{i}" not in loss_config or f"points_{i+1}" not in loss_config:
            src, tgt, _, _ = get_landmark_correspondences(
                data.initial_states[i],
                data.initial_states[i + 1],
                grid_dims,
                device=device,
                open_ui=not args.no_ui,
            )
        elif isinstance(loss_config[f"points_{i}"], str) or isinstance(
            loss_config[f"points_{i+1}"], str
        ):
            with open(loss_config[f"points_{i}"], "r") as fin:
                src = np.array(yaml.safe_load(fin))

            with open(loss_config[f"points_{i+1}"], "r") as fin:
                tgt = np.array(yaml.safe_load(fin))
        else:
            src = np.array(loss_config[f"points_{i}"])
            tgt = np.array(loss_config[f"points_{i+1}"])

        src = torch.Tensor(src).float().to(device)
        tgt = torch.Tensor(tgt).float().to(device)

    with open(osp.join(output_path, "config.yaml"), "w") as fout:
        yaml.dump(config, fout)

    # ------- constructing the model ------- #
    network_config = config["network"]
    network_config["internal_network_type"] = network_config.get(
        "internal_network_type", "siren"
    )

    model = NeuralODE.load_from_config(network_config).to(device)
    model.internal_module = model.internal_module.to(device)

    # ------- setting the optimizer and training schedulers ------- #
    lr = float(config["optimizer"].get("lr", 1e-4))
    optim = torch.optim.Adam(params=model.internal_module.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=0.5,
        patience=100,
        cooldown=900,
        threshold=1e-4,
        min_lr=1e-5,
    )

    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=1e-1, end_factor=1.0, total_iters=100
    )

    # ---------- loss setup ----------#
    dims = network_config.get("dim", 2)
    _grid = torch.linspace(-1, 1, 100)
    grid = torch.cartesian_prod(*[_grid] * dims).to(device)

    constraint_weights = loss_config.get("constraint_weights", None)
    for k, v in constraint_weights.items():
        constraint_weights[k] = float(v)

    matching_loss_weight = constraint_weights.get("matching_loss_weight", 1024.0)
    jacobian_loss_weight = constraint_weights.get("jacobian_loss_weight", 1.0)
    timesteps = loss_config.get("timesteps", [0.0, 0.125, 0.25, 0.375, 0.5])

    loss_func = NeuralODELoss(
        matching_loss_weight=matching_loss_weight,
        jacobian_loss_weight=jacobian_loss_weight,
        timesteps=timesteps,
    ).to(device)

    # ---------- training ----------#
    # best_loss = np.inf
    # best_step = warmup_steps
    training_loss = {}
    for step in range(training_steps + 1):
        loss = loss_func(model, src, tgt, grid)
        matching_error = loss["matching_loss"]
        running_loss = torch.zeros((1, 1)).to(device)
        for k, v in loss.items():
            if not k.startswith("_"):
                running_loss += v
            if k not in training_loss:
                training_loss[k] = [v.item()]
            else:
                training_loss[k].append(v.item())

        # Logging the total training loss and individual terms.
        if logger_type == LoggerType.TENSORBOARD:
            for k, v in loss.items():
                writer.add_scalar(f"train_loss/{k}", v.item(), step)

            writer.add_scalar("train_loss/total_loss", running_loss.item(), step)

        if not step % 1000:
            try:
                print(
                    step,
                    running_loss.item(),
                    config["initial_conditions"]["0"],
                    config["initial_conditions"]["1"],
                )
            except KeyError:
                print(
                    step,
                    running_loss.item(),
                    config["initial_conditions"][0],
                    config["initial_conditions"][1],
                )

        if checkpoint_steps is not None and step > 0 and not step % checkpoint_steps:
            torch.save(
                model,
                osp.join(output_path, f"checkpoint_{step}.pth"),
            )

        if (
            reconstruction_steps is not None
            and step > 0
            and not step % reconstruction_steps
        ):
            print(f"Reconstruction at {step}")
            model = model.eval()
            with torch.no_grad():
                create_node_morphing(
                    warp_net=model,
                    frame0=None,
                    frame1=None,
                    output_path=osp.join(output_path, f"rec_{step}_w_landmarks.mp4"),
                    frame_dims=grid_dims,
                    n_frames=n_frames,
                    fps=fps,
                    device=device,
                    landmark_src=src,
                    landmark_tgt=tgt,
                    overlay_landmarks=True,
                    frame_collection=data.initial_states,
                )

                create_node_morphing(
                    warp_net=model,
                    frame0=None,
                    frame1=None,
                    output_path=osp.join(output_path, f"rec_{step}_no_landmarks.mp4"),
                    frame_dims=grid_dims,
                    n_frames=n_frames,
                    fps=fps,
                    device=device,
                    landmark_src=src,
                    landmark_tgt=tgt,
                    overlay_landmarks=False,
                    frame_collection=data.initial_states,
                )
            model = model.train()

        optim.zero_grad()
        running_loss.backward()
        optim.step()
        scheduler_warmup.step()
        scheduler.step(running_loss)

    print("Training done. Losses:", loss)
    print("Max mismatch t=1/2:", matching_error.sqrt())

    if logger_type == LoggerType.TENSORBOARD:
        writer.add_scalar(
            "train_loss/max_mismatch_t=1/2",
            matching_error.detach(),
            step,
        )
        writer.close()

    model = model.eval()
    with torch.no_grad():
        torch.save(model.state_dict(), osp.join(output_path, "weights.pth"))

    loss_df = pd.DataFrame.from_dict(training_loss)
    loss_df.to_csv(osp.join(output_path, "loss.csv"), sep=";")

    if not args.no_reconstruction:
        print("Running the inference for the morphing.")
        with torch.no_grad():
            create_node_morphing(
                warp_net=model,
                frame0=None,
                frame1=None,
                output_path=osp.join(output_path, "morphing_w_landmarks.mp4"),
                frame_dims=grid_dims,
                n_frames=n_frames,
                fps=fps,
                device=device,
                landmark_src=src,
                landmark_tgt=tgt,
                overlay_landmarks=True,
                frame_collection=data.initial_states,
            )

            create_node_morphing(
                warp_net=model,
                frame0=None,
                frame1=None,
                output_path=osp.join(output_path, "morphing_no_landmarks.mp4"),
                frame_dims=grid_dims,
                n_frames=n_frames,
                fps=fps,
                device=device,
                landmark_src=src,
                landmark_tgt=tgt,
                overlay_landmarks=False,
                frame_collection=data.initial_states,
            )
        print("Inference done.")


if __name__ == "__main__":
    parser = get_default_argumentparser()
    args = parser.parse_args()

    # print all args:
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    paths_to_run = []
    for p in args.config_path:
        fname = osp.splitext(osp.split(p)[-1])[0]  # removing the file extension
        outpath = osp.join(args.output_path, f"{fname}")
        weights_path = osp.join(outpath, "weights.pth")
        if args.skip_finished and osp.exists(weights_path):
            print(f"{fname} is already trained. Skipping")
            continue
        paths_to_run.append((p, outpath))

    if args.n_tasks > 1:
        pool = Pool(processes=args.n_tasks)
        for p, o in paths_to_run:
            pool.apply_async(func=train_warping, args=(p, o, args))

        pool.close()
        pool.join()
    else:
        for p, o in paths_to_run:
            train_warping(p, o, args)
