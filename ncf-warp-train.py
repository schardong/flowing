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
from ifmorph.dataset import WarpingDataset
from ifmorph.loss_functions import FlowLoss
from ifmorph.util import create_morphing, get_landmark_correspondences
from ifmorph.neural_conjugate_flows import *
from ifmorph.fd_models import convert_to_fd
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
        raise FileNotFoundError("Experiment configuration file not found.")

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

    devstr = ""
    if args.device:
        devstr = args.device
    else:
        devstr = config.get("device", "cuda:0")

    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.", file=sys.stderr)
    else:
        torch.cuda.empty_cache()

    device = torch.device(devstr)

    initial_conditions = [(v, k) for k, v in config["initial_conditions"].items()]
    data = WarpingDataset(
        initial_conditions, num_samples=config["training"]["n_samples"], device=device
    )

    # --------define neural conjugate flow (NCF)----------
    # Construct layers
    network_config = config["network"]

    # Construct model
    model = NeuralConjugate.load_from_config(network_config)
    model = convert_to_fd(model).to(device)

    optim_config = config["optimizer"]
    optim = torch.optim.Adam(lr=optim_config["lr"], params=model.parameters())

    loss_config = config["loss"]
    training_config = config["training"]
    training_steps = training_config["n_steps"]
    warmup_steps = training_config.get("warmup_steps", training_steps // 10)
    checkpoint_steps = training_config.get("checkpoint_steps", None)
    reconstruct_steps = None
    if not args.no_reconstruction:
        reconstruct_steps = training_config.get("reconstruction_steps", None)

    reconstruct_config = config["reconstruct"]
    n_frames = reconstruct_config.get("n_frames", 100)
    fps = reconstruct_config.get("fps", 10)
    grid_dims = reconstruct_config.get("frame_dims", (320, 320))

    best_loss = np.inf
    best_step = warmup_steps
    training_loss = {}

    src, tgt = None, None

    if len(initial_conditions) < 2:
        raise ValueError(
            "The provided config file should contain at least 2 images as initial conditions"
        )

    points_sequence = [None for _ in range(len(data.initial_conditions))]
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

        constraint_weights = loss_config.get("constraint_weights", None)
        intermediate_times = loss_config.get("intermediate_times", [0.0, 1.0])

        for k, v in constraint_weights.items():
            constraint_weights[k] = float(v)

        src = torch.Tensor(src).float().to(device)
        tgt = torch.Tensor(tgt).float().to(device)

        points_sequence[i] = src
        points_sequence[i + 1] = tgt

        if len(src) == 0:
            raise ValueError(f"The config files does not have landmarks for image {i}")
        elif len(tgt) == 0:
            raise ValueError(
                f"The config files does not have landmarks for image {i+1}"
            )

        loss_config[f"points_{i}"] = src.detach().clone().cpu().numpy().tolist()
        loss_config[f"points_{i+1}"] = tgt.detach().clone().cpu().numpy().tolist()

    config["loss"] = loss_config
    loss_func = FlowLoss(
        constraint_weights=constraint_weights, intermediate_times=intermediate_times
    )

    with open(osp.join(output_path, "config.yaml"), "w") as fout:
        yaml.dump(config, fout)

    # Logging the images and landmarks
    # if logger_type == LoggerType.TENSORBOARD:
    #     writer.add_image(
    #         "initial_states/source", src_img, dataformats="HW"
    #     )
    #     writer.add_image(
    #         "initial_states/target", tgt_img, dataformats="HW"
    #     )

    points_sequence = torch.stack(points_sequence, dim=0).to(device)

    for step in range(training_steps + 1):
        loss = loss_func(points_sequence, model)
        # Accumulating the losses.
        running_loss = torch.zeros((1, 1)).to(device)
        for k, v in loss.items():
            if not k.startswith("_"):
                running_loss += v
            if k not in training_loss:
                training_loss[k] = [v.item()]
            else:
                training_loss[k].append(v.item())

            # Logging individual loss terms.
            if logger_type == LoggerType.TENSORBOARD:
                writer.add_scalar(f"train_loss/{k}", v.item(), step)

        # Logging the total training loss.
        if logger_type == LoggerType.TENSORBOARD:
            writer.add_scalar("train_loss/total_loss", running_loss.item(), step)

        if not step % 1000:
            # print(step, running_loss.item())
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

        if step > warmup_steps and running_loss.item() < best_loss:
            best_step = step
            best_loss = running_loss.item()
            best_weights = deepcopy(model.state_dict())

        if checkpoint_steps is not None and step > 0 and not step % checkpoint_steps:
            torch.save(
                # model.state_dict(),
                model,
                osp.join(output_path, f"checkpoint_{step}.pth"),
            )

        if reconstruct_steps is not None and step > 0 and not step % reconstruct_steps:
            print("Running the inference.")
            model = model.eval()
            vidpath = osp.join(output_path, f"rec_{step}.mp4")
            with torch.no_grad():
                create_morphing(
                    warp_net=model,
                    frame0=None,
                    frame1=None,
                    output_path=vidpath,
                    frame_dims=grid_dims,
                    n_frames=n_frames,
                    fps=fps,
                    device=device,
                    landmark_src=src,
                    landmark_tgt=tgt,
                    blending_type="linear",
                    overlay_landmarks=True,
                    frame_collection=data.initial_states,
                )
            print("Inference done.")
            model = model.train().to(device)

        optim.zero_grad()
        running_loss.backward()
        optim.step()

    print("Training done.")
    print(f"Best results at step {best_step}, with loss {best_loss}.")
    print(f"Saving the results in folder {output_path}.")
    model = model.eval()
    with torch.no_grad():
        # model.update_omegas(w0=1, ww=None)
        torch.save(
            # model.state_dict(), osp.join(output_path, "weights.pth")
            model,
            osp.join(output_path, "weights.pth"),
        )

        # model.w0 = network_config["omega_0"]
        # model.ww = network_config["omega_w"]
        model.load_state_dict(best_weights)
        # model.update_omegas(w0=1, ww=None)
        torch.save(
            # model.state_dict(), osp.join(output_path, "best.pth")
            model,
            osp.join(output_path, "best.pth"),
        )

    loss_df = pd.DataFrame.from_dict(training_loss)
    loss_df.to_csv(osp.join(output_path, "loss.csv"), sep=";")

    if not args.no_reconstruction:
        print("Running the inference.")

        vidpath = osp.join(output_path, "video.mp4")
        create_morphing(
            warp_net=model,
            frame0=None,
            frame1=None,
            output_path=vidpath,
            frame_dims=grid_dims,
            n_frames=n_frames,
            fps=fps,
            device=device,
            landmark_src=src,
            landmark_tgt=tgt,
            overlay_landmarks=True,
            frame_collection=data.initial_states,
        )
        print("Inference done.")

    if logger_type == LoggerType.TENSORBOARD:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", nargs="+", help="Path to experiment configuration" " files."
    )
    parser.add_argument(
        "--logging",
        default="tensorboard",
        type=str,
        help="Type of logging to use. May be one of: tensorboard, none."
        'By default is "tensorboard".',
    )
    parser.add_argument(
        "--seed", default=123, type=int, help="Seed for the RNG. By default its 123."
    )
    parser.add_argument(
        "--n-tasks",
        default=1,
        type=int,
        help="Number of parallel trainings to run. By default is set to 1,"
        " meaning that we run serially.",
    )
    parser.add_argument(
        "--skip-finished",
        action="store_true",
        help="Skips running an experiment if the output path contains the"
        ' "weights.pth" file.',
    )
    parser.add_argument(
        "--output-path",
        "-o",
        default="results",
        help="Optional output path to store experimental results. By default"
        " we use the experiment filename and create a matching directory"
        ' under folder "results".',
    )
    parser.add_argument(
        "--no-ui",
        "-n",
        action="store_true",
        default=False,
        help="Does not open the UI for point adjustments. Useful when running"
        " in batches.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        help="Device to run the"
        " training. Overrides the experiment configuration if present.",
    )
    parser.add_argument(
        "--no-reconstruction",
        action="store_true",
        help="Bypasses the"
        " configuration and runs no intermediate/final reconstructions.",
    )
    parser.add_argument(
        "--from-pretrained",
        type=str,
        default=None,
        help="Path to the yaml" " with eights to load the model from.",
    )

    args = parser.parse_args()

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
