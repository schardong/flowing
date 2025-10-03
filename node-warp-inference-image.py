#!/usr/bin/env python
# coding: utf-8

"""Runs the inference on a pre-trained Flow model. Output is a series of
images at each `T`. For the video output, see `warp-inference-vid.py`"""

import argparse
import os
import os.path as osp
import cv2
import torch
import yaml
from ifmorph.dataset import check_network_type, ImageDataset, NotTorchFile
from ifmorph.model import from_pth
from ifmorph.neural_odes import NeuralODE
from ifmorph.util import (get_grid, blend_frames, plot_landmarks,
                          warp_shapenet_inference, warp_points)

WITH_MRNET = True
try:
    from ext.mrimg.src.networks.mrnet import MRFactory
except (ModuleNotFoundError, ImportError):
    WITH_MRNET = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configpath",
        help="Path to experiment configuration file stored with the output"
        " PTHs."
    )
    parser.add_argument(
        "--outputdir", "-o", default=os.getcwd(),
        help="Path to the output directory. By default is the current working"
        "directory, and the files are named \"frame_{checkpoint}_{t}.png\","
        " where \"checkpoint\" is the chosen checkpoint and \"t\" is the"
        " timestep."
    )
    parser.add_argument(
        "--landmarks", "-l", default=False, action="store_true",
        help="Whether to overlay the source/target landmarks on the resulting"
        " images."
    )
    parser.add_argument(
        "--device", "-d", default="cuda:0",
        help="The device to run the inference on. By default its set as"
        " \"cuda:0\" If CUDA is not supported, then the CPU will be used."
    )
    parser.add_argument(
        "--checkpoint", "-c", default="best",
        help="The checkpoint weigths to perform reconstruction. By default"
        " we use the best weights, saved as \"weights.pth\". Note that this is"
        " a number which will be used to compose the name"
        " \"checkpoint_CHECKPOINT.pth\", unless the default value is kept."
    )
    parser.add_argument(
        "--timesteps", "-t", default=[0, 1], nargs='+', help="The timesteps to"
        " use as input for flow. Must be in range [-1, 1]. For each timestep,"
        " an image will be saved. If no timesteps are given, we assume"
        " [0, 1]."
    )
    parser.add_argument(
        "--framedims", "-f", nargs='+', help="Dimensions (in pixels) for the"
        " output image. Note that it must contain two numbers separated by a"
        " space, e.g. \"-f 800 600\"."
    )
    parser.add_argument(
        "--blending", "-b", default="linear",
        help="The type of blending to use. May be any of \"linear\", \"min\","
        " \"max\", \"dist\", \"src\", \"tgt\", \"seamless_{normal,mix}\"."
    )
    args = parser.parse_args()

    devstr = args.device
    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.")
    else:
        torch.cuda.empty_cache()
    device = torch.device(devstr)

    if not osp.exists(args.configpath):
        raise FileNotFoundError("Configuration file not found at"
                                f" \"{args.configpath}\". Aborting.")

    with open(args.configpath, 'r') as fin:
        config = yaml.safe_load(fin)

    modelfilename = "weights.pth"
    if args.checkpoint != "best":
        modelfilename = f"checkpoint_{args.checkpoint}.pth"

    basepath = osp.split(osp.expanduser(args.configpath))[0]
    modelpath = osp.join(basepath, modelfilename)
    if not osp.exists(modelpath):
        raise FileNotFoundError(f"Model file \"{modelpath}\" not found.")

    network_config = config["network"]
    if "type" in network_config and network_config["type"] == "neural_ode":
        model = NeuralODE.load_from_config(network_config).to(device)
        model.load_weights_from_model_path(modelpath, device=device)
        model.eval()
    else:
        raise ValueError("node-warp-inference-image.py only supports neural_ode models.")

    reconstruct_config = config["reconstruct"]
    if args.framedims:
        grid_dims = [int(d) for d in args.framedims]
    else:
        grid_dims = reconstruct_config.get("frame_dims", [640, 640])

    initialstates = [None] * len(config["initial_conditions"])
    discretep = False
    for i, p in enumerate(config["initial_conditions"].values()):
        try:
            nettype = check_network_type(p)
        except NotTorchFile:
            initialstates[i] = ImageDataset(p, sidelen=grid_dims)
            discretep = True
        else:
            if nettype == "siren":
                initialstates[i] = from_pth(p, w0=1, device=device)
            elif nettype == "mrnet" and WITH_MRNET:
                initialstates[i] = MRFactory.load_state_dict(p).to(device)
            else:
                raise ValueError(f"Unknown network type: {nettype}")

    try:
        p0 = config["initial_conditions"]['0'].split("/")[-1].split("_")[0]
    except KeyError:
        p0 = config["initial_conditions"][0].split("/")[-1].split("_")[0]

    try:
        p1 = config["initial_conditions"]['1'].split("/")[-1].split("_")[0]
    except KeyError:
        p1 = config["initial_conditions"][1].split("/")[-1].split("_")[0]

    imbasename = f"{p0}-{p1}.png"
    baseimpath = osp.join(osp.expanduser(args.outputdir), imbasename)

    if args.timesteps:
        timesteps = [float(t) for t in args.timesteps]
    orginal_timesteps = timesteps.copy()

    # add 0 at the start and 1 at the end if they are not there
    if timesteps[0] != 0:
        timesteps = [0] + timesteps
    if timesteps[-1] != 1:
        timesteps = timesteps + [1]

    blending_type = args.blending

    grid = get_grid(grid_dims).to(device).requires_grad_(False)

    in_between_n = 6

    expanded_timesteps = []
    for i in range(len(timesteps) - 1):
        expanded_timesteps.append(torch.linspace(timesteps[i], timesteps[i+1], in_between_n)[:-1].to(device).float())
    expanded_timesteps.append(torch.tensor([timesteps[-1]]).to(device).float())

    _t = torch.cat(expanded_timesteps).to(device).float()

    model = model.to(device)

    with torch.no_grad():
        model.eval()
        warped_src = model(-_t, grid)
        warped_tgt = model(_t, grid)

        for i, t in enumerate(_t):
            if not (t.item() in orginal_timesteps):
                continue

            if discretep:
                coords0 = warped_src[i]
                coords1 = warped_tgt[-1-i]

                rec0 = initialstates[0].pixels(coords0)
                rec0 = rec0.reshape([
                    grid_dims[0], grid_dims[1], initialstates[0].n_channels
                ])
                rec1 = initialstates[1].pixels(coords1)
                rec1 = rec1.reshape([
                    grid_dims[0], grid_dims[1], initialstates[1].n_channels
                ])
            else:
                raise NotImplementedError("Shapenet inference not implemented yet for NODEs.")

            # print device type of rec0 and rec1
            rec0 = rec0.to(device)
            rec1 = rec1.to(device)
            imbasename_t = f"{p0}-{p1}_{t}.png"

            print("Blending", blending_type, ":", imbasename_t, "...")

            frame = blend_frames(rec0, rec1, t, blending_type)
            if args.landmarks:
                color = None
                lms = None
                ts = None
                if blending_type == "src":
                    color = (225, 0, 0)
                    lms = config["loss"]["sources"]
                    ts = t
                elif blending_type == "tgt":
                    color = (0, 225, 0)
                    lms = config["loss"]["targets"]
                    ts = t - 1
                else:
                    color = [(225, 0, 0), (0, 225, 0)]
                    lms = [
                        config["loss"]["sources"],
                        config["loss"]["targets"]
                    ]
                    ts = [t, t - 1]

                if isinstance(lms, list):
                    for c, lm, t2 in zip(color, lms, ts):
                        lm, _ = warp_points(
                            model, torch.Tensor(lm).to(device=device).float(), t2
                        ).detach().cpu().numpy()
                        frame = plot_landmarks(frame, lm, c=c, r=3)
                else:
                    lm, _ = warp_points(
                        model, torch.Tensor(lms).to(device=device).float(), ts
                    ).detach().cpu().numpy()
                    frame = plot_landmarks(frame, lm=lm, c=color, r=3)

            impath =  osp.join(osp.expanduser(args.outputdir), imbasename_t)
            if not osp.exists(args.outputdir):
                os.makedirs(args.outputdir)
            cv2.imwrite(
                impath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
            print(f"Output image at {t} written to \"{impath}\"")
