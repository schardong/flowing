#!/usr/bin/env python
# coding: utf-8

import argparse
from multiprocessing import Pool
import os
import os.path as osp
import sys
import cv2
import torch
from ifmorph.dataset import check_network_type, NotTorchFile
from ifmorph.model import from_pth
from ifmorph.util import get_landmarks_dlib, image_inference, get_device


def process_face_landmarks(impath, args):
    print(f'Running for image "{impath}".')
    device = get_device(args.device, "cpu")
    dims = [int(d) for d in args.dims.split(",")]

    try:
        _ = check_network_type(impath)
    except NotTorchFile:
        img = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2RGB)
    else:
        faceim = from_pth(impath, device=device).eval()
        img = image_inference(faceim, dims, device=device)

    try:
        lms = get_landmarks_dlib(img).astype(float)
    except ValueError:
        print(f'Failed to find landmarks for "{impath}". Skipping.')
        return
    else:
        key = osp.splitext(osp.split(impath)[1])[0]

    if args.saveim:
        outimpath = osp.splitext(osp.split(impath)[-1])[0] + ".png"
        if args.plot_landmarks:
            lms_copy = lms.astype(int)
            for lm in lms_copy:
                cv2.circle(img, lm[::-1], 3, (0, 255, 0), -1)
        cv2.imwrite(
            osp.join(args.output_path, outimpath),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

    lms[:, 0] = 2.0 * (lms[:, 0] / img.shape[0]) - 1.0
    lms[:, 1] = 2.0 * (lms[:, 1] / img.shape[1]) - 1.0
    outpath = osp.join(args.output_path, key) + ".dat"
    lms.dump(outpath)
    print(f'Saved landmarks under "{outpath}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detects landmarks using DLib from a series of images which"
        " may be encoded as neural networks or not."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Path to the images or PyTorch files representing face images.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        default=".",
        help="Path to the output directory where the landmark files will be"
        " stored. It will be created if it does not exists.",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda:0",
        help='Device to run the inference. By default its "cuda:0". Will'
        ' switch to "cpu" if no CUDA capable device is found.',
    )
    parser.add_argument(
        "--dims",
        type=str,
        default="512,512",
        help="Image dimensions to consider when running the inference. Won't"
        " have any effect if the images are not encoded as neural networks. We"
        " use width,height notation. Note that DLib runs on a face image, thus"
        " larger images may result in more precise landmark placement. Default"
        "is 512,512.",
    )
    parser.add_argument(
        "--n-tasks",
        default=1,
        type=int,
        help="Number of parallel processes to run. By default is 1, meaning a"
        " serial run.",
    )
    parser.add_argument(
        "--saveim",
        "-s",
        action="store_true",
        default=False,
        help="Saves the output images. Saves a copy of the input images if they"
        " are not encoded as neural networks.",
    )
    parser.add_argument(
        "--plot-landmarks",
        "-p",
        action="store_true",
        default=False,
        help="Plots the landmarks over the saved images. Only makes sense when"
        ' passing "--saveim" as well, ignored otherwise.',
    )
    args = parser.parse_args()

    if not osp.exists(args.output_path):
        os.makedirs(args.output_path)

    impaths_to_run = []
    for p in args.images:
        impaths_to_run.append(p)

    if args.n_tasks > 1:
        pool = Pool(processes=args.n_tasks)
        for p in impaths_to_run:
            pool.apply_async(func=process_face_landmarks, args=(p, args))

        pool.close()
        pool.join()
    else:
        for p in impaths_to_run:
            process_face_landmarks(p, args)
