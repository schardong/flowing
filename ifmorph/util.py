# coding: utf-8

import argparse
import math
import os.path as osp
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List
import platform
from ifmorph.point_editor import FaceInteractor

try:
    import dlib
except ImportError:
    print("dlib module not found. Some features may not work.", file=sys.stderr)
    dlib = None

try:
    import mediapipe as mp

    if platform.system() != "Windows":
        mp_face_mesh = mp.solutions.face_mesh
except ImportError:
    print("mediapipe module not found. Some features may not work.", file=sys.stderr)
    mp_face_mesh = None


def warp_points_ncf(
    model: torch.nn.Module, points: torch.Tensor, t: float, preserve_grad: bool = False
) -> torch.Tensor:
    repetitions = 1
    zero_pads = 0 * repetitions
    zeros = torch.zeros(points.shape[0], zero_pads).to(points.device)
    p = points.repeat(1, repetitions)
    p = torch.cat((p, zeros), dim=1)
    t_points = torch.cat((p, torch.full_like(points[..., -1:], t)), dim=-1)
    out = model(t_points)
    return out["model_out"]


def warp_points_node(
    model: torch.nn.Module, points: torch.Tensor, t: float, preserve_grad: bool = False
) -> torch.Tensor:
    repetitions = 1
    zero_pads = 0 * repetitions
    zeros = torch.zeros(points.shape[0], zero_pads).to(points.device)
    p = points.repeat(1, repetitions)
    p = torch.cat((p, zeros), dim=1)
    t_points = torch.cat((p, torch.full_like(points[..., -1:], t)), dim=-1)
    out = model(torch.full_like(points[..., -1:], t), p)
    return out


def get_dlib_coord_dict():
    return {
        "sillhouete": list(range(0, 17)),
        "left_eyebrow": list(range(22, 27)),
        "right_eyebrow": list(range(17, 22)),
        "nose": list(range(27, 31)),
        "lower_nose": list(range(31, 36)),
        "left_eye": list(range(42, 48)),
        "right_eye": list(range(36, 42)),
        "upper_lip_outer": list(range(48, 55)),
        "lower_lip_outer": list(range(55, 51)),
        "upper_lip_inner": [61, 62, 63],
        "lower_lip_inner": [65, 66, 67],
    }


def get_landmarks_dlib(img):
    """Uses DLib 68 point model to find face landmarks in `img`.

    Parameters
    ----------
    img: np.array
        The image containing the face. We didn't test with multiple faces.

    Returns
    -------
    landmarks: np.array
        An [N, 2] shaped array with coordinates of the landmarks in `img`.

    Raises
    ------
    ValueError: if no faces are detected
    """
    detector_dlib = dlib.get_frontal_face_detector()
    predictor_dlib = dlib.shape_predictor(
        osp.join("landmark_models", "shape_predictor_68_face_landmarks_GTX.dat")
    )
    landmarks = []
    try:
        facerect = detector_dlib(img, 1)[0]
    except IndexError:
        print("No faces detected.")
        raise ValueError("No faces detected.")
    shape = predictor_dlib(img, facerect)
    for i in range(shape.num_parts):
        p = shape.part(i)
        landmarks.append((p.y, p.x))

    landmarks = np.array(landmarks)
    return landmarks


def get_landmarks_hull(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 100, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    return biggest_contour[:, 0, :]


def get_grid(dims, requires_grad=False, list_of_coords=True):
    """Returns a `dims`-shaped grid.

    Parameters
    ----------
    dims: number, tuple[numbers]
        Dimensions of the output grid

    requires_grad: boolean, optional
        Set to True if you want to retain the gradient of the output grid.
        Default value is False.

    list_of_coords: boolean, optional
        Set to False to return the dims shaped grid, or keep it True to
        return a prod(dims), len(dims) shaped tensor, i.e., a list of
        coordinates. Default value is True.

    Returns
    -------
    coords: torch.Tensor
        Note that the coordinates will always be in range [-1, 1].
    """
    if isinstance(dims, int):
        dims = [dims]
    tensors = tuple(
        [torch.linspace(-1 + (1.0 / d), 1 - (1.0 / d), steps=d) for d in dims]
    )
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    if list_of_coords:
        mgrid = mgrid.reshape(-1, len(dims))
    mgrid.requires_grad_(requires_grad)
    return mgrid


def get_silhouette_lm(img: np.ndarray, method: str = "dlib") -> np.ndarray:
    """Returns the silhouette landmarks from `img` in pixel coordinates.

    Parameters
    ----------
    img: numpy.ndarray
        The input image with shape [H, W, C] with value range in [0, 255].

    method: str, optional
        The face detection/landmark method to use. Only accepts "dlib" for now.

    Returns
    -------
    landmarks: list
        A list of pairs of coordinates with locations of each landmark in image
        space.
    """
    landmarks = []
    if method == "dlib":
        coord_dict = get_dlib_coord_dict()
        maskpts = coord_dict["sillhouete"]
        maskpts.extend(coord_dict["left_eyebrow"][::-1])
        maskpts.extend(coord_dict["right_eyebrow"][::-1])

        detector_dlib = dlib.get_frontal_face_detector()
        predictor_dlib = dlib.shape_predictor(
            osp.join("landmark_models", "shape_predictor_68_face_landmarks_GTX.dat")
        )
        try:
            faces = detector_dlib(img, 1)
            facerect = faces[0]
        except IndexError:
            print("No faces detected.")
            raise
        shape = predictor_dlib(img, facerect)
        for i in range(shape.num_parts):
            p = shape.part(i)
            landmarks.append((p.x, p.y))
        landmarks = np.array(landmarks)[maskpts]
    elif method == "convex-hull":
        landmarks = get_landmarks_hull(img)
    else:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        baseopts = BaseOptions(
            model_asset_path=osp.join("landmark_models", "face_landmarker.task"),
        )
        lmopts = vision.FaceLandmarkerOptions(
            base_options=baseopts,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        detector = vision.FaceLandmarker.create_from_options(lmopts)

        mpim = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detections = detector.detect(mpim)
        landmarks = detections.face_landmarks[0]

        landmarks = np.array([[lm.x, lm.y] for lm in landmarks])
        landmarks[:, 0] *= img.shape[1]
        landmarks[:, 1] *= img.shape[0]
        landmarks = landmarks.astype(np.int32)
        landmarks = landmarks[
            [
                10,
                109,
                67,
                103,
                54,
                21,
                162,
                127,
                234,
                93,
                132,
                58,
                172,
                136,
                150,
                149,
                176,
                148,
                152,
                377,
                400,
                378,
                379,
                365,
                397,
                288,
                361,
                323,
                454,
                356,
                389,
                251,
                284,
                332,
                297,
                338,
            ],
            :,
        ]

    return landmarks


def get_bottom_face_lm(img):
    landmarks = []
    coord_dict = get_dlib_coord_dict()
    maskpts = coord_dict["sillhouete"][1:-1]
    maskpts.extend(coord_dict["lower_nose"][::-1])

    detector_dlib = dlib.get_frontal_face_detector()
    predictor_dlib = dlib.shape_predictor(
        osp.join("landmark_models", "shape_predictor_68_face_landmarks_GTX.dat")
    )
    try:
        faces = detector_dlib(img, 1)
        facerect = faces[0]
    except IndexError:
        print("No faces detected.")
        raise
    shape = predictor_dlib(img, facerect)
    for i in range(shape.num_parts):
        p = shape.part(i)
        landmarks.append((p.x, p.y))
    landmarks = np.array(landmarks)[maskpts]

    return landmarks


def plot_scalars(t: torch.Tensor):
    """Plots a batch of the input tensors as images. The input tensors must
    contain scalar values, e.g. their shape must be 1xHxW

    Parameters
    ----------
    t: torch.Tensor
        The input tensor with shape Bx1xHxW.

    Returns
    -------
    fig: matplotlib.Figure
    ax: matplotlib.Axes
    """
    N = t.shape[0]
    rows = math.ceil(math.sqrt(N))
    cols = math.ceil(N / rows)
    fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
        for i in range(N):
            img = t[i, ...].permute(1, 2, 0)
            ax[i].imshow(img, cmap="gray")
            ax[i].axis("off")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    else:
        img = t[0, ...].permute(1, 2, 0)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig, ax


def batched_predict(model: torch.nn.Module, coord_arr: torch.Tensor, batch_size: int):
    """Runs model prediction in batches to avoid memory issues."""
    with torch.no_grad():
        n = coord_arr.shape[0]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + batch_size, n)
            yhat = model(coord_arr[ql:qr, :])["model_out"]
            preds.append(yhat)
            ql = qr
        preds = torch.cat(preds, dim=0)
    return preds


def warp_shapenet_inference(
    grid,
    t,
    warpnet,
    shapenet,
    framedims=None,
    bggray=0,
    preserve_grad=False,
    normalize_to_byte=True,
):
    """Performs the inference on a `warpnet` and feeds the results to a
    `shapenet`, normalizing the areas out-of-domain.

    Parameters
    ----------
    grid: torch.Tensor
        [N, 2] tensor of points to be warped and used for `shapenet` inference.

    t: float
        Parameter in [-1, 1] range to warp the points in `grid`.

    warpnet: torch.nn.Module
        A network that performs the warping of coordinates in `grid_with_t`.
        Must accept a [N, 3] tensor as input, and outputs a [N, 2] tensor.

    shapenet: torch.nn.Module
        A network that receives the [N, 2] output of `flownet` and runs the
        inference on those points. It must output a an Nx{1,3} tensor,
        depending on the number of channels of the output image.

    framedims: tuple with *2* ints, optional
        The number of rows and columns of the output image. Note that the
        number of channels is inferred from `shapenet`. By default is `None`,
        meaning that no reshape will be performed.

    preserve_grad: boolean, optional
        Default is False.

    normalize_to_byte: boolean, optional
        Default is True.

    Returns
    -------
    img: torch.Tensor
        The resulting image with shape `(framedims[0], framedims[1],
        shapenet.out_features)`

    coords: torch.Tensor
        The warped coordinates used as input for `shapenet`.
    """
    wcoords, _ = warp_points(warpnet, grid, t, preserve_grad=preserve_grad)
    out = shapenet(wcoords, preserve_grad=preserve_grad)
    img = out["model_out"].clamp(0, 1)
    coords = out["model_in"]
    if preserve_grad is False:
        img = img.detach()
    if normalize_to_byte:
        img *= 255

    # restrict to the image domain [-1,1]^2
    img = torch.where(
        torch.abs(coords[..., 0].unsqueeze(-1)) < 1.0, img, torch.full_like(img, bggray)
    )
    img = torch.where(
        torch.abs(coords[..., 1].unsqueeze(-1)) < 1.0, img, torch.full_like(img, bggray)
    )
    if framedims is not None:
        img = img.reshape([framedims[0], framedims[1], shapenet.out_features])
    return img, coords


def warp_points(
    model: torch.nn.Module,
    points: torch.Tensor,
    t: float,
    preserve_grad: bool = False,
    clip: bool = True,
) -> torch.Tensor:
    """Warps `points` by parameter `t` in range [-1, 1] using `model`.

    # TODO: Add this as a method to the network models. This would remove the
    workaround below.

    Parameters
    ----------
    model: torch.nn.Module
        A warping network that maps [N, 3] points to [N, 2] points.

    points: torch.Tensor
        [N, 2] tensor of points to be warped.

    t: number
        Parameter in [-1, 1] range to warp the points.

    preserve_grad: boolean, optional
        Default is False.

    Returns
    -------
    warped_points: torch.Tensor
        Output of `model(points)`.

    input_points: torch.Tensor
        Input of `model(points)`. Needed if `preserve_grad=True`, otherwise
        its a perfect copy of `points`.
    """

    # --- GAMBIARRA ---
    repetitions = 1
    zero_pads = 0 * repetitions
    zeros = torch.zeros(points.shape[0], zero_pads).to(points.device)
    p = points.repeat(1, repetitions)
    p = torch.cat((p, zeros), dim=1)
    t_points = torch.cat((p, torch.full_like(points[..., -1:], t)), dim=-1)  # data[0]
    # --- GAMBIARRA ---

    out = model(t_points, preserve_grad=preserve_grad)

    if clip:
        out["model_out"] = out["model_out"].clamp(-1, 1)

    return out["model_out"][..., :2], out["model_in"][..., :2]


def time_warp_points(
    model: torch.nn.Module,
    points: torch.Tensor,
    t: float,
    preserve_grad: bool = False,
    clip: bool = True,
) -> torch.Tensor:
    """Warps `points` by parameter `t` in range [-1, 1] using `model`.

    Parameters
    ----------
    model: torch.nn.Module
        A warping network that maps [N, 3] points to [N, 2] points.

    points: torch.Tensor
        [N, 2] tensor of points to be warped.

    t: number
        Parameter in [-1, 1] range to warp the points.

    preserve_grad: boolean, optional
        Default is False.

    Returns
    -------
    warped_points: torch.Tensor
        Output of `model(points)`.

    input_points: torch.Tensor
        Input of `model(points)`. Needed if `preserve_grad=True`, otherwise
        its a perfect copy of `points`.
    """

    p = points
    t_points = torch.cat((torch.full_like(points[..., -1:], t), p), dim=-1)  # data[0]

    out = model(t_points, preserve_grad=preserve_grad)

    if clip:
        out["model_out"] = out["model_out"].clamp(-1, 1)

    return out["model_out"][..., 1:3], out["model_in"][..., 1:3]


def plot_landmarks(im: np.array, landmarks: np.array, c=(0, 255, 0), r=1) -> np.array:
    """Overlays landmarks `lm` on the image `im` with colors `c` and radius
    `r`.

    Parameters
    ----------
    im: np.array
        The HxWx3 image.

    lm: np.array
        The Nx2 landmark coordinates. If they are in range [-1, 1] or [0, 1],
        we will normalize them using `im.shape`.

    c: tuple, optional
        The RGB landmark color.

    r: int, optional
        The landmark radius

    Returns
    -------
    imnp: np.ndarray
        The input image in numpy format with landmarks overlaid.
    """
    imc = im.copy()
    lmn = landmarks.copy()

    if landmarks.max() <= 1.0:  # assumed to be in [?, 1] range
        if landmarks.min() < 0:  # assumed to be in range [-1, 1]
            lmn = (lmn + 1.0) / 2.0
        lmn[:, 0] *= imc.shape[0]
        lmn[:, 1] *= imc.shape[1]
    lmn = lmn.astype(np.uint32)

    # rad = int(math.ceil(r / 2))
    for lm in lmn:
        imc = cv2.circle(imc, lm[::-1], radius=r, color=c, thickness=-1)

    return imc


def blend_frames(
    f1: torch.Tensor,
    f2: torch.Tensor,
    t: float,
    blending_type: str,
    mask_0: torch.Tensor = None,
    mask_1: torch.Tensor = None,
    landmark_detection: str = "mediapipe",
) -> np.array:
    """Blends frames `f1` and `f2` following `blending_type`.

    Parameters
    ----------
    f1: torch.Tensor
        The [M,N] shaped initial frame.

    f2: torch.Tensor
        The [M,N] shaped final frame.

    t: float
        The time parameter for blending. For most blending types, `t=0` returns
        `f1` and `t=1` returns `f2`.

    blending_type: str
        The blending method. May be any one of: linear{_masked}, src, tgt, min,
        max, seamless_{mix,clone}_{src,tgt}

    Returns
    -------
    blended_frame: np.array
        The blended frame as a numpy array. Already normalized to 0-255 range
        and with dtype=np.uint8.
    """
    if mask_0 is None:
        mask_0 = torch.ones_like(f1) == 1
    if mask_1 is None:
        mask_1 = torch.ones_like(f2) == 1

    if blending_type == "linear":
        xor_mask = mask_0 ^ mask_1
        left_edge_weight = xor_mask * (mask_0 * t - mask_1 * (1 - t))
        right_edge_weight = xor_mask * (mask_1 * (1 - t) - mask_0 * t)
        rec = (1 - t + left_edge_weight) * f1 + (t + right_edge_weight) * f2
        rec = (1 - t) * f1 + t * f2
    elif blending_type == "linear_masked":
        f1np = f1.detach().cpu().numpy()
        if f1np.max() <= 1.0:
            f1np *= 255.0
        f1np = f1np.astype(np.uint8)

        f2np = f2.detach().cpu().numpy()
        if f2np.max() <= 1.0:
            f2np *= 255.0
        f2np = f2np.astype(np.uint8)

        landmarks = get_silhouette_lm(f1np, method=landmark_detection).astype(np.int32)
        mask = np.zeros(f2.shape, dtype=np.uint8)
        mask_full = cv2.fillPoly(
            mask, np.array([landmarks], dtype=np.int32), (255, 255, 255)
        )
        mask = mask_full[..., 0]

        f1_masked = cv2.bitwise_and(f1np, f1np, mask=mask)
        f2_masked = cv2.bitwise_and(f2np, f2np, mask=mask)

        ft_masked = (1 - t) * f1_masked + t * f2_masked
        rec = cv2.bitwise_and(f1np, f1np, mask=cv2.bitwise_not(mask))
        rec = rec.astype(np.uint8)

        # Use seamless cloning to blend the masked interpolation with the
        # background
        br = cv2.boundingRect(mask)
        center = (int(br[0] + br[2] / 2), int(br[1] + br[3] / 2))

        rec = cv2.seamlessClone(
            ft_masked.astype(np.uint8),
            f1np,
            mask_full,
            p=center,
            flags=cv2.NORMAL_CLONE,
        )

    elif "min" in blending_type:
        rec = torch.minimum(f1, f2)
    elif "max" in blending_type:
        xor_mask = mask_0 ^ mask_1
        xor_mask = ~xor_mask
        left_mask = xor_mask | mask_0
        right_mask = xor_mask | mask_1
        rec = torch.maximum(f1 * left_mask, f2 * right_mask)
    elif "bump" in blending_type:
        right_coeff = 0.5 * (torch.tanh(6 * (torch.tensor(t) - 0.5)) + 1)
        left_coeff = 1 - right_coeff
        rec = left_coeff * f1 + right_coeff * f2
    elif "src" in blending_type:
        rec = f1
    elif "tgt" in blending_type:
        rec = f2
    elif "seamless" in blending_type:
        # Invert source and target images.
        if "tgt" in blending_type:
            f1, f2 = f2, f1

        flags = cv2.NORMAL_CLONE
        if "mix" in blending_type:
            flags = cv2.MIXED_CLONE
        else:
            f2 = (1 - t) * f1 + t * f2

        f1np = f1.detach().cpu().numpy()
        if f1np.max() <= 1.0:
            f1np *= 255.0
        f1np = f1np.astype(np.uint8)

        f2np = f2.detach().cpu().numpy()
        if f2np.max() <= 1.0:
            f2np *= 255
        f2np = f2np.astype(np.uint8)

        landmarks = get_silhouette_lm(f1np, method=landmark_detection).astype(np.int32)

        mask = np.zeros(f2.shape, dtype=np.uint8)
        mask = cv2.fillPoly(
            mask, np.array([landmarks], dtype=np.int32), (255, 255, 255)
        )
        br = cv2.boundingRect(mask[:, :, 0])
        center = (int(br[0] + br[2] / 2), int(br[1] + br[3] / 2))
        kernel = np.ones((5, 5), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations=1)

        rec = cv2.seamlessClone(f2np, f1np, mask_eroded, p=center, flags=flags)

    if isinstance(rec, torch.Tensor):
        if rec.max() <= 1.0:
            rec *= 255.0
        rec = rec.detach().cpu().numpy().astype(np.uint8)

    return rec


def create_morphing(
    warp_net: torch.nn.Module,
    frame0: torch.nn.Module,
    frame1: torch.nn.Module,
    output_path: str,
    frame_dims: tuple,
    n_frames: int,
    fps: int,
    device: torch.device,
    landmark_src,
    landmark_tgt,
    overlay_landmarks=True,
    blending_type="linear",
    frame_collection: List[torch.nn.Module] = None,
    edge_mask=False,
):
    """Creates a video file given a model and output path.

    Parameters
    ----------
    warp_net: torch.nn.Module
        The warping model. It must be a coordinate model with three
        inputs (u, v, t), where u, v, t are in range[-1, 1]. u, v are spatial
        coordinates, while t is the time.

    output_path: str, PathLike
        Output path to save the generated video.

    frame_dims: tuple of ints
        The (width, height) of each video frame.

    n_frames: int
        The number of frames in the final video.

    fps: int
        Frames-per-second of the output video.

    device: torch.device
        The device to run the inference for all networks. All intermediate
        tensors will be allocated with this device option.

    landmark_src: torch.Tensor
        Warping source landmark.

    landmark_tgt: torch.Tensor
        Warping target landmarks.

    overlay_landmarks: boolean, optional
        Switch to plot the warping points over the video (if True), or not
        (if False, default behaviour)

    blending_type: str, optional
        How to perform the blending of the initial states, may be "linear"
        (default), which performs a linear interpolation of the states.
        "minimum" and "maximum" get minimum(maximum) color values between the
        inferences of `shape_net0` and `shape_net1` at the warped coordinates.
        Finally, `dist` performs a distance based blending.

    frame_collection: list, optional
        List of frames to be used in the morphing. If not given, frame0 and frame1 will be used

    Returns
    -------
    Nothing
    """
    warp_net = warp_net.eval()
    continuousp = False
    if frame_collection is None:
        frame_collection = [frame0, frame1]

    # If the frames are continuous images (i.e. SIRENs), we must set them to
    # "eval" mode.
    if isinstance(frame_collection[0], torch.nn.Module):
        for frame in frame_collection:
            frame.eval()
        continuousp = True

    t1 = 0
    t2 = 1
    times = np.arange(t1, t2, (t2 - t1) / n_frames)
    grid = get_grid(frame_dims).to(device).requires_grad_(False)

    print(
        f"Doing morphing with parameters | n_frames: {n_frames} | fps: {fps} | continuousp {continuousp} | frame0 {frame0} "
    )
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_dims[::-1], True
    )

    with torch.no_grad():
        if isinstance(landmark_src, torch.Tensor):
            landmark_src = landmark_src.clone().detach()
        else:
            landmark_src = torch.Tensor(landmark_src).to(device).float()

        if isinstance(landmark_tgt, torch.Tensor):
            landmark_tgt = landmark_tgt.clone().detach()
        else:
            landmark_tgt = torch.Tensor(landmark_tgt).to(device).float()

        for t in times:
            mask_0 = None
            mask_1 = None

            left_index = math.floor(t * (len(frame_collection) - 1))
            right_index = left_index + 1
            left_t = left_index / (len(frame_collection) - 1)
            right_t = right_index / (len(frame_collection) - 1)

            frame_left = frame_collection[left_index]
            frame_right = frame_collection[right_index]

            if continuousp:
                rec0, _ = warp_shapenet_inference(
                    grid, left_t - t, warp_net, frame_left, frame_dims
                )
                rec1, _ = warp_shapenet_inference(
                    grid, right_t - t, warp_net, frame_right, frame_dims
                )
            else:
                landmark_src_warped, _ = warp_points(warp_net, grid, left_t - t)
                wpoints = landmark_src_warped
                mask_0 = (
                    (wpoints[..., 0] > -1)
                    & (wpoints[..., 0] < 1)
                    & (wpoints[..., 1] > -1)
                    & (wpoints[..., 1] < 1)
                )
                mask_0 = mask_0.reshape([frame_dims[0], frame_dims[1], 1])
                rec0 = frame_left.pixels(wpoints).reshape(
                    [frame_dims[0], frame_dims[1], frame_left.n_channels]
                )
                mask_0 = mask_0.to(rec0.device)

                landmark_tgt_warped, _ = warp_points(warp_net, grid, right_t - t)
                wpoints = landmark_tgt_warped
                mask_1 = (
                    (wpoints[..., 0] > -1)
                    & (wpoints[..., 0] < 1)
                    & (wpoints[..., 1] > -1)
                    & (wpoints[..., 1] < 1)
                )
                mask_1 = mask_1.reshape([frame_dims[0], frame_dims[1], 1])

                rec1 = frame_right.pixels(wpoints).reshape(
                    [frame_dims[0], frame_dims[1], frame_right.n_channels]
                )

                mask_1 = mask_1.to(rec1.device)

            blending_coeff = (t - left_t) / (right_t - left_t)
            if not edge_mask:
                mask_0 = None
                mask_1 = None
            rec = blend_frames(
                rec0,
                rec1,
                blending_coeff,
                blending_type=blending_type,
                mask_0=mask_0,
                mask_1=mask_1,
            )
            rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
            cv2.putText(
                img=rec,
                text="Time = %.2f" % (t.item()),
                org=(0, 30),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
            )

            if overlay_landmarks:
                for c, pts, ts in zip(
                    [(255, 0, 0), (0, 255, 255)],
                    [landmark_src, landmark_tgt],
                    [t, t - 1],
                ):
                    y, _ = warp_points(
                        warp_net, torch.Tensor(pts).to(device).float(), ts
                    )
                    y = y.detach().cpu().numpy()
                    rec = plot_landmarks(rec, y, c=c, r=4)

            out.write(rec)
    out.release()


def create_node_morphing(
    warp_net: torch.nn.Module,
    frame0: torch.nn.Module,
    frame1: torch.nn.Module,
    output_path: str,
    frame_dims: tuple,
    n_frames: int,
    fps: int,
    device: torch.device,
    landmark_src,
    landmark_tgt,
    overlay_landmarks=True,
    blending_type="linear",
    frame_collection: List[torch.nn.Module] = None,
    edge_mask=False,
):
    """Creates a video file given a model and output path.

    Parameters
    ----------
    warp_net: torch.nn.Module
        The warping model. It must be a coordinate model with three
        inputs (u, v, t), where u, v, t are in range[-1, 1]. u, v are spatial
        coordinates, while t is the time.

    output_path: str, PathLike
        Output path to save the generated video.

    frame_dims: tuple of ints
        The (width, height) of each video frame.

    n_frames: int
        The number of frames in the final video.

    fps: int
        Frames-per-second of the output video.

    device: torch.device
        The device to run the inference for all networks. All intermediate
        tensors will be allocated with this device option.

    landmark_src: torch.Tensor
        Warping source landmark.

    landmark_tgt: torch.Tensor
        Warping target landmarks.

    overlay_landmarks: boolean, optional
        Switch to plot the warping points over the video (if True), or not
        (if False, default behaviour)

    blending_type: str, optional
        How to perform the blending of the initial states, may be "linear"
        (default), which performs a linear interpolation of the states.
        "minimum" and "maximum" get minimum(maximum) color values between the
        inferences of `shape_net0` and `shape_net1` at the warped coordinates.
        Finally, `dist` performs a distance based blending.

    frame_collection: list, optional
        List of frames to be used in the morphing. If not given, frame0 and frame1 will be used

    Returns
    -------
    Nothing
    """
    warp_net = warp_net.eval().to(device)
    continuousp = False
    if frame_collection is None:
        frame_collection = [frame0, frame1]

    # If the frames are continuous images (i.e. SIRENs), we must set them to
    # "eval" mode.
    if isinstance(frame_collection[0], torch.nn.Module):
        for frame in frame_collection:
            frame.eval()
        continuousp = True

    t1 = 0
    t2 = 1
    times = np.arange(t1, t2, (t2 - t1) / n_frames)
    grid = get_grid(frame_dims).to(device).requires_grad_(False)

    print(
        f"Doing morphing with parameters | n_frames: {n_frames} | fps: {fps} | continuousp {continuousp} | frame0 {frame0} "
    )
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_dims[::-1], True
    )

    with torch.no_grad():
        if isinstance(landmark_src, torch.Tensor):
            landmark_src = landmark_src.clone().detach()
        else:
            landmark_src = torch.Tensor(landmark_src).to(device).float()

        if isinstance(landmark_tgt, torch.Tensor):
            landmark_tgt = landmark_tgt.clone().detach()
        else:
            landmark_tgt = torch.Tensor(landmark_tgt).to(device).float()

        _times = torch.tensor(times).to(device).float()
        warped_src = warp_net(-_times, grid)  # .detach().cpu().numpy()
        mark_src = warp_net(_times, landmark_src).detach().cpu().numpy()
        warped_tgt = warp_net(_times, grid)  # .detach().cpu().numpy()
        mark_tgt = warp_net(-_times, landmark_tgt).detach().cpu().numpy()

        for i, t in enumerate(times):

            left_index = math.floor(t * (len(frame_collection) - 1))
            right_index = left_index + 1
            left_t = left_index / (len(frame_collection) - 1)
            right_t = right_index / (len(frame_collection) - 1)

            frame_left = frame_collection[left_index]
            frame_right = frame_collection[right_index]

            landmark_src_warped = warped_src[i]
            wpoints = landmark_src_warped
            rec0 = frame_left.pixels(wpoints).reshape(
                [frame_dims[0], frame_dims[1], frame_left.n_channels]
            )

            landmark_tgt_warped = warped_tgt[-1 - i]
            wpoints = landmark_tgt_warped

            rec1 = frame_right.pixels(wpoints).reshape(
                [frame_dims[0], frame_dims[1], frame_right.n_channels]
            )

            blending_coeff = (t - left_t) / (right_t - left_t)
            if not edge_mask:
                mask_0 = None
                mask_1 = None
            rec = blend_frames(
                rec0,
                rec1,
                blending_coeff,
                blending_type=blending_type,
                mask_0=mask_0,
                mask_1=mask_1,
            )
            rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
            cv2.putText(
                img=rec,
                text="Time = %.2f" % (t.item()),
                org=(0, 30),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
            )

            if overlay_landmarks:
                rec = plot_landmarks(rec, mark_src[i], c=(255, 0, 0), r=4)
                rec = plot_landmarks(rec, mark_tgt[-1 - i], c=(0, 255, 255), r=4)

            out.write(rec)
    out.release()


def create_time_morphing(
    warp_net: torch.nn.Module,
    frame0: torch.nn.Module,
    frame1: torch.nn.Module,
    output_path: str,
    frame_dims: tuple,
    n_frames: int,
    fps: int,
    device: torch.device,
    landmark_src,
    landmark_tgt,
    overlay_landmarks=True,
    blending_type="linear",
    frame_collection: List[torch.nn.Module] = None,
    edge_mask=False,
):
    """Creates a video file given a model and output path.

    Parameters
    ----------
    warp_net: torch.nn.Module
        The warping model. It must be a coordinate model with three
        inputs (u, v, t), where u, v, t are in range[-1, 1]. u, v are spatial
        coordinates, while t is the time.

    output_path: str, PathLike
        Output path to save the generated video.

    frame_dims: tuple of ints
        The (width, height) of each video frame.

    n_frames: int
        The number of frames in the final video.

    fps: int
        Frames-per-second of the output video.

    device: torch.device
        The device to run the inference for all networks. All intermediate
        tensors will be allocated with this device option.

    landmark_src: torch.Tensor
        Warping source landmark.

    landmark_tgt: torch.Tensor
        Warping target landmarks.

    overlay_landmarks: boolean, optional
        Switch to plot the warping points over the video (if True), or not
        (if False, default behaviour)

    blending_type: str, optional
        How to perform the blending of the initial states, may be "linear"
        (default), which performs a linear interpolation of the states.
        "minimum" and "maximum" get minimum(maximum) color values between the
        inferences of `shape_net0` and `shape_net1` at the warped coordinates.
        Finally, `dist` performs a distance based blending.

    frame_collection: list, optional
        List of frames to be used in the morphing. If not given, frame0 and frame1 will be used

    Returns
    -------
    Nothing
    """
    warp_net = warp_net.eval()
    continuousp = False
    if frame_collection is None:
        frame_collection = [frame0, frame1]

    # If the frames are continuous images (i.e. SIRENs), we must set them to
    # "eval" mode.
    if isinstance(frame_collection[0], torch.nn.Module):
        for frame in frame_collection:
            frame.eval()
        continuousp = True

    t1 = 0
    t2 = 1
    times = np.arange(t1, t2, (t2 - t1) / n_frames)
    # frame_dims = (*frame_dims, 1)
    x_grid = torch.linspace(-1, 1, frame_dims[0])
    y_grid = torch.linspace(-1, 1, frame_dims[1])
    grid0 = (
        torch.cartesian_prod(torch.zeros(1), x_grid, y_grid)
        .to(device)
        .requires_grad_(False)
    )
    grid1 = (
        torch.cartesian_prod(torch.ones(1), x_grid, y_grid)
        .to(device)
        .requires_grad_(False)
    )

    print(
        f"Doing morphing with parameters | n_frames: {n_frames} | fps: {fps} | continuousp {continuousp} | frame0 {frame0} "
    )
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_dims[::-1], True
    )

    with torch.no_grad():
        if isinstance(landmark_src, torch.Tensor):
            landmark_src = landmark_src.clone().detach()
        else:
            landmark_src = torch.Tensor(landmark_src).to(device).float()

        if isinstance(landmark_tgt, torch.Tensor):
            landmark_tgt = landmark_tgt.clone().detach()
        else:
            landmark_tgt = torch.Tensor(landmark_tgt).to(device).float()

        for t in times:
            mask_0 = None
            mask_1 = None

            left_index = math.floor(t * (len(frame_collection) - 1))
            right_index = left_index + 1
            left_t = left_index / (len(frame_collection) - 1)
            right_t = right_index / (len(frame_collection) - 1)

            frame_left = frame_collection[left_index]
            frame_right = frame_collection[right_index]

            if continuousp:
                rec0, _ = warp_shapenet_inference(
                    grid0, left_t - t, warp_net, frame_left, frame_dims
                )
                rec1, _ = warp_shapenet_inference(
                    grid1, right_t - t, warp_net, frame_right, frame_dims
                )
            else:
                wpoints, _ = time_warp_points(warp_net, grid1, left_t - t)
                mask_0 = (
                    (wpoints[..., 0] > -1)
                    & (wpoints[..., 0] < 1)
                    & (wpoints[..., 1] > -1)
                    & (wpoints[..., 1] < 1)
                )
                mask_0 = mask_0.reshape([frame_dims[0], frame_dims[1], 1])

                rec0 = frame_left.pixels(wpoints).reshape(
                    [frame_dims[0], frame_dims[1], frame_left.n_channels]
                )
                mask_0 = mask_0.to(rec0.device)

                wpoints, _ = time_warp_points(warp_net, grid0, right_t - t)
                mask_1 = (
                    (wpoints[..., 0] > -1)
                    & (wpoints[..., 0] < 1)
                    & (wpoints[..., 1] > -1)
                    & (wpoints[..., 1] < 1)
                )
                mask_1 = mask_1.reshape([frame_dims[0], frame_dims[1], 1])

                rec1 = frame_right.pixels(wpoints).reshape(
                    [frame_dims[0], frame_dims[1], frame_right.n_channels]
                )

                mask_1 = mask_1.to(rec1.device)

            blending_coeff = (t - left_t) / (right_t - left_t)
            if not edge_mask:
                mask_0 = None
                mask_1 = None
            rec = blend_frames(
                rec0,
                rec1,
                blending_coeff,
                blending_type=blending_type,
                mask_0=mask_0,
                mask_1=mask_1,
            )
            rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
            cv2.putText(
                img=rec,
                text="Time = %.2f" % (t.item()),
                org=(0, 30),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
            )

            if overlay_landmarks:
                for c, pts, ts in zip(
                    [(255, 0, 0), (0, 255, 255)],
                    [landmark_src, landmark_tgt],
                    [t, t - 1],
                ):
                    y, _ = warp_points(
                        warp_net, torch.Tensor(pts).to(device).float(), ts
                    )
                    y = y.detach().cpu().numpy()
                    rec = plot_landmarks(rec, y, c=c, r=4)

            out.write(rec)
    out.release()


def grid_image(coords):
    N = 8
    M = 8
    x = coords[..., 0].unsqueeze(-1) * N
    y = coords[..., 1].unsqueeze(-1) * M

    colors = torch.exp(-((x - torch.floor(x)) ** 2) / 0.001)
    colors += torch.exp(-((y - torch.floor(y)) ** 2) / 0.001)
    colors = 1 - torch.clamp(colors, 0, 1)

    return colors


def image_inference(model, grid_dims, device=torch.device("cpu")):
    """Runs inference on the model. We assume that `model` represents an
    image.

    Parameters
    ----------
    model: torch.nn.Module
        The model to run inference on. The `forward` call must return a
        dictionary with a `model_out` key.

    grid_dims: collection of *two* ints
        (height, width) of the output image. We don't check for invalid
        values (<= 0).

    device: str or torch.Device, optional
        The device to run the inference on. We will create an evenly spaced
        grid, transfer the model and run the inference on this device. Note
        that we don't transfer the model back to its original device, nor
        check if the device is the same as the old one. Default value is
        `"cpu"`.

    Returns
    -------
    img: torch.Tensor
        The inference results with shape (grid_dims[0], grid_dims[1],
        n_channels), where n_channels is model.out_features.
    """
    model = model.eval().to(device)
    n_channels = model.out_features
    grid = get_grid(grid_dims).to(device)

    with torch.no_grad():
        img = model(grid)["model_out"].detach().clamp(0, 1) * 255
        img = (
            img.cpu()
            .reshape((grid_dims[0], grid_dims[1], n_channels))
            .numpy()
            .astype(np.uint8)
        )

    return img


def get_landmarks(face_mesh, img):
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    landmarks = results.multi_face_landmarks[0].landmark
    landmarks_list = [[landmark.y, landmark.x] for landmark in landmarks]

    larndmarks_np = np.array(landmarks_list)

    dict_face = {
        "silhouette": [
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
        ],
        "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
        "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
        "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
        "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
        "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
        # rightEyeUpper1: [247, 30, 29, 27, 28, 56, 190],
        # rightEyeLower1: [130, 25, 110, 24, 23, 22, 26, 112, 243],
        # rightEyeUpper2: [113, 225, 224, 223, 222, 221, 189],
        # rightEyeLower2: [226, 31, 228, 229, 230, 231, 232, 233, 244],
        # rightEyeLower3: [143, 111, 117, 118, 119, 120, 121, 128, 245],
        "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
        # "rightEyebrowLower": [35, 124, 46, 53, 52, 65],
        # 'rightEyeIris': [473, 474, 475, 476, 477],
        "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
        "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
        # "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
        # "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
        # "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
        # "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
        # "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],
        "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
        # "leftEyebrowLower": [265, 353, 276, 283, 282, 295],
        # 'leftEyeIris': [468, 469, 470, 471, 472],
        "midwayBetweenEyes": [168],
        "noseTip": [1],
        "noseBottom": [2],
        "noseRightCorner": [98],
        "noseLeftCorner": [327],
        "rightCheek": [205],
        "leftCheek": [425],
    }

    list_final = []
    for k, v in dict_face.items():
        list_final = list_final + v

    return larndmarks_np[list_final]


def get_landmark_correspondences(
    frame0,
    frame1,
    frame_dims,
    src_pts=None,
    tgt_pts=None,
    device="cuda:0",
    method=None,
    open_ui=True,
):
    """Opens the `FaceInteractor` UI to get the landmark correspondences
    between `frame0` and `frame1`.

    Its possible for this function to be a giant no-op if `method` is `None`
    and `open_ui` is `False`, since the images will be resized (or, if
    `frame{0,1}` are networks, their inference will be performed) and no
    landmarks will be created. We test for this condition and return empty
    lists if detected.

    Parameters
    ----------
    frame0: torch.nn.Module or np.array
        The source frame as a neural network or a discrete image.

    frame1: torch.nn.Module or np.array
        The target frame as a neural network or a discrete image.

    frame_dims: list[int, int]
        The frame dimensions. If `frame{0,1}` are neural networks, we will
        create a grid with frame_dims[0] rows and frame_dims[1] colunms to
        perform the inference. If they are images, we will resize them before
        opening performing the landmark detection or opening the UI.

    src_pts: np.array, optional
        Default is None

    tgt_pts: np.array, optional
        Default is None

    device: torch.device or str, optional
        Only used if `frame0` or `frame1` are encoded as neural networks,
        ignored otherwise. Default is "cuda:0".

    method: str, optional
        The landmark detection method to use. Allowed values are None
        (default), "" (same as None), "dlib", "mediapipe".

    open_ui: boolean, optional
        Opens the `FaceInteractor` UI to allow landmark edition.

    Returns
    -------
    landmark_src: np.array
        The normalized landmarks for `frame0`, in range [-1, 1]

    landmark_tgt: np.array
        The normalized landmarks for `frame1`, in range [-1, 1]
    """
    if method is None or not len(method) and not open_ui:
        print(
            f'"method" is "{method}" and open_ui is False. This is an'
            " expensive no-op. Bailing out.",
            file=sys.stderr,
        )
        return [], []

    dims = (frame_dims[0], frame_dims[1], 3)
    grid = get_grid(frame_dims, requires_grad=False).to(device)
    if isinstance(frame0, torch.nn.Module):
        frame0 = frame0.eval().to(device)
        with torch.no_grad():
            frame0 = frame0(grid)["model_out"].detach().clamp(0, 1) * 255
            frame0 = frame0.cpu().reshape(dims).numpy().astype(np.uint8)
    else:
        frame0 = cv2.resize(frame0, frame_dims)

    if isinstance(frame1, torch.nn.Module):
        frame1 = frame1.eval().to(device)
        with torch.no_grad():
            frame1 = frame1(grid)["model_out"].detach().clamp(0, 1) * 255
            frame1 = frame1.cpu().reshape(dims).numpy().astype(np.uint8)
    else:
        frame1 = cv2.resize(frame1, frame_dims)

    landmark_src, landmark_tgt = None, None
    if method is not None and len(method):
        if method == "dlib":
            landmark_src = get_landmarks_dlib(frame0).astype(float)
            landmark_src[:, 0] /= frame_dims[0]
            landmark_src[:, 1] /= frame_dims[1]

            landmark_tgt = get_landmarks_dlib(frame1).astype(float)
            landmark_tgt[:, 0] /= frame_dims[0]
            landmark_tgt[:, 1] /= frame_dims[1]
        elif method == "mediapipe":
            with mp_face_mesh.FaceMesh(
                static_image_mode=True, min_detection_confidence=0.75
            ) as fm:
                landmark_src = get_landmarks(fm, frame0)
                landmark_tgt = get_landmarks(fm, frame1)
        else:
            raise ValueError(f'Landmark method "{method}" not recognized.' " Aborting.")

        # Need to return to original coordinates
        landmark_src = landmark_src * 2 - 1
        landmark_tgt = landmark_tgt * 2 - 1

    p = FaceInteractor(frame0, frame1, src_pts=landmark_src, tgt_pts=landmark_tgt)
    plt.show()

    return p.landmarks


def test_mediapipe_landmark_detection(img):
    im_landmarks = imr.copy()
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, min_detection_confidence=0.75
    )
    landmarks = get_landmarks(face_mesh, img)
    landmarks[:, 0] *= img.shape[1]
    landmarks[:, 1] *= img.shape[0]
    landmarks = landmarks.astype(int)

    for p in landmarks:
        im_landmarks[p[0] - 1 : p[0] + 1, p[1] - 1 : p[1] + 1] = (0, 255, 0)
    return im_landmarks


def test_dlib_landmark_detection(img):
    im_landmarks = img.copy()
    landmarks = get_landmarks_dlib(img)
    for p in landmarks:
        im_landmarks[p[1] - 1 : p[1] + 1, p[0] - 1 : p[0] + 1] = (0, 255, 0)
    return im_landmarks


def get_default_argumentparser() -> argparse.ArgumentParser:
    """Builds and returns an argument parser with sane arguments.

    Returns
    -------
    parser: argparse.ArgumentParser
        A parser with sane mandatory and optional parameters for all experiments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", nargs="+", help="Path to experiment configuration files."
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
    return parser


def get_device(desired_dev, backup_dev: str = "cpu"):
    devstr = ""
    if desired_dev:
        devstr = desired_dev
    else:
        devstr = backup_dev

    if "cuda" in devstr and not torch.cuda.is_available():
        devstr = "cpu"
        print("No CUDA available devices found on system. Using CPU.", file=sys.stderr)
    else:
        torch.cuda.empty_cache()

    return torch.device(devstr)


if __name__ == "__main__":
    imnames = [f"{i}_03.jpg".zfill(10) for i in range(1, 10)]
    for i, imname in enumerate(imnames):
        imgpath = osp.join("data", "frll", imname)
        print(imgpath)
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imr = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)

        imdlib = test_dlib_landmark_detection(imr)
        cv2.imwrite(f"{i}_landmarks_dlib.png", cv2.cvtColor(imdlib, cv2.COLOR_RGB2BGR))

        immp = test_mediapipe_landmark_detection(imr)
        cv2.imwrite(f"{i}_landmarks_mp.png", cv2.cvtColor(immp, cv2.COLOR_RGB2BGR))
