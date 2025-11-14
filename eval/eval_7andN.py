import os
import sys

# Ensure project root is on sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from collections import defaultdict

from dust3r.losses import L21
from dust3r.utils.geometry import geotrf
from spann3r.datasets import SevenScenes, NRGBD
from spann3r.loss import Regr3D_t_ScaleShiftInv
from spann3r.model import Spann3R
from spann3r.tools.eval_recon import accuracy, completion


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/spann3r.pth",
        help="path to Spann3R checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/outputs/eval_7andN/",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--kf", type=int, default=2, help="key frame")
    parser.add_argument(
        "--dust3r_ckpt",
        type=str,
        default="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        help="path to DUSt3R backbone checkpoint",
    )
    return parser


def main(args):
    # Ensure deterministic numpy sampling (e.g., point subsampling)
    np.random.seed(0)

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
    else:
        raise NotImplementedError
    datasets_all = {
        # "7scenes": SevenScenes(
        #     split="test",
        #     ROOT="/root/autodl-tmp/data/7-scenes",
        #     resolution=resolution,
        #     num_seq=1,
        #     full_video=True,
        #     kf_every=args.kf,
        # ),  # 20),
        "NRGBD": NRGBD(
            split="test",
            ROOT="/root/autodl-tmp/data/neural_rgbd_data/",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf,
        ),
    }

    device = args.device

    model = Spann3R(dus3r_name=args.dust3r_ckpt, use_feat=False).to(device)
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    del checkpoint
    os.makedirs(osp.join(args.output_dir, f"{args.kf}"), exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(osp.join(args.output_dir, f"{args.kf}"), name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, "logs.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0
            scene_infer_times = defaultdict(list)

            for data_idx in tqdm(range(len(dataset))):
                batch = default_collate([dataset[data_idx]])
                ignore_keys = set(
                    [
                        "depthmap",
                        "dataset",
                        "label",
                        "instance",
                        "idx",
                        "true_shape",
                        "rng",
                    ]
                )
                for view in batch:
                    for name in view.keys():  # pseudo_focal
                        if name in ignore_keys:
                            continue
                        if isinstance(view[name], tuple) or isinstance(
                            view[name], list
                        ):
                            view[name] = [
                                x.to(device, non_blocking=True) for x in view[name]
                            ]
                        else:
                            view[name] = view[name].to(device, non_blocking=True)

                pts_all = []
                pts_gt_all = []
                images_all = []
                masks_all = []
                conf_all = []
                in_camera1 = None

                should_sync = device.startswith("cuda") and torch.cuda.is_available()
                if should_sync:
                    torch.cuda.synchronize()
                start = time.time()
                preds, preds_all = model.forward(batch)
                if should_sync:
                    torch.cuda.synchronize()
                end = time.time()
                elapsed_s = end - start
                frame_count = len(batch)
                fps = frame_count / elapsed_s if elapsed_s > 0 else float("inf")
                print(f"Inference FPS (frames/s): {fps:.2f}")

                valid_length = len(preds) // args.revisit
                if args.revisit > 1 and valid_length > 0:
                    preds = preds[-valid_length:]
                    batch = batch[-valid_length:]
                    if preds_all:
                        keep_pairs = max(len(preds) - 1, 0)
                        preds_all = preds_all[-keep_pairs:] if keep_pairs > 0 else []

                # Evaluation
                print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                    criterion.get_all_pts3d_t(batch, preds_all)
                )

                gt_shift_value = None
                if monitoring and "gt_shift_z" in monitoring:
                    gt_shift = monitoring["gt_shift_z"]
                    if torch.is_tensor(gt_shift):
                        gt_shift_value = float(gt_shift.detach().cpu().mean())
                    else:
                        gt_shift_value = float(gt_shift)

                in_camera1 = None
                pts_all = []
                pts_gt_all = []
                images_all = []
                masks_all = []
                conf_all = []

                if isinstance(pred_pts, (list, tuple)) and len(pred_pts) == 2:
                    pred_pts_l, pred_pts_r = pred_pts
                else:
                    pred_pts_l, pred_pts_r = pred_pts, None

                for j, view in enumerate(batch):
                    if in_camera1 is None:
                        in_camera1 = view["camera_pose"][0].cpu()

                    image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                    mask = view["valid_mask"].cpu().numpy()[0]

                    if pred_pts_r is not None:
                        if j < len(pred_pts_l):
                            pts_tensor = pred_pts_l[j]
                        else:
                            pts_tensor = pred_pts_r[-1]
                    else:
                        pts_tensor = pred_pts_l[j]

                    pts = pts_tensor.detach().cpu().numpy()[0]
                    conf_tensor = preds[j]["conf"]
                    if torch.is_tensor(conf_tensor):
                        if conf_tensor.ndim == 3:
                            conf = conf_tensor[0].detach().cpu().numpy()
                        else:
                            conf = conf_tensor.detach().cpu().numpy()
                    else:
                        conf = np.asarray(conf_tensor)
                    pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                    if gt_shift_value is not None:
                        pts[..., 2] += gt_shift_value
                        pts_gt[..., 2] += gt_shift_value

                    if in_camera1 is not None:
                        pts_trf = geotrf(in_camera1, pts)
                        pts_gt_trf = geotrf(in_camera1, pts_gt)
                        if torch.is_tensor(pts_trf):
                            pts = pts_trf.detach().cpu().numpy()
                        else:
                            pts = np.asarray(pts_trf)
                        if torch.is_tensor(pts_gt_trf):
                            pts_gt = pts_gt_trf.detach().cpu().numpy()
                        else:
                            pts_gt = np.asarray(pts_gt_trf)

                    H, W = image.shape[:2]
                    cx = W // 2
                    cy = H // 2
                    l, t = cx - 112, cy - 112
                    r, b = cx + 112, cy + 112
                    image = image[t:b, l:r]
                    mask = mask[t:b, l:r]
                    pts = pts[t:b, l:r]
                    pts_gt = pts_gt[t:b, l:r]
                    conf = conf[t:b, l:r]

                    image = (image + 1.0) / 2.0

                    images_all.append(image[None, ...])
                    pts_all.append(pts[None, ...])
                    pts_gt_all.append(pts_gt[None, ...])
                    masks_all.append(mask[None, ...])
                    conf_all.append(conf[None, ...])

                images_all = np.concatenate(images_all, axis=0)
                pts_all = np.concatenate(pts_all, axis=0)
                pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                masks_all = np.concatenate(masks_all, axis=0)
                conf_all = np.concatenate(conf_all, axis=0)

                scene_id = view["label"][0].rsplit("/", 1)[0]
                # Record FPS per scene for averaging later
                try:
                    scene_infer_times[scene_id].append(float(fps))
                except Exception:
                    pass

                save_params = {}

                save_params["images_all"] = images_all
                save_params["pts_all"] = pts_all
                save_params["pts_gt_all"] = pts_gt_all
                save_params["masks_all"] = masks_all
                save_params["conf_all"] = conf_all

                pts_all_masked = pts_all[masks_all > 0]
                pts_gt_all_masked = pts_gt_all[masks_all > 0]
                images_all_masked = images_all[masks_all > 0]

                mask = np.isfinite(pts_all_masked)
                pts_all_masked = pts_all_masked[mask]

                mask_gt = np.isfinite(pts_gt_all_masked)
                pts_gt_all_masked = pts_gt_all_masked[mask_gt]
                images_all_masked = images_all_masked[mask]

                # Reshape to point cloud (N, 3) before sampling
                pts_all_masked = pts_all_masked.reshape(-1, 3)
                pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                images_all_masked = images_all_masked.reshape(-1, 3)

                # If number of points exceeds threshold, sample by points
                if pts_all_masked.shape[0] > 999999:
                    sample_indices = np.random.choice(
                        pts_all_masked.shape[0], 999999, replace=False
                    )
                    pts_all_masked = pts_all_masked[sample_indices]
                    images_all_masked = images_all_masked[sample_indices]

                # Apply the same sampling to GT point cloud
                if pts_gt_all_masked.shape[0] > 999999:
                    sample_indices_gt = np.random.choice(
                        pts_gt_all_masked.shape[0], 999999, replace=False
                    )
                    pts_gt_all_masked = pts_gt_all_masked[sample_indices_gt]

                if args.use_proj:

                    def umeyama_alignment(
                        src: np.ndarray, dst: np.ndarray, with_scale: bool = True
                    ):
                        assert src.shape == dst.shape
                        N, dim = src.shape

                        mu_src = src.mean(axis=0)
                        mu_dst = dst.mean(axis=0)
                        src_c = src - mu_src
                        dst_c = dst - mu_dst

                        Sigma = dst_c.T @ src_c / N  # (3,3)

                        U, D, Vt = np.linalg.svd(Sigma)

                        S = np.eye(dim)
                        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
                            S[-1, -1] = -1

                        R = U @ S @ Vt

                        if with_scale:
                            var_src = (src_c**2).sum() / N
                            s = (D * S.diagonal()).sum() / var_src
                        else:
                            s = 1.0

                        t = mu_dst - s * R @ mu_src

                        return s, R, t

                    pts_all_masked = pts_all_masked.reshape(-1, 3)
                    pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
                    s, R, t = umeyama_alignment(
                        pts_all_masked, pts_gt_all_masked, with_scale=True
                    )
                    pts_all_aligned = (s * (R @ pts_all_masked.T)).T + t  # (N,3)
                    pts_all_masked = pts_all_aligned

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_all_masked)
                pcd.colors = o3d.utility.Vector3dVector(images_all_masked)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked)
                pcd_gt.colors = o3d.utility.Vector3dVector(images_all_masked)

                trans_init = np.eye(4)

                threshold = 0.1
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd,
                    pcd_gt,
                    threshold,
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )

                transformation = reg_p2p.transformation

                pcd = pcd.transform(transformation)
                pcd.estimate_normals()
                pcd_gt.estimate_normals()

                gt_normal = np.asarray(pcd_gt.normals)
                pred_normal = np.asarray(pcd.normals)

                acc, acc_med, nc1, nc1_med = accuracy(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                comp, comp_med, nc2, nc2_med = completion(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                print(
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, FPS: {fps:.2f}"
                )
                print(
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}, FPS: {fps:.2f}",
                    file=open(log_file, "a"),
                )

                acc_all += acc
                comp_all += comp
                nc1_all += nc1
                nc2_all += nc2

                acc_all_med += acc_med
                comp_all_med += comp_med
                nc1_all_med += nc1_med
                nc2_all_med += nc2_med

                # release cuda memory
                torch.cuda.empty_cache()

            # Get depth from pcd and run TSDFusion
            to_write = ""
            # Read the log file
            if os.path.exists(osp.join(save_path, "logs.txt")):
                with open(osp.join(save_path, "logs.txt"), "r") as f_sub:
                    to_write += f_sub.read()

            with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                log_data = to_write
                metrics = defaultdict(list)
                for line in log_data.strip().split("\n"):
                    match = regex.match(line)
                    if match:
                        data = match.groupdict()
                        # Exclude 'scene_id' from metrics as it's an identifier
                        for key, value in data.items():
                            if key == "scene_id" or value is None:
                                continue
                            metrics[key].append(float(value))
                        metrics["nc"].append(
                            (float(data["nc1"]) + float(data["nc2"])) / 2
                        )
                        metrics["nc_med"].append(
                            (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                        )
                mean_metrics = {
                    metric: sum(values) / len(values)
                    for metric, values in metrics.items()
                }

                c_name = "mean"
                print_str = f"{c_name.ljust(20)}: "
                for m_name in mean_metrics:
                    print_num = np.mean(mean_metrics[m_name])
                    print_str = print_str + f"{m_name}: {print_num:.3f} | "
                print_str = print_str + "\n"
                # Summarize per-scene average FPS
                time_lines = []
                for sid, times in scene_infer_times.items():
                    if len(times) > 0:
                        avg_fps = np.mean(times)
                        time_lines.append(f"Idx: {sid}, FPS_avg: {avg_fps:.2f}")
                time_block = "\n".join(time_lines) + (
                    "\n" if len(time_lines) > 0 else ""
                )

                f.write(to_write + time_block + print_str)


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
    (?:,\s*FPS:\s*(?P<fps>[^,]+))?
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
