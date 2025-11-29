import os
import sys

# Ensure project root is on sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import re
from pathlib import Path
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
import cv2
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from collections import defaultdict
import hashlib

from dust3r.losses import L21
from dust3r.utils.geometry import geotrf
from spann3r.datasets import SevenScenes, NRGBD
from spann3r.loss import Regr3D_t_ScaleShiftInv
from spann3r.model import Spann3R
from spann3r.tools.eval_recon import accuracy, completion

from vggt.utils.eval_utils import build_frame_selection, load_images_rgb

# Suppress OpenCV imread warnings
try:
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


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
        default="./outputs/eval_7andN_span3r/",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--kf", type=int, default=2, help="key frame")
    parser.add_argument("--input_frame", type=int, default=200, help="max frames per scene")
    parser.add_argument(
        "--dust3r_ckpt",
        type=str,
        default="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        help="path to DUSt3R backbone checkpoint",
    )
    return parser


def _safe_torch_load(path, map_location):
    try:
        obj = torch.load(path, map_location=map_location)
        return obj, None
    except Exception as e:
        return None, e

def _file_info(path):
    try:
        size = os.path.getsize(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return {"size": size, "sha256": h.hexdigest()}
    except Exception:
        return None

def _state_dict_from_checkpoint(obj):
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj if isinstance(obj, dict) else None

def _detect_ckpt_kind(state_dict):
    if not isinstance(state_dict, dict):
        return "unknown"
    keys = list(state_dict.keys())
    if any(k.startswith("dust3r.") for k in keys) or any(k.startswith("attn_head_1") or k.startswith("value_encoder") for k in keys):
        return "spann3r"
    dust_keys = [
        "mask_token",
        "patch_embed.proj.weight",
        "enc_blocks.0.norm1.weight",
        "dec_blocks.0.norm1.weight",
        "head1.fc.0.weight",
    ]
    if any(k in keys for k in dust_keys) or any(k.startswith("enc_blocks.") or k.startswith("dec_blocks.") or k.startswith("patch_embed.") for k in keys):
        return "dust3r"
    return "unknown"

def _print_version_warnings():
    try:
        ver = torch.__version__
        print(f"PyTorch version: {ver}")
    except Exception:
        pass

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
    
    """
    datasets_all = {
        "7scenes": SevenScenes(
            split="test",
            ROOT="/home/jovyan/shared/xinzeli/fastplus/7-scenes",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf,
        ),  # 20),
    """
    datasets_all = {
        "7scenes": SevenScenes(
            split="test",
            ROOT="/home/jovyan/shared/xinzeli/fastplus/7-scenes",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf,
        ),  # 20),
        "NRGBD": NRGBD(
            split="test",
            ROOT="/home/jovyan/shared/xinzeli/fastplus/nrgbd",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf,
        ),
    }

    device = args.device

    # Resolve and validate checkpoints
    dust3r_ckpt = osp.abspath(osp.expanduser(args.dust3r_ckpt)) if args.dust3r_ckpt else None
    spann3r_ckpt = osp.abspath(osp.expanduser(args.ckpt_path)) if args.ckpt_path else None

    # If user accidentally passed DUSt3R to --ckpt_path, try to detect and warn
    if (dust3r_ckpt is None or not osp.isfile(dust3r_ckpt)) and (
        spann3r_ckpt and osp.isfile(spann3r_ckpt) and "DUSt3R" in osp.basename(spann3r_ckpt)
    ):
        print(
            f"Detected DUSt3R checkpoint at --ckpt_path: '{spann3r_ckpt}'. Using it as --dust3r_ckpt.",
            flush=True,
        )
        dust3r_ckpt = spann3r_ckpt

    if not (dust3r_ckpt and osp.isfile(dust3r_ckpt)):
        raise FileNotFoundError(
            f"DUSt3R checkpoint not found: '{args.dust3r_ckpt}'. Provide a valid path via --dust3r_ckpt."
        )

    skip_spann3r_load = False
    if not (spann3r_ckpt and osp.isfile(spann3r_ckpt)):
        print(
            f"Spann3R checkpoint not found: '{args.ckpt_path}'. Proceeding without loading Spann3R-specific weights.",
            flush=True,
        )
        skip_spann3r_load = True

    _print_version_warnings()
    info_dust = _file_info(dust3r_ckpt)
    if info_dust:
        print(f"DUSt3R ckpt size: {info_dust['size']} bytes, sha256: {info_dust['sha256']}")
    model = Spann3R(dus3r_name=dust3r_ckpt, use_feat=False).to(device)
    if not skip_spann3r_load:
        obj, err = _safe_torch_load(spann3r_ckpt, map_location=device)
        if err is not None:
            print(f"Failed to load Spann3R checkpoint: {err}. Skipping load.")
            skip_spann3r_load = True
        else:
            info_span = _file_info(spann3r_ckpt)
            if info_span:
                print(f"Spann3R ckpt size: {info_span['size']} bytes, sha256: {info_span['sha256']}")
            sd = _state_dict_from_checkpoint(obj)
            kind = _detect_ckpt_kind(sd)
            if kind == "dust3r":
                print("Checkpoint appears to be DUSt3R-only. Backbone already loaded. Skipping Spann3R load.")
                skip_spann3r_load = True
            elif kind == "spann3r":
                try:
                    model.load_state_dict(sd)
                except Exception as e:
                    print(f"Strict load failed: {e}. Retrying with non-strict.")
                    missing, unexpected = model.load_state_dict(sd, strict=False)
                    if missing:
                        print(f"Missing keys: {len(missing)}")
                    if unexpected:
                        print(f"Unexpected keys: {len(unexpected)}")
            else:
                print("Unknown checkpoint format. Skipping Spann3R load.")
                skip_spann3r_load = True
    model.eval()
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

                labels = []
                for v in batch:
                    try:
                        labels.append(v["label"][0])
                    except Exception:
                        labels.append("")

                frame_ids = []
                for i, lbl in enumerate(labels):
                    stem = Path(lbl).stem
                    nums = re.findall(r"\d+", stem)
                    fid = int(nums[0]) if len(nums) > 0 else i
                    frame_ids.append(fid)

                synthetic_paths = [Path(f"/__synthetic__/{fid}.png") for fid in frame_ids]
                available_pose_frame_ids = np.array(frame_ids)
                sel_frame_ids, _, sel_indices = build_frame_selection(
                    synthetic_paths,
                    available_pose_frame_ids,
                    args.input_frame,
                )
                print(
                    f"[FrameSelection] total_frames={len(available_pose_frame_ids)} selected_ids={sel_frame_ids}"
                )
                if len(sel_indices) > 0:
                    batch = [batch[i] for i in sel_indices]
                    sel_image_paths = [Path(labels[i]) for i in sel_indices]
                    try:
                        _ = load_images_rgb(sel_image_paths)
                    except Exception:
                        pass
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
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    start_time = time.time()
                preds, preds_all = model.forward(batch)
                if should_sync:
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start_event.elapsed_time(end_event)
                    elapsed_s = elapsed_ms / 1000.0
                else:
                    elapsed_s = time.time() - start_time
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
