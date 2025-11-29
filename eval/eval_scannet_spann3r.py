"""
Spann3R ScanNet Evaluation

Image size requirements:
- Height must be a multiple of 16
- Width must be a multiple of 16

This script centrally crops input images to the largest size that satisfies
the 16-pixel multiple constraint, then applies DUSt3R-compatible normalization.

Set RUN_SIZE_TESTS=1 to run size handling unit tests before evaluation.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import time
import os
import sys
import re
from PIL import Image
import torchvision.transforms as tvf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.utils.eval_utils import (
    load_poses,
    get_sorted_image_paths,
    get_all_scenes,
    build_frame_selection,
    evaluate_scene_and_save,
    compute_average_metrics_and_save,
)

from spann3r.model import Spann3R


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path, default="/home/jovyan/shared/xinzeli/scannetv2/process_scannet"
    )
    parser.add_argument(
        "--gt_ply_dir",
        type=Path,
        default="/home/jovyan/shared/xinzeli/scannetv2/scannet",
    )
    parser.add_argument("--output_path", type=Path, default="./eval_results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=3.0,
        help="Depth confidence threshold for filtering low confidence depth values",
    )
    parser.add_argument(
        "--chamfer_max_dist",
        type=float,
        default=0.5,
        help="Maximum distance threshold in Chamfer Distance computation, distances exceeding this value will be clipped",
    )
    parser.add_argument(
        "--input_frame",
        type=int,
        default=200,
        help="Maximum number of frames selected for processing per scene",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=50,
        help="Maximum number of scenes to evaluate",
    )
    parser.add_argument(
        "--dust3r_ckpt",
        type=str,
        default="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        help="path to DUSt3R backbone checkpoint",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/spann3r.pth",
        help="path to Spann3R checkpoint",
    )
    parser.add_argument(
        "--vis_attn_map",
        action="store_true",
        help="Whether to visualize attention maps during inference",
    )
    args = parser.parse_args()
    torch.manual_seed(33)
    np.random.seed(0)

    scannet_scenes = get_all_scenes(args.data_dir, args.num_scenes)
    print(f"Evaluate {len(scannet_scenes)} scenes")

    all_scenes_metrics = {"scenes": {}, "average": {}}
    from collections import defaultdict
    scene_infer_times = defaultdict(list)
    dtype = torch.bfloat16
    device = torch.device(args.device)

    dust3r_ckpt = os.path.abspath(os.path.expanduser(args.dust3r_ckpt)) if args.dust3r_ckpt else None
    spann3r_ckpt = os.path.abspath(os.path.expanduser(args.ckpt_path)) if args.ckpt_path else None

    if not (dust3r_ckpt and os.path.isfile(dust3r_ckpt)):
        raise FileNotFoundError(
            f"DUSt3R checkpoint not found: '{args.dust3r_ckpt}'. Provide a valid path via --dust3r_ckpt."
        )

    model = Spann3R(dus3r_name=dust3r_ckpt, use_feat=False).to(device)
    if spann3r_ckpt and os.path.isfile(spann3r_ckpt):
        try:
            obj = torch.load(spann3r_ckpt, map_location=device)
            sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
            try:
                model.load_state_dict(sd)
            except Exception:
                model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"Failed to load Spann3R checkpoint: {e}. Proceeding with backbone only.")
    model.eval()

    img_transform = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    for scene in scannet_scenes:
        scene_dir = args.data_dir / f"{scene}"
        output_scene_dir = args.output_path / f"input_frame_{args.input_frame}" / scene
        if (output_scene_dir / "metrics.json").exists():
            continue

        images_dir = scene_dir / "color"
        pose_path = scene_dir / "pose"
        image_paths = get_sorted_image_paths(images_dir)
        poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
        if (
            poses_gt is None
            or first_gt_pose is None
            or available_pose_frame_ids is None
        ):
            print(f"Skipping scene {scene}: no pose data")
            continue

        selected_frame_ids, selected_image_paths, selected_pose_indices = (
            build_frame_selection(
                image_paths, available_pose_frame_ids, args.input_frame
            )
        )

        c2ws = poses_gt[selected_pose_indices]
        image_paths = selected_image_paths
        frame_ids = selected_frame_ids

        if len(image_paths) == 0:
            print(f"No images found in {images_dir}")
            continue

        print("🚩Processing", scene, f"Found {len(image_paths)} images")
        all_cam_to_world_mat = []
        all_world_points = []

        try:
            def _crop_pil_to_16(img):
                w, h = img.size
                new_w = (w // 16) * 16
                new_h = (h // 16) * 16
                if new_w <= 0 or new_h <= 0:
                    return img
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                return img.crop((left, top, left + new_w, top + new_h))

            def _assert_tensor_hw16(t):
                if isinstance(t, torch.Tensor) and t.ndim == 3:
                    _, h, w = t.shape
                    assert h % 16 == 0 and w % 16 == 0, f"tensor size {h}x{w} not multiples of 16"

            views = []
            for p in image_paths:
                img = Image.open(p).convert("RGB")
                img = _crop_pil_to_16(img)
                img_t = img_transform(img)
                _assert_tensor_hw16(img_t)
                views.append({"img": img_t})

            from torch.utils.data._utils.collate import default_collate
            batch = default_collate([views])

            for view in batch:
                if isinstance(view.get("img"), torch.Tensor):
                    view["img"] = view["img"].to(device, non_blocking=True)

            should_sync = device.type == "cuda" and torch.cuda.is_available()
            if should_sync:
                torch.cuda.synchronize()
            t0 = time.time()
            preds, preds_all = model.forward(batch)
            if should_sync:
                torch.cuda.synchronize()
            t1 = time.time()
            elapsed_s = t1 - t0
            frame_count = len(batch)
            fps = frame_count / elapsed_s if elapsed_s > 0 else float("inf")
            print(f"Inference FPS (frames/s): {fps:.2f} [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            scene_infer_times[scene].append(float(fps))
            inference_time_ms = float(elapsed_s * 1000.0)

            all_cam_to_world_mat = [c2ws[i] for i in range(len(c2ws))]
            for j in range(len(preds)):
                pred = preds[j]
                pts = None
                if isinstance(pred, dict):
                    if "pts3d" in pred:
                        pts = pred["pts3d"]
                    elif "pts3d_in_other_view" in pred:
                        pts = pred["pts3d_in_other_view"]
                if pts is None:
                    continue
                pts_np = pts.detach().to(torch.float32).cpu().numpy()[0]
                pts_np = pts_np.reshape(-1, 3)
                c2w = c2ws[j if j < len(c2ws) else -1]
                R = c2w[:3, :3]
                t = c2w[:3, 3]
                pts_world = (R @ pts_np.T).T + t
                all_world_points.append(pts_world)

            if len(all_world_points) == 0:
                print(f"Skipping {scene}: no valid points predicted")
                continue

            merged_points = np.vstack(all_world_points)
            if merged_points.shape[0] > 999999:
                sample_indices = np.random.choice(
                    merged_points.shape[0], 999999, replace=False
                )
                merged_points = merged_points[sample_indices]
            all_world_points = [merged_points]

            if not all_cam_to_world_mat or not all_world_points:
                print(f"Skipping {scene}: failed to obtain valid camera poses or point clouds")
                continue

            metrics = evaluate_scene_and_save(
                scene,
                c2ws,
                first_gt_pose,
                frame_ids,
                all_cam_to_world_mat,
                all_world_points,
                output_scene_dir,
                args.gt_ply_dir,
                args.chamfer_max_dist,
                inference_time_ms,
                args.plot,
            )
            if metrics is not None:
                scene_metrics = {
                    key: float(value)
                    for key, value in metrics.items()
                    if key
                    in [
                        "chamfer_distance",
                        "ate",
                        "are",
                        "rpe_rot",
                        "rpe_trans",
                        "inference_time_ms",
                    ]
                }
                scene_metrics["fps"] = float(fps)
                all_scenes_metrics["scenes"][scene] = scene_metrics
                print("Complete metrics", all_scenes_metrics["scenes"][scene])

        except Exception as e:
            print(f"Error processing scene {scene}: {e}")
            import traceback
            traceback.print_exc()

    for sid, times in scene_infer_times.items():
        if len(times) > 0:
            avg_fps = np.mean(times)
            print(f"Idx: {sid}, FPS_avg: {avg_fps:.2f}")
    compute_average_metrics_and_save(
        all_scenes_metrics,
        args.output_path,
        args.input_frame,
    )

    if os.environ.get("RUN_SIZE_TESTS") == "1":
        def _run_size_tests():
            img = Image.new("RGB", (968, 1001), (128, 128, 128))
            cropped = (lambda im: im.crop(((im.size[0] - (im.size[0]//16)*16)//2,
                                           (im.size[1] - (im.size[1]//16)*16)//2,
                                           (im.size[0] + (im.size[0]//16)*16)//2,
                                           (im.size[1] + (im.size[1]//16)*16)//2)))(img)
            assert cropped.size[0] % 16 == 0 and cropped.size[1] % 16 == 0
            t = img_transform(cropped)
            assert t.shape[1] % 16 == 0 and t.shape[2] % 16 == 0
            print("Size tests passed for Spann3R preprocessing")
        _run_size_tests()
