import argparse
from pathlib import Path
import numpy as np
import torch
import time
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

FAST3R_PROJECT_DIR = os.path.abspath(os.path.join(ROOT_DIR, "fast3r"))
FAST3R_PKG_DIR = os.path.abspath(os.path.join(ROOT_DIR, "fast3r", "fast3r"))
for p in (FAST3R_PROJECT_DIR, FAST3R_PKG_DIR):
    if p not in sys.path:
        sys.path.append(p)

from fast3r.dust3r.utils.image import load_images as fast3r_load_images
from fast3r.dust3r.inference_multiview import inference as fast3r_inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from vggt.utils.eval_utils import (
    load_poses,
    get_sorted_image_paths,
    get_all_scenes,
    build_frame_selection,
    evaluate_scene_and_save,
    compute_average_metrics_and_save,
)


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
    parser.add_argument("--fast3r_ckpt", type=str, default="jedyang97/Fast3R_ViT_Large_512")
    parser.add_argument(
        "--vis_attn_map",
        action="store_true",
        help="Whether to visualize attention maps during inference",
    )
    args = parser.parse_args()
    torch.manual_seed(33)
    np.random.seed(0)

    # Scene sampling
    scannet_scenes = get_all_scenes(args.data_dir, args.num_scenes)
    print(f"Evaluate {len(scannet_scenes)} scenes")

    all_scenes_metrics = {"scenes": {}, "average": {}}
    from collections import defaultdict
    scene_infer_times = defaultdict(list)
    dtype = torch.bfloat16
    device = torch.device(args.device)
    model = Fast3R.from_pretrained(args.fast3r_ckpt)
    model = model.to(device).to(torch.bfloat16).eval()
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    lit_module.eval()

    # Process each scene
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

        # Frame filtering
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
            fast3r_views = fast3r_load_images([str(p) for p in image_paths], size=512, verbose=False)
            output_dict, profiling_info = fast3r_inference(
                fast3r_views,
                model,
                device,
                dtype=dtype,
                verbose=False,
                profiling=True,
            )
            poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
                output_dict['preds'],
                niter_PnP=100,
                focal_length_estimation_method='first_view_from_global_head'
            )
            camera_poses = poses_c2w_batch[0]
            all_cam_to_world_mat = [camera_poses[i] for i in range(len(camera_poses))]
            all_world_points = []
            for view_idx, pred in enumerate(output_dict['preds']):
                pts = pred['pts3d_in_other_view']
                pts_np = pts.to(torch.float32).cpu().numpy()[0]
                pts_np = pts_np.reshape(-1, 3)
                c2w = camera_poses[view_idx]
                R = c2w[:3, :3]
                t = c2w[:3, 3]
                pts_world = (R @ pts_np.T).T + t
                all_world_points.append(pts_world)
            total_time = profiling_info.get('total_time', None) if isinstance(profiling_info, dict) else None
            if total_time is None:
                total_time = 0.0
            inference_time_ms = float(total_time * 1000.0)
            frame_count = len(fast3r_views)
            fps = frame_count / total_time if total_time > 0 else float("inf")
            print(f"Inference FPS (frames/s): {fps:.2f} [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            scene_infer_times[scene].append(float(fps))

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

            # Evaluate and save
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
                all_scenes_metrics["scenes"][scene] = {
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
                all_scenes_metrics["scenes"][scene]["fps"] = float(fps)
                print("Complete metrics", all_scenes_metrics["scenes"][scene])

        except Exception as e:
            print(f"Error processing scene {scene}: {e}")
            import traceback

            traceback.print_exc()

    for sid, times in scene_infer_times.items():
        if len(times) > 0:
            avg_fps = np.mean(times)
            print(f"Idx: {sid}, FPS_avg: {avg_fps:.2f}")
    # Summarize average metrics and save
    compute_average_metrics_and_save(
        all_scenes_metrics,
        args.output_path,
        args.input_frame,
    )
