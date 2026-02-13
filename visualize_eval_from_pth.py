import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.append("src/")

from eval.pose_evaluation.evo_utils import eval_metrics, load_traj


SINTEL_TAG_FLOAT = 202021.25


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _c2w_to_tumpose(c2w: np.ndarray) -> np.ndarray:
    xyz = c2w[:3, 3]
    qx, qy, qz, qw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
    return np.array([xyz[0], xyz[1], xyz[2], qw, qx, qy, qz], dtype=np.float64)


def _get_tum_poses(poses: np.ndarray):
    tt = np.arange(len(poses), dtype=np.float64)
    tum_poses = np.stack([_c2w_to_tumpose(p) for p in poses], axis=0)
    return [tum_poses, tt]


def _depth_evaluation_local(predicted_depth_original: np.ndarray, ground_truth_depth_original: np.ndarray, align: str):
    pr = predicted_depth_original.astype(np.float64).reshape(-1)
    gt = ground_truth_depth_original.astype(np.float64).reshape(-1)

    mask = np.isfinite(pr) & np.isfinite(gt) & (gt > 0)
    pr = pr[mask]
    gt = gt[mask]
    if pr.size == 0:
        return {
            "Abs Rel": 0.0, "Sq Rel": 0.0, "RMSE": 0.0, "Log RMSE": 0.0,
            "δ < 1.": 0.0, "δ < 1.25": 0.0, "δ < 1.25^2": 0.0, "δ < 1.25^3": 0.0,
            "valid_pixels": 0,
        }

    if align == "metric":
        pr_aligned = pr
    elif align == "scale":
        s = np.median(gt) / (np.median(pr) + 1e-8)
        pr_aligned = pr * s
    elif align == "scale&shift":
        A = np.stack([pr, np.ones_like(pr)], axis=1)
        x, *_ = np.linalg.lstsq(A, gt, rcond=None)
        pr_aligned = x[0] * pr + x[1]
    else:
        raise ValueError(f"Unknown align mode: {align}")

    pr_aligned = np.clip(pr_aligned, 1e-5, None)
    gt_clip = np.clip(gt, 1e-5, None)

    abs_rel = float(np.mean(np.abs(pr_aligned - gt_clip) / gt_clip))
    sq_rel = float(np.mean(((pr_aligned - gt_clip) ** 2) / gt_clip))
    rmse = float(np.sqrt(np.mean((pr_aligned - gt_clip) ** 2)))
    log_rmse = float(np.sqrt(np.mean((np.log(pr_aligned) - np.log(gt_clip)) ** 2)))

    ratio = np.maximum(pr_aligned / gt_clip, gt_clip / pr_aligned)
    th0 = float(np.mean((ratio < 1.0).astype(np.float64)))
    th1 = float(np.mean((ratio < 1.25).astype(np.float64)))
    th2 = float(np.mean((ratio < 1.25 ** 2).astype(np.float64)))
    th3 = float(np.mean((ratio < 1.25 ** 3).astype(np.float64)))

    return {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "δ < 1.": th0,
        "δ < 1.25": th1,
        "δ < 1.25^2": th2,
        "δ < 1.25^3": th3,
        "valid_pixels": int(pr.size),
    }


def _load_predictions(path: str) -> dict:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)}")
    required = ["world_points", "world_points_conf", "depth", "images", "extrinsic"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in prediction pth: {missing}")

    preds = {k: _to_numpy(v) for k, v in data.items()}
    return preds


def _normalize_images(images: np.ndarray) -> np.ndarray:
    # input from run_inference.py is (S,3,H,W)
    if images.ndim != 4:
        raise ValueError(f"Expected images ndim=4, got {images.shape}")

    if images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))

    images = images.astype(np.float32)
    if images.max() > 1.0:
        images = images / 255.0
    images = np.clip(images, 0.0, 1.0)
    return images


def _camera_from_extrinsic_intrinsic(extrinsic: np.ndarray, intrinsic: Optional[np.ndarray], h: int, w: int):
    # extrinsic is world->cam (S,3,4)
    s = extrinsic.shape[0]
    w2c = np.tile(np.eye(4, dtype=np.float32)[None], (s, 1, 1))
    w2c[:, :3, :] = extrinsic
    c2w = np.linalg.inv(w2c)

    r_list = c2w[:, :3, :3]
    t_list = c2w[:, :3, 3]

    if intrinsic is not None:
        focal = intrinsic[:, 0, 0]
        pp = intrinsic[:, :2, 2]
    else:
        focal = np.full((s,), w / 2.0, dtype=np.float32)
        pp = np.tile(np.array([w / 2.0, h / 2.0], dtype=np.float32)[None], (s, 1))

    return {"R": r_list, "t": t_list, "focal": focal, "pp": pp}, c2w


def _save_basic_artifacts(preds: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "prediction_arrays.npz"),
        world_points=preds["world_points"],
        world_points_conf=preds["world_points_conf"],
        depth=preds["depth"],
        extrinsic=preds["extrinsic"],
        intrinsic=preds.get("intrinsic", None),
    )


def _run_pose_metrics(c2w: np.ndarray, gt_pose_path: str, gt_pose_format: str, out_json: str):
    pred_traj = _get_tum_poses(c2w)
    gt_traj = load_traj(gt_pose_path, traj_format=gt_pose_format, num_frames=len(pred_traj[0]))

    txt_out = os.path.splitext(out_json)[0] + ".txt"
    ate, rpe_t, rpe_r = eval_metrics(pred_traj, gt_traj, seq="from_pth", filename=txt_out)
    metrics = {
        "ATE_RMSE": float(ate),
        "RPE_trans_RMSE": float(rpe_t),
        "RPE_rot_RMSE_deg": float(rpe_r),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _read_depth(path: str) -> np.ndarray:
    import imageio.v2 as iio

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".png":
        img = iio.imread(path)
        if img is None:
            raise ValueError(f"Failed to read depth png: {path}")
        if img.dtype == np.uint16:
            return img.astype(np.float32) / 256.0
        return img.astype(np.float32)
    if ext == ".dpt":
        with open(path, "rb") as f:
            tag = np.fromfile(f, dtype=np.float32, count=1)[0]
            if tag != SINTEL_TAG_FLOAT:
                raise ValueError(f"Bad Sintel tag in {path}: {tag}")
            width = np.fromfile(f, dtype=np.int32, count=1)[0]
            height = np.fromfile(f, dtype=np.int32, count=1)[0]
            return np.fromfile(f, dtype=np.float32).reshape((height, width))
    raise ValueError(f"Unsupported depth extension: {ext}")


def _run_depth_metrics(pred_depth: np.ndarray, gt_depth_glob: str, align: str, out_json: str):
    from PIL import Image

    gt_paths = sorted(glob.glob(gt_depth_glob))
    if not gt_paths:
        raise FileNotFoundError(f"No GT depth files for pattern: {gt_depth_glob}")

    n = min(len(pred_depth), len(gt_paths))
    per_frame = []
    for i in range(n):
        gt = _read_depth(gt_paths[i]).astype(np.float32)
        pr = pred_depth[i].astype(np.float32)
        if pr.ndim == 3:
            pr = pr.squeeze(-1)
        if pr.shape != gt.shape:
            pr = np.array(
                Image.fromarray(pr).resize((gt.shape[1], gt.shape[0]), resample=Image.BICUBIC),
                dtype=np.float32,
            )

        result = _depth_evaluation_local(pr, gt, align=align)
        per_frame.append(result)

    keys = [k for k in per_frame[0].keys() if k != "valid_pixels"]
    weights = np.array([r["valid_pixels"] for r in per_frame], dtype=np.float64)
    weights = np.maximum(weights, 1.0)

    avg = {}
    for k in keys:
        vals = np.array([r[k] for r in per_frame], dtype=np.float64)
        avg[k] = float(np.average(vals, weights=weights))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(avg, f, indent=2)
    return avg




def _parse_frame_range(frame_range: Optional[str], total_frames: int) -> tuple[int, int]:
    if total_frames <= 0:
        raise ValueError("No frames available in prediction file")

    if frame_range is None:
        return 0, total_frames - 1

    try:
        left, right = frame_range.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
    except Exception as exc:
        raise ValueError(
            f"Invalid --frame_range '{frame_range}', expected format like '100-200'"
        ) from exc

    if start > end:
        raise ValueError(f"Invalid --frame_range '{frame_range}': start > end")

    start = max(0, start)
    end = min(total_frames - 1, end)
    if start > end:
        raise ValueError(
            f"Frame range '{frame_range}' is outside available frames [0, {total_frames - 1}]"
        )
    return start, end


def _select_frame_indices(total_frames: int, frame_stride: int, max_frames: int, frame_range: Optional[str]) -> np.ndarray:
    if frame_stride <= 0:
        raise ValueError("--frame_stride must be > 0")
    if max_frames <= 0:
        raise ValueError("--max_frames must be > 0")

    start, end = _parse_frame_range(frame_range, total_frames)
    indices = np.arange(start, end + 1, frame_stride, dtype=np.int64)

    if len(indices) == 0:
        indices = np.array([start], dtype=np.int64)

    if len(indices) > max_frames:
        sample_pos = np.linspace(0, len(indices) - 1, num=max_frames, dtype=np.int64)
        indices = indices[sample_pos]

    return np.unique(indices)

def main():
    parser = argparse.ArgumentParser(description="Visualize and evaluate StreamVGGT predictions from output .pth")
    parser.add_argument("--pred_pth", type=str, required=True, help="Path to run_inference output .pth")
    parser.add_argument("--output_dir", type=str, default="./pth_post_results", help="Where to save metrics/artifacts")

    parser.add_argument("--run_viser", action="store_true", help="Launch interactive viser visualization")
    parser.add_argument("--port", type=int, default=9999, help="Viser port")
    parser.add_argument("--vis_threshold", type=float, default=1.5, help="Visibility threshold for point cloud")
    parser.add_argument("--frame_stride", type=int, default=1, help="Take every N-th frame before visualization sampling")
    parser.add_argument("--max_frames", type=int, default=50, help="Maximum number of frames to visualize")
    parser.add_argument("--frame_range", type=str, default=None, help="Optional frame range for visualization, e.g. '100-200'")

    parser.add_argument("--gt_pose_path", type=str, default=None, help="Optional GT trajectory file/folder")
    parser.add_argument(
        "--gt_pose_format",
        type=str,
        default="tum",
        choices=["tum", "replica", "sintel", "tartanair"],
        help="GT trajectory format",
    )

    parser.add_argument("--gt_depth_glob", type=str, default=None, help="Optional GT depth file glob")
    parser.add_argument(
        "--depth_align",
        type=str,
        default="scale",
        choices=["scale&shift", "scale", "metric"],
        help="Depth alignment mode",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    preds = _load_predictions(args.pred_pth)
    images = _normalize_images(preds["images"])
    pts = preds["world_points"]
    conf = preds["world_points_conf"]
    depth = preds["depth"]
    extrinsic = preds["extrinsic"]
    intrinsic = preds.get("intrinsic", None)

    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf.squeeze(-1)

    h, w = images.shape[1], images.shape[2]
    cam_dict, c2w = _camera_from_extrinsic_intrinsic(extrinsic, intrinsic, h, w)

    _save_basic_artifacts(preds, args.output_dir)

    summary = {
        "num_frames": int(images.shape[0]),
        "pred_pth": args.pred_pth,
    }

    if args.gt_pose_path:
        pose_json = os.path.join(args.output_dir, "pose_metrics.json")
        summary["pose_metrics"] = _run_pose_metrics(c2w, args.gt_pose_path, args.gt_pose_format, pose_json)

    if args.gt_depth_glob:
        depth_json = os.path.join(args.output_dir, "depth_metrics.json")
        summary["depth_metrics"] = _run_depth_metrics(depth, args.gt_depth_glob, args.depth_align, depth_json)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    if args.run_viser:
        from viser_utils import PointCloudViewer

        selected_indices = _select_frame_indices(
            total_frames=len(pts),
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            frame_range=args.frame_range,
        )

        pts_vis = [pts[i] for i in selected_indices]
        images_vis = [images[i] for i in selected_indices]
        conf_vis = [conf[i] for i in selected_indices]
        cam_vis = {
            "R": cam_dict["R"][selected_indices],
            "t": cam_dict["t"][selected_indices],
            "focal": cam_dict["focal"][selected_indices],
            "pp": cam_dict["pp"][selected_indices],
        }

        summary["visualization_frames"] = {
            "count": int(len(selected_indices)),
            "indices": selected_indices.tolist(),
            "frame_stride": int(args.frame_stride),
            "max_frames": int(args.max_frames),
            "frame_range": args.frame_range,
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[Viser] Visualizing {len(selected_indices)} frames from total {len(pts)}")

        viewer = PointCloudViewer(
            model=None,
            state_args=None,
            pc_list=pts_vis,
            color_list=images_vis,
            conf_list=conf_vis,
            cam_dict=cam_vis,
            gt_poses=None,
            device="cpu",
            vis_threshold=args.vis_threshold,
            size=max(h, w),
            port=args.port,
            edge_color_list=[None] * len(pts_vis),
            show_camera=True,
        )
        viewer.run()


if __name__ == "__main__":
    main()
