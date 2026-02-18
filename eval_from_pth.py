import argparse
import json
import os
from typing import Optional

import numpy as np
import trimesh
import torch
from scipy.spatial import cKDTree


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_predictions(path: str) -> dict:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)}")
    required = ["world_points", "world_points_conf"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in prediction pth: {missing}")
    return {k: _to_numpy(v) for k, v in data.items()}


def _parse_frame_range(frame_range: Optional[str], total_frames: int) -> tuple[int, int]:
    if frame_range is None:
        return 0, total_frames - 1
    left, right = frame_range.split("-", 1)
    start = max(0, int(left.strip()))
    end = min(total_frames - 1, int(right.strip()))
    if start > end:
        raise ValueError(f"Invalid frame range: {frame_range}")
    return start, end


def _select_indices(total_frames: int, frame_stride: int, max_frames: int, frame_range: Optional[str]) -> np.ndarray:
    if frame_stride <= 0:
        raise ValueError("--frame_stride must be > 0")
    if max_frames <= 0:
        raise ValueError("--max_frames must be > 0")
    start, end = _parse_frame_range(frame_range, total_frames)
    indices = np.arange(start, end + 1, frame_stride, dtype=np.int64)
    if len(indices) > max_frames:
        sample_pos = np.linspace(0, len(indices) - 1, num=max_frames, dtype=np.int64)
        indices = indices[sample_pos]
    return np.unique(indices)


def _build_pred_pointcloud(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    selected_indices: np.ndarray,
    conf_threshold: float,
    max_points: int,
    rng_seed: int,
) -> np.ndarray:
    if world_points_conf.ndim == 4 and world_points_conf.shape[-1] == 1:
        world_points_conf = world_points_conf.squeeze(-1)

    pts = world_points[selected_indices]
    conf = world_points_conf[selected_indices]

    flat_pts = pts.reshape(-1, 3)
    flat_conf = conf.reshape(-1)

    valid = np.isfinite(flat_pts).all(axis=1) & np.isfinite(flat_conf) & (flat_conf >= conf_threshold)
    pred = flat_pts[valid]

    if len(pred) == 0:
        raise ValueError("No valid predicted points after confidence filtering.")

    if len(pred) > max_points:
        rng = np.random.default_rng(rng_seed)
        choose = rng.choice(len(pred), size=max_points, replace=False)
        pred = pred[choose]

    return pred.astype(np.float32)


def _load_gt_pcd(path: str, max_points: int, rng_seed: int) -> np.ndarray:
    mesh_or_pc = trimesh.load(path, process=False)
    if hasattr(mesh_or_pc, "vertices"):
        pts = np.asarray(mesh_or_pc.vertices, dtype=np.float32)
    elif isinstance(mesh_or_pc, trimesh.Scene):
        pts_list = []
        for geom in mesh_or_pc.geometry.values():
            if hasattr(geom, "vertices"):
                pts_list.append(np.asarray(geom.vertices, dtype=np.float32))
        if not pts_list:
            raise ValueError(f"Loaded scene without vertices from {path}")
        pts = np.concatenate(pts_list, axis=0)
    else:
        raise ValueError(f"Unsupported GT geometry type: {type(mesh_or_pc)}")
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if len(pts) == 0:
        raise ValueError("No valid points in GT PCD")
    if len(pts) > max_points:
        rng = np.random.default_rng(rng_seed + 1)
        choose = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[choose]
    return pts


def _compute_pointcloud_metrics(pred_pts: np.ndarray, gt_pts: np.ndarray, threshold: float) -> dict:
    pred_tree = cKDTree(pred_pts)
    gt_tree = cKDTree(gt_pts)

    d_pred_to_gt, _ = gt_tree.query(pred_pts, k=1, workers=-1)
    d_gt_to_pred, _ = pred_tree.query(gt_pts, k=1, workers=-1)

    chamfer_l1 = float(np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred))
    chamfer_l2 = float(np.mean(d_pred_to_gt**2) + np.mean(d_gt_to_pred**2))

    precision = float(np.mean(d_pred_to_gt <= threshold))
    recall = float(np.mean(d_gt_to_pred <= threshold))
    fscore = float(2 * precision * recall / (precision + recall + 1e-8))

    return {
        "chamfer_l1": chamfer_l1,
        "chamfer_l2": chamfer_l2,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "threshold": float(threshold),
        "pred_points": int(len(pred_pts)),
        "gt_points": int(len(gt_pts)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted point cloud from inference .pth against GT .pcd")
    parser.add_argument("--pred_pth", type=str, required=True, help="Path to run_inference output .pth")
    parser.add_argument("--gt_pcd", type=str, required=True, help="Path to GT point cloud (.pcd/.ply)")
    parser.add_argument("--output_dir", type=str, default="./pth_eval_results", help="Where to save metrics/artifacts")

    parser.add_argument("--frame_stride", type=int, default=1, help="Take every N-th frame before evaluation")
    parser.add_argument("--max_frames", type=int, default=1000000, help="Maximum frames used for evaluation")
    parser.add_argument("--frame_range", type=str, default=None, help="Optional frame range, e.g. '100-200'")

    parser.add_argument("--conf_threshold", type=float, default=0.0, help="Confidence threshold for predicted points")
    parser.add_argument("--distance_threshold", type=float, default=0.05, help="Distance threshold used by precision/recall/fscore")
    parser.add_argument("--max_points", type=int, default=2000000, help="Max points per cloud for metric computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for downsampling")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    preds = _load_predictions(args.pred_pth)
    world_points = preds["world_points"]
    world_points_conf = preds["world_points_conf"]

    selected_indices = _select_indices(
        total_frames=len(world_points),
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        frame_range=args.frame_range,
    )

    pred_pts = _build_pred_pointcloud(
        world_points=world_points,
        world_points_conf=world_points_conf,
        selected_indices=selected_indices,
        conf_threshold=args.conf_threshold,
        max_points=args.max_points,
        rng_seed=args.seed,
    )
    gt_pts = _load_gt_pcd(args.gt_pcd, max_points=args.max_points, rng_seed=args.seed)

    metrics = _compute_pointcloud_metrics(pred_pts, gt_pts, threshold=args.distance_threshold)
    metrics["pred_pth"] = args.pred_pth
    metrics["gt_pcd"] = args.gt_pcd
    metrics["num_frames_used"] = int(len(selected_indices))
    metrics["frame_indices"] = selected_indices.tolist()

    out_json = os.path.join(args.output_dir, "pointcloud_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_pcd_path = os.path.join(args.output_dir, "pred_points_sampled.ply")
    gt_pcd_path = os.path.join(args.output_dir, "gt_points_sampled.ply")
    trimesh.points.PointCloud(pred_pts.astype(np.float64)).export(pred_pcd_path)
    trimesh.points.PointCloud(gt_pts.astype(np.float64)).export(gt_pcd_path)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
