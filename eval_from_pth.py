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


def _pcd_numpy_dtype(type_char: str, size: int):
    type_char = type_char.upper()
    if type_char == "F":
        return {4: np.float32, 8: np.float64}[size]
    if type_char == "I":
        return {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[size]
    if type_char == "U":
        return {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[size]
    raise ValueError(f"Unsupported PCD TYPE/SIZE combination: TYPE={type_char}, SIZE={size}")




def _lzf_decompress_variant(data: bytes, expected_length: int, literal_add: int, backref_add: int, ref_minus_one: bool) -> bytes:
    i = 0
    out = bytearray()
    data_len = len(data)
    while i < data_len and len(out) < expected_length:
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            run = ctrl + literal_add
            if run < 0 or i + run > data_len:
                raise ValueError("Invalid LZF stream: literal run exceeds input size")
            out.extend(data[i:i + run])
            i += run
        else:
            length = (ctrl >> 5)
            ref_offset = (ctrl & 0x1F) << 8
            if i >= data_len:
                raise ValueError("Invalid LZF stream: missing back-reference offset byte")
            ref_offset += data[i]
            i += 1
            if length == 7:
                if i >= data_len:
                    raise ValueError("Invalid LZF stream: missing extended length byte")
                length += data[i]
                i += 1
            length += backref_add
            ref_pos = len(out) - ref_offset - (1 if ref_minus_one else 0)
            if ref_pos < 0:
                raise ValueError("Invalid LZF stream: back-reference before output start")
            for _ in range(length):
                if ref_pos >= len(out):
                    raise ValueError("Invalid LZF stream: back-reference out of range")
                out.append(out[ref_pos])
                ref_pos += 1
    if len(out) != expected_length:
        raise ValueError(f"LZF decompression size mismatch: got {len(out)}, expected {expected_length}")
    return bytes(out)


def _lzf_decompress(data: bytes, expected_length: int) -> bytes:
    variants = [
        (1, 2, True),   # standard LZF
        (1, 3, True),
        (0, 2, True),
        (1, 2, False),
    ]
    last_err = None
    for literal_add, backref_add, ref_minus_one in variants:
        try:
            return _lzf_decompress_variant(data, expected_length, literal_add, backref_add, ref_minus_one)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise ValueError(f"Failed to decompress LZF payload with supported variants: {last_err}")


def _extract_xyz_from_pcd_bytes(raw: bytes, fields, sizes, types, counts, points: int) -> np.ndarray:
    x_vals = y_vals = z_vals = None
    offset = 0
    for name, size, tchar, count in zip(fields, sizes, types, counts):
        elem_bytes = size * count
        field_bytes = elem_bytes * points
        chunk = raw[offset:offset + field_bytes]
        if len(chunk) != field_bytes:
            raise ValueError(f"Invalid PCD payload: insufficient bytes for field '{name}'")

        dtype = _pcd_numpy_dtype(tchar, size)
        arr = np.frombuffer(chunk, dtype=dtype)
        if count > 1:
            arr = arr.reshape(points, count)
            arr = arr[:, 0]
        if name == "x":
            x_vals = arr
        elif name == "y":
            y_vals = arr
        elif name == "z":
            z_vals = arr
        offset += field_bytes

    if x_vals is None or y_vals is None or z_vals is None:
        raise ValueError("PCD file missing x/y/z fields")

    pts = np.stack([x_vals, y_vals, z_vals], axis=1)
    return pts.astype(np.float32)

def _read_pcd_points(path: str) -> np.ndarray:
    header = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD file (missing DATA line): {path}")
            line_str = line.decode("utf-8", errors="ignore").strip()
            if not line_str or line_str.startswith("#"):
                continue
            parts = line_str.split()
            key = parts[0].lower()
            values = parts[1:]
            header[key] = values
            if key == "data":
                break

        fields = header.get("fields")
        sizes = [int(v) for v in header.get("size", [])]
        types = header.get("type", [])
        counts = [int(v) for v in header.get("count", ["1"] * len(fields))]
        points = int(header.get("points", ["0"])[0])
        data_type = header["data"][0].lower()

        if not fields or not sizes or not types:
            raise ValueError(f"Invalid PCD header in {path}: missing FIELDS/SIZE/TYPE")
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f"Invalid PCD header in {path}: inconsistent FIELDS/SIZE/TYPE/COUNT lengths")

        if data_type == "ascii":
            raw = np.loadtxt(f, dtype=np.float64)
            if raw.ndim == 1:
                raw = raw[None, :]
            offsets = []
            csum = 0
            for c in counts:
                offsets.append(csum)
                csum += c
            idx_x = offsets[fields.index("x")]
            idx_y = offsets[fields.index("y")]
            idx_z = offsets[fields.index("z")]
            pts = np.stack([raw[:, idx_x], raw[:, idx_y], raw[:, idx_z]], axis=1)
            return pts.astype(np.float32)

        if data_type == "binary":
            dtype_descr = []
            for name, size, tchar, count in zip(fields, sizes, types, counts):
                base_dtype = _pcd_numpy_dtype(tchar, size)
                if count == 1:
                    dtype_descr.append((name, base_dtype))
                else:
                    dtype_descr.append((name, base_dtype, (count,)))

            packed = np.fromfile(f, dtype=np.dtype(dtype_descr), count=points)
            if not all(k in packed.dtype.names for k in ("x", "y", "z")):
                raise ValueError(f"PCD file missing x/y/z fields: {path}")
            pts = np.stack([packed["x"], packed["y"], packed["z"]], axis=1)
            return pts.astype(np.float32)

        if data_type == "binary_compressed":
            sizes_u32 = np.fromfile(f, dtype=np.uint32, count=2)
            if len(sizes_u32) != 2:
                raise ValueError(f"Invalid PCD binary_compressed header in {path}")
            compressed_size, uncompressed_size = int(sizes_u32[0]), int(sizes_u32[1])
            compressed = f.read(compressed_size)
            if len(compressed) != compressed_size:
                raise ValueError(f"Invalid PCD binary_compressed payload size in {path}")
            try:
                raw = _lzf_decompress(compressed, uncompressed_size)
                return _extract_xyz_from_pcd_bytes(raw, fields, sizes, types, counts, points)
            except Exception as exc:  # noqa: BLE001
                try:
                    import open3d as o3d

                    pcd = o3d.io.read_point_cloud(path)
                    pts = np.asarray(pcd.points, dtype=np.float32)
                    if pts.size == 0:
                        raise ValueError("Open3D returned empty point cloud")
                    return pts
                except Exception as o3d_exc:  # noqa: BLE001
                    raise ValueError(
                        f"Failed to parse binary_compressed PCD via internal decoder ({exc}) and Open3D fallback ({o3d_exc})"
                    )

        raise ValueError(f"Unsupported PCD DATA mode '{data_type}' in {path}")


def _load_gt_pcd(path: str, max_points: int, rng_seed: int) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pcd":
        pts = _read_pcd_points(path)
    else:
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


def _umeyama_alignment(src_pts: np.ndarray, dst_pts: np.ndarray, estimate_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    cov = (dst_centered.T @ src_centered) / max(1, src_pts.shape[0])
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[-1, -1] = -1

    r = u @ s_mat @ vt

    if estimate_scale:
        var_src = np.mean(np.sum(src_centered**2, axis=1))
        if var_src <= 1e-12:
            scale = 1.0
        else:
            scale = float(np.trace(np.diag(d) @ s_mat) / var_src)
    else:
        scale = 1.0

    t = dst_mean - scale * (r @ src_mean)
    return scale, r, t


def _apply_similarity(pts: np.ndarray, scale: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (pts @ r.T)) + t[None, :]


def _align_pred_to_gt(pred_pts: np.ndarray, gt_pts: np.ndarray, align_mode: str, icp_iters: int) -> tuple[np.ndarray, dict]:
    if align_mode == "none":
        return pred_pts, {"align_mode": align_mode, "scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]}

    estimate_scale = align_mode == "sim3"
    aligned = pred_pts.astype(np.float64)
    gt_tree = cKDTree(gt_pts)

    total_scale = 1.0
    total_r = np.eye(3, dtype=np.float64)
    total_t = np.zeros(3, dtype=np.float64)

    for _ in range(max(1, icp_iters)):
        _, nn_idx = gt_tree.query(aligned, k=1, workers=-1)
        matched_gt = gt_pts[nn_idx].astype(np.float64)
        s, r, t = _umeyama_alignment(aligned, matched_gt, estimate_scale=estimate_scale)
        aligned = _apply_similarity(aligned, s, r, t)

        total_t = s * (r @ total_t) + t
        total_r = r @ total_r
        total_scale = s * total_scale

    info = {
        "align_mode": align_mode,
        "scale": float(total_scale),
        "rotation": total_r.tolist(),
        "translation": total_t.tolist(),
        "icp_iters": int(max(1, icp_iters)),
    }
    return aligned.astype(np.float32), info


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




def _compute_cut3r_style_metrics(pred_pts: np.ndarray, gt_pts: np.ndarray) -> dict:
    """Match CUT3R mv_recon metric naming: accuracy/completion (+ median)."""
    gt_tree = cKDTree(gt_pts)
    pred_tree = cKDTree(pred_pts)

    d_pred_to_gt, _ = gt_tree.query(pred_pts, k=1, workers=-1)
    d_gt_to_pred, _ = pred_tree.query(gt_pts, k=1, workers=-1)

    return {
        "accuracy_mean": float(np.mean(d_pred_to_gt)),
        "accuracy_median": float(np.median(d_pred_to_gt)),
        "completion_mean": float(np.mean(d_gt_to_pred)),
        "completion_median": float(np.median(d_gt_to_pred)),
    }


def _parse_threshold_list(text: str) -> list[float]:
    vals = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("At least one threshold must be provided")
    return vals


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
    parser.add_argument(
        "--align_mode",
        type=str,
        default="sim3",
        choices=["none", "rigid", "sim3"],
        help="Point cloud alignment mode before evaluation: none, rigid(SE3), or sim3(scale+rotation+translation).",
    )
    parser.add_argument("--icp_iters", type=int, default=10, help="ICP refinement iterations for rigid/sim3 alignment")
    parser.add_argument(
        "--eval_protocol",
        type=str,
        default="cut3r",
        choices=["cut3r", "legacy"],
        help="Metric reporting protocol. cut3r reports accuracy/completion style metrics.",
    )
    parser.add_argument(
        "--fscore_thresholds",
        type=str,
        default="0.05",
        help="Comma separated thresholds used for F-score reporting (e.g. '0.05,0.1').",
    )

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

    pred_pts_aligned, align_info = _align_pred_to_gt(pred_pts, gt_pts, align_mode=args.align_mode, icp_iters=args.icp_iters)

    metrics = _compute_pointcloud_metrics(pred_pts_aligned, gt_pts, threshold=args.distance_threshold)
    if args.eval_protocol == "cut3r":
        metrics.update(_compute_cut3r_style_metrics(pred_pts_aligned, gt_pts))

    for th in _parse_threshold_list(args.fscore_thresholds):
        th_metrics = _compute_pointcloud_metrics(pred_pts_aligned, gt_pts, threshold=th)
        metrics[f"fscore@{th:g}"] = th_metrics["fscore"]
        metrics[f"precision@{th:g}"] = th_metrics["precision"]
        metrics[f"recall@{th:g}"] = th_metrics["recall"]

    metrics.update(align_info)
    metrics["pred_pth"] = args.pred_pth
    metrics["gt_pcd"] = args.gt_pcd
    metrics["num_frames_used"] = int(len(selected_indices))
    metrics["frame_indices"] = selected_indices.tolist()

    out_json = os.path.join(args.output_dir, "pointcloud_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_pcd_path = os.path.join(args.output_dir, "pred_points_sampled.ply")
    gt_pcd_path = os.path.join(args.output_dir, "gt_points_sampled.ply")
    pred_aligned_pcd_path = os.path.join(args.output_dir, "pred_points_aligned_sampled.ply")
    trimesh.points.PointCloud(pred_pts.astype(np.float64)).export(pred_pcd_path)
    trimesh.points.PointCloud(gt_pts.astype(np.float64)).export(gt_pcd_path)
    trimesh.points.PointCloud(pred_pts_aligned.astype(np.float64)).export(pred_aligned_pcd_path)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
