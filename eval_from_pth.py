import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from eval.mv_recon.utils import accuracy, completion


def _depthmap_to_absolute_camera_coordinates(depthmap: np.ndarray, camera_intrinsics: np.ndarray, camera_pose: np.ndarray):
    try:
        from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates as _impl

        return _impl(
            depthmap=depthmap,
            camera_intrinsics=camera_intrinsics,
            camera_pose=camera_pose,
        )
    except Exception:  # noqa: BLE001
        h, w = depthmap.shape
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        z = depthmap.astype(np.float32)
        x = (xx - cx) * z / max(float(fx), 1e-8)
        y = (yy - cy) * z / max(float(fy), 1e-8)
        cam = np.stack([x, y, z], axis=-1)
        cam_h = np.concatenate([cam.reshape(-1, 3), np.ones((h * w, 1), dtype=np.float32)], axis=1)
        world = (camera_pose @ cam_h.T).T[:, :3].reshape(h, w, 3).astype(np.float32)
        valid = np.isfinite(z) & (z > 1e-4)
        return world, valid


def transpose_to_landscape(view: Dict[str, Any]):
    height, width = view["true_shape"]

    if width < height:
        view["img"] = view["img"].swapaxes(1, 2)
        view["valid_mask"] = view["valid_mask"].swapaxes(0, 1)
        view["depthmap"] = view["depthmap"].swapaxes(0, 1)
        view["pts3d"] = view["pts3d"].swapaxes(0, 1)
        view["camera_intrinsics"] = view["camera_intrinsics"][[1, 0, 2]]


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_inputs_or_predictions(path: str) -> Dict[str, Any]:
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
        (1, 2, True),
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
            raw = _lzf_decompress(compressed, uncompressed_size)
            return _extract_xyz_from_pcd_bytes(raw, fields, sizes, types, counts, points)

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


def _default_intrinsics(width: int, height: int) -> np.ndarray:
    f = float(max(width, height))
    return np.array(
        [[f, 0.0, width * 0.5], [0.0, f, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _ensure_frame_hw3(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D frame tensor, got {arr.shape}")
    if arr.shape[-1] == 3:
        return arr
    if arr.shape[1] == 3:
        return np.transpose(arr, (0, 2, 3, 1))
    raise ValueError(f"Cannot infer frame channel layout for shape={arr.shape}")


def _infer_image_stack(preds: Dict[str, Any], total_frames: int, h: int, w: int) -> np.ndarray:
    for key in ("img", "images", "rgb", "image"):
        if key in preds and preds[key] is not None:
            img = _to_numpy(preds[key])
            img = _ensure_frame_hw3(img)
            if img.shape[0] == total_frames:
                return img.astype(np.float32)
    return np.zeros((total_frames, h, w, 3), dtype=np.float32)


def _get_pose_stack(preds: Dict[str, Any], total_frames: int) -> np.ndarray:
    for key in ("camera_pose", "camera_poses", "poses", "world_to_cam", "extrinsics"):
        if key in preds and preds[key] is not None:
            pose = _to_numpy(preds[key]).astype(np.float32)
            if pose.ndim == 2 and pose.shape == (4, 4):
                pose = np.repeat(pose[None, ...], total_frames, axis=0)
            if pose.shape[0] == total_frames and pose.shape[-2:] == (4, 4):
                return pose
    eye = np.eye(4, dtype=np.float32)
    return np.repeat(eye[None, ...], total_frames, axis=0)


def _get_intr_stack(preds: Dict[str, Any], total_frames: int, h: int, w: int) -> np.ndarray:
    for key in ("camera_intrinsics", "intrinsics", "K"):
        if key in preds and preds[key] is not None:
            k = _to_numpy(preds[key]).astype(np.float32)
            if k.ndim == 2 and k.shape == (3, 3):
                k = np.repeat(k[None, ...], total_frames, axis=0)
            if k.shape[0] == total_frames and k.shape[-2:] == (3, 3):
                return k
    K = _default_intrinsics(w, h)
    return np.repeat(K[None, ...], total_frames, axis=0)


def _world_points_to_depth(world_pts: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    if world_pts.ndim != 3 or world_pts.shape[-1] != 3:
        raise ValueError(f"Expected per-frame HxWx3 world points, got {world_pts.shape}")
    pose_inv = np.linalg.inv(camera_pose)
    pts_flat = world_pts.reshape(-1, 3)
    pts_h = np.concatenate([pts_flat, np.ones((pts_flat.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (pose_inv @ pts_h.T).T[:, :3]
    depth = cam[:, 2].reshape(world_pts.shape[0], world_pts.shape[1]).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 1e-4] = 0.0
    return depth


def _build_or_adapt_views(preds: Dict[str, Any], selected_indices: np.ndarray) -> List[Dict[str, Any]]:
    world_points = _ensure_frame_hw3(_to_numpy(preds["world_points"]).astype(np.float32))
    world_points_conf = _to_numpy(preds["world_points_conf"]).astype(np.float32)
    if world_points_conf.ndim == 4 and world_points_conf.shape[-1] == 1:
        world_points_conf = world_points_conf[..., 0]

    total_frames, h, w, _ = world_points.shape
    imgs = _infer_image_stack(preds, total_frames, h, w)
    poses = _get_pose_stack(preds, total_frames)
    intrs = _get_intr_stack(preds, total_frames, h, w)

    depth_stack = None
    for key in ("depthmap", "depth", "depth_maps"):
        if key in preds and preds[key] is not None:
            depth_stack = _to_numpy(preds[key]).astype(np.float32)
            break
    if depth_stack is None:
        depth_stack = np.stack([
            _world_points_to_depth(world_points[i], poses[i]) for i in range(total_frames)
        ], axis=0)

    views: List[Dict[str, Any]] = []
    for idx in selected_indices.tolist():
        img = imgs[idx]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img_chw = np.transpose(img, (2, 0, 1)).astype(np.float32)

        depthmap = np.nan_to_num(depth_stack[idx].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        depthmap[depthmap < 1e-4] = 0.0

        view = {
            "img": torch.from_numpy(img_chw),
            "depthmap": depthmap,
            "camera_pose": poses[idx].astype(np.float32),
            "camera_intrinsics": intrs[idx].astype(np.float32),
            "dataset": "pth_eval",
            "label": f"scene/{idx:06d}",
            "instance": str(idx),
            "true_shape": np.int32((depthmap.shape[0], depthmap.shape[1])),
            "idx": 0,
            "img_mask": True,
            "ray_mask": False,
            "ray_map": torch.full((6, depthmap.shape[0], depthmap.shape[1]), torch.nan),
            "update": True,
            "reset": False,
        }

        pts3d_calc, valid_mask = _depthmap_to_absolute_camera_coordinates(
            depthmap=view["depthmap"],
            camera_intrinsics=view["camera_intrinsics"],
            camera_pose=view["camera_pose"],
        )

        pts3d_world = world_points[idx]
        pts3d = np.where(np.isfinite(pts3d_world), pts3d_world, pts3d_calc).astype(np.float32)
        valid_conf = np.isfinite(world_points_conf[idx]) & (world_points_conf[idx] > 0)
        view["pts3d"] = pts3d
        view["valid_mask"] = (valid_mask & np.isfinite(pts3d).all(axis=-1) & valid_conf)

        transpose_to_landscape(view)
        views.append(view)

    return views


def _move_batch_to_device(views: Sequence[Dict[str, Any]], device: torch.device) -> List[Dict[str, Any]]:
    ignore_keys = {"depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"}
    out = []
    for view in views:
        moved = {}
        for k, v in view.items():
            if k in ignore_keys:
                moved[k] = v
            elif isinstance(v, torch.Tensor):
                moved[k] = v.to(device, non_blocking=True)
            elif isinstance(v, list):
                moved[k] = [x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in v]
            elif isinstance(v, tuple):
                moved[k] = tuple(x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in v)
            else:
                moved[k] = v
        out.append(moved)
    return out


def _run_inference_or_decode_pth(views: Sequence[Dict[str, Any]], world_points_conf: np.ndarray) -> List[Dict[str, torch.Tensor]]:
    preds = []
    for i, view in enumerate(views):
        pts = torch.from_numpy(np.asarray(view["pts3d"], dtype=np.float32))[None]
        conf = torch.from_numpy(np.asarray(world_points_conf[i], dtype=np.float32))[None]
        preds.append({"pts3d_in_other_view": pts, "conf": conf})
    return preds


def _convert_outputs_to_mv_recon_preds(views: Sequence[Dict[str, Any]], preds_np: Sequence[Dict[str, torch.Tensor]]) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    gts: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []
    for view, pred in zip(views, preds_np):
        img_tensor = view["img"][None] if isinstance(view["img"], torch.Tensor) else torch.from_numpy(np.asarray(view["img"], dtype=np.float32))[None]
        ray_map = view.get("ray_map")
        if isinstance(ray_map, torch.Tensor):
            ray_map_tensor = ray_map[None]
        else:
            ray_map_tensor = torch.full((1, 6, img_tensor.shape[-2], img_tensor.shape[-1]), torch.nan)
        gt = {
            "pts3d": torch.from_numpy(np.asarray(view["pts3d"], dtype=np.float32))[None],
            "valid_mask": torch.from_numpy(np.asarray(view["valid_mask"], dtype=bool))[None],
            "camera_pose": torch.from_numpy(np.asarray(view["camera_pose"], dtype=np.float32))[None],
            "camera_intrinsics": torch.from_numpy(np.asarray(view["camera_intrinsics"], dtype=np.float32))[None],
            "depthmap": torch.from_numpy(np.asarray(view["depthmap"], dtype=np.float32))[None],
            "img": img_tensor,
            "dataset": [view["dataset"]],
            "label": [view["label"]],
            "instance": [view["instance"]],
            "idx": torch.tensor([int(view.get("idx", 0))], dtype=torch.int64),
            "true_shape": torch.from_numpy(np.asarray(view["true_shape"], dtype=np.int32))[None],
            "img_mask": torch.tensor([bool(view.get("img_mask", True))]),
            "ray_mask": torch.tensor([bool(view.get("ray_mask", False))]),
            "ray_map": ray_map_tensor,
            "update": torch.tensor([bool(view.get("update", True))]),
            "reset": torch.tensor([bool(view.get("reset", False))]),
        }
        gts.append(gt)
        preds.append(pred)
    return gts, preds


def _flatten_scene_points(gt_pts_t: List[torch.Tensor], pred_pts_t: List[torch.Tensor], masks_t: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    all_gt = []
    all_pr = []
    for gt_t, pr_t, m_t in zip(gt_pts_t, pred_pts_t, masks_t):
        gt = gt_t.detach().cpu().numpy()[0]
        pr = pr_t.detach().cpu().numpy()[0]
        m = m_t.detach().cpu().numpy()[0].astype(bool)
        valid_joint = m & np.isfinite(gt).all(axis=-1) & np.isfinite(pr).all(axis=-1)
        if np.any(valid_joint):
            all_gt.append(gt[valid_joint])
            all_pr.append(pr[valid_joint])
    if not all_gt:
        raise ValueError("No valid scene points after applying valid_mask and finite joint mask")
    return np.concatenate(all_gt, axis=0), np.concatenate(all_pr, axis=0)


def _estimate_normals(points: np.ndarray, k: int = 24) -> np.ndarray:
    if points.shape[0] < 8:
        return np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (points.shape[0], 1))
    tree = cKDTree(points)
    k_eff = min(max(8, k), points.shape[0])
    _, nn_idx = tree.query(points, k=k_eff, workers=-1)
    normals = np.zeros_like(points, dtype=np.float32)
    for i in range(points.shape[0]):
        patch = points[nn_idx[i]]
        c = patch.mean(axis=0, keepdims=True)
        cov = (patch - c).T @ (patch - c) / max(1, patch.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        n = eigvecs[:, np.argmin(eigvals)]
        if n[2] < 0:
            n = -n
        normals[i] = n.astype(np.float32)
    n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(n_norm, 1e-8)
    return normals


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
        scale = 1.0 if var_src <= 1e-12 else float(np.trace(np.diag(d) @ s_mat) / var_src)
    else:
        scale = 1.0

    t = dst_mean - scale * (r @ src_mean)
    return scale, r, t


def _apply_similarity(pts: np.ndarray, scale: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (pts @ r.T)) + t[None, :]


def _align_pred_to_gt(pred_pts: np.ndarray, gt_pts: np.ndarray, align_mode: str, icp_iters: int) -> tuple[np.ndarray, dict]:
    if align_mode == "none":
        return pred_pts, {"align_mode": align_mode, "scale": 1.0, "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]}

    aligned = pred_pts.astype(np.float64)
    gt_tree = cKDTree(gt_pts)

    total_scale = 1.0
    total_r = np.eye(3, dtype=np.float64)
    total_t = np.zeros(3, dtype=np.float64)

    # Align behavior with mv_recon launch:
    # - rigid: point-to-point ICP (SE3, no scale)
    # - sim3: one scale alignment first, then rigid ICP refinement
    if align_mode == "sim3":
        _, nn_idx0 = gt_tree.query(aligned, k=1, workers=-1)
        matched_gt0 = gt_pts[nn_idx0].astype(np.float64)
        s0, r0, t0 = _umeyama_alignment(aligned, matched_gt0, estimate_scale=True)
        aligned = _apply_similarity(aligned, s0, r0, t0)
        total_t = s0 * (r0 @ total_t) + t0
        total_r = r0 @ total_r
        total_scale = s0 * total_scale

    for _ in range(max(1, icp_iters)):
        _, nn_idx = gt_tree.query(aligned, k=1, workers=-1)
        matched_gt = gt_pts[nn_idx].astype(np.float64)
        s, r, t = _umeyama_alignment(aligned, matched_gt, estimate_scale=False)
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


def _compute_cut3r_style_metrics(pred_pts: np.ndarray, gt_pts: np.ndarray, gt_normals: np.ndarray, pred_normals: np.ndarray) -> dict:
    acc, acc_med, nc1, nc1_med = accuracy(gt_pts, pred_pts, gt_normals, pred_normals)
    comp, comp_med, nc2, nc2_med = completion(gt_pts, pred_pts, gt_normals, pred_normals)
    return {
        "accuracy_mean": float(acc),
        "accuracy_median": float(acc_med),
        "completion_mean": float(comp),
        "completion_median": float(comp_med),
        "normal_consistency_accuracy_mean": float(nc1),
        "normal_consistency_accuracy_median": float(nc1_med),
        "normal_consistency_completion_mean": float(nc2),
        "normal_consistency_completion_median": float(nc2_med),
        "normal_consistency": float(0.5 * (nc1 + nc2)),
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


def _evaluate_one_scene(
    views: Sequence[Dict[str, Any]],
    world_points_conf: np.ndarray,
    gt_pcd_path: str,
    max_points: int,
    seed: int,
    align_mode: str,
    icp_iters: int,
    distance_threshold: float,
    eval_protocol: str,
    fscore_thresholds: Sequence[float],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    criterion = None
    try:
        from eval.mv_recon.criterion import L21 as _L21, Regr3D_t_ScaleShiftInv as _Regr3D_t_ScaleShiftInv

        criterion = _Regr3D_t_ScaleShiftInv(_L21, norm_mode=False, gt_scale=True)
    except Exception:  # noqa: BLE001
        criterion = None
    preds_np = _run_inference_or_decode_pth(views, world_points_conf)
    gts_t, preds_t = _convert_outputs_to_mv_recon_preds(views, preds_np)

    if criterion is not None:
        with torch.no_grad():
            gt_pts_t, pred_pts_t, _, _, masks_t, _ = criterion.get_all_pts3d_t(gts_t, preds_t)
        gt_scene, pred_scene = _flatten_scene_points(gt_pts_t, pred_pts_t, masks_t)
    else:
        gt_scene = np.concatenate([
            np.asarray(v["pts3d"], dtype=np.float32)[np.asarray(v["valid_mask"], dtype=bool)] for v in views
        ], axis=0)
        pred_scene = np.concatenate([
            np.asarray(v["pts3d"], dtype=np.float32)[np.asarray(v["valid_mask"], dtype=bool)] for v in views
        ], axis=0)

    if gt_scene.shape[0] > max_points:
        rng = np.random.default_rng(seed + 9)
        idx = rng.choice(gt_scene.shape[0], size=max_points, replace=False)
        gt_scene = gt_scene[idx]
        pred_scene = pred_scene[idx]

    gt_gt = _load_gt_pcd(gt_pcd_path, max_points=max_points, rng_seed=seed)

    pred_aligned, align_info = _align_pred_to_gt(pred_scene, gt_gt, align_mode=align_mode, icp_iters=icp_iters)

    metrics = _compute_pointcloud_metrics(pred_aligned, gt_gt, threshold=distance_threshold)

    pred_normals = _estimate_normals(pred_aligned)
    gt_normals = _estimate_normals(gt_gt)

    if eval_protocol == "cut3r":
        metrics.update(_compute_cut3r_style_metrics(pred_aligned, gt_gt, gt_normals, pred_normals))

    for th in fscore_thresholds:
        th_metrics = _compute_pointcloud_metrics(pred_aligned, gt_gt, threshold=th)
        metrics[f"fscore@{th:g}"] = th_metrics["fscore"]
        metrics[f"precision@{th:g}"] = th_metrics["precision"]
        metrics[f"recall@{th:g}"] = th_metrics["recall"]

    metrics.update(align_info)
    return metrics, pred_scene, gt_gt, pred_aligned


def _save_scene_artifacts(output_dir: str, pred_pts: np.ndarray, gt_pts: np.ndarray, pred_aligned_pts: np.ndarray):
    pred_pcd_path = os.path.join(output_dir, "pred_points_sampled.ply")
    gt_pcd_path = os.path.join(output_dir, "gt_points_sampled.ply")
    pred_aligned_pcd_path = os.path.join(output_dir, "pred_points_aligned_sampled.ply")
    trimesh.points.PointCloud(pred_pts.astype(np.float64)).export(pred_pcd_path)
    trimesh.points.PointCloud(gt_pts.astype(np.float64)).export(gt_pcd_path)
    trimesh.points.PointCloud(pred_aligned_pts.astype(np.float64)).export(pred_aligned_pcd_path)


def _aggregate_logs(output_dir: str, scene_line: str):
    log_file = os.path.join(output_dir, "logs_0.txt")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(scene_line + "\n")
    with open(log_file, "r", encoding="utf-8") as f:
        all_lines = f.read()
    with open(os.path.join(output_dir, "logs_all.txt"), "w", encoding="utf-8") as f:
        f.write(all_lines)


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

    preds = _load_inputs_or_predictions(args.pred_pth)
    world_points = _ensure_frame_hw3(preds["world_points"].astype(np.float32))
    world_points_conf = preds["world_points_conf"].astype(np.float32)
    if world_points_conf.ndim == 4 and world_points_conf.shape[-1] == 1:
        world_points_conf = world_points_conf[..., 0]

    selected_indices = _select_indices(
        total_frames=len(world_points),
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        frame_range=args.frame_range,
    )

    views = _build_or_adapt_views(preds, selected_indices)

    selected_conf = world_points_conf[selected_indices]
    if selected_conf.ndim == 3:
        pass
    else:
        selected_conf = np.squeeze(selected_conf)

    conf_mask = np.isfinite(selected_conf)
    conf_mask &= selected_conf >= float(args.conf_threshold)
    for i, view in enumerate(views):
        vm = np.asarray(view["valid_mask"], dtype=bool)
        if conf_mask[i].shape == vm.shape:
            view["valid_mask"] = vm & conf_mask[i]

    fscore_ths = _parse_threshold_list(args.fscore_thresholds)

    metrics, pred_pts, gt_pts, pred_aligned_pts = _evaluate_one_scene(
        views=views,
        world_points_conf=selected_conf,
        gt_pcd_path=args.gt_pcd,
        max_points=args.max_points,
        seed=args.seed,
        align_mode=args.align_mode,
        icp_iters=args.icp_iters,
        distance_threshold=args.distance_threshold,
        eval_protocol=args.eval_protocol,
        fscore_thresholds=fscore_ths,
    )

    metrics["pred_pth"] = args.pred_pth
    metrics["gt_pcd"] = args.gt_pcd
    metrics["num_frames_used"] = int(len(selected_indices))
    metrics["frame_indices"] = selected_indices.tolist()

    out_json = os.path.join(args.output_dir, "pointcloud_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _save_scene_artifacts(args.output_dir, pred_pts, gt_pts, pred_aligned_pts)

    scene_id = os.path.splitext(os.path.basename(args.gt_pcd))[0]
    scene_line = (
        f"Idx: {scene_id}, Acc: {metrics.get('accuracy_mean', np.nan)}, Comp: {metrics.get('completion_mean', np.nan)}, "
        f"NC: {metrics.get('normal_consistency', np.nan)} - Acc_med: {metrics.get('accuracy_median', np.nan)}, "
        f"Comp_med: {metrics.get('completion_median', np.nan)}"
    )
    _aggregate_logs(args.output_dir, scene_line)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
