import os
import torch
import numpy as np
import sys
import glob
import time
import argparse
import logging
from typing import List, Dict, Optional

# Add project source to the Python path
sys.path.append("src/")

# Import necessary components from the StreamVGGT project
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from streamvggt.utils.geometry import FrameDiskCache

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

def run_inference(args: argparse.Namespace):
    """
    Main function to load the model, run inference on input images, and save the results.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("Error: CUDA device not available.")
        return

    print("Initializing and loading StreamVGGT model ...")

    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return

    frame_writer = None
    cache_results = not args.no_cache_results

    if args.frame_cache_dir:
        frame_writer = FrameDiskCache(args.frame_cache_dir)

    model_total_budget = args.total_budget if args.use_geo_kv_prune else 1200000
    print(f"Geo prune budget config: total_budget={model_total_budget} (arg_total_budget={args.total_budget}, use_geo_kv_prune={args.use_geo_kv_prune})")
    aggregator_kwargs = {
        "geo_conf_ema_alpha": args.geo_conf_ema_alpha,
        "geo_var_ema_alpha": args.geo_var_ema_alpha,
        "geo_invisible_read_weight": args.geo_invisible_read_weight,
        "geo_bucket_quantile_target": args.geo_bucket_quantile_target,
        "geo_bucket_quantile_min": args.geo_bucket_quantile_min,
        "geo_bucket_quantile_max": args.geo_bucket_quantile_max,
        "geo_anchor_conf_enter": args.geo_anchor_conf_enter,
        "geo_anchor_conf_exit": args.geo_anchor_conf_exit,
        "geo_anchor_min_support": args.geo_anchor_min_support,
        "geo_anchor_max_pos_var": args.geo_anchor_max_pos_var,
        "geo_max_voxels": args.geo_max_voxels,
        "geo_anchor_voxel_budget": args.geo_anchor_voxel_budget,
        "geo_anchor_read_quota": args.geo_anchor_read_quota,
        "geo_local_budget_ratio": args.geo_local_budget_ratio,
        "geo_local_budget_cap_per_frame": args.geo_local_budget_cap_per_frame,
        "geo_anchor_budget_ratio": args.geo_anchor_budget_ratio,
        "geo_local_coverage_grid": args.geo_local_coverage_grid,
        "geo_frame0_patch_cap": args.geo_frame0_patch_cap,
        "geo_anchor_invisible_read_weight": args.geo_anchor_invisible_read_weight,
        "geo_max_old_frames_to_score": args.geo_max_old_frames_to_score,
        "geo_max_candidate_tokens": args.geo_max_candidate_tokens,
        "geo_selection_interval": args.geo_selection_interval,
        "geo_anchor_refresh_interval": args.geo_anchor_refresh_interval,
        "geo_stable_keyframe_topk_per_frame": args.geo_stable_keyframe_topk_per_frame,
        "geo_layer_budget_cap": args.geo_layer_budget_cap,
        "geo_reloc_frames": args.geo_reloc_frames,
        "geo_reloc_trigger_overlap": args.geo_reloc_trigger_overlap,
        "geo_reloc_trigger_visible_ratio": args.geo_reloc_trigger_visible_ratio,
        "geo_reloc_local_budget_ratio": args.geo_reloc_local_budget_ratio,
        "geo_reloc_stable_read_budget_ratio": args.geo_reloc_stable_read_budget_ratio,
        "geo_reloc_hard_frames": args.geo_reloc_hard_frames,
        "geo_bootstrap_frames": args.geo_bootstrap_frames,
        "geo_bootstrap_min_voxels": args.geo_bootstrap_min_voxels,
        "geo_bootstrap_min_refs": args.geo_bootstrap_min_refs,
        "geo_prune_start_ratio": args.geo_prune_start_ratio,
        "geo_console_log_interval": args.geo_console_log_interval,
    }
    model = StreamVGGT(total_budget=model_total_budget, aggregator_kwargs=aggregator_kwargs)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")

    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()
    del ckpt
    print("Model loaded successfully onto the GPU.")

    print(f"Loading images from input directory: {args.input_dir}")
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    image_names = sorted(
        p for p in glob.glob(os.path.join(args.input_dir, "*"))
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts
    )

    if args.max_views >= 0:
        image_names = image_names[: args.max_views]
    
    if not image_names:
        print(f"Error: No images found in {args.input_dir}. Please check the path and file extensions.")
        return
        
    print(f"Found {len(image_names)} images to process.")
    images = load_and_preprocess_images(image_names)
    if device == "cuda" and images.device.type == "cpu":
        images = images.pin_memory()
    print(f"Preprocessed images tensor shape: {images.shape}")

    frames: List[Dict[str, torch.Tensor]] = []
    for i in range(images.shape[0]):
        image_frame = images[i].unsqueeze(0)
        frame = {"img": image_frame}
        frames.append(frame)

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time_model = time.time()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            if args.use_geo_kv_prune:
                output = model.inference(
                    frames,
                    frame_writer=frame_writer,
                    cache_results=cache_results,
                    use_geo_kv_prune=True,
                    geo_voxel_size=args.geo_voxel_size,
                    geo_topk_per_voxel=args.geo_topk_per_voxel,
                    geo_recent_frames=args.geo_recent_frames,
                    geo_near=args.geo_near,
                    geo_far=args.geo_far,
                    show_progress=not args.no_progress,
                    memory_diagnostics=args.memory_diagnostics,
                    memory_log_interval=args.memory_log_interval,
                )
            else:
                output = model.inference(
                    frames,
                    frame_writer=frame_writer,
                    cache_results=cache_results,
                )

    torch.cuda.synchronize()
    end_time_model = time.time()

    model_execution_time = end_time_model - start_time_model
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory_bytes / (1024**3)

    print("\n" + "="*50)
    print("INFERENCE PERFORMANCE")
    print(f"  Model Execution Time: {model_execution_time:.4f} seconds")
    print(f"  Peak GPU Memory Usage: {peak_memory_gb:.2f} GB")
    print("="*50 + "\n")
    
    if (not cache_results) or output.ress is None or len(output.ress) == 0:
        summary = {"per_frame_only": True}
        if args.frame_cache_dir:
            summary["frame_cache_dir"] = args.frame_cache_dir
        return summary

    # Extract results from the output structure
    all_pts3d = [res['pts3d_in_other_view'].squeeze(0) for res in output.ress]
    all_conf = [res['conf'].squeeze(0) for res in output.ress]
    all_depth = [res['depth'].squeeze(0) for res in output.ress]
    all_depth_conf = [res['depth_conf'].squeeze(0) for res in output.ress]
    all_camera_pose = [res['camera_pose'].squeeze(0) for res in output.ress]

    # Create a dictionary to hold all prediction tensors
    predictions = {
        "world_points": torch.stack(all_pts3d, dim=0),
        "world_points_conf": torch.stack(all_conf, dim=0),
        "depth": torch.stack(all_depth, dim=0),
        "depth_conf": torch.stack(all_depth_conf, dim=0),
        "pose_enc": torch.stack(all_camera_pose, dim=0),
    }
    if args.use_geo_kv_prune:
        predictions["image_paths"] = [os.path.relpath(p, PROJECT_ROOT) for p in image_names]
    else:
        predictions["images"] = images

    # Convert pose encoding to extrinsic and intrinsic matrices
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"].unsqueeze(0), 
        images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic.squeeze(0)
    predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None

    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            predictions[key] = value.detach().cpu()

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run InfiniteVGGT inference from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Python logging level (e.g. DEBUG, INFO, WARNING).",
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/examples",  
        help="Path to the directory containing input images."
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="/checkpoints.pth", 
        help="Path to the model checkpoint file (.pth)."
    )
    parser.add_argument(
        "--frame_cache_dir",
        type=str,
        default=None,
        help="Write the prediction for each frame to cache dir",
    )
    parser.add_argument(
        "--no_cache_results",
        action="store_true",
        help="Prediction results will not be accumulated in GPU memory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./inference_results",
        help="Path to the directory containing the complete results"
    )
    parser.add_argument(
        "--use_geo_kv_prune",
        action="store_true",
        help="Enable geometry-guided KV cache pruning.",
    )
    parser.add_argument(
        "--geo_voxel_size",
        type=float,
        default=0.2,
        help="Voxel size (in world units) for geometry-guided KV pruning.",
    )
    parser.add_argument(
        "--geo_topk_per_voxel",
        type=int,
        default=4,
        help="Maximum number of representative tokens retained per voxel.",
    )
    parser.add_argument(
        "--geo_recent_frames",
        type=int,
        default=2,
        help="Always keep tokens from the most recent N frames.",
    )
    parser.add_argument(
        "--geo_near",
        type=float,
        default=0.05,
        help="Near plane distance for frustum activation.",
    )
    parser.add_argument(
        "--geo_far",
        type=float,
        default=200.0,
        help="Far plane distance for frustum activation.",
    )
    parser.add_argument(
        "--total_budget",
        type=int,
        default=1200000,
        help="Global token budget used to initialize StreamVGGT.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable per-frame progress bar during inference.",
    )
    parser.add_argument(
        "--memory_diagnostics",
        action="store_true",
        help="Print per-frame CUDA memory diagnostics logs.",
    )
    parser.add_argument(
        "--memory_log_interval",
        type=int,
        default=1,
        help="Log CUDA memory diagnostics every N frames when enabled.",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=-1,
        help="Maximum number of views to infer. -1 means use all views.",
    )
    parser.add_argument("--geo_conf_ema_alpha", type=float, default=0.9)
    parser.add_argument("--geo_var_ema_alpha", type=float, default=0.9)
    parser.add_argument("--geo_invisible_read_weight", type=float, default=0.2)
    parser.add_argument("--geo_bucket_quantile_target", type=float, default=0.6)
    parser.add_argument("--geo_bucket_quantile_min", type=float, default=0.3)
    parser.add_argument("--geo_bucket_quantile_max", type=float, default=0.9)
    parser.add_argument("--geo_anchor_conf_enter", type=float, default=1.25)
    parser.add_argument("--geo_anchor_conf_exit", type=float, default=1.05)
    parser.add_argument("--geo_anchor_min_support", type=float, default=2.0)
    parser.add_argument("--geo_anchor_max_pos_var", type=float, default=0.05)
    parser.add_argument("--geo_max_voxels", type=int, default=200000)
    parser.add_argument("--geo_anchor_voxel_budget", type=int, default=4096)
    parser.add_argument("--geo_anchor_read_quota", type=int, default=2048)
    parser.add_argument("--geo_local_budget_ratio", type=float, default=0.75)
    parser.add_argument("--geo_local_budget_cap_per_frame", type=int, default=1369)
    parser.add_argument("--geo_anchor_budget_ratio", type=float, default=0.35)
    parser.add_argument("--geo_local_coverage_grid", type=int, default=4)
    parser.add_argument("--geo_frame0_patch_cap", type=int, default=1000000)
    parser.add_argument("--geo_anchor_invisible_read_weight", type=float, default=0.5)
    parser.add_argument("--geo_max_old_frames_to_score", type=int, default=24)
    parser.add_argument("--geo_max_candidate_tokens", type=int, default=15000)
    parser.add_argument("--geo_selection_interval", type=int, default=1)
    parser.add_argument("--geo_anchor_refresh_interval", type=int, default=1)
    parser.add_argument("--geo_stable_keyframe_topk_per_frame", type=int, default=2)
    parser.add_argument("--geo_layer_budget_cap", type=int, default=8192)
    parser.add_argument("--geo_reloc_frames", type=int, default=8)
    parser.add_argument("--geo_reloc_trigger_overlap", type=int, default=128)
    parser.add_argument("--geo_reloc_trigger_visible_ratio", type=float, default=0.75)
    parser.add_argument("--geo_reloc_local_budget_ratio", type=float, default=0.1)
    parser.add_argument("--geo_reloc_stable_read_budget_ratio", type=float, default=0.4)
    parser.add_argument("--geo_reloc_hard_frames", type=int, default=2)
    parser.add_argument("--geo_bootstrap_frames", type=int, default=64)
    parser.add_argument("--geo_bootstrap_min_voxels", type=int, default=4096)
    parser.add_argument("--geo_bootstrap_min_refs", type=int, default=64)
    parser.add_argument("--geo_prune_start_ratio", type=float, default=0.9)
    parser.add_argument(
        "--geo_console_log_interval",
        type=int,
        default=50,
        help="Print geo prune console stats every N views; set -1 to disable.",
    )
    
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    result = run_inference(args)

    if result is None:
        print("Inference aborted due to previous errors.")
    elif result.get("per_frame_only", False):
        cache_dir = result.get("frame_cache_dir", args.frame_cache_dir)
        if cache_dir:
            print(f"Inference finished. Per-frame outputs saved under {cache_dir}.")
        else:
            print("Inference finished. Per-frame outputs were written via custom frame_writer.")
    else:
        torch.save(result, args.output_path)
        print(f"Inference finished. Results saved to {args.output_path}")
