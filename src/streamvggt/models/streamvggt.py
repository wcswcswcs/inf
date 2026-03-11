import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from tqdm.auto import tqdm
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any, Callable
from dataclasses import dataclass

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None
    state: Optional[dict] = None

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        total_budget=1200000,
        aggregator_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        aggregator_kwargs = aggregator_kwargs or {}
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            **aggregator_kwargs,
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.total_budget = total_budget
    


    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
    
    def inference(
        self, 
        frames, 
        query_points: torch.Tensor = None, 
        past_key_values=None, 
        past_key_values_camera=None,
        current_view: Optional[dict] = None,
        last_reliable_view: Optional[dict] = None,
        geo_state: Optional[dict] = None,
        rolling_state: Optional[dict] = None,
        frame_writer: Optional[Callable[[int, dict, dict], None]] = None,
        cache_results: bool = True,
        total_budget=None,
        use_geo_kv_prune: bool = False,
        geo_voxel_size: float = 0.2,
        geo_topk_per_voxel: int = 4,
        geo_recent_frames: int = 2,
        geo_near: float = 0.05,
        geo_far: float = 200.0,
        show_progress: bool = True,
        memory_diagnostics: bool = False,
        memory_log_interval: int = 1,
    ):
        if past_key_values is None:
            past_key_values = [None] * self.aggregator.depth
        if past_key_values_camera is None:
            past_key_values_camera = [None] * self.camera_head.trunk_depth
        if total_budget is None:
            total_budget = self.total_budget
        if use_geo_kv_prune:
            if geo_state is None:
                self.aggregator.reset_geo_cache_state()
                current_view = None
                last_reliable_view = None
            else:
                self.aggregator.load_geo_cache_state(geo_state)
        if rolling_state is None:
            prev_world_to_cam_cpu = None
            prev_conf_mean = None
        else:
            prev_world_to_cam_cpu = rolling_state.get("prev_world_to_cam_cpu", None)
            prev_conf_mean = rolling_state.get("prev_conf_mean", None)
        model_device = next(self.parameters()).device

        all_ress = []
        processed_frames = []

        frame_iter = frames
        if show_progress:
            frame_iter = tqdm(frames, total=len(frames), desc="Inference", unit="frame")

        log_interval = max(1, int(memory_log_interval))

        total_frames = len(frames)
        no_progress_log_interval = 50
        for i, frame in enumerate(frame_iter):
            if not show_progress and (
                ((i + 1) % no_progress_log_interval == 0)
                or (i == 0)
                or (i + 1 == total_frames)
            ):
                print(f"Inference step {i + 1}/{total_frames}", flush=True)

            images = frame["img"].unsqueeze(0).to(model_device, non_blocking=True)
            selected_view = current_view
            if use_geo_kv_prune:
                policy = self.aggregator._geo_peek_adaptive_policy(
                    frame_idx=i,
                    total_tokens=0,
                    max_past_tokens=None,
                    current_view=current_view,
                    observation=self.aggregator._geo_get_last_observation(),
                    selector_diag=self.aggregator._geo_get_last_selector_diag(),
                )
                use_stale_view = bool(policy["use_stale_view"])
                selected_view = None if (not use_stale_view) else current_view
                if use_stale_view and last_reliable_view is not None and self.aggregator.geo_trust_score < self.aggregator.geo_selection_low_trust_threshold:
                    selected_view = last_reliable_view
            aggregator_output = self.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i,
                total_budget=total_budget,
                use_geo_kv_prune=use_geo_kv_prune,
                geo_topk_per_voxel=geo_topk_per_voxel,
                geo_recent_frames=geo_recent_frames,
                geo_near=geo_near,
                geo_far=geo_far,
                current_view=selected_view,
            )

            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output        
            
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]
                
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0] 
                    pts3d_conf = pts3d_conf[:, 0]

                if use_geo_kv_prune and self.point_head is not None and self.camera_head is not None:
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        pose_enc,
                        images.shape[-2:]
                    )
                    world_to_cam = torch.eye(4, device=extrinsic.device, dtype=extrinsic.dtype).unsqueeze(0).repeat(extrinsic.shape[0], 1, 1)
                    world_to_cam[:, :3, :4] = extrinsic[:, 0]
                    intrinsic_cur = intrinsic[:, 0] if intrinsic is not None else None

                    # Keep pruning view metadata on CPU to avoid repeated per-layer GPU->CPU transfers.
                    world_to_cam_cpu = world_to_cam.detach().cpu()
                    intrinsic_cpu = intrinsic_cur.detach().cpu() if intrinsic_cur is not None else None
                    pose_delta = 0.0
                    if prev_world_to_cam_cpu is not None:
                        pose_delta = float((world_to_cam_cpu[:, :3, 3] - prev_world_to_cam_cpu[:, :3, 3]).norm(dim=-1).mean().item())

                    conf_mean = float(pts3d_conf.detach().to(torch.float32).mean().item())
                    conf_drop = 0.0 if prev_conf_mean is None else max(0.0, float(prev_conf_mean - conf_mean))

                    geo_meta_stats = self.aggregator.update_geo_frame_metadata(
                        frame_idx=i,
                        pts3d=pts3d.detach(),
                        conf=pts3d_conf.detach(),
                        world_to_cam=world_to_cam_cpu,
                        intrinsic=intrinsic_cpu,
                        voxel_size=geo_voxel_size,
                    )

                    current_view = {
                        "world_to_cam": world_to_cam_cpu,
                        "intrinsic": intrinsic_cpu,
                        "img_hw": tuple(int(x) for x in images.shape[-2:]),
                        "pose_delta": pose_delta,
                        "conf_drop": conf_drop,
                        "new_voxel_ratio": float(geo_meta_stats.get("new_voxel_ratio", 0.0)),
                    }
                    trust_now = float(geo_meta_stats.get("trust_score", float(self.aggregator.geo_trust_score)))
                    matched_ratio_now = float(geo_meta_stats.get("matched_ratio", 0.0))
                    if trust_now >= float(self.aggregator.geo_selection_low_trust_threshold) and matched_ratio_now >= 0.05:
                        last_reliable_view = current_view
                    prev_world_to_cam_cpu = world_to_cam_cpu
                    prev_conf_mean = conf_mean

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            res_gpu = {
                "pts3d_in_other_view": pts3d,
                "conf": pts3d_conf,
                "depth": depth,
                "depth_conf": depth_conf,
                "camera_pose": camera_pose,
                **({"valid_mask": frame["valid_mask"]} if "valid_mask" in frame else {}),
                **(
                    {"track": track, "vis": vis, "track_conf": track_conf}
                    if query_points is not None
                    else {}
                ),
            }
            needs_cpu_export = (frame_writer is not None) or cache_results
            if needs_cpu_export:
                res_cpu = {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in res_gpu.items()
                }

                if frame_writer is not None:
                    frame_writer(i, frame, res_cpu)

                if cache_results:
                    all_ress.append(res_cpu)
                    processed_frames.append(
                        {nk: nv.detach().cpu() if isinstance(nv, torch.Tensor) else nv for nk, nv in frame.items()}
                    )

            if memory_diagnostics and model_device.type == "cuda" and ((i + 1) % log_interval == 0 or (i + 1) == len(frames)):
                allocated_gb = torch.cuda.memory_allocated(model_device) / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved(model_device) / (1024 ** 3)
                max_allocated_gb = torch.cuda.max_memory_allocated(model_device) / (1024 ** 3)
                print(
                    f"[MEM][frame {i + 1}/{len(frames)}] "
                    f"allocated={allocated_gb:.2f} GB, reserved={reserved_gb:.2f} GB, max_allocated={max_allocated_gb:.2f} GB"
                )

            del res_gpu

        final_state = {
            "past_key_values": past_key_values,
            "past_key_values_camera": past_key_values_camera,
            "geo_state": self.aggregator.export_geo_cache_state() if use_geo_kv_prune else None,
            "current_view": current_view,
            "last_reliable_view": last_reliable_view,
            "rolling_state": {
                "prev_world_to_cam_cpu": prev_world_to_cam_cpu,
                "prev_conf_mean": prev_conf_mean,
            },
        }

        return StreamVGGTOutput(
            ress=all_ress if cache_results else None,
            views=processed_frames if cache_results else None,
            state=final_state,
        )
