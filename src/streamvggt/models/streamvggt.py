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
    
    @staticmethod
    def _kv_list_to_cpu(kv_list):
        out = []
        for kv in kv_list:
            if kv is None:
                out.append(None)
            else:
                out.append((kv[0].detach().cpu(), kv[1].detach().cpu()))
        return out

    @staticmethod
    def _kv_list_to_device(kv_list, device):
        out = []
        for kv in kv_list:
            if kv is None:
                out.append(None)
            else:
                out.append((
                    kv[0].to(device, non_blocking=True),
                    kv[1].to(device, non_blocking=True),
                ))
        return out

    @staticmethod
    def _kv_list_device(kv_list):
        if kv_list is None:
            return None
        for kv in kv_list:
            if kv is not None:
                return kv[0].device
        return None

    @staticmethod
    def _has_any_kv(kv_list) -> bool:
        return bool(
            kv_list is not None
            and any(kv is not None for kv in kv_list)
        )

    def _clear_loaded_geo_token_cache_state(self):
        self.aggregator.geo_cached_landmark_identity_keep = torch.empty((0,), dtype=torch.long)
        self.aggregator.geo_pending_console_log = None

        for layer_idx in range(self.aggregator.depth):
            self.aggregator.geo_token_meta[layer_idx] = {
                "frame_idx": torch.empty(0, dtype=torch.long),
                "is_special": torch.empty(0, dtype=torch.bool),
                "is_keyframe": torch.empty(0, dtype=torch.bool),
                "local_patch_idx": torch.empty(0, dtype=torch.long),
                "identity_local": torch.empty(0, dtype=torch.long),
                "global_id": torch.empty(0, dtype=torch.long),
                "is_anchor": torch.empty(0, dtype=torch.bool),
                "is_landmark": torch.empty(0, dtype=torch.bool),
                "is_reference": torch.empty(0, dtype=torch.bool),
                "geo_role": torch.empty(0, dtype=torch.long),
            }

        # clear selector-local continuity signals tied to token-aligned KV cache
        self.aggregator.geo_last_selector_diag = {
            "frame_idx": -1,
            "stable_visible_overlap": 0.0,
            "stable_visible_ratio": 0.0,
            "visible_total": 0.0,
            "selected_total": 0.0,
        }
        self.aggregator.geo_selector_overlap_ema = 0.0
        self.aggregator.geo_selector_visible_ratio_ema = 0.0
        self.aggregator.geo_handover_ready_streak = 0
        self.aggregator.geo_handover_unready_streak = 0
        self.aggregator.geo_recovery_enter_streak = 0
        self.aggregator.geo_recovery_exit_streak = 0

        # logging/debug state tied to old selector continuity
        self.aggregator.geo_last_policy_inputs = {}
        self.aggregator.geo_last_policy_metrics = {}
        self.aggregator.geo_last_commit_guard_frame = -1

    def _validate_geo_kv_alignment(self, past_key_values):
        for layer_idx in range(self.aggregator.depth):
            kv = None if past_key_values is None else past_key_values[layer_idx]
            kv_len = 0 if kv is None else int(kv[0].shape[2])

            layer_meta = self.aggregator.geo_token_meta[layer_idx]
            frame_idx_meta = layer_meta.get("frame_idx", torch.empty((0,), dtype=torch.long))
            meta_len = int(frame_idx_meta.numel())

            if meta_len != kv_len:
                raise ValueError(
                    f"Inconsistent geo resume state at layer {layer_idx}: "
                    f"geo_token_meta has {meta_len} tokens but past_key_values has {kv_len}."
                )

    @torch.inference_mode()
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
        geo_voxel_size: float = 0.15,
        geo_topk_per_voxel: int = 4,
        geo_recent_frames: int = 2,
        geo_near: float = 0.05,
        geo_far: float = 200.0,
        show_progress: bool = True,
        memory_diagnostics: bool = False,
        memory_log_interval: int = 1,
        offload_camera_cache_to_cpu: bool = False,
        frame_start_idx: Optional[int] = None,
    ):
        if past_key_values is None:
            past_key_values = [None] * self.aggregator.depth
        if past_key_values_camera is None:
            past_key_values_camera = [None] * self.camera_head.trunk_depth
        has_past_kv = self._has_any_kv(past_key_values)
        has_past_kv_camera = self._has_any_kv(past_key_values_camera)
        is_resuming_from_cache = bool(has_past_kv or has_past_kv_camera)

        if total_budget is None:
            total_budget = self.total_budget
        if use_geo_kv_prune:
            if geo_state is None:
                self.aggregator.reset_geo_cache_state()
                current_view = None
                last_reliable_view = None
            else:
                self.aggregator.load_geo_cache_state(geo_state)
                if has_past_kv:
                    self._validate_geo_kv_alignment(past_key_values)
                else:
                    self._clear_loaded_geo_token_cache_state()
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
        expected_next_from_roll = None
        if rolling_state is not None and rolling_state.get("next_frame_idx", None) is not None:
            expected_next_from_roll = int(rolling_state["next_frame_idx"])

        expected_next_from_geo = None
        if use_geo_kv_prune and geo_state is not None:
            expected_next_from_geo = int(getattr(self.aggregator, "geo_max_frame_idx", -1)) + 1

        if use_geo_kv_prune and has_past_kv and geo_state is None:
            raise ValueError(
                "Resuming geo KV pruning with past_key_values requires geo_state. "
                "Provide geo_state together with cached aggregator KV."
            )

        has_explicit_resume_source = bool(
            (frame_start_idx is not None)
            or (expected_next_from_roll is not None)
            or (expected_next_from_geo is not None)
        )
        if is_resuming_from_cache and (not has_explicit_resume_source):
            raise ValueError(
                "Resuming with cached KV requires an explicit frame index source. "
                "Provide one of: frame_start_idx, rolling_state['next_frame_idx'], "
                "or geo_state with a valid next frame."
            )

        if frame_start_idx is not None:
            frame_start_idx = int(frame_start_idx)
            if expected_next_from_roll is not None and expected_next_from_roll != frame_start_idx:
                raise ValueError(
                    f"Inconsistent resume state: frame_start_idx={frame_start_idx} "
                    f"but rolling_state.next_frame_idx={expected_next_from_roll}"
                )
            if expected_next_from_geo is not None and expected_next_from_geo != frame_start_idx:
                raise ValueError(
                    f"Inconsistent resume state: frame_start_idx={frame_start_idx} "
                    f"but geo_state implies next_frame_idx={expected_next_from_geo}"
                )
            base_frame_idx = frame_start_idx
        else:
            if (
                expected_next_from_roll is not None
                and expected_next_from_geo is not None
                and expected_next_from_roll != expected_next_from_geo
            ):
                raise ValueError(
                    f"Inconsistent resume state: rolling_state.next_frame_idx={expected_next_from_roll} "
                    f"but geo_state implies next_frame_idx={expected_next_from_geo}"
                )
            if expected_next_from_roll is not None:
                base_frame_idx = expected_next_from_roll
            elif expected_next_from_geo is not None:
                base_frame_idx = expected_next_from_geo
            else:
                base_frame_idx = 0

        no_progress_log_interval = 50
        for i, frame in enumerate(frame_iter):
            frame_idx_abs = int(base_frame_idx + i)
            if not show_progress and (
                ((i + 1) % no_progress_log_interval == 0)
                or (i == 0)
                or (i + 1 == total_frames)
            ):
                print(f"Inference step {i + 1}/{total_frames}", flush=True)

            images = frame["img"].unsqueeze(0).to(model_device, non_blocking=True)
            policy_view = current_view
            selector_view = policy_view
            policy_view_source = "current_view" if policy_view is not None else "none"
            selector_view_source = policy_view_source
            geo_use_view_pruning = True
            if use_geo_kv_prune:
                policy = self.aggregator._geo_peek_effective_policy_for_inference(
                    past_key_values=past_key_values,
                    past_frame_idx=frame_idx_abs,
                    total_budget=total_budget,
                    current_view=policy_view,
                    geo_recent_frames=geo_recent_frames,
                )
                geo_use_view_pruning = bool(policy.get("use_view_pruning", True))
                prefer_last_reliable_view = bool(policy.get("prefer_last_reliable_view", False))
                health = self.aggregator._geo_compute_controller_health(frame_idx=int(frame_idx_abs))
                selector_fallback_ready = bool(
                    self.aggregator._geo_structure_ready()
                    or self.aggregator._geo_prestructure_reference_ready()
                )
                current_release_ready = bool(self.aggregator._geo_current_release_ready())
                soft_current_ready = bool(self.aggregator._geo_soft_current_ready(int(frame_idx_abs)))
                healthy_selector = bool(
                    health["selector_fresh"]
                    and float(health["controller_stress"]) <= 0.20
                    and (not bool(health["runtime_bad"]))
                    and (not bool(health["external_drift_bad"]))
                )
                force_current_selector = bool(
                    self.aggregator._geo_structure_ready()
                    and current_release_ready
                    and soft_current_ready
                    and healthy_selector
                )
                self.aggregator.geo_last_policy_inputs["inference_force_current_selector"] = bool(force_current_selector)
                allow_last_reliable = bool(
                    prefer_last_reliable_view
                    and last_reliable_view is not None
                    and (
                    bool(health["runtime_bad"])
                    or float(health["controller_stress"]) >= 0.45
                    or bool(health["external_drift_bad"])
                    )
                )
                self.aggregator.geo_last_policy_inputs["inference_allow_last_reliable"] = bool(allow_last_reliable)
                if (
                    ((not force_current_selector) or self.aggregator.geo_trust_score < self.aggregator.geo_selection_low_trust_threshold)
                    and selector_fallback_ready
                    and allow_last_reliable
                ):
                    selector_view = last_reliable_view
                    selector_view_source = "last_reliable_view"
                else:
                    selector_view = policy_view
                    selector_view_source = "current_view" if selector_view is not None else "none"
            aggregator_output = self.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=frame_idx_abs,
                total_budget=total_budget,
                use_geo_kv_prune=use_geo_kv_prune,
                geo_topk_per_voxel=geo_topk_per_voxel,
                geo_recent_frames=geo_recent_frames,
                geo_near=geo_near,
                geo_far=geo_far,
                geo_use_view_pruning=geo_use_view_pruning,
                current_view=selector_view,
                policy_view=policy_view,
                policy_view_source=policy_view_source,
                selector_view_source=selector_view_source,
            )

            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output        
            
            needs_cpu_export = (frame_writer is not None) or cache_results
            need_camera = self.camera_head is not None
            need_depth = (self.depth_head is not None) and bool(needs_cpu_export)
            need_point = (self.point_head is not None) and bool(needs_cpu_export or use_geo_kv_prune)
            need_track = (self.track_head is not None) and (query_points is not None)

            camera_cache_for_step = past_key_values_camera
            camera_cache_device = self._kv_list_device(camera_cache_for_step)
            if (
                camera_cache_for_step is not None
                and camera_cache_device is not None
                and camera_cache_device != model_device
            ):
                camera_cache_for_step = self._kv_list_to_device(camera_cache_for_step, model_device)

            pose_enc = None
            camera_pose = None
            depth = None
            depth_conf = None
            pts3d = None
            pts3d_conf = None
            track = None
            vis = None
            track_conf = None
            new_camera_cache = None
            extrinsic = None
            intrinsic = None
            world_to_cam = None
            intrinsic_cur = None

            with torch.cuda.amp.autocast(enabled=False):
                if need_camera:
                    pose_enc, new_camera_cache = self.camera_head(
                        aggregated_tokens,
                        past_key_values_camera=camera_cache_for_step,
                        use_cache=True,
                    )
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :] if needs_cpu_export else None

                if need_depth:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0]
                    depth_conf = depth_conf[:, 0]

                if need_point:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0]
                    pts3d_conf = pts3d_conf[:, 0]

                if use_geo_kv_prune and need_point and need_camera and pose_enc is not None and pts3d_conf is not None:
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
                        t_cur = world_to_cam_cpu[:, :3, 3]
                        t_prev = prev_world_to_cam_cpu[:, :3, 3]
                        trans_delta = float((t_cur - t_prev).norm(dim=-1).mean().item())

                        r_cur = world_to_cam_cpu[:, :3, :3]
                        r_prev = prev_world_to_cam_cpu[:, :3, :3]
                        r_rel = torch.matmul(r_cur, r_prev.transpose(-1, -2))
                        trace = r_rel[:, 0, 0] + r_rel[:, 1, 1] + r_rel[:, 2, 2]
                        cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
                        rot_delta = float(torch.acos(cos_theta).mean().item())

                        pose_delta = float(trans_delta + 0.5 * rot_delta)

                    conf_mean = float(pts3d_conf.detach().to(torch.float32).mean().item())
                    conf_drop = 0.0 if prev_conf_mean is None else max(0.0, float(prev_conf_mean - conf_mean))

                    geo_meta_stats = self.aggregator.update_geo_frame_metadata(
                        frame_idx=frame_idx_abs,
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

                if need_track:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                    )
                    track = track_list[-1][:, 0]
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            if need_camera:
                if offload_camera_cache_to_cpu:
                    past_key_values_camera = self._kv_list_to_cpu(new_camera_cache)
                else:
                    past_key_values_camera = new_camera_cache

            res_cpu = None
            if needs_cpu_export:
                res_cpu = {
                    "frame_idx_abs": int(frame_idx_abs),
                }
                if pts3d is not None:
                    res_cpu["pts3d_in_other_view"] = pts3d.detach().cpu()
                    res_cpu["conf"] = pts3d_conf.detach().cpu() if pts3d_conf is not None else None
                if depth is not None:
                    res_cpu["depth"] = depth.detach().cpu()
                    res_cpu["depth_conf"] = depth_conf.detach().cpu() if depth_conf is not None else None
                if camera_pose is not None:
                    res_cpu["camera_pose"] = camera_pose.detach().cpu()
                if "valid_mask" in frame:
                    res_cpu["valid_mask"] = (
                        frame["valid_mask"].detach().cpu()
                        if isinstance(frame["valid_mask"], torch.Tensor)
                        else frame["valid_mask"]
                    )
                if need_track and track is not None:
                    res_cpu["track"] = track.detach().cpu()
                    res_cpu["vis"] = vis.detach().cpu() if vis is not None else None
                    res_cpu["track_conf"] = track_conf.detach().cpu() if track_conf is not None else None

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

            del aggregated_tokens
            if pose_enc is not None:
                del pose_enc
            if camera_pose is not None:
                del camera_pose
            if depth is not None:
                del depth
            if depth_conf is not None:
                del depth_conf
            if pts3d is not None:
                del pts3d
            if pts3d_conf is not None:
                del pts3d_conf
            if track is not None:
                del track
            if vis is not None:
                del vis
            if track_conf is not None:
                del track_conf
            if extrinsic is not None:
                del extrinsic
            if intrinsic is not None:
                del intrinsic
            if world_to_cam is not None:
                del world_to_cam
            if intrinsic_cur is not None:
                del intrinsic_cur
            if camera_cache_for_step is not None and offload_camera_cache_to_cpu:
                del camera_cache_for_step
            if new_camera_cache is not None and offload_camera_cache_to_cpu:
                del new_camera_cache
            del images

        next_frame_idx = int(base_frame_idx + total_frames)
        final_state = {
            "past_key_values": past_key_values,
            "past_key_values_camera": past_key_values_camera,
            "geo_state": self.aggregator.export_geo_cache_state() if use_geo_kv_prune else None,
            "current_view": current_view,
            "last_reliable_view": last_reliable_view,
            "next_frame_idx": next_frame_idx,
            "rolling_state": {
                "prev_world_to_cam_cpu": prev_world_to_cam_cpu,
                "prev_conf_mean": prev_conf_mean,
                "next_frame_idx": next_frame_idx,
                "base_frame_idx_used": int(base_frame_idx),
            },
            "resume_source_info": {
                "next_frame_idx": next_frame_idx,
                "base_frame_idx_used": int(base_frame_idx),
                "has_geo_state": bool(use_geo_kv_prune),
                "input_had_geo_state": bool(use_geo_kv_prune and geo_state is not None),
                "geo_resume_mode": (
                    "full"
                    if (use_geo_kv_prune and has_past_kv)
                    else "map_only"
                    if use_geo_kv_prune
                    else "disabled"
                ),
                "geo_selector_state_reset": bool(use_geo_kv_prune and geo_state is not None and (not has_past_kv)),
                "has_rolling_state": bool(rolling_state is not None),
                "used_frame_start_idx": bool(frame_start_idx is not None),
            },
        }

        return StreamVGGTOutput(
            ress=all_ress if cache_results else None,
            views=processed_frames if cache_results else None,
            state=final_state,
        )
