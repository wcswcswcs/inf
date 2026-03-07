# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any
from collections import defaultdict

from streamvggt.layers import PatchEmbed
from streamvggt.layers.block import Block
from streamvggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from streamvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        geo_conf_ema_alpha: float = 0.9,
        geo_var_ema_alpha: float = 0.9,
        geo_invisible_read_weight: float = 0.2,
        geo_bucket_quantile_target: float = 0.6,
        geo_bucket_quantile_min: float = 0.3,
        geo_bucket_quantile_max: float = 0.9,
        geo_anchor_conf_enter: float = 1.25,
        geo_anchor_conf_exit: float = 1.05,
        geo_anchor_min_support: float = 2.0,
        geo_anchor_max_pos_var: float = 0.05,
        geo_max_voxels: int = 200000,
        geo_anchor_voxel_budget: int = 4096,
        geo_anchor_read_quota: int = 2048,
        geo_local_budget_ratio: float = 0.75,
        geo_local_budget_cap_per_frame: int = 1369,
        geo_anchor_budget_ratio: float = 0.35,
        geo_local_coverage_grid: int = 4,
        geo_frame0_patch_cap: int = 1000000,
        geo_anchor_invisible_read_weight: float = 0.5,
        geo_max_old_frames_to_score: int = 24,
        geo_max_candidate_tokens: int = 15000,
        geo_selection_interval: int = 3,
        geo_anchor_refresh_interval: int = 4,
        geo_anchor_replace_ratio: float = 0.25,
        geo_anchor_min_ttl: int = 12,
        geo_bank_coarse_stride: int = 4,
        geo_bank_bucket_reserve: int = 1,
        geo_bank_trim_interval: int = 8,
        geo_trim_scan_ratio: float = 0.25,
        geo_trim_drop_factor: float = 4.0,
        geo_keyframe_interval: int = 24,
        geo_keyframe_max_count: int = 96,
        geo_keyframe_token_quota: int = 1024,
        geo_keyframe_frozen_per_frame: int = 96,
        geo_keyframe_level0_max: int = 8,
        geo_keyframe_level1_max: int = 8,
        geo_keyframe_level2_max: int = 8,
        geo_keyframe_time_bins: int = 4,
        geo_anchor_stable_ratio: float = 0.6,
        geo_stable_anchor_min_support: float = 6.0,
        geo_stable_anchor_max_pos_var: float = 0.02,
        geo_stable_anchor_min_conf: float = 1.1,
        geo_anchor_adaptive_age_decay: float = 0.05,
        geo_anchor_stable_age_decay: float = 0.005,
        geo_bank_inlier_drift2_thr: float = 0.02,
        geo_bank_trust_residual_thr: float = 0.05,
        geo_selection_low_trust_threshold: float = 0.35,
        geo_full_select_pose_delta: float = 0.15,
        geo_full_select_conf_drop: float = 0.15,
        geo_full_select_new_voxel_ratio: float = 0.25,
        geo_keyframe_protected_quota: int = 256,
        geo_landmark_per_keyframe: int = 256,
        geo_landmark_max_count: int = 8192,
        geo_landmark_token_quota: int = 512,
        geo_reference_max_count: int = 4096,
        geo_reference_token_quota: int = 256,
        geo_reference_min_overlap: int = 32,
        geo_reference_overlap_bonus: float = 0.25,
        geo_reference_drift_threshold: float = 0.08,
        geo_recovery_frames: int = 12,
        geo_recovery_ref_boost: float = 2.0,
        geo_stable_map_ratio: float = 0.4,
        geo_stable_read_budget_ratio: float = 0.2,
        geo_stable_invisible_quota_ratio: float = 0.1,
        geo_stable_topk_per_voxel: int = 2,
        geo_stable_keyframe_topk_per_frame: int = 2,
        geo_stable_overlap_low_threshold: int = 256,
        geo_overlap_retrieval_budget_ratio: float = 0.15,
        geo_overlap_retrieval_top_frames: int = 4,
        geo_stable_quality_visible_ratio_thr: float = 0.6,
        geo_stable_quality_overlap_thr: int = 128,
        geo_stable_quality_streak_thr: int = 3,
        geo_recovery_stable_ratio_scale: float = 0.5,
        geo_console_log_interval: int = 50,
        geo_warmup_frames: int = 8,
        geo_warmup_local_budget_ratio: float = 1.0,
        geo_frame_meta_recent_keep: int = 64,
        geo_layer_budget_cap: int = 8192,
        geo_reloc_frames: int = 8,
        geo_reloc_trigger_overlap: int = 128,
        geo_reloc_trigger_visible_ratio: float = 0.75,
        geo_reloc_local_budget_ratio: float = 0.1,
        geo_reloc_stable_read_budget_ratio: float = 0.4,
        geo_bootstrap_frames: int = 64,
        geo_bootstrap_min_voxels: int = 4096,
        geo_bootstrap_min_refs: int = 64,
        geo_prune_start_ratio: float = 0.9,
        geo_reloc_hard_frames: int = 2,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )
        self.last_scores = torch.zeros(self.depth)
        self.geo_conf_ema_alpha = geo_conf_ema_alpha
        self.geo_var_ema_alpha = geo_var_ema_alpha
        self.geo_invisible_read_weight = geo_invisible_read_weight
        self.geo_bucket_quantile_target = geo_bucket_quantile_target
        self.geo_bucket_quantile_min = geo_bucket_quantile_min
        self.geo_bucket_quantile_max = geo_bucket_quantile_max
        self.geo_anchor_conf_enter = geo_anchor_conf_enter
        self.geo_anchor_conf_exit = geo_anchor_conf_exit
        self.geo_anchor_min_support = geo_anchor_min_support
        self.geo_anchor_max_pos_var = geo_anchor_max_pos_var
        self.geo_max_voxels = geo_max_voxels
        self.geo_anchor_voxel_budget = geo_anchor_voxel_budget
        self.geo_anchor_read_quota = geo_anchor_read_quota
        self.geo_local_budget_ratio = geo_local_budget_ratio
        self.geo_local_budget_cap_per_frame = geo_local_budget_cap_per_frame
        self.geo_anchor_budget_ratio = geo_anchor_budget_ratio
        self.geo_local_coverage_grid = geo_local_coverage_grid
        self.geo_frame0_patch_cap = geo_frame0_patch_cap
        self.geo_anchor_invisible_read_weight = geo_anchor_invisible_read_weight
        self.geo_max_old_frames_to_score = geo_max_old_frames_to_score
        self.geo_max_candidate_tokens = geo_max_candidate_tokens
        self.geo_selection_interval = geo_selection_interval
        self.geo_anchor_refresh_interval = geo_anchor_refresh_interval
        self.geo_anchor_replace_ratio = geo_anchor_replace_ratio
        self.geo_anchor_min_ttl = geo_anchor_min_ttl
        self.geo_bank_coarse_stride = max(1, int(geo_bank_coarse_stride))
        self.geo_bank_bucket_reserve = max(0, int(geo_bank_bucket_reserve))
        self.geo_bank_trim_interval = max(1, int(geo_bank_trim_interval))
        self.geo_trim_scan_ratio = float(min(max(geo_trim_scan_ratio, 0.05), 1.0))
        self.geo_trim_drop_factor = max(1.0, float(geo_trim_drop_factor))
        self.geo_keyframe_interval = max(1, int(geo_keyframe_interval))
        self.geo_keyframe_max_count = max(1, int(geo_keyframe_max_count))
        self.geo_keyframe_token_quota = max(0, int(geo_keyframe_token_quota))
        self.geo_keyframe_frozen_per_frame = max(1, int(geo_keyframe_frozen_per_frame))
        self.geo_keyframe_level0_max = max(1, int(geo_keyframe_level0_max))
        self.geo_keyframe_level1_max = max(1, int(geo_keyframe_level1_max))
        self.geo_keyframe_level2_max = max(1, int(geo_keyframe_level2_max))
        self.geo_keyframe_time_bins = max(1, int(geo_keyframe_time_bins))
        self.geo_anchor_stable_ratio = float(min(max(geo_anchor_stable_ratio, 0.0), 1.0))
        self.geo_stable_anchor_min_support = max(1.0, float(geo_stable_anchor_min_support))
        self.geo_stable_anchor_max_pos_var = max(0.0, float(geo_stable_anchor_max_pos_var))
        self.geo_stable_anchor_min_conf = max(0.0, float(geo_stable_anchor_min_conf))
        self.geo_anchor_adaptive_age_decay = max(0.0, float(geo_anchor_adaptive_age_decay))
        self.geo_anchor_stable_age_decay = max(0.0, float(geo_anchor_stable_age_decay))
        self.geo_bank_inlier_drift2_thr = max(1e-8, float(geo_bank_inlier_drift2_thr))
        self.geo_bank_trust_residual_thr = max(1e-8, float(geo_bank_trust_residual_thr))
        self.geo_selection_low_trust_threshold = float(min(max(geo_selection_low_trust_threshold, 0.0), 1.0))
        self.geo_full_select_pose_delta = max(0.0, float(geo_full_select_pose_delta))
        self.geo_full_select_conf_drop = max(0.0, float(geo_full_select_conf_drop))
        self.geo_full_select_new_voxel_ratio = max(0.0, float(geo_full_select_new_voxel_ratio))
        self.geo_keyframe_protected_quota = max(0, int(geo_keyframe_protected_quota))
        self.geo_landmark_per_keyframe = max(0, int(geo_landmark_per_keyframe))
        self.geo_landmark_max_count = max(1, int(geo_landmark_max_count))
        self.geo_landmark_token_quota = max(0, int(geo_landmark_token_quota))
        self.geo_reference_max_count = max(1, int(geo_reference_max_count))
        self.geo_reference_token_quota = max(0, int(geo_reference_token_quota))
        self.geo_reference_min_overlap = max(1, int(geo_reference_min_overlap))
        self.geo_reference_overlap_bonus = max(0.0, float(geo_reference_overlap_bonus))
        self.geo_reference_drift_threshold = max(1e-8, float(geo_reference_drift_threshold))
        self.geo_recovery_frames = max(1, int(geo_recovery_frames))
        self.geo_recovery_ref_boost = max(1.0, float(geo_recovery_ref_boost))
        self.geo_stable_map_ratio = float(min(max(geo_stable_map_ratio, 0.05), 0.9))
        self.geo_stable_read_budget_ratio = float(min(max(geo_stable_read_budget_ratio, 0.0), 0.8))
        self.geo_stable_invisible_quota_ratio = float(min(max(geo_stable_invisible_quota_ratio, 0.0), 1.0))
        self.geo_stable_topk_per_voxel = max(1, int(geo_stable_topk_per_voxel))
        self.geo_stable_keyframe_topk_per_frame = max(1, int(geo_stable_keyframe_topk_per_frame))
        self.geo_stable_min_voxels = max(64, int(self.geo_anchor_voxel_budget * 0.25))
        self.geo_stable_overlap_low_threshold = max(0, int(geo_stable_overlap_low_threshold))
        self.geo_overlap_retrieval_budget_ratio = float(min(max(geo_overlap_retrieval_budget_ratio, 0.0), 1.0))
        self.geo_overlap_retrieval_top_frames = max(1, int(geo_overlap_retrieval_top_frames))
        self.geo_stable_quality_visible_ratio_thr = float(min(max(geo_stable_quality_visible_ratio_thr, 0.0), 1.0))
        self.geo_stable_quality_overlap_thr = max(0, int(geo_stable_quality_overlap_thr))
        self.geo_stable_quality_streak_thr = max(1, int(geo_stable_quality_streak_thr))
        self.geo_recovery_stable_ratio_scale = float(min(max(geo_recovery_stable_ratio_scale, 0.0), 1.0))
        self.geo_console_log_interval = int(geo_console_log_interval)
        self.geo_warmup_frames = max(0, int(geo_warmup_frames))
        self.geo_warmup_local_budget_ratio = float(min(max(geo_warmup_local_budget_ratio, 0.0), 1.0))
        self.geo_frame_meta_recent_keep = max(8, int(geo_frame_meta_recent_keep))
        self.geo_layer_budget_cap = max(512, int(geo_layer_budget_cap))
        self.geo_reloc_frames = max(1, int(geo_reloc_frames))
        self.geo_reloc_trigger_overlap = max(0, int(geo_reloc_trigger_overlap))
        self.geo_reloc_trigger_visible_ratio = float(min(max(geo_reloc_trigger_visible_ratio, 0.0), 1.0))
        self.geo_reloc_local_budget_ratio = float(min(max(geo_reloc_local_budget_ratio, 0.0), 1.0))
        self.geo_reloc_stable_read_budget_ratio = float(min(max(geo_reloc_stable_read_budget_ratio, 0.0), 1.0))
        self.geo_bootstrap_frames = max(0, int(geo_bootstrap_frames))
        self.geo_bootstrap_min_voxels = max(1, int(geo_bootstrap_min_voxels))
        self.geo_bootstrap_min_refs = max(1, int(geo_bootstrap_min_refs))
        self.geo_prune_start_ratio = float(min(max(geo_prune_start_ratio, 0.0), 1.0))
        self.geo_reloc_hard_frames = max(1, int(geo_reloc_hard_frames))
        self.geo_identity_stride = 1 << 21
        self.geo_identity_offset = 1 << 18
        self.reset_geo_cache_state()

    def reset_geo_cache_state(self):
        self.geo_frame_meta: Dict[int, Dict[str, Any]] = {}
        self.geo_frame_anchor_mask: Dict[int, torch.Tensor] = {}
        self.geo_max_frame_idx = -1
        self.geo_voxel_bank: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        self.geo_anchor_voxels: set[Tuple[int, int, int]] = set()
        self.geo_anchor_voxel_list: List[Tuple[int, int, int]] = []
        self.geo_anchor_birth: Dict[Tuple[int, int, int], int] = {}
        self.geo_anchor_hash_tensor = torch.empty((0,), dtype=torch.long)
        self.geo_stable_anchor_voxels: set[Tuple[int, int, int]] = set()
        self.geo_stable_anchor_voxel_list: List[Tuple[int, int, int]] = []
        self.geo_stable_anchor_hash_tensor = torch.empty((0,), dtype=torch.long)
        self.geo_trust_score: float = 1.0
        self.geo_landmark_voxels: set[Tuple[int, int, int]] = set()
        self.geo_landmark_voxel_list: List[Tuple[int, int, int]] = []
        self.geo_landmark_birth: Dict[Tuple[int, int, int], int] = {}
        self.geo_landmark_hash_tensor = torch.empty((0,), dtype=torch.long)
        self.geo_reference_bank: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        self.geo_reference_voxels: set[Tuple[int, int, int]] = set()
        self.geo_reference_voxel_list: List[Tuple[int, int, int]] = []
        self.geo_reference_hash_tensor = torch.empty((0,), dtype=torch.long)
        self.geo_recovery_frames_left: int = 0
        self.geo_stable_map_voxels: set[Tuple[int, int, int]] = set()
        self.geo_adaptive_map_voxels: set[Tuple[int, int, int]] = set()
        self.geo_bad_stable_quality_streak: int = 0
        self.geo_reloc_frames_left: int = 0
        self.geo_reloc_state: str = "off"
        self.geo_reloc_hard_left: int = 0
        self.geo_reloc_good_streak: int = 0
        self.geo_cached_landmark_keep: torch.Tensor = torch.empty((0,), dtype=torch.long)
        self.geo_trim_cursor = 0
        self.geo_last_console_log_frame = -1
        self.geo_anchor_version = 0
        self.geo_frame_anchor_version: Dict[int, int] = {}
        self.geo_keyframes: List[int] = []
        self.geo_keyframe_set: set[int] = set()
        self.geo_keyframe_frozen_local: Dict[int, torch.Tensor] = {}
        self.geo_keyframe_pyramid: Dict[int, List[int]] = {0: [], 1: [], 2: []}
        self.geo_token_meta: Dict[int, Dict[str, torch.Tensor]] = {
            i: {
                "frame_idx": torch.empty(0, dtype=torch.long),
                "is_special": torch.empty(0, dtype=torch.bool),
                "is_keyframe": torch.empty(0, dtype=torch.bool),
                "local_patch_idx": torch.empty(0, dtype=torch.long),
                "identity_local": torch.empty(0, dtype=torch.long),
                "global_id": torch.empty(0, dtype=torch.long),
                "is_anchor": torch.empty(0, dtype=torch.bool),
                "is_landmark": torch.empty(0, dtype=torch.bool),
                "is_reference": torch.empty(0, dtype=torch.bool),
            }
            for i in range(self.depth)
        }

    def _update_geo_keyframes(self, frame_idx: int):
        frame_idx = int(frame_idx)
        should_add = (frame_idx == 0) or (frame_idx % int(self.geo_keyframe_interval) == 0)
        if not should_add or frame_idx in self.geo_keyframe_set:
            return
        self.geo_keyframes.append(frame_idx)
        self.geo_keyframe_set.add(frame_idx)

        max_count = int(self.geo_keyframe_max_count)
        while len(self.geo_keyframes) > max_count:
            drop = self.geo_keyframes.pop(0)
            if drop == 0:
                self.geo_keyframes.insert(0, drop)
                break
            self.geo_keyframe_set.discard(drop)
            self.geo_keyframe_frozen_local.pop(int(drop), None)
            for lvl in self.geo_keyframe_pyramid.values():
                if int(drop) in lvl:
                    lvl[:] = [v for v in lvl if int(v) != int(drop)]

    def _insert_keyframe_pyramid(self, frame_idx: int):
        f = int(frame_idx)
        if f in self.geo_keyframe_pyramid[0] or f in self.geo_keyframe_pyramid[1] or f in self.geo_keyframe_pyramid[2]:
            return
        self.geo_keyframe_pyramid[0].append(f)
        if len(self.geo_keyframe_pyramid[0]) > int(self.geo_keyframe_level0_max):
            moved = self.geo_keyframe_pyramid[0].pop(0)
            self.geo_keyframe_pyramid[1].append(moved)
        if len(self.geo_keyframe_pyramid[1]) > int(self.geo_keyframe_level1_max):
            moved = self.geo_keyframe_pyramid[1].pop(0)
            self.geo_keyframe_pyramid[2].append(moved)
        if len(self.geo_keyframe_pyramid[2]) > int(self.geo_keyframe_level2_max):
            self.geo_keyframe_pyramid[2].pop(0)

    def _freeze_keyframe_tokens(self, frame_idx: int, conf_flat: torch.Tensor):
        frame_idx = int(frame_idx)
        if frame_idx not in self.geo_keyframe_set:
            return
        if frame_idx in self.geo_keyframe_frozen_local:
            return
        if conf_flat is None or conf_flat.numel() == 0:
            self.geo_keyframe_frozen_local[frame_idx] = torch.empty((0,), dtype=torch.long)
            return

        quota = min(int(self.geo_keyframe_frozen_per_frame), int(conf_flat.numel()))
        if quota <= 0:
            self.geo_keyframe_frozen_local[frame_idx] = torch.empty((0,), dtype=torch.long)
            return

        side = max(1, int(round(float(conf_flat.numel()) ** 0.5)))
        grid_n = max(1, int(self.geo_local_coverage_grid))
        bin_h = max(1, side // grid_n)
        bin_w = max(1, side // grid_n)
        best_per_cell: Dict[Tuple[int, int], Tuple[float, int]] = {}
        for lp in range(int(conf_flat.numel())):
            y, x = lp // side, lp % side
            cell = (min(grid_n - 1, y // bin_h), min(grid_n - 1, x // bin_w))
            sc = float(conf_flat[lp].item())
            prev = best_per_cell.get(cell)
            if prev is None or sc > prev[0]:
                best_per_cell[cell] = (sc, lp)

        chosen = [idx for _, idx in best_per_cell.values()]
        if len(chosen) < quota:
            order = torch.argsort(conf_flat.to(torch.float32), descending=True)
            for lp in order.tolist():
                if lp not in chosen:
                    chosen.append(int(lp))
                if len(chosen) >= quota:
                    break
        else:
            chosen = chosen[:quota]

        frozen = torch.unique(torch.tensor(chosen, dtype=torch.long), sorted=True)
        self.geo_keyframe_frozen_local[frame_idx] = frozen
        self._insert_keyframe_pyramid(frame_idx)

    def _voxel_importance(self, item: Dict[str, float], now_frame_idx: int, age_decay: Optional[float] = None) -> float:
        age = max(0.0, float(now_frame_idx) - float(item["last_seen"]))
        decay = self.geo_anchor_adaptive_age_decay if age_decay is None else max(0.0, float(age_decay))
        return (
            float(item["conf_ema"])
            * torch.log1p(torch.tensor(float(item["support"]))).item()
            * (1.0 / (1.0 + float(item["pos_var"])))
            * (1.0 / (1.0 + decay * age))
        )

    def _update_landmarks_from_keyframe(
        self,
        frame_idx: int,
        uniq_vox: torch.Tensor,
        conf_mean_all: torch.Tensor,
    ):
        if int(self.geo_landmark_per_keyframe) <= 0 or uniq_vox.numel() == 0:
            return
        if int(frame_idx) not in self.geo_keyframe_set:
            return

        quota = min(int(self.geo_landmark_per_keyframe), int(uniq_vox.shape[0]))
        if quota <= 0:
            return

        # spatially-diverse selection: one winner per coarse bucket, then fill by confidence.
        coarse_s = int(self.geo_bank_coarse_stride)
        best_by_bucket: Dict[Tuple[int, int, int], Tuple[float, Tuple[int, int, int]]] = {}
        for i in range(int(uniq_vox.shape[0])):
            key = tuple(int(v) for v in uniq_vox[i].tolist())
            coarse = (key[0] // coarse_s, key[1] // coarse_s, key[2] // coarse_s)
            sc = float(conf_mean_all[i].item())
            prev = best_by_bucket.get(coarse)
            if prev is None or sc > prev[0]:
                best_by_bucket[coarse] = (sc, key)

        chosen: List[Tuple[int, int, int]] = [v[1] for v in best_by_bucket.values()]
        if len(chosen) < quota:
            order = torch.argsort(conf_mean_all, descending=True)
            for i in order.tolist():
                key = tuple(int(v) for v in uniq_vox[i].tolist())
                if key not in self.geo_landmark_voxels and key not in chosen:
                    chosen.append(key)
                if len(chosen) >= quota:
                    break
        else:
            chosen = chosen[:quota]

        for key in chosen:
            self.geo_landmark_voxels.add(key)
            if key not in self.geo_landmark_birth:
                self.geo_landmark_birth[key] = int(frame_idx)

        if len(self.geo_landmark_voxels) > int(self.geo_landmark_max_count):
            # keep frame0-era and newest landmarks when overflowing
            items = sorted(self.geo_landmark_voxels, key=lambda k: int(self.geo_landmark_birth.get(k, frame_idx)))
            keep = set(items[-int(self.geo_landmark_max_count):])
            keep.update(k for k, b in self.geo_landmark_birth.items() if int(b) == 0)
            while len(keep) > int(self.geo_landmark_max_count):
                keep.remove(next(iter(keep)))
            self.geo_landmark_voxels = keep
            self.geo_landmark_birth = {k: self.geo_landmark_birth.get(k, int(frame_idx)) for k in keep}

        self.geo_landmark_voxel_list = sorted(self.geo_landmark_voxels)
        if self.geo_landmark_voxel_list:
            vox = torch.tensor(self.geo_landmark_voxel_list, dtype=torch.long)
            self.geo_landmark_hash_tensor = self._voxel_hash(vox)
        else:
            self.geo_landmark_hash_tensor = torch.empty((0,), dtype=torch.long)

    def _update_reference_bank_from_keyframe(
        self,
        frame_idx: int,
        uniq_vox: torch.Tensor,
        conf_mean_all: torch.Tensor,
    ):
        if int(frame_idx) not in self.geo_keyframe_set:
            return
        if uniq_vox.numel() == 0:
            return

        for i in range(int(uniq_vox.shape[0])):
            key = tuple(int(v) for v in uniq_vox[i].tolist())
            if key in self.geo_reference_bank:
                continue
            bank = self.geo_voxel_bank.get(key)
            if bank is None:
                continue
            if float(bank.get("support", 0.0)) < self.geo_stable_anchor_min_support:
                continue
            if float(bank.get("pos_var", 1e9)) > self.geo_stable_anchor_max_pos_var:
                continue
            score = float(conf_mean_all[i].item()) * float(bank.get("conf_ema", 0.0))
            self.geo_reference_bank[key] = {
                "pos_x": float(bank["pos_x"]),
                "pos_y": float(bank["pos_y"]),
                "pos_z": float(bank["pos_z"]),
                "score": float(score),
                "created_frame": float(frame_idx),
            }

        if len(self.geo_reference_bank) > int(self.geo_reference_max_count):
            ranked = sorted(
                self.geo_reference_bank.items(),
                key=lambda kv: (float(kv[1].get("score", 0.0)), float(kv[1].get("created_frame", 0.0))),
            )
            drop_n = len(self.geo_reference_bank) - int(self.geo_reference_max_count)
            for key, _ in ranked[:drop_n]:
                self.geo_reference_bank.pop(key, None)

        self.geo_reference_voxels = set(self.geo_reference_bank.keys())
        self.geo_reference_voxel_list = sorted(self.geo_reference_voxels)
        if self.geo_reference_voxel_list:
            vox = torch.tensor(self.geo_reference_voxel_list, dtype=torch.long)
            self.geo_reference_hash_tensor = self._voxel_hash(vox)
        else:
            self.geo_reference_hash_tensor = torch.empty((0,), dtype=torch.long)

    def _active_frames_for_anchor_refresh(self, current_frame_idx: int) -> set[int]:
        current_frame_idx = int(current_frame_idx)
        must_keep: set[int] = {0, current_frame_idx}
        must_keep.update(int(v) for v in self.geo_keyframes)

        kv_frames: set[int] = set()
        for meta in self.geo_token_meta.values():
            fi = meta.get("frame_idx")
            if fi is None or fi.numel() == 0:
                continue
            vals = torch.unique(fi).detach().cpu().tolist()
            kv_frames.update(int(v) for v in vals)

        must_keep.update(kv_frames)

        # Optional extras: only keep a bounded recent tail of non-critical history.
        optional = sorted(k for k in self.geo_frame_meta.keys() if int(k) not in must_keep)
        if self.geo_max_old_frames_to_score > 0 and len(optional) > self.geo_max_old_frames_to_score:
            optional = optional[-self.geo_max_old_frames_to_score :]

        active = set(must_keep)
        active.update(int(v) for v in optional)
        return active

    def _update_stable_anchor_voxels(self):
        ranked: List[Tuple[float, Tuple[int, int, int]]] = []
        for key, item in self.geo_voxel_bank.items():
            if float(item.get("support", 0.0)) < self.geo_stable_anchor_min_support:
                continue
            if float(item.get("pos_var", 1e9)) > self.geo_stable_anchor_max_pos_var:
                continue
            if float(item.get("conf_ema", 0.0)) < self.geo_stable_anchor_min_conf:
                continue
            if int(item.get("outlier_count", 0)) > 0:
                continue
            score = float(item["conf_ema"]) * torch.log1p(torch.tensor(float(item["support"]))).item() / (1.0 + float(item["pos_var"]))
            ranked.append((score, key))

        cap = max(0, int(int(self.geo_anchor_voxel_budget) * float(self.geo_anchor_stable_ratio)))
        if cap <= 0 or not ranked:
            self.geo_stable_anchor_voxels = set()
            self.geo_stable_anchor_voxel_list = []
            self.geo_stable_anchor_hash_tensor = torch.empty((0,), dtype=torch.long)
            return

        ranked.sort(key=lambda x: (-x[0], x[1]))
        self.geo_stable_anchor_voxel_list = [k for _, k in ranked[:cap]]
        self.geo_stable_anchor_voxels = set(self.geo_stable_anchor_voxel_list)
        vox = torch.tensor(self.geo_stable_anchor_voxel_list, dtype=torch.long)
        self.geo_stable_anchor_hash_tensor = self._voxel_hash(vox)

    def _is_stable_voxel(self, item: Dict[str, float]) -> bool:
        return (
            float(item.get("support", 0.0)) >= float(self.geo_stable_anchor_min_support)
            and float(item.get("pos_var", 1e9)) <= float(self.geo_stable_anchor_max_pos_var)
            and float(item.get("conf_ema", 0.0)) >= float(self.geo_stable_anchor_min_conf)
            and int(item.get("outlier_count", 0)) <= 0
        )

    def _rebuild_stable_adaptive_maps(self, now_frame_idx: Optional[int] = None):
        stable = set()
        adaptive = set()
        scored: List[Tuple[float, Tuple[int, int, int]]] = []
        for key, item in self.geo_voxel_bank.items():
            support = float(item.get("support", 0.0))
            conf_ema = float(item.get("conf_ema", 0.0))
            pos_var = float(item.get("pos_var", 1e9))
            score = conf_ema * float(torch.log1p(torch.tensor(support)).item()) / (1.0 + pos_var)
            scored.append((score, key))
            if self._is_stable_voxel(item):
                stable.add(key)
            else:
                adaptive.add(key)

        now_idx = int(self.geo_max_frame_idx if now_frame_idx is None else now_frame_idx)
        stable_target = int(round(float(self.geo_max_voxels) * float(self.geo_stable_map_ratio)))
        stable_target = max(stable_target, len(self.geo_stable_anchor_voxel_list), int(self.geo_stable_min_voxels))
        stable_target = min(max(0, stable_target), len(scored))
        need = max(0, stable_target - len(stable))
        if need > 0 and adaptive:
            cand: List[Tuple[float, Tuple[int, int, int]]] = []
            for key in adaptive:
                item = self.geo_voxel_bank.get(key)
                if item is None:
                    continue
                if int(item.get("outlier_count", 0)) > 0:
                    continue
                cand.append((float(self._voxel_importance(item, now_idx)), key))
            cand.sort(key=lambda x: (-x[0], x[1]))
            for _, key in cand[:need]:
                if key in adaptive:
                    adaptive.remove(key)
                    stable.add(key)

        self.geo_stable_map_voxels = stable
        self.geo_adaptive_map_voxels = adaptive

    def _prune_geo_frame_meta(self, current_frame_idx: int):
        if not self.geo_frame_meta:
            return

        keep: set[int] = {0, int(current_frame_idx)}
        recent_min = max(0, int(current_frame_idx) - int(self.geo_frame_meta_recent_keep))
        keep.update(int(f) for f in self.geo_frame_meta.keys() if int(f) >= recent_min)
        keep.update(int(f) for f in self.geo_keyframes)
        for lvl in self.geo_keyframe_pyramid.values():
            keep.update(int(f) for f in lvl)

        for meta in self.geo_token_meta.values():
            fi = meta.get("frame_idx")
            if fi is None or fi.numel() == 0:
                continue
            keep.update(int(v) for v in torch.unique(fi).detach().cpu().tolist())

        drop = [int(f) for f in self.geo_frame_meta.keys() if int(f) not in keep]
        for f in drop:
            self.geo_frame_meta.pop(f, None)
            self.geo_frame_anchor_mask.pop(f, None)
            self.geo_frame_anchor_version.pop(f, None)

        stale_frozen = [int(f) for f in self.geo_keyframe_frozen_local.keys() if int(f) not in keep]
        for f in stale_frozen:
            self.geo_keyframe_frozen_local.pop(f, None)

    def _refresh_geo_anchor_voxels(self, now_frame_idx: int):
        if not self.geo_voxel_bank:
            self.geo_anchor_voxels = set()
            self.geo_anchor_voxel_list = []
            self.geo_anchor_birth = {}
            self.geo_anchor_hash_tensor = torch.empty((0,), dtype=torch.long)
            self.geo_anchor_version += 1
            return

        prev_anchors = self.geo_anchor_voxels
        self._update_stable_anchor_voxels()
        stable_ranked = []
        adaptive_ranked = []
        for key, item in self.geo_voxel_bank.items():
            conf_ema = float(item["conf_ema"])
            support = float(item["support"])
            pos_var = float(item["pos_var"])
            in_prev = key in prev_anchors
            conf_ok = conf_ema >= (self.geo_anchor_conf_exit if in_prev else self.geo_anchor_conf_enter)
            support_ok = support >= self.geo_anchor_min_support
            var_ok = pos_var <= self.geo_anchor_max_pos_var
            if conf_ok and support_ok and var_ok:
                stable_ranked.append((self._voxel_importance(item, now_frame_idx, self.geo_anchor_stable_age_decay), key))
                adaptive_ranked.append((self._voxel_importance(item, now_frame_idx, self.geo_anchor_adaptive_age_decay), key))

        # Avoid full sort on the whole bank: keep only a bounded top candidate set.
        candidate_pool = max(int(self.geo_anchor_voxel_budget) * 8, int(self.geo_anchor_voxel_budget) + 1024)
        budget = int(self.geo_anchor_voxel_budget)
        stable_budget = min(budget, max(0, int(round(budget * self.geo_anchor_stable_ratio))))
        adaptive_budget = max(0, budget - stable_budget)

        stable_candidates = heapq.nlargest(candidate_pool, stable_ranked, key=lambda x: x[0])
        adaptive_candidates = heapq.nlargest(candidate_pool, adaptive_ranked, key=lambda x: x[0])
        stable_candidates.sort(key=lambda x: (-x[0], x[1]))
        adaptive_candidates.sort(key=lambda x: (-x[0], x[1]))

        candidate: List[Tuple[int, int, int]] = []
        seen_candidate: set[Tuple[int, int, int]] = set()
        for _, key in stable_candidates:
            if key in seen_candidate:
                continue
            candidate.append(key)
            seen_candidate.add(key)
            if len(candidate) >= stable_budget:
                break

        if adaptive_budget > 0:
            for _, key in adaptive_candidates:
                if key in seen_candidate:
                    continue
                candidate.append(key)
                seen_candidate.add(key)
                if len(candidate) >= budget:
                    break

        if len(candidate) < budget:
            for _, key in stable_candidates:
                if key in seen_candidate:
                    continue
                candidate.append(key)
                seen_candidate.add(key)
                if len(candidate) >= budget:
                    break

        if budget <= 0:
            self.geo_anchor_voxel_list = []
            self.geo_anchor_voxels = set()
            self.geo_anchor_birth = {}
        else:
            prev = list(self.geo_anchor_voxel_list)
            prev_set = set(prev)
            protected_prev = []
            for k in prev:
                birth = int(self.geo_anchor_birth.get(k, now_frame_idx))
                age = max(0, int(now_frame_idx) - birth)
                if age < int(self.geo_anchor_min_ttl):
                    protected_prev.append(k)

            max_replace = max(1, int(budget * float(self.geo_anchor_replace_ratio)))
            max_replace = min(max_replace, budget)

            selected: List[Tuple[int, int, int]] = []
            selected_set: set[Tuple[int, int, int]] = set()

            for k in self.geo_stable_anchor_voxel_list:
                if len(selected) >= budget:
                    break
                if k in self.geo_voxel_bank and k not in selected_set:
                    selected.append(k)
                    selected_set.add(k)

            for k in protected_prev:
                if len(selected) >= budget:
                    break
                if k in prev_set and k in self.geo_voxel_bank and k not in selected_set:
                    selected.append(k)
                    selected_set.add(k)

            for k in prev:
                if len(selected) >= budget:
                    break
                if k in self.geo_voxel_bank and k not in selected_set:
                    selected.append(k)
                    selected_set.add(k)

            replace_used = 0
            for k in candidate:
                if len(selected) >= budget:
                    break
                if k in selected_set:
                    continue
                if k not in prev_set:
                    if replace_used >= max_replace:
                        continue
                    replace_used += 1
                selected.append(k)
                selected_set.add(k)

            if len(selected) < budget:
                for k in candidate:
                    if len(selected) >= budget:
                        break
                    if k in selected_set:
                        continue
                    selected.append(k)
                    selected_set.add(k)

            self.geo_anchor_voxel_list = selected[:budget]
            self.geo_anchor_voxels = set(self.geo_anchor_voxel_list)

            new_birth: Dict[Tuple[int, int, int], int] = {}
            for k in self.geo_anchor_voxel_list:
                if k in self.geo_anchor_birth:
                    new_birth[k] = self.geo_anchor_birth[k]
                else:
                    new_birth[k] = int(now_frame_idx)
            self.geo_anchor_birth = new_birth

        if self.geo_anchor_voxel_list:
            vox = torch.tensor(self.geo_anchor_voxel_list, dtype=torch.long)
            self.geo_anchor_hash_tensor = self._voxel_hash(vox)
        else:
            self.geo_anchor_hash_tensor = torch.empty((0,), dtype=torch.long)
        self.geo_anchor_version += 1

    @staticmethod
    def _voxel_hash(voxel_ids: torch.Tensor) -> torch.Tensor:
        if voxel_ids.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        v = voxel_ids.to(torch.long)
        return (v[:, 0] * 73856093) ^ (v[:, 1] * 19349663) ^ (v[:, 2] * 83492791)

    def _update_frame_anchor_mask(self, frame_idx: int):
        frame_meta = self.geo_frame_meta.get(int(frame_idx))
        if frame_meta is None:
            return
        vox = frame_meta.get("voxel_ids")
        if vox is None or vox.numel() == 0 or self.geo_anchor_hash_tensor.numel() == 0:
            self.geo_frame_anchor_mask[int(frame_idx)] = torch.zeros((0 if vox is None else vox.shape[0],), dtype=torch.bool)
            self.geo_frame_anchor_version[int(frame_idx)] = int(self.geo_anchor_version)
            return
        hashes = self._voxel_hash(vox)
        self.geo_frame_anchor_mask[int(frame_idx)] = torch.isin(hashes, self.geo_anchor_hash_tensor)
        self.geo_frame_anchor_version[int(frame_idx)] = int(self.geo_anchor_version)

    def _get_frame_anchor_mask(self, frame_idx: int) -> Optional[torch.Tensor]:
        frame_idx = int(frame_idx)
        if frame_idx not in self.geo_frame_meta:
            return None
        if self.geo_frame_anchor_version.get(frame_idx, -1) != int(self.geo_anchor_version):
            self._update_frame_anchor_mask(frame_idx)
        return self.geo_frame_anchor_mask.get(frame_idx)

    def _compute_dynamic_bucket_threshold(
        self,
        values: List[float],
        target_count: int,
    ) -> float:
        if not values:
            return float("inf")
        vals = torch.tensor(values, dtype=torch.float32)
        q_target = float(self.geo_bucket_quantile_target)
        q_target = min(max(q_target, self.geo_bucket_quantile_min), self.geo_bucket_quantile_max)

        if target_count > 0:
            n = vals.numel()
            ratio = max(0.0, min(1.0, float(target_count) / max(1.0, float(n))))
            q_target = 1.0 - ratio
            q_target = min(max(q_target, self.geo_bucket_quantile_min), self.geo_bucket_quantile_max)

        return float(torch.quantile(vals, q_target).item())

    def _derive_anchor_mask_from_meta(self, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n == 0:
            return torch.empty(0, dtype=torch.bool)

        out = torch.zeros((n,), dtype=torch.bool)
        if not self.geo_anchor_voxels:
            return out

        frame_idx = meta["frame_idx"]
        local_idx = meta["local_patch_idx"]
        valid = local_idx >= 0
        if valid.sum().item() == 0:
            return out

        for fidx in torch.unique(frame_idx[valid]).tolist():
            fidx = int(fidx)
            frame_anchor_mask = self._get_frame_anchor_mask(fidx)
            if frame_anchor_mask is None or frame_anchor_mask.numel() == 0:
                continue

            mask_f = valid & (frame_idx == fidx)
            idx_global = torch.nonzero(mask_f, as_tuple=False).flatten()
            local_f = local_idx.index_select(0, idx_global).long()
            in_range = (local_f >= 0) & (local_f < frame_anchor_mask.shape[0])
            if in_range.sum().item() == 0:
                continue

            idx_global = idx_global[in_range]
            local_f = local_f[in_range]
            anchor_mask = frame_anchor_mask.index_select(0, local_f)
            if anchor_mask.numel() > 0:
                out.index_put_((idx_global,), anchor_mask)

        return out

    def _derive_landmark_mask_from_meta(self, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n == 0:
            return torch.empty(0, dtype=torch.bool)
        if self.geo_landmark_hash_tensor.numel() == 0:
            return torch.zeros((n,), dtype=torch.bool)

        out = torch.zeros((n,), dtype=torch.bool)
        frame_idx = meta["frame_idx"]
        local_idx = meta["local_patch_idx"]
        valid = local_idx >= 0
        if valid.sum().item() == 0:
            return out

        for fidx in torch.unique(frame_idx[valid]).tolist():
            fidx = int(fidx)
            frame_meta = self.geo_frame_meta.get(fidx)
            if frame_meta is None:
                continue
            vox = frame_meta.get("voxel_ids")
            if vox is None or vox.numel() == 0:
                continue

            mask_f = valid & (frame_idx == fidx)
            idx_global = torch.nonzero(mask_f, as_tuple=False).flatten()
            local_f = local_idx.index_select(0, idx_global).long()
            in_range = (local_f >= 0) & (local_f < vox.shape[0])
            if in_range.sum().item() == 0:
                continue

            idx_global = idx_global[in_range]
            local_f = local_f[in_range]
            hashes = self._voxel_hash(vox.index_select(0, local_f))
            lm_mask = torch.isin(hashes, self.geo_landmark_hash_tensor)
            if lm_mask.numel() > 0:
                out.index_put_((idx_global,), lm_mask)

        return out

    def _derive_reference_mask_from_meta(self, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n == 0:
            return torch.empty(0, dtype=torch.bool)
        if self.geo_reference_hash_tensor.numel() == 0:
            return torch.zeros((n,), dtype=torch.bool)

        out = torch.zeros((n,), dtype=torch.bool)
        frame_idx = meta["frame_idx"]
        local_idx = meta["local_patch_idx"]
        valid = local_idx >= 0
        if valid.sum().item() == 0:
            return out

        for fidx in torch.unique(frame_idx[valid]).tolist():
            fidx = int(fidx)
            frame_meta = self.geo_frame_meta.get(fidx)
            if frame_meta is None:
                continue
            vox = frame_meta.get("voxel_ids")
            if vox is None or vox.numel() == 0:
                continue

            mask_f = valid & (frame_idx == fidx)
            idx_global = torch.nonzero(mask_f, as_tuple=False).flatten()
            local_f = local_idx.index_select(0, idx_global).long()
            in_range = (local_f >= 0) & (local_f < vox.shape[0])
            if in_range.sum().item() == 0:
                continue

            idx_global = idx_global[in_range]
            local_f = local_f[in_range]
            hashes = self._voxel_hash(vox.index_select(0, local_f))
            ref_mask = torch.isin(hashes, self.geo_reference_hash_tensor)
            if ref_mask.numel() > 0:
                out.index_put_((idx_global,), ref_mask)

        return out

    def _geo_bank_ready(self, frame_idx: int) -> bool:
        return int(frame_idx) >= int(self.geo_bootstrap_frames) and len(self.geo_voxel_bank) >= int(self.geo_bootstrap_min_voxels)

    def _geo_ref_ready(self) -> bool:
        return len(self.geo_reference_bank) >= int(self.geo_bootstrap_min_refs)

    def _refresh_cached_keyframe_flags(self, frame_idx: int):
        frozen = self.geo_keyframe_frozen_local.get(int(frame_idx))
        if frozen is None or frozen.numel() == 0:
            return
        frozen = frozen.detach().cpu().long()
        for layer_idx in range(self.depth):
            meta = self.geo_token_meta[layer_idx]
            if meta["frame_idx"].numel() == 0:
                continue
            mask = (meta["frame_idx"] == int(frame_idx)) & (meta["local_patch_idx"] >= 0)
            if not mask.any():
                continue
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            local = meta["local_patch_idx"].index_select(0, idx).long()
            take = torch.isin(local, frozen)
            if take.any():
                meta["is_keyframe"][idx[take]] = True

    def update_geo_frame_metadata(
        self,
        frame_idx: int,
        pts3d: torch.Tensor,
        conf: torch.Tensor,
        world_to_cam: Optional[torch.Tensor],
        intrinsic: Optional[torch.Tensor],
        voxel_size: float,
    ) -> Dict[str, float]:
        if pts3d is None or conf is None:
            return {"new_voxel_ratio": 0.0, "trust_score": float(self.geo_trust_score), "matched_ratio": 0.0, "ref_overlap": 0.0}
        if pts3d.ndim != 4 or conf.ndim != 3:
            return {"new_voxel_ratio": 0.0, "trust_score": float(self.geo_trust_score), "matched_ratio": 0.0, "ref_overlap": 0.0}

        _, H, W, _ = pts3d.shape
        gh, gw = H // self.patch_size, W // self.patch_size
        if gh <= 0 or gw <= 0:
            return {"new_voxel_ratio": 0.0, "trust_score": float(self.geo_trust_score), "matched_ratio": 0.0, "ref_overlap": 0.0}

        pts_patch = F.adaptive_avg_pool2d(pts3d.permute(0, 3, 1, 2), (gh, gw)).permute(0, 2, 3, 1)
        conf_patch = F.adaptive_avg_pool2d(conf.unsqueeze(1), (gh, gw)).squeeze(1)

        pts_flat = pts_patch.reshape(-1, 3).detach().cpu()
        conf_flat = conf_patch.reshape(-1).detach().cpu()

        voxel_ids = torch.floor(pts_flat / max(voxel_size, 1e-6)).to(torch.int32)
        meta = {
            "pts": pts_flat,
            "conf": conf_flat,
            "voxel_ids": voxel_ids,
            "world_to_cam": world_to_cam.detach().cpu()
            if (world_to_cam is not None and world_to_cam.device.type != "cpu")
            else world_to_cam.detach()
            if world_to_cam is not None
            else None,
            "intrinsic": intrinsic.detach().cpu()
            if (intrinsic is not None and intrinsic.device.type != "cpu")
            else intrinsic.detach()
            if intrinsic is not None
            else None,
        }
        self.geo_frame_meta[frame_idx] = meta
        self.geo_frame_anchor_mask[frame_idx] = torch.zeros((voxel_ids.shape[0],), dtype=torch.bool)
        self.geo_frame_anchor_version[frame_idx] = -1
        self.geo_max_frame_idx = max(self.geo_max_frame_idx, frame_idx)
        self._update_geo_keyframes(frame_idx)
        self._freeze_keyframe_tokens(frame_idx, conf_flat)
        self._refresh_cached_keyframe_flags(frame_idx)

        # Update global voxel landmark bank (conf/support/stability/recency)
        uniq_vox, inverse, counts = torch.unique(
            voxel_ids.to(torch.long), dim=0, return_inverse=True, return_counts=True
        )
        num_groups = int(uniq_vox.shape[0])
        conf_sum = torch.zeros((num_groups,), dtype=torch.float32)
        conf_sum.index_add_(0, inverse, conf_flat.to(torch.float32))
        pts_sum = torch.zeros((num_groups, 3), dtype=torch.float32)
        pts_sum.index_add_(0, inverse, pts_flat.to(torch.float32))
        counts_f = counts.to(torch.float32).clamp_min(1.0)
        conf_mean_all = conf_sum / counts_f
        pos_mean_all = pts_sum / counts_f.unsqueeze(1)

        drift2_map: Dict[Tuple[int, int, int], float] = {}
        residuals: List[float] = []
        for g in range(num_groups):
            key = tuple(int(v) for v in uniq_vox[g].tolist())
            if key in self.geo_voxel_bank:
                item = self.geo_voxel_bank[key]
                pos_mean = pos_mean_all[g]
                prev_pos = torch.tensor([item["pos_x"], item["pos_y"], item["pos_z"]], dtype=pos_mean.dtype)
                drift2 = float(((pos_mean - prev_pos) ** 2).mean().item())
                drift2_map[key] = drift2
                residuals.append(drift2)

        matched_existing = int(len(residuals))
        matched_ratio = float(matched_existing) / float(max(1, num_groups))

        ref_residuals: List[float] = []
        for g in range(num_groups):
            key = tuple(int(v) for v in uniq_vox[g].tolist())
            ref = self.geo_reference_bank.get(key)
            if ref is None:
                continue
            pos_mean = pos_mean_all[g]
            ref_pos = torch.tensor([ref["pos_x"], ref["pos_y"], ref["pos_z"]], dtype=pos_mean.dtype)
            ref_residuals.append(float(((pos_mean - ref_pos) ** 2).mean().sqrt().item()))
        ref_overlap = int(len(ref_residuals))

        bank_ready = self._geo_bank_ready(frame_idx)
        ref_ready = self._geo_ref_ready()
        if not bank_ready:
            self.geo_trust_score = 1.0
            self.geo_recovery_frames_left = 0
            low_trust = False
            recovery_mode = False
            allow_new_voxels = True
            allow_promote_landmark = True
            allow_promote_reference = False
        else:
            if residuals:
                med = float(torch.tensor(residuals, dtype=torch.float32).median().item())
                residual_trust = float(max(0.0, min(1.0, 1.0 - (med / max(self.geo_bank_trust_residual_thr, 1e-8)))))
            else:
                residual_trust = 0.0
            overlap_trust = float(max(0.0, min(1.0, matched_ratio / 0.25)))
            self.geo_trust_score = float(min(residual_trust, overlap_trust))

            bad_overlap = matched_ratio < 0.05
            if ref_ready:
                bad_overlap = bad_overlap or (ref_overlap < max(4, int(self.geo_reference_min_overlap // 2)))
            if ref_overlap >= int(self.geo_reference_min_overlap):
                ref_med = float(torch.tensor(ref_residuals, dtype=torch.float32).median().item())
                if ref_med > float(self.geo_reference_drift_threshold):
                    self.geo_recovery_frames_left = int(self.geo_recovery_frames)
            if bad_overlap or self.geo_trust_score < float(self.geo_selection_low_trust_threshold):
                self.geo_recovery_frames_left = max(int(self.geo_recovery_frames_left), int(self.geo_recovery_frames))
            if self.geo_recovery_frames_left > 0:
                self.geo_recovery_frames_left = max(0, int(self.geo_recovery_frames_left) - 1)
            low_trust = self.geo_trust_score < float(self.geo_selection_low_trust_threshold)
            recovery_mode = self.geo_recovery_frames_left > 0
            allow_new_voxels = (not recovery_mode) and (not low_trust)
            allow_promote_landmark = (not recovery_mode)
            allow_promote_reference = (not recovery_mode) and (float(self.geo_trust_score) >= float(self.geo_selection_low_trust_threshold))

        if int(self.geo_reloc_frames_left) > 0 or str(self.geo_reloc_state) != "off":
            allow_new_voxels = False
            allow_promote_landmark = False
            allow_promote_reference = False

        new_voxels = 0
        for g in range(num_groups):
            key = tuple(int(v) for v in uniq_vox[g].tolist())
            conf_mean = float(conf_mean_all[g].item())
            pos_mean = pos_mean_all[g]

            if key not in self.geo_voxel_bank:
                if not allow_new_voxels:
                    continue
                new_voxels += 1
                self.geo_voxel_bank[key] = {
                    "conf_ema": conf_mean,
                    "support": 1.0,
                    "last_seen": float(frame_idx),
                    "pos_x": float(pos_mean[0].item()),
                    "pos_y": float(pos_mean[1].item()),
                    "pos_z": float(pos_mean[2].item()),
                    "pos_var": 0.0,
                    "outlier_count": 0,
                }
            else:
                item = self.geo_voxel_bank[key]
                drift2 = float(drift2_map.get(key, 0.0))
                if recovery_mode:
                    item["last_seen"] = float(frame_idx)
                    continue
                if low_trust and drift2 > float(self.geo_bank_inlier_drift2_thr):
                    item["outlier_count"] = int(item.get("outlier_count", 0)) + 1
                    item["last_seen"] = float(frame_idx)
                    continue

                item["conf_ema"] = (
                    self.geo_conf_ema_alpha * item["conf_ema"]
                    + (1.0 - self.geo_conf_ema_alpha) * conf_mean
                )
                item["support"] += 1.0
                item["last_seen"] = float(frame_idx)
                item["outlier_count"] = 0
                item["pos_x"] = float(
                    self.geo_conf_ema_alpha * item["pos_x"]
                    + (1.0 - self.geo_conf_ema_alpha) * pos_mean[0].item()
                )
                item["pos_y"] = float(
                    self.geo_conf_ema_alpha * item["pos_y"]
                    + (1.0 - self.geo_conf_ema_alpha) * pos_mean[1].item()
                )
                item["pos_z"] = float(
                    self.geo_conf_ema_alpha * item["pos_z"]
                    + (1.0 - self.geo_conf_ema_alpha) * pos_mean[2].item()
                )
                item["pos_var"] = (
                    self.geo_var_ema_alpha * item["pos_var"]
                    + (1.0 - self.geo_var_ema_alpha) * drift2
                )

        if allow_promote_landmark:
            self._update_landmarks_from_keyframe(frame_idx, uniq_vox, conf_mean_all)
        if allow_promote_reference:
            self._update_reference_bank_from_keyframe(frame_idx, uniq_vox, conf_mean_all)

        self._rebuild_stable_adaptive_maps(frame_idx)

        # Keep global bank bounded (stable/adaptive two-tier trim).
        do_trim = (
            len(self.geo_voxel_bank) > self.geo_max_voxels
            and (
                (frame_idx % int(self.geo_bank_trim_interval) == 0)
                or (len(self.geo_voxel_bank) > int(self.geo_max_voxels * 1.2))
            )
        )
        if do_trim:
            keep_budget = int(self.geo_max_voxels)
            excess = max(0, len(self.geo_voxel_bank) - keep_budget)
            if excess > 0:
                protected = set(self.geo_anchor_voxel_list)
                protected.update(self.geo_stable_anchor_voxels)
                protected.update(self.geo_landmark_voxels)
                protected.update(self.geo_reference_voxels)

                stable_cap = max(1, int(round(float(keep_budget) * float(self.geo_stable_map_ratio))))
                stable_pool = [k for k in self.geo_stable_map_voxels if k in self.geo_voxel_bank]
                adaptive_pool = [k for k in self.geo_voxel_bank.keys() if k not in self.geo_stable_map_voxels]

                # Stable map: light trim only when stable pool itself overflows its cap.
                stable_drop_budget = max(0, len(stable_pool) - stable_cap)
                if stable_drop_budget > 0:
                    stable_scored: List[Tuple[float, Tuple[int, int, int]]] = []
                    for key in stable_pool:
                        if key in protected:
                            continue
                        stable_scored.append((self._voxel_importance(self.geo_voxel_bank[key], frame_idx), key))
                    for _, key in heapq.nsmallest(stable_drop_budget, stable_scored, key=lambda x: x[0]):
                        self.geo_voxel_bank.pop(key, None)
                        excess = max(0, excess - 1)

                if excess > 0:
                    n_all = len(adaptive_pool)
                    if n_all > 0:
                        scan_size = max(
                            int(excess * self.geo_trim_drop_factor),
                            int(n_all * self.geo_trim_scan_ratio),
                            min(n_all, 1024),
                        )
                        scan_size = min(n_all, scan_size)

                        start = int(self.geo_trim_cursor % max(1, n_all))
                        stop = start + scan_size
                        if stop <= n_all:
                            sample_keys = adaptive_pool[start:stop]
                        else:
                            sample_keys = adaptive_pool[start:] + adaptive_pool[: stop - n_all]
                        self.geo_trim_cursor = (start + scan_size) % max(1, n_all)

                        scored: List[Tuple[float, Tuple[int, int, int]]] = []
                        for key in sample_keys:
                            if key in protected:
                                continue
                            scored.append((self._voxel_importance(self.geo_voxel_bank[key], frame_idx), key))

                        if scored:
                            drop_n = min(excess, len(scored))
                            for _, key in heapq.nsmallest(drop_n, scored, key=lambda x: x[0]):
                                self.geo_voxel_bank.pop(key, None)

        self._rebuild_stable_adaptive_maps(frame_idx)

        if frame_idx % self.geo_anchor_refresh_interval == 0:
            self._refresh_geo_anchor_voxels(frame_idx)
            active_frames = self._active_frames_for_anchor_refresh(frame_idx)
            for fidx in active_frames:
                self._update_frame_anchor_mask(fidx)
        else:
            self._update_frame_anchor_mask(frame_idx)

        self._prune_geo_frame_meta(frame_idx)

        logger.info(
            "[geo_bootstrap] frame=%d voxel_bank=%d ref_bank=%d stable_anchors=%d trust=%.4f recovery=%d reloc=%d matched_ratio=%.4f ref_overlap=%d",
            int(frame_idx),
            int(len(self.geo_voxel_bank)),
            int(len(self.geo_reference_bank)),
            int(len(self.geo_stable_anchor_voxels)),
            float(self.geo_trust_score),
            int(self.geo_recovery_frames_left),
            int(self.geo_reloc_frames_left),
            float(matched_ratio),
            int(ref_overlap),
        )

        return {
            "new_voxel_ratio": float(new_voxels) / max(1.0, float(num_groups)),
            "trust_score": float(self.geo_trust_score),
            "matched_ratio": float(matched_ratio),
            "ref_overlap": float(ref_overlap),
        }

    @staticmethod
    def _frustum_mask(
        pts: torch.Tensor,
        world_to_cam: torch.Tensor,
        intrinsic: torch.Tensor,
        near: float,
        far: float,
        img_hw: Optional[Tuple[int, int]] = None,
    ):
        if pts.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=pts.device)

        if world_to_cam.ndim == 3:
            world_to_cam = world_to_cam[0]
        if intrinsic.ndim == 3:
            intrinsic = intrinsic[0]
        ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
        pts_h = torch.cat([pts, ones], dim=-1)
        cam_h = pts_h @ world_to_cam.t()   # [N, 4]
        cam = cam_h[:, :3]                 # [N, 3]

        Xc, Yc, Zc = cam[:, 0], cam[:, 1], cam[:, 2]
        valid_z = (Zc > near) & (Zc < far)

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        inv_z = Zc.clamp_min(1e-6).reciprocal()
        u = fx * (Xc * inv_z) + cx
        v = fy * (Yc * inv_z) + cy

        if img_hw is not None:
            H, W = img_hw
            H = max(float(H), 1.0)
            W = max(float(W), 1.0)
        else:
            W = max((cx * 2.0).item(), 1.0)
            H = max((cy * 2.0).item(), 1.0)

        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return valid_z & inside

    @staticmethod
    def _index_meta(meta: Dict[str, torch.Tensor], keep_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        keep_cpu = keep_idx.detach().cpu().long()
        return {
            "frame_idx": meta["frame_idx"].index_select(0, keep_cpu),
            "is_special": meta["is_special"].index_select(0, keep_cpu),
            "is_keyframe": meta["is_keyframe"].index_select(0, keep_cpu)
            if "is_keyframe" in meta
            else torch.zeros((keep_cpu.numel(),), dtype=torch.bool),
            "local_patch_idx": meta["local_patch_idx"].index_select(0, keep_cpu),
            "identity_local": meta["identity_local"].index_select(0, keep_cpu)
            if "identity_local" in meta
            else meta["local_patch_idx"].index_select(0, keep_cpu),
            "global_id": meta["global_id"].index_select(0, keep_cpu)
            if "global_id" in meta
            else torch.empty((keep_cpu.numel(),), dtype=torch.long),
            "is_anchor": meta["is_anchor"].index_select(0, keep_cpu)
            if "is_anchor" in meta
            else torch.zeros((keep_cpu.numel(),), dtype=torch.bool),
            "is_landmark": meta["is_landmark"].index_select(0, keep_cpu)
            if "is_landmark" in meta
            else torch.zeros((keep_cpu.numel(),), dtype=torch.bool),
            "is_reference": meta["is_reference"].index_select(0, keep_cpu)
            if "is_reference" in meta
            else torch.zeros((keep_cpu.numel(),), dtype=torch.bool),
        }

    @staticmethod
    def _concat_meta(meta_a: Dict[str, torch.Tensor], meta_b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "frame_idx": torch.cat([meta_a["frame_idx"], meta_b["frame_idx"]], dim=0),
            "is_special": torch.cat([meta_a["is_special"], meta_b["is_special"]], dim=0),
            "is_keyframe": torch.cat(
                [
                    meta_a.get("is_keyframe", torch.zeros_like(meta_a["is_special"])),
                    meta_b.get("is_keyframe", torch.zeros_like(meta_b["is_special"])),
                ],
                dim=0,
            ),
            "local_patch_idx": torch.cat([meta_a["local_patch_idx"], meta_b["local_patch_idx"]], dim=0),
            "identity_local": torch.cat(
                [
                    meta_a.get("identity_local", meta_a["local_patch_idx"]),
                    meta_b.get("identity_local", meta_b["local_patch_idx"]),
                ],
                dim=0,
            ),
            "global_id": torch.cat(
                [
                    meta_a.get("global_id", torch.empty((0,), dtype=torch.long)),
                    meta_b.get("global_id", torch.empty((0,), dtype=torch.long)),
                ],
                dim=0,
            ),
            "is_anchor": torch.cat(
                [
                    meta_a.get("is_anchor", torch.zeros_like(meta_a["is_special"])),
                    meta_b.get("is_anchor", torch.zeros_like(meta_b["is_special"])),
                ],
                dim=0,
            ),
            "is_landmark": torch.cat(
                [
                    meta_a.get("is_landmark", torch.zeros_like(meta_a["is_special"])),
                    meta_b.get("is_landmark", torch.zeros_like(meta_b["is_special"])),
                ],
                dim=0,
            ),
            "is_reference": torch.cat(
                [
                    meta_a.get("is_reference", torch.zeros_like(meta_a["is_special"])),
                    meta_b.get("is_reference", torch.zeros_like(meta_b["is_special"])),
                ],
                dim=0,
            ),
        }

    @staticmethod
    def _hard_protected_mask(meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        frame_idx = meta["frame_idx"]
        is_special = meta["is_special"]
        if frame_idx.numel() == 0:
            return torch.empty(0, dtype=torch.bool)
        # Keep hard protection minimal to avoid stale long-horizon token privilege.
        return is_special

    @staticmethod
    def _protected_mask(meta: Dict[str, torch.Tensor], recent_frames: int) -> torch.Tensor:
        frame_idx = meta["frame_idx"]
        if frame_idx.numel() == 0:
            return torch.empty(0, dtype=torch.bool)
        current_frame_idx = int(frame_idx.max().item())
        recent_min = max(0, current_frame_idx - int(recent_frames))
        return Aggregator._hard_protected_mask(meta) | (frame_idx >= recent_min)

    def _cap_keep_with_protection(
        self,
        meta: Dict[str, torch.Tensor],
        keep_idx: torch.Tensor,
        budget: int,
        recent_frames: int,
    ) -> torch.Tensor:
        if keep_idx.numel() == 0:
            return keep_idx

        keep = torch.unique(keep_idx.detach().cpu().long(), sorted=True)
        keep = keep[(keep >= 0) & (keep < meta["frame_idx"].numel())]
        if keep.numel() == 0:
            return keep

        if budget <= 0:
            return torch.empty(0, dtype=torch.long)

        frame_idx_all = meta["frame_idx"]
        is_special_all = meta["is_special"]
        is_anchor_all = meta.get("is_anchor", torch.zeros_like(is_special_all))
        current_frame_idx = int(frame_idx_all.max().item()) if frame_idx_all.numel() > 0 else 0
        recent_min = max(0, current_frame_idx - int(recent_frames))

        frame_keep = frame_idx_all.index_select(0, keep)
        special_keep = is_special_all.index_select(0, keep)
        keyframe_keep = meta.get("is_keyframe", torch.zeros_like(is_special_all)).index_select(0, keep)
        anchor_keep = is_anchor_all.index_select(0, keep)
        landmark_keep = meta.get("is_landmark", torch.zeros_like(is_special_all)).index_select(0, keep)
        reference_keep = meta.get("is_reference", torch.zeros_like(is_special_all)).index_select(0, keep)

        frame0_keep = frame_keep == 0
        recent_keep = frame_keep >= recent_min

        def _take_tail(idx_tensor: torch.Tensor, n: int) -> torch.Tensor:
            if n <= 0 or idx_tensor.numel() == 0:
                return torch.empty((0,), dtype=torch.long)
            if idx_tensor.numel() <= n:
                return idx_tensor
            return idx_tensor[-n:]

        # Hard reservation groups (must be retained before any soft eviction).
        hard_special = keep[special_keep]
        frame0_patch = keep[frame0_keep & (~special_keep)]
        frame0_quota = min(int(max(0, budget // 16)), 64)
        hard_frame0 = _take_tail(frame0_patch, frame0_quota)
        anchor_quota = max(1, int(float(budget) * float(self.geo_anchor_budget_ratio)))
        reference_quota = int(self.geo_reference_token_quota)
        if self.geo_recovery_frames_left > 0:
            reference_quota = int(max(reference_quota, round(reference_quota * self.geo_recovery_ref_boost)))
        hard_anchor = _take_tail(keep[anchor_keep], min(anchor_quota, budget))
        hard_reference = _take_tail(keep[reference_keep], min(reference_quota, budget))
        hard_landmark = _take_tail(keep[landmark_keep], min(int(self.geo_landmark_token_quota), budget))
        hard_keyframe = _take_tail(keep[keyframe_keep], min(int(self.geo_keyframe_protected_quota), budget))

        hard_idx = torch.unique(
            torch.cat([hard_special, hard_frame0, hard_reference, hard_landmark, hard_anchor, hard_keyframe], dim=0),
            sorted=True,
        )
        if hard_idx.numel() >= budget:
            # Budget cannot fit all hard-reserved tokens: preserve frame0/special first, then anchors, then keyframes.
            parts: List[torch.Tensor] = []
            remain = int(budget)
            for part in [
                torch.unique(torch.cat([hard_special, hard_frame0], dim=0), sorted=True),
                torch.unique(hard_reference, sorted=True),
                torch.unique(hard_landmark, sorted=True),
                torch.unique(hard_anchor, sorted=True),
                torch.unique(hard_keyframe, sorted=True),
            ]:
                if remain <= 0:
                    break
                chosen = _take_tail(part, remain)
                if chosen.numel() > 0:
                    parts.append(chosen)
                    remain -= int(chosen.numel())
            if not parts:
                return torch.empty((0,), dtype=torch.long)
            return torch.unique(torch.cat(parts, dim=0), sorted=True)

        # Soft pools: evict from global/recent only after hard reservation is satisfied.
        remain = budget - int(hard_idx.numel())
        soft_recent = keep[recent_keep & ~(special_keep | frame0_keep)]
        soft_global = keep[
            ~(special_keep | frame0_keep | reference_keep | landmark_keep | anchor_keep | keyframe_keep | recent_keep)
        ]

        pick_recent = _take_tail(soft_recent, remain)
        remain_after_recent = remain - int(pick_recent.numel())
        pick_global = _take_tail(soft_global, remain_after_recent)

        out = torch.cat([hard_idx, pick_recent, pick_global], dim=0)
        return torch.unique(out, sorted=True)

    @staticmethod
    def _sanitize_keep_idx(keep_idx: torch.Tensor, meta_len: int, kv_len: int) -> torch.Tensor:
        if keep_idx is None or keep_idx.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        upper = min(int(meta_len), int(kv_len))
        if upper <= 0:
            return torch.empty(0, dtype=torch.long)
        keep = torch.unique(keep_idx.detach().cpu().long(), sorted=True)
        keep = keep[(keep >= 0) & (keep < upper)]
        return keep

    @staticmethod
    def _is_full_range_keep(keep_idx: torch.Tensor, length: int) -> bool:
        if keep_idx is None:
            return False
        if int(keep_idx.numel()) != int(length):
            return False
        if length == 0:
            return True
        keep_cpu = keep_idx.detach().cpu().long()
        return bool(torch.equal(keep_cpu, torch.arange(length, dtype=torch.long)))

    def _build_current_frame_meta(self, frame_idx: int, tokens_per_frame: int) -> Dict[str, torch.Tensor]:
        special = self.patch_start_idx
        patch_tokens = max(tokens_per_frame - special, 0)

        frame_idx_t = torch.full((tokens_per_frame,), int(frame_idx), dtype=torch.long)
        is_special = torch.zeros((tokens_per_frame,), dtype=torch.bool)
        is_special[:special] = True
        is_keyframe = torch.zeros((tokens_per_frame,), dtype=torch.bool)

        local_patch_idx = torch.full((tokens_per_frame,), -1, dtype=torch.long)
        identity_local = torch.full((tokens_per_frame,), -1, dtype=torch.long)
        if special > 0:
            identity_local[:special] = -(torch.arange(special, dtype=torch.long) + 1)
        if patch_tokens > 0:
            local_patch_idx[special:] = torch.arange(patch_tokens, dtype=torch.long)
            identity_local[special:] = local_patch_idx[special:]
            if int(frame_idx) in self.geo_keyframe_set:
                frozen_local = self.geo_keyframe_frozen_local.get(int(frame_idx), torch.empty((0,), dtype=torch.long))
                if frozen_local.numel() > 0:
                    valid = (frozen_local >= 0) & (frozen_local < patch_tokens)
                    if valid.any():
                        kf_pos = frozen_local[valid].long() + special
                        is_keyframe.index_fill_(0, kf_pos, True)
        global_id = frame_idx_t * int(self.geo_identity_stride) + (identity_local + int(self.geo_identity_offset))

        return {
            "frame_idx": frame_idx_t,
            "is_special": is_special,
            "is_keyframe": is_keyframe,
            "local_patch_idx": local_patch_idx,
            "identity_local": identity_local,
            "global_id": global_id,
            "is_anchor": torch.zeros((tokens_per_frame,), dtype=torch.bool),
            "is_landmark": torch.zeros((tokens_per_frame,), dtype=torch.bool),
            "is_reference": torch.zeros((tokens_per_frame,), dtype=torch.bool),
        }

    @staticmethod
    def _build_identity_keep_from_meta(meta: Dict[str, torch.Tensor], keep_idx: Optional[torch.Tensor]) -> torch.Tensor:
        if keep_idx is None or keep_idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        keep = torch.unique(keep_idx.detach().cpu().long(), sorted=True)
        if keep.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        frame = meta["frame_idx"].index_select(0, keep)
        global_id = meta.get("global_id", torch.empty((0,), dtype=torch.long)).index_select(0, keep)
        is_special = meta["is_special"].index_select(0, keep)
        is_anchor = meta.get("is_anchor", torch.zeros_like(meta["is_special"])).index_select(0, keep)
        is_reference = meta.get("is_reference", torch.zeros_like(meta["is_special"])).index_select(0, keep)

        def _rank(i: int):
            f = int(frame[i].item())
            return (
                1 if bool(is_special[i].item()) else 0,
                1 if bool(is_reference[i].item()) else 0,
                1 if bool(is_anchor[i].item()) else 0,
                1 if f == 0 else 0,
                f,
            )

        order = sorted(range(keep.numel()), key=_rank, reverse=True)
        out: List[int] = []
        seen = set()
        for i in order:
            key = int(global_id[i].item())
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        if not out:
            return torch.empty((0,), dtype=torch.long)
        return torch.tensor(out, dtype=torch.long)

    def _cap_identity_keep_with_protection(
        self,
        meta: Dict[str, torch.Tensor],
        identity_keep: torch.Tensor,
        budget: int,
        recent_frames: int,
    ) -> torch.Tensor:
        """
        Apply strict hard-cap in identity space, then keep original identity order.

        This keeps the shared keep-plan semantic in identity space instead of
        implicitly relying on per-layer positional index alignment.
        """
        if identity_keep is None or identity_keep.numel() == 0 or budget <= 0:
            return torch.empty((0,), dtype=torch.long)

        idx = self._identity_keep_to_index(meta, identity_keep)
        if idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        idx = self._cap_keep_with_protection(meta, idx, budget=budget, recent_frames=recent_frames)
        if idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        gid = meta.get("global_id", torch.empty((0,), dtype=torch.long))
        if gid.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        chosen_gid = torch.unique(gid.index_select(0, idx.detach().cpu().long()), sorted=False)
        identity_keep_cpu = identity_keep.detach().cpu().long()
        keep_mask = torch.isin(identity_keep_cpu, chosen_gid)
        out = identity_keep_cpu[keep_mask]
        if out.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        return out

    @staticmethod
    def _ensure_identity_lookup(meta: Dict[str, torch.Tensor]):
        gid = meta.get("global_id", torch.empty((0,), dtype=torch.long))
        gid_len = int(gid.numel())
        if gid.numel() == 0:
            meta["_gid_to_pos"] = {}
            meta["_gid_len"] = 0
            meta["_gid_is_sorted"] = True
            return
        if "_gid_to_pos" in meta and int(meta.get("_gid_len", -1)) == gid_len:
            return

        gid_cpu = gid.detach().cpu().long()
        gid_is_sorted = bool(torch.all(gid_cpu[1:] >= gid_cpu[:-1]).item()) if gid_cpu.numel() > 1 else True
        meta["_gid_is_sorted"] = gid_is_sorted
        if gid_is_sorted:
            meta["_gid_sorted"] = gid_cpu
            meta["_gid_len"] = gid_len
            meta["_gid_to_pos"] = {}
            return

        gid_to_pos: Dict[int, int] = {}
        for i, g in enumerate(gid_cpu.tolist()):
            gid_to_pos[int(g)] = i
        meta["_gid_to_pos"] = gid_to_pos
        meta["_gid_len"] = gid_len

    @staticmethod
    def _identity_keep_to_index(meta: Dict[str, torch.Tensor], identity_keep: torch.Tensor) -> torch.Tensor:
        if identity_keep is None or identity_keep.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        Aggregator._ensure_identity_lookup(meta)
        if bool(meta.get("_gid_is_sorted", False)):
            gid_sorted = meta.get("_gid_sorted", torch.empty((0,), dtype=torch.long))
            if gid_sorted.numel() == 0:
                return torch.empty((0,), dtype=torch.long)
            keys = torch.unique(identity_keep.detach().cpu().long(), sorted=True)
            where = torch.searchsorted(gid_sorted, keys)
            valid = where < gid_sorted.numel()
            if valid.sum().item() == 0:
                return torch.empty((0,), dtype=torch.long)
            where = where[valid]
            keys_v = keys[valid]
            matched = gid_sorted.index_select(0, where) == keys_v
            if matched.sum().item() == 0:
                return torch.empty((0,), dtype=torch.long)
            return torch.unique(where[matched], sorted=True)

        gid_to_pos = meta.get("_gid_to_pos", {})
        if not gid_to_pos:
            return torch.empty(0, dtype=torch.long)

        out: List[int] = []
        seen = set()
        for key in identity_keep.detach().cpu().long().tolist():
            k = int(key)
            if k in seen:
                continue
            seen.add(k)
            if k in gid_to_pos:
                out.append(int(gid_to_pos[k]))
        if not out:
            return torch.empty((0,), dtype=torch.long)
        return torch.tensor(out, dtype=torch.long)

    @staticmethod
    def _group_topk_by_hash(
        voxel_hash: torch.Tensor,
        scores: torch.Tensor,
        token_idx: torch.Tensor,
        topk_per_voxel: int,
    ) -> torch.Tensor:
        if voxel_hash.numel() == 0 or token_idx.numel() == 0 or topk_per_voxel <= 0:
            return torch.empty((0,), dtype=torch.long)

        order_score = torch.argsort(scores, descending=True, stable=True)
        h = voxel_hash.index_select(0, order_score)
        idx = token_idx.index_select(0, order_score)

        order_hash = torch.argsort(h, stable=True)
        h = h.index_select(0, order_hash)
        idx = idx.index_select(0, order_hash)

        _, counts = torch.unique_consecutive(h, return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts
        rank = torch.arange(h.numel(), dtype=torch.long) - torch.repeat_interleave(starts, counts)
        return idx[rank < int(topk_per_voxel)]

    def _should_force_full_geo_selection(
        self,
        current_frame_idx: int,
        current_view: Optional[Dict[str, Any]],
    ) -> bool:
        if current_frame_idx <= int(self.geo_warmup_frames):
            return True
        if current_frame_idx % int(self.geo_keyframe_interval) == 0:
            return True
        if current_view is None:
            return False
        pose_delta = float(current_view.get("pose_delta", 0.0) or 0.0)
        conf_drop = float(current_view.get("conf_drop", 0.0) or 0.0)
        new_voxel_ratio = float(current_view.get("new_voxel_ratio", 0.0) or 0.0)
        return (
            pose_delta >= self.geo_full_select_pose_delta
            or conf_drop >= self.geo_full_select_conf_drop
            or new_voxel_ratio >= self.geo_full_select_new_voxel_ratio
        )

    @staticmethod
    def _multiscale_old_frame_sample(sorted_frames: torch.Tensor, max_count: int) -> torch.Tensor:
        if sorted_frames.numel() <= max_count:
            return sorted_frames
        if max_count <= 1:
            return sorted_frames[-1:]

        max_count = int(max_count)
        near = max(1, max_count // 2)
        far = max_count - near
        out = [int(v) for v in sorted_frames[-near:].tolist()]

        older = sorted_frames[:-near]
        if far > 0 and older.numel() > 0:
            older_len = int(older.numel())
            # Log-spaced picks from far history.
            pos = torch.unique(
                torch.clamp(
                    torch.floor(torch.logspace(0, 1, steps=far, base=max(10.0, float(older_len + 1))) - 1).long(),
                    min=0,
                    max=max(0, older_len - 1),
                ),
                sorted=True,
            )
            for p in pos.tolist():
                out.append(int(older[p].item()))

        return torch.unique(torch.tensor(out, dtype=torch.long), sorted=True)

    def _maybe_console_geo_log(
        self,
        current_frame_idx: int,
        total_tokens: int,
        candidate_count: int,
        visible_total: int,
        selected_count: int,
        anchor_count: int,
        stable_count: int,
        tau_bucket: float,
        stable_visible_voxel_overlap: Optional[int] = None,
        stable_selected_visible: Optional[int] = None,
        stable_selected_invisible: Optional[int] = None,
        fast_path: Optional[int] = None,
        cache_size: Optional[int] = None,
        keep_overlap_cache: Optional[int] = None,
        reanchor_added: Optional[int] = None,
        reanchor_overlap_avg: Optional[float] = None,
        budget: Optional[int] = None,
    ):
        if current_frame_idx < 0:
            return
        interval = int(self.geo_console_log_interval)
        if interval < 0:
            return
        interval = max(1, interval)
        if (int(current_frame_idx) % interval) != 0:
            return
        if int(self.geo_last_console_log_frame) == int(current_frame_idx):
            return
        self.geo_last_console_log_frame = int(current_frame_idx)
        print(
            f"[geo_prune] total={int(total_tokens)} candidate={int(candidate_count)} "
            f"visible={int(visible_total)} selected={int(selected_count)} "
            f"anchor_selected={int(anchor_count)} stable_selected={int(stable_count)} "
            f"tau_bucket={float(tau_bucket):.4f}",
            flush=True,
        )
        if stable_visible_voxel_overlap is not None:
            print(
                f"[geo_prune] overlap_visible_stable_voxels={int(stable_visible_voxel_overlap)}",
                flush=True,
            )
        if stable_selected_visible is not None and stable_selected_invisible is not None:
            denom = max(1, int(stable_selected_visible) + int(stable_selected_invisible))
            vis_ratio = float(stable_selected_visible) / float(denom)
            print(
                f"[geo_prune] stable_keep_visible={int(stable_selected_visible)} "
                f"stable_keep_invisible={int(stable_selected_invisible)} stable_keep_visible_ratio={vis_ratio:.4f}",
                flush=True,
            )
        if fast_path is not None or cache_size is not None or keep_overlap_cache is not None or reanchor_added is not None or reanchor_overlap_avg is not None or budget is not None:
            overlap_avg = float(reanchor_overlap_avg) if reanchor_overlap_avg is not None else 0.0
            print(
                f"[geo_prune] fast_path={int(fast_path or 0)} cache_size={int(cache_size or 0)} "
                f"keep_overlap_cache={int(keep_overlap_cache or 0)} reanchor_added={int(reanchor_added or 0)} "
                f"reanchor_overlap_avg={overlap_avg:.2f} budget={int(budget or 0)}",
                flush=True,
            )

    def _extract_landmark_cache(
        self,
        meta: Dict[str, torch.Tensor],
        keep_idx: torch.Tensor,
        max_past_tokens: Optional[int],
    ) -> torch.Tensor:
        if keep_idx is None or keep_idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        keep = torch.unique(keep_idx.detach().cpu().long(), sorted=True)
        is_special = meta["is_special"].index_select(0, keep)
        is_reference = meta.get("is_reference", torch.zeros_like(meta["is_special"])).index_select(0, keep)
        is_landmark = meta.get("is_landmark", torch.zeros_like(meta["is_special"])).index_select(0, keep)
        is_anchor = meta.get("is_anchor", torch.zeros_like(meta["is_special"])).index_select(0, keep)
        is_keyframe = meta.get("is_keyframe", torch.zeros_like(meta["is_special"])).index_select(0, keep)

        cache = keep[(~is_special) & (is_reference | is_landmark | is_anchor | is_keyframe)]
        if max_past_tokens is not None:
            cap = min(2048, max(64, int(max_past_tokens * 0.15)))
        else:
            cap = 1024
        if cache.numel() > cap:
            cache = cache[-cap:]
        return cache

    @staticmethod
    def _count_keep_cache_overlap(keep_idx: torch.Tensor, cache_idx: torch.Tensor) -> int:
        if keep_idx is None or cache_idx is None:
            return 0
        if keep_idx.numel() == 0 or cache_idx.numel() == 0:
            return 0
        keep_cpu = keep_idx.detach().cpu().long()
        cache_cpu = cache_idx.detach().cpu().long()
        return int(torch.isin(keep_cpu, cache_cpu).sum().item())

    def _select_keyframe_tokens_stratified(
        self,
        meta: Dict[str, torch.Tensor],
        keyframe_patch_idx: torch.Tensor,
        quota: int,
    ) -> torch.Tensor:
        if quota <= 0 or keyframe_patch_idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        frame_idx = meta["frame_idx"]
        local_idx = meta["local_patch_idx"]
        token_frame = frame_idx.index_select(0, keyframe_patch_idx)
        unique_frames = torch.unique(token_frame).sort().values
        if unique_frames.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        pyramid_frames: List[List[int]] = [
            [int(f) for f in self.geo_keyframe_pyramid.get(0, []) if int(f) in set(unique_frames.tolist())],
            [int(f) for f in self.geo_keyframe_pyramid.get(1, []) if int(f) in set(unique_frames.tolist())],
            [int(f) for f in self.geo_keyframe_pyramid.get(2, []) if int(f) in set(unique_frames.tolist())],
        ]
        if any(len(v) > 0 for v in pyramid_frames):
            frame_chunks = [torch.tensor(v, dtype=torch.long) for v in pyramid_frames if len(v) > 0]
            # level0 (recent-mid) gets largest share; levels 1/2 keep guaranteed long-term footholds.
            base_weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
            weights = base_weights[[i for i, v in enumerate(pyramid_frames) if len(v) > 0]]
            weights = weights / weights.sum()
            raw_alloc = weights * float(quota)
            alloc = torch.floor(raw_alloc).to(torch.long)
        else:
            bin_count = min(int(self.geo_keyframe_time_bins), int(unique_frames.numel()), int(quota))
            frame_chunks = list(torch.chunk(unique_frames, bin_count))
            weights = torch.arange(1, bin_count + 1, dtype=torch.float32)
            raw_alloc = weights / weights.sum() * float(quota)
            alloc = torch.floor(raw_alloc).to(torch.long)

        alloc = torch.maximum(alloc, torch.ones_like(alloc))
        while int(alloc.sum().item()) > int(quota):
            for i in range(int(alloc.numel())):
                if alloc[i] > 1 and int(alloc.sum().item()) > int(quota):
                    alloc[i] -= 1
        extra = int(quota - int(alloc.sum().item()))
        b = int(alloc.numel()) - 1
        while extra > 0 and alloc.numel() > 0:
            alloc[b] += 1
            extra -= 1
            b = max(0, b - 1)

        selected: List[int] = []
        for i, frames_bin in enumerate(frame_chunks):
            q_bin = int(alloc[i].item())
            if q_bin <= 0 or frames_bin.numel() == 0:
                continue
            in_bin = (token_frame.unsqueeze(1) == frames_bin.unsqueeze(0)).any(dim=1)
            idx_bin = keyframe_patch_idx[in_bin]
            if idx_bin.numel() == 0:
                continue

            frame_bin = frame_idx.index_select(0, idx_bin)
            local_bin = local_idx.index_select(0, idx_bin).long()
            score = torch.zeros((idx_bin.numel(),), dtype=torch.float32)
            for j in range(idx_bin.numel()):
                f = int(frame_bin[j].item())
                lp = int(local_bin[j].item())
                fm = self.geo_frame_meta.get(f)
                if fm is not None and 0 <= lp < int(fm["conf"].shape[0]):
                    score[j] = float(fm["conf"][lp].item())
                else:
                    score[j] = float(f)

            k = min(q_bin, int(idx_bin.numel()))
            top = torch.topk(score, k=k, largest=True).indices
            selected.extend(idx_bin.index_select(0, top).tolist())

        if not selected:
            return torch.empty((0,), dtype=torch.long)
        return torch.unique(torch.tensor(selected, dtype=torch.long), sorted=True)

    def _geo_map_ready_for_prune(self, cache_frame_idx: int) -> bool:
        return self._geo_bank_ready(cache_frame_idx) and (self._geo_ref_ready() or (len(self.geo_stable_anchor_voxels) >= 64))

    def _geo_prune_ready(self, meta: Dict[str, torch.Tensor], max_past_tokens: Optional[int], cache_frame_idx: int) -> bool:
        if max_past_tokens is None:
            return False
        total_tokens = int(meta["frame_idx"].numel())
        if total_tokens <= int(float(max_past_tokens) * float(self.geo_prune_start_ratio)):
            return False
        return self._geo_map_ready_for_prune(cache_frame_idx)

    def _simple_non_geo_keep(self, meta: Dict[str, torch.Tensor], budget: int, recent_frames: int = 2) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n <= 0 or budget <= 0:
            return torch.empty((0,), dtype=torch.long)
        idx_all = torch.arange(n, dtype=torch.long)
        is_special = meta.get("is_special", torch.zeros((n,), dtype=torch.bool))
        frame_idx = meta.get("frame_idx", torch.zeros((n,), dtype=torch.long))
        max_frame = int(frame_idx.max().item()) if frame_idx.numel() > 0 else -1
        recent_mask = frame_idx >= max(-1, max_frame - int(recent_frames) + 1)
        keep = torch.unique(torch.cat([idx_all[is_special], idx_all[recent_mask]], dim=0), sorted=True)
        if keep.numel() > int(budget):
            keep = keep[-int(budget):]
        return keep

    def _build_reloc_identity_keep(
        self,
        meta: Dict[str, torch.Tensor],
        max_past_tokens: int,
        recent_frames: int,
    ) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n == 0 or int(max_past_tokens) <= 0:
            return torch.empty((0,), dtype=torch.long)
        idx_all = torch.arange(n, dtype=torch.long)
        frame_idx = meta["frame_idx"]
        is_special = meta.get("is_special", torch.zeros((n,), dtype=torch.bool))
        is_reference = meta.get("is_reference", torch.zeros((n,), dtype=torch.bool))
        is_landmark = meta.get("is_landmark", torch.zeros((n,), dtype=torch.bool))
        is_anchor = meta.get("is_anchor", torch.zeros((n,), dtype=torch.bool))
        is_keyframe = meta.get("is_keyframe", torch.zeros((n,), dtype=torch.bool))

        max_frame = int(frame_idx.max().item()) if frame_idx.numel() > 0 else -1
        recent_mask = frame_idx >= max(-1, max_frame - max(1, int(recent_frames)) + 1)
        recency = (frame_idx.to(torch.float32) - float(max_frame)).clamp(min=-16.0, max=0.0)
        score = 0.05 * (recency + 16.0)
        score = score + is_reference.to(torch.float32) * 3.0 + is_landmark.to(torch.float32) * 2.0 + is_anchor.to(torch.float32) * 1.5 + is_keyframe.to(torch.float32) * 1.0

        selected_parts: List[torch.Tensor] = [idx_all[is_special]]
        selected_mask = torch.zeros((n,), dtype=torch.bool)
        if selected_parts[0].numel() > 0:
            selected_mask[selected_parts[0]] = True

        def _pick(mask: torch.Tensor, quota: int):
            nonlocal selected_mask
            if quota <= 0:
                return
            cand = idx_all[mask & (~selected_mask)]
            if cand.numel() == 0:
                return
            k = min(int(quota), int(cand.numel()))
            sc = score.index_select(0, cand)
            top = torch.topk(sc, k=k, largest=True).indices
            out = cand.index_select(0, top)
            selected_parts.append(out)
            selected_mask[out] = True

        _pick(is_reference, 2048)
        _pick(is_landmark, 1024)
        _pick(is_anchor, 1024)
        _pick(is_keyframe, 2048)
        _pick(recent_mask & (~is_special), 512)

        keep = torch.unique(torch.cat([p for p in selected_parts if p.numel() > 0], dim=0), sorted=True)
        hard_mode = str(self.geo_reloc_state) == "hard"
        if (not hard_mode) and keep.numel() < int(max_past_tokens):
            remain = int(max_past_tokens) - int(keep.numel())
            extra = idx_all[(~selected_mask) & (is_reference | is_landmark | is_anchor | is_keyframe | recent_mask)]
            if extra.numel() > 0 and remain > 0:
                k = min(int(extra.numel()), int(min(remain, 512)))
                sc = score.index_select(0, extra)
                top = torch.topk(sc, k=k, largest=True).indices
                keep = torch.unique(torch.cat([keep, extra.index_select(0, top)], dim=0), sorted=True)

        keep = self._sanitize_keep_idx(keep, meta_len=n, kv_len=n)
        keep = self._cap_keep_with_protection(meta, keep, budget=int(max_past_tokens), recent_frames=recent_frames)
        return self._build_identity_keep_from_meta(meta, keep)

    def _select_geo_active_indices(
        self,
        meta: Dict[str, torch.Tensor],
        topk_per_voxel: int,
        recent_frames: int,
        near: float,
        far: float,
        current_view: Optional[Dict[str, torch.Tensor]],
        max_past_tokens: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        total_tokens = int(meta["frame_idx"].numel())
        if total_tokens == 0:
            return None
        if max_past_tokens is not None and total_tokens <= int(float(max_past_tokens) * float(self.geo_prune_start_ratio)):
            keep_all = torch.arange(total_tokens, dtype=torch.long)
            logger.info(
                "[geo_keep] total_tokens=%d budget=%d selected=%d selected_ratio=%.4f skip_prune=1",
                int(total_tokens),
                int(max_past_tokens),
                int(keep_all.numel()),
                float(keep_all.numel()) / float(max(1, total_tokens)),
            )
            self.geo_cached_landmark_keep = self._extract_landmark_cache(meta, keep_all, max_past_tokens)
            overlap = self._count_keep_cache_overlap(keep_all, self.geo_cached_landmark_keep)
            self._maybe_console_geo_log(
                current_frame_idx=int(meta["frame_idx"].max().item()) if meta["frame_idx"].numel() > 0 else -1,
                total_tokens=total_tokens,
                candidate_count=0,
                visible_total=0,
                selected_count=int(keep_all.numel()),
                anchor_count=int(meta.get("is_anchor", torch.zeros_like(meta["is_special"])).sum().item()),
                stable_count=0,
                tau_bucket=float("nan"),
                stable_visible_voxel_overlap=0,
                stable_selected_visible=0,
                stable_selected_invisible=0,
                fast_path=2,
                cache_size=int(self.geo_cached_landmark_keep.numel()),
                keep_overlap_cache=overlap,
                reanchor_added=0,
                reanchor_overlap_avg=0.0,
                budget=int(max_past_tokens or 0),
            )
            return keep_all

        selected = set()
        frame_idx = meta["frame_idx"]
        is_special = meta["is_special"]
        is_keyframe = meta.get("is_keyframe", torch.zeros_like(is_special))
        local_idx = meta["local_patch_idx"]
        is_anchor = meta.get("is_anchor", torch.zeros_like(is_special))
        is_landmark = meta.get("is_landmark", torch.zeros_like(is_special))
        is_reference = meta.get("is_reference", torch.zeros_like(is_special))

        # Always keep special tokens.
        special_idx = torch.nonzero(is_special, as_tuple=False).flatten().tolist()
        selected.update(special_idx)

        # Keep previously established anchor tokens (pinning within cached KV).
        anchor_token_idx = torch.nonzero(is_anchor, as_tuple=False).flatten().tolist()
        selected.update(anchor_token_idx)
        reference_token_idx = torch.nonzero(is_reference, as_tuple=False).flatten().tolist()
        selected.update(reference_token_idx)
        landmark_token_idx = torch.nonzero(is_landmark, as_tuple=False).flatten().tolist()
        selected.update(landmark_token_idx)

        # Frame0: keep special tokens always, patch tokens by a fixed cap.
        frame0_mask = frame_idx == 0
        frame0_special_idx = torch.nonzero(frame0_mask & is_special, as_tuple=False).flatten().tolist()
        selected.update(frame0_special_idx)

        frame0_patch_idx = torch.nonzero(frame0_mask & (~is_special) & (local_idx >= 0), as_tuple=False).flatten()
        if frame0_patch_idx.numel() > 0:
            frame0_local = local_idx[frame0_patch_idx].long()
            # try conf-guided selection if metadata for frame0 exists
            frame0_meta = self.geo_frame_meta.get(0)
            if frame0_meta is not None and frame0_meta["conf"].numel() > 0:
                in_range = (frame0_local >= 0) & (frame0_local < frame0_meta["conf"].shape[0])
                frame0_patch_idx = frame0_patch_idx[in_range]
                frame0_local = frame0_local[in_range]
                if frame0_patch_idx.numel() > 0:
                    conf0 = frame0_meta["conf"].index_select(0, frame0_local).to(torch.float32)
                    k0 = min(int(self.geo_frame0_patch_cap), int(frame0_patch_idx.numel()))
                    top_idx = torch.topk(conf0, k=k0, largest=True).indices
                    selected.update(frame0_patch_idx.index_select(0, top_idx).tolist())
            else:
                k0 = min(int(self.geo_frame0_patch_cap), int(frame0_patch_idx.numel()))
                selected.update(frame0_patch_idx[:k0].tolist())

        # Reserve sparse keyframe tokens to preserve long-horizon constraints.
        keyframe_patch_idx = torch.nonzero(is_keyframe & (~is_special) & (~frame0_mask) & (local_idx >= 0), as_tuple=False).flatten()
        if keyframe_patch_idx.numel() > 0 and int(self.geo_keyframe_token_quota) > 0:
            keyframe_keep = self._select_keyframe_tokens_stratified(
                meta,
                keyframe_patch_idx,
                quota=int(self.geo_keyframe_token_quota),
            )
            selected.update(keyframe_keep.tolist())

        current_frame_idx = int(frame_idx.max().item()) if frame_idx.numel() > 0 else 0
        recent_min = max(0, current_frame_idx - int(recent_frames))
        recent_mask = frame_idx >= recent_min

        force_full_select = self._should_force_full_geo_selection(current_frame_idx, current_view)

        # Optional fast-path (event/interval gated): only run full geo selection on key/unstable frames.
        if self.geo_selection_interval > 1 and (not force_full_select) and (current_frame_idx % self.geo_selection_interval != 0):
            recent_idx = torch.nonzero(recent_mask, as_tuple=False).flatten().tolist()
            selected.update(recent_idx)
            keep_fast = torch.tensor(sorted(selected), dtype=torch.long)
            if max_past_tokens is not None:
                keep_fast = self._cap_keep_with_protection(
                    meta,
                    keep_fast,
                    budget=max(0, int(max_past_tokens)),
                    recent_frames=recent_frames,
                )
            self.geo_cached_landmark_keep = self._extract_landmark_cache(meta, keep_fast, max_past_tokens)
            overlap = self._count_keep_cache_overlap(keep_fast, self.geo_cached_landmark_keep)
            self._maybe_console_geo_log(
                current_frame_idx=current_frame_idx,
                total_tokens=total_tokens,
                candidate_count=0,
                visible_total=0,
                selected_count=int(keep_fast.numel()),
                anchor_count=len(anchor_token_idx),
                stable_count=0,
                tau_bucket=float("nan"),
                stable_visible_voxel_overlap=0,
                stable_selected_visible=0,
                stable_selected_invisible=0,
                fast_path=1,
                cache_size=int(self.geo_cached_landmark_keep.numel()),
                keep_overlap_cache=overlap,
                reanchor_added=0,
                reanchor_overlap_avg=0.0,
                budget=int(max_past_tokens or 0),
            )
            return keep_fast

        # Build a budgeted local-tracking pool for recent patches (not all recent patches).
        if max_past_tokens is not None:
            recent_frames_count = max(1, int(torch.unique(frame_idx[recent_mask]).numel()))
            local_ratio = float(self.geo_local_budget_ratio)
            if self.geo_reloc_frames_left > 0:
                local_ratio = min(local_ratio, float(self.geo_reloc_local_budget_ratio))
            if current_frame_idx <= int(self.geo_warmup_frames):
                local_ratio = max(local_ratio, float(self.geo_warmup_local_budget_ratio))
            local_budget = min(
                int(max_past_tokens * local_ratio),
                int(self.geo_local_budget_cap_per_frame) * recent_frames_count,
            )
        else:
            local_budget = 0
        local_selected: List[int] = []
        recent_special_idx = torch.nonzero(recent_mask & is_special, as_tuple=False).flatten().tolist()
        selected.update(recent_special_idx)

        recent_patch_indices = torch.nonzero(recent_mask & (~is_special) & (local_idx >= 0), as_tuple=False).flatten()
        if recent_patch_indices.numel() > 0 and local_budget > 0:
            per_frame_budget = max(1, local_budget // max(1, int(torch.unique(frame_idx[recent_patch_indices]).numel())))
            grid_n = max(1, int(self.geo_local_coverage_grid))
            for fidx in torch.unique(frame_idx[recent_patch_indices]).tolist():
                fidx = int(fidx)
                frame_meta = self.geo_frame_meta.get(fidx)
                if frame_meta is None:
                    continue
                idx_f = recent_patch_indices[frame_idx[recent_patch_indices] == fidx]
                if idx_f.numel() == 0:
                    continue

                local_f = local_idx[idx_f].long()
                in_range = (local_f >= 0) & (local_f < frame_meta["conf"].shape[0])
                if in_range.sum().item() == 0:
                    continue
                idx_f = idx_f[in_range]
                local_f = local_f[in_range]
                conf_f = frame_meta["conf"].index_select(0, local_f).to(torch.float32)

                # High-confidence top-k
                k_top = min(int(per_frame_budget), int(idx_f.numel()))
                if k_top > 0:
                    top_idx = torch.topk(conf_f, k=k_top, largest=True).indices
                    local_selected.extend(idx_f.index_select(0, top_idx).tolist())

                # Coverage fallback across local patch grid
                patch_n = int(frame_meta["conf"].shape[0])
                side = max(1, int(round(patch_n ** 0.5)))
                bin_h = max(1, side // grid_n)
                bin_w = max(1, side // grid_n)
                best_per_cell: Dict[Tuple[int, int], Tuple[float, int]] = {}
                for j in range(idx_f.numel()):
                    lp = int(local_f[j].item())
                    y, x = lp // side, lp % side
                    cell = (min(grid_n - 1, y // bin_h), min(grid_n - 1, x // bin_w))
                    sc = float(conf_f[j].item())
                    gid = int(idx_f[j].item())
                    prev = best_per_cell.get(cell)
                    if prev is None or sc > prev[0]:
                        best_per_cell[cell] = (sc, gid)
                local_selected.extend(v[1] for v in best_per_cell.values())

        if local_selected:
            # keep deterministic order by (frame_idx, token_idx)
            local_u = sorted(set(int(i) for i in local_selected), key=lambda i: (int(frame_idx[i].item()), i))
            if local_budget > 0 and len(local_u) > local_budget:
                local_u = local_u[:local_budget]
            selected.update(local_u)

        if current_view is None or current_view.get("world_to_cam") is None or current_view.get("intrinsic") is None:
            return torch.tensor(sorted(selected), dtype=torch.long)

        world_to_cam = current_view["world_to_cam"]
        intrinsic = current_view["intrinsic"]
        if isinstance(world_to_cam, torch.Tensor):
            world_to_cam = world_to_cam.detach()
            if world_to_cam.device.type != "cpu":
                world_to_cam = world_to_cam.cpu()
        if isinstance(intrinsic, torch.Tensor):
            intrinsic = intrinsic.detach()
            if intrinsic.device.type != "cpu":
                intrinsic = intrinsic.cpu()
        if world_to_cam.ndim == 3:
            world_to_cam = world_to_cam[0]
        if intrinsic.ndim == 3:
            intrinsic = intrinsic[0]

        img_hw = current_view.get("img_hw") if current_view is not None else None

        # Candidates for geometry-based pruning: non-special and non-recent tokens
        candidate_mask = (~is_special) & (~recent_mask) & (local_idx >= 0)
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            return torch.tensor(sorted(selected), dtype=torch.long)

        # Acceleration guard 1: restrict scoring to recent old frames only.
        if self.geo_max_old_frames_to_score > 0:
            uniq = torch.unique(frame_idx[candidate_indices])
            if uniq.numel() > self.geo_max_old_frames_to_score:
                keep_frames = self._multiscale_old_frame_sample(
                    uniq.sort().values,
                    int(self.geo_max_old_frames_to_score),
                )
                keep_mask = (frame_idx[candidate_indices].unsqueeze(1) == keep_frames.unsqueeze(0)).any(dim=1)
                candidate_indices = candidate_indices[keep_mask]

        # Acceleration guard 2: cap candidate token count per layer.
        if self.geo_max_candidate_tokens > 0 and candidate_indices.numel() > self.geo_max_candidate_tokens:
            cf = frame_idx[candidate_indices]
            order = torch.argsort(cf)
            candidate_indices = candidate_indices.index_select(0, order)[-self.geo_max_candidate_tokens :]

        # Global per-voxel top-k across all old frames (tensorized grouping to reduce Python overhead).
        gather_idx: List[torch.Tensor] = []
        gather_score: List[torch.Tensor] = []
        gather_bank_conf: List[torch.Tensor] = []
        gather_voxel_hash: List[torch.Tensor] = []
        gather_visible: List[torch.Tensor] = []
        candidate_count = int(candidate_indices.numel())
        visible_total = 0

        for fidx in torch.unique(frame_idx[candidate_indices]).tolist():
            fidx = int(fidx)
            frame_meta = self.geo_frame_meta.get(fidx)
            if frame_meta is None:
                continue

            in_frame = candidate_indices[frame_idx[candidate_indices] == fidx]
            if in_frame.numel() == 0:
                continue

            local = local_idx[in_frame]
            valid_local = (local >= 0) & (local < frame_meta["pts"].shape[0])
            if valid_local.sum().item() == 0:
                continue

            in_frame = in_frame[valid_local]
            local = local[valid_local].long()

            pts = frame_meta["pts"].index_select(0, local).to(torch.float32)
            conf = frame_meta["conf"].index_select(0, local).to(torch.float32)
            vox = frame_meta["voxel_ids"].index_select(0, local)

            visible = self._frustum_mask(
                pts,
                world_to_cam.to(torch.float32),
                intrinsic.to(torch.float32),
                near=near,
                far=far,
                img_hw=img_hw,
            )
            visible_total += int(visible.sum().item())

            bank_conf_l: List[float] = []
            bank_support_l: List[float] = []
            bank_var_l: List[float] = []
            is_anchor_vox_l: List[bool] = []
            is_ref_vox_l: List[bool] = []
            for key in (tuple(int(v) for v in row) for row in vox.tolist()):
                bank = self.geo_voxel_bank.get(key)
                if bank is None:
                    bank_conf_l.append(0.0)
                    bank_support_l.append(1.0)
                    bank_var_l.append(0.0)
                else:
                    bank_conf_l.append(float(bank["conf_ema"]))
                    bank_support_l.append(float(bank["support"]))
                    bank_var_l.append(float(bank["pos_var"]))
                is_anchor_vox_l.append(key in self.geo_anchor_voxels)
                is_ref_vox_l.append(key in self.geo_reference_voxels)

            bank_conf_t = torch.tensor(bank_conf_l, dtype=torch.float32)
            bank_support_t = torch.tensor(bank_support_l, dtype=torch.float32)
            bank_var_t = torch.tensor(bank_var_l, dtype=torch.float32)
            anchor_vox_t = torch.tensor(is_anchor_vox_l, dtype=torch.bool)
            ref_vox_t = torch.tensor(is_ref_vox_l, dtype=torch.bool)

            conf_safe = conf.to(torch.float32).clamp_min(1e-6)
            bank_conf_eff = torch.where(bank_conf_t > 0, bank_conf_t, conf_safe)
            support_gain = torch.log1p(bank_support_t)
            stability = (1.0 / (1.0 + bank_var_t)).to(torch.float32)
            base_score = conf_safe * bank_conf_eff.clamp_min(1e-6) * support_gain * stability
            if ref_vox_t.any() and float(self.geo_reference_overlap_bonus) > 0.0:
                base_score = base_score * torch.where(
                    ref_vox_t,
                    torch.full_like(base_score, 1.0 + float(self.geo_reference_overlap_bonus)),
                    torch.ones_like(base_score),
                )

            if (self.geo_recovery_frames_left <= 0) and (float(self.geo_trust_score) >= float(self.geo_selection_low_trust_threshold)):
                vis_weight = torch.where(
                    visible,
                    torch.ones_like(base_score),
                    torch.where(
                        anchor_vox_t,
                        torch.full_like(base_score, float(self.geo_anchor_invisible_read_weight)),
                        torch.full_like(base_score, float(self.geo_invisible_read_weight)),
                    ),
                )
            else:
                vis_weight = torch.ones_like(base_score)
            score_t = base_score * vis_weight

            gather_idx.append(in_frame.to(torch.long))
            gather_score.append(score_t)
            gather_bank_conf.append(bank_conf_eff)
            gather_voxel_hash.append(self._voxel_hash(vox))
            gather_visible.append(visible.to(torch.bool))

        if not gather_idx:
            return torch.tensor(sorted(selected), dtype=torch.long)

        idx_all = torch.cat(gather_idx, dim=0)
        score_all = torch.cat(gather_score, dim=0)
        bank_conf_all = torch.cat(gather_bank_conf, dim=0)
        hash_all = torch.cat(gather_voxel_hash, dim=0)
        visible_all = torch.cat(gather_visible, dim=0)

        # Adaptive bucket threshold to control bucket size.
        remaining_budget = None
        if max_past_tokens is not None:
            remaining_budget = max(0, int(max_past_tokens) - len(selected))
        tau_bucket = self._compute_dynamic_bucket_threshold(bank_conf_all.tolist(), remaining_budget or 0)

        # Global anchor quota from ordered anchor list (deterministic).
        anchor_count = 0
        selected_global: set[int] = set()
        if self.geo_anchor_voxel_list:
            anchor_quota = int(self.geo_anchor_read_quota)
            if max_past_tokens is not None:
                anchor_quota = min(anchor_quota, max(0, int(max_past_tokens * self.geo_anchor_budget_ratio)))

            anchor_hash_to_best: Dict[int, Tuple[float, int]] = {}
            valid_anchor = bank_conf_all >= float(self.geo_anchor_conf_exit)
            if valid_anchor.any():
                idx_v = idx_all[valid_anchor]
                score_v = score_all[valid_anchor]
                hash_v = hash_all[valid_anchor]
                for i in range(idx_v.numel()):
                    h = int(hash_v[i].item())
                    sc = float(score_v[i].item())
                    token = int(idx_v[i].item())
                    prev = anchor_hash_to_best.get(h)
                    if prev is None or sc > prev[0]:
                        anchor_hash_to_best[h] = (sc, token)

            ordered_anchor = list(self.geo_stable_anchor_voxel_list) + [
                v for v in self.geo_anchor_voxel_list if v not in self.geo_stable_anchor_voxels
            ]
            for vox in ordered_anchor:
                h = int(self._voxel_hash(torch.tensor([vox], dtype=torch.long))[0].item())
                if h not in anchor_hash_to_best:
                    continue
                token = int(anchor_hash_to_best[h][1])
                selected.add(token)
                selected_global.add(token)
                anchor_count += 1
                if anchor_count >= anchor_quota:
                    break

        # Stable-map hard read budget: preserve long-horizon constraints even under adaptive churn.
        stable_count = 0
        stable_visible_selected = 0
        stable_invisible_selected = 0
        stable_selected_tokens: List[int] = []
        stable_visible_voxel_overlap = 0
        reanchor_added = 0
        reanchor_overlap_sum = 0
        reanchor_frames_used = 0
        vis_hash_unique = torch.unique(hash_all[visible_all]) if visible_all.any() else torch.empty((0,), dtype=hash_all.dtype)
        stable_source_voxels = self.geo_stable_map_voxels
        if self.geo_recovery_frames_left > 0 and self.geo_reference_voxels:
            overlap_ref = self.geo_stable_map_voxels.intersection(self.geo_reference_voxels)
            stable_source_voxels = overlap_ref if overlap_ref else self.geo_reference_voxels
        if stable_source_voxels:
            stable_hash = self._voxel_hash(torch.tensor(sorted(stable_source_voxels), dtype=torch.long))
            stable_mask = torch.isin(hash_all, stable_hash)
            vis_stable_mask = stable_mask & visible_all
            if vis_stable_mask.any():
                stable_visible_voxel_overlap = int(torch.unique(hash_all[vis_stable_mask]).numel())
            if stable_mask.any():
                stable_quota = 0
                if max_past_tokens is not None:
                    stable_quota = max(0, int(max_past_tokens * float(self.geo_stable_read_budget_ratio)))
                stable_quota = max(stable_quota, len(self.geo_stable_anchor_voxel_list))
                if self.geo_reloc_frames_left > 0 and max_past_tokens is not None:
                    reloc_quota = max(0, int(max_past_tokens * float(self.geo_reloc_stable_read_budget_ratio)))
                    stable_quota = max(stable_quota, reloc_quota)
                if self.geo_recovery_frames_left > 0:
                    stable_quota = int(round(float(stable_quota) * float(self.geo_recovery_stable_ratio_scale)))
                stable_quota = min(stable_quota, int(stable_mask.sum().item()))
                if stable_quota > 0:
                    stable_idx = idx_all[stable_mask]
                    stable_score = score_all[stable_mask]
                    stable_vis = visible_all[stable_mask]
                    stable_hash_all = hash_all[stable_mask]

                    vis_idx_raw = stable_idx[stable_vis]
                    vis_score_raw = stable_score[stable_vis]
                    vis_hash_raw = stable_hash_all[stable_vis]
                    invis_idx_raw = stable_idx[~stable_vis]
                    invis_score_raw = stable_score[~stable_vis]
                    invis_hash_raw = stable_hash_all[~stable_vis]

                    vis_idx = self._group_topk_by_hash(
                        vis_hash_raw,
                        vis_score_raw,
                        vis_idx_raw,
                        topk_per_voxel=int(self.geo_stable_topk_per_voxel),
                    )
                    invis_idx = self._group_topk_by_hash(
                        invis_hash_raw,
                        invis_score_raw,
                        invis_idx_raw,
                        topk_per_voxel=1,
                    )

                    vis_score_map = {int(vis_idx_raw[i].item()): float(vis_score_raw[i].item()) for i in range(vis_idx_raw.numel())}
                    invis_score_map = {int(invis_idx_raw[i].item()): float(invis_score_raw[i].item()) for i in range(invis_idx_raw.numel())}

                    vis_quota = stable_quota - int(round(stable_quota * float(self.geo_stable_invisible_quota_ratio)))
                    vis_quota = max(0, min(stable_quota, vis_quota))
                    invis_quota = max(0, stable_quota - vis_quota)

                    # Overlap-driven stable budget: avoid token flooding when visible stable overlap is low.
                    vis_overlap_cap = int(stable_visible_voxel_overlap) * int(self.geo_stable_topk_per_voxel)
                    if vis_overlap_cap > 0:
                        vis_quota = min(vis_quota, vis_overlap_cap)

                    if self.geo_recovery_frames_left > 0:
                        invis_quota = min(invis_quota, max(0, int(round(stable_quota * 0.02))))
                    vis_quota = max(0, min(vis_quota, int(vis_idx.numel())))
                    invis_quota = max(0, min(invis_quota, int(invis_idx.numel())))
                    stable_quota_eff = vis_quota + invis_quota

                    for part_idx, part_score_map, part_quota, is_visible_part in [
                        (vis_idx, vis_score_map, vis_quota, True),
                        (invis_idx, invis_score_map, invis_quota, False),
                    ]:
                        if part_quota <= 0:
                            continue
                        if part_idx.numel() == 0:
                            continue
                        part_score = torch.tensor([part_score_map.get(int(v.item()), 0.0) for v in part_idx], dtype=torch.float32)
                        order_part = torch.argsort(part_score, descending=True)
                        picked_part = 0
                        for i in order_part.tolist():
                            token = int(part_idx[i].item())
                            if token in selected_global:
                                continue
                            selected.add(token)
                            selected_global.add(token)
                            stable_selected_tokens.append(token)
                            stable_count += 1
                            picked_part += 1
                            if is_visible_part:
                                stable_visible_selected += 1
                            else:
                                stable_invisible_selected += 1
                            if picked_part >= int(part_quota) or stable_count >= stable_quota_eff:
                                break
                        if stable_count >= stable_quota_eff:
                            break

                    # NOTE: keep invisible quota as a hard cap. Do not backfill with invisible stable tokens.
                    if stable_count < stable_quota_eff and vis_quota > stable_visible_selected:
                        if vis_idx.numel() > 0:
                            vis_score = torch.tensor([vis_score_map.get(int(v.item()), 0.0) for v in vis_idx], dtype=torch.float32)
                            order_vis = torch.argsort(vis_score, descending=True)
                            for i in order_vis.tolist():
                                token = int(vis_idx[i].item())
                                if token in selected_global:
                                    continue
                                selected.add(token)
                                selected_global.add(token)
                                stable_selected_tokens.append(token)
                                stable_count += 1
                                stable_visible_selected += 1
                                if stable_count >= stable_quota_eff or stable_visible_selected >= vis_quota:
                                    break

                    # Overlap low: use retrieval-like keyframe tokens to fill remaining long-horizon constraints.
                    stable_deficit = max(0, int(stable_quota - stable_count))
                    if (
                        stable_deficit > 0
                        and int(stable_visible_voxel_overlap) < int(self.geo_stable_overlap_low_threshold)
                        and self.geo_keyframes
                        and max_past_tokens is not None
                    ):
                        retrieval_budget = min(
                            stable_deficit,
                            max(0, int(max_past_tokens * float(self.geo_overlap_retrieval_budget_ratio))),
                        )
                        if retrieval_budget > 0 and vis_hash_unique.numel() > 0:
                            frame_scores: List[Tuple[int, int]] = []
                            for f in self.geo_keyframes:
                                fm = self.geo_frame_meta.get(int(f))
                                if fm is None or fm.get("voxel_ids") is None or fm["voxel_ids"].numel() == 0:
                                    continue
                                fh = self._voxel_hash(fm["voxel_ids"])
                                overlap = int(torch.isin(fh, vis_hash_unique).sum().item())
                                if overlap > 0:
                                    frame_scores.append((overlap, int(f)))

                            frame_scores.sort(key=lambda x: (-x[0], x[1]))
                            chosen_scored = frame_scores[: int(self.geo_overlap_retrieval_top_frames)]
                            chosen_frames = [f for _, f in chosen_scored]
                            if chosen_frames:
                                reanchor_overlap_sum += int(sum(int(v) for v, _ in chosen_scored))
                                reanchor_frames_used += int(len(chosen_frames))
                                per_frame_q = max(1, retrieval_budget // max(1, len(chosen_frames)))
                                for f in chosen_frames:
                                    idx_f = torch.nonzero((frame_idx == int(f)) & (~is_special) & (local_idx >= 0), as_tuple=False).flatten()
                                    if idx_f.numel() == 0:
                                        continue
                                    fm = self.geo_frame_meta.get(int(f))
                                    if fm is None or fm.get("conf") is None or fm["conf"].numel() == 0 or fm.get("voxel_ids") is None:
                                        continue
                                    local_f = local_idx.index_select(0, idx_f).long()
                                    in_range = (local_f >= 0) & (local_f < fm["conf"].shape[0]) & (local_f < fm["voxel_ids"].shape[0])
                                    if in_range.sum().item() == 0:
                                        continue
                                    idx_f = idx_f[in_range]
                                    local_f = local_f[in_range]
                                    conf_f = fm["conf"].index_select(0, local_f).to(torch.float32)
                                    hash_f = self._voxel_hash(fm["voxel_ids"].index_select(0, local_f))
                                    overlap_mask = torch.isin(hash_f, vis_hash_unique)
                                    if overlap_mask.sum().item() == 0:
                                        continue
                                    idx_f = idx_f[overlap_mask]
                                    conf_f = conf_f[overlap_mask]
                                    hash_f = hash_f[overlap_mask]
                                    top_idx = self._group_topk_by_hash(hash_f, conf_f, idx_f, topk_per_voxel=max(1, int(self.geo_stable_keyframe_topk_per_frame)))
                                    if top_idx.numel() == 0:
                                        continue
                                    picked = 0
                                    for token_t in top_idx.tolist():
                                        token = int(token_t)
                                        if token in selected_global:
                                            continue
                                        selected.add(token)
                                        selected_global.add(token)
                                        stable_selected_tokens.append(token)
                                        reanchor_added += 1
                                        picked += 1
                                        if picked >= per_frame_q:
                                            break

        stable_total_selected = int(stable_visible_selected + stable_invisible_selected)
        stable_visible_ratio = float(stable_visible_selected) / float(max(1, stable_total_selected))
        bad_stable_quality = (
            stable_total_selected == 0
            or stable_visible_ratio < float(self.geo_stable_quality_visible_ratio_thr)
            or int(stable_visible_voxel_overlap) < int(self.geo_stable_quality_overlap_thr)
            or int(visible_total) == 0
        )
        if bad_stable_quality:
            self.geo_bad_stable_quality_streak += 1
        else:
            self.geo_bad_stable_quality_streak = 0
        if self.geo_bad_stable_quality_streak >= int(self.geo_stable_quality_streak_thr):
            self.geo_recovery_frames_left = max(int(self.geo_recovery_frames_left), int(self.geo_recovery_frames))

        reloc_trigger = (
            stable_total_selected == 0
            or int(visible_total) == 0
            or int(stable_visible_voxel_overlap) < int(self.geo_reloc_trigger_overlap)
            or float(stable_visible_ratio) < float(self.geo_reloc_trigger_visible_ratio)
            or float(self.geo_trust_score) < float(self.geo_selection_low_trust_threshold)
        )
        good_geo = (
            float(self.geo_trust_score) >= float(self.geo_selection_low_trust_threshold)
            and int(stable_visible_voxel_overlap) >= int(self.geo_reloc_trigger_overlap)
            and int(visible_total) > 0
        )
        if reloc_trigger and str(self.geo_reloc_state) == "off":
            self.geo_reloc_state = "hard"
            self.geo_reloc_hard_left = int(self.geo_reloc_hard_frames)
            self.geo_reloc_frames_left = max(int(self.geo_reloc_frames_left), int(self.geo_reloc_frames))
            self.geo_reloc_good_streak = 0

        if str(self.geo_reloc_state) != "off":
            if good_geo:
                self.geo_reloc_good_streak += 1
            else:
                self.geo_reloc_good_streak = 0
            if str(self.geo_reloc_state) == "hard":
                self.geo_reloc_hard_left = max(0, int(self.geo_reloc_hard_left) - 1)
                if int(self.geo_reloc_hard_left) <= 0:
                    self.geo_reloc_state = "recover"
            if self.geo_reloc_frames_left > 0:
                self.geo_reloc_frames_left = max(0, int(self.geo_reloc_frames_left) - 1)
            if int(self.geo_reloc_frames_left) <= 0 or int(self.geo_reloc_good_streak) >= 3:
                self.geo_reloc_state = "off"
                self.geo_reloc_frames_left = 0
                self.geo_reloc_hard_left = 0
                self.geo_reloc_good_streak = 0

        bad_mode = (
            int(self.geo_recovery_frames_left) > 0
            or int(self.geo_reloc_frames_left) > 0
            or int(visible_total) == 0
            or int(stable_visible_voxel_overlap) < int(self.geo_reloc_trigger_overlap)
        )
        if bad_mode:
            ref_mask_all = is_reference.index_select(0, idx_all)
            valid_global = visible_all & (bank_conf_all >= float(tau_bucket))
            idx_valid = idx_all[valid_global]
            score_valid = score_all[valid_global]
            hash_valid = hash_all[valid_global]
            tiny_invis_quota = 96
            invis_ref_mask = (~visible_all) & ref_mask_all & (bank_conf_all >= float(tau_bucket))
            if invis_ref_mask.any() and tiny_invis_quota > 0:
                idx_invis = idx_all[invis_ref_mask]
                sc_invis = score_all[invis_ref_mask]
                if idx_invis.numel() > tiny_invis_quota:
                    top_invis = torch.topk(sc_invis, k=tiny_invis_quota, largest=True).indices
                    idx_invis = idx_invis.index_select(0, top_invis)
                    sc_invis = sc_invis.index_select(0, top_invis)
                idx_valid = torch.cat([idx_valid, idx_invis], dim=0)
                score_valid = torch.cat([score_valid, sc_invis], dim=0)
                hash_valid = torch.cat([hash_valid, torch.full((idx_invis.numel(),), -1, dtype=hash_all.dtype)], dim=0)
        else:
            valid_global = bank_conf_all >= float(tau_bucket)
            idx_valid = idx_all[valid_global]
            score_valid = score_all[valid_global]
            hash_valid = hash_all[valid_global]
        grouped_idx = self._group_topk_by_hash(hash_valid, score_valid, idx_valid, topk_per_voxel=max(1, int(topk_per_voxel)))

        if grouped_idx.numel() > 0:
            if max_past_tokens is not None:
                selected_base = len(selected)
                left_budget = max(0, int(max_past_tokens) - selected_base)
            else:
                left_budget = int(grouped_idx.numel())

            global_budget = left_budget

            addable = [int(v) for v in grouped_idx.tolist() if int(v) not in selected_global]
            if addable and global_budget > 0:
                addable = addable[:global_budget]
                selected.update(addable)

        if stable_selected_tokens and "is_reference" in meta:
            stable_idx = torch.unique(torch.tensor(stable_selected_tokens, dtype=torch.long), sorted=True)
            stable_idx = stable_idx[(stable_idx >= 0) & (stable_idx < meta["is_reference"].numel())]
            if stable_idx.numel() > 0:
                meta["is_reference"][stable_idx] = True

        if not selected:
            return None

        logger.debug(
            "[geo_prune] total=%d candidate=%d visible=%d selected=%d anchor_selected=%d stable_selected=%d tau_bucket=%.4f",
            total_tokens,
            candidate_count,
            visible_total,
            len(selected),
            anchor_count,
            stable_count,
            tau_bucket,
        )
        keep = torch.tensor(sorted(i for i in selected if 0 <= i < total_tokens), dtype=torch.long)
        if max_past_tokens is not None and keep.numel() > int(max_past_tokens):
            keep = self._cap_keep_with_protection(
                meta,
                keep,
                budget=max(0, int(max_past_tokens)),
                recent_frames=recent_frames,
            )
        self.geo_cached_landmark_keep = self._extract_landmark_cache(meta, keep, max_past_tokens)
        overlap = self._count_keep_cache_overlap(keep, self.geo_cached_landmark_keep)
        reanchor_overlap_avg = (float(reanchor_overlap_sum) / float(max(1, reanchor_frames_used))) if reanchor_frames_used > 0 else 0.0
        self._maybe_console_geo_log(
            current_frame_idx=current_frame_idx,
            total_tokens=total_tokens,
            candidate_count=candidate_count,
            visible_total=visible_total,
            selected_count=int(keep.numel()),
            anchor_count=anchor_count,
            stable_count=stable_count,
            tau_bucket=tau_bucket,
            stable_visible_voxel_overlap=stable_visible_voxel_overlap,
            stable_selected_visible=stable_visible_selected,
            stable_selected_invisible=stable_invisible_selected,
            fast_path=0,
            cache_size=int(self.geo_cached_landmark_keep.numel()),
            keep_overlap_cache=overlap,
            reanchor_added=int(reanchor_added),
            reanchor_overlap_avg=float(reanchor_overlap_avg),
            budget=int(max_past_tokens or 0),
        )
        return keep
    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        total_budget=0,
        use_geo_kv_prune: bool = False,
        geo_topk_per_voxel: int = 4,
        geo_recent_frames: int = 2,
        geo_near: float = 0.05,
        geo_far: float = 200.0,
        current_view: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if use_cache and past_key_values[0] is not None:
            # _, _, S_true, _, _ = past_key_values[0][0].shape
            S_true = past_frame_idx + 1
        else:
            S_true = S
        
        if use_cache and S > 1:
            print(f"Use KV cache expects S=1, got S={S}")

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean.to(images.device)) / self._resnet_std.to(images.device)

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.reshape(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if use_cache:
            # Streaming fast-path: only materialize current-frame special tokens.
            token_sel = 0 if past_frame_idx == 0 else 1
            camera_token = self.camera_token[:, token_sel : token_sel + 1, ...].expand(B, 1, 1, C).reshape(B * S, 1, C)
            register_token = self.register_token[:, token_sel : token_sel + 1, ...].expand(B, 1, self.register_token.shape[2], C).reshape(
                B * S, self.register_token.shape[2], C
            )
        else:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        current_budgets = self._calculate_dynamic_budgets(total_budget)
        scores = []

        # In geo mode, build one shared keep plan per frame instead of re-running
        # expensive Python/CPU geo selection for every global layer.
        geo_shared_identity_keep: Optional[torch.Tensor] = None
        geo_prune_ready = False
        geo_reloc_active = False
        if use_cache and use_geo_kv_prune and any(kv is not None for kv in past_key_values):
            ref_layer_idx = None
            for idx, kv in enumerate(past_key_values):
                if kv is not None and self.geo_token_meta[idx]["frame_idx"].numel() > 0:
                    ref_layer_idx = idx
                    break

            if ref_layer_idx is not None:
                ref_meta = self.geo_token_meta[ref_layer_idx]
                raw_ref_budget = max(0, int(current_budgets.min().item()) - P)
                ref_budget = min(raw_ref_budget, int(self.geo_layer_budget_cap))
                cache_frame_idx = int(ref_meta["frame_idx"].max().item()) if ref_meta["frame_idx"].numel() > 0 else max(-1, int(past_frame_idx) - 1)
                geo_reloc_active = int(self.geo_reloc_frames_left) > 0 or str(self.geo_reloc_state) != "off"
                geo_prune_ready = self._geo_prune_ready(ref_meta, ref_budget, cache_frame_idx)

                if geo_reloc_active:
                    geo_shared_identity_keep = self._build_reloc_identity_keep(
                        meta=ref_meta,
                        max_past_tokens=max(0, int(ref_budget)),
                        recent_frames=max(1, int(geo_recent_frames)),
                    )
                elif geo_prune_ready:
                    geo_shared_keep_idx = self._select_geo_active_indices(
                        meta=ref_meta,
                        topk_per_voxel=geo_topk_per_voxel,
                        recent_frames=geo_recent_frames,
                        near=geo_near,
                        far=geo_far,
                        current_view=current_view,
                        max_past_tokens=ref_budget,
                    )
                    protected_ref = torch.nonzero(self._hard_protected_mask(ref_meta), as_tuple=False).view(-1)
                    if geo_shared_keep_idx is None or geo_shared_keep_idx.numel() == 0:
                        geo_shared_keep_idx = protected_ref
                    elif protected_ref.numel() > 0:
                        geo_shared_keep_idx = torch.unique(
                            torch.cat([geo_shared_keep_idx.detach().cpu().long(), protected_ref], dim=0),
                            sorted=True,
                        )
                    geo_shared_identity_keep = self._build_identity_keep_from_meta(ref_meta, geo_shared_keep_idx)
                else:
                    geo_shared_identity_keep = None


        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if use_cache:
                        layer_idx = global_idx
                        raw_layer_budget = int(current_budgets[layer_idx].item())
                        layer_budget = min(raw_layer_budget, int(self.geo_layer_budget_cap)) if use_geo_kv_prune else raw_layer_budget
                        debug_protected = torch.empty((0,), dtype=torch.long)
                        debug_keep_idx = torch.empty((0,), dtype=torch.long)
                        debug_pre_keep = torch.empty((0,), dtype=torch.long)
                        past_kv_block = past_key_values[layer_idx] if past_key_values[layer_idx] is not None else None
                        kv_before_len = int(past_kv_block[0].shape[2]) if past_kv_block is not None else 0
                        past_meta = self.geo_token_meta[layer_idx]

                        if use_geo_kv_prune and past_kv_block is not None:
                            max_past_tokens = max(0, layer_budget - P)
                            if geo_reloc_active:
                                layer_identity_keep = self._build_reloc_identity_keep(
                                    meta=past_meta,
                                    max_past_tokens=max_past_tokens,
                                    recent_frames=max(1, int(geo_recent_frames)),
                                )
                                keep_idx = self._identity_keep_to_index(past_meta, layer_identity_keep)
                            elif geo_prune_ready:
                                layer_identity_keep = self._cap_identity_keep_with_protection(
                                    past_meta,
                                    geo_shared_identity_keep
                                    if geo_shared_identity_keep is not None
                                    else torch.empty((0,), dtype=torch.long),
                                    budget=max_past_tokens,
                                    recent_frames=geo_recent_frames,
                                )
                                keep_idx = self._identity_keep_to_index(past_meta, layer_identity_keep)
                            else:
                                kv_len = int(past_kv_block[0].shape[2])
                                if kv_len <= max_past_tokens:
                                    keep_idx = torch.arange(kv_len, dtype=torch.long)
                                else:
                                    keep_idx = self._simple_non_geo_keep(
                                        past_meta,
                                        budget=max_past_tokens,
                                        recent_frames=geo_recent_frames,
                                    )

                            debug_keep_idx = keep_idx
                            keep_idx = self._sanitize_keep_idx(
                                keep_idx,
                                meta_len=past_meta["frame_idx"].numel(),
                                kv_len=past_kv_block[0].shape[2],
                            )

                            protected_idx = torch.nonzero(self._hard_protected_mask(past_meta), as_tuple=False).view(-1)
                            debug_protected = protected_idx
                            if keep_idx.numel() > 0 and protected_idx.numel() > 0:
                                pre_keep_all = torch.unique(
                                    torch.cat([keep_idx.detach().cpu().long(), protected_idx], dim=0),
                                    sorted=True,
                                )
                            elif keep_idx.numel() > 0:
                                pre_keep_all = keep_idx
                            elif protected_idx.numel() > 0:
                                pre_keep_all = protected_idx
                            else:
                                pre_keep_all = torch.arange(past_kv_block[0].shape[2], dtype=torch.long)
                            pre_keep = self._cap_keep_with_protection(
                                past_meta,
                                pre_keep_all,
                                budget=max_past_tokens,
                                recent_frames=geo_recent_frames,
                            )
                            debug_pre_keep = pre_keep
                            pre_keep = self._sanitize_keep_idx(
                                pre_keep,
                                meta_len=past_meta["frame_idx"].numel(),
                                kv_len=past_kv_block[0].shape[2],
                            )
                            if pre_keep.numel() == 0:
                                past_kv_block = None
                                past_meta = self.geo_token_meta[layer_idx]
                            elif not self._is_full_range_keep(pre_keep, past_kv_block[0].shape[2]):
                                pre_keep_dev = pre_keep.to(past_kv_block[0].device)
                                past_kv_block = (
                                    torch.index_select(past_kv_block[0], 2, pre_keep_dev),
                                    torch.index_select(past_kv_block[1], 2, pre_keep_dev),
                                )
                                past_meta = self._index_meta(past_meta, pre_keep)

                        tokens, global_idx, global_intermediates, new_kv, current_scores = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_kv_block,
                            use_cache=True,
                            past_frame_idx=past_frame_idx,
                            cache_budget=layer_budget
                        )

                        if use_geo_kv_prune:
                            current_meta = self._build_current_frame_meta(past_frame_idx, P)
                            merged_meta = self._concat_meta(past_meta, current_meta)
                            merged_meta["is_anchor"] = self._derive_anchor_mask_from_meta(merged_meta)
                            merged_meta["is_landmark"] = self._derive_landmark_mask_from_meta(merged_meta)
                            merged_meta["is_reference"] = self._derive_reference_mask_from_meta(merged_meta)

                            # Keep explicit hard cap in geo mode with protection for anchor/special/recent tokens.
                            if new_kv[0].shape[2] > layer_budget:
                                cap_all = torch.arange(new_kv[0].shape[2], dtype=torch.long)
                                cap_keep = self._cap_keep_with_protection(
                                    merged_meta,
                                    cap_all,
                                    budget=layer_budget,
                                    recent_frames=geo_recent_frames,
                                )
                                cap_keep = self._sanitize_keep_idx(
                                    cap_keep,
                                    meta_len=merged_meta["frame_idx"].numel(),
                                    kv_len=new_kv[0].shape[2],
                                )
                                cap_keep_dev = cap_keep.to(new_kv[0].device)
                                new_kv = (
                                    torch.index_select(new_kv[0], 2, cap_keep_dev),
                                    torch.index_select(new_kv[1], 2, cap_keep_dev),
                                )
                                merged_meta = self._index_meta(merged_meta, cap_keep)
                            frame0_in_cache = int((merged_meta["frame_idx"] == 0).sum().item())
                            ref_in_cache = int(merged_meta.get("is_reference", torch.zeros_like(merged_meta["is_special"])).sum().item())
                            landmark_in_cache = int(merged_meta.get("is_landmark", torch.zeros_like(merged_meta["is_special"])).sum().item())
                            anchor_in_cache = int(merged_meta.get("is_anchor", torch.zeros_like(merged_meta["is_special"])).sum().item())
                            logger.info(
                                "[geo_debug] layer=%d kv_before=%d meta_before=%d protected=%d keep_idx=%d pre_keep=%d new_kv=%d merged_meta=%d layer_budget=%d trust=%.4f recovery=%d reloc=%d frame0_in_cache=%d ref_in_cache=%d landmark_in_cache=%d anchor_in_cache=%d",
                                int(layer_idx),
                                int(kv_before_len),
                                int(past_meta["frame_idx"].numel()) if past_meta is not None and "frame_idx" in past_meta else 0,
                                int(debug_protected.numel()),
                                int(debug_keep_idx.numel()),
                                int(debug_pre_keep.numel()),
                                int(new_kv[0].shape[2]),
                                int(merged_meta["frame_idx"].numel()),
                                int(layer_budget),
                                float(self.geo_trust_score),
                                int(self.geo_recovery_frames_left),
                                int(self.geo_reloc_frames_left),
                                int(frame0_in_cache),
                                int(ref_in_cache),
                                int(landmark_in_cache),
                                int(anchor_in_cache),
                            )
                            assert int(merged_meta["frame_idx"].numel()) == int(new_kv[0].shape[2]), "geo meta/KV length mismatch"
                            self.geo_token_meta[layer_idx] = merged_meta

                        past_key_values[global_idx - 1] = new_kv
                        if current_scores is not None: # pruning happened
                            scores.append(current_scores)
                        else:
                            scores.append(self.last_scores[global_idx-1].item())
                    else: 
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
        if scores: # update scores
            self.last_scores = torch.tensor(scores, device=self.last_scores.device, dtype=self.last_scores.dtype)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        if use_cache:      
            return output_list, self.patch_start_idx, past_key_values
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):

            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        use_cache=False,
        past_frame_idx=0,
        cache_budget=None
    ) -> Union[Tuple[torch.Tensor, int, List[torch.Tensor]], Tuple[torch.Tensor, int, List[torch.Tensor], List]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
                """
        
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)
            
        intermediates = []

        for _ in range(self.aa_block_size):
            if not use_cache:
                L = S * P
                frame_ids = torch.arange(L, device=tokens.device) // P  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(tokens.dtype) * torch.finfo(tokens.dtype).min
            else:
                attn_mask = None
            
            scores = None
            if use_cache:
                tokens, block_kv, scores = self.global_blocks[global_idx](
                    tokens, 
                    pos=pos, 
                    attn_mask=attn_mask, 
                    past_key_values=past_key_values_block,
                    use_cache=True,
                    cache_budget=cache_budget
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, attn_mask=attn_mask)

            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if use_cache:
            return tokens, global_idx, intermediates, block_kv, scores
        return tokens, global_idx, intermediates
        
    def _calculate_dynamic_budgets(self, total_budget):

        with torch.no_grad():
            diversity_scores = 1.0 - self.last_scores
            scaled_scores = diversity_scores / 0.5
            proportions = torch.softmax(scaled_scores, dim=0)
            if total_budget < 0:
                total_budget = 0
            budgets = proportions * total_budget

        return budgets.int()


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined
