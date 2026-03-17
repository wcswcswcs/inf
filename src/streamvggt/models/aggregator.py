# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import heapq
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any, Iterable
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
        geo_recovery_new_voxel_quota_per_frame: int = 32,
        geo_recovery_reference_refresh: bool = True,
        geo_recovery_landmark_quota_per_frame: int = 16,
        geo_recovery_reference_quota_per_frame: int = 8,
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
        geo_layer_budget_cap: int = 65536,
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
        geo_bootstrap_until: int = 8,
        geo_legacy_selector_until: int = 768,
        geo_legacy_hard_recent_frames: int = 16,
        geo_legacy_recent_window: int = 20,
        geo_legacy_soft_recent_frames: int = 24,
        geo_legacy_min_keep_per_recent_frame: int = 256,
        geo_hard_recent_frames: int = 6,
        geo_early_stabilize_frames: int = 200,
        geo_early_recent_frames: int = 8,
        geo_early_budget_floor: int = 16384,
        geo_cap_ramp_start: int = 960,
        geo_cap_ramp_end: int = 1216,
        geo_anchor_enable_after: int = 0,
        geo_landmark_enable_after: int = 128,
        geo_reference_enable_after: int = 192,
        geo_reloc_enable_after: int = 192,
        geo_stable_map_ratio_runtime: float = 0.25,
        geo_stable_min_voxels_runtime: int = 128,
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
        self.geo_recovery_new_voxel_quota_per_frame = max(0, int(geo_recovery_new_voxel_quota_per_frame))
        self.geo_recovery_reference_refresh = bool(geo_recovery_reference_refresh)
        self.geo_recovery_landmark_quota_per_frame = max(0, int(geo_recovery_landmark_quota_per_frame))
        self.geo_recovery_reference_quota_per_frame = max(0, int(geo_recovery_reference_quota_per_frame))
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
        self.geo_bootstrap_until = max(0, int(geo_bootstrap_until))
        self.geo_legacy_selector_until = max(int(self.geo_bootstrap_until), int(geo_legacy_selector_until))  # deprecated: retained for backward config compatibility; not used in current logic
        self.geo_legacy_hard_recent_frames = max(1, int(geo_legacy_hard_recent_frames))
        self.geo_legacy_recent_window = max(1, int(geo_legacy_recent_window))
        self.geo_legacy_soft_recent_frames = max(1, int(geo_legacy_soft_recent_frames))
        self.geo_legacy_min_keep_per_recent_frame = max(1, int(geo_legacy_min_keep_per_recent_frame))
        self.geo_hard_recent_frames = max(1, int(geo_hard_recent_frames))
        self.geo_early_stabilize_frames = max(0, int(geo_early_stabilize_frames))
        self.geo_early_recent_frames = max(1, int(geo_early_recent_frames))
        self.geo_early_budget_floor = max(0, int(geo_early_budget_floor))
        self.geo_cap_ramp_start = max(0, int(geo_cap_ramp_start))  # deprecated: retained for backward config compatibility; not used in current logic
        self.geo_cap_ramp_end = max(int(self.geo_cap_ramp_start) + 1, int(geo_cap_ramp_end))  # deprecated: retained for backward config compatibility; not used in current logic
        self.geo_anchor_enable_after = max(0, int(geo_anchor_enable_after))
        self.geo_landmark_enable_after = max(int(self.geo_bootstrap_frames), int(geo_landmark_enable_after))
        self.geo_reference_enable_after = max(int(self.geo_landmark_enable_after), int(geo_reference_enable_after))
        self.geo_reloc_enable_after = max(int(self.geo_reference_enable_after), int(geo_reloc_enable_after))
        self.geo_stable_map_ratio_runtime = float(min(max(geo_stable_map_ratio_runtime, 0.05), 0.9))
        self.geo_stable_min_voxels_runtime = max(32, int(geo_stable_min_voxels_runtime))
        self.geo_identity_stride = 1 << 21
        self.geo_identity_offset = 1 << 18
        # Hard-backbone defaults (kept for config compatibility of old-stable behavior).
        self.geo_frame0_backbone_quota = 512
        self.geo_reference_hard_quota = 1024
        self.geo_anchor_hard_quota = 1024
        self.geo_keyframe_hard_quota = 256
        self.geo_bootstrap_min_stable_anchors = 256
        self.geo_bootstrap_ref_overlap_thr = 128.0
        self.geo_bootstrap_visible_ratio_thr = 0.50
        self.geo_bootstrap_ready_streak = 8
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
        self.geo_structure_ready_streak: int = 0
        self.geo_structure_ready_latched: bool = False
        self.geo_structure_unready_streak: int = 0
        self.geo_reloc_frames_left: int = 0
        self.geo_reloc_state: str = "off"
        self.geo_reloc_hard_left: int = 0
        self.geo_reloc_good_streak: int = 0
        # Legacy positional cache is intentionally disabled; continuity uses identity cache only.
        self.geo_cached_landmark_keep: torch.Tensor = torch.empty((0,), dtype=torch.long)
        self.geo_cached_landmark_identity_keep: torch.Tensor = torch.empty((0,), dtype=torch.long)
        self.geo_trim_cursor = 0
        self.geo_last_console_log_frame = -1
        self.geo_pending_console_log: Optional[Dict[str, Any]] = None
        self.geo_runtime_ready_latched: bool = False
        self.geo_runtime_ready_streak: int = 0
        self.geo_runtime_unready_streak: int = 0
        self.geo_runtime_ready_last_frame: int = -1
        self.geo_selector_mode: str = "legacy"
        self.geo_maturity_ema: float = 0.0
        self.geo_instability_ema: float = 0.0
        self.geo_pressure_ema: float = 0.0
        self.geo_motion_ema: float = 0.0
        self.geo_confdrop_ema: float = 0.0
        self.geo_matched_ema: float = 0.0
        self.geo_new_voxel_ema: float = 0.0
        self.geo_ref_overlap_ema: float = 0.0
        self.geo_selector_overlap_ema: float = 0.0
        self.geo_selector_visible_ratio_ema: float = 0.0
        self.geo_handover_ready_streak: int = 0
        self.geo_handover_unready_streak: int = 0
        self.geo_recovery_enter_streak: int = 0
        self.geo_recovery_exit_streak: int = 0
        self.geo_last_observation: Dict[str, float] = {
            "frame_idx": -1,
            "matched_ratio": 0.0,
            "new_voxel_ratio": 0.0,
            "ref_overlap": 0.0,
            "trust_score": 1.0,
        }
        self.geo_last_selector_diag: Dict[str, float] = {
            "frame_idx": -1,
            "stable_visible_overlap": 0.0,
            "stable_visible_ratio": 0.0,
            "visible_total": 0.0,
            "selected_total": 0.0,
        }
        self.geo_last_committed_policy: Optional[Dict[str, Any]] = None
        self.geo_last_policy_frame: int = -1
        self.geo_last_policy_inputs: Dict[str, Any] = {}
        self.geo_last_policy_metrics: Dict[str, float] = {}
        self.geo_last_commit_guard_frame: int = -1
        self.geo_last_debug_log_frame = -1
        self.geo_last_bootstrap_log_frame = -1
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
                "geo_role": torch.empty(0, dtype=torch.long),
            }
            for i in range(self.depth)
        }

    def export_geo_cache_state(self) -> Dict[str, Any]:
        def _clone_tensor_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in d.items():
                if torch.is_tensor(v):
                    out[k] = v.detach().cpu().clone()
                elif isinstance(v, dict):
                    out[k] = _clone_tensor_dict(v)
                else:
                    out[k] = copy.deepcopy(v)
            return out

        return {
            "geo_frame_meta": _clone_tensor_dict(self.geo_frame_meta),
            "geo_frame_anchor_mask": copy.deepcopy(self.geo_frame_anchor_mask),
            "geo_voxel_bank": copy.deepcopy(self.geo_voxel_bank),
            "geo_anchor_voxels": set(self.geo_anchor_voxels),
            "geo_anchor_voxel_list": list(self.geo_anchor_voxel_list),
            "geo_anchor_birth": copy.deepcopy(self.geo_anchor_birth),
            "geo_stable_anchor_voxels": set(self.geo_stable_anchor_voxels),
            "geo_stable_anchor_voxel_list": list(self.geo_stable_anchor_voxel_list),
            "geo_landmark_voxels": set(self.geo_landmark_voxels),
            "geo_landmark_voxel_list": list(self.geo_landmark_voxel_list),
            "geo_landmark_birth": copy.deepcopy(self.geo_landmark_birth),
            "geo_reference_bank": copy.deepcopy(self.geo_reference_bank),
            "geo_reference_voxels": set(self.geo_reference_voxels),
            "geo_reference_voxel_list": list(self.geo_reference_voxel_list),
            "geo_stable_map_voxels": set(self.geo_stable_map_voxels),
            "geo_adaptive_map_voxels": set(self.geo_adaptive_map_voxels),
            "geo_keyframes": list(self.geo_keyframes),
            "geo_keyframe_set": set(self.geo_keyframe_set),
            "geo_keyframe_frozen_local": {
                int(k): v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                for k, v in self.geo_keyframe_frozen_local.items()
            },
            "geo_keyframe_pyramid": copy.deepcopy(self.geo_keyframe_pyramid),
            "geo_token_meta": {
                int(layer_idx): _clone_tensor_dict(layer_meta)
                for layer_idx, layer_meta in self.geo_token_meta.items()
            },
            "geo_runtime_ready_latched": bool(self.geo_runtime_ready_latched),
            "geo_runtime_ready_streak": int(self.geo_runtime_ready_streak),
            "geo_runtime_unready_streak": int(self.geo_runtime_unready_streak),
            "geo_runtime_ready_last_frame": int(self.geo_runtime_ready_last_frame),
            "geo_recovery_frames_left": int(self.geo_recovery_frames_left),
            "geo_reloc_state": str(self.geo_reloc_state),
            "geo_reloc_frames_left": int(self.geo_reloc_frames_left),
            "geo_reloc_hard_left": int(self.geo_reloc_hard_left),
            "geo_reloc_good_streak": int(self.geo_reloc_good_streak),
            "geo_trust_score": float(self.geo_trust_score),
            "geo_anchor_version": int(self.geo_anchor_version),
            "geo_frame_anchor_version": copy.deepcopy(self.geo_frame_anchor_version),
            "geo_trim_cursor": int(self.geo_trim_cursor),
            "geo_cached_landmark_identity_keep": self.geo_cached_landmark_identity_keep.detach().cpu().clone(),
            "geo_pending_console_log": copy.deepcopy(self.geo_pending_console_log),
            "geo_last_console_log_frame": int(self.geo_last_console_log_frame),
            "geo_last_debug_log_frame": int(self.geo_last_debug_log_frame),
            "geo_last_bootstrap_log_frame": int(self.geo_last_bootstrap_log_frame),
            "geo_selector_mode": str(self.geo_selector_mode),
            "geo_structure_ready_streak": int(self.geo_structure_ready_streak),
            "geo_structure_ready_latched": bool(self.geo_structure_ready_latched),
            "geo_structure_unready_streak": int(self.geo_structure_unready_streak),
            "geo_maturity_ema": float(self.geo_maturity_ema),
            "geo_instability_ema": float(self.geo_instability_ema),
            "geo_pressure_ema": float(self.geo_pressure_ema),
            "geo_motion_ema": float(self.geo_motion_ema),
            "geo_confdrop_ema": float(self.geo_confdrop_ema),
            "geo_matched_ema": float(self.geo_matched_ema),
            "geo_new_voxel_ema": float(self.geo_new_voxel_ema),
            "geo_ref_overlap_ema": float(self.geo_ref_overlap_ema),
            "geo_selector_overlap_ema": float(self.geo_selector_overlap_ema),
            "geo_selector_visible_ratio_ema": float(self.geo_selector_visible_ratio_ema),
            "geo_handover_ready_streak": int(self.geo_handover_ready_streak),
            "geo_handover_unready_streak": int(self.geo_handover_unready_streak),
            "geo_recovery_enter_streak": int(self.geo_recovery_enter_streak),
            "geo_recovery_exit_streak": int(self.geo_recovery_exit_streak),
            "geo_last_observation": copy.deepcopy(self.geo_last_observation),
            "geo_last_selector_diag": copy.deepcopy(self.geo_last_selector_diag),
            "geo_last_committed_policy": copy.deepcopy(self.geo_last_committed_policy),
            "geo_last_policy_frame": int(self.geo_last_policy_frame),
            "geo_last_policy_inputs": copy.deepcopy(self.geo_last_policy_inputs),
            "geo_last_policy_metrics": copy.deepcopy(self.geo_last_policy_metrics),
            "geo_last_commit_guard_frame": int(self.geo_last_commit_guard_frame),
            "last_scores": self.last_scores.detach().cpu().clone(),
        }

    def load_geo_cache_state(self, state: Dict[str, Any]) -> None:
        def _clone_tensor_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in d.items():
                if torch.is_tensor(v):
                    out[k] = v.detach().cpu().clone()
                elif isinstance(v, dict):
                    out[k] = _clone_tensor_dict(v)
                else:
                    out[k] = copy.deepcopy(v)
            return out

        self.geo_frame_meta = _clone_tensor_dict(state.get("geo_frame_meta", {}))
        self.geo_frame_anchor_mask = copy.deepcopy(state.get("geo_frame_anchor_mask", {}))
        self.geo_voxel_bank = copy.deepcopy(state.get("geo_voxel_bank", {}))
        self.geo_anchor_voxels = set(state.get("geo_anchor_voxels", set()))
        self.geo_anchor_voxel_list = list(state.get("geo_anchor_voxel_list", []))
        self.geo_anchor_birth = copy.deepcopy(state.get("geo_anchor_birth", {}))
        self.geo_stable_anchor_voxels = set(state.get("geo_stable_anchor_voxels", set()))
        self.geo_stable_anchor_voxel_list = list(state.get("geo_stable_anchor_voxel_list", []))
        self.geo_landmark_voxels = set(state.get("geo_landmark_voxels", set()))
        self.geo_landmark_voxel_list = list(state.get("geo_landmark_voxel_list", []))
        self.geo_landmark_birth = copy.deepcopy(state.get("geo_landmark_birth", {}))
        self.geo_reference_bank = copy.deepcopy(state.get("geo_reference_bank", {}))
        self.geo_reference_voxels = set(state.get("geo_reference_voxels", set()))
        self.geo_reference_voxel_list = list(state.get("geo_reference_voxel_list", []))
        self.geo_stable_map_voxels = set(state.get("geo_stable_map_voxels", set()))
        self.geo_adaptive_map_voxels = set(state.get("geo_adaptive_map_voxels", set()))
        self.geo_keyframes = list(state.get("geo_keyframes", []))
        self.geo_keyframe_set = set(state.get("geo_keyframe_set", set()))
        self.geo_keyframe_frozen_local = {
            int(k): v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in state.get("geo_keyframe_frozen_local", {}).items()
        }
        self.geo_keyframe_pyramid = copy.deepcopy(state.get("geo_keyframe_pyramid", {0: [], 1: [], 2: []}))
        token_meta_state = state.get("geo_token_meta")
        if isinstance(token_meta_state, dict):
            self.geo_token_meta = {
                int(layer_idx): _clone_tensor_dict(layer_meta)
                for layer_idx, layer_meta in token_meta_state.items()
            }
            for layer_idx, layer_meta in self.geo_token_meta.items():
                if "geo_role" not in layer_meta:
                    layer_meta["geo_role"] = self._compute_primary_geo_role(layer_meta)
        self.geo_runtime_ready_latched = bool(state.get("geo_runtime_ready_latched", False))
        self.geo_runtime_ready_streak = int(state.get("geo_runtime_ready_streak", 0))
        self.geo_runtime_unready_streak = int(state.get("geo_runtime_unready_streak", 0))
        self.geo_runtime_ready_last_frame = int(state.get("geo_runtime_ready_last_frame", -1))
        self.geo_recovery_frames_left = int(state.get("geo_recovery_frames_left", 0))
        self.geo_reloc_state = str(state.get("geo_reloc_state", "off"))
        self.geo_reloc_frames_left = int(state.get("geo_reloc_frames_left", 0))
        self.geo_reloc_hard_left = int(state.get("geo_reloc_hard_left", 0))
        self.geo_reloc_good_streak = int(state.get("geo_reloc_good_streak", 0))
        self.geo_trust_score = float(state.get("geo_trust_score", 1.0))
        self.geo_anchor_version = int(state.get("geo_anchor_version", 0))
        self.geo_frame_anchor_version = copy.deepcopy(state.get("geo_frame_anchor_version", {}))
        # Legacy positional cache is ignored by design to avoid index-based continuity drift.
        self.geo_cached_landmark_keep = torch.empty((0,), dtype=torch.long)

        cached_identity_keep = state.get("geo_cached_landmark_identity_keep", torch.empty((0,), dtype=torch.long))
        if torch.is_tensor(cached_identity_keep):
            self.geo_cached_landmark_identity_keep = cached_identity_keep.detach().cpu().to(torch.long).clone()
        else:
            # Backward compatibility: do not reuse legacy positional cache across frames.
            self.geo_cached_landmark_identity_keep = torch.empty((0,), dtype=torch.long)
        self.geo_trim_cursor = int(state.get("geo_trim_cursor", 0))
        self.geo_pending_console_log = copy.deepcopy(state.get("geo_pending_console_log", None))
        self.geo_last_console_log_frame = int(state.get("geo_last_console_log_frame", -1))
        self.geo_last_debug_log_frame = int(state.get("geo_last_debug_log_frame", -1))
        self.geo_last_bootstrap_log_frame = int(state.get("geo_last_bootstrap_log_frame", -1))
        self.geo_selector_mode = str(state.get("geo_selector_mode", "legacy"))
        self.geo_structure_ready_streak = int(state.get("geo_structure_ready_streak", 0))
        self.geo_structure_ready_latched = bool(state.get("geo_structure_ready_latched", False))
        self.geo_structure_unready_streak = int(state.get("geo_structure_unready_streak", 0))
        _ = state.get("geo_current_selector_latched", None)
        _ = state.get("geo_current_selector_ready_streak", None)
        _ = state.get("geo_current_selector_unready_streak", None)
        self.geo_maturity_ema = float(state.get("geo_maturity_ema", 0.0))
        self.geo_instability_ema = float(state.get("geo_instability_ema", 0.0))
        self.geo_pressure_ema = float(state.get("geo_pressure_ema", 0.0))
        self.geo_motion_ema = float(state.get("geo_motion_ema", 0.0))
        self.geo_confdrop_ema = float(state.get("geo_confdrop_ema", 0.0))
        self.geo_matched_ema = float(state.get("geo_matched_ema", 0.0))
        self.geo_new_voxel_ema = float(state.get("geo_new_voxel_ema", 0.0))
        self.geo_ref_overlap_ema = float(state.get("geo_ref_overlap_ema", 0.0))
        self.geo_selector_overlap_ema = float(state.get("geo_selector_overlap_ema", 0.0))
        self.geo_selector_visible_ratio_ema = float(state.get("geo_selector_visible_ratio_ema", 0.0))
        self.geo_handover_ready_streak = int(state.get("geo_handover_ready_streak", 0))
        self.geo_handover_unready_streak = int(state.get("geo_handover_unready_streak", 0))
        self.geo_recovery_enter_streak = int(state.get("geo_recovery_enter_streak", 0))
        self.geo_recovery_exit_streak = int(state.get("geo_recovery_exit_streak", 0))
        legacy_stats = copy.deepcopy(state.get("geo_last_frame_stats", {}))
        self.geo_last_observation = copy.deepcopy(state.get("geo_last_observation", self.geo_last_observation))
        if (not isinstance(self.geo_last_observation, dict)) or int(self.geo_last_observation.get("frame_idx", -1)) < 0:
            self.geo_last_observation = {
                "frame_idx": int(state.get("geo_last_observation_frame", -1)),
                "matched_ratio": float(legacy_stats.get("matched_ratio", 0.0)),
                "new_voxel_ratio": float(legacy_stats.get("new_voxel_ratio", 0.0)),
                "ref_overlap": float(legacy_stats.get("ref_overlap", 0.0)),
                "trust_score": float(self.geo_trust_score),
            }
        self.geo_last_selector_diag = copy.deepcopy(state.get("geo_last_selector_diag", self.geo_last_selector_diag))
        self.geo_last_committed_policy = copy.deepcopy(state.get("geo_last_committed_policy", None))
        self.geo_last_policy_frame = int(state.get("geo_last_policy_frame", -1))
        self.geo_last_policy_inputs = copy.deepcopy(state.get("geo_last_policy_inputs", {}))
        self.geo_last_policy_metrics = copy.deepcopy(state.get("geo_last_policy_metrics", {}))
        self.geo_last_commit_guard_frame = int(state.get("geo_last_commit_guard_frame", -1))
        scores_state = state.get("last_scores", None)
        if torch.is_tensor(scores_state):
            self.last_scores = scores_state.detach().cpu().to(self.last_scores.dtype).clone()
        self.geo_max_frame_idx = max(self.geo_keyframes) if self.geo_keyframes else max(self.geo_frame_meta.keys(), default=-1)
        self._refresh_runtime_hash_tensors()

    def _refresh_runtime_hash_tensors(self) -> None:
        self.geo_anchor_voxels = set(self.geo_anchor_voxel_list)
        self.geo_stable_anchor_voxels = set(self.geo_stable_anchor_voxel_list)
        self.geo_landmark_voxels = set(self.geo_landmark_voxel_list)
        self.geo_reference_voxels = set(self.geo_reference_voxel_list)

        def _hash_from_voxels(vox_list):
            if not vox_list:
                return torch.empty((0,), dtype=torch.long)
            vox = torch.tensor([list(v) for v in vox_list], dtype=torch.int32)
            return self._voxel_hash(vox)

        self.geo_anchor_hash_tensor = _hash_from_voxels(self.geo_anchor_voxel_list)
        self.geo_stable_anchor_hash_tensor = _hash_from_voxels(self.geo_stable_anchor_voxel_list)
        self.geo_landmark_hash_tensor = _hash_from_voxels(self.geo_landmark_voxel_list)
        self.geo_reference_hash_tensor = _hash_from_voxels(self.geo_reference_voxel_list)

        if self.geo_frame_meta:
            self.geo_max_frame_idx = max(int(k) for k in self.geo_frame_meta.keys())
        else:
            self.geo_max_frame_idx = -1

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
        quota_override: Optional[int] = None,
    ):
        if int(self.geo_landmark_per_keyframe) <= 0 or uniq_vox.numel() == 0:
            return
        if int(frame_idx) not in self.geo_keyframe_set:
            return

        quota_cfg = int(self.geo_landmark_per_keyframe) if quota_override is None else int(quota_override)
        quota = min(int(max(0, quota_cfg)), int(uniq_vox.shape[0]))
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
            ranked_keep = sorted(
                self.geo_landmark_voxels,
                key=lambda k: (
                    int(self.geo_landmark_birth.get(k, frame_idx)),
                    float(self.geo_voxel_bank.get(k, {}).get("conf_ema", 0.0)),
                    -float(self.geo_voxel_bank.get(k, {}).get("pos_var", 0.0)),
                    k,
                ),
                reverse=True,
            )
            keep = set(ranked_keep[: int(self.geo_landmark_max_count)])
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
        quota_override: Optional[int] = None,
        allow_existing_refresh: bool = False,
    ):
        if int(frame_idx) not in self.geo_keyframe_set:
            return
        if uniq_vox.numel() == 0:
            return

        quota_new = int(1e9) if quota_override is None else max(0, int(quota_override))
        new_added = 0
        for i in range(int(uniq_vox.shape[0])):
            key = tuple(int(v) for v in uniq_vox[i].tolist())
            if key in self.geo_reference_bank:
                if allow_existing_refresh:
                    bank = self.geo_voxel_bank.get(key)
                    if bank is not None:
                        score = float(conf_mean_all[i].item()) * float(bank.get("conf_ema", 0.0))
                        self.geo_reference_bank[key].update(
                            {
                                "pos_x": float(bank["pos_x"]),
                                "pos_y": float(bank["pos_y"]),
                                "pos_z": float(bank["pos_z"]),
                                "score": float(max(float(self.geo_reference_bank[key].get("score", 0.0)), score)),
                            }
                        )
                continue
            if new_added >= quota_new:
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
            new_added += 1

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
        bank_size = int(len(scored))
        if bank_size <= 0:
            self.geo_stable_map_voxels = set()
            self.geo_adaptive_map_voxels = set()
            return
        stable_target = int(round(float(bank_size) * float(self.geo_stable_map_ratio_runtime)))
        stable_target = max(stable_target, len(self.geo_stable_anchor_voxel_list), int(self.geo_stable_min_voxels_runtime))
        stable_target = min(stable_target, bank_size)
        stable_upper = max(len(self.geo_stable_anchor_voxel_list) + 256, int(0.5 * bank_size))
        stable_target = min(stable_target, stable_upper, bank_size)
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

    def _ema(self, old: float, new: float, alpha: float = 0.8) -> float:
        return float(alpha) * float(old) + (1.0 - float(alpha)) * float(new)

    def _geo_default_policy(self, frame_idx: int) -> Dict[str, Any]:
        return {
            "mode": "legacy",
            "use_anchor_labels": False,
            "use_landmark_labels": False,
            "use_reference_labels": False,
            "use_recovery": False,
            "use_reloc": False,
            "use_cap": False,
            "cap_alpha": 0.0,
            "hard_recent_frames": int(max(6, int(self.geo_hard_recent_frames))),
            "recent_window": int(max(12, int(self.geo_legacy_recent_window))),
            "soft_recent_window": int(max(16, int(self.geo_legacy_soft_recent_frames))),
            "anchor_quota_ratio": float(max(0.03, min(0.08, float(self.geo_anchor_budget_ratio)))),
            "local_budget_ratio": float(max(0.45, min(0.85, float(self.geo_local_budget_ratio)))),
            "stable_read_budget_ratio": float(max(0.15, min(0.45, float(self.geo_stable_read_budget_ratio)))),
            "frame0_patch_cap": int(max(256, min(2048, int(self.geo_frame0_patch_cap)))),
            "use_view_pruning": True,
            "prefer_last_reliable_view": False,
        }

    def _geo_has_complete_policy_context(self, ref_meta: Optional[Dict[str, torch.Tensor]], ref_past_budget: Optional[int]) -> bool:
        return bool(ref_meta is not None and ref_meta.get("frame_idx") is not None and ref_meta["frame_idx"].numel() > 0 and ref_past_budget is not None and int(ref_past_budget) > 0)

    def _geo_get_last_observation(self) -> Optional[Dict[str, Any]]:
        obs = copy.deepcopy(self.geo_last_observation)
        return None if int(obs.get("frame_idx", -1)) < 0 else obs

    def _geo_get_last_selector_diag(self) -> Optional[Dict[str, Any]]:
        diag = copy.deepcopy(self.geo_last_selector_diag)
        return None if int(diag.get("frame_idx", -1)) < 0 else diag

    def _geo_collect_policy_signals(
        self,
        *,
        frame_idx: int,
        total_tokens: int,
        max_past_tokens: Optional[int],
        current_view: Optional[Dict[str, Any]],
        observation: Optional[Dict[str, Any]],
        selector_diag: Optional[Dict[str, Any]],
    ) -> Dict[str, Optional[float]]:
        pressure: Optional[float] = None
        if max_past_tokens is not None and int(max_past_tokens) > 0:
            pressure = float(total_tokens) / float(max(1, int(max_past_tokens)))

        pose_delta: Optional[float] = None
        conf_drop: Optional[float] = None
        if current_view is not None:
            if current_view.get("pose_delta") is not None:
                pose_delta = float(current_view.get("pose_delta") or 0.0)
            if current_view.get("conf_drop") is not None:
                conf_drop = float(current_view.get("conf_drop") or 0.0)

        matched_ratio: Optional[float] = None
        new_voxel_ratio: Optional[float] = None
        ref_overlap: Optional[float] = None
        trust_score: Optional[float] = None
        if observation is not None:
            matched_ratio = float(observation.get("matched_ratio", 0.0)) if observation.get("matched_ratio") is not None else None
            new_voxel_ratio = float(observation.get("new_voxel_ratio", 0.0)) if observation.get("new_voxel_ratio") is not None else None
            ref_overlap = float(observation.get("ref_overlap", 0.0)) if observation.get("ref_overlap") is not None else None
            trust_score = float(observation.get("trust_score", self.geo_trust_score)) if observation.get("trust_score") is not None else None

        selector_overlap: Optional[float] = None
        selector_visible_ratio: Optional[float] = None
        hard_keep_continuity: Optional[float] = None
        frame0_pin_ratio: Optional[float] = None
        if selector_diag is not None:
            selector_overlap = float(selector_diag.get("stable_visible_overlap", 0.0)) if selector_diag.get("stable_visible_overlap") is not None else None
            selector_visible_ratio = float(selector_diag.get("stable_visible_ratio", 0.0)) if selector_diag.get("stable_visible_ratio") is not None else None
            hard_keep_continuity = float(selector_diag.get("hard_keep_continuity", 1.0)) if selector_diag.get("hard_keep_continuity") is not None else None
            frame0_pin_ratio = float(selector_diag.get("frame0_pin_ratio", 1.0)) if selector_diag.get("frame0_pin_ratio") is not None else None

        return {
            "frame_idx": float(frame_idx),
            "pressure": pressure,
            "pose_delta": pose_delta,
            "conf_drop": conf_drop,
            "matched_ratio": matched_ratio,
            "new_voxel_ratio": new_voxel_ratio,
            "ref_overlap": ref_overlap,
            "trust_score": trust_score,
            "selector_overlap": selector_overlap,
            "selector_visible_ratio": selector_visible_ratio,
            "hard_keep_continuity": hard_keep_continuity,
            "frame0_pin_ratio": frame0_pin_ratio,
        }

    def _geo_preview_maturity(self, *, matched_ema: float, trust_score: Optional[float]) -> float:
        m_vox = min(1.0, float(len(self.geo_voxel_bank)) / 1024.0)
        m_stable = min(1.0, float(len(self.geo_stable_anchor_voxels)) / 256.0)
        m_land = min(1.0, float(len(self.geo_landmark_voxels)) / 256.0)
        m_ref = min(1.0, float(len(self.geo_reference_bank)) / 128.0)
        m_match = min(1.0, float(matched_ema) / 0.25)
        trust_now = float(self.geo_trust_score) if trust_score is None else float(trust_score)
        m_trust = min(1.0, max(0.0, trust_now))
        maturity_raw = 0.20 * m_vox + 0.20 * m_stable + 0.15 * m_land + 0.15 * m_ref + 0.15 * m_match + 0.15 * m_trust
        return self._ema(self.geo_maturity_ema, maturity_raw)

    def _geo_preview_instability(
        self,
        *,
        trust_score: Optional[float],
        matched_ema: float,
        new_voxel_ema: float,
        selector_overlap_ema: float,
        selector_visible_ratio_ema: float,
        motion_ema: float,
        confdrop_ema: float,
        pressure_ema: float,
        hard_keep_continuity: Optional[float],
        frame0_pin_ratio: Optional[float],
    ) -> float:
        trust_now = float(self.geo_trust_score) if trust_score is None else float(trust_score)
        i_trust = 1.0 - min(1.0, max(0.0, trust_now))
        i_match = max(0.0, min(1.0, (0.15 - float(matched_ema)) / 0.15))
        i_novel = min(1.0, float(new_voxel_ema) / 0.4)
        i_overlap = max(0.0, min(1.0, (16.0 - float(selector_overlap_ema)) / 16.0))
        i_vis = max(0.0, min(1.0, (0.6 - float(selector_visible_ratio_ema)) / 0.6))
        i_motion = min(1.0, float(motion_ema) / max(1e-6, float(self.geo_full_select_pose_delta)))
        i_confdrop = min(1.0, float(confdrop_ema) / max(1e-6, float(self.geo_full_select_conf_drop)))
        i_pressure = max(0.0, min(1.0, (float(pressure_ema) - 0.85) / 0.15))
        i_backbone = 0.0 if hard_keep_continuity is None else max(0.0, min(1.0, 1.0 - float(hard_keep_continuity)))
        i_frame0 = 0.0 if frame0_pin_ratio is None else max(0.0, min(1.0, 1.0 - float(frame0_pin_ratio)))
        instability_raw = (
            0.16 * i_trust
            + 0.12 * i_match
            + 0.08 * i_novel
            + 0.12 * i_overlap
            + 0.08 * i_vis
            + 0.08 * i_motion
            + 0.08 * i_confdrop
            + 0.08 * i_pressure
            + 0.10 * i_backbone
            + 0.10 * i_frame0
        )
        return self._ema(self.geo_instability_ema, instability_raw)

    def _geo_preview_adaptive_metrics(self, *, signals: Dict[str, Optional[float]]) -> Dict[str, float]:
        def _ema_preview(old: float, key: str) -> float:
            val = signals.get(key)
            return float(old) if val is None else self._ema(old, float(val))

        pressure_ema = _ema_preview(self.geo_pressure_ema, "pressure")
        motion_ema = _ema_preview(self.geo_motion_ema, "pose_delta")
        confdrop_ema = _ema_preview(self.geo_confdrop_ema, "conf_drop")
        matched_ema = _ema_preview(self.geo_matched_ema, "matched_ratio")
        new_voxel_ema = _ema_preview(self.geo_new_voxel_ema, "new_voxel_ratio")
        ref_overlap_ema = _ema_preview(self.geo_ref_overlap_ema, "ref_overlap")
        selector_overlap_ema = _ema_preview(self.geo_selector_overlap_ema, "selector_overlap")
        selector_visible_ratio_ema = _ema_preview(self.geo_selector_visible_ratio_ema, "selector_visible_ratio")

        maturity = self._geo_preview_maturity(matched_ema=matched_ema, trust_score=signals.get("trust_score"))
        instability = self._geo_preview_instability(
            trust_score=signals.get("trust_score"),
            matched_ema=matched_ema,
            new_voxel_ema=new_voxel_ema,
            selector_overlap_ema=selector_overlap_ema,
            selector_visible_ratio_ema=selector_visible_ratio_ema,
            motion_ema=motion_ema,
            confdrop_ema=confdrop_ema,
            pressure_ema=pressure_ema,
            hard_keep_continuity=signals.get("hard_keep_continuity"),
            frame0_pin_ratio=signals.get("frame0_pin_ratio"),
        )

        return {
            "pressure_ema": pressure_ema,
            "motion_ema": motion_ema,
            "confdrop_ema": confdrop_ema,
            "matched_ema": matched_ema,
            "new_voxel_ema": new_voxel_ema,
            "ref_overlap_ema": ref_overlap_ema,
            "selector_overlap_ema": selector_overlap_ema,
            "selector_visible_ratio_ema": selector_visible_ratio_ema,
            "maturity": maturity,
            "instability": instability,
        }

    def _geo_preview_adaptive_policy(self, *, frame_idx: int, metrics: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        maturity = float(metrics["maturity"])
        instability = float(metrics["instability"])
        matched_ema = float(metrics["matched_ema"])
        selector_overlap_ema = float(metrics["selector_overlap_ema"])

        anchor_phase_open = int(frame_idx) >= int(self.geo_anchor_enable_after)
        landmark_phase_open = int(frame_idx) >= int(self.geo_landmark_enable_after)
        reference_phase_open = int(frame_idx) >= int(self.geo_reference_enable_after)
        reloc_phase_open = int(frame_idx) >= int(self.geo_reloc_enable_after)

        bootstrap_bank_ready = self._geo_bootstrap_bank_ready(int(frame_idx))
        structure_ready = self._geo_structure_ready()

        anchor_ready = bool(anchor_phase_open and self._geo_anchor_ready(int(frame_idx)))
        landmark_growth_ready = bool(
            landmark_phase_open
            and bootstrap_bank_ready
            and maturity >= 0.45
            and matched_ema >= 0.12
        )
        selector_ready_overlap = max(float(selector_overlap_ema), float(metrics["ref_overlap_ema"]))
        reference_growth_ready = bool(
            reference_phase_open
            and bootstrap_bank_ready
            and landmark_growth_ready
            and (
                len(self.geo_stable_anchor_voxels) >= 64
                or len(self.geo_landmark_voxels) >= 64
            )
            and selector_ready_overlap >= 8.0
        )

        landmark_label_ready = bool(landmark_phase_open and structure_ready and landmark_growth_ready)
        reference_label_ready = bool(reference_phase_open and structure_ready and reference_growth_ready)

        handover_ready_streak = int(self.geo_handover_ready_streak)
        handover_unready_streak = int(self.geo_handover_unready_streak)
        recovery_enter_streak = int(self.geo_recovery_enter_streak)
        recovery_exit_streak = int(self.geo_recovery_exit_streak)
        selector_mode = str(self.geo_selector_mode)

        if reference_label_ready and instability <= 0.45:
            handover_ready_streak += 1
            handover_unready_streak = 0
        else:
            handover_ready_streak = 0
            handover_unready_streak += 1

        if selector_mode == "legacy":
            if handover_ready_streak >= 5:
                selector_mode = "current"
        elif selector_mode == "current":
            if instability >= 0.70:
                recovery_enter_streak += 1
            else:
                recovery_enter_streak = 0
            if recovery_enter_streak >= 3:
                selector_mode = "recovery"
            elif handover_unready_streak >= 8:
                selector_mode = "legacy"
        else:
            if instability <= 0.35:
                recovery_exit_streak += 1
            else:
                recovery_exit_streak = 0
            if recovery_exit_streak >= 8:
                selector_mode = "current"

        prev_budget = max(
            1,
            int(
                self.geo_last_policy_inputs.get(
                    "frame_keep_budget_min",
                    self.geo_last_policy_inputs.get("final_ref_budget", 1),
                ) or 1
            ),
        )
        plain_patch_final_prev = int(
            self.geo_last_policy_inputs.get(
                "frame_keep_plain_patch_final_min",
                self.geo_last_policy_inputs.get("keep_plain_patch_final", 0),
            ) or 0
        )
        keep_plain_patch_reserved_prev = int(
            self.geo_last_policy_inputs.get(
                "frame_keep_plain_patch_reserved_min",
                self.geo_last_policy_inputs.get("keep_plain_patch_reserved", 0),
            ) or 0
        )
        plain_ratio_prev = float(plain_patch_final_prev) / float(prev_budget)
        reserved_ratio_prev = float(keep_plain_patch_reserved_prev) / float(prev_budget)
        stable_visible_ratio_prev = float(self.geo_last_selector_diag.get("stable_visible_ratio", 1.0)) if isinstance(self.geo_last_selector_diag, dict) else 1.0
        visible_total_prev = int(self.geo_last_selector_diag.get("visible_total", self.geo_last_policy_inputs.get("selector_diag_true_visible_total", 0))) if isinstance(self.geo_last_selector_diag, dict) else int(self.geo_last_policy_inputs.get("selector_diag_true_visible_total", 0) or 0)
        stable_overlap_prev = int(self.geo_last_selector_diag.get("stable_visible_overlap", self.geo_last_selector_diag.get("stable_visible_voxel_overlap", 0))) if isinstance(self.geo_last_selector_diag, dict) else 0

        plain_stress = min(1.0, max(0.0, (0.45 - plain_ratio_prev) / 0.20))
        reserved_target = 0.06
        reserved_stress = min(1.0, max(0.0, (reserved_target - reserved_ratio_prev) / reserved_target))
        visible_stress = min(1.0, max(0.0, (0.75 - stable_visible_ratio_prev) / 0.25))
        observation_stress = max(plain_stress, reserved_stress, visible_stress)

        observation_collapse = bool(
            plain_ratio_prev < 0.30
            or reserved_ratio_prev < 0.02
            or stable_visible_ratio_prev < 0.60
        )

        visible_count_collapse = bool(int(visible_total_prev) < 4096)
        visible_ratio_collapse = bool(float(stable_visible_ratio_prev) < 0.35)
        stable_overlap_collapse = bool(int(stable_overlap_prev) < 256)
        legacy_observation_break = bool(
            selector_mode == "legacy"
            and (not bool(bootstrap_bank_ready))
            and (
                visible_count_collapse
                or visible_ratio_collapse
                or stable_overlap_collapse
            )
        )

        if selector_mode == "current" and observation_collapse:
            selector_mode = "recovery"

        reference_support_ready = bool(
            len(self.geo_reference_bank) >= max(
                8,
                min(16, int(self._geo_effective_bootstrap_thresholds()["refs"])),
            )
            or len(self.geo_stable_anchor_voxels) >= 64
            or len(self.geo_landmark_voxels) >= 128
        )
        allow_reloc_trigger = bool(
            reloc_phase_open
            and reference_support_ready
        )
        reloc_signal = bool(
            instability >= 0.50
            or matched_ema < 0.08
            or selector_ready_overlap < 8.0
            or float(self.geo_trust_score) < float(self.geo_selection_low_trust_threshold)
        )
        ongoing_recovery = bool(int(self.geo_recovery_frames_left) > 0)
        ongoing_reloc = bool(
            int(self.geo_reloc_frames_left) > 0
            or str(self.geo_reloc_state) != "off"
        )

        base_use_cap = bool(reference_label_ready)
        base_cap_alpha = 0.0 if (not reference_label_ready) else min(1.0, max(0.0, (maturity - 0.65) / 0.35))

        policy = {
            "mode": selector_mode,
            "use_anchor_labels": bool(anchor_ready),
            "use_landmark_labels": bool(landmark_label_ready),
            "use_reference_labels": bool(reference_label_ready),
            "allow_landmark_growth": bool(landmark_growth_ready),
            "allow_reference_growth": bool(reference_growth_ready),
            "landmark_growth_ready": bool(landmark_growth_ready),
            "reference_growth_ready": bool(reference_growth_ready),
            "landmark_label_ready": bool(landmark_label_ready),
            "reference_label_ready": bool(reference_label_ready),
            "use_recovery": bool((selector_mode == "recovery") or ongoing_recovery),
            "ongoing_recovery": bool(ongoing_recovery),
            "allow_reloc_trigger": bool(allow_reloc_trigger),
            "use_reloc": bool(ongoing_reloc or (allow_reloc_trigger and reloc_signal)),
            "use_cap": bool(base_use_cap),
            "cap_alpha": float(base_cap_alpha),
            "hard_recent_frames": int(max(6, min(24, round(8 + 10 * instability)))),
            "recent_window": int(max(12, min(32, round(12 + 12 * instability)))),
            "soft_recent_window": int(max(16, min(40, round(16 + 12 * instability)))),
            "anchor_quota_ratio": float(max(0.03, min(0.08, 0.03 + 0.05 * (1.0 - maturity)))),
            "local_budget_ratio": float(max(0.45, min(0.85, 0.55 + 0.25 * instability - 0.15 * maturity))),
            "stable_read_budget_ratio": float(max(0.15, min(0.45, 0.15 + 0.30 * instability))),
            "frame0_patch_cap": int(max(256, min(2048, round((0.12 * 2048.0) * (1.0 - 0.5 * maturity))))),
            "use_view_pruning": bool(False if (not anchor_ready) else (True if selector_mode == "legacy" else instability < 0.85)),
            "prefer_last_reliable_view": bool(anchor_ready and instability >= 0.55),
            "anchor_phase_open": bool(anchor_phase_open),
            "landmark_phase_open": bool(landmark_phase_open),
            "reference_phase_open": bool(reference_phase_open),
            "reloc_phase_open": bool(reloc_phase_open),
            "bootstrap_bank_ready": bool(bootstrap_bank_ready),
            "structure_ready": bool(structure_ready),
            "observation_collapse": bool(observation_collapse),
            "observation_stress": float(observation_stress),
            "plain_patch_final_prev": int(plain_patch_final_prev),
            "keep_plain_patch_reserved_prev": int(keep_plain_patch_reserved_prev),
            "plain_ratio_prev": float(plain_ratio_prev),
            "reserved_ratio_prev": float(reserved_ratio_prev),
            "reserved_target_effective": float(reserved_target),
            "stable_visible_ratio_prev": float(stable_visible_ratio_prev),
            "visible_total_prev": int(visible_total_prev),
            "stable_overlap_prev": int(stable_overlap_prev),
            "legacy_observation_break": bool(legacy_observation_break),
            "prev_budget": int(prev_budget),
        }
        if legacy_observation_break:
            policy["legacy_break_force_recent_plain"] = True
            policy["legacy_break_anchor_scale"] = 0.25
            policy["legacy_break_frame0_scale"] = 0.50
            policy["legacy_break_recent_plain_ratio"] = 0.12
            policy["use_cap"] = False
            policy["cap_alpha"] = 0.0
            policy["local_budget_ratio"] = max(float(policy["local_budget_ratio"]), 0.80)
            policy["stable_read_budget_ratio"] = max(float(policy["stable_read_budget_ratio"]), 0.35)
            policy["hard_recent_frames"] = max(int(policy["hard_recent_frames"]), 8)
            policy["recent_window"] = max(int(policy["recent_window"]), 24)
            policy["soft_recent_window"] = max(int(policy["soft_recent_window"]), 28)
            policy["anchor_quota_ratio"] = min(float(policy["anchor_quota_ratio"]), 0.03)
        else:
            policy["legacy_break_force_recent_plain"] = False
            policy["legacy_break_anchor_scale"] = 1.0
            policy["legacy_break_frame0_scale"] = 1.0
            policy["legacy_break_recent_plain_ratio"] = 0.08

        if selector_mode in {"current", "recovery"}:
            policy["cap_alpha"] = float(policy["cap_alpha"]) * (1.0 - 0.85 * float(observation_stress))
            policy["local_budget_ratio"] = max(
                float(policy["local_budget_ratio"]),
                0.60 + 0.20 * float(observation_stress),
            )
            policy["stable_read_budget_ratio"] = max(
                float(policy["stable_read_budget_ratio"]),
                0.22 + 0.18 * float(observation_stress),
            )
            policy["anchor_quota_ratio"] = max(
                0.015,
                float(policy["anchor_quota_ratio"]) * (1.0 - 0.50 * float(observation_stress)),
            )
            policy["frame0_hard_scale"] = float(max(0.50, 1.0 - 0.50 * float(observation_stress)))
            policy["reference_hard_scale"] = float(max(0.40, 1.0 - 0.60 * float(observation_stress)))
        else:
            policy["frame0_hard_scale"] = 1.0
            policy["reference_hard_scale"] = 1.0

        if legacy_observation_break:
            policy["frame0_hard_scale"] = min(float(policy.get("frame0_hard_scale", 1.0)), 0.75)
            policy["reference_hard_scale"] = min(float(policy.get("reference_hard_scale", 1.0)), 0.75)

        if observation_stress >= 0.85:
            policy["use_cap"] = False
            policy["cap_alpha"] = 0.0
        elif observation_stress >= 0.55:
            policy["cap_alpha"] = min(float(policy["cap_alpha"]), 0.35)

        fsm = {
            "selector_mode_next": selector_mode,
            "handover_ready_streak_next": handover_ready_streak,
            "handover_unready_streak_next": handover_unready_streak,
            "recovery_enter_streak_next": recovery_enter_streak,
            "recovery_exit_streak_next": recovery_exit_streak,
        }
        return policy, fsm

    def _geo_commit_adaptive_policy_once(
        self,
        *,
        frame_idx: int,
        total_tokens: int,
        max_past_tokens: Optional[int],
        current_view: Optional[Dict[str, Any]],
        observation: Optional[Dict[str, Any]],
        selector_diag: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if int(self.geo_last_commit_guard_frame) == int(frame_idx):
            return copy.deepcopy(self.geo_last_committed_policy or self._geo_default_policy(frame_idx))

        signals = self._geo_collect_policy_signals(
            frame_idx=frame_idx,
            total_tokens=total_tokens,
            max_past_tokens=max_past_tokens,
            current_view=current_view,
            observation=observation,
            selector_diag=selector_diag,
        )
        metrics = self._geo_preview_adaptive_metrics(signals=signals)
        policy, fsm = self._geo_preview_adaptive_policy(frame_idx=frame_idx, metrics=metrics)
        if max_past_tokens is not None:
            policy["frame0_patch_cap"] = int(max(256, min(2048, round((0.12 * float(max_past_tokens or 2048)) * (1.0 - 0.5 * float(metrics["maturity"]))))))

        self.geo_pressure_ema = float(metrics["pressure_ema"])
        self.geo_motion_ema = float(metrics["motion_ema"])
        self.geo_confdrop_ema = float(metrics["confdrop_ema"])
        self.geo_matched_ema = float(metrics["matched_ema"])
        self.geo_new_voxel_ema = float(metrics["new_voxel_ema"])
        self.geo_ref_overlap_ema = float(metrics["ref_overlap_ema"])
        self.geo_selector_overlap_ema = float(metrics["selector_overlap_ema"])
        self.geo_selector_visible_ratio_ema = float(metrics["selector_visible_ratio_ema"])
        self.geo_maturity_ema = float(metrics["maturity"])
        self.geo_instability_ema = float(metrics["instability"])

        self.geo_selector_mode = str(fsm["selector_mode_next"])
        self.geo_handover_ready_streak = int(fsm["handover_ready_streak_next"])
        self.geo_handover_unready_streak = int(fsm["handover_unready_streak_next"])
        self.geo_recovery_enter_streak = int(fsm["recovery_enter_streak_next"])
        self.geo_recovery_exit_streak = int(fsm["recovery_exit_streak_next"])

        self.geo_last_committed_policy = copy.deepcopy(policy)
        self.geo_last_policy_frame = int(frame_idx)
        self.geo_last_policy_inputs = {
            "frame_idx": int(frame_idx),
            "total_tokens": int(total_tokens),
            "max_past_tokens": int(max_past_tokens) if max_past_tokens is not None else None,
            "observation_frame": int(observation.get("frame_idx", -1)) if observation else -1,
            "selector_diag_frame": int(selector_diag.get("frame_idx", -1)) if selector_diag else -1,
            "used_current_view": bool(current_view is not None),
            "observation_collapse": bool(policy.get("observation_collapse", False)),
            "observation_stress": float(policy.get("observation_stress", 0.0)),
            "plain_patch_final_prev": int(policy.get("plain_patch_final_prev", 0) or 0),
            "keep_plain_patch_reserved_prev": int(policy.get("keep_plain_patch_reserved_prev", 0) or 0),
            "plain_ratio_prev": float(policy.get("plain_ratio_prev", 0.0)),
            "reserved_ratio_prev": float(policy.get("reserved_ratio_prev", 0.0)),
            "reserved_target_effective": float(policy.get("reserved_target_effective", 0.06)),
            "stable_visible_ratio_prev": float(policy.get("stable_visible_ratio_prev", 1.0)),
            "prev_budget": int(policy.get("prev_budget", 0) or 0),
            "frame0_hard_scale": float(policy.get("frame0_hard_scale", 1.0)),
            "reference_hard_scale": float(policy.get("reference_hard_scale", 1.0)),
        }
        self.geo_last_policy_metrics = copy.deepcopy(metrics)
        self.geo_last_commit_guard_frame = int(frame_idx)
        return copy.deepcopy(policy)

    def _geo_peek_adaptive_policy(
        self,
        *,
        frame_idx: int,
        total_tokens: int,
        max_past_tokens: Optional[int],
        current_view: Optional[Dict[str, Any]],
        observation: Optional[Dict[str, Any]],
        selector_diag: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        signals = self._geo_collect_policy_signals(
            frame_idx=frame_idx,
            total_tokens=total_tokens,
            max_past_tokens=max_past_tokens,
            current_view=current_view,
            observation=observation,
            selector_diag=selector_diag,
        )
        metrics = self._geo_preview_adaptive_metrics(signals=signals)
        policy, _ = self._geo_preview_adaptive_policy(frame_idx=frame_idx, metrics=metrics)
        if max_past_tokens is not None:
            policy["frame0_patch_cap"] = int(max(256, min(2048, round((0.12 * float(max_past_tokens or 2048)) * (1.0 - 0.5 * float(metrics["maturity"]))))))
        return policy

    def _geo_preview_adaptive_policy_compat(
        self,
        current_frame_idx: int,
        total_tokens: int,
        max_past_tokens: Optional[int],
        current_view: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Deprecated compatibility wrapper: pure preview only (no state mutation).
        return self._geo_peek_adaptive_policy(
            frame_idx=int(current_frame_idx),
            total_tokens=int(total_tokens),
            max_past_tokens=max_past_tokens,
            current_view=current_view,
            observation=self._geo_get_last_observation(),
            selector_diag=self._geo_get_last_selector_diag(),
        )

    def _geo_preview_policy_with_budget_fixed_point(
        self,
        *,
        frame_idx: int,
        ref_meta: Optional[Dict[str, torch.Tensor]],
        raw_ref_layer_budget: int,
        current_view: Optional[Dict[str, Any]],
        P: int,
        allow_cap: bool,
        max_iters: int = 8,
    ) -> Tuple[Dict[str, Any], int, int, int, bool]:
        total_tokens = (
            int(ref_meta["frame_idx"].numel())
            if ref_meta is not None and ref_meta.get("frame_idx") is not None
            else 0
        )

        past_budget = max(0, int(raw_ref_layer_budget) - int(P))
        final_policy: Dict[str, Any] = self._geo_default_policy(int(frame_idx))
        final_layer_budget = int(raw_ref_layer_budget)

        for it in range(1, max(1, int(max_iters)) + 1):
            policy_now = self._geo_peek_adaptive_policy(
                frame_idx=int(frame_idx),
                total_tokens=int(total_tokens),
                max_past_tokens=int(past_budget),
                current_view=current_view,
                observation=self._geo_get_last_observation(),
                selector_diag=self._geo_get_last_selector_diag(),
            )
            if allow_cap:
                layer_budget_now = self._scheduled_layer_budget(
                    int(raw_ref_layer_budget),
                    int(frame_idx),
                    policy=policy_now,
                )
            else:
                layer_budget_now = int(raw_ref_layer_budget)
            new_past_budget = max(0, int(layer_budget_now) - int(P))

            final_policy = policy_now
            final_layer_budget = int(layer_budget_now)

            if int(new_past_budget) == int(past_budget):
                return final_policy, int(new_past_budget), int(final_layer_budget), int(it), True

            past_budget = int(new_past_budget)

        final_policy = self._geo_peek_adaptive_policy(
            frame_idx=int(frame_idx),
            total_tokens=int(total_tokens),
            max_past_tokens=int(past_budget),
            current_view=current_view,
            observation=self._geo_get_last_observation(),
            selector_diag=self._geo_get_last_selector_diag(),
        )
        final_layer_budget = int(past_budget) + int(P)
        return final_policy, int(past_budget), int(final_layer_budget), int(max_iters), False

    def _geo_peek_effective_policy_for_inference(
        self,
        *,
        past_key_values,
        past_frame_idx: int,
        total_budget: int,
        current_view: Optional[Dict[str, Any]],
        geo_recent_frames: int,
    ) -> Dict[str, Any]:
        _ = int(max(1, geo_recent_frames))
        guard_before = (
            float(self.geo_pressure_ema),
            float(self.geo_motion_ema),
            float(self.geo_confdrop_ema),
            float(self.geo_matched_ema),
            float(self.geo_new_voxel_ema),
            float(self.geo_ref_overlap_ema),
            float(self.geo_selector_overlap_ema),
            float(self.geo_selector_visible_ratio_ema),
            float(self.geo_maturity_ema),
            float(self.geo_instability_ema),
            int(self.geo_handover_ready_streak),
            int(self.geo_handover_unready_streak),
            int(self.geo_recovery_enter_streak),
            int(self.geo_recovery_exit_streak),
            str(self.geo_selector_mode),
        )

        current_budgets = self._calculate_dynamic_budgets(total_budget)
        ref_layer_idx = None
        if past_key_values is not None:
            for idx, kv in enumerate(past_key_values):
                if kv is not None and self.geo_token_meta[idx]["frame_idx"].numel() > 0:
                    ref_layer_idx = idx
                    break

        if ref_layer_idx is None:
            policy = copy.deepcopy(self.geo_last_committed_policy or self._geo_default_policy(int(past_frame_idx)))
        else:
            ref_meta = self.geo_token_meta[ref_layer_idx]
            cache_frame_idx = int(ref_meta["frame_idx"].max().item()) if ref_meta["frame_idx"].numel() > 0 else max(-1, int(past_frame_idx) - 1)
            structure_ready = self._geo_structure_ready()
            _, raw_ref_layer_budget, _, allow_cap = self._geo_get_shared_ref_budget(
                current_budgets=current_budgets,
                P=int(self.patch_start_idx),
                structure_ready=structure_ready,
                geo_policy=self.geo_last_committed_policy,
                frame_idx=int(past_frame_idx),
            )
            local_idx = ref_meta.get("local_patch_idx", torch.empty((0,), dtype=torch.long))
            valid_local = local_idx[local_idx >= 0]
            patch_count_hint = int(valid_local.max().item()) + 1 if valid_local.numel() > 0 else int(max(0, raw_ref_layer_budget - self.patch_start_idx))
            P = int(self.patch_start_idx + max(0, patch_count_hint))
            policy, _, _, _, _ = self._geo_preview_policy_with_budget_fixed_point(
                frame_idx=int(cache_frame_idx),
                ref_meta=ref_meta,
                raw_ref_layer_budget=int(raw_ref_layer_budget),
                current_view=current_view,
                P=int(P),
                allow_cap=bool(allow_cap),
            )

        guard_after = (
            float(self.geo_pressure_ema),
            float(self.geo_motion_ema),
            float(self.geo_confdrop_ema),
            float(self.geo_matched_ema),
            float(self.geo_new_voxel_ema),
            float(self.geo_ref_overlap_ema),
            float(self.geo_selector_overlap_ema),
            float(self.geo_selector_visible_ratio_ema),
            float(self.geo_maturity_ema),
            float(self.geo_instability_ema),
            int(self.geo_handover_ready_streak),
            int(self.geo_handover_unready_streak),
            int(self.geo_recovery_enter_streak),
            int(self.geo_recovery_exit_streak),
            str(self.geo_selector_mode),
        )
        assert guard_before == guard_after, "peek_effective_policy_for_inference must be pure"
        return policy

    def _geo_get_effective_policy_for_forward(
        self,
        *,
        frame_idx: int,
        ref_meta: Optional[Dict[str, torch.Tensor]],
        ref_past_budget: Optional[int],
        current_view: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self._geo_has_complete_policy_context(ref_meta, ref_past_budget):
            return self._geo_commit_adaptive_policy_once(
                frame_idx=int(frame_idx),
                total_tokens=int(ref_meta["frame_idx"].numel()),
                max_past_tokens=int(ref_past_budget) if ref_past_budget is not None else None,
                current_view=current_view,
                observation=self._geo_get_last_observation(),
                selector_diag=self._geo_get_last_selector_diag(),
            )
        if self.geo_last_committed_policy is not None:
            return copy.deepcopy(self.geo_last_committed_policy)
        return self._geo_default_policy(int(frame_idx))

    def _geo_get_shared_ref_budget(
        self,
        *,
        current_budgets: torch.Tensor,
        P: int,
        structure_ready: bool,
        geo_policy: Optional[Dict[str, Any]],
        frame_idx: int,
    ) -> Tuple[int, int, str, bool]:
        """
        Returns:
            base_ref_past_budget,
            base_ref_layer_budget,
            ref_budget_source,
            allow_cap
        """
        _ = int(frame_idx)
        raw_max = int(current_budgets.max().item())
        raw_min = int(current_budgets.min().item())

        apply_early_floor = False
        upper_budget = int(raw_max)
        if not structure_ready:
            apply_early_floor = bool(
                (int(frame_idx) <= int(self.geo_early_stabilize_frames))
                or (not bool((geo_policy or {}).get("use_anchor_labels", False)))
            )
            if apply_early_floor:
                upper_budget = max(int(raw_max), int(self.geo_early_budget_floor))
                base_ref_layer_budget = int(upper_budget)
                ref_budget_source = "floor" if upper_budget > raw_max else "max"
            else:
                base_ref_layer_budget = int(raw_max)
                ref_budget_source = "max"
            allow_cap = False
        else:
            upper_budget = int(raw_max)
            base_ref_layer_budget = int(raw_min)
            ref_budget_source = "min"
            allow_cap = bool(geo_policy is not None and bool(geo_policy.get("use_cap", False)))
        self.geo_last_policy_inputs["shared_ref_early_floor_applied"] = bool(apply_early_floor)

        prev_ref_layer_budget = int(self.geo_last_policy_inputs.get("final_ref_layer_budget", raw_max) or raw_max)
        target_ref_budget = int(base_ref_layer_budget)
        max_down = max(512, int(0.12 * max(1, prev_ref_layer_budget)))
        max_up = max(1024, int(0.20 * max(1, prev_ref_layer_budget)))
        if target_ref_budget < prev_ref_layer_budget:
            target_ref_budget = max(int(target_ref_budget), int(prev_ref_layer_budget - max_down))
        else:
            target_ref_budget = min(int(target_ref_budget), int(prev_ref_layer_budget + max_up))
        base_ref_layer_budget = int(max(raw_min, min(int(upper_budget), target_ref_budget)))
        self.geo_last_policy_inputs["shared_ref_budget_upper"] = int(upper_budget)
        self.geo_last_policy_inputs["shared_ref_prev_layer_budget"] = int(prev_ref_layer_budget)

        base_ref_past_budget = max(0, int(base_ref_layer_budget) - int(P))
        return int(base_ref_past_budget), int(base_ref_layer_budget), str(ref_budget_source), bool(allow_cap)

    def _scheduled_layer_budget(self, raw_layer_budget: int, frame_idx: int, policy: Optional[Dict[str, Any]] = None) -> int:
        raw_b = max(0, int(raw_layer_budget))
        cap_b = min(raw_b, int(self.geo_layer_budget_cap))

        if policy is None or (not bool(policy.get("use_cap", False))):
            return raw_b

        alpha = max(0.0, min(1.0, float(policy.get("cap_alpha", 0.0))))
        blended = int(round((1.0 - alpha) * float(raw_b) + alpha * float(cap_b)))
        return max(cap_b, blended)

    def _label_eligible_mask(self, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n == 0:
            return torch.empty((0,), dtype=torch.bool)
        frame_idx = meta["frame_idx"]
        is_special = meta.get("is_special", torch.zeros((n,), dtype=torch.bool))
        is_keyframe = meta.get("is_keyframe", torch.zeros((n,), dtype=torch.bool))
        max_frame = int(frame_idx.max().item()) if n > 0 else -1
        recent_mask = frame_idx >= max(-1, max_frame - 1)
        return (~is_special) & (is_keyframe | recent_mask)

    def _hard_recent_patch_idx(self, meta: Dict[str, torch.Tensor], hard_recent_frames: int) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n == 0 or hard_recent_frames <= 0:
            return torch.empty((0,), dtype=torch.long)

        frame_idx = meta["frame_idx"]
        is_special = meta.get("is_special", torch.zeros((n,), dtype=torch.bool))
        local_idx = meta.get("local_patch_idx", torch.full((n,), -1, dtype=torch.long))

        current_frame_idx = int(frame_idx.max().item())
        recent_min = max(0, current_frame_idx - int(hard_recent_frames) + 1)
        mask = (frame_idx >= recent_min) & (~is_special) & (local_idx >= 0)
        return torch.nonzero(mask, as_tuple=False).flatten()

    def _take_recent_quota(self, idx: torch.Tensor, frame_idx: torch.Tensor, quota: int) -> torch.Tensor:
        if idx.numel() == 0 or quota <= 0:
            return torch.empty((0,), dtype=torch.long)
        if idx.numel() <= int(quota):
            return idx
        order = torch.argsort(frame_idx.index_select(0, idx), descending=True)
        return idx.index_select(0, order[: int(quota)])

    def _bounded_label_from_mask(
        self,
        meta: Dict[str, torch.Tensor],
        raw_mask: torch.Tensor,
        per_frame_quota: int,
    ) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n == 0 or raw_mask.numel() == 0 or per_frame_quota <= 0:
            return torch.zeros((n,), dtype=torch.bool)

        out = torch.zeros((n,), dtype=torch.bool)
        frame_idx = meta["frame_idx"]
        local_idx = meta.get("local_patch_idx", torch.full((n,), -1, dtype=torch.long))

        frames = torch.unique(frame_idx[raw_mask]) if raw_mask.any() else torch.empty((0,), dtype=frame_idx.dtype)
        for f in frames.tolist():
            f = int(f)
            idx_f = torch.nonzero(raw_mask & (frame_idx == f), as_tuple=False).flatten()
            if idx_f.numel() == 0:
                continue
            if idx_f.numel() <= int(per_frame_quota):
                out[idx_f] = True
                continue

            fm = self.geo_frame_meta.get(f)
            if fm is not None and fm.get("conf") is not None:
                lp = local_idx.index_select(0, idx_f).long()
                valid = (lp >= 0) & (lp < fm["conf"].shape[0])
                idx_f = idx_f[valid]
                lp = lp[valid]
                if idx_f.numel() == 0:
                    continue
                score = fm["conf"].index_select(0, lp).to(torch.float32)
            else:
                score = torch.arange(idx_f.numel(), dtype=torch.float32)

            k = min(int(per_frame_quota), int(idx_f.numel()))
            top = torch.topk(score, k=k, largest=True).indices
            out[idx_f.index_select(0, top)] = True
        return out

    def _decay_persistent_labels(self, meta: Dict[str, torch.Tensor], current_frame_idx: int):
        n = int(meta.get("frame_idx", torch.empty((0,), dtype=torch.long)).numel())
        if n <= 0:
            return
        frame_idx = meta["frame_idx"]
        age = int(current_frame_idx) - frame_idx
        anchor_ttl = 12 if not self._geo_structure_ready() else 6
        self.geo_last_policy_inputs["anchor_ttl_effective"] = int(anchor_ttl)
        landmark_ttl = 256
        reference_ttl = 256
        if "is_anchor" in meta and meta["is_anchor"].numel() == n:
            meta["is_anchor"] = meta["is_anchor"] & (age <= anchor_ttl)
        if "is_landmark" in meta and meta["is_landmark"].numel() == n:
            meta["is_landmark"] = meta["is_landmark"] & (age <= landmark_ttl)
        if "is_reference" in meta and meta["is_reference"].numel() == n:
            meta["is_reference"] = meta["is_reference"] & (age <= reference_ttl)

    def _fill_keep_to_budget(
        self,
        meta: Dict[str, torch.Tensor],
        keep: torch.Tensor,
        budget: int,
        mode: str = "balanced",
    ) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n <= 0 or keep.numel() >= int(budget):
            return keep

        idx_all = torch.arange(n, dtype=torch.long)
        frame_idx = meta["frame_idx"]
        leftover = idx_all[~torch.isin(idx_all, keep)]
        if leftover.numel() == 0:
            return keep

        need = int(budget) - int(keep.numel())
        if need <= 0:
            return keep

        if str(mode) == "legacy":
            left_frames = frame_idx.index_select(0, leftover)
            uniq_frames = torch.unique(left_frames, sorted=True)
            per_frame = max(16, min(64, int(need // max(1, uniq_frames.numel()))))
            selected_fill: List[torch.Tensor] = []
            for f in uniq_frames.tolist():
                idx_f = leftover[left_frames == int(f)]
                if idx_f.numel() > 0:
                    selected_fill.append(idx_f[: min(int(per_frame), int(idx_f.numel()))])
            if selected_fill:
                fill = torch.unique(torch.cat(selected_fill, dim=0), sorted=True)
                return torch.unique(torch.cat([keep, fill[:need]], dim=0), sorted=True)
            return keep

        if str(mode) == "balanced":
            left_frames = frame_idx.index_select(0, leftover)
            uniq_frames = torch.unique(left_frames, sorted=True)
            selected_fill: List[torch.Tensor] = []
            per_frame = max(16, min(64, int(need // max(1, uniq_frames.numel()))))
            step_denom = max(1, int(need // max(1, per_frame)))
            stride = max(1, int(math.ceil(float(max(1, uniq_frames.numel())) / float(step_denom))))
            for f in uniq_frames[::stride].tolist():
                idx_f = leftover[left_frames == int(f)]
                if idx_f.numel() > 0:
                    selected_fill.append(idx_f[: min(int(per_frame), int(idx_f.numel()))])
            if selected_fill:
                fill = torch.unique(torch.cat(selected_fill, dim=0), sorted=True)
                if fill.numel() < need:
                    rem = leftover[~torch.isin(leftover, fill)]
                    if rem.numel() > 0:
                        fill = torch.unique(torch.cat([fill, rem[: max(0, need - fill.numel())]], dim=0), sorted=True)
                return torch.unique(torch.cat([keep, fill[:need]], dim=0), sorted=True)
            return keep

        left_frames = frame_idx.index_select(0, leftover)
        uniq_frames = torch.unique(left_frames, sorted=True)
        selected_fill: List[torch.Tensor] = []
        per_frame = max(16, min(64, int(need // max(1, uniq_frames.numel()))))
        for f in uniq_frames.tolist():
            idx_f = leftover[left_frames == int(f)]
            if idx_f.numel() > 0:
                selected_fill.append(idx_f[: min(int(per_frame), int(idx_f.numel()))])
        if selected_fill:
            fill = torch.unique(torch.cat(selected_fill, dim=0), sorted=True)
            return torch.unique(torch.cat([keep, fill[:need]], dim=0), sorted=True)
        return keep

    def _fill_keep_to_budget_preserve_order(
        self,
        meta: Dict[str, torch.Tensor],
        keep: torch.Tensor,
        budget: int,
        mode: str = "balanced",
    ) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n <= 0 or keep.numel() >= int(budget):
            return self._sanitize_keep_idx_preserve_order(keep, meta_len=n, kv_len=n)

        keep_ord = self._sanitize_keep_idx_preserve_order(keep, meta_len=n, kv_len=n)
        idx_all = torch.arange(n, dtype=torch.long)
        frame_idx = meta["frame_idx"]

        leftover = idx_all[~torch.isin(idx_all, keep_ord)]
        if leftover.numel() == 0:
            return keep_ord

        need = int(budget) - int(keep_ord.numel())
        if need <= 0:
            return keep_ord

        left_frames = frame_idx.index_select(0, leftover)
        uniq_frames = torch.unique(left_frames, sorted=True)
        frame_to_tokens: Dict[int, torch.Tensor] = {}
        for f in uniq_frames.tolist():
            frame_to_tokens[int(f)] = leftover[left_frames == int(f)]

        picked_extra: List[int] = []

        if str(mode) == "balanced":
            rev_frames = torch.flip(uniq_frames, dims=[0])
            per_frame = max(16, min(64, int(need // max(1, rev_frames.numel()))))
            step_denom = max(1, int(need // max(1, per_frame)))
            stride = max(1, int(math.ceil(float(max(1, rev_frames.numel())) / float(step_denom))))
            chosen_frames = rev_frames[::stride]
            for f in chosen_frames.tolist():
                idx_f = frame_to_tokens[int(f)]
                if idx_f.numel() > 0:
                    picked_extra.extend(int(v) for v in idx_f[: min(int(per_frame), int(idx_f.numel()))].tolist())

        elif str(mode) == "legacy":
            rev_frames = torch.flip(uniq_frames, dims=[0])
            per_frame = max(16, min(64, int(need // max(1, rev_frames.numel()))))
            for f in rev_frames.tolist():
                idx_f = frame_to_tokens[int(f)]
                if idx_f.numel() > 0:
                    picked_extra.extend(int(v) for v in idx_f[: min(int(per_frame), int(idx_f.numel()))].tolist())

        else:
            rev_frames = torch.flip(uniq_frames, dims=[0])
            for f in rev_frames.tolist():
                idx_f = frame_to_tokens[int(f)]
                if idx_f.numel() > 0:
                    picked_extra.extend(int(v) for v in idx_f.tolist())
                    if len(picked_extra) >= need:
                        break

        if len(picked_extra) < need:
            picked_extra_set = set(picked_extra)
            rev_frames = torch.flip(uniq_frames, dims=[0])
            for f in rev_frames.tolist():
                idx_f = frame_to_tokens[int(f)]
                for v in idx_f.tolist():
                    iv = int(v)
                    if iv not in picked_extra_set:
                        picked_extra.append(iv)
                        picked_extra_set.add(iv)
                    if len(picked_extra) >= need:
                        break
                if len(picked_extra) >= need:
                    break

        fill = torch.tensor(picked_extra[:need], dtype=torch.long) if picked_extra else torch.empty((0,), dtype=torch.long)
        out = self._unique_preserve_order_long(torch.cat([keep_ord, fill], dim=0))
        return out

    def _bounded_special_idx(
        self,
        meta: Dict[str, torch.Tensor],
        idx_all: torch.Tensor,
        budget: int,
        recent_frames: int,
    ) -> torch.Tensor:
        if idx_all.numel() == 0 or budget <= 0:
            return torch.empty((0,), dtype=torch.long)
        frame_idx = meta.get("frame_idx", torch.zeros((idx_all.numel(),), dtype=torch.long))
        is_special = meta.get("is_special", torch.zeros((idx_all.numel(),), dtype=torch.bool))
        identity_local = meta.get("identity_local", meta.get("local_patch_idx", torch.full((idx_all.numel(),), -1, dtype=torch.long)))
        max_frame = int(frame_idx.max().item()) if frame_idx.numel() > 0 else -1
        special_quota = min(max(16, int(budget // 8)), 128)
        special = idx_all[is_special]
        if special.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        frame0_camera = special[
            (frame_idx.index_select(0, special) == 0)
            & (identity_local.index_select(0, special) == -1)
        ]
        recent_special = special[
            frame_idx.index_select(0, special) >= max(0, max_frame - max(1, int(recent_frames)) + 1)
        ]
        bounded = torch.unique(torch.cat([frame0_camera, recent_special], dim=0), sorted=True)
        if bounded.numel() > special_quota:
            bounded = bounded[-special_quota:]
        return bounded

    def _warmup_keep(self, meta: Dict[str, torch.Tensor], budget: int, recent_frames: int = 8) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n <= 0 or budget <= 0:
            return torch.empty((0,), dtype=torch.long)
        if n <= int(budget):
            return torch.arange(n, dtype=torch.long)
        idx_all = torch.arange(n, dtype=torch.long)
        frame_idx = meta["frame_idx"]
        is_special = meta.get("is_special", torch.zeros((n,), dtype=torch.bool))
        max_frame = int(frame_idx.max().item()) if frame_idx.numel() > 0 else -1
        recent_mask = frame_idx >= max(-1, max_frame - int(recent_frames) + 1)

        bounded_special = self._bounded_special_idx(meta, idx_all, budget=budget, recent_frames=recent_frames)
        recent_keep = idx_all[recent_mask & (~is_special)]
        keep = torch.unique(torch.cat([bounded_special, recent_keep], dim=0), sorted=True)

        if keep.numel() < int(budget):
            leftover = idx_all[~torch.isin(idx_all, keep)]
            if leftover.numel() > 0:
                left_frames = frame_idx.index_select(0, leftover)
                uniq_frames = torch.unique(left_frames, sorted=True)
                stride = max(1, int(math.ceil(float(max(1, uniq_frames.numel())) / float(max(1, int(budget - keep.numel()))))))
                chosen_frames = uniq_frames[::stride]
                sampled: List[torch.Tensor] = []
                for f in chosen_frames.tolist():
                    f_idx = leftover[left_frames == int(f)]
                    if f_idx.numel() > 0:
                        sampled.append(f_idx[: min(64, int(f_idx.numel()))])
                if sampled:
                    sampled_idx = torch.unique(torch.cat(sampled, dim=0), sorted=True)
                    fill_budget = int(budget) - int(keep.numel())
                    keep = torch.unique(torch.cat([keep, sampled_idx[:fill_budget]], dim=0), sorted=True)

        if keep.numel() > int(budget):
            keep = keep[-int(budget):]
        return keep

    def _build_safe_warmup_keep(
        self,
        meta: Dict[str, torch.Tensor],
        *,
        budget: int,
        recent_frames: int,
        current_frame_idx: int,
        policy: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        # Hard-backbone-first warmup keep. Used only before bootstrap_bank_ready.
        b = max(0, int(budget))
        if b <= 0:
            return torch.empty((0,), dtype=torch.long)

        hard_keep = self._build_hard_backbone_keep(
            meta,
            current_frame_idx=int(current_frame_idx),
            max_past_tokens=b,
            policy=policy,
        )

        selected: set[int] = set()
        selected_order: List[int] = []
        self._ordered_add(selected, selected_order, hard_keep)

        n = int(meta.get("frame_idx", torch.empty((0,), dtype=torch.long)).numel())
        if n <= 0:
            return torch.empty((0,), dtype=torch.long)

        frame_idx = meta["frame_idx"]
        is_special = meta.get("is_special", torch.zeros((n,), dtype=torch.bool))
        local_idx = meta.get("local_patch_idx", torch.full((n,), -1, dtype=torch.long))

        recent_min = max(0, int(current_frame_idx) - int(recent_frames) + 1)
        recent_patch = torch.nonzero(
            (frame_idx >= recent_min) & (~is_special) & (local_idx >= 0),
            as_tuple=False,
        ).flatten()
        if recent_patch.numel() > 0:
            remain_budget = max(0, b - len(selected))
            recent_quota = min(int(recent_patch.numel()), max(128, remain_budget))
            recent_keep = self._take_recent_quota(recent_patch, frame_idx=frame_idx, quota=int(recent_quota))
            self._ordered_add(selected, selected_order, recent_keep)

        keep = torch.tensor(selected_order, dtype=torch.long) if selected_order else torch.empty((0,), dtype=torch.long)
        keep = self._cap_keep_with_hard_protection(
            meta=meta,
            keep_idx=keep,
            hard_keep=hard_keep,
            budget=b,
            recent_frames=max(1, int(recent_frames)),
            priority_keep_idx=keep,
            policy=policy,
        )

        return keep

    def _geo_anchor_ready(self, frame_idx: int) -> bool:
        min_start = max(int(self.geo_bootstrap_until), int(self.geo_anchor_enable_after))
        if int(frame_idx) < min_start:
            return False
        return bool(
            len(self.geo_stable_anchor_voxels) >= 16
            or len(self.geo_voxel_bank) >= 64
        )

    def _update_geo_runtime_ready(self, frame_idx: int, force_refresh: bool = False) -> bool:
        if (not force_refresh) and int(frame_idx) == int(self.geo_runtime_ready_last_frame):
            return bool(self.geo_runtime_ready_latched)
        self.geo_runtime_ready_last_frame = int(frame_idx)
        raw_ready = self._geo_map_ready_for_prune(frame_idx)
        if raw_ready:
            self.geo_runtime_ready_streak += 1
            self.geo_runtime_unready_streak = 0
        else:
            self.geo_runtime_ready_streak = 0
            self.geo_runtime_unready_streak += 1

        if (not self.geo_runtime_ready_latched) and self.geo_runtime_ready_streak >= 2:
            self.geo_runtime_ready_latched = True
        if self.geo_runtime_ready_latched and self.geo_runtime_unready_streak >= 8:
            self.geo_runtime_ready_latched = False
        return bool(self.geo_runtime_ready_latched)

    def _geo_ref_ready(self) -> bool:
        return len(self.geo_reference_bank) >= int(self.geo_bootstrap_min_refs)

    def _update_geo_runtime_state(
        self,
        frame_idx: int,
        matched_ratio: float,
        ref_overlap: int,
        runtime_map_ready: Optional[bool] = None,
        policy: Optional[Dict[str, Any]] = None,
    ):
        allow_reloc_trigger = (
            bool(policy.get("allow_reloc_trigger", policy.get("use_reloc", True)))
            if policy is not None else True
        )
        ongoing_reloc = bool(int(self.geo_reloc_frames_left) > 0 or str(self.geo_reloc_state) != "off")

        if (not allow_reloc_trigger) and (not ongoing_reloc):
            self.geo_reloc_state = "off"
            self.geo_reloc_frames_left = 0
            self.geo_reloc_hard_left = 0
            self.geo_reloc_good_streak = 0
            return

        runtime_map_ready = bool(self._update_geo_runtime_ready(frame_idx) if runtime_map_ready is None else runtime_map_ready)
        structure_ready = self._geo_structure_ready()

        if str(self.geo_reloc_state) != "off":
            self.geo_reloc_frames_left = max(0, int(self.geo_reloc_frames_left) - 1)
            if str(self.geo_reloc_state) == "hard":
                self.geo_reloc_hard_left = max(0, int(self.geo_reloc_hard_left) - 1)
                if int(self.geo_reloc_hard_left) <= 0:
                    self.geo_reloc_state = "recover"
            if int(self.geo_reloc_frames_left) <= 0:
                self.geo_reloc_state = "off"
                self.geo_reloc_hard_left = 0
                self.geo_reloc_good_streak = 0

        ongoing_after_decay = bool(
            int(self.geo_reloc_frames_left) > 0
            or str(self.geo_reloc_state) != "off"
        )
        if (not allow_reloc_trigger) and (not ongoing_after_decay):
            self.geo_reloc_state = "off"
            self.geo_reloc_frames_left = 0
            self.geo_reloc_hard_left = 0
            self.geo_reloc_good_streak = 0
            return

        if runtime_map_ready:
            bad_runtime = (
                float(self.geo_trust_score) < float(self.geo_selection_low_trust_threshold)
                or float(matched_ratio) < 0.05
                or (
                    len(self.geo_reference_bank) >= 16
                    and int(ref_overlap) < max(4, int(self.geo_reference_min_overlap // 2))
                )
            )
        else:
            bad_runtime = (
                structure_ready
                and int(frame_idx) >= int(self.geo_bootstrap_frames) + 32
                and (
                    len(self.geo_reference_bank) < 8
                    or len(self.geo_landmark_voxels) < 64
                )
            )

        if bool(allow_reloc_trigger) and bad_runtime and str(self.geo_reloc_state) == "off":
            self.geo_reloc_state = "hard"
            self.geo_reloc_hard_left = int(self.geo_reloc_hard_frames)
            self.geo_reloc_frames_left = int(self.geo_reloc_frames)
            self.geo_reloc_good_streak = 0

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

    def _allow_repair_existing_voxel(
        self,
        *,
        key: Tuple[int, int, int],
        conf_mean: float,
        drift2: float,
        high_conf_thr: float,
        near_stable_or_reference_fn,
    ) -> bool:
        trusted_key = (
            key in self.geo_reference_bank
            or key in self.geo_stable_anchor_voxels
            or key in self.geo_reference_voxels
        )
        near_trusted = bool(near_stable_or_reference_fn(key))
        conf_ok = float(conf_mean) >= float(high_conf_thr)

        strict_drift_thr = float(self.geo_bank_inlier_drift2_thr)
        loose_drift_thr = 4.0 * strict_drift_thr

        strict_drift_ok = float(drift2) <= strict_drift_thr
        loose_drift_ok = float(drift2) <= loose_drift_thr

        return bool(
            (trusted_key and strict_drift_ok)
            or (trusted_key and conf_ok and loose_drift_ok)
            or (near_trusted and conf_ok and loose_drift_ok)
        )

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

        policy = copy.deepcopy(self.geo_last_committed_policy) if self.geo_last_committed_policy is not None else self._geo_default_policy(int(frame_idx))
        safe_warmup = not bool(policy["use_anchor_labels"])

        bootstrap_bank_ready = self._geo_bootstrap_bank_ready(frame_idx)
        structure_ready = self._geo_structure_ready()
        runtime_map_ready = False
        promote_reference_ready = (
            len(self.geo_stable_anchor_voxels) >= 64
            or len(self.geo_reference_bank) >= 16
            or len(self.geo_landmark_voxels) >= 128
        )
        recovery_frames_next = int(self.geo_recovery_frames_left)
        low_trust = False
        bad_overlap = False
        tiny_bootstrap = (int(frame_idx) < int(self.geo_bootstrap_frames)) and (len(self.geo_voxel_bank) < 256)
        if tiny_bootstrap:
            self.geo_trust_score = 1.0
            recovery_frames_next = 0
            low_trust = False
        else:
            if residuals:
                med = float(torch.tensor(residuals, dtype=torch.float32).median().item())
                residual_trust = float(max(0.0, min(1.0, 1.0 - (med / max(self.geo_bank_trust_residual_thr, 1e-8)))))
            else:
                residual_trust = 0.0
            overlap_trust = float(max(0.0, min(1.0, matched_ratio / 0.25)))
            self.geo_trust_score = float(min(residual_trust, overlap_trust))

            bad_overlap = matched_ratio < 0.05
            if structure_ready and len(self.geo_reference_bank) >= 16:
                bad_overlap = bad_overlap or (ref_overlap < max(4, int(self.geo_reference_min_overlap // 2)))
            if structure_ready and ref_overlap >= int(self.geo_reference_min_overlap) and ref_residuals:
                ref_med = float(torch.tensor(ref_residuals, dtype=torch.float32).median().item())
                if ref_med > float(self.geo_reference_drift_threshold):
                    recovery_frames_next = max(int(recovery_frames_next), int(self.geo_recovery_frames))
            low_trust = self.geo_trust_score < float(self.geo_selection_low_trust_threshold)
            if bad_overlap or self.geo_trust_score < float(self.geo_selection_low_trust_threshold):
                recovery_frames_next = max(int(recovery_frames_next), int(self.geo_recovery_frames))
            elif recovery_frames_next > 0:
                recovery_frames_next = max(0, int(recovery_frames_next) - 1)

        if tiny_bootstrap:
            self.geo_recovery_frames_left = 0
        else:
            self.geo_recovery_frames_left = max(0, int(recovery_frames_next))
        recovery_mode = bool(int(self.geo_recovery_frames_left) > 0)

        allow_new_voxels = (not recovery_mode) and (not low_trust)
        allow_promote_landmark = (
            bool(policy.get("allow_landmark_growth", False))
            and bool(bootstrap_bank_ready)
            and (not recovery_mode)
        )
        allow_promote_reference = (
            bool(policy.get("allow_reference_growth", False))
            and bool(bootstrap_bank_ready)
            and (not recovery_mode)
            and (float(self.geo_trust_score) >= float(self.geo_selection_low_trust_threshold))
            and bool(promote_reference_ready)
        )

        allow_reloc_trigger = bool(policy.get("allow_reloc_trigger", policy.get("use_reloc", False)))
        ongoing_reloc = bool(int(self.geo_reloc_frames_left) > 0 or str(self.geo_reloc_state) != "off")
        if (not allow_reloc_trigger) and (not ongoing_reloc):
            self.geo_reloc_state = "off"
            self.geo_reloc_frames_left = 0
            self.geo_reloc_hard_left = 0
            self.geo_reloc_good_streak = 0

        repair_mode = bool(recovery_mode) or bool(int(self.geo_reloc_frames_left) > 0 or str(self.geo_reloc_state) != "off")
        allow_reference_refresh_only = bool(
            repair_mode
            and bool(self.geo_recovery_reference_refresh)
            and bool(bootstrap_bank_ready)
            and bool(promote_reference_ready)
        )
        if repair_mode:
            allow_new_voxels = True
            allow_promote_landmark = bool(policy.get("allow_landmark_growth", False)) and bool(bootstrap_bank_ready)
            allow_promote_reference = (
                bool(policy.get("allow_reference_growth", False))
                and bool(bootstrap_bank_ready)
                and bool(promote_reference_ready)
            )

        recovery_new_voxel_quota = int(self.geo_recovery_new_voxel_quota_per_frame) if repair_mode else int(1e9)
        recovery_landmark_quota = int(self.geo_recovery_landmark_quota_per_frame) if repair_mode else None
        recovery_reference_quota = int(self.geo_recovery_reference_quota_per_frame) if repair_mode else None
        high_conf_thr = max(float(self.geo_stable_anchor_min_conf), 1.1)
        stable_or_ref_set = set(self.geo_stable_anchor_voxels)
        stable_or_ref_set.update(self.geo_reference_voxels)

        def _near_stable_or_reference(voxel_key: Tuple[int, int, int]) -> bool:
            if not stable_or_ref_set:
                return False
            x, y, z = voxel_key
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if (x + dx, y + dy, z + dz) in stable_or_ref_set:
                            return True
            return False

        new_voxels = 0
        for g in range(num_groups):
            key = tuple(int(v) for v in uniq_vox[g].tolist())
            conf_mean = float(conf_mean_all[g].item())
            pos_mean = pos_mean_all[g]

            if key not in self.geo_voxel_bank:
                if not allow_new_voxels:
                    continue
                if repair_mode:
                    if int(new_voxels) >= int(recovery_new_voxel_quota):
                        continue
                    if float(conf_mean) < float(high_conf_thr):
                        continue
                    if not _near_stable_or_reference(key):
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
                if repair_mode:
                    if not self._allow_repair_existing_voxel(
                        key=key,
                        conf_mean=float(conf_mean),
                        drift2=float(drift2),
                        high_conf_thr=float(high_conf_thr),
                        near_stable_or_reference_fn=_near_stable_or_reference,
                    ):
                        item["outlier_count"] = int(item.get("outlier_count", 0)) + 1
                        item["last_seen"] = float(frame_idx)
                        continue

                    repair_conf_alpha = max(float(self.geo_conf_ema_alpha), 0.95)
                    repair_var_alpha = max(float(self.geo_var_ema_alpha), 0.95)

                    item["outlier_count"] = 0
                    item["conf_ema"] = (
                        repair_conf_alpha * item["conf_ema"]
                        + (1.0 - repair_conf_alpha) * conf_mean
                    )
                    item["support"] += 1.0
                    item["pos_x"] = float(
                        repair_conf_alpha * item["pos_x"]
                        + (1.0 - repair_conf_alpha) * pos_mean[0].item()
                    )
                    item["pos_y"] = float(
                        repair_conf_alpha * item["pos_y"]
                        + (1.0 - repair_conf_alpha) * pos_mean[1].item()
                    )
                    item["pos_z"] = float(
                        repair_conf_alpha * item["pos_z"]
                        + (1.0 - repair_conf_alpha) * pos_mean[2].item()
                    )
                    item["pos_var"] = (
                        repair_var_alpha * item.get("pos_var", 0.0)
                        + (1.0 - repair_var_alpha) * drift2
                    )
                    item["last_seen"] = float(frame_idx)
                    if self.geo_recovery_reference_refresh and key in self.geo_reference_bank:
                        self.geo_reference_bank[key].update(
                            {
                                "pos_x": float(item["pos_x"]),
                                "pos_y": float(item["pos_y"]),
                                "pos_z": float(item["pos_z"]),
                                "score": float(
                                    max(
                                        float(self.geo_reference_bank[key].get("score", 0.0)),
                                        conf_mean * float(item.get("conf_ema", 0.0)),
                                    )
                                ),
                            }
                        )
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
            self._update_landmarks_from_keyframe(
                frame_idx,
                uniq_vox,
                conf_mean_all,
                quota_override=recovery_landmark_quota,
            )
        if allow_promote_reference:
            self._update_reference_bank_from_keyframe(
                frame_idx,
                uniq_vox,
                conf_mean_all,
                quota_override=recovery_reference_quota,
                allow_existing_refresh=bool(repair_mode and self.geo_recovery_reference_refresh),
            )
        elif allow_reference_refresh_only:
            self._update_reference_bank_from_keyframe(
                frame_idx,
                uniq_vox,
                conf_mean_all,
                quota_override=0,
                allow_existing_refresh=True,
            )

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
        runtime_map_ready = self._update_geo_runtime_ready(frame_idx, force_refresh=True)
        self._update_geo_runtime_state(
            frame_idx=frame_idx,
            matched_ratio=float(matched_ratio),
            ref_overlap=int(ref_overlap),
            runtime_map_ready=runtime_map_ready,
            policy=policy,
        )

        structure_ready_now = self._update_geo_structure_ready_streak(frame_idx)
        self.geo_last_observation = {
            "frame_idx": int(frame_idx),
            "matched_ratio": float(matched_ratio),
            "new_voxel_ratio": float(new_voxels) / max(1.0, float(num_groups)),
            "ref_overlap": float(ref_overlap),
            "trust_score": float(self.geo_trust_score),
            "bootstrap_bank_ready": float(1.0 if bootstrap_bank_ready else 0.0),
            "structure_ready": float(1.0 if structure_ready_now else 0.0),
            "allow_reference_refresh_only": float(1.0 if allow_reference_refresh_only else 0.0),
        }

        bootstrap_voxel_count = int(len(self.geo_voxel_bank))
        bootstrap_stable_anchor_count = int(len(self.geo_stable_anchor_voxels))
        bootstrap_keyframe_count = int(len(self.geo_keyframes))
        bootstrap_ready_recomputed = bool(self._geo_bootstrap_bank_ready(frame_idx))
        self.geo_last_policy_inputs["bootstrap_voxel_count"] = int(bootstrap_voxel_count)
        self.geo_last_policy_inputs["bootstrap_stable_anchor_count"] = int(bootstrap_stable_anchor_count)
        self.geo_last_policy_inputs["bootstrap_keyframe_count"] = int(bootstrap_keyframe_count)
        self.geo_last_policy_inputs["bootstrap_ready_recomputed"] = bool(bootstrap_ready_recomputed)

        if self._should_log_geo_bootstrap(int(frame_idx)):
            logger.info(
                "[geo_bootstrap] frame=%d voxel_bank=%d ref_bank=%d stable_anchors=%d keyframes=%d bootstrap_ready_recomputed=%d trust=%.4f recovery=%d reloc=%d matched_ratio=%.4f ref_overlap=%d",
                int(frame_idx),
                int(bootstrap_voxel_count),
                int(len(self.geo_reference_bank)),
                int(bootstrap_stable_anchor_count),
                int(bootstrap_keyframe_count),
                int(bootstrap_ready_recomputed),
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
            "allow_reference_refresh_only": float(1.0 if allow_reference_refresh_only else 0.0),
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
            "geo_role": meta.get("geo_role", Aggregator._compute_primary_geo_role(meta)).index_select(0, keep_cpu),
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
            "geo_role": torch.cat(
                [
                    meta_a.get("geo_role", Aggregator._compute_primary_geo_role(meta_a)),
                    meta_b.get("geo_role", Aggregator._compute_primary_geo_role(meta_b)),
                ],
                dim=0,
            ),
        }

    @staticmethod
    def _hard_protected_mask(meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        frame_idx = meta["frame_idx"]
        if frame_idx.numel() == 0:
            return torch.empty(0, dtype=torch.bool)
        local_idx = meta.get("local_patch_idx", torch.full_like(frame_idx, -1))
        frame0_patch = (frame_idx == 0) & (local_idx >= 0)
        # Keep frame-0 patch backbone hard-protected across geo modes.
        return frame0_patch

    @staticmethod
    def _ordered_add(
        selected_set: set[int],
        selected_order: List[int],
        tokens: Union[List[int], torch.Tensor, Iterable[int]],
    ) -> None:
        if torch.is_tensor(tokens):
            items = [int(v) for v in tokens.detach().cpu().long().tolist()]
        else:
            items = [int(v) for v in tokens]
        for token in items:
            if token not in selected_set:
                selected_set.add(token)
                selected_order.append(token)

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

        # Critical guard: never re-rank/prune when already under budget.
        if keep.numel() <= int(budget):
            return keep

        frame_idx_all = meta["frame_idx"]
        is_special_all = meta["is_special"]
        geo_role_all = meta.get("geo_role", self._compute_primary_geo_role(meta))
        current_frame_idx = int(frame_idx_all.max().item()) if frame_idx_all.numel() > 0 else 0
        recent_min = max(0, current_frame_idx - int(recent_frames))

        frame_keep = frame_idx_all.index_select(0, keep)
        special_keep = is_special_all.index_select(0, keep)
        identity_local_all = meta.get("identity_local", meta.get("local_patch_idx", torch.full_like(frame_idx_all, -1)))
        identity_local_keep = identity_local_all.index_select(0, keep)
        role_keep = geo_role_all.index_select(0, keep)
        keyframe_keep = role_keep == 1
        anchor_keep = role_keep == 3
        reference_keep = role_keep == 4

        frame0_keep = frame_keep == 0
        recent_keep = frame_keep >= recent_min
        local_keep = meta.get("local_patch_idx", torch.full_like(frame_idx_all, -1)).index_select(0, keep)

        def _take_tail(idx_tensor: torch.Tensor, n: int) -> torch.Tensor:
            if n <= 0 or idx_tensor.numel() == 0:
                return torch.empty((0,), dtype=torch.long)
            if idx_tensor.numel() <= n:
                return idx_tensor
            return idx_tensor[-n:]

        # Hard reservation groups (must be retained before any soft eviction).
        special_recent_frames = max(1, int(recent_frames))
        special_quota = min(max(16, int(budget // 8)), 128)
        frame0_camera = keep[(frame_keep == 0) & special_keep & (identity_local_keep == -1)]
        recent_special = keep[special_keep & (frame_keep >= max(0, current_frame_idx - special_recent_frames + 1))]
        hard_special = _take_tail(torch.unique(torch.cat([frame0_camera, recent_special], dim=0), sorted=True), special_quota)
        frame0_patch = keep[frame0_keep & (~special_keep)]
        frame0_quota = min(int(self.geo_frame0_backbone_quota), max(128, int(max(0, budget // 8))))
        hard_frame0 = _take_tail(frame0_patch, frame0_quota)
        anchor_quota = max(1, int(float(budget) * float(self.geo_anchor_budget_ratio)))
        reference_quota = int(self.geo_reference_token_quota)
        if self.geo_recovery_frames_left > 0:
            reference_quota = int(max(reference_quota, round(reference_quota * self.geo_recovery_ref_boost)))
        hard_anchor = _take_tail(keep[anchor_keep], min(anchor_quota, budget))
        hard_reference = _take_tail(keep[reference_keep], min(reference_quota, budget))
        hard_keyframe = _take_tail(keep[keyframe_keep], min(int(self.geo_keyframe_protected_quota), budget))

        hard_idx = torch.unique(
            torch.cat([hard_special, hard_reference, hard_anchor, hard_keyframe, hard_frame0], dim=0),
            sorted=True,
        )
        if hard_idx.numel() >= budget:
            # Budget cannot fit all hard-reserved tokens: preserve frame0/special first, then anchors, then keyframes.
            parts: List[torch.Tensor] = []
            remain = int(budget)
            for part in [
                torch.unique(torch.cat([hard_special, hard_frame0], dim=0), sorted=True),
                torch.unique(hard_keyframe, sorted=True),
                torch.unique(hard_reference, sorted=True),
                torch.unique(hard_anchor, sorted=True),
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

        # Soft pools: after hard reservation, fill from recent then all leftovers.
        remain = budget - int(hard_idx.numel())
        soft_recent = keep[recent_keep & ~(special_keep | frame0_keep)]
        hard_mask = torch.isin(keep, hard_idx)
        soft_global = keep[(~hard_mask) & (~torch.isin(keep, soft_recent)) & (~special_keep)]

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
    def _sanitize_keep_idx_preserve_order(keep_idx: torch.Tensor, meta_len: int, kv_len: int) -> torch.Tensor:
        if keep_idx is None or keep_idx.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        upper = min(int(meta_len), int(kv_len))
        if upper <= 0:
            return torch.empty(0, dtype=torch.long)

        keep_cpu = keep_idx.detach().cpu().long().view(-1)
        out: List[int] = []
        seen = set()
        for v in keep_cpu.tolist():
            iv = int(v)
            if iv < 0 or iv >= upper:
                continue
            if iv in seen:
                continue
            seen.add(iv)
            out.append(iv)
        if not out:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(out, dtype=torch.long)

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
        geo_role = self._compute_primary_geo_role({
            "frame_idx": frame_idx_t,
            "is_special": is_special,
            "is_keyframe": is_keyframe,
            "is_anchor": torch.zeros((tokens_per_frame,), dtype=torch.bool),
            "is_landmark": torch.zeros((tokens_per_frame,), dtype=torch.bool),
            "is_reference": torch.zeros((tokens_per_frame,), dtype=torch.bool),
        })

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
            "geo_role": geo_role,
        }

    @staticmethod
    def _unique_preserve_order_long(x: torch.Tensor) -> torch.Tensor:
        if x is None or x.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        x_cpu = x.detach().cpu().long().view(-1)
        out: List[int] = []
        seen = set()
        for v in x_cpu.tolist():
            iv = int(v)
            if iv in seen:
                continue
            seen.add(iv)
            out.append(iv)
        if not out:
            return torch.empty((0,), dtype=torch.long)
        return torch.tensor(out, dtype=torch.long)

    @staticmethod
    def _build_identity_keep_from_meta(
        meta: Dict[str, torch.Tensor],
        keep_idx: Optional[torch.Tensor],
        preserve_order: bool = True,
    ) -> torch.Tensor:
        if keep_idx is None or keep_idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        keep = (
            Aggregator._unique_preserve_order_long(keep_idx)
            if preserve_order
            else torch.unique(keep_idx.detach().cpu().long(), sorted=True)
        )
        if keep.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        gid = meta.get("global_id", torch.empty((0,), dtype=torch.long))
        if gid.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        keep = keep[(keep >= 0) & (keep < gid.numel())]
        if keep.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        if preserve_order:
            global_id = gid.index_select(0, keep)
            return Aggregator._unique_preserve_order_long(global_id)

        frame = meta["frame_idx"].index_select(0, keep)
        global_id = gid.index_select(0, keep)
        geo_role = meta.get("geo_role", Aggregator._compute_primary_geo_role(meta)).index_select(0, keep)

        def _rank(i: int):
            f = int(frame[i].item())
            r = int(geo_role[i].item())
            return (
                1 if r == 5 else 0,
                1 if r == 4 else 0,
                1 if r == 3 else 0,
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
        policy: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Apply strict hard-cap in identity space, then keep original identity order.

        This keeps the shared keep-plan semantic in identity space instead of
        implicitly relying on per-layer positional index alignment.
        """
        if identity_keep is None or identity_keep.numel() == 0 or budget <= 0:
            return torch.empty((0,), dtype=torch.long)

        idx = self._identity_keep_to_index(meta, identity_keep, preserve_order=True)
        if idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        idx = self._cap_keep_for_geo_mode(
            meta=meta,
            keep_idx=idx,
            budget=int(budget),
            recent_frames=int(recent_frames),
            current_frame_idx=int(meta["frame_idx"].max().item()) if meta["frame_idx"].numel() > 0 else -1,
            policy=policy,
            priority_keep_idx=idx,
        )
        if idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        chosen_identity = self._build_identity_keep_from_meta(
            meta,
            idx,
            preserve_order=True,
        )
        if chosen_identity.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        identity_keep_cpu = self._unique_preserve_order_long(identity_keep)
        keep_mask = torch.isin(identity_keep_cpu, chosen_identity)
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
    def _identity_keep_to_index(
        meta: Dict[str, torch.Tensor],
        identity_keep: torch.Tensor,
        preserve_order: bool = True,
    ) -> torch.Tensor:
        if identity_keep is None or identity_keep.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        Aggregator._ensure_identity_lookup(meta)

        keys = (
            Aggregator._unique_preserve_order_long(identity_keep)
            if preserve_order
            else torch.unique(identity_keep.detach().cpu().long(), sorted=True)
        )
        if keys.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        if bool(meta.get("_gid_is_sorted", False)):
            gid_sorted = meta.get("_gid_sorted", torch.empty((0,), dtype=torch.long))
            if gid_sorted.numel() == 0:
                return torch.empty((0,), dtype=torch.long)
            where = torch.searchsorted(gid_sorted, keys)
            valid = where < gid_sorted.numel()
            if valid.sum().item() == 0:
                return torch.empty((0,), dtype=torch.long)
            where = where[valid]
            keys_v = keys[valid]
            matched = gid_sorted.index_select(0, where) == keys_v
            if matched.sum().item() == 0:
                return torch.empty((0,), dtype=torch.long)
            out = where[matched]
            return (
                Aggregator._unique_preserve_order_long(out)
                if preserve_order
                else torch.unique(out, sorted=True)
            )

        gid_to_pos = meta.get("_gid_to_pos", {})
        if not gid_to_pos:
            return torch.empty(0, dtype=torch.long)

        out: List[int] = []
        seen = set()
        for key in keys.detach().cpu().long().tolist():
            k = int(key)
            if k in seen:
                continue
            seen.add(k)
            if k in gid_to_pos:
                out.append(int(gid_to_pos[k]))
        if not out:
            return torch.empty((0,), dtype=torch.long)
        out_t = torch.tensor(out, dtype=torch.long)
        return (
            Aggregator._unique_preserve_order_long(out_t)
            if preserve_order
            else torch.unique(out_t, sorted=True)
        )


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

    def _summarize_kv_meta(
        self,
        meta: Optional[Dict[str, torch.Tensor]],
        recent_frames: int = 2,
        subset_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, int]:
        empty = {
            "total": 0,
            "special": 0,
            "frame0": 0,
            "recent": 0,
            "keyframe": 0,
            "anchor": 0,
            "landmark": 0,
            "reference": 0,
            "anchor_only": 0,
            "landmark_only": 0,
            "reference_only": 0,
            "keyframe_only": 0,
            "multi_tag": 0,
            "plain_patch": 0,
        }
        if meta is None or "frame_idx" not in meta:
            return empty

        frame_idx = meta["frame_idx"]
        n = int(frame_idx.numel())
        if n == 0:
            return empty

        if subset_idx is None:
            take = torch.arange(n, dtype=torch.long)
        else:
            take = torch.unique(subset_idx.detach().cpu().long(), sorted=True)
            take = take[(take >= 0) & (take < n)]
            if take.numel() == 0:
                return empty

        def _get(name: str) -> torch.Tensor:
            base = meta.get(name, torch.zeros((n,), dtype=torch.bool))
            return base.index_select(0, take)

        frame_take = frame_idx.index_select(0, take)
        m = int(frame_take.numel())
        is_special = _get("is_special")
        role_src = meta.get("geo_role", self._compute_primary_geo_role(meta))
        geo_role = role_src.index_select(0, take)
        is_keyframe = geo_role == 1
        is_landmark = geo_role == 2
        is_anchor = geo_role == 3
        is_reference = geo_role == 4

        max_frame = int(frame_take.max().item()) if m > 0 else -1
        recent_mask = frame_take >= max(-1, max_frame - int(recent_frames) + 1)
        frame0_mask = frame_take == 0

        non_special = ~is_special
        anchor_only = non_special & (geo_role == 3)
        landmark_only = non_special & (geo_role == 2)
        reference_only = non_special & (geo_role == 4)
        keyframe_only = non_special & (geo_role == 1)

        multi_tag = torch.zeros_like(non_special)
        plain_patch = non_special & (geo_role == 0)

        return {
            "total": m,
            "special": int(is_special.sum().item()),
            "frame0": int(frame0_mask.sum().item()),
            "recent": int(recent_mask.sum().item()),
            "keyframe": int(is_keyframe.sum().item()),
            "anchor": int(is_anchor.sum().item()),
            "landmark": int(is_landmark.sum().item()),
            "reference": int(is_reference.sum().item()),
            "anchor_only": int(anchor_only.sum().item()),
            "landmark_only": int(landmark_only.sum().item()),
            "reference_only": int(reference_only.sum().item()),
            "keyframe_only": int(keyframe_only.sum().item()),
            "multi_tag": int(multi_tag.sum().item()),
            "plain_patch": int(plain_patch.sum().item()),
        }

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
        kv_comp_old: Optional[Dict[str, int]] = None,
        kv_comp_keep: Optional[Dict[str, int]] = None,
        kv_comp_new: Optional[Dict[str, int]] = None,
    ):
        self._emit_console_geo_log(
            current_frame_idx=current_frame_idx,
            total_tokens=total_tokens,
            candidate_count=candidate_count,
            visible_total=visible_total,
            selected_count=selected_count,
            anchor_count=anchor_count,
            stable_count=stable_count,
            tau_bucket=tau_bucket,
            stable_visible_voxel_overlap=stable_visible_voxel_overlap,
            stable_selected_visible=stable_selected_visible,
            stable_selected_invisible=stable_selected_invisible,
            fast_path=fast_path,
            cache_size=cache_size,
            keep_overlap_cache=keep_overlap_cache,
            reanchor_added=reanchor_added,
            reanchor_overlap_avg=reanchor_overlap_avg,
            budget=budget,
            kv_comp_old=kv_comp_old,
            kv_comp_keep=kv_comp_keep,
            kv_comp_new=kv_comp_new,
            guard_frame=True,
        )

    def _update_geo_selector_diag(
        self,
        *,
        current_frame_idx: int,
        stable_visible_voxel_overlap: int,
        stable_selected_visible: int,
        stable_selected_invisible: int,
        visible_total: int,
        selected_total: int,
        hard_keep_continuity: Optional[float] = None,
        frame0_pin_ratio: Optional[float] = None,
    ) -> None:
        vis = float(stable_selected_visible)
        invis = float(stable_selected_invisible)
        stable_vis_ratio = vis / float(max(1.0, vis + invis))
        self.geo_last_selector_diag = {
            "frame_idx": int(current_frame_idx),
            "stable_visible_overlap": float(stable_visible_voxel_overlap),
            "stable_visible_ratio": float(stable_vis_ratio),
            "visible_total": float(visible_total),
            "selected_total": float(selected_total),
            "hard_keep_continuity": None if hard_keep_continuity is None else float(hard_keep_continuity),
            "frame0_pin_ratio": None if frame0_pin_ratio is None else float(frame0_pin_ratio),
        }

    def _queue_geo_console_log(
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
        kv_comp_old: Optional[Dict[str, int]] = None,
        kv_comp_keep: Optional[Dict[str, int]] = None,
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
        self.geo_pending_console_log = {
            "current_frame_idx": int(current_frame_idx),
            "total_tokens": int(total_tokens),
            "candidate_count": int(candidate_count),
            "visible_total": int(visible_total),
            "selected_count": int(selected_count),
            "anchor_count": int(anchor_count),
            "stable_count": int(stable_count),
            "tau_bucket": float(tau_bucket),
            "stable_visible_voxel_overlap": stable_visible_voxel_overlap,
            "stable_selected_visible": stable_selected_visible,
            "stable_selected_invisible": stable_selected_invisible,
            "fast_path": fast_path,
            "cache_size": cache_size,
            "keep_overlap_cache": keep_overlap_cache,
            "reanchor_added": reanchor_added,
            "reanchor_overlap_avg": reanchor_overlap_avg,
            "budget": budget,
            "kv_comp_old": kv_comp_old,
            "kv_comp_keep": kv_comp_keep,
        }

    def _flush_geo_console_log(self, kv_comp_new: Optional[Dict[str, int]] = None):
        payload = self.geo_pending_console_log
        if payload is None:
            return
        self._emit_console_geo_log(
            current_frame_idx=int(payload["current_frame_idx"]),
            total_tokens=int(payload["total_tokens"]),
            candidate_count=int(payload["candidate_count"]),
            visible_total=int(payload["visible_total"]),
            selected_count=int(payload["selected_count"]),
            anchor_count=int(payload["anchor_count"]),
            stable_count=int(payload["stable_count"]),
            tau_bucket=float(payload["tau_bucket"]),
            stable_visible_voxel_overlap=payload.get("stable_visible_voxel_overlap"),
            stable_selected_visible=payload.get("stable_selected_visible"),
            stable_selected_invisible=payload.get("stable_selected_invisible"),
            fast_path=payload.get("fast_path"),
            cache_size=payload.get("cache_size"),
            keep_overlap_cache=payload.get("keep_overlap_cache"),
            reanchor_added=payload.get("reanchor_added"),
            reanchor_overlap_avg=payload.get("reanchor_overlap_avg"),
            budget=payload.get("budget"),
            kv_comp_old=payload.get("kv_comp_old"),
            kv_comp_keep=payload.get("kv_comp_keep"),
            kv_comp_new=kv_comp_new,
            guard_frame=True,
        )
        self.geo_pending_console_log = None

    def _emit_console_geo_log(
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
        kv_comp_old: Optional[Dict[str, int]] = None,
        kv_comp_keep: Optional[Dict[str, int]] = None,
        kv_comp_new: Optional[Dict[str, int]] = None,
        guard_frame: bool = True,
    ):
        if current_frame_idx < 0:
            return
        interval = int(self.geo_console_log_interval)
        if interval < 0:
            return
        interval = max(1, interval)
        if (int(current_frame_idx) % interval) != 0:
            return
        if guard_frame and int(self.geo_last_console_log_frame) == int(current_frame_idx):
            return
        if guard_frame:
            self.geo_last_console_log_frame = int(current_frame_idx)

        print(
            f"[geo_prune] total={int(total_tokens)} candidate={int(candidate_count)} "
            f"visible={int(visible_total)} selected={int(selected_count)} "
            f"anchor_in_cache={int(anchor_count)} stable_selected={int(stable_count)} "
            f"tau_bucket={float(tau_bucket):.4f}",
            flush=True,
        )
        if kv_comp_old is not None or kv_comp_keep is not None or kv_comp_new is not None:
            def _fmt_raw(prefix: str, comp: Optional[Dict[str, int]]) -> str:
                if comp is None:
                    return f"{prefix}=NA"
                return (
                    f"{prefix}_total={int(comp['total'])} {prefix}_special={int(comp['special'])} "
                    f"{prefix}_frame0={int(comp['frame0'])} {prefix}_recent={int(comp['recent'])} "
                    f"{prefix}_keyframe={int(comp['keyframe'])} {prefix}_anchor={int(comp['anchor'])} "
                    f"{prefix}_landmark={int(comp['landmark'])} {prefix}_reference={int(comp['reference'])}"
                )

            def _fmt_split(prefix: str, comp: Optional[Dict[str, int]]) -> str:
                if comp is None:
                    return f"{prefix}=NA"
                return (
                    f"{prefix}_special={int(comp['special'])} {prefix}_anchor_only={int(comp['anchor_only'])} "
                    f"{prefix}_landmark_only={int(comp['landmark_only'])} {prefix}_reference_only={int(comp['reference_only'])} "
                    f"{prefix}_keyframe_only={int(comp['keyframe_only'])} {prefix}_multi_tag={int(comp['multi_tag'])} "
                    f"{prefix}_plain_patch={int(comp['plain_patch'])}"
                )

            print(
                f"[geo_prune_kv_raw] {_fmt_raw('old', kv_comp_old)} {_fmt_raw('keep', kv_comp_keep)} {_fmt_raw('new', kv_comp_new)}",
                flush=True,
            )
            print(
                f"[geo_prune_kv_split] {_fmt_split('old', kv_comp_old)} {_fmt_split('keep', kv_comp_keep)} {_fmt_split('new', kv_comp_new)}",
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

        policy = self.geo_last_committed_policy or self._geo_default_policy(int(current_frame_idx))
        inp = self.geo_last_policy_inputs if isinstance(self.geo_last_policy_inputs, dict) else {}
        met = self.geo_last_policy_metrics if isinstance(self.geo_last_policy_metrics, dict) else {}
        bootstrap_voxel_count_live = int(len(self.geo_voxel_bank))
        bootstrap_stable_anchor_count_live = int(len(self.geo_stable_anchor_voxels))
        bootstrap_keyframe_count_live = int(len(self.geo_keyframes))
        bootstrap_ready_recomputed_live = bool(self._geo_bootstrap_bank_ready(int(current_frame_idx)))
        print(
            f"[geo_policy] preview_policy_frame={int(inp.get('preview_policy_frame', -1))} "
            f"commit_policy_frame={int(inp.get('commit_policy_frame', self.geo_last_policy_frame))} "
            f"policy_view_source={str(inp.get('policy_view_source', 'none'))} "
            f"selector_view_source={str(inp.get('selector_view_source', 'none'))} "
            f"policy_mode={str(policy.get('mode', 'legacy'))} "
            f"effective_mode={str(inp.get('effective_mode', 'legacy'))} "
            f"selector_exec_mode={str(inp.get('effective_mode', 'legacy'))} "
            f"observation_frame={int(inp.get('observation_frame', -1))} selector_diag_frame={int(inp.get('selector_diag_frame', -1))} "
            f"total_tokens={int(inp.get('total_tokens', 0) or 0)} max_past_tokens={inp.get('max_past_tokens', None)} "
            f"raw_ref_budget={int(inp.get('raw_ref_budget', 0) or 0)} base_ref_layer_budget={int(inp.get('base_ref_layer_budget', 0) or 0)} final_ref_budget={int(inp.get('final_ref_budget', 0) or 0)} "
            f"final_ref_layer_budget={int(inp.get('final_ref_layer_budget', 0) or 0)} allow_cap={bool(inp.get('allow_cap', False))} "
            f"fixed_point_iters_used={int(inp.get('fixed_point_iters_used', 0) or 0)} fixed_point_converged={bool(inp.get('fixed_point_converged', False))} "
            f"maturity={float(met.get('maturity', self.geo_maturity_ema)):.4f} instability={float(met.get('instability', self.geo_instability_ema)):.4f} "
            f"use_view_pruning={bool(inp.get('use_view_pruning', policy.get('use_view_pruning', True)))} "
            f"use_anchor_labels={bool(inp.get('use_anchor_labels', policy.get('use_anchor_labels', False)))} "
            f"safe_warmup={bool(inp.get('safe_warmup', False))} "
            f"bootstrap_bank_ready={bool(inp.get('bootstrap_bank_ready', policy.get('bootstrap_bank_ready', False)))} "
            f"bootstrap_voxel_count={int(bootstrap_voxel_count_live)} "
            f"bootstrap_stable_anchor_count={int(bootstrap_stable_anchor_count_live)} "
            f"bootstrap_keyframe_count={int(bootstrap_keyframe_count_live)} "
            f"bootstrap_ready_recomputed={bool(bootstrap_ready_recomputed_live)} "
            f"structure_ready={bool(inp.get('structure_ready', policy.get('structure_ready', False)))} "
            f"exec_policy_mode={str(inp.get('exec_policy_mode', inp.get('effective_mode', 'legacy')))} "
            f"exec_use_cap={bool(inp.get('exec_use_cap', False))} "
            f"layer_cap_policy_mode={str(inp.get('layer_cap_policy_mode', inp.get('exec_policy_mode', 'legacy')))} "
            f"early_budget_floor_applied={bool(inp.get('early_budget_floor_applied', False))} "
            f"shared_ref_early_floor_applied={bool(inp.get('shared_ref_early_floor_applied', False))} "
            f"shared_keep_order_preserved={bool(inp.get('shared_keep_order_preserved', False))} "
            f"allow_fill_effective={bool(inp.get('allow_fill_effective', True))} "
            f"fast_path_allow_fill={bool(inp.get('fast_path_allow_fill', False))} "
            f"selector_diag_updated={bool(inp.get('selector_diag_updated', True))} "
            f"selector_diag_proxy_backfill={bool(inp.get('selector_diag_proxy_backfill', False))} "
            f"selector_diag_true_visible_total={int(inp.get('selector_diag_true_visible_total', 0) or 0)} "
            f"legacy_observation_break={bool(inp.get('legacy_observation_break', policy.get('legacy_observation_break', False)))} "
            f"legacy_break_force_recent_plain={bool(inp.get('legacy_break_force_recent_plain', False))} "
            f"legacy_break_anchor_scale={float(inp.get('legacy_break_anchor_scale', 1.0) or 1.0):.4f} "
            f"legacy_break_frame0_scale={float(inp.get('legacy_break_frame0_scale', 1.0) or 1.0):.4f} "
            f"legacy_break_recent_plain_ratio={float(inp.get('legacy_break_recent_plain_ratio', 0.08) or 0.08):.4f} "
            f"observation_stress={float(inp.get('observation_stress', 0.0) or 0.0):.4f} "
            f"reserved_ratio_prev={float(inp.get('reserved_ratio_prev', 0.0) or 0.0):.4f} "
            f"reserved_target_effective={float(inp.get('reserved_target_effective', 0.06) or 0.06):.4f} "
            f"keep_plain_patch_reserved_requested={int(inp.get('keep_plain_patch_reserved_requested', 0) or 0)} "
            f"frame0_hard_scale={float(inp.get('frame0_hard_scale', 1.0) or 1.0):.4f} "
            f"reference_hard_scale={float(inp.get('reference_hard_scale', 1.0) or 1.0):.4f} "
            f"shared_ref_budget_upper={int(inp.get('shared_ref_budget_upper', 0) or 0)} "
            f"shared_ref_prev_layer_budget={int(inp.get('shared_ref_prev_layer_budget', 0) or 0)} "
            f"visible_ref_quota_effective={int(inp.get('visible_ref_quota_effective', 0) or 0)} "
            f"invis_ref_quota_effective={int(inp.get('invis_ref_quota_effective', 0) or 0)} "
            f"recent_plain_floor_diverse={bool(inp.get('recent_plain_floor_diverse', False))} "
            f"recent_plain_ratio_effective={float(inp.get('recent_plain_ratio_effective', 0.06) or 0.06):.4f} "
            f"frame_keep_plain_patch_final_min={int(inp.get('frame_keep_plain_patch_final_min', 0) or 0)} "
            f"frame_keep_plain_patch_reserved_min={int(inp.get('frame_keep_plain_patch_reserved_min', 0) or 0)} "
            f"frame_keep_budget_min={int(inp.get('frame_keep_budget_min', 0) or 0)} "
            f"priority_keep_fastpath_has_plain_floor={bool(inp.get('priority_keep_fastpath_has_plain_floor', False))} "
            f"implicit_recent_plain_floor_used={bool(inp.get('implicit_recent_plain_floor_used', False))} "
            f"fastpath_recent_plain_floor_added={int(inp.get('fastpath_recent_plain_floor_added', 0) or 0)} "
            f"fastpath_recent_frames_eff={int(inp.get('fastpath_recent_frames_eff', 0) or 0)} "
            f"keep_plain_patch_reserved_prev_is_fastpath_safe={bool(inp.get('keep_plain_patch_reserved_prev_is_fastpath_safe', False))} "
            f"frame0_priority_after_plain={bool(inp.get('frame0_priority_after_plain', False))} "
            f"current_recovery_ref_before_frame0={bool(inp.get('current_recovery_ref_before_frame0', False))} "
            f"extra_frame0_soft_promotion_enabled={bool(inp.get('extra_frame0_soft_promotion_enabled', False))} "
            f"frame0_hard_capped_diverse={int(inp.get('frame0_hard_capped_diverse', 0) or 0)} "
            f"keep_plain_patch_hard_floor={int(inp.get('keep_plain_patch_hard_floor', 0) or 0)} "
            f"hard_cap_unique_budget={bool(inp.get('hard_cap_unique_budget', False))} "
            f"frame0_quota_effective={int(inp.get('frame0_quota_effective', 0) or 0)} "
            f"anchor_ttl_effective={int(inp.get('anchor_ttl_effective', 0) or 0)} "
            f"keep_plain_patch_reserved={int(inp.get('keep_plain_patch_reserved', 0) or 0)} "
            f"keep_plain_patch_final={int(inp.get('keep_plain_patch_final', 0) or 0)} "
            f"frame0_hard_kept={int(inp.get('frame0_hard_kept', 0) or 0)} "
            f"structure_ready_latched={bool(inp.get('structure_ready_latched', self.geo_structure_ready_latched))} "
            f"structure_unready_streak={int(inp.get('structure_unready_streak', self.geo_structure_unready_streak) or 0)} "
            f"landmark_growth_ready={bool(inp.get('landmark_growth_ready', policy.get('landmark_growth_ready', False)))} "
            f"reference_growth_ready={bool(inp.get('reference_growth_ready', policy.get('reference_growth_ready', False)))} "
            f"landmark_label_ready={bool(inp.get('landmark_label_ready', policy.get('landmark_label_ready', policy.get('use_landmark_labels', False))))} "
            f"reference_label_ready={bool(inp.get('reference_label_ready', policy.get('reference_label_ready', policy.get('use_reference_labels', False))))} "
            f"anchor_phase_open={bool(inp.get('anchor_phase_open', policy.get('anchor_phase_open', False)))} "
            f"landmark_phase_open={bool(inp.get('landmark_phase_open', policy.get('landmark_phase_open', False)))} "
            f"reference_phase_open={bool(inp.get('reference_phase_open', policy.get('reference_phase_open', False)))} "
            f"reloc_phase_open={bool(inp.get('reloc_phase_open', policy.get('reloc_phase_open', False)))} "
            f"use_recovery={bool(inp.get('use_recovery', policy.get('use_recovery', False)))} "
            f"allow_reloc_trigger={bool(inp.get('allow_reloc_trigger', policy.get('allow_reloc_trigger', False)))} "
            f"reloc_gate_open={bool(inp.get('reloc_gate_open', inp.get('allow_reloc_trigger', policy.get('allow_reloc_trigger', False))))} "
            f"use_reloc={bool(inp.get('use_reloc', policy.get('use_reloc', False)))} "
            f"ongoing_recovery={bool(inp.get('ongoing_recovery', int(self.geo_recovery_frames_left) > 0))} "
            f"recovery_timer_active={bool(inp.get('recovery_timer_active', int(self.geo_recovery_frames_left) > 0))} "
            f"ongoing_reloc={bool(inp.get('ongoing_reloc', int(self.geo_reloc_frames_left) > 0 or str(self.geo_reloc_state) != 'off'))} "
            f"recovery_selector={bool(inp.get('recovery_selector', False))} "
            f"allow_reference_refresh_only={bool(inp.get('allow_reference_refresh_only', False))} "
            f"ref_budget_source={str(inp.get('ref_budget_source', 'na'))} "
            f"prefer_last_reliable_view={bool(inp.get('prefer_last_reliable_view', policy.get('prefer_last_reliable_view', False)))} "
            f"use_cap={bool(policy.get('use_cap', False))} cap_alpha={float(policy.get('cap_alpha', 0.0)):.4f}",
            flush=True,
        )

    def _should_log_geo_debug(self, current_frame_idx: int) -> bool:
        if current_frame_idx < 0:
            return False
        interval = int(self.geo_console_log_interval)
        if interval < 0:
            return False
        interval = max(1, interval)
        if (int(current_frame_idx) % interval) != 0:
            return False
        if int(self.geo_last_debug_log_frame) == int(current_frame_idx):
            return False
        self.geo_last_debug_log_frame = int(current_frame_idx)
        return True

    def _should_log_geo_bootstrap(self, current_frame_idx: int) -> bool:
        if current_frame_idx < 0:
            return False
        interval = int(self.geo_console_log_interval)
        if interval < 0:
            return False
        interval = max(1, interval)
        if (int(current_frame_idx) % interval) != 0:
            return False
        if int(self.geo_last_bootstrap_log_frame) == int(current_frame_idx):
            return False
        self.geo_last_bootstrap_log_frame = int(current_frame_idx)
        return True

    def _extract_landmark_identity_cache(
        self,
        meta: Dict[str, torch.Tensor],
        keep_idx: torch.Tensor,
        max_past_tokens: Optional[int],
    ) -> torch.Tensor:
        if keep_idx is None or keep_idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        keep = self._sanitize_keep_idx_preserve_order(
            keep_idx.detach().cpu().long(),
            meta_len=int(meta["frame_idx"].numel()),
            kv_len=int(meta["frame_idx"].numel()),
        )
        is_special = meta["is_special"].index_select(0, keep)
        geo_role = meta.get("geo_role", self._compute_primary_geo_role(meta)).index_select(0, keep)

        cache_idx = keep[(~is_special) & ((geo_role == 4) | (geo_role == 3) | (geo_role == 1))]
        if cache_idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        gid = meta.get("global_id", torch.empty((0,), dtype=torch.long))
        if gid.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        cache_identity = gid.index_select(0, cache_idx)
        if max_past_tokens is not None:
            cap = min(2048, max(64, int(max_past_tokens * 0.15)))
        else:
            cap = 1024
        if cache_identity.numel() > cap:
            cache_identity = cache_identity[-cap:]
        return self._unique_preserve_order_long(cache_identity.detach().cpu().long())

    @staticmethod
    def _count_keep_cache_overlap_identity(
        meta: Dict[str, torch.Tensor],
        keep_idx: torch.Tensor,
        cache_identity: torch.Tensor,
    ) -> int:
        if keep_idx is None or cache_identity is None:
            return 0
        if keep_idx.numel() == 0 or cache_identity.numel() == 0:
            return 0
        gid = meta.get("global_id", torch.empty((0,), dtype=torch.long))
        if gid.numel() == 0:
            return 0
        keep_ids = gid.index_select(0, keep_idx.detach().cpu().long())
        return int(torch.isin(keep_ids, cache_identity.detach().cpu().long()).sum().item())

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
        if int(cache_frame_idx) < int(self.geo_bootstrap_frames):
            return False

        stable_ready = len(self.geo_stable_anchor_voxels) >= 128
        landmark_ready = len(self.geo_landmark_voxels) >= 128
        ref_ready = len(self.geo_reference_bank) >= 16
        bank_target = max(512, int(1.2 * len(self.geo_stable_anchor_voxels)))
        bank_ready = len(self.geo_voxel_bank) >= int(bank_target)
        return bool(stable_ready and (landmark_ready or ref_ready or bank_ready))

    def _simple_non_geo_keep(self, meta: Dict[str, torch.Tensor], budget: int, recent_frames: int = 2) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n <= 0 or budget <= 0:
            return torch.empty((0,), dtype=torch.long)
        if n <= int(budget):
            return torch.arange(n, dtype=torch.long)
        idx_all = torch.arange(n, dtype=torch.long)
        is_special = meta.get("is_special", torch.zeros((n,), dtype=torch.bool))
        is_keyframe = meta.get("is_keyframe", torch.zeros((n,), dtype=torch.bool))
        is_anchor = meta.get("is_anchor", torch.zeros((n,), dtype=torch.bool))
        is_landmark = meta.get("is_landmark", torch.zeros((n,), dtype=torch.bool))
        is_reference = meta.get("is_reference", torch.zeros((n,), dtype=torch.bool))
        frame_idx = meta.get("frame_idx", torch.zeros((n,), dtype=torch.long))
        max_frame = int(frame_idx.max().item()) if frame_idx.numel() > 0 else -1
        recent_mask = frame_idx >= max(-1, max_frame - int(recent_frames) + 1)

        def _take_recent_quota(idx: torch.Tensor, quota: int) -> torch.Tensor:
            if idx.numel() == 0 or quota <= 0:
                return torch.empty((0,), dtype=torch.long)
            if idx.numel() <= int(quota):
                return idx
            order = torch.argsort(frame_idx.index_select(0, idx))
            idx = idx.index_select(0, order)
            return idx[-int(quota):]

        special_recent_frames = max(1, int(recent_frames))
        special_quota = min(max(16, int(budget // 8)), 128)
        special = idx_all[is_special]
        identity_local = meta.get("identity_local", meta.get("local_patch_idx", torch.full((n,), -1, dtype=torch.long)))
        frame0_camera = special[
            (frame_idx.index_select(0, special) == 0)
            & (identity_local.index_select(0, special) == -1)
        ] if special.numel() > 0 else torch.empty((0,), dtype=torch.long)
        recent_special = special[
            frame_idx.index_select(0, special) >= max(0, max_frame - special_recent_frames + 1)
        ] if special.numel() > 0 else torch.empty((0,), dtype=torch.long)
        bounded_special = torch.unique(torch.cat([frame0_camera, recent_special], dim=0), sorted=True)
        if bounded_special.numel() > special_quota:
            bounded_special = bounded_special[-special_quota:]

        current_frame_idx = int(frame_idx.max().item()) if frame_idx.numel() > 0 else -1
        anchor_ready = self._geo_anchor_ready(current_frame_idx)

        ref_keep = _take_recent_quota(idx_all[is_reference], quota=256)
        landmark_keep = _take_recent_quota(idx_all[is_landmark], quota=512)
        anchor_keep = _take_recent_quota(idx_all[is_anchor], quota=256) if anchor_ready else torch.empty((0,), dtype=torch.long)
        keyframe_keep = _take_recent_quota(idx_all[is_keyframe], quota=256)

        keep = torch.unique(
            torch.cat([bounded_special, ref_keep, landmark_keep, anchor_keep, keyframe_keep, idx_all[recent_mask]], dim=0),
            sorted=True,
        )
        if keep.numel() >= int(budget):
            return self._cap_keep_with_protection(meta, keep, budget=int(budget), recent_frames=recent_frames)

        leftover = idx_all[~torch.isin(idx_all, keep)]
        if leftover.numel() > 0:
            order = torch.argsort(frame_idx.index_select(0, leftover))
            leftover = leftover.index_select(0, order)
            need = int(budget) - int(keep.numel())
            keep = torch.unique(torch.cat([keep, leftover[-need:]], dim=0), sorted=True)

        if keep.numel() > int(budget):
            keep = self._cap_keep_with_protection(meta, keep, budget=int(budget), recent_frames=recent_frames)
        return keep

    def _build_reloc_identity_keep(
        self,
        meta: Dict[str, torch.Tensor],
        max_past_tokens: int,
        recent_frames: int,
        policy: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        n = int(meta["frame_idx"].numel())
        if n == 0 or int(max_past_tokens) <= 0:
            return torch.empty((0,), dtype=torch.long)
        idx_all = torch.arange(n, dtype=torch.long)
        frame_idx = meta["frame_idx"]
        is_special = meta.get("is_special", torch.zeros((n,), dtype=torch.bool))
        geo_role = meta.get("geo_role", self._compute_primary_geo_role(meta))
        is_reference = geo_role == 4
        is_landmark = geo_role == 2
        is_anchor = geo_role == 3
        is_keyframe = geo_role == 1

        max_frame = int(frame_idx.max().item()) if frame_idx.numel() > 0 else -1
        recent_mask = frame_idx >= max(-1, max_frame - max(1, int(recent_frames)) + 1)
        recency = (frame_idx.to(torch.float32) - float(max_frame)).clamp(min=-16.0, max=0.0)
        score = 0.05 * (recency + 16.0)
        score = score + is_reference.to(torch.float32) * 3.0 + is_landmark.to(torch.float32) * 2.0 + is_anchor.to(torch.float32) * 1.5 + is_keyframe.to(torch.float32) * 1.0

        bounded_special = self._bounded_special_idx(meta, idx_all, budget=int(max_past_tokens), recent_frames=recent_frames)
        selected_set: set[int] = set()
        selected_order: List[int] = []
        selected_mask = torch.zeros((n,), dtype=torch.bool)
        self._ordered_add(selected_set, selected_order, bounded_special)
        if bounded_special.numel() > 0:
            selected_mask[bounded_special] = True

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
            self._ordered_add(selected_set, selected_order, out)
            selected_mask[out] = True

        if policy is not None:
            local_ratio = max(float(self.geo_reloc_local_budget_ratio), float(policy["local_budget_ratio"]))
            stable_ratio = max(float(self.geo_reloc_stable_read_budget_ratio), float(policy["stable_read_budget_ratio"]))
            base_budget = int(max_past_tokens)
            q_ref = max(256, int(base_budget * stable_ratio))
            q_land = max(128, int(base_budget * stable_ratio * 0.5))
            q_anchor = max(128, int(base_budget * local_ratio * 0.5))
            q_key = max(256, int(base_budget * local_ratio))
            q_recent = max(128, int(base_budget * local_ratio * 0.5))
        else:
            q_ref, q_land, q_anchor, q_key, q_recent = 2048, 1024, 1024, 2048, 512
        _pick(is_reference, q_ref)
        _pick(is_landmark, q_land)
        _pick(is_anchor, q_anchor)
        _pick(is_keyframe, q_key)
        _pick(recent_mask & (~is_special), q_recent)

        keep = torch.tensor(selected_order, dtype=torch.long) if selected_order else torch.empty((0,), dtype=torch.long)
        hard_mode = str(self.geo_reloc_state) == "hard"
        if (not hard_mode) and keep.numel() < int(max_past_tokens):
            remain = int(max_past_tokens) - int(keep.numel())
            extra = idx_all[(~selected_mask) & (is_reference | is_landmark | is_anchor | is_keyframe | recent_mask)]
            if extra.numel() > 0 and remain > 0:
                k = min(int(extra.numel()), int(min(remain, 512)))
                sc = score.index_select(0, extra)
                top = torch.topk(sc, k=k, largest=True).indices
                extra_tokens = extra.index_select(0, top)
                self._ordered_add(selected_set, selected_order, extra_tokens)
                keep = torch.tensor(selected_order, dtype=torch.long) if selected_order else torch.empty((0,), dtype=torch.long)

        keep = self._sanitize_keep_idx_preserve_order(keep, meta_len=n, kv_len=n)
        keep = self._cap_keep_for_geo_mode(
            meta=meta,
            keep_idx=keep,
            budget=int(max_past_tokens),
            recent_frames=int(recent_frames),
            current_frame_idx=int(meta["frame_idx"].max().item()) if meta["frame_idx"].numel() > 0 else -1,
            policy=policy,
            priority_keep_idx=keep,
        )
        return self._build_identity_keep_from_meta(meta, keep, preserve_order=True)

    def _select_patch_diverse(
        self,
        patch_idx: torch.Tensor,
        patch_local: torch.Tensor,
        patch_conf: torch.Tensor,
        quota: int,
        full_patch_count: int,
        grid_n: int = 4,
    ) -> torch.Tensor:
        if patch_idx.numel() == 0 or int(quota) <= 0:
            return torch.empty((0,), dtype=torch.long)

        patch_n = max(1, int(full_patch_count))
        side = max(1, int(round(float(patch_n) ** 0.5)))
        bin_h = max(1, side // max(1, int(grid_n)))
        bin_w = max(1, side // max(1, int(grid_n)))

        best_per_cell: Dict[Tuple[int, int], Tuple[float, int]] = {}
        g = max(1, int(grid_n))
        for j in range(int(patch_idx.numel())):
            lp = int(patch_local[j].item())
            y = lp // side
            x = lp % side
            cell = (min(g - 1, y // bin_h), min(g - 1, x // bin_w))
            sc = float(patch_conf[j].item())
            gid = int(patch_idx[j].item())
            prev = best_per_cell.get(cell)
            if prev is None or sc > prev[0]:
                best_per_cell[cell] = (sc, gid)

        chosen: List[int] = [v[1] for v in best_per_cell.values()]
        if len(chosen) < int(quota):
            order = torch.argsort(patch_conf, descending=True)
            for k in order.tolist():
                gid = int(patch_idx[int(k)].item())
                if gid not in chosen:
                    chosen.append(gid)
                if len(chosen) >= int(quota):
                    break

        return self._unique_preserve_order_long(torch.tensor(chosen[: int(quota)], dtype=torch.long))

    def _select_frame0_patch_diverse(
        self,
        frame0_patch_idx: torch.Tensor,
        frame0_local: torch.Tensor,
        frame0_conf: torch.Tensor,
        quota: int,
        full_patch_count: int,
        grid_n: int = 4,
    ) -> torch.Tensor:
        return self._select_patch_diverse(
            frame0_patch_idx,
            frame0_local,
            frame0_conf,
            quota=int(quota),
            full_patch_count=int(full_patch_count),
            grid_n=int(grid_n),
        )

    def _select_geo_active_indices_legacy_early_core(
        self,
        meta: Dict[str, torch.Tensor],
        topk_per_voxel: int,
        recent_frames: int,
        near: float,
        far: float,
        current_view: Optional[Dict[str, torch.Tensor]],
        use_view_pruning: bool = True,
        max_past_tokens: Optional[int] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> Tuple[set[int], Dict[str, Any]]:
        total_tokens = int(meta["frame_idx"].numel())
        if total_tokens == 0:
            return set(), {
                "current_frame_idx": 0,
                "recent_frames_eff": int(recent_frames),
                "candidate_count": 0,
                "visible_total": 0,
                "anchor_count": 0,
                "stable_count": 0,
                "tau_bucket": float("nan"),
                "stable_visible_voxel_overlap": 0,
                "stable_selected_visible": 0,
                "stable_selected_invisible": 0,
                "fast_path": 8,
                "reanchor_added": 0,
                "reanchor_overlap_avg": 0.0,
            }

        frame_idx = meta["frame_idx"]
        is_special = meta.get("is_special", torch.zeros((total_tokens,), dtype=torch.bool))
        geo_role = meta.get("geo_role", self._compute_primary_geo_role(meta))
        is_keyframe = geo_role == 1
        is_anchor = geo_role == 3
        is_reference = geo_role == 4
        _ = is_reference
        local_idx = meta.get("local_patch_idx", torch.full((total_tokens,), -1, dtype=torch.long))

        current_frame_idx = int(frame_idx.max().item()) if frame_idx.numel() > 0 else 0
        hard_recent_frames = int(policy["hard_recent_frames"]) if policy is not None else int(self.geo_legacy_hard_recent_frames)
        recent_frames_eff = int(policy["recent_window"]) if policy is not None else max(int(recent_frames), int(self.geo_legacy_recent_window), int(self.geo_early_recent_frames))
        soft_recent_frames_eff = int(policy["soft_recent_window"]) if policy is not None else max(int(recent_frames_eff), int(self.geo_legacy_soft_recent_frames))
        if current_frame_idx < int(self.geo_early_stabilize_frames):
            hard_recent_frames = max(hard_recent_frames, 20)
        hard_recent_frames = min(hard_recent_frames, int(soft_recent_frames_eff))
        recent_min = max(0, current_frame_idx - int(recent_frames_eff))
        soft_recent_min = max(0, current_frame_idx - int(soft_recent_frames_eff))
        recent_mask = frame_idx >= recent_min

        candidate_indices = torch.empty((0,), dtype=torch.long)
        gather_idx: List[torch.Tensor] = []
        gather_score: List[torch.Tensor] = []
        gather_hash: List[torch.Tensor] = []
        gather_bank_conf: List[torch.Tensor] = []
        gather_visible: List[torch.Tensor] = []

        idx_all = torch.empty((0,), dtype=torch.long)
        score_all = torch.empty((0,), dtype=torch.float32)
        hash_all = torch.empty((0,), dtype=torch.long)
        bank_conf_all = torch.empty((0,), dtype=torch.float32)
        visible_all = torch.empty((0,), dtype=torch.bool)

        stable_hash = torch.empty((0,), dtype=torch.long)

        world_to_cam_diag = None
        intrinsic_diag = None
        img_hw_diag = None

        selected: set[int] = set()
        selected_order: List[int] = []
        self._ordered_add(selected, selected_order, torch.nonzero(is_special, as_tuple=False).flatten())

        hard_recent_idx = self._hard_recent_patch_idx(meta, hard_recent_frames=hard_recent_frames)
        if hard_recent_idx.numel() > 0:
            self._ordered_add(selected, selected_order, hard_recent_idx)

        frame0_patch = torch.nonzero((frame_idx == 0) & (~is_special) & (local_idx >= 0), as_tuple=False).flatten()
        if frame0_patch.numel() > 0:
            frame0_patch_cap = int((policy or {}).get("frame0_patch_cap", self.geo_frame0_patch_cap))
            if bool((policy or {}).get("legacy_observation_break", False)):
                frame0_patch_cap = max(128, int(round(float(frame0_patch_cap) * float((policy or {}).get("legacy_break_frame0_scale", 0.50)))))
            q0 = min(int(frame0_patch_cap), int(frame0_patch.numel()))
            frame0_meta = self.geo_frame_meta.get(0)
            if frame0_meta is not None and frame0_meta.get("conf") is not None and frame0_meta["conf"].numel() > 0:
                frame0_local = local_idx.index_select(0, frame0_patch).long()
                valid = (frame0_local >= 0) & (frame0_local < frame0_meta["conf"].shape[0])
                frame0_patch = frame0_patch[valid]
                frame0_local = frame0_local[valid]
                q0 = min(int(frame0_patch_cap), int(frame0_patch.numel()))
                if q0 > 0 and frame0_patch.numel() > 0:
                    conf0 = frame0_meta["conf"].index_select(0, frame0_local).to(torch.float32)
                    frame0_keep = self._select_frame0_patch_diverse(
                        frame0_patch,
                        frame0_local,
                        conf0,
                        quota=q0,
                        full_patch_count=int(frame0_meta["conf"].shape[0]),
                        grid_n=4,
                    )
                    self._ordered_add(selected, selected_order, frame0_keep)
            # If frame-0 confidence metadata is missing, skip frame0 patch promotion
            # instead of falling back to positional slicing.

        keyframe_idx = torch.nonzero(is_keyframe & (~is_special), as_tuple=False).flatten()
        if keyframe_idx.numel() > 0:
            kq = min(int(self.geo_keyframe_protected_quota), int(keyframe_idx.numel()))
            self._ordered_add(selected, selected_order, keyframe_idx[-kq:])

        anchor_idx = torch.nonzero(is_anchor & (~is_special), as_tuple=False).flatten()
        if anchor_idx.numel() > 0:
            if max_past_tokens is not None:
                anchor_quota = min(256, max(64, int((float(policy["anchor_quota_ratio"]) if policy is not None else 0.05) * int(max_past_tokens))))
            else:
                anchor_quota = 256
            if bool((policy or {}).get("legacy_observation_break", False)):
                anchor_quota = max(64, int(round(float(anchor_quota) * float((policy or {}).get("legacy_break_anchor_scale", 0.25)))))
            anchor_keep = self._take_recent_quota(anchor_idx, frame_idx=frame_idx, quota=int(anchor_quota))
            self._ordered_add(selected, selected_order, anchor_keep)

        hard_recent_flag = torch.zeros((total_tokens,), dtype=torch.bool)
        if hard_recent_idx.numel() > 0:
            hard_recent_flag.index_fill_(0, hard_recent_idx, True)
        soft_recent_mask = (frame_idx >= soft_recent_min) & (~hard_recent_flag)

        # Strengthen early legacy retention: keep an inner recent ring directly,
        # then let local-budget logic act on the outer soft-recent ring.
        inner_recent_frames = min(
            int(self.geo_legacy_soft_recent_frames),
            max(int(self.geo_legacy_hard_recent_frames) + 4, int(self.geo_legacy_hard_recent_frames)),
        )
        if current_frame_idx < int(self.geo_early_stabilize_frames):
            inner_recent_frames = min(int(self.geo_legacy_soft_recent_frames), max(inner_recent_frames, 8))
        inner_recent_min = max(0, current_frame_idx - int(inner_recent_frames))
        inner_recent_idx = torch.nonzero(
            (frame_idx >= inner_recent_min) & (~is_special) & (local_idx >= 0) & (~hard_recent_flag),
            as_tuple=False,
        ).flatten()
        if inner_recent_idx.numel() > 0:
            self._ordered_add(selected, selected_order, inner_recent_idx)

        recent_patch_idx = torch.nonzero(
            soft_recent_mask & (~is_special) & (local_idx >= 0) & (frame_idx < inner_recent_min),
            as_tuple=False,
        ).flatten()
        if recent_patch_idx.numel() > 0:
            recent_frame_count = max(1, int(torch.unique(frame_idx.index_select(0, recent_patch_idx)).numel()))
            floor_budget = int(self.geo_legacy_min_keep_per_recent_frame) * int(recent_frame_count)
            local_budget = int(max_past_tokens) if max_past_tokens is not None else int(recent_patch_idx.numel())
            local_budget = min(
                local_budget,
                int(max(floor_budget, round(float(local_budget) * float(policy["local_budget_ratio"])) if policy is not None else round(float(local_budget) * float(self.geo_local_budget_ratio)))),
            )
            take: List[torch.Tensor] = []
            frame_for_recent = frame_idx.index_select(0, recent_patch_idx)
            for f in range(soft_recent_min, current_frame_idx + 1):
                idx_f = recent_patch_idx[frame_for_recent == int(f)]
                if idx_f.numel() == 0:
                    continue
                take.append(idx_f[: min(int(self.geo_local_budget_cap_per_frame), int(idx_f.numel()))])
            if take:
                take_idx = torch.unique(torch.cat(take, dim=0), sorted=True)
                if take_idx.numel() > local_budget:
                    take_idx = take_idx[-local_budget:]
                self._ordered_add(selected, selected_order, take_idx)

        legacy_diag_overlap = 0
        legacy_diag_stable_selected_visible = 0
        legacy_diag_stable_selected_invisible = 0
        legacy_diag_visible_total = 0
        legacy_diag_measured = False
        selector_diag_proxy_backfill = False
        legacy_recent_plain_reserved = torch.empty((0,), dtype=torch.long)
        if use_view_pruning and current_view is not None and current_view.get("world_to_cam") is not None and current_view.get("intrinsic") is not None:
            world_to_cam_diag = current_view["world_to_cam"]
            intrinsic_diag = current_view["intrinsic"]
            if isinstance(world_to_cam_diag, torch.Tensor):
                world_to_cam_diag = world_to_cam_diag.detach().cpu()
            if isinstance(intrinsic_diag, torch.Tensor):
                intrinsic_diag = intrinsic_diag.detach().cpu()
            if isinstance(world_to_cam_diag, torch.Tensor) and world_to_cam_diag.ndim == 3:
                world_to_cam_diag = world_to_cam_diag[0]
            if isinstance(intrinsic_diag, torch.Tensor) and intrinsic_diag.ndim == 3:
                intrinsic_diag = intrinsic_diag[0]
            img_hw_diag = current_view.get("img_hw")

            candidate_mask = (~is_special) & (~recent_mask) & (local_idx >= 0)
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            for fidx in torch.unique(frame_idx[candidate_indices]).tolist() if candidate_indices.numel() > 0 else []:
                fm = self.geo_frame_meta.get(int(fidx))
                if fm is None:
                    continue
                in_frame = candidate_indices[frame_idx[candidate_indices] == int(fidx)]
                if in_frame.numel() == 0:
                    continue
                local = local_idx.index_select(0, in_frame).long()
                valid = (local >= 0) & (local < fm["pts"].shape[0])
                if valid.sum().item() == 0:
                    continue
                in_frame = in_frame[valid]
                local = local[valid]
                pts = fm["pts"].index_select(0, local).to(torch.float32)
                conf = fm["conf"].index_select(0, local).to(torch.float32)
                vox = fm["voxel_ids"].index_select(0, local)
                visible = self._frustum_mask(pts, world_to_cam_diag.to(torch.float32), intrinsic_diag.to(torch.float32), near=near, far=far, img_hw=img_hw_diag)
                bank_conf = []
                bank_sup = []
                bank_var = []
                for key in (tuple(int(v) for v in row) for row in vox.tolist()):
                    it = self.geo_voxel_bank.get(key)
                    if it is None:
                        bank_conf.append(0.0); bank_sup.append(1.0); bank_var.append(0.0)
                    else:
                        bank_conf.append(float(it.get("conf_ema", 0.0))); bank_sup.append(float(it.get("support", 1.0))); bank_var.append(float(it.get("pos_var", 0.0)))
                bank_conf_t = torch.tensor(bank_conf, dtype=torch.float32)
                conf_safe = conf.clamp_min(1e-6)
                bank_conf_eff = torch.where(bank_conf_t > 0, bank_conf_t, conf_safe)
                score = conf_safe * bank_conf_eff.clamp_min(1e-6) * torch.log1p(torch.tensor(bank_sup, dtype=torch.float32)) * (1.0 / (1.0 + torch.tensor(bank_var, dtype=torch.float32)))
                score = score * torch.where(visible, torch.ones_like(score), torch.full_like(score, float(self.geo_invisible_read_weight)))
                gather_idx.append(in_frame.to(torch.long))
                gather_score.append(score)
                gather_hash.append(self._voxel_hash(vox))
                gather_bank_conf.append(bank_conf_eff)
                gather_visible.append(visible.to(torch.bool))
            if gather_idx:
                idx_all = torch.cat(gather_idx, dim=0)
                score_all = torch.cat(gather_score, dim=0)
                hash_all = torch.cat(gather_hash, dim=0)
                bank_conf_all = torch.cat(gather_bank_conf, dim=0)
                visible_all = torch.cat(gather_visible, dim=0)
                legacy_diag_visible_total = int(visible_all.sum().item())
                tau = self._compute_dynamic_bucket_threshold(bank_conf_all.tolist(), int(max_past_tokens or 0))
                valid = bank_conf_all >= float(tau)
                grouped = self._group_topk_by_hash(hash_all[valid], score_all[valid], idx_all[valid], topk_per_voxel=max(1, int(topk_per_voxel)))
                if grouped.numel() > 0:
                    self._ordered_add(selected, selected_order, grouped)

                stable_source = self.geo_reference_voxels if len(self.geo_reference_voxels) > 0 else self.geo_stable_map_voxels
                if stable_source:
                    stable_hash = self._voxel_hash(torch.tensor(sorted(stable_source), dtype=torch.long))
                    stable_mask = torch.isin(hash_all, stable_hash)
                    selected_mask = torch.isin(idx_all, torch.tensor(sorted(selected), dtype=torch.long)) if idx_all.numel() > 0 else torch.zeros((0,), dtype=torch.bool)
                    legacy_diag_overlap = int(torch.unique(hash_all[stable_mask & visible_all]).numel())
                    legacy_diag_stable_selected_visible = int((selected_mask & stable_mask & visible_all).sum().item())
                    legacy_diag_stable_selected_invisible = int((selected_mask & stable_mask & (~visible_all)).sum().item())
                    legacy_diag_measured = True

        if bool((policy or {}).get("legacy_observation_break", False)):
            legacy_recent_plain_reserved = self._legacy_recent_plain_floor_idx(
                meta=meta,
                current_frame_idx=int(current_frame_idx),
                recent_frames_eff=int(recent_frames_eff),
                max_past_tokens=max_past_tokens,
                policy=policy,
            )
            if legacy_recent_plain_reserved.numel() > 0:
                self._ordered_add(selected, selected_order, legacy_recent_plain_reserved)

        proxy = None
        if not legacy_diag_measured:
            proxy = self._geo_proxy_selector_diag(
                current_frame_idx=current_frame_idx,
                selected_total=len(selected),
            )

        if proxy is not None:
            selector_diag_proxy_backfill = True
            if int(legacy_diag_overlap) <= 0:
                legacy_diag_overlap = int(proxy["stable_visible_overlap"])
            if int(legacy_diag_stable_selected_visible) <= 0:
                legacy_diag_stable_selected_visible = int(proxy["stable_selected_visible"])
            if int(legacy_diag_stable_selected_invisible) <= 0:
                legacy_diag_stable_selected_invisible = int(proxy["stable_selected_invisible"])
            if int(legacy_diag_visible_total) <= 0:
                legacy_diag_visible_total = int(proxy["visible_total"])

        diag_payload: Dict[str, Any] = {
            "current_frame_idx": int(current_frame_idx),
            "recent_frames_eff": int(recent_frames_eff),
            "candidate_count": int(candidate_indices.numel()),
            "visible_total": int(legacy_diag_visible_total),
            "anchor_count": int(anchor_idx.numel()),
            "stable_count": int(0),
            "tau_bucket": float("nan"),
            "stable_visible_voxel_overlap": int(legacy_diag_overlap),
            "stable_selected_visible": int(legacy_diag_stable_selected_visible),
            "stable_selected_invisible": int(legacy_diag_stable_selected_invisible),
            "fast_path": int(8),
            "reanchor_added": int(0),
            "reanchor_overlap_avg": float(0.0),
            "selected_order": list(selected_order),
            "diag_idx_all": idx_all.detach().cpu(),
            "diag_hash_all": hash_all.detach().cpu(),
            "diag_visible_all": visible_all.detach().cpu(),
            "diag_stable_hash": stable_hash.detach().cpu(),
            "diag_world_to_cam": world_to_cam_diag.detach().cpu() if isinstance(world_to_cam_diag, torch.Tensor) else None,
            "diag_intrinsic": intrinsic_diag.detach().cpu() if isinstance(intrinsic_diag, torch.Tensor) else None,
            "diag_img_hw": img_hw_diag,
            "diag_near": float(near),
            "diag_far": float(far),
            "selector_diag_proxy_backfill": bool(selector_diag_proxy_backfill),
            "selector_diag_true_visible_total": int(legacy_diag_visible_total),
            "legacy_recent_plain_reserved_idx": legacy_recent_plain_reserved.detach().cpu(),
        }
        return selected, diag_payload


    def _select_geo_active_indices_legacy_early(
        self,
        meta: Dict[str, torch.Tensor],
        topk_per_voxel: int,
        recent_frames: int,
        near: float,
        far: float,
        current_view: Optional[Dict[str, torch.Tensor]],
        use_view_pruning: bool = True,
        max_past_tokens: Optional[int] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        selected, diag = self._select_geo_active_indices_legacy_early_core(
            meta=meta,
            topk_per_voxel=topk_per_voxel,
            recent_frames=recent_frames,
            near=near,
            far=far,
            current_view=current_view,
            use_view_pruning=use_view_pruning,
            max_past_tokens=max_past_tokens,
            policy=policy,
        )
        legacy_recent_plain_reserved = diag.get("legacy_recent_plain_reserved_idx", torch.empty((0,), dtype=torch.long))
        priority_keep_idx = (
            torch.tensor(diag.get("selected_order", []), dtype=torch.long)
            if diag.get("selected_order")
            else torch.empty((0,), dtype=torch.long)
        )
        return self._finalize_geo_keep(
            meta=meta,
            selected=selected,
            selected_order=diag.get("selected_order"),
            current_frame_idx=int(diag["current_frame_idx"]),
            recent_frames_eff=int(diag["recent_frames_eff"]),
            max_past_tokens=max_past_tokens,
            candidate_count=int(diag["candidate_count"]),
            visible_total=int(diag["visible_total"]),
            anchor_count=int(diag["anchor_count"]),
            stable_count=int(diag["stable_count"]),
            tau_bucket=float(diag["tau_bucket"]),
            stable_visible_voxel_overlap=int(diag["stable_visible_voxel_overlap"]),
            stable_selected_visible=int(diag["stable_selected_visible"]),
            stable_selected_invisible=int(diag["stable_selected_invisible"]),
            fast_path=int(diag["fast_path"]),
            hard_keep_idx=None,
            reanchor_added=int(diag["reanchor_added"]),
            reanchor_overlap_avg=float(diag["reanchor_overlap_avg"]),
            diag_payload=diag,
            allow_fill=False,
            policy=policy,
            priority_keep_idx=priority_keep_idx,
            plain_reserved_idx=legacy_recent_plain_reserved,
        )

    def _recompute_selector_diag_from_final_keep(
        self,
        *,
        meta: Dict[str, torch.Tensor],
        final_keep_idx: torch.Tensor,
        diag_payload: Optional[Dict[str, Any]],
        fallback: Dict[str, int],
    ) -> Dict[str, int]:
        if diag_payload is None:
            return fallback

        idx_all = diag_payload.get("diag_idx_all", torch.empty((0,), dtype=torch.long))
        hash_all = diag_payload.get("diag_hash_all", torch.empty((0,), dtype=torch.long))
        visible_all = diag_payload.get("diag_visible_all", torch.empty((0,), dtype=torch.bool))
        stable_hash = diag_payload.get("diag_stable_hash", torch.empty((0,), dtype=torch.long))

        if idx_all.numel() == 0 or hash_all.numel() == 0 or visible_all.numel() == 0 or stable_hash.numel() == 0:
            return fallback

        keep_cpu = final_keep_idx.detach().cpu().long()
        stable_mask = torch.isin(hash_all, stable_hash)
        visible_stable_hashes = torch.unique(hash_all[stable_mask & visible_all]).detach().cpu().long()
        stable_visible_voxel_overlap = int(visible_stable_hashes.numel())
        visible_total = int(visible_all.sum().item())
        if keep_cpu.numel() == 0:
            return {
                "stable_visible_voxel_overlap": int(stable_visible_voxel_overlap),
                "stable_selected_visible": 0,
                "stable_selected_invisible": 0,
                "visible_total": int(visible_total),
            }

        selected_mask = torch.isin(idx_all, keep_cpu)
        stable_selected_visible = int((selected_mask & stable_mask & visible_all).sum().item())
        stable_selected_invisible = int((selected_mask & stable_mask & (~visible_all)).sum().item())

        # Supplement keep tokens outside measured candidate universe (e.g., hard_keep only tokens).
        in_measured = torch.isin(keep_cpu, idx_all)
        extra_keep = keep_cpu[~in_measured]
        extra_visible_hashes: List[torch.Tensor] = []
        if extra_keep.numel() > 0:
            world_to_cam = diag_payload.get("diag_world_to_cam", None)
            intrinsic = diag_payload.get("diag_intrinsic", None)
            img_hw = diag_payload.get("diag_img_hw", None)
            near_v = float(diag_payload.get("diag_near", 0.05))
            far_v = float(diag_payload.get("diag_far", 200.0))
            if isinstance(world_to_cam, torch.Tensor) and isinstance(intrinsic, torch.Tensor):
                frame_idx_all = meta.get("frame_idx", torch.empty((0,), dtype=torch.long))
                local_idx_all = meta.get("local_patch_idx", torch.full((frame_idx_all.numel(),), -1, dtype=torch.long))
                is_special_all = meta.get("is_special", torch.zeros((frame_idx_all.numel(),), dtype=torch.bool))
                for token in extra_keep.tolist():
                    t = int(token)
                    if t < 0 or t >= int(frame_idx_all.numel()):
                        continue
                    if bool(is_special_all[t].item()):
                        continue
                    fidx = int(frame_idx_all[t].item())
                    lp = int(local_idx_all[t].item())
                    if lp < 0:
                        continue
                    fm = self.geo_frame_meta.get(fidx)
                    if fm is None or fm.get("pts") is None or fm.get("voxel_ids") is None:
                        continue
                    if lp >= int(fm["pts"].shape[0]) or lp >= int(fm["voxel_ids"].shape[0]):
                        continue
                    pts = fm["pts"][lp : lp + 1].to(torch.float32)
                    vox = fm["voxel_ids"][lp : lp + 1]
                    vh = self._voxel_hash(vox).detach().cpu().long()
                    if vh.numel() == 0:
                        continue
                    if not bool(torch.isin(vh, stable_hash).item()):
                        continue
                    vis = self._frustum_mask(
                        pts,
                        world_to_cam.to(torch.float32),
                        intrinsic.to(torch.float32),
                        near=near_v,
                        far=far_v,
                        img_hw=img_hw,
                    )
                    if bool(vis.item()):
                        stable_selected_visible += 1
                        extra_visible_hashes.append(vh)
                    else:
                        stable_selected_invisible += 1

        if extra_visible_hashes:
            extra_visible_hashes_t = torch.unique(torch.cat(extra_visible_hashes, dim=0), sorted=True)
            visible_stable_hashes = torch.unique(
                torch.cat([visible_stable_hashes, extra_visible_hashes_t], dim=0),
                sorted=True,
            )
            stable_visible_voxel_overlap = int(visible_stable_hashes.numel())

        return {
            "stable_visible_voxel_overlap": int(stable_visible_voxel_overlap),
            "stable_selected_visible": int(stable_selected_visible),
            "stable_selected_invisible": int(stable_selected_invisible),
            "visible_total": int(visible_total),
        }

    def _select_geo_active_indices_bootstrap(
        self,
        meta: Dict[str, torch.Tensor],
        topk_per_voxel: int,
        recent_frames: int,
        near: float,
        far: float,
        current_view: Optional[Dict[str, torch.Tensor]],
        hard_keep: Optional[torch.Tensor],
        use_view_pruning: bool = True,
        max_past_tokens: Optional[int] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        legacy_selected, diag = self._select_geo_active_indices_legacy_early_core(
            meta=meta,
            topk_per_voxel=topk_per_voxel,
            recent_frames=recent_frames,
            near=near,
            far=far,
            current_view=current_view,
            use_view_pruning=use_view_pruning,
            max_past_tokens=max_past_tokens,
            policy=policy,
        )
        hard = torch.unique(hard_keep.detach().cpu().long(), sorted=True) if hard_keep is not None else torch.empty((0,), dtype=torch.long)
        merged = set(int(v) for v in legacy_selected)
        merged_order = list(diag.get("selected_order", []))
        self._ordered_add(merged, merged_order, hard)
        legacy_recent_plain_reserved = diag.get("legacy_recent_plain_reserved_idx", torch.empty((0,), dtype=torch.long))
        return self._finalize_geo_keep(
            meta=meta,
            selected=merged,
            selected_order=merged_order,
            current_frame_idx=int(diag["current_frame_idx"]),
            recent_frames_eff=int(diag["recent_frames_eff"]),
            max_past_tokens=max_past_tokens,
            candidate_count=int(diag["candidate_count"]),
            visible_total=int(diag["visible_total"]),
            anchor_count=int(diag["anchor_count"]),
            stable_count=int(diag["stable_count"]),
            tau_bucket=float(diag["tau_bucket"]),
            stable_visible_voxel_overlap=int(diag["stable_visible_voxel_overlap"]),
            stable_selected_visible=int(diag["stable_selected_visible"]),
            stable_selected_invisible=int(diag["stable_selected_invisible"]),
            fast_path=int(diag["fast_path"]),
            hard_keep_idx=hard,
            reanchor_added=int(diag["reanchor_added"]),
            reanchor_overlap_avg=float(diag["reanchor_overlap_avg"]),
            diag_payload=diag,
            allow_fill=False,
            policy=policy,
            plain_reserved_idx=legacy_recent_plain_reserved,
        )

    def _legacy_recent_plain_floor_idx(
        self,
        meta: Dict[str, torch.Tensor],
        *,
        current_frame_idx: int,
        recent_frames_eff: int,
        max_past_tokens: Optional[int],
        policy: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        n = int(meta.get("frame_idx", torch.empty((0,), dtype=torch.long)).numel())
        if n <= 0:
            return torch.empty((0,), dtype=torch.long)

        if not bool((policy or {}).get("legacy_observation_break", False)):
            return torch.empty((0,), dtype=torch.long)

        frame_idx = meta["frame_idx"]
        is_special = meta.get("is_special", torch.zeros((n,), dtype=torch.bool))
        geo_role = meta.get("geo_role", self._compute_primary_geo_role(meta))
        local_idx = meta.get("local_patch_idx", torch.full((n,), -1, dtype=torch.long))

        recent_min = max(0, int(current_frame_idx) - int(recent_frames_eff))
        recent_plain_idx = torch.nonzero(
            (frame_idx >= recent_min)
            & (~is_special)
            & (geo_role == 0)
            & (local_idx >= 0),
            as_tuple=False,
        ).flatten()
        if recent_plain_idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        if bool((policy or {}).get("legacy_break_force_recent_plain", False)):
            ratio = float((policy or {}).get("legacy_break_recent_plain_ratio", 0.12))
            if max_past_tokens is not None:
                quota = max(512, int(ratio * int(max_past_tokens)))
            else:
                quota = int(recent_plain_idx.numel())
        else:
            if max_past_tokens is not None:
                quota = max(256, int(0.08 * int(max_past_tokens)))
            else:
                quota = int(recent_plain_idx.numel())

        recent_plain_frame = frame_idx.index_select(0, recent_plain_idx)
        uniq_recent_frames = torch.unique(recent_plain_frame).sort(descending=True).values
        per_frame_quota = max(
            16,
            int(math.ceil(float(quota) / float(max(1, uniq_recent_frames.numel())))),
        )

        picked_parts: List[torch.Tensor] = []
        for f in uniq_recent_frames.tolist():
            idx_f = recent_plain_idx[recent_plain_frame == int(f)]
            if idx_f.numel() == 0:
                continue
            local_f = local_idx.index_select(0, idx_f).long()
            fm = self.geo_frame_meta.get(int(f))
            if fm is not None and fm.get("conf") is not None and fm["conf"].numel() > 0:
                valid = (local_f >= 0) & (local_f < fm["conf"].shape[0])
                idx_f = idx_f[valid]
                local_f = local_f[valid]
                if idx_f.numel() == 0:
                    continue
                conf_f = fm["conf"].index_select(0, local_f).to(torch.float32)
                kf = min(int(per_frame_quota), int(idx_f.numel()))
                picked_parts.append(
                    self._select_patch_diverse(
                        idx_f,
                        local_f,
                        conf_f,
                        quota=int(kf),
                        full_patch_count=int(fm["conf"].shape[0]),
                        grid_n=max(2, int(self.geo_local_coverage_grid)),
                    )
                )
            else:
                kf = min(int(per_frame_quota), int(idx_f.numel()))
                picked_parts.append(idx_f[:kf])

        if not picked_parts:
            return torch.empty((0,), dtype=torch.long)

        reserve = self._unique_preserve_order_long(torch.cat(picked_parts, dim=0)).detach().cpu().long()
        if reserve.numel() > int(quota):
            reserve = reserve[: int(quota)]
        return reserve

    def _implicit_recent_plain_floor_idx(
        self,
        meta: Dict[str, torch.Tensor],
        *,
        current_frame_idx: int,
        recent_frames_eff: int,
        max_past_tokens: Optional[int],
        policy: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        n = int(meta.get("frame_idx", torch.empty((0,), dtype=torch.long)).numel())
        if n <= 0:
            return torch.empty((0,), dtype=torch.long)

        mode_now = str((policy or {}).get("mode", "legacy"))
        if mode_now not in {"current", "recovery"}:
            return torch.empty((0,), dtype=torch.long)

        frame_idx = meta["frame_idx"]
        is_special = meta.get("is_special", torch.zeros((n,), dtype=torch.bool))
        geo_role = meta.get("geo_role", self._compute_primary_geo_role(meta))
        local_idx = meta.get("local_patch_idx", torch.full((n,), -1, dtype=torch.long))

        recent_min = max(0, int(current_frame_idx) - int(recent_frames_eff))
        recent_plain_idx = torch.nonzero(
            (frame_idx >= recent_min)
            & (~is_special)
            & (geo_role == 0)
            & (local_idx >= 0),
            as_tuple=False,
        ).flatten()
        if recent_plain_idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)

        obs_stress = float((policy or {}).get("observation_stress", 0.0))
        recent_plain_ratio = 0.06 + 0.04 * float(obs_stress)
        self.geo_last_policy_inputs["recent_plain_ratio_effective"] = float(recent_plain_ratio)
        if max_past_tokens is not None:
            quota = max(192, int(recent_plain_ratio * int(max_past_tokens)))
        else:
            quota = int(recent_plain_idx.numel())

        recent_plain_frame = frame_idx.index_select(0, recent_plain_idx)
        uniq_recent_frames = torch.unique(recent_plain_frame).sort(descending=True).values
        per_frame_quota = max(
            16,
            int(math.ceil(float(quota) / float(max(1, uniq_recent_frames.numel())))),
        )

        picked_parts: List[torch.Tensor] = []
        for f in uniq_recent_frames.tolist():
            idx_f = recent_plain_idx[recent_plain_frame == int(f)]
            if idx_f.numel() == 0:
                continue
            local_f = local_idx.index_select(0, idx_f).long()
            fm = self.geo_frame_meta.get(int(f))
            if fm is not None and fm.get("conf") is not None and fm["conf"].numel() > 0:
                valid = (local_f >= 0) & (local_f < fm["conf"].shape[0])
                idx_f = idx_f[valid]
                local_f = local_f[valid]
                if idx_f.numel() == 0:
                    continue
                conf_f = fm["conf"].index_select(0, local_f).to(torch.float32)
                kf = min(int(per_frame_quota), int(idx_f.numel()))
                picked_parts.append(
                    self._select_patch_diverse(
                        idx_f,
                        local_f,
                        conf_f,
                        quota=int(kf),
                        full_patch_count=int(fm["conf"].shape[0]),
                        grid_n=max(2, int(self.geo_local_coverage_grid)),
                    )
                )
            else:
                kf = min(int(per_frame_quota), int(idx_f.numel()))
                picked_parts.append(idx_f[:kf])

        if not picked_parts:
            return torch.empty((0,), dtype=torch.long)

        reserve = self._unique_preserve_order_long(torch.cat(picked_parts, dim=0)).detach().cpu().long()
        if reserve.numel() > int(quota):
            reserve = reserve[: int(quota)]
        return reserve

    def _augment_fastpath_with_recent_plain_floor(
        self,
        *,
        meta: Dict[str, torch.Tensor],
        selected_set: set[int],
        selected_order: List[int],
        current_frame_idx: int,
        recent_frames_eff: int,
        max_past_tokens: Optional[int],
        policy: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        reserve = self._implicit_recent_plain_floor_idx(
            meta,
            current_frame_idx=int(current_frame_idx),
            recent_frames_eff=int(recent_frames_eff),
            max_past_tokens=max_past_tokens,
            policy=policy,
        )
        if reserve.numel() > 0:
            self._ordered_add(selected_set, selected_order, reserve)
        self.geo_last_policy_inputs["implicit_recent_plain_floor_used"] = bool(reserve.numel() > 0)
        self.geo_last_policy_inputs["fastpath_recent_plain_floor_added"] = int(reserve.numel())
        self.geo_last_policy_inputs["fastpath_recent_frames_eff"] = int(recent_frames_eff)
        self.geo_last_policy_inputs["keep_plain_patch_reserved_prev_is_fastpath_safe"] = bool(True)
        return reserve

    def _finalize_geo_keep(
        self,
        *,
        meta: Dict[str, torch.Tensor],
        selected: set[int],
        selected_order: Optional[List[int]] = None,
        current_frame_idx: int,
        recent_frames_eff: int,
        max_past_tokens: Optional[int],
        candidate_count: int,
        visible_total: int,
        anchor_count: int,
        stable_count: int,
        tau_bucket: float,
        stable_visible_voxel_overlap: int,
        stable_selected_visible: int,
        stable_selected_invisible: int,
        fast_path: int,
        reanchor_added: int = 0,
        reanchor_overlap_avg: float = 0.0,
        hard_keep_idx: Optional[torch.Tensor] = None,
        diag_payload: Optional[Dict[str, Any]] = None,
        allow_fill: bool = True,
        priority_keep_idx: Optional[torch.Tensor] = None,
        policy: Optional[Dict[str, Any]] = None,
        update_selector_diag: bool = True,
        plain_reserved_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.geo_last_policy_inputs["allow_fill_effective"] = bool(allow_fill)
        self.geo_last_policy_inputs["selector_diag_updated"] = bool(update_selector_diag)
        self.geo_last_policy_inputs["fast_path_allow_fill"] = bool(allow_fill) if int(fast_path) != 0 else bool(False)
        total_tokens = int(meta["frame_idx"].numel()) if meta.get("frame_idx") is not None else 0
        if selected_order is not None:
            order: List[int] = []
            seen: set[int] = set()
            for tok in selected_order:
                t = int(tok)
                if 0 <= t < total_tokens and t not in seen and t in selected:
                    order.append(t)
                    seen.add(t)
            for t in sorted(i for i in selected if 0 <= i < total_tokens):
                if t not in seen:
                    order.append(int(t))
                    seen.add(int(t))
            keep = torch.tensor(order, dtype=torch.long)
        else:
            keep = torch.tensor(sorted(i for i in selected if 0 <= i < total_tokens), dtype=torch.long)
        if max_past_tokens is not None:
            if keep.numel() > int(max_past_tokens):
                if hard_keep_idx is not None and hard_keep_idx.numel() > 0:
                    keep = self._cap_keep_with_hard_protection(
                        meta=meta,
                        keep_idx=keep,
                        hard_keep=hard_keep_idx,
                        budget=max(0, int(max_past_tokens)),
                        recent_frames=recent_frames_eff,
                        priority_keep_idx=priority_keep_idx if priority_keep_idx is not None else keep,
                        policy=policy,
                    )
                else:
                    keep = self._cap_keep_with_protection(
                        meta,
                        keep,
                        budget=max(0, int(max_past_tokens)),
                        recent_frames=recent_frames_eff,
                    )
            elif keep.numel() < int(max_past_tokens) and bool(allow_fill):
                keep = self._fill_keep_to_budget_preserve_order(meta, keep, budget=int(max_past_tokens), mode="balanced")

        keep_plain_reserved_requested = int(0)
        keep_plain_reserved_final = int(0)
        if plain_reserved_idx is not None and plain_reserved_idx.numel() > 0:
            plain_reserved_idx = self._sanitize_keep_idx_preserve_order(
                plain_reserved_idx.detach().cpu().long(),
                meta_len=total_tokens,
                kv_len=total_tokens,
            )
            keep_plain_reserved_requested = int(plain_reserved_idx.numel())
            if keep.numel() > 0 and plain_reserved_idx.numel() > 0:
                keep_plain_reserved_final = int(torch.isin(plain_reserved_idx, keep).sum().item())
        self.geo_last_policy_inputs["keep_plain_patch_reserved_requested"] = int(keep_plain_reserved_requested)
        self.geo_last_policy_inputs["keep_plain_patch_reserved"] = int(keep_plain_reserved_final)
        self.geo_last_policy_inputs["implicit_recent_plain_floor_used"] = bool(plain_reserved_idx is not None and keep_plain_reserved_requested > 0)
        self.geo_last_policy_inputs["keep_plain_patch_reserved_prev_is_fastpath_safe"] = bool(plain_reserved_idx is not None)

        if isinstance(diag_payload, dict):
            self.geo_last_policy_inputs["selector_diag_proxy_backfill"] = bool(diag_payload.get("selector_diag_proxy_backfill", False))
            self.geo_last_policy_inputs["selector_diag_true_visible_total"] = int(diag_payload.get("selector_diag_true_visible_total", visible_total) or 0)
        else:
            self.geo_last_policy_inputs["selector_diag_proxy_backfill"] = bool(False)
            self.geo_last_policy_inputs["selector_diag_true_visible_total"] = int(visible_total)

        diag_final = self._recompute_selector_diag_from_final_keep(
            meta=meta,
            final_keep_idx=keep,
            diag_payload=diag_payload,
            fallback={
                "stable_visible_voxel_overlap": int(stable_visible_voxel_overlap),
                "stable_selected_visible": int(stable_selected_visible),
                "stable_selected_invisible": int(stable_selected_invisible),
                "visible_total": int(visible_total),
            },
        )

        geo_role_all = meta.get("geo_role", self._compute_primary_geo_role(meta))
        keep_role = geo_role_all.index_select(0, keep.detach().cpu().long()) if keep.numel() > 0 else torch.empty((0,), dtype=torch.long)
        anchor_count_final = int((keep_role == 3).sum().item())
        plain_keep_final = int((keep_role == 0).sum().item())
        self.geo_last_policy_inputs["keep_plain_patch_final"] = int(plain_keep_final)
        layer_keep_budget = int(max_past_tokens) if max_past_tokens is not None else int(keep.numel())
        self.geo_last_policy_inputs["frame_keep_plain_patch_final_last"] = int(plain_keep_final)
        self.geo_last_policy_inputs["frame_keep_plain_patch_reserved_last"] = int(keep_plain_reserved_final)
        self.geo_last_policy_inputs["frame_keep_budget_last"] = int(layer_keep_budget)
        prev_min_plain = self.geo_last_policy_inputs.get("frame_keep_plain_patch_final_min", None)
        prev_min_reserved = self.geo_last_policy_inputs.get("frame_keep_plain_patch_reserved_min", None)
        prev_min_budget = self.geo_last_policy_inputs.get("frame_keep_budget_min", None)
        if prev_min_plain is None or prev_min_reserved is None or prev_min_budget is None:
            self.geo_last_policy_inputs["frame_keep_plain_patch_final_min"] = int(plain_keep_final)
            self.geo_last_policy_inputs["frame_keep_plain_patch_reserved_min"] = int(keep_plain_reserved_final)
            self.geo_last_policy_inputs["frame_keep_budget_min"] = int(layer_keep_budget)
        else:
            old_plain_ratio = float(prev_min_plain) / float(max(1, int(prev_min_budget)))
            new_plain_ratio = float(plain_keep_final) / float(max(1, int(layer_keep_budget)))
            old_reserved_ratio = float(prev_min_reserved) / float(max(1, int(prev_min_budget)))
            new_reserved_ratio = float(keep_plain_reserved_final) / float(max(1, int(layer_keep_budget)))
            old_score = min(old_plain_ratio, old_reserved_ratio)
            new_score = min(new_plain_ratio, new_reserved_ratio)
            if new_score < old_score:
                self.geo_last_policy_inputs["frame_keep_plain_patch_final_min"] = int(plain_keep_final)
                self.geo_last_policy_inputs["frame_keep_plain_patch_reserved_min"] = int(keep_plain_reserved_final)
                self.geo_last_policy_inputs["frame_keep_budget_min"] = int(layer_keep_budget)

        prev_cache_identity = self.geo_cached_landmark_identity_keep.detach().cpu().clone()
        overlap = self._count_keep_cache_overlap_identity(meta, keep, prev_cache_identity)
        prev_cache_size = int(prev_cache_identity.numel())
        hard_keep_continuity = float(overlap) / float(max(1, prev_cache_size))
        frame0_mask = (meta["frame_idx"] == 0) & (~meta["is_special"]) & (meta["local_patch_idx"] >= 0)
        frame0_keep_count = int((frame0_mask.index_select(0, keep.detach().cpu().long())).sum().item()) if keep.numel() > 0 else 0
        frame0_pin_ratio = float(frame0_keep_count) / float(max(1, int(self.geo_frame0_backbone_quota)))
        new_cache_identity = self._extract_landmark_identity_cache(meta, keep, max_past_tokens)
        self.geo_cached_landmark_identity_keep = new_cache_identity
        kv_old = self._summarize_kv_meta(meta, recent_frames=recent_frames_eff)
        kv_keep = self._summarize_kv_meta(meta, recent_frames=recent_frames_eff, subset_idx=keep)
        if bool(update_selector_diag):
            self._update_geo_selector_diag(
                current_frame_idx=current_frame_idx,
                stable_visible_voxel_overlap=int(diag_final["stable_visible_voxel_overlap"]),
                stable_selected_visible=int(diag_final["stable_selected_visible"]),
                stable_selected_invisible=int(diag_final["stable_selected_invisible"]),
                visible_total=int(diag_final["visible_total"]),
                selected_total=int(keep.numel()),
                hard_keep_continuity=float(hard_keep_continuity),
                frame0_pin_ratio=float(min(1.0, max(0.0, frame0_pin_ratio))),
            )
        self._queue_geo_console_log(
            current_frame_idx=current_frame_idx,
            total_tokens=total_tokens,
            candidate_count=int(candidate_count),
            visible_total=int(diag_final["visible_total"]),
            selected_count=int(keep.numel()),
            anchor_count=int(anchor_count_final),
            stable_count=int(stable_count),
            tau_bucket=float(tau_bucket),
            stable_visible_voxel_overlap=int(diag_final["stable_visible_voxel_overlap"]),
            stable_selected_visible=int(diag_final["stable_selected_visible"]),
            stable_selected_invisible=int(diag_final["stable_selected_invisible"]),
            fast_path=int(fast_path),
            cache_size=int(new_cache_identity.numel()),
            keep_overlap_cache=overlap,
            reanchor_added=int(reanchor_added),
            reanchor_overlap_avg=float(reanchor_overlap_avg),
            budget=int(max_past_tokens or 0),
            kv_comp_old=kv_old,
            kv_comp_keep=kv_keep,
        )
        return keep

    def _geo_proxy_selector_diag(
        self,
        *,
        current_frame_idx: int,
        selected_total: int,
    ) -> Dict[str, int]:
        _ = int(current_frame_idx)
        prev_diag = self._geo_get_last_selector_diag()
        prev_obs = self._geo_get_last_observation()

        proxy_overlap = max(
            int(prev_diag.get("stable_visible_overlap", 0.0)) if prev_diag else 0,
            int(prev_obs.get("ref_overlap", 0.0)) if prev_obs else 0,
        )
        proxy_vis_ratio = float(prev_diag.get("stable_visible_ratio", 0.0)) if prev_diag else 0.0
        proxy_visible_total = int(prev_diag.get("visible_total", 0.0)) if prev_diag else int(proxy_overlap)

        selected_total_i = int(max(0, selected_total))
        proxy_selected_visible = int(round(float(selected_total_i) * float(proxy_vis_ratio)))
        proxy_selected_visible = max(0, min(proxy_selected_visible, selected_total_i))
        proxy_selected_invisible = max(0, selected_total_i - proxy_selected_visible)

        return {
            "stable_visible_overlap": int(proxy_overlap),
            "stable_selected_visible": int(proxy_selected_visible),
            "stable_selected_invisible": int(proxy_selected_invisible),
            "visible_total": int(proxy_visible_total),
        }


    def _seed_geo_backbone_keep(
        self,
        *,
        meta: Dict[str, torch.Tensor],
        current_frame_idx: int,
        max_past_tokens: Optional[int],
        recent_frames_eff: int,
        policy: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        _ = int(current_frame_idx)
        _ = int(max(1, recent_frames_eff))
        n = int(meta["frame_idx"].numel()) if meta.get("frame_idx") is not None else 0
        if n <= 0:
            return torch.empty((0,), dtype=torch.long)

        ordered_parts: List[torch.Tensor] = []
        cached = self._identity_keep_to_index(
            meta,
            self.geo_cached_landmark_identity_keep,
            preserve_order=True,
        )
        if cached.numel() > 0:
            if max_past_tokens is not None:
                cache_cap = min(int(cached.numel()), max(64, int(float(max_past_tokens) * 0.25)))
                if cached.numel() > cache_cap:
                    cached = cached[-cache_cap:]
            ordered_parts.append(cached)

        idx_all = torch.arange(n, dtype=torch.long)
        frame_idx = meta["frame_idx"]
        is_special = meta.get("is_special", torch.zeros((n,), dtype=torch.bool))
        geo_role = meta.get("geo_role", self._compute_primary_geo_role(meta))
        is_reference = geo_role == 4
        is_anchor = geo_role == 3
        is_keyframe = geo_role == 1

        def _bounded_recent(mask: torch.Tensor, quota: int) -> torch.Tensor:
            if int(quota) <= 0:
                return torch.empty((0,), dtype=torch.long)
            cand = idx_all[mask & (~is_special)]
            if cand.numel() == 0:
                return cand
            return self._take_recent_quota(cand, frame_idx=frame_idx, quota=int(quota))

        if max_past_tokens is not None:
            anchor_ratio = float(policy["anchor_quota_ratio"]) if policy is not None else float(self.geo_anchor_budget_ratio)
            anchor_quota = min(
                int(self.geo_anchor_read_quota),
                max(1, int(float(max_past_tokens) * anchor_ratio)),
            )
            reference_quota = min(int(self.geo_reference_token_quota), max(16, int(float(max_past_tokens) * 0.05)))
            keyframe_quota = min(int(self.geo_keyframe_protected_quota), max(32, int(float(max_past_tokens) * 0.10)))
        else:
            anchor_quota = int(self.geo_anchor_read_quota)
            reference_quota = int(self.geo_reference_token_quota)
            keyframe_quota = int(self.geo_keyframe_protected_quota)

        ref_keep = _bounded_recent(is_reference, reference_quota)
        anchor_keep = _bounded_recent(is_anchor, anchor_quota)
        keyframe_keep = _bounded_recent(is_keyframe, keyframe_quota)

        if ref_keep.numel() > 0:
            ordered_parts.append(ref_keep)
        if anchor_keep.numel() > 0:
            ordered_parts.append(anchor_keep)
        if keyframe_keep.numel() > 0:
            ordered_parts.append(keyframe_keep)

        if not ordered_parts:
            return torch.empty((0,), dtype=torch.long)
        return self._unique_preserve_order_long(torch.cat(ordered_parts, dim=0))

    @staticmethod
    def _compute_primary_geo_role(meta: Dict[str, torch.Tensor]) -> torch.LongTensor:
        frame_idx = meta.get("frame_idx", torch.empty((0,), dtype=torch.long))
        role = torch.zeros_like(frame_idx, dtype=torch.long)
        is_special = meta.get("is_special", torch.zeros_like(frame_idx, dtype=torch.bool))
        is_keyframe = meta.get("is_keyframe", torch.zeros_like(is_special))
        is_landmark = meta.get("is_landmark", torch.zeros_like(is_special))
        is_anchor = meta.get("is_anchor", torch.zeros_like(is_special))
        is_reference = meta.get("is_reference", torch.zeros_like(is_special))
        role[is_keyframe] = 1
        role[is_landmark] = 2
        role[is_anchor] = 3
        role[is_reference] = 4
        role[is_special] = 5
        return role

    def _geo_effective_bootstrap_thresholds(self) -> Dict[str, Any]:
        return {
            "voxels": min(int(self.geo_bootstrap_min_voxels), 1536),
            "stable_anchors": min(int(self.geo_bootstrap_min_stable_anchors), 96),
            "refs": min(int(self.geo_bootstrap_min_refs), 24),
            "ref_overlap": min(float(self.geo_bootstrap_ref_overlap_thr), 32.0),
            "visible_ratio": min(float(self.geo_bootstrap_visible_ratio_thr), 0.30),
            "ready_streak": min(int(self.geo_bootstrap_ready_streak), 4),
        }

    def _geo_bootstrap_bank_ready(self, frame_idx: int) -> bool:
        thr = self._geo_effective_bootstrap_thresholds()
        if int(frame_idx) < int(self.geo_bootstrap_frames):
            return False
        return bool(
            len(self.geo_voxel_bank) >= int(thr["voxels"])
            and len(self.geo_stable_anchor_voxels) >= max(32, int(thr["stable_anchors"]) // 2)
            and len(self.geo_keyframes) >= 2
        )

    def _geo_raw_structure_ready(self) -> bool:
        thr = self._geo_effective_bootstrap_thresholds()
        bank_ok = bool(
            len(self.geo_voxel_bank) >= int(thr["voxels"])
            and len(self.geo_stable_anchor_voxels) >= int(thr["stable_anchors"])
            and len(self.geo_reference_bank) >= int(thr["refs"])
            and float(self.geo_ref_overlap_ema) >= float(thr["ref_overlap"])
        )
        if not bank_ok:
            return False
        mature_ref_bank = len(self.geo_reference_bank) >= max(32, int(1.5 * int(thr["refs"])))
        if mature_ref_bank:
            return True

        visible_thr_enter = float(thr["visible_ratio"])
        visible_thr_exit = min(visible_thr_enter, 0.10)
        visible_thr = visible_thr_exit if self._geo_structure_ready() else visible_thr_enter
        return bool(
            float(self.geo_selector_visible_ratio_ema) >= float(visible_thr)
        )

    def _update_geo_structure_ready_streak(self, frame_idx: int) -> bool:
        thr = self._geo_effective_bootstrap_thresholds()
        raw_ready = self._geo_raw_structure_ready()
        if int(frame_idx) < int(self.geo_bootstrap_frames):
            self.geo_structure_ready_streak = 0
            self.geo_structure_unready_streak = 0
            self.geo_structure_ready_latched = False
            return False

        if raw_ready:
            self.geo_structure_ready_streak += 1
            self.geo_structure_unready_streak = 0
        else:
            self.geo_structure_ready_streak = 0
            self.geo_structure_unready_streak += 1

        if (not self.geo_structure_ready_latched) and self.geo_structure_ready_streak >= int(thr["ready_streak"]):
            self.geo_structure_ready_latched = True

        exit_streak = max(3, int(thr["ready_streak"]))
        if self.geo_structure_ready_latched and self.geo_structure_unready_streak >= int(exit_streak):
            self.geo_structure_ready_latched = False

        return bool(self.geo_structure_ready_latched)

    def _geo_structure_ready(self) -> bool:
        return bool(self.geo_structure_ready_latched)

    def _build_hard_backbone_keep(
        self,
        meta: Dict[str, torch.Tensor],
        *,
        current_frame_idx: int,
        max_past_tokens: Optional[int],
        policy: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        _ = int(current_frame_idx)
        _ = policy
        total = int(meta.get("frame_idx", torch.empty((0,), dtype=torch.long)).numel())
        if total == 0:
            return torch.empty((0,), dtype=torch.long)

        frame_idx = meta["frame_idx"]
        is_special = meta.get("is_special", torch.zeros((total,), dtype=torch.bool))
        geo_role = meta.get("geo_role", self._compute_primary_geo_role(meta))
        is_reference = geo_role == 4
        is_anchor = geo_role == 3
        is_keyframe = geo_role == 1
        local_idx = meta.get("local_patch_idx", torch.full((total,), -1, dtype=torch.long))

        keep: set[int] = set()

        special_recent_frames = int(policy["recent_window"]) if policy is not None else int(self.geo_legacy_recent_window)
        bounded_special_budget = int(max_past_tokens) if max_past_tokens is not None else 1024
        bounded_special = self._bounded_special_idx(
            meta,
            torch.arange(total, dtype=torch.long),
            budget=int(bounded_special_budget),
            recent_frames=max(1, int(special_recent_frames)),
        )
        if bounded_special.numel() > 0:
            keep.update(int(v) for v in bounded_special.tolist())

        mode_now = str((policy or {}).get("mode", "legacy"))

        frame0_mask = (frame_idx == 0) & (~is_special) & (local_idx >= 0)
        frame0_idx = torch.nonzero(frame0_mask, as_tuple=False).flatten()
        frame0_keep = frame0_idx
        if frame0_idx.numel() > 0:
            if mode_now == "legacy" or max_past_tokens is None:
                frame0_quota = int(frame0_idx.numel())
                frame0_keep = frame0_idx
            else:
                frame0_hard_scale = float((policy or {}).get("frame0_hard_scale", 1.0))
                base_frame0_quota = min(
                    int(frame0_idx.numel()),
                    min(256, max(64, int(0.06 * int(max_past_tokens)))),
                )
                frame0_quota = max(48, int(round(float(base_frame0_quota) * float(frame0_hard_scale))))
                frame0_meta = self.geo_frame_meta.get(0)
                if (
                    frame0_meta is not None
                    and frame0_meta.get("conf") is not None
                    and frame0_meta["conf"].numel() > 0
                ):
                    frame0_local = local_idx.index_select(0, frame0_idx).long()
                    valid = (frame0_local >= 0) & (frame0_local < frame0_meta["conf"].shape[0])
                    frame0_idx_valid = frame0_idx[valid]
                    frame0_local_valid = frame0_local[valid]
                    if frame0_idx_valid.numel() > 0:
                        conf0 = frame0_meta["conf"].index_select(0, frame0_local_valid).to(torch.float32)
                        frame0_keep = self._select_frame0_patch_diverse(
                            frame0_idx_valid,
                            frame0_local_valid,
                            conf0,
                            quota=int(frame0_quota),
                            full_patch_count=int(frame0_meta["conf"].shape[0]),
                            grid_n=4,
                        )
                    else:
                        frame0_keep = frame0_idx[: int(frame0_quota)]
                else:
                    frame0_keep = frame0_idx[: int(frame0_quota)]
            keep.update(int(v) for v in frame0_keep.tolist())
        self.geo_last_policy_inputs["frame0_hard_kept"] = int(frame0_keep.numel())
        self.geo_last_policy_inputs["frame0_quota_effective"] = int(frame0_quota) if frame0_idx.numel() > 0 else int(0)

        plain_visible_reserve = 0
        if max_past_tokens is not None and mode_now in {"current", "recovery"}:
            plain_visible_reserve = max(256, int(0.12 * int(max_past_tokens)))

        special_count = int(bounded_special.numel())
        frame0_count = int(frame0_keep.numel())
        remaining_geo_budget = None
        if max_past_tokens is not None:
            remaining_geo_budget = max(
                0,
                int(max_past_tokens) - special_count - frame0_count - plain_visible_reserve,
            )

        base_ref_quota = int(self.geo_reference_hard_quota)
        base_anchor_quota = int(self.geo_anchor_hard_quota)
        base_key_quota = int(self.geo_keyframe_hard_quota)
        if remaining_geo_budget is not None:
            ref_quota = min(base_ref_quota, max(0, int(0.45 * remaining_geo_budget)))
            anchor_quota = min(base_anchor_quota, max(0, int(0.35 * remaining_geo_budget)))
            key_quota = min(base_key_quota, max(0, int(0.20 * remaining_geo_budget)))
        else:
            ref_quota = base_ref_quota
            anchor_quota = base_anchor_quota
            key_quota = base_key_quota

        if mode_now in {"current", "recovery"}:
            ref_scale = float((policy or {}).get("reference_hard_scale", 1.0))
            ref_quota = max(48, int(round(float(ref_quota) * float(ref_scale))))

        ref_idx = torch.nonzero(is_reference & (~is_special), as_tuple=False).flatten()
        if ref_idx.numel() > 0 and int(ref_quota) > 0:
            ref_keep = self._take_recent_quota(ref_idx, frame_idx=frame_idx, quota=ref_quota)
            keep.update(int(v) for v in ref_keep.tolist())

        anchor_idx = torch.nonzero(is_anchor & (~is_special) & (~is_reference), as_tuple=False).flatten()
        if anchor_idx.numel() > 0 and int(anchor_quota) > 0:
            anchor_keep = self._take_recent_quota(anchor_idx, frame_idx=frame_idx, quota=anchor_quota)
            keep.update(int(v) for v in anchor_keep.tolist())

        key_idx = torch.nonzero(is_keyframe & (~is_special) & (~is_reference) & (~is_anchor), as_tuple=False).flatten()
        if key_idx.numel() > 0 and int(key_quota) > 0:
            key_keep = self._select_keyframe_tokens_stratified(meta, key_idx, quota=key_quota)
            keep.update(int(v) for v in key_keep.tolist())

        return torch.unique(torch.tensor(sorted(keep), dtype=torch.long), sorted=True)

    def _bounded_special_from_subset(
        self,
        meta: Dict[str, torch.Tensor],
        subset_idx: torch.Tensor,
        budget: int,
        recent_frames: int,
    ) -> torch.Tensor:
        if subset_idx is None or subset_idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        total = int(meta["frame_idx"].numel())
        subset = self._sanitize_keep_idx_preserve_order(
            subset_idx.detach().cpu().long(),
            meta_len=total,
            kv_len=total,
        )
        if subset.numel() == 0:
            return subset
        bounded_global = self._bounded_special_idx(
            meta,
            torch.arange(total, dtype=torch.long),
            budget=int(budget),
            recent_frames=max(1, int(recent_frames)),
        )
        keep_mask = torch.isin(bounded_global, subset)
        return bounded_global[keep_mask]

    def _cap_keep_with_hard_protection(
        self,
        *,
        meta: Dict[str, torch.Tensor],
        keep_idx: torch.Tensor,
        hard_keep: torch.Tensor,
        budget: Optional[int],
        recent_frames: int,
        priority_keep_idx: Optional[torch.Tensor] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        meta_len = int(meta["frame_idx"].numel())
        if budget is None:
            return self._sanitize_keep_idx_preserve_order(
                keep_idx,
                meta_len=meta_len,
                kv_len=meta_len,
            )
        b = max(0, int(budget))
        self.geo_last_policy_inputs["keep_plain_patch_hard_floor"] = int(0)
        self.geo_last_policy_inputs["frame0_hard_capped_diverse"] = int(0)
        self.geo_last_policy_inputs["frame0_priority_after_plain"] = bool(False)
        self.geo_last_policy_inputs["hard_cap_unique_budget"] = bool(False)
        keep = self._sanitize_keep_idx_preserve_order(
            keep_idx,
            meta_len=meta_len,
            kv_len=meta_len,
        )
        hard = self._sanitize_keep_idx_preserve_order(
            hard_keep.detach().cpu().long() if hard_keep is not None else torch.empty((0,), dtype=torch.long),
            meta_len=meta_len,
            kv_len=meta_len,
        )
        if keep.numel() <= b:
            return keep
        if hard.numel() >= b:
            plain_floor_idx = torch.empty((0,), dtype=torch.long)
            if priority_keep_idx is not None and policy is not None:
                mode_now = str(policy.get("mode", "legacy"))
                if mode_now in {"current", "recovery"}:
                    priority = self._sanitize_keep_idx_preserve_order(
                        priority_keep_idx.detach().cpu().long(),
                        meta_len=meta_len,
                        kv_len=meta_len,
                    )
                    geo_role_all = meta.get("geo_role", self._compute_primary_geo_role(meta))
                    is_special_all = meta.get("is_special", torch.zeros((meta_len,), dtype=torch.bool))
                    plain_mask = (~is_special_all.index_select(0, priority)) & (geo_role_all.index_select(0, priority) == 0)
                    plain_floor_idx = priority[plain_mask & (~torch.isin(priority, hard))]
            return self._cap_hard_backbone_only(
                meta=meta,
                hard_idx=hard,
                budget=b,
                plain_floor_idx=plain_floor_idx,
                policy=policy,
            )

        selected = set(int(v) for v in hard.tolist())
        take_n = max(0, b - len(selected))
        if take_n <= 0:
            out = hard
        else:
            if priority_keep_idx is not None and priority_keep_idx.numel() > 0:
                priority = self._sanitize_keep_idx_preserve_order(
                    priority_keep_idx.detach().cpu().long(),
                    meta_len=meta_len,
                    kv_len=meta_len,
                )
                soft_order = priority[
                    torch.isin(priority, keep) & (~torch.isin(priority, hard))
                ]
                pick_soft = soft_order[:take_n]
            else:
                remain = keep[~torch.isin(keep, hard)]
                frame_idx = meta["frame_idx"]
                if remain.numel() > 0:
                    remain = remain.index_select(
                        0,
                        torch.argsort(
                            frame_idx.index_select(0, remain),
                            descending=True,
                            stable=True,
                        ),
                    )
                pick_soft = remain[:take_n]
            out = self._unique_preserve_order_long(torch.cat([hard, pick_soft], dim=0))
        if out.numel() > b:
            out = self._cap_hard_backbone_only(
                meta=meta,
                hard_idx=out,
                budget=b,
                policy=policy,
            )
        return out

    def _cap_frame0_hard_subset(
        self,
        meta: Dict[str, torch.Tensor],
        frame0_idx: torch.Tensor,
        quota: int,
    ) -> torch.Tensor:
        if frame0_idx is None or frame0_idx.numel() == 0 or int(quota) <= 0:
            return torch.empty((0,), dtype=torch.long)

        frame0_idx = frame0_idx.detach().cpu().long()
        if frame0_idx.numel() <= int(quota):
            return frame0_idx

        local_idx_all = meta.get(
            "local_patch_idx",
            torch.full((int(meta["frame_idx"].numel()),), -1, dtype=torch.long),
        )
        frame0_meta = self.geo_frame_meta.get(0)

        if (
            frame0_meta is not None
            and frame0_meta.get("conf") is not None
            and frame0_meta["conf"].numel() > 0
        ):
            frame0_local = local_idx_all.index_select(0, frame0_idx).long()
            valid = (frame0_local >= 0) & (frame0_local < frame0_meta["conf"].shape[0])
            frame0_idx_valid = frame0_idx[valid]
            frame0_local_valid = frame0_local[valid]
            if frame0_idx_valid.numel() > 0:
                conf0 = frame0_meta["conf"].index_select(0, frame0_local_valid).to(torch.float32)
                return self._select_frame0_patch_diverse(
                    frame0_idx_valid,
                    frame0_local_valid,
                    conf0,
                    quota=int(quota),
                    full_patch_count=int(frame0_meta["conf"].shape[0]),
                    grid_n=4,
                )

        return frame0_idx[: int(quota)]

    def _cap_hard_backbone_only(
        self,
        *,
        meta: Dict[str, torch.Tensor],
        hard_idx: torch.Tensor,
        budget: int,
        plain_floor_idx: Optional[torch.Tensor] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Cap already-hard-protected indices using strict backbone priority only.
        Priority order:
          - current/recovery: special -> plain_floor -> frame0 -> reference -> anchor -> keyframe
          - legacy/other:    special -> frame0 -> plain_floor -> reference -> anchor -> keyframe
        """
        b = max(0, int(budget))
        if b <= 0:
            return torch.empty((0,), dtype=torch.long)

        total = int(meta["frame_idx"].numel())
        hard = self._sanitize_keep_idx(
            hard_idx.detach().cpu().long(),
            meta_len=total,
            kv_len=total,
        )
        if hard.numel() <= b:
            return hard

        frame_idx_all = meta["frame_idx"]
        geo_role_all = meta.get("geo_role", self._compute_primary_geo_role(meta))
        is_special_all = meta.get(
            "is_special", torch.zeros_like(frame_idx_all, dtype=torch.bool)
        )
        local_idx_all = meta.get(
            "local_patch_idx", torch.full_like(frame_idx_all, -1, dtype=torch.long)
        )

        frame_keep = frame_idx_all.index_select(0, hard)
        role_keep = geo_role_all.index_select(0, hard)
        special_keep = is_special_all.index_select(0, hard)
        local_keep = local_idx_all.index_select(0, hard)

        frame0_keep = (frame_keep == 0) & (~special_keep) & (local_keep >= 0)
        ref_keep = role_keep == 4
        anchor_keep = role_keep == 3
        key_keep = role_keep == 1

        part_special = self._bounded_special_from_subset(
            meta,
            hard[special_keep],
            budget=b,
            recent_frames=max(2, int(self.geo_legacy_recent_window)),
        )
        part_frame0 = hard[frame0_keep]
        part_ref = hard[ref_keep]
        part_anchor = hard[anchor_keep]
        part_key = hard[key_keep]

        mode_now = str((policy or {}).get("mode", "legacy"))
        plain_floor = torch.empty((0,), dtype=torch.long)
        if plain_floor_idx is not None and plain_floor_idx.numel() > 0 and mode_now in {"current", "recovery"}:
            plain_floor_quota = max(64, int(0.08 * int(b)))
            plain_floor = plain_floor_idx[: min(int(plain_floor_idx.numel()), int(plain_floor_quota))]
        self.geo_last_policy_inputs["keep_plain_patch_hard_floor"] = int(plain_floor.numel())

        def _take_recent_quota(idx_tensor: torch.Tensor, quota: int) -> torch.Tensor:
            if quota <= 0 or idx_tensor.numel() == 0:
                return torch.empty((0,), dtype=torch.long)
            if idx_tensor.numel() <= int(quota):
                return idx_tensor
            frame_take = frame_idx_all.index_select(0, idx_tensor)
            order = torch.argsort(frame_take, descending=True, stable=True)
            return idx_tensor.index_select(0, order[: int(quota)])

        def _exclude_picked(idx_tensor: torch.Tensor, picked_ids: torch.Tensor) -> torch.Tensor:
            if idx_tensor is None or idx_tensor.numel() == 0:
                return torch.empty((0,), dtype=torch.long)
            if picked_ids is None or picked_ids.numel() == 0:
                return idx_tensor
            return idx_tensor[~torch.isin(idx_tensor, picked_ids)]

        remaining = b
        picked_parts: List[torch.Tensor] = []
        picked_ids = torch.empty((0,), dtype=torch.long)

        chosen_special = _take_recent_quota(part_special, remaining)
        if chosen_special.numel() > 0:
            picked_parts.append(chosen_special)
            picked_ids = self._unique_preserve_order_long(torch.cat([picked_ids, chosen_special], dim=0))
            remaining -= int(chosen_special.numel())

        self.geo_last_policy_inputs["frame0_hard_capped_diverse"] = int(0)
        frame0_priority_after_plain = bool(mode_now in {"current", "recovery"})
        self.geo_last_policy_inputs["frame0_priority_after_plain"] = bool(frame0_priority_after_plain)
        self.geo_last_policy_inputs["current_recovery_ref_before_frame0"] = bool(frame0_priority_after_plain)

        if frame0_priority_after_plain:
            plain_avail = _exclude_picked(plain_floor, picked_ids)
            if remaining > 0 and plain_avail.numel() > 0:
                chosen_plain = plain_avail[:remaining]
                if chosen_plain.numel() > 0:
                    picked_parts.append(chosen_plain)
                    picked_ids = self._unique_preserve_order_long(torch.cat([picked_ids, chosen_plain], dim=0))
                    remaining -= int(chosen_plain.numel())
            ref_avail = _exclude_picked(part_ref, picked_ids)
            if remaining > 0 and ref_avail.numel() > 0:
                chosen_ref = _take_recent_quota(ref_avail, remaining)
                if chosen_ref.numel() > 0:
                    picked_parts.append(chosen_ref)
                    picked_ids = self._unique_preserve_order_long(torch.cat([picked_ids, chosen_ref], dim=0))
                    remaining -= int(chosen_ref.numel())
            frame0_avail = _exclude_picked(part_frame0, picked_ids)
            if remaining > 0 and frame0_avail.numel() > 0:
                chosen_frame0 = self._cap_frame0_hard_subset(meta, frame0_avail, quota=remaining)
                if chosen_frame0.numel() > 0:
                    picked_parts.append(chosen_frame0)
                    picked_ids = self._unique_preserve_order_long(torch.cat([picked_ids, chosen_frame0], dim=0))
                    remaining -= int(chosen_frame0.numel())
                    self.geo_last_policy_inputs["frame0_hard_capped_diverse"] = int(chosen_frame0.numel())
            trailing_parts = [part_anchor, part_key]
        else:
            frame0_avail = _exclude_picked(part_frame0, picked_ids)
            if remaining > 0 and frame0_avail.numel() > 0:
                chosen_frame0 = self._cap_frame0_hard_subset(meta, frame0_avail, quota=remaining)
                if chosen_frame0.numel() > 0:
                    picked_parts.append(chosen_frame0)
                    picked_ids = self._unique_preserve_order_long(torch.cat([picked_ids, chosen_frame0], dim=0))
                    remaining -= int(chosen_frame0.numel())
                    self.geo_last_policy_inputs["frame0_hard_capped_diverse"] = int(chosen_frame0.numel())
            plain_avail = _exclude_picked(plain_floor, picked_ids)
            if remaining > 0 and plain_avail.numel() > 0:
                chosen_plain = plain_avail[:remaining]
                if chosen_plain.numel() > 0:
                    picked_parts.append(chosen_plain)
                    picked_ids = self._unique_preserve_order_long(torch.cat([picked_ids, chosen_plain], dim=0))
                    remaining -= int(chosen_plain.numel())
            trailing_parts = [part_ref, part_anchor, part_key]

        for part in trailing_parts:
            if remaining <= 0:
                break
            part_avail = _exclude_picked(part, picked_ids)
            chosen = _take_recent_quota(part_avail, remaining)
            if chosen.numel() > 0:
                picked_parts.append(chosen)
                picked_ids = self._unique_preserve_order_long(torch.cat([picked_ids, chosen], dim=0))
                remaining -= int(chosen.numel())

        self.geo_last_policy_inputs["hard_cap_unique_budget"] = bool(True)
        if not picked_parts:
            return torch.empty((0,), dtype=torch.long)
        return self._unique_preserve_order_long(torch.cat(picked_parts, dim=0))

    def _cap_keep_for_geo_mode(
        self,
        *,
        meta: Dict[str, torch.Tensor],
        keep_idx: torch.Tensor,
        budget: int,
        recent_frames: int,
        current_frame_idx: int,
        policy: Optional[Dict[str, Any]],
        priority_keep_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        End-to-end geo-mode cap:
        1) sanitize keep_idx
        2) rebuild hard backbone on current meta
        3) union keep_idx with hard backbone
        4) apply hard-backbone protection cap
        """
        b = max(0, int(budget))
        if b <= 0:
            return torch.empty((0,), dtype=torch.long)

        total = int(meta["frame_idx"].numel())
        keep = self._sanitize_keep_idx_preserve_order(
            keep_idx.detach().cpu().long(),
            meta_len=total,
            kv_len=total,
        )
        if keep.numel() == 0:
            return keep

        hard_keep = self._build_hard_backbone_keep(
            meta,
            current_frame_idx=int(current_frame_idx),
            max_past_tokens=int(b),
            policy=policy,
        )

        keep_all = self._unique_preserve_order_long(torch.cat([keep, hard_keep], dim=0))
        if priority_keep_idx is None:
            priority_keep_idx = keep
        out = self._cap_keep_with_hard_protection(
            meta=meta,
            keep_idx=keep_all,
            hard_keep=hard_keep,
            budget=int(b),
            recent_frames=int(recent_frames),
            priority_keep_idx=priority_keep_idx,
            policy=policy,
        )

        return self._sanitize_keep_idx_preserve_order(
            out,
            meta_len=total,
            kv_len=total,
        )

    def _select_geo_active_indices(
        self,
        meta: Dict[str, torch.Tensor],
        topk_per_voxel: int,
        recent_frames: int,
        near: float,
        far: float,
        current_view: Optional[Dict[str, torch.Tensor]],
        trigger_view: Optional[Dict[str, Any]] = None,
        use_view_pruning: bool = True,
        max_past_tokens: Optional[int] = None,
        enable_reference_logic: bool = True,
        enable_landmark_logic: bool = True,
        enable_stable_logic: bool = True,
        enable_reanchor_logic: bool = True,
        policy: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        total_tokens = int(meta["frame_idx"].numel())
        if total_tokens == 0:
            return None

        frame_idx = meta["frame_idx"]
        meta["geo_role"] = self._compute_primary_geo_role(meta)
        current_frame_idx = int(frame_idx.max().item()) if frame_idx.numel() > 0 else 0
        structure_ready = self._geo_structure_ready()
        recent_frames_eff = int(policy["recent_window"]) if policy is not None else int(recent_frames)
        hard_recent_frames_eff = int(policy["hard_recent_frames"]) if policy is not None else int(self.geo_hard_recent_frames)
        if max_past_tokens is not None and total_tokens <= int(float(max_past_tokens) * float(self.geo_prune_start_ratio)):
            keep_all = torch.arange(total_tokens, dtype=torch.long)
            logger.info(
                "[geo_keep] total_tokens=%d budget=%d selected=%d selected_ratio=%.4f skip_prune=1",
                int(total_tokens),
                int(max_past_tokens),
                int(keep_all.numel()),
                float(keep_all.numel()) / float(max(1, total_tokens)),
            )
            proxy = self._geo_proxy_selector_diag(
                current_frame_idx=current_frame_idx,
                selected_total=int(keep_all.numel()),
            )
            selected_fast = set(int(v) for v in keep_all.tolist())
            selected_fast_order = [int(v) for v in keep_all.tolist()]
            fast_plain_reserved = self._augment_fastpath_with_recent_plain_floor(
                meta=meta,
                selected_set=selected_fast,
                selected_order=selected_fast_order,
                current_frame_idx=current_frame_idx,
                recent_frames_eff=recent_frames_eff,
                max_past_tokens=max_past_tokens,
                policy=policy,
            )
            priority_for_cap_fast = self._unique_preserve_order_long(
                torch.cat(
                    [
                        fast_plain_reserved,
                        torch.tensor(selected_fast_order, dtype=torch.long),
                    ],
                    dim=0,
                )
            )
            self.geo_last_policy_inputs["priority_keep_fastpath_has_plain_floor"] = bool(fast_plain_reserved.numel() > 0)
            return self._finalize_geo_keep(
                meta=meta,
                selected=selected_fast,
                selected_order=selected_fast_order,
                current_frame_idx=current_frame_idx,
                recent_frames_eff=recent_frames_eff,
                max_past_tokens=max_past_tokens,
                candidate_count=0,
                visible_total=int(proxy["visible_total"]),
                anchor_count=int(meta.get("is_anchor", torch.zeros_like(meta["is_special"])).sum().item()),
                stable_count=0,
                tau_bucket=float("nan"),
                stable_visible_voxel_overlap=int(proxy["stable_visible_overlap"]),
                stable_selected_visible=int(proxy["stable_selected_visible"]),
                stable_selected_invisible=int(proxy["stable_selected_invisible"]),
                fast_path=2,
                reanchor_added=0,
                reanchor_overlap_avg=0.0,
                policy=policy,
                allow_fill=False,
                update_selector_diag=False,
                plain_reserved_idx=fast_plain_reserved,
                priority_keep_idx=priority_for_cap_fast,
            )

        hard_keep = self._build_hard_backbone_keep(
            meta,
            current_frame_idx=current_frame_idx,
            max_past_tokens=max_past_tokens,
            policy=policy,
        )
        selected = set(int(v) for v in hard_keep.tolist())
        selected_order: List[int] = [int(v) for v in hard_keep.tolist()]
        is_special = meta["is_special"]
        is_keyframe = meta.get("is_keyframe", torch.zeros_like(is_special))
        local_idx = meta["local_patch_idx"]
        geo_role = meta.get("geo_role", self._compute_primary_geo_role(meta))
        is_anchor = geo_role == 3
        is_landmark = geo_role == 2
        is_reference = geo_role == 4

        mode_now = str((policy or {}).get("mode", "legacy"))

        # In current/recovery, do NOT re-add all special tokens here.
        # bounded special has already been handled in _build_hard_backbone_keep().
        if mode_now == "legacy":
            special_idx = torch.nonzero(is_special, as_tuple=False).flatten().tolist()
            self._ordered_add(selected, selected_order, special_idx)

        # NOTE: do not unconditionally keep all anchor/reference/landmark tokens.
        # They are selected later with visibility/score-aware bounded quotas.

        # Frame0: keep special tokens always, patch tokens by a fixed cap.
        frame0_mask = frame_idx == 0
        frame0_special_idx = torch.nonzero(frame0_mask & is_special, as_tuple=False).flatten().tolist()
        self._ordered_add(selected, selected_order, frame0_special_idx)

        extra_frame0_soft_promotion_enabled = bool(mode_now == "legacy")
        self.geo_last_policy_inputs["extra_frame0_soft_promotion_enabled"] = bool(extra_frame0_soft_promotion_enabled)

        frame0_patch_idx = torch.nonzero(frame0_mask & (~is_special) & (local_idx >= 0), as_tuple=False).flatten()
        if frame0_patch_idx.numel() > 0 and extra_frame0_soft_promotion_enabled:
            frame0_local = local_idx[frame0_patch_idx].long()
            # try conf-guided selection if metadata for frame0 exists
            frame0_meta = self.geo_frame_meta.get(0)
            if frame0_meta is not None and frame0_meta["conf"].numel() > 0:
                in_range = (frame0_local >= 0) & (frame0_local < frame0_meta["conf"].shape[0])
                frame0_patch_idx = frame0_patch_idx[in_range]
                frame0_local = frame0_local[in_range]
                if frame0_patch_idx.numel() > 0:
                    conf0 = frame0_meta["conf"].index_select(0, frame0_local).to(torch.float32)
                    k0 = min(int((policy or {}).get("frame0_patch_cap", self.geo_frame0_patch_cap)), int(frame0_patch_idx.numel()))
                    frame0_keep = self._select_frame0_patch_diverse(
                        frame0_patch_idx,
                        frame0_local,
                        conf0,
                        quota=int(k0),
                        full_patch_count=int(frame0_meta["conf"].shape[0]),
                        grid_n=4,
                    )
                    self._ordered_add(selected, selected_order, frame0_keep)
            else:
                # No confidence metadata: do not use positional slicing for frame-0 patches.
                pass

        # Reserve sparse keyframe tokens to preserve long-horizon constraints.
        keyframe_patch_idx = torch.nonzero(is_keyframe & (~is_special) & (~frame0_mask) & (local_idx >= 0), as_tuple=False).flatten()
        if keyframe_patch_idx.numel() > 0 and int(self.geo_keyframe_token_quota) > 0:
            keyframe_keep = self._select_keyframe_tokens_stratified(
                meta,
                keyframe_patch_idx,
                quota=int(self.geo_keyframe_token_quota),
            )
            self._ordered_add(selected, selected_order, keyframe_keep)

        recent_min = max(0, current_frame_idx - recent_frames_eff)
        recent_mask = frame_idx >= recent_min
        hard_recent_idx = self._hard_recent_patch_idx(meta, hard_recent_frames_eff)

        view_for_trigger = trigger_view if trigger_view is not None else current_view
        force_full_select = (not structure_ready) or (bool(use_view_pruning) and self._should_force_full_geo_selection(current_frame_idx, view_for_trigger))

        # Optional fast-path (event/interval gated): only run full geo selection on key/unstable frames.
        if structure_ready and self.geo_selection_interval > 1 and (not force_full_select) and (current_frame_idx % self.geo_selection_interval != 0):
            recent_idx = torch.nonzero(recent_mask, as_tuple=False).flatten().tolist()
            self._ordered_add(selected, selected_order, recent_idx)
            selected_fast = set(selected)
            selected_fast_order = list(selected_order)
            self._ordered_add(
                selected_fast,
                selected_fast_order,
                self._seed_geo_backbone_keep(
                    meta=meta,
                    current_frame_idx=current_frame_idx,
                    max_past_tokens=max_past_tokens,
                    recent_frames_eff=recent_frames_eff,
                    policy=policy,
                ),
            )
            fast_plain_reserved = self._augment_fastpath_with_recent_plain_floor(
                meta=meta,
                selected_set=selected_fast,
                selected_order=selected_fast_order,
                current_frame_idx=current_frame_idx,
                recent_frames_eff=recent_frames_eff,
                max_past_tokens=max_past_tokens,
                policy=policy,
            )
            priority_for_cap_fast = self._unique_preserve_order_long(
                torch.cat(
                    [
                        hard_keep.detach().cpu().long() if hard_keep is not None else torch.empty((0,), dtype=torch.long),
                        fast_plain_reserved,
                        torch.tensor(selected_fast_order, dtype=torch.long),
                    ],
                    dim=0,
                )
            )
            self.geo_last_policy_inputs["priority_keep_fastpath_has_plain_floor"] = bool(fast_plain_reserved.numel() > 0)
            proxy = self._geo_proxy_selector_diag(
                current_frame_idx=current_frame_idx,
                selected_total=len(selected_fast),
            )
            return self._finalize_geo_keep(
                meta=meta,
                selected=selected_fast,
                selected_order=selected_fast_order,
                current_frame_idx=current_frame_idx,
                recent_frames_eff=recent_frames_eff,
                max_past_tokens=max_past_tokens,
                candidate_count=0,
                visible_total=int(proxy["visible_total"]),
                anchor_count=int(meta.get("is_anchor", torch.zeros_like(meta["is_special"])).sum().item()),
                stable_count=0,
                tau_bucket=float("nan"),
                stable_visible_voxel_overlap=int(proxy["stable_visible_overlap"]),
                stable_selected_visible=int(proxy["stable_selected_visible"]),
                stable_selected_invisible=int(proxy["stable_selected_invisible"]),
                fast_path=1,
                hard_keep_idx=hard_keep,
                reanchor_added=0,
                reanchor_overlap_avg=0.0,
                policy=policy,
                allow_fill=False,
                update_selector_diag=False,
                plain_reserved_idx=fast_plain_reserved,
                priority_keep_idx=priority_for_cap_fast,
            )

        # Build a budgeted local-tracking pool for recent patches (not all recent patches).
        if max_past_tokens is not None:
            recent_frames_count = max(1, int(torch.unique(frame_idx[recent_mask]).numel()))
            local_ratio = float(policy["local_budget_ratio"]) if policy is not None else float(self.geo_local_budget_ratio)
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
        self._ordered_add(selected, selected_order, recent_special_idx)

        if hard_recent_idx.numel() > 0:
            self._ordered_add(selected, selected_order, hard_recent_idx)

        soft_recent_mask = recent_mask.clone()
        if hard_recent_idx.numel() > 0:
            hard_recent_flag = torch.zeros((total_tokens,), dtype=torch.bool)
            hard_recent_flag.index_fill_(0, hard_recent_idx, True)
            soft_recent_mask = recent_mask & (~hard_recent_flag)

        recent_patch_indices = torch.nonzero(soft_recent_mask & (~is_special) & (local_idx >= 0), as_tuple=False).flatten()
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
            self._ordered_add(selected, selected_order, local_u)

        if (not use_view_pruning) or current_view is None or current_view.get("world_to_cam") is None or current_view.get("intrinsic") is None:
            selected_fast = set(selected)
            selected_fast_order = list(selected_order)
            self._ordered_add(
                selected_fast,
                selected_fast_order,
                self._seed_geo_backbone_keep(
                    meta=meta,
                    current_frame_idx=current_frame_idx,
                    max_past_tokens=max_past_tokens,
                    recent_frames_eff=recent_frames_eff,
                    policy=policy,
                ),
            )
            fast_plain_reserved = self._augment_fastpath_with_recent_plain_floor(
                meta=meta,
                selected_set=selected_fast,
                selected_order=selected_fast_order,
                current_frame_idx=current_frame_idx,
                recent_frames_eff=recent_frames_eff,
                max_past_tokens=max_past_tokens,
                policy=policy,
            )
            priority_for_cap_fast = self._unique_preserve_order_long(
                torch.cat(
                    [
                        hard_keep.detach().cpu().long() if hard_keep is not None else torch.empty((0,), dtype=torch.long),
                        fast_plain_reserved,
                        torch.tensor(selected_fast_order, dtype=torch.long),
                    ],
                    dim=0,
                )
            )
            self.geo_last_policy_inputs["priority_keep_fastpath_has_plain_floor"] = bool(fast_plain_reserved.numel() > 0)
            proxy = self._geo_proxy_selector_diag(
                current_frame_idx=current_frame_idx,
                selected_total=len(selected_fast),
            )
            return self._finalize_geo_keep(
                meta=meta,
                selected=selected_fast,
                selected_order=selected_fast_order,
                current_frame_idx=current_frame_idx,
                recent_frames_eff=recent_frames_eff,
                max_past_tokens=max_past_tokens,
                candidate_count=0,
                visible_total=int(proxy["visible_total"]),
                anchor_count=0,
                stable_count=0,
                tau_bucket=float("nan"),
                stable_visible_voxel_overlap=int(proxy["stable_visible_overlap"]),
                stable_selected_visible=int(proxy["stable_selected_visible"]),
                stable_selected_invisible=int(proxy["stable_selected_invisible"]),
                fast_path=3,
                hard_keep_idx=hard_keep,
                reanchor_added=0,
                reanchor_overlap_avg=0.0,
                policy=policy,
                allow_fill=False,
                update_selector_diag=False,
                plain_reserved_idx=fast_plain_reserved,
                priority_keep_idx=priority_for_cap_fast,
            )

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
            selected_fast = set(selected)
            selected_fast_order = list(selected_order)
            self._ordered_add(
                selected_fast,
                selected_fast_order,
                self._seed_geo_backbone_keep(
                    meta=meta,
                    current_frame_idx=current_frame_idx,
                    max_past_tokens=max_past_tokens,
                    recent_frames_eff=recent_frames_eff,
                    policy=policy,
                ),
            )
            fast_plain_reserved = self._augment_fastpath_with_recent_plain_floor(
                meta=meta,
                selected_set=selected_fast,
                selected_order=selected_fast_order,
                current_frame_idx=current_frame_idx,
                recent_frames_eff=recent_frames_eff,
                max_past_tokens=max_past_tokens,
                policy=policy,
            )
            priority_for_cap_fast = self._unique_preserve_order_long(
                torch.cat(
                    [
                        hard_keep.detach().cpu().long() if hard_keep is not None else torch.empty((0,), dtype=torch.long),
                        fast_plain_reserved,
                        torch.tensor(selected_fast_order, dtype=torch.long),
                    ],
                    dim=0,
                )
            )
            self.geo_last_policy_inputs["priority_keep_fastpath_has_plain_floor"] = bool(fast_plain_reserved.numel() > 0)
            proxy = self._geo_proxy_selector_diag(
                current_frame_idx=current_frame_idx,
                selected_total=len(selected_fast),
            )
            return self._finalize_geo_keep(
                meta=meta,
                selected=selected_fast,
                selected_order=selected_fast_order,
                current_frame_idx=current_frame_idx,
                recent_frames_eff=recent_frames_eff,
                max_past_tokens=max_past_tokens,
                candidate_count=0,
                visible_total=int(proxy["visible_total"]),
                anchor_count=0,
                stable_count=0,
                tau_bucket=float("nan"),
                stable_visible_voxel_overlap=int(proxy["stable_visible_overlap"]),
                stable_selected_visible=int(proxy["stable_selected_visible"]),
                stable_selected_invisible=int(proxy["stable_selected_invisible"]),
                fast_path=6,
                hard_keep_idx=hard_keep,
                reanchor_added=0,
                reanchor_overlap_avg=0.0,
                policy=policy,
                allow_fill=False,
                update_selector_diag=False,
                plain_reserved_idx=fast_plain_reserved,
                priority_keep_idx=priority_for_cap_fast,
            )

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
            if enable_reference_logic and ref_vox_t.any() and float(self.geo_reference_overlap_bonus) > 0.0:
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
            selected_fast = set(selected)
            selected_fast_order = list(selected_order)
            self._ordered_add(
                selected_fast,
                selected_fast_order,
                self._seed_geo_backbone_keep(
                    meta=meta,
                    current_frame_idx=current_frame_idx,
                    max_past_tokens=max_past_tokens,
                    recent_frames_eff=recent_frames_eff,
                    policy=policy,
                ),
            )
            fast_plain_reserved = self._augment_fastpath_with_recent_plain_floor(
                meta=meta,
                selected_set=selected_fast,
                selected_order=selected_fast_order,
                current_frame_idx=current_frame_idx,
                recent_frames_eff=recent_frames_eff,
                max_past_tokens=max_past_tokens,
                policy=policy,
            )
            priority_for_cap_fast = self._unique_preserve_order_long(
                torch.cat(
                    [
                        hard_keep.detach().cpu().long() if hard_keep is not None else torch.empty((0,), dtype=torch.long),
                        fast_plain_reserved,
                        torch.tensor(selected_fast_order, dtype=torch.long),
                    ],
                    dim=0,
                )
            )
            self.geo_last_policy_inputs["priority_keep_fastpath_has_plain_floor"] = bool(fast_plain_reserved.numel() > 0)
            proxy = self._geo_proxy_selector_diag(
                current_frame_idx=current_frame_idx,
                selected_total=len(selected_fast),
            )
            return self._finalize_geo_keep(
                meta=meta,
                selected=selected_fast,
                selected_order=selected_fast_order,
                current_frame_idx=current_frame_idx,
                recent_frames_eff=recent_frames_eff,
                max_past_tokens=max_past_tokens,
                candidate_count=int(candidate_count),
                visible_total=int(proxy["visible_total"]),
                anchor_count=0,
                stable_count=0,
                tau_bucket=float("nan"),
                stable_visible_voxel_overlap=int(proxy["stable_visible_overlap"]),
                stable_selected_visible=int(proxy["stable_selected_visible"]),
                stable_selected_invisible=int(proxy["stable_selected_invisible"]),
                fast_path=7,
                hard_keep_idx=hard_keep,
                reanchor_added=0,
                reanchor_overlap_avg=0.0,
                policy=policy,
                allow_fill=False,
                update_selector_diag=False,
                plain_reserved_idx=fast_plain_reserved,
                priority_keep_idx=priority_for_cap_fast,
            )

        idx_all = torch.cat(gather_idx, dim=0)
        score_all = torch.cat(gather_score, dim=0)
        bank_conf_all = torch.cat(gather_bank_conf, dim=0)
        hash_all = torch.cat(gather_voxel_hash, dim=0)
        visible_all = torch.cat(gather_visible, dim=0)

        assert int(idx_all.numel()) == int(score_all.numel()), (
            f"idx_all/score_all mismatch: {idx_all.numel()} vs {score_all.numel()}"
        )
        assert int(idx_all.numel()) == int(visible_all.numel()), (
            f"idx_all/visible_all mismatch: {idx_all.numel()} vs {visible_all.numel()}"
        )
        assert int(idx_all.numel()) == int(bank_conf_all.numel()), (
            f"idx_all/bank_conf_all mismatch: {idx_all.numel()} vs {bank_conf_all.numel()}"
        )

        # Adaptive bucket threshold to control bucket size.
        remaining_budget = None
        if max_past_tokens is not None:
            remaining_budget = max(0, int(max_past_tokens) - len(selected))
        tau_bucket = self._compute_dynamic_bucket_threshold(bank_conf_all.tolist(), remaining_budget or 0)

        # Global anchor quota from ordered anchor list (deterministic).
        anchor_count = 0
        selected_global: set[int] = set(selected)
        if self.geo_anchor_voxel_list:
            anchor_quota = int(self.geo_anchor_read_quota)
            if max_past_tokens is not None:
                anchor_ratio = float(policy["anchor_quota_ratio"]) if policy is not None else float(self.geo_anchor_budget_ratio)
                anchor_quota = min(anchor_quota, max(0, int(max_past_tokens * anchor_ratio)))

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
                self._ordered_add(selected, selected_order, [token])
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
        stable_hash_for_diag = torch.empty((0,), dtype=torch.long)
        if enable_stable_logic and self.geo_recovery_frames_left > 0 and self.geo_reference_voxels:
            overlap_ref = self.geo_stable_map_voxels.intersection(self.geo_reference_voxels)
            stable_source_voxels = overlap_ref if overlap_ref else self.geo_reference_voxels
        if enable_stable_logic and stable_source_voxels:
            stable_hash = self._voxel_hash(torch.tensor(sorted(stable_source_voxels), dtype=torch.long))
            stable_hash_for_diag = stable_hash.detach().cpu()
            stable_mask = torch.isin(hash_all, stable_hash)
            vis_stable_mask = stable_mask & visible_all
            if vis_stable_mask.any():
                stable_visible_voxel_overlap = int(torch.unique(hash_all[vis_stable_mask]).numel())
            if stable_mask.any():
                stable_quota = 0
                if max_past_tokens is not None:
                    stable_ratio = float(policy["stable_read_budget_ratio"]) if policy is not None else float(self.geo_stable_read_budget_ratio)
                    stable_quota = max(0, int(max_past_tokens * stable_ratio))
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
                            self._ordered_add(selected, selected_order, [token])
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
                                self._ordered_add(selected, selected_order, [token])
                                selected_global.add(token)
                                stable_selected_tokens.append(token)
                                stable_count += 1
                                stable_visible_selected += 1
                                if stable_count >= stable_quota_eff or stable_visible_selected >= vis_quota:
                                    break

                    # Overlap low: use retrieval-like keyframe tokens to fill remaining long-horizon constraints.
                    stable_deficit = max(0, int(stable_quota - stable_count))
                    if (
                        enable_reanchor_logic
                        and stable_deficit > 0
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
                                        self._ordered_add(selected, selected_order, [token])
                                        selected_global.add(token)
                                        stable_selected_tokens.append(token)
                                        reanchor_added += 1
                                        picked += 1
                                        if picked >= per_frame_q:
                                            break

        stable_total_selected = int(stable_visible_selected + stable_invisible_selected)
        stable_visible_ratio = float(stable_visible_selected) / float(max(1, stable_total_selected))
        bad_mode = False
        if enable_stable_logic:
            bad_stable_quality = (
                stable_total_selected == 0
                or stable_visible_ratio < float(self.geo_stable_quality_visible_ratio_thr)
                or int(stable_visible_voxel_overlap) < int(self.geo_stable_quality_overlap_thr)
                or int(visible_total) == 0
            )
            reloc_trigger = (
                stable_total_selected == 0
                or int(visible_total) == 0
                or int(stable_visible_voxel_overlap) < int(self.geo_reloc_trigger_overlap)
                or float(stable_visible_ratio) < float(self.geo_reloc_trigger_visible_ratio)
                or float(self.geo_trust_score) < float(self.geo_selection_low_trust_threshold)
            )
            bad_mode = bool(bad_stable_quality or reloc_trigger)

        if bad_mode:
            valid_global = visible_all & (bank_conf_all >= float(tau_bucket))
            idx_valid = idx_all[valid_global]
            score_valid = score_all[valid_global]
            hash_valid = hash_all[valid_global]
            tiny_invis_quota = 96
            ref_mask_all = (meta["geo_role"].index_select(0, idx_all) == 4) if enable_reference_logic else torch.zeros_like(visible_all)
            invis_ref_mask = (~visible_all) & ref_mask_all & (bank_conf_all >= float(tau_bucket))
            if enable_reference_logic and invis_ref_mask.any() and tiny_invis_quota > 0:
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

        recent_plain_floor_kept = torch.empty((0,), dtype=torch.long)
        recent_plain_floor_count = int(recent_plain_floor_kept.numel())
        if mode_now in {"current", "recovery"} and max_past_tokens is not None:
            recent_plain_idx = torch.nonzero(
                recent_mask
                & (~is_special)
                & (geo_role == 0)
                & (local_idx >= 0),
                as_tuple=False,
            ).flatten()
            if recent_plain_idx.numel() > 0:
                obs_stress = float((policy or {}).get("observation_stress", 0.0))
                recent_plain_ratio = 0.06 + 0.04 * float(obs_stress)
                self.geo_last_policy_inputs["recent_plain_ratio_effective"] = float(recent_plain_ratio)
                recent_plain_quota = max(192, int(recent_plain_ratio * int(max_past_tokens)))
                recent_plain_frame = frame_idx.index_select(0, recent_plain_idx)
                uniq_recent_frames = torch.unique(recent_plain_frame).sort(descending=True).values
                per_frame_quota = max(
                    16,
                    int(math.ceil(float(recent_plain_quota) / float(max(1, uniq_recent_frames.numel())))),
                )
                picked_parts: List[torch.Tensor] = []
                for f in uniq_recent_frames.tolist():
                    idx_f = recent_plain_idx[recent_plain_frame == int(f)]
                    if idx_f.numel() == 0:
                        continue
                    local_f = local_idx.index_select(0, idx_f).long()
                    fm = self.geo_frame_meta.get(int(f))
                    if fm is not None and fm.get("conf") is not None and fm["conf"].numel() > 0:
                        valid = (local_f >= 0) & (local_f < fm["conf"].shape[0])
                        idx_f = idx_f[valid]
                        local_f = local_f[valid]
                        if idx_f.numel() == 0:
                            continue
                        conf_f = fm["conf"].index_select(0, local_f).to(torch.float32)
                        kf = min(int(per_frame_quota), int(idx_f.numel()))
                        picked_parts.append(
                            self._select_patch_diverse(
                                idx_f,
                                local_f,
                                conf_f,
                                quota=int(kf),
                                full_patch_count=int(fm["conf"].shape[0]),
                                grid_n=max(2, int(self.geo_local_coverage_grid)),
                            )
                        )
                    else:
                        kf = min(int(per_frame_quota), int(idx_f.numel()))
                        picked_parts.append(idx_f[:kf])
                if picked_parts:
                    recent_plain_floor_kept = self._unique_preserve_order_long(torch.cat(picked_parts, dim=0)).detach().cpu().long()
                    if recent_plain_floor_kept.numel() > int(recent_plain_quota):
                        recent_plain_floor_kept = recent_plain_floor_kept[: int(recent_plain_quota)]
                    recent_plain_floor_count = int(recent_plain_floor_kept.numel())
                    for t in recent_plain_floor_kept.tolist():
                        token = int(t)
                        if token in selected_global:
                            continue
                        self._ordered_add(selected, selected_order, [token])
                        selected_global.add(token)

        role_all_idx = geo_role.index_select(0, idx_all)
        plain_visible_mask = (
            ~is_special.index_select(0, idx_all)
            & (role_all_idx == 0)
            & visible_all
        )
        plain_visible_idx = idx_all[plain_visible_mask]
        plain_visible_score = score_all[plain_visible_mask]
        plain_visible_hash = hash_all[plain_visible_mask]
        plain_visible_grouped = self._group_topk_by_hash(
            plain_visible_hash,
            plain_visible_score,
            plain_visible_idx,
            topk_per_voxel=max(1, int(topk_per_voxel)),
        )
        plain_patch_reserved_count = int(recent_plain_floor_count)
        plain_visible_keep_final = torch.empty((0,), dtype=torch.long)
        if plain_visible_grouped.numel() > 0:
            if max_past_tokens is not None:
                plain_visible_quota = max(256, int(0.12 * int(max_past_tokens)))
            else:
                plain_visible_quota = 512
            if plain_visible_grouped.numel() > int(plain_visible_quota):
                grouped_mask = torch.isin(plain_visible_idx, plain_visible_grouped)
                grouped_plain_idx = plain_visible_idx[grouped_mask]
                grouped_plain_score = plain_visible_score[grouped_mask]
                top = torch.topk(grouped_plain_score, k=int(plain_visible_quota), largest=True).indices
                plain_visible_keep = grouped_plain_idx.index_select(0, top)
            else:
                plain_visible_keep = plain_visible_grouped
            plain_visible_keep_final = plain_visible_keep.detach().cpu().long()
            for t in plain_visible_keep.tolist():
                token = int(t)
                if token in selected_global:
                    continue
                self._ordered_add(selected, selected_order, [token])
                selected_global.add(token)
        self.geo_last_policy_inputs["recent_plain_floor_diverse"] = bool(True)
        self.geo_last_policy_inputs["keep_plain_patch_reserved_requested"] = int(recent_plain_floor_count)
        self.geo_last_policy_inputs["keep_plain_patch_final"] = int(plain_visible_keep_final.numel())

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
                for token in addable:
                    if token in selected_global:
                        continue
                    self._ordered_add(selected, selected_order, [token])
                    selected_global.add(token)

        # Post-score bounded keep for labeled geo tokens (visible first).
        if idx_all.numel() > 0:
            role_all = meta["geo_role"].index_select(0, idx_all)
            ref_mask_all = role_all == 4
            land_mask_all = role_all == 2
            anchor_mask_all = role_all == 3
            key_mask_all = role_all == 1
            _ = key_mask_all

            def _masked_topk_from_candidate_space(mask: torch.Tensor, quota: int) -> torch.Tensor:
                if quota <= 0:
                    return torch.empty((0,), dtype=torch.long)
                if idx_all.numel() == 0 or score_all.numel() == 0 or mask.numel() == 0:
                    return torch.empty((0,), dtype=torch.long)
                if idx_all.numel() != score_all.numel() or idx_all.numel() != mask.numel():
                    raise RuntimeError(
                        f"_masked_topk_from_candidate_space mismatch: "
                        f"idx_all={int(idx_all.numel())}, score_all={int(score_all.numel())}, mask={int(mask.numel())}"
                    )
                token_idx = idx_all[mask]
                token_score = score_all[mask]
                if token_idx.numel() == 0:
                    return torch.empty((0,), dtype=torch.long)
                if token_idx.numel() != token_score.numel():
                    raise RuntimeError(
                        f"_masked_topk_from_candidate_space token mismatch: "
                        f"token_idx={int(token_idx.numel())}, token_score={int(token_score.numel())}"
                    )
                k = min(int(quota), int(token_idx.numel()))
                top = torch.topk(token_score, k=k, largest=True).indices
                return token_idx.index_select(0, top)

            if max_past_tokens is not None:
                anchor_quota = min(
                    int(self.geo_anchor_read_quota),
                    max(
                        1,
                        int(
                            float(max_past_tokens)
                            * float(
                                policy["anchor_quota_ratio"] if policy is not None else self.geo_anchor_budget_ratio
                            )
                        ),
                    ),
                )
            else:
                anchor_quota = int(self.geo_anchor_read_quota)
            anchor_keep = _masked_topk_from_candidate_space(
                anchor_mask_all & visible_all,
                anchor_quota,
            )
            for t in anchor_keep.tolist():
                token = int(t)
                if token in selected_global:
                    continue
                self._ordered_add(selected, selected_order, [token])
                selected_global.add(token)

            if enable_landmark_logic:
                land_keep = _masked_topk_from_candidate_space(
                    land_mask_all & visible_all,
                    int(self.geo_landmark_token_quota),
                )
                for t in land_keep.tolist():
                    token = int(t)
                    if token in selected_global:
                        continue
                    self._ordered_add(selected, selected_order, [token])
                    selected_global.add(token)

            if enable_reference_logic:
                ref_scale = float((policy or {}).get("reference_hard_scale", 1.0))
                mode_now_soft_ref = str((policy or {}).get("mode", "legacy"))
                if mode_now_soft_ref in {"current", "recovery"}:
                    obs_stress = float((policy or {}).get("observation_stress", 0.0))
                    if obs_stress < 0.40:
                        min_visible_ref = 24
                        min_invis_ref = 8
                    elif obs_stress < 0.70:
                        min_visible_ref = 12
                        min_invis_ref = 4
                    else:
                        min_visible_ref = 4
                        min_invis_ref = 1
                    visible_ref_quota = max(
                        int(min_visible_ref),
                        int(round(float(self.geo_reference_token_quota) * float(ref_scale))),
                    )
                    invis_ref_quota = max(
                        int(min_invis_ref),
                        int(round(64.0 * float(ref_scale))),
                    )
                else:
                    visible_ref_quota = int(self.geo_reference_token_quota)
                    invis_ref_quota = 64
                self.geo_last_policy_inputs["visible_ref_quota_effective"] = int(visible_ref_quota)
                self.geo_last_policy_inputs["invis_ref_quota_effective"] = int(invis_ref_quota)

                ref_keep = _masked_topk_from_candidate_space(
                    ref_mask_all & visible_all,
                    int(visible_ref_quota),
                )
                for t in ref_keep.tolist():
                    token = int(t)
                    if token in selected_global:
                        continue
                    self._ordered_add(selected, selected_order, [token])
                    selected_global.add(token)

                tiny_invis_ref_quota = int(invis_ref_quota)
                ref_invis_keep = _masked_topk_from_candidate_space(
                    ref_mask_all & (~visible_all),
                    tiny_invis_ref_quota,
                )
                for t in ref_invis_keep.tolist():
                    token = int(t)
                    if token in selected_global:
                        continue
                    self._ordered_add(selected, selected_order, [token])
                    selected_global.add(token)

        if not selected:
            return None

        logger.debug(
            "[geo_prune] total=%d candidate=%d visible=%d selected=%d anchor_in_cache=%d stable_selected=%d tau_bucket=%.4f",
            total_tokens,
            candidate_count,
            visible_total,
            len(selected),
            anchor_count,
            stable_count,
            tau_bucket,
        )
        reanchor_overlap_avg = (float(reanchor_overlap_sum) / float(max(1, reanchor_frames_used))) if reanchor_frames_used > 0 else 0.0
        diag_payload = {
            "diag_idx_all": idx_all.detach().cpu(),
            "diag_hash_all": hash_all.detach().cpu(),
            "diag_visible_all": visible_all.detach().cpu(),
            "diag_stable_hash": stable_hash_for_diag,
            "diag_world_to_cam": world_to_cam.detach().cpu() if isinstance(world_to_cam, torch.Tensor) else None,
            "diag_intrinsic": intrinsic.detach().cpu() if isinstance(intrinsic, torch.Tensor) else None,
            "diag_img_hw": img_hw,
            "diag_near": float(near),
            "diag_far": float(far),
            "stable_visible_voxel_overlap": int(stable_visible_voxel_overlap),
            "stable_selected_visible": int(stable_visible_selected),
            "stable_selected_invisible": int(stable_invisible_selected),
            "visible_total": int(visible_total),
        }
        priority_for_cap = self._unique_preserve_order_long(
            torch.cat(
                [
                    hard_keep.detach().cpu().long() if hard_keep is not None else torch.empty((0,), dtype=torch.long),
                    recent_plain_floor_kept,
                    plain_visible_keep_final,
                    torch.tensor(selected_order, dtype=torch.long),
                ],
                dim=0,
            )
        )
        return self._finalize_geo_keep(
            meta=meta,
            selected=selected,
            selected_order=selected_order,
            current_frame_idx=current_frame_idx,
            recent_frames_eff=recent_frames_eff,
            max_past_tokens=max_past_tokens,
            candidate_count=candidate_count,
            visible_total=visible_total,
            anchor_count=anchor_count,
            stable_count=stable_count,
            tau_bucket=tau_bucket,
            stable_visible_voxel_overlap=stable_visible_voxel_overlap,
            stable_selected_visible=stable_visible_selected,
            stable_selected_invisible=stable_invisible_selected,
            fast_path=0,
            hard_keep_idx=hard_keep,
            reanchor_added=int(reanchor_added),
            reanchor_overlap_avg=float(reanchor_overlap_avg),
            diag_payload=diag_payload,
            priority_keep_idx=priority_for_cap,
            policy=policy,
            allow_fill=False,
            plain_reserved_idx=recent_plain_floor_kept,
        )
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
        geo_use_view_pruning: bool = True,
        geo_topk_per_voxel: int = 4,
        geo_recent_frames: int = 2,
        geo_near: float = 0.05,
        geo_far: float = 200.0,
        current_view: Optional[Dict[str, torch.Tensor]] = None,
        policy_view: Optional[Dict[str, Any]] = None,
        policy_view_source: str = "none",
        selector_view_source: str = "none",
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
        if policy_view is None:
            policy_view = current_view
            policy_view_source = "current_view" if current_view is not None else "none"

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
        self.geo_pending_console_log = None
        geo_policy: Optional[Dict[str, Any]] = None
        effective_mode = "legacy"
        safe_warmup = False
        bootstrap_bank_ready_now = False
        structure_ready_now = False
        enable_landmark_logic = True
        enable_reference_logic = True
        enable_stable_logic = True
        enable_reanchor_logic = True
        if use_geo_kv_prune:
            geo_policy = self._geo_get_effective_policy_for_forward(
                frame_idx=int(past_frame_idx),
                ref_meta=None,
                ref_past_budget=None,
                current_view=policy_view,
            )

        # In geo mode, build one shared keep plan per frame instead of re-running
        # expensive Python/CPU geo selection for every global layer.
        geo_shared_identity_keep: Optional[torch.Tensor] = None
        geo_prune_ready = False
        geo_reloc_active = False
        ref_past_budget = 0
        if use_cache and use_geo_kv_prune and any(kv is not None for kv in past_key_values):
            ref_layer_idx = None
            for idx, kv in enumerate(past_key_values):
                if kv is not None and self.geo_token_meta[idx]["frame_idx"].numel() > 0:
                    ref_layer_idx = idx
                    break

            ref_meta = self.geo_token_meta[ref_layer_idx] if ref_layer_idx is not None else None
            structure_ready_now = self._geo_structure_ready()
            last_obs = self._geo_get_last_observation()
            bootstrap_bank_ready_now = bool(last_obs is not None and float(last_obs.get("bootstrap_bank_ready", 0.0)) > 0.5)
            base_ref_past_budget, base_ref_layer_budget, ref_budget_source, allow_cap = self._geo_get_shared_ref_budget(
                current_budgets=current_budgets,
                P=int(P),
                structure_ready=structure_ready_now,
                geo_policy=geo_policy,
                frame_idx=int(past_frame_idx),
            )
            cache_frame_idx = int(ref_meta["frame_idx"].max().item()) if (ref_meta is not None and ref_meta["frame_idx"].numel() > 0) else max(-1, int(past_frame_idx) - 1)
            if self._geo_has_complete_policy_context(ref_meta, max(0, int(base_ref_layer_budget) - P)):
                policy_preview, ref_past_budget, ref_layer_budget, fp_iters_used, fp_converged = self._geo_preview_policy_with_budget_fixed_point(
                    frame_idx=int(cache_frame_idx),
                    ref_meta=ref_meta,
                    raw_ref_layer_budget=int(base_ref_layer_budget),
                    current_view=policy_view,
                    P=int(P),
                    allow_cap=bool(allow_cap),
                )
                geo_policy = self._geo_commit_adaptive_policy_once(
                    frame_idx=int(cache_frame_idx),
                    total_tokens=int(ref_meta["frame_idx"].numel()),
                    max_past_tokens=int(ref_past_budget),
                    current_view=policy_view,
                    observation=self._geo_get_last_observation(),
                    selector_diag=self._geo_get_last_selector_diag(),
                )
                self.geo_last_policy_inputs["preview_policy_frame"] = int(cache_frame_idx)
                self.geo_last_policy_inputs["commit_policy_frame"] = int(self.geo_last_policy_frame)
                self.geo_last_policy_inputs["raw_ref_budget"] = int(base_ref_layer_budget)
                self.geo_last_policy_inputs["base_ref_layer_budget"] = int(base_ref_layer_budget)
                self.geo_last_policy_inputs["final_ref_budget"] = int(ref_past_budget)
                self.geo_last_policy_inputs["final_ref_layer_budget"] = int(ref_layer_budget)
                self.geo_last_policy_inputs["policy_view_source"] = str(policy_view_source)
                self.geo_last_policy_inputs["selector_view_source"] = str(selector_view_source)
                self.geo_last_policy_inputs["use_view_pruning"] = bool((geo_policy or {}).get("use_view_pruning", geo_use_view_pruning))
                self.geo_last_policy_inputs["prefer_last_reliable_view"] = bool((policy_preview or {}).get("prefer_last_reliable_view", False))
                self.geo_last_policy_inputs["selector_mode"] = str((geo_policy or {}).get("mode", "legacy"))
                self.geo_last_policy_inputs["fixed_point_iters_used"] = int(fp_iters_used)
                self.geo_last_policy_inputs["fixed_point_converged"] = bool(fp_converged)
                _ = policy_preview
            elif self.geo_last_committed_policy is not None:
                geo_policy = copy.deepcopy(self.geo_last_committed_policy)
                ref_layer_budget = self._scheduled_layer_budget(base_ref_layer_budget, int(past_frame_idx), policy=geo_policy) if allow_cap else int(base_ref_layer_budget)
                ref_past_budget = max(0, int(ref_layer_budget) - P)
                self.geo_last_policy_inputs["final_ref_layer_budget"] = int(ref_layer_budget)
                self.geo_last_policy_inputs["fixed_point_iters_used"] = int(0)
                self.geo_last_policy_inputs["fixed_point_converged"] = bool(False)
            else:
                geo_policy = self._geo_default_policy(int(cache_frame_idx))
                ref_layer_budget = self._scheduled_layer_budget(base_ref_layer_budget, int(past_frame_idx), policy=geo_policy) if allow_cap else int(base_ref_layer_budget)
                ref_past_budget = max(0, int(ref_layer_budget) - P)
                self.geo_last_policy_inputs["final_ref_layer_budget"] = int(ref_layer_budget)
                self.geo_last_policy_inputs["fixed_point_iters_used"] = int(0)
                self.geo_last_policy_inputs["fixed_point_converged"] = bool(False)

            selector_use_view_pruning = bool((geo_policy or {}).get("use_view_pruning", geo_use_view_pruning))
            self.geo_last_policy_inputs["structure_ready"] = bool(structure_ready_now)
            self.geo_last_policy_inputs["bootstrap_bank_ready"] = bool(bootstrap_bank_ready_now)
            self.geo_last_policy_inputs["ref_budget_source"] = str(ref_budget_source)
            self.geo_last_policy_inputs["use_cap"] = bool(allow_cap)
            self.geo_last_policy_inputs["allow_cap"] = bool(allow_cap)
            self.geo_last_policy_inputs["base_ref_layer_budget"] = int(base_ref_layer_budget)
            self.geo_last_policy_inputs["base_ref_budget"] = int(base_ref_past_budget)

            self.geo_last_policy_inputs["policy_view_source"] = str(policy_view_source)
            self.geo_last_policy_inputs["selector_view_source"] = str(selector_view_source)
            self.geo_last_policy_inputs["use_view_pruning"] = bool(selector_use_view_pruning)
            self.geo_last_policy_inputs["prefer_last_reliable_view"] = bool((geo_policy or {}).get("prefer_last_reliable_view", False))
            self.geo_last_policy_inputs["selector_mode"] = str((geo_policy or {}).get("mode", "legacy"))
            self.geo_last_policy_inputs["landmark_growth_ready"] = bool((geo_policy or {}).get("landmark_growth_ready", False))
            self.geo_last_policy_inputs["reference_growth_ready"] = bool((geo_policy or {}).get("reference_growth_ready", False))
            self.geo_last_policy_inputs["landmark_label_ready"] = bool((geo_policy or {}).get("landmark_label_ready", (geo_policy or {}).get("use_landmark_labels", False)))
            self.geo_last_policy_inputs["reference_label_ready"] = bool((geo_policy or {}).get("reference_label_ready", (geo_policy or {}).get("use_reference_labels", False)))
            self.geo_last_policy_inputs["anchor_phase_open"] = bool((geo_policy or {}).get("anchor_phase_open", False))
            self.geo_last_policy_inputs["landmark_phase_open"] = bool((geo_policy or {}).get("landmark_phase_open", False))
            self.geo_last_policy_inputs["reference_phase_open"] = bool((geo_policy or {}).get("reference_phase_open", False))
            self.geo_last_policy_inputs["reloc_phase_open"] = bool((geo_policy or {}).get("reloc_phase_open", False))
            self.geo_last_policy_inputs["use_recovery"] = bool((geo_policy or {}).get("use_recovery", False))
            self.geo_last_policy_inputs["legacy_observation_break"] = bool((geo_policy or {}).get("legacy_observation_break", False))
            self.geo_last_policy_inputs["legacy_break_force_recent_plain"] = bool((geo_policy or {}).get("legacy_break_force_recent_plain", False))
            self.geo_last_policy_inputs["legacy_break_anchor_scale"] = float((geo_policy or {}).get("legacy_break_anchor_scale", 1.0) or 1.0)
            self.geo_last_policy_inputs["legacy_break_frame0_scale"] = float((geo_policy or {}).get("legacy_break_frame0_scale", 1.0) or 1.0)
            self.geo_last_policy_inputs["legacy_break_recent_plain_ratio"] = float((geo_policy or {}).get("legacy_break_recent_plain_ratio", 0.08) or 0.08)
            self.geo_last_policy_inputs["allow_reloc_trigger"] = bool((geo_policy or {}).get("allow_reloc_trigger", False))
            self.geo_last_policy_inputs["reloc_gate_open"] = bool((geo_policy or {}).get("allow_reloc_trigger", False))
            self.geo_last_policy_inputs["use_reloc"] = bool((geo_policy or {}).get("use_reloc", False))
            self.geo_last_policy_inputs["ongoing_recovery"] = bool(int(self.geo_recovery_frames_left) > 0)
            self.geo_last_policy_inputs["recovery_timer_active"] = bool(int(self.geo_recovery_frames_left) > 0)
            self.geo_last_policy_inputs["ongoing_reloc"] = bool(int(self.geo_reloc_frames_left) > 0 or str(self.geo_reloc_state) != "off")

            policy_mode = str((geo_policy or {}).get("mode", "legacy"))
            ongoing_recovery = bool(int(self.geo_recovery_frames_left) > 0)
            ongoing_reloc = bool(int(self.geo_reloc_frames_left) > 0 or str(self.geo_reloc_state) != "off")
            if ongoing_reloc:
                effective_mode = "reloc"
            elif policy_mode == "recovery" or ongoing_recovery:
                effective_mode = "recovery"
            elif (policy_mode == "current") and (not bool(structure_ready_now)):
                effective_mode = "legacy"
            else:
                effective_mode = policy_mode
            self.geo_last_policy_inputs["policy_mode"] = str(policy_mode)
            self.geo_last_policy_inputs["effective_mode"] = str(effective_mode)
            exec_policy = copy.deepcopy(geo_policy or {})
            exec_policy["mode"] = str(effective_mode)
            if str(effective_mode) in {"legacy", "recovery", "reloc"}:
                exec_policy["use_cap"] = False
                exec_policy["cap_alpha"] = 0.0
            if bool(allow_cap):
                ref_layer_budget = self._scheduled_layer_budget(base_ref_layer_budget, int(past_frame_idx), policy=exec_policy)
                ref_past_budget = max(0, int(ref_layer_budget) - P)
                self.geo_last_policy_inputs["final_ref_budget"] = int(ref_past_budget)
                self.geo_last_policy_inputs["final_ref_layer_budget"] = int(ref_layer_budget)
            self.geo_last_policy_inputs["exec_policy_mode"] = str(exec_policy.get("mode", "legacy"))
            self.geo_last_policy_inputs["exec_use_cap"] = bool(exec_policy.get("use_cap", False))
            self.geo_last_policy_inputs["layer_cap_policy_mode"] = str(exec_policy.get("mode", "legacy"))
            self.geo_last_policy_inputs["structure_ready_latched"] = bool(self.geo_structure_ready_latched)
            self.geo_last_policy_inputs["structure_unready_streak"] = int(self.geo_structure_unready_streak)
            self.geo_last_policy_inputs["ongoing_recovery"] = bool(ongoing_recovery)
            self.geo_last_policy_inputs["recovery_timer_active"] = bool(int(self.geo_recovery_frames_left) > 0)
            self.geo_last_policy_inputs["ongoing_reloc"] = bool(ongoing_reloc)
            self.geo_last_policy_inputs["allow_reference_refresh_only"] = bool(
                last_obs is not None and float(last_obs.get("allow_reference_refresh_only", 0.0)) > 0.5
            )
            self.geo_last_policy_inputs["structure_ready"] = bool(structure_ready_now)
            self.geo_last_policy_inputs["bootstrap_bank_ready"] = bool(bootstrap_bank_ready_now)
            self.geo_last_policy_inputs["recovery_selector"] = bool(False)
            self.geo_last_policy_inputs["shared_keep_order_preserved"] = bool(False)
            self.geo_last_policy_inputs["frame_keep_plain_patch_final_min"] = None
            self.geo_last_policy_inputs["frame_keep_plain_patch_reserved_min"] = None
            self.geo_last_policy_inputs["frame_keep_budget_min"] = None
            self.geo_last_policy_inputs["frame_keep_plain_patch_final_last"] = None
            self.geo_last_policy_inputs["frame_keep_plain_patch_reserved_last"] = None
            self.geo_last_policy_inputs["frame_keep_budget_last"] = None

            if ref_meta is not None:
                mode_now = str(effective_mode)
                geo_prune_ready = True
                hard_keep_for_bootstrap = self._build_hard_backbone_keep(
                    ref_meta,
                    current_frame_idx=int(cache_frame_idx),
                    max_past_tokens=ref_past_budget,
                    policy=exec_policy,
                )
                if mode_now == "legacy":
                    self.geo_last_policy_inputs["recovery_selector"] = bool(False)
                    geo_reloc_active = False
                    geo_shared_keep_idx = self._select_geo_active_indices_bootstrap(
                        meta=ref_meta,
                        topk_per_voxel=geo_topk_per_voxel,
                        recent_frames=geo_recent_frames,
                        near=geo_near,
                        far=geo_far,
                        current_view=current_view,
                        hard_keep=hard_keep_for_bootstrap,
                        use_view_pruning=bool(selector_use_view_pruning),
                        max_past_tokens=ref_past_budget,
                        policy=exec_policy,
                    )
                    geo_shared_identity_keep = self._build_identity_keep_from_meta(ref_meta, geo_shared_keep_idx)
                elif mode_now in {"current", "recovery"}:
                    geo_reloc_active = False
                    recovery_selector = bool(mode_now == "recovery" or int(self.geo_recovery_frames_left) > 0)
                    self.geo_last_policy_inputs["recovery_selector"] = bool(recovery_selector)
                    selector_policy = copy.deepcopy(exec_policy)
                    if recovery_selector:
                        selector_policy["use_cap"] = False
                        selector_policy["local_budget_ratio"] = max(
                            float(selector_policy.get("local_budget_ratio", self.geo_local_budget_ratio)),
                            0.70,
                        )
                        selector_policy["stable_read_budget_ratio"] = max(
                            float(selector_policy.get("stable_read_budget_ratio", self.geo_stable_read_budget_ratio)),
                            0.30,
                        )
                        selector_policy["hard_recent_frames"] = max(
                            int(selector_policy.get("hard_recent_frames", self.geo_hard_recent_frames)),
                            8,
                        )
                        selector_policy["recent_window"] = max(
                            int(selector_policy.get("recent_window", self.geo_legacy_recent_window)),
                            16,
                        )
                        selector_policy["soft_recent_window"] = max(
                            int(selector_policy.get("soft_recent_window", self.geo_legacy_soft_recent_frames)),
                            20,
                        )

                    geo_shared_keep_idx = self._select_geo_active_indices(
                        meta=ref_meta,
                        topk_per_voxel=geo_topk_per_voxel,
                        recent_frames=geo_recent_frames,
                        near=geo_near,
                        far=geo_far,
                        current_view=current_view,
                        trigger_view=policy_view,
                        use_view_pruning=bool(selector_use_view_pruning),
                        max_past_tokens=ref_past_budget,
                        enable_reference_logic=bool(selector_policy.get("use_reference_labels", False) or recovery_selector),
                        enable_landmark_logic=bool(selector_policy.get("use_landmark_labels", False) or recovery_selector),
                        enable_stable_logic=bool(selector_policy.get("use_reference_labels", False) or recovery_selector),
                        enable_reanchor_logic=bool(selector_policy.get("use_reloc", False) or recovery_selector),
                        policy=selector_policy,
                    )
                    protected_ref = torch.nonzero(self._hard_protected_mask(ref_meta), as_tuple=False).view(-1)
                    if geo_shared_keep_idx is None or geo_shared_keep_idx.numel() == 0:
                        geo_shared_keep_idx = protected_ref
                    elif protected_ref.numel() > 0:
                        geo_shared_keep_idx = self._unique_preserve_order_long(
                            torch.cat([geo_shared_keep_idx.detach().cpu().long(), protected_ref], dim=0)
                        )
                    self.geo_last_policy_inputs["shared_keep_order_preserved"] = bool(True)
                    geo_shared_identity_keep = self._build_identity_keep_from_meta(ref_meta, geo_shared_keep_idx)
                elif mode_now == "reloc":
                    geo_reloc_active = True
                    geo_shared_identity_keep = self._build_reloc_identity_keep(
                        meta=ref_meta,
                        max_past_tokens=max(0, int(ref_past_budget)),
                        recent_frames=max(1, int(geo_recent_frames)),
                        policy=exec_policy,
                    )
                else:
                    raise RuntimeError(f"Unknown effective_mode: {mode_now}")
            else:
                self._queue_geo_console_log(
                    current_frame_idx=max(-1, int(past_frame_idx)),
                    total_tokens=0,
                    candidate_count=0,
                    visible_total=0,
                    selected_count=0,
                    anchor_count=0,
                    stable_count=0,
                    tau_bucket=float("nan"),
                    stable_visible_voxel_overlap=0,
                    stable_selected_visible=0,
                    stable_selected_invisible=0,
                    fast_path=4,
                    cache_size=0,
                    keep_overlap_cache=0,
                    reanchor_added=0,
                    reanchor_overlap_avg=0.0,
                    budget=int(ref_past_budget),
                    kv_comp_old=self._summarize_kv_meta(None, recent_frames=geo_recent_frames),
                )
        elif use_cache and use_geo_kv_prune:
            raw_ref_layer_budget = max(0, int(current_budgets.min().item()))
            ref_layer_budget = self._scheduled_layer_budget(raw_ref_layer_budget, int(past_frame_idx), policy=geo_policy)
            ref_past_budget = max(0, int(ref_layer_budget) - P)
            self._queue_geo_console_log(
                current_frame_idx=max(-1, int(past_frame_idx)),
                total_tokens=0,
                candidate_count=0,
                visible_total=0,
                selected_count=0,
                anchor_count=0,
                stable_count=0,
                tau_bucket=float("nan"),
                stable_visible_voxel_overlap=0,
                stable_selected_visible=0,
                stable_selected_invisible=0,
                fast_path=5,
                cache_size=0,
                keep_overlap_cache=0,
                reanchor_added=0,
                reanchor_overlap_avg=0.0,
                budget=int(ref_past_budget),
                kv_comp_old=self._summarize_kv_meta(None, recent_frames=geo_recent_frames),
            )

        if use_geo_kv_prune:
            geo_policy = copy.deepcopy(geo_policy or self._geo_default_policy(int(past_frame_idx)))
            effective_mode = str(effective_mode or (geo_policy or {}).get("mode", "legacy"))
            exec_policy = copy.deepcopy(geo_policy)
            exec_policy["mode"] = str(effective_mode)
            if str(effective_mode) in {"legacy", "recovery", "reloc"}:
                exec_policy["use_cap"] = False
                exec_policy["cap_alpha"] = 0.0
            self.geo_last_policy_inputs["exec_policy_mode"] = str(exec_policy.get("mode", "legacy"))
            self.geo_last_policy_inputs["exec_use_cap"] = bool(exec_policy.get("use_cap", False))
            self.geo_last_policy_inputs["layer_cap_policy_mode"] = str(exec_policy.get("mode", "legacy"))
            safe_warmup = bool(
                (effective_mode == "legacy")
                and (not bool((geo_policy or {}).get("use_anchor_labels", False)))
            )
            self.geo_last_policy_inputs["use_anchor_labels"] = bool((geo_policy or {}).get("use_anchor_labels", False))
            self.geo_last_policy_inputs["safe_warmup"] = bool(safe_warmup)
            enable_landmark_logic = bool(geo_policy["use_landmark_labels"])
            enable_reference_logic = bool(geo_policy["use_reference_labels"])
            enable_stable_logic = bool(geo_policy["use_reference_labels"])
            enable_reanchor_logic = bool(geo_policy["use_reloc"])

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
                        apply_early_floor = bool(
                            use_geo_kv_prune
                            and (effective_mode == "legacy")
                            and (
                                bool(safe_warmup)
                                or int(past_frame_idx) <= int(self.geo_early_stabilize_frames)
                            )
                        )
                        self.geo_last_policy_inputs["early_budget_floor_applied"] = bool(apply_early_floor)
                        if apply_early_floor:
                            raw_layer_budget = max(raw_layer_budget, int(self.geo_early_budget_floor))
                        if use_geo_kv_prune and effective_mode != "legacy":
                            layer_budget = self._scheduled_layer_budget(raw_layer_budget, int(past_frame_idx), policy=exec_policy)
                        else:
                            layer_budget = raw_layer_budget
                        debug_protected = torch.empty((0,), dtype=torch.long)
                        debug_keep_idx = torch.empty((0,), dtype=torch.long)
                        debug_pre_keep = torch.empty((0,), dtype=torch.long)
                        past_kv_block = past_key_values[layer_idx] if past_key_values[layer_idx] is not None else None
                        kv_before_len = int(past_kv_block[0].shape[2]) if past_kv_block is not None else 0
                        past_meta = self.geo_token_meta[layer_idx]

                        if use_geo_kv_prune and past_kv_block is not None:
                            max_past_tokens = max(0, layer_budget - P)
                            if safe_warmup:
                                kv_len = int(past_kv_block[0].shape[2])
                                if kv_len <= max_past_tokens:
                                    keep_idx = torch.arange(kv_len, dtype=torch.long)
                                else:
                                    keep_idx = self._build_safe_warmup_keep(
                                        past_meta,
                                        budget=max_past_tokens,
                                        recent_frames=max(8, int(geo_recent_frames)),
                                        current_frame_idx=int(past_frame_idx),
                                        policy=exec_policy,
                                    )
                            elif geo_reloc_active:
                                layer_identity_keep = self._build_reloc_identity_keep(
                                    meta=past_meta,
                                    max_past_tokens=max_past_tokens,
                                    recent_frames=max(1, int(geo_recent_frames)),
                                    policy=exec_policy,
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
                                    policy=exec_policy,
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
                            keep_idx = self._sanitize_keep_idx_preserve_order(
                                keep_idx,
                                meta_len=past_meta["frame_idx"].numel(),
                                kv_len=past_kv_block[0].shape[2],
                            )

                            protected_idx = torch.nonzero(self._hard_protected_mask(past_meta), as_tuple=False).view(-1)
                            debug_protected = protected_idx
                            if keep_idx.numel() > 0 and protected_idx.numel() > 0:
                                pre_keep_all = self._unique_preserve_order_long(
                                    torch.cat([keep_idx.detach().cpu().long(), protected_idx], dim=0)
                                )
                            elif keep_idx.numel() > 0:
                                pre_keep_all = keep_idx
                            elif protected_idx.numel() > 0:
                                pre_keep_all = protected_idx
                            else:
                                pre_keep_all = torch.arange(past_kv_block[0].shape[2], dtype=torch.long)
                            pre_keep = self._cap_keep_for_geo_mode(
                                meta=past_meta,
                                keep_idx=pre_keep_all,
                                budget=int(max_past_tokens),
                                recent_frames=int(geo_recent_frames),
                                current_frame_idx=int(past_frame_idx),
                                policy=exec_policy,
                                priority_keep_idx=keep_idx if keep_idx is not None else pre_keep_all,
                            )
                            debug_pre_keep = pre_keep
                            pre_keep = self._sanitize_keep_idx_preserve_order(
                                pre_keep,
                                meta_len=past_meta["frame_idx"].numel(),
                                kv_len=past_kv_block[0].shape[2],
                            )
                            if pre_keep.numel() == 0:
                                past_kv_block = None
                                past_meta = self.geo_token_meta[layer_idx]
                            elif not self._is_full_range_keep(pre_keep, past_kv_block[0].shape[2]):
                                pre_keep_mat = torch.sort(pre_keep).values
                                pre_keep_dev = pre_keep_mat.to(past_kv_block[0].device)
                                past_kv_block = (
                                    torch.index_select(past_kv_block[0], 2, pre_keep_dev),
                                    torch.index_select(past_kv_block[1], 2, pre_keep_dev),
                                )
                                past_meta = self._index_meta(past_meta, pre_keep_mat)

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
                            self._decay_persistent_labels(merged_meta, int(past_frame_idx))
                            anchor_enabled = bool((geo_policy or {}).get("use_anchor_labels", False))
                            landmark_enabled = bool((geo_policy or {}).get("use_landmark_labels", False))
                            reference_enabled = bool((geo_policy or {}).get("use_reference_labels", False))

                            n_meta = int(merged_meta["frame_idx"].numel())
                            zeros = torch.zeros((n_meta,), dtype=torch.bool)
                            anchor_raw = self._derive_anchor_mask_from_meta(merged_meta)
                            self.geo_last_policy_inputs["anchor_count_raw"] = int(anchor_raw.sum().item())

                            if not anchor_enabled:
                                merged_meta["is_anchor"] = zeros
                                merged_meta["is_landmark"] = zeros
                                merged_meta["is_reference"] = zeros
                            else:
                                prev_anchor = merged_meta.get("is_anchor", zeros)
                                layer_budget_eff = max(int(P), int(layer_budget))
                                if not bool(structure_ready_now):
                                    anchor_quota = max(96, min(256, int(0.05 * layer_budget_eff)))
                                else:
                                    anchor_quota = max(64, min(160, int(0.03 * layer_budget_eff)))
                                new_anchor = self._bounded_label_from_mask(
                                    merged_meta,
                                    anchor_raw,
                                    per_frame_quota=int(anchor_quota),
                                )
                                merged_meta["is_anchor"] = prev_anchor | new_anchor
                                eligible = self._label_eligible_mask(merged_meta)
                                prev_landmark = merged_meta.get("is_landmark", zeros)
                                prev_reference = merged_meta.get("is_reference", zeros)

                                if not bool(structure_ready_now):
                                    merged_meta["is_landmark"] = prev_landmark
                                    merged_meta["is_reference"] = prev_reference
                                elif not landmark_enabled:
                                    merged_meta["is_landmark"] = prev_landmark
                                    merged_meta["is_reference"] = prev_reference
                                elif not reference_enabled:
                                    landmark_raw = self._derive_landmark_mask_from_meta(merged_meta)
                                    new_landmark = self._bounded_label_from_mask(
                                        merged_meta,
                                        eligible & landmark_raw,
                                        per_frame_quota=128,
                                    )
                                    merged_meta["is_landmark"] = prev_landmark | new_landmark
                                    merged_meta["is_reference"] = prev_reference
                                else:
                                    landmark_raw = self._derive_landmark_mask_from_meta(merged_meta)
                                    reference_raw = self._derive_reference_mask_from_meta(merged_meta)
                                    new_landmark = self._bounded_label_from_mask(
                                        merged_meta,
                                        eligible & landmark_raw,
                                        per_frame_quota=128,
                                    )
                                    new_reference = self._bounded_label_from_mask(
                                        merged_meta,
                                        eligible & reference_raw,
                                        per_frame_quota=64,
                                    )
                                    merged_meta["is_landmark"] = prev_landmark | new_landmark
                                    merged_meta["is_reference"] = prev_reference | new_reference

                            merged_meta["geo_role"] = self._compute_primary_geo_role(merged_meta)

                            # Keep explicit hard cap in geo mode with hard-backbone-first protection.
                            if new_kv[0].shape[2] > layer_budget:
                                cap_all = torch.arange(new_kv[0].shape[2], dtype=torch.long)
                                frame_idx_all = merged_meta["frame_idx"]
                                current_frame_tokens = cap_all[frame_idx_all == int(past_frame_idx)]
                                past_tokens = cap_all[frame_idx_all < int(past_frame_idx)]
                                priority_cap_idx = self._unique_preserve_order_long(
                                    torch.cat([current_frame_tokens, past_tokens], dim=0)
                                )
                                hard_keep_final = self._build_hard_backbone_keep(
                                    merged_meta,
                                    current_frame_idx=int(past_frame_idx),
                                    max_past_tokens=int(layer_budget),
                                    policy=exec_policy,
                                )
                                cap_keep = self._cap_keep_with_hard_protection(
                                    meta=merged_meta,
                                    keep_idx=cap_all,
                                    hard_keep=hard_keep_final,
                                    budget=int(layer_budget),
                                    recent_frames=int(geo_recent_frames),
                                    priority_keep_idx=priority_cap_idx,
                                    policy=exec_policy,
                                )
                                cap_keep = self._sanitize_keep_idx_preserve_order(
                                    cap_keep,
                                    meta_len=merged_meta["frame_idx"].numel(),
                                    kv_len=new_kv[0].shape[2],
                                )
                                cap_keep_mat = torch.sort(cap_keep).values
                                cap_keep_dev = cap_keep_mat.to(new_kv[0].device)
                                new_kv = (
                                    torch.index_select(new_kv[0], 2, cap_keep_dev),
                                    torch.index_select(new_kv[1], 2, cap_keep_dev),
                                )
                                merged_meta = self._index_meta(merged_meta, cap_keep_mat)
                            frame0_in_cache = int((merged_meta["frame_idx"] == 0).sum().item())
                            ref_in_cache = int(merged_meta.get("is_reference", torch.zeros_like(merged_meta["is_special"])).sum().item())
                            landmark_in_cache = int(merged_meta.get("is_landmark", torch.zeros_like(merged_meta["is_special"])).sum().item())
                            anchor_in_cache = int(merged_meta.get("is_anchor", torch.zeros_like(merged_meta["is_special"])).sum().item())
                            debug_frame_idx = int(past_frame_idx)
                            if self._should_log_geo_debug(debug_frame_idx):
                                logger.info(
                                    "[geo_debug] layer=%d kv_before=%d meta_before=%d protected=%d keep_idx=%d pre_keep=%d new_kv=%d merged_meta=%d layer_budget=%d trust=%.4f recovery=%d reloc=%d safe_warmup=%d bootstrap_bank_ready=%d structure_ready=%d exec_use_cap=%d layer_cap_policy_mode=%s use_anchor_labels=%d anchor_count_raw=%d frame0_in_cache=%d ref_in_cache=%d landmark_in_cache=%d anchor_in_cache=%d keep_plain_patch_reserved=%d keep_plain_patch_final=%d frame0_hard_kept=%d keep_plain_patch_hard_floor=%d frame0_hard_capped_diverse=%d early_budget_floor_applied=%d shared_ref_early_floor_applied=%d shared_keep_order_preserved=%d allow_fill_effective=%d fast_path_allow_fill=%d selector_diag_updated=%d frame0_priority_after_plain=%d current_recovery_ref_before_frame0=%d extra_frame0_soft_promotion_enabled=%d keep_plain_patch_reserved_requested=%d reserved_ratio_prev=%.4f reserved_target_effective=%.4f frame0_hard_scale=%.4f reference_hard_scale=%.4f shared_ref_budget_upper=%d shared_ref_prev_layer_budget=%d visible_ref_quota_effective=%d invis_ref_quota_effective=%d recent_plain_floor_diverse=%d recent_plain_ratio_effective=%.4f frame_keep_plain_patch_final_min=%d frame_keep_plain_patch_reserved_min=%d frame_keep_budget_min=%d priority_keep_fastpath_has_plain_floor=%d implicit_recent_plain_floor_used=%d fastpath_recent_plain_floor_added=%d fastpath_recent_frames_eff=%d keep_plain_patch_reserved_prev_is_fastpath_safe=%d hard_cap_unique_budget=%d frame0_quota_effective=%d anchor_ttl_effective=%d allow_reference_refresh_only=%d",
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
                                    int(bool(safe_warmup)),
                                    int(bool(bootstrap_bank_ready_now)),
                                    int(bool(structure_ready_now)),
                                    int(bool(self.geo_last_policy_inputs.get("exec_use_cap", False))),
                                    str(self.geo_last_policy_inputs.get("layer_cap_policy_mode", "legacy")),
                                    int(bool((geo_policy or {}).get("use_anchor_labels", False))),
                                    int(self.geo_last_policy_inputs.get("anchor_count_raw", 0) or 0),
                                    int(frame0_in_cache),
                                    int(ref_in_cache),
                                    int(landmark_in_cache),
                                    int(anchor_in_cache),
                                    int(self.geo_last_policy_inputs.get("keep_plain_patch_reserved", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("keep_plain_patch_final", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("frame0_hard_kept", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("keep_plain_patch_hard_floor", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("frame0_hard_capped_diverse", 0) or 0),
                                    int(bool(self.geo_last_policy_inputs.get("early_budget_floor_applied", False))),
                                    int(bool(self.geo_last_policy_inputs.get("shared_ref_early_floor_applied", False))),
                                    int(bool(self.geo_last_policy_inputs.get("shared_keep_order_preserved", False))),
                                    int(bool(self.geo_last_policy_inputs.get("allow_fill_effective", True))),
                                    int(bool(self.geo_last_policy_inputs.get("fast_path_allow_fill", False))),
                                    int(bool(self.geo_last_policy_inputs.get("selector_diag_updated", True))),
                                    int(bool(self.geo_last_policy_inputs.get("frame0_priority_after_plain", False))),
                                    int(bool(self.geo_last_policy_inputs.get("current_recovery_ref_before_frame0", False))),
                                    int(bool(self.geo_last_policy_inputs.get("extra_frame0_soft_promotion_enabled", False))),
                                    int(self.geo_last_policy_inputs.get("keep_plain_patch_reserved_requested", 0) or 0),
                                    float(self.geo_last_policy_inputs.get("reserved_ratio_prev", 0.0) or 0.0),
                                    float(self.geo_last_policy_inputs.get("reserved_target_effective", 0.06) or 0.06),
                                    float(self.geo_last_policy_inputs.get("frame0_hard_scale", 1.0) or 1.0),
                                    float(self.geo_last_policy_inputs.get("reference_hard_scale", 1.0) or 1.0),
                                    int(self.geo_last_policy_inputs.get("shared_ref_budget_upper", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("shared_ref_prev_layer_budget", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("visible_ref_quota_effective", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("invis_ref_quota_effective", 0) or 0),
                                    int(bool(self.geo_last_policy_inputs.get("recent_plain_floor_diverse", False))),
                                    float(self.geo_last_policy_inputs.get("recent_plain_ratio_effective", 0.06) or 0.06),
                                    int(self.geo_last_policy_inputs.get("frame_keep_plain_patch_final_min", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("frame_keep_plain_patch_reserved_min", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("frame_keep_budget_min", 0) or 0),
                                    int(bool(self.geo_last_policy_inputs.get("priority_keep_fastpath_has_plain_floor", False))),
                                    int(bool(self.geo_last_policy_inputs.get("implicit_recent_plain_floor_used", False))),
                                    int(self.geo_last_policy_inputs.get("fastpath_recent_plain_floor_added", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("fastpath_recent_frames_eff", 0) or 0),
                                    int(bool(self.geo_last_policy_inputs.get("keep_plain_patch_reserved_prev_is_fastpath_safe", False))),
                                    int(bool(self.geo_last_policy_inputs.get("hard_cap_unique_budget", False))),
                                    int(self.geo_last_policy_inputs.get("frame0_quota_effective", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("anchor_ttl_effective", 0) or 0),
                                    int(self.geo_last_policy_inputs.get("allow_reference_refresh_only", False)),
                                )
                            assert int(merged_meta["frame_idx"].numel()) == int(new_kv[0].shape[2]), "geo meta/KV length mismatch"
                            self.geo_token_meta[layer_idx] = merged_meta
                            if int(layer_idx) == 0:
                                kv_new = self._summarize_kv_meta(
                                    merged_meta,
                                    recent_frames=geo_recent_frames,
                                )
                                self._flush_geo_console_log(kv_comp_new=kv_new)

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

        self._flush_geo_console_log(kv_comp_new=None)

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
