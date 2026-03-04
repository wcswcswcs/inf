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
        geo_keyframe_time_bins: int = 4,
        geo_anchor_stable_ratio: float = 0.6,
        geo_anchor_adaptive_age_decay: float = 0.05,
        geo_anchor_stable_age_decay: float = 0.005,
        geo_full_select_pose_delta: float = 0.15,
        geo_full_select_conf_drop: float = 0.15,
        geo_full_select_new_voxel_ratio: float = 0.25,
        geo_keyframe_protected_quota: int = 256,
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
        self.geo_keyframe_time_bins = max(1, int(geo_keyframe_time_bins))
        self.geo_anchor_stable_ratio = float(min(max(geo_anchor_stable_ratio, 0.0), 1.0))
        self.geo_anchor_adaptive_age_decay = max(0.0, float(geo_anchor_adaptive_age_decay))
        self.geo_anchor_stable_age_decay = max(0.0, float(geo_anchor_stable_age_decay))
        self.geo_full_select_pose_delta = max(0.0, float(geo_full_select_pose_delta))
        self.geo_full_select_conf_drop = max(0.0, float(geo_full_select_conf_drop))
        self.geo_full_select_new_voxel_ratio = max(0.0, float(geo_full_select_new_voxel_ratio))
        self.geo_keyframe_protected_quota = max(0, int(geo_keyframe_protected_quota))
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
        self.geo_trim_cursor = 0
        self.geo_anchor_version = 0
        self.geo_frame_anchor_version: Dict[int, int] = {}
        self.geo_keyframes: List[int] = []
        self.geo_keyframe_set: set[int] = set()
        self.geo_token_meta: Dict[int, Dict[str, torch.Tensor]] = {
            i: {
                "frame_idx": torch.empty(0, dtype=torch.long),
                "is_special": torch.empty(0, dtype=torch.bool),
                "is_keyframe": torch.empty(0, dtype=torch.bool),
                "local_patch_idx": torch.empty(0, dtype=torch.long),
                "identity_local": torch.empty(0, dtype=torch.long),
                "global_id": torch.empty(0, dtype=torch.long),
                "is_anchor": torch.empty(0, dtype=torch.bool),
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

    def _voxel_importance(self, item: Dict[str, float], now_frame_idx: int, age_decay: Optional[float] = None) -> float:
        age = max(0.0, float(now_frame_idx) - float(item["last_seen"]))
        decay = self.geo_anchor_adaptive_age_decay if age_decay is None else max(0.0, float(age_decay))
        return (
            float(item["conf_ema"])
            * torch.log1p(torch.tensor(float(item["support"]))).item()
            * (1.0 / (1.0 + float(item["pos_var"])))
            * (1.0 / (1.0 + decay * age))
        )

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

    def _refresh_geo_anchor_voxels(self, now_frame_idx: int):
        if not self.geo_voxel_bank:
            self.geo_anchor_voxels = set()
            self.geo_anchor_voxel_list = []
            self.geo_anchor_birth = {}
            self.geo_anchor_hash_tensor = torch.empty((0,), dtype=torch.long)
            self.geo_anchor_version += 1
            return

        prev_anchors = self.geo_anchor_voxels
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
            return {"new_voxel_ratio": 0.0}
        if pts3d.ndim != 4 or conf.ndim != 3:
            return {"new_voxel_ratio": 0.0}

        _, H, W, _ = pts3d.shape
        gh, gw = H // self.patch_size, W // self.patch_size
        if gh <= 0 or gw <= 0:
            return {"new_voxel_ratio": 0.0}

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

        new_voxels = 0
        for g in range(num_groups):
            key = tuple(int(v) for v in uniq_vox[g].tolist())
            conf_mean = float(conf_mean_all[g].item())
            pos_mean = pos_mean_all[g]

            if key not in self.geo_voxel_bank:
                new_voxels += 1
                self.geo_voxel_bank[key] = {
                    "conf_ema": conf_mean,
                    "support": 1.0,
                    "last_seen": float(frame_idx),
                    "pos_x": float(pos_mean[0].item()),
                    "pos_y": float(pos_mean[1].item()),
                    "pos_z": float(pos_mean[2].item()),
                    "pos_var": 0.0,
                }
            else:
                item = self.geo_voxel_bank[key]
                prev_pos = torch.tensor([item["pos_x"], item["pos_y"], item["pos_z"]], dtype=pos_mean.dtype)
                drift2 = float(((pos_mean - prev_pos) ** 2).mean().item())
                item["conf_ema"] = (
                    self.geo_conf_ema_alpha * item["conf_ema"]
                    + (1.0 - self.geo_conf_ema_alpha) * conf_mean
                )
                item["support"] += 1.0
                item["last_seen"] = float(frame_idx)
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

        # Keep global bank bounded.
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
                keys_all = list(self.geo_voxel_bank.keys())
                n_all = len(keys_all)
                scan_size = max(
                    int(excess * self.geo_trim_drop_factor),
                    int(n_all * self.geo_trim_scan_ratio),
                    min(n_all, 1024),
                )
                scan_size = min(n_all, scan_size)

                start = int(self.geo_trim_cursor % max(1, n_all))
                stop = start + scan_size
                if stop <= n_all:
                    sample_keys = keys_all[start:stop]
                else:
                    sample_keys = keys_all[start:] + keys_all[: stop - n_all]
                self.geo_trim_cursor = (start + scan_size) % max(1, n_all)

                protected = set(self.geo_anchor_voxel_list)
                scored: List[Tuple[float, Tuple[int, int, int]]] = []
                for key in sample_keys:
                    if key in protected:
                        continue
                    scored.append((self._voxel_importance(self.geo_voxel_bank[key], frame_idx), key))

                if scored:
                    drop_n = min(excess, len(scored))
                    for _, key in heapq.nsmallest(drop_n, scored, key=lambda x: x[0]):
                        self.geo_voxel_bank.pop(key, None)

        if frame_idx % self.geo_anchor_refresh_interval == 0:
            self._refresh_geo_anchor_voxels(frame_idx)
            active_frames = self._active_frames_for_anchor_refresh(frame_idx)
            for fidx in active_frames:
                self._update_frame_anchor_mask(fidx)
        else:
            self._update_frame_anchor_mask(frame_idx)

        return {
            "new_voxel_ratio": float(new_voxels) / max(1.0, float(num_groups)),
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
        }

    @staticmethod
    def _protected_mask(meta: Dict[str, torch.Tensor], recent_frames: int) -> torch.Tensor:
        frame_idx = meta["frame_idx"]
        is_special = meta["is_special"]
        is_keyframe = meta.get("is_keyframe", torch.zeros_like(is_special))
        is_anchor = meta.get("is_anchor", torch.zeros_like(is_special))
        if frame_idx.numel() == 0:
            return torch.empty(0, dtype=torch.bool)
        current_frame_idx = int(frame_idx.max().item())
        recent_min = max(0, current_frame_idx - int(recent_frames))
        return is_special | is_keyframe | is_anchor | (frame_idx == 0) | (frame_idx >= recent_min)

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

        protected = special_keep | keyframe_keep | anchor_keep | (frame_keep == 0) | (frame_keep >= recent_min)
        prot_idx = keep[protected]
        if prot_idx.numel() >= budget:
            # O(n) truncation from tail: keep most recent protected tokens without full sort.
            return torch.unique(prot_idx[-budget:], sorted=True)

        remain = budget - int(prot_idx.numel())
        non_prot_idx = keep[~protected]
        if non_prot_idx.numel() > remain:
            non_prot_idx = non_prot_idx[-remain:]

        out = torch.cat([prot_idx, non_prot_idx], dim=0)
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
        is_keyframe = torch.full((tokens_per_frame,), bool(int(frame_idx) in self.geo_keyframe_set), dtype=torch.bool)

        local_patch_idx = torch.full((tokens_per_frame,), -1, dtype=torch.long)
        identity_local = torch.full((tokens_per_frame,), -1, dtype=torch.long)
        if special > 0:
            identity_local[:special] = -(torch.arange(special, dtype=torch.long) + 1)
        if patch_tokens > 0:
            local_patch_idx[special:] = torch.arange(patch_tokens, dtype=torch.long)
            identity_local[special:] = local_patch_idx[special:]
        global_id = frame_idx_t * int(self.geo_identity_stride) + (identity_local + int(self.geo_identity_offset))

        return {
            "frame_idx": frame_idx_t,
            "is_special": is_special,
            "is_keyframe": is_keyframe,
            "local_patch_idx": local_patch_idx,
            "identity_local": identity_local,
            "global_id": global_id,
            "is_anchor": torch.zeros((tokens_per_frame,), dtype=torch.bool),
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

        def _rank(i: int):
            f = int(frame[i].item())
            return (
                1 if bool(is_special[i].item()) else 0,
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
            return
        if "_gid_to_pos" in meta and int(meta.get("_gid_len", -1)) == gid_len:
            return

        gid_to_pos: Dict[int, int] = {}
        for i, g in enumerate(gid.detach().cpu().long().tolist()):
            gid_to_pos[int(g)] = i
        meta["_gid_to_pos"] = gid_to_pos
        meta["_gid_len"] = gid_len

    @staticmethod
    def _identity_keep_to_index(meta: Dict[str, torch.Tensor], identity_keep: torch.Tensor) -> torch.Tensor:
        if identity_keep is None or identity_keep.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        Aggregator._ensure_identity_lookup(meta)
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
        if current_frame_idx == 0:
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

        bin_count = min(int(self.geo_keyframe_time_bins), int(unique_frames.numel()), int(quota))
        frame_chunks = torch.chunk(unique_frames, bin_count)
        weights = torch.arange(1, bin_count + 1, dtype=torch.float32)
        raw_alloc = weights / weights.sum() * float(quota)
        alloc = torch.floor(raw_alloc).to(torch.long)
        alloc = torch.maximum(alloc, torch.ones_like(alloc))
        while int(alloc.sum().item()) > int(quota):
            for i in range(bin_count):
                if alloc[i] > 1 and int(alloc.sum().item()) > int(quota):
                    alloc[i] -= 1
        extra = int(quota - int(alloc.sum().item()))
        b = bin_count - 1
        while extra > 0:
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

        selected = set()
        frame_idx = meta["frame_idx"]
        is_special = meta["is_special"]
        is_keyframe = meta.get("is_keyframe", torch.zeros_like(is_special))
        local_idx = meta["local_patch_idx"]
        is_anchor = meta.get("is_anchor", torch.zeros_like(is_special))

        # Always keep special tokens.
        special_idx = torch.nonzero(is_special, as_tuple=False).flatten().tolist()
        selected.update(special_idx)

        # Keep previously established anchor tokens (pinning within cached KV).
        anchor_token_idx = torch.nonzero(is_anchor, as_tuple=False).flatten().tolist()
        selected.update(anchor_token_idx)

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
            return keep_fast

        # Build a budgeted local-tracking pool for recent patches (not all recent patches).
        if max_past_tokens is not None:
            recent_frames_count = max(1, int(torch.unique(frame_idx[recent_mask]).numel()))
            local_budget = min(
                int(max_past_tokens * self.geo_local_budget_ratio),
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

            bank_conf_t = torch.tensor(bank_conf_l, dtype=torch.float32)
            bank_support_t = torch.tensor(bank_support_l, dtype=torch.float32)
            bank_var_t = torch.tensor(bank_var_l, dtype=torch.float32)
            anchor_vox_t = torch.tensor(is_anchor_vox_l, dtype=torch.bool)

            conf_safe = conf.to(torch.float32).clamp_min(1e-6)
            bank_conf_eff = torch.where(bank_conf_t > 0, bank_conf_t, conf_safe)
            support_gain = torch.log1p(bank_support_t)
            stability = (1.0 / (1.0 + bank_var_t)).to(torch.float32)
            base_score = conf_safe * bank_conf_eff.clamp_min(1e-6) * support_gain * stability
            vis_weight = torch.where(
                visible,
                torch.ones_like(base_score),
                torch.where(
                    anchor_vox_t,
                    torch.full_like(base_score, float(self.geo_anchor_invisible_read_weight)),
                    torch.full_like(base_score, float(self.geo_invisible_read_weight)),
                ),
            )
            score_t = base_score * vis_weight

            gather_idx.append(in_frame.to(torch.long))
            gather_score.append(score_t)
            gather_bank_conf.append(bank_conf_eff)
            gather_voxel_hash.append(self._voxel_hash(vox))

        if not gather_idx:
            return torch.tensor(sorted(selected), dtype=torch.long)

        idx_all = torch.cat(gather_idx, dim=0)
        score_all = torch.cat(gather_score, dim=0)
        bank_conf_all = torch.cat(gather_bank_conf, dim=0)
        hash_all = torch.cat(gather_voxel_hash, dim=0)

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

            for vox in self.geo_anchor_voxel_list:
                h = int(self._voxel_hash(torch.tensor([vox], dtype=torch.long))[0].item())
                if h not in anchor_hash_to_best:
                    continue
                token = int(anchor_hash_to_best[h][1])
                selected.add(token)
                selected_global.add(token)
                anchor_count += 1
                if anchor_count >= anchor_quota:
                    break

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
            if max_past_tokens is not None:
                global_floor = max(1, int(max_past_tokens * self.geo_anchor_budget_ratio))
                global_budget = max(global_floor, left_budget)

            addable = [int(v) for v in grouped_idx.tolist() if int(v) not in selected_global]
            if addable and global_budget > 0:
                addable = addable[:global_budget]
                selected.update(addable)

        if not selected:
            return None

        logger.debug(
            "[geo_prune] total=%d candidate=%d visible=%d selected=%d anchor_selected=%d tau_bucket=%.4f",
            total_tokens,
            candidate_count,
            visible_total,
            len(selected),
            anchor_count,
            tau_bucket,
        )

        keep = torch.tensor(sorted(i for i in selected if 0 <= i < total_tokens), dtype=torch.long)
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
        if use_cache and use_geo_kv_prune and any(kv is not None for kv in past_key_values):
            ref_layer_idx = None
            for idx, kv in enumerate(past_key_values):
                if kv is not None and self.geo_token_meta[idx]["frame_idx"].numel() > 0:
                    ref_layer_idx = idx
                    break

            if ref_layer_idx is not None:
                ref_meta = self.geo_token_meta[ref_layer_idx]
                ref_budget = max(0, int(current_budgets.max().item()) - P)
                geo_shared_keep_idx = self._select_geo_active_indices(
                    meta=ref_meta,
                    topk_per_voxel=geo_topk_per_voxel,
                    recent_frames=geo_recent_frames,
                    near=geo_near,
                    far=geo_far,
                    current_view=current_view,
                    max_past_tokens=ref_budget,
                )
                geo_shared_identity_keep = self._build_identity_keep_from_meta(ref_meta, geo_shared_keep_idx)

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if use_cache:
                        layer_idx = global_idx
                        layer_budget = int(current_budgets[layer_idx].item())
                        past_kv_block = past_key_values[layer_idx] if past_key_values[layer_idx] is not None else None
                        past_meta = self.geo_token_meta[layer_idx]

                        if use_geo_kv_prune and past_kv_block is not None:
                            layer_identity_keep = self._cap_identity_keep_with_protection(
                                past_meta,
                                geo_shared_identity_keep
                                if geo_shared_identity_keep is not None
                                else torch.empty((0,), dtype=torch.long),
                                budget=max(0, layer_budget - P),
                                recent_frames=geo_recent_frames,
                            )

                            keep_idx = self._identity_keep_to_index(past_meta, layer_identity_keep)
                            keep_idx = self._sanitize_keep_idx(
                                keep_idx,
                                meta_len=past_meta["frame_idx"].numel(),
                                kv_len=past_kv_block[0].shape[2],
                            )

                            # Hard pre-attention cap with protection for anchor/special/recent tokens.
                            max_past_tokens = max(0, layer_budget - P)
                            if keep_idx.numel() > 0:
                                pre_keep_all = keep_idx
                            else:
                                pre_keep_all = torch.arange(past_kv_block[0].shape[2], dtype=torch.long)
                            pre_keep = self._cap_keep_with_protection(
                                past_meta,
                                pre_keep_all,
                                budget=max_past_tokens,
                                recent_frames=geo_recent_frames,
                            )
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
