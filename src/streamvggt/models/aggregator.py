# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
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
        geo_selection_interval: int = 1,
        geo_anchor_refresh_interval: int = 1,
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
        self.geo_identity_stride = 1 << 21
        self.geo_identity_offset = 1 << 18
        self.reset_geo_cache_state()

    def reset_geo_cache_state(self):
        self.geo_frame_meta: Dict[int, Dict[str, Any]] = {}
        self.geo_max_frame_idx = -1
        self.geo_voxel_bank: Dict[Tuple[int, int, int], Dict[str, float]] = {}
        self.geo_anchor_voxels: set[Tuple[int, int, int]] = set()
        self.geo_anchor_voxel_list: List[Tuple[int, int, int]] = []
        self.geo_token_meta: Dict[int, Dict[str, torch.Tensor]] = {
            i: {
                "frame_idx": torch.empty(0, dtype=torch.long),
                "is_special": torch.empty(0, dtype=torch.bool),
                "local_patch_idx": torch.empty(0, dtype=torch.long),
                "identity_local": torch.empty(0, dtype=torch.long),
                "global_id": torch.empty(0, dtype=torch.long),
                "is_anchor": torch.empty(0, dtype=torch.bool),
            }
            for i in range(self.depth)
        }

    def _voxel_importance(self, item: Dict[str, float], now_frame_idx: int) -> float:
        age = max(0.0, float(now_frame_idx) - float(item["last_seen"]))
        return (
            float(item["conf_ema"])
            * torch.log1p(torch.tensor(float(item["support"]))).item()
            * (1.0 / (1.0 + float(item["pos_var"])))
            * (1.0 / (1.0 + 0.05 * age))
        )

    def _refresh_geo_anchor_voxels(self, now_frame_idx: int):
        if not self.geo_voxel_bank:
            self.geo_anchor_voxels = set()
            self.geo_anchor_voxel_list = []
            return

        prev_anchors = self.geo_anchor_voxels
        ranked = []
        for key, item in self.geo_voxel_bank.items():
            conf_ema = float(item["conf_ema"])
            support = float(item["support"])
            pos_var = float(item["pos_var"])
            in_prev = key in prev_anchors
            conf_ok = conf_ema >= (self.geo_anchor_conf_exit if in_prev else self.geo_anchor_conf_enter)
            support_ok = support >= self.geo_anchor_min_support
            var_ok = pos_var <= self.geo_anchor_max_pos_var
            if conf_ok and support_ok and var_ok:
                ranked.append((self._voxel_importance(item, now_frame_idx), key))

        ranked.sort(key=lambda x: (-x[0], x[1]))
        self.geo_anchor_voxel_list = [k for _, k in ranked[: self.geo_anchor_voxel_budget]]
        self.geo_anchor_voxels = set(self.geo_anchor_voxel_list)

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
            frame_meta = self.geo_frame_meta.get(fidx)
            if frame_meta is None:
                continue

            mask_f = valid & (frame_idx == fidx)
            idx_global = torch.nonzero(mask_f, as_tuple=False).flatten()
            local_f = local_idx.index_select(0, idx_global).long()
            in_range = (local_f >= 0) & (local_f < frame_meta["voxel_ids"].shape[0])
            if in_range.sum().item() == 0:
                continue

            idx_global = idx_global[in_range]
            local_f = local_f[in_range]
            vox = frame_meta["voxel_ids"].index_select(0, local_f)

            anchor_mask = torch.tensor(
                [tuple(int(v) for v in row.tolist()) in self.geo_anchor_voxels for row in vox],
                dtype=torch.bool,
            )
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
    ):
        if pts3d is None or conf is None:
            return
        if pts3d.ndim != 4 or conf.ndim != 3:
            return

        _, H, W, _ = pts3d.shape
        gh, gw = H // self.patch_size, W // self.patch_size
        if gh <= 0 or gw <= 0:
            return

        pts_patch = F.adaptive_avg_pool2d(pts3d.permute(0, 3, 1, 2), (gh, gw)).permute(0, 2, 3, 1)
        conf_patch = F.adaptive_avg_pool2d(conf.unsqueeze(1), (gh, gw)).squeeze(1)

        pts_flat = pts_patch.reshape(-1, 3).detach().cpu()
        conf_flat = conf_patch.reshape(-1).detach().cpu()

        voxel_ids = torch.floor(pts_flat / max(voxel_size, 1e-6)).to(torch.int32)
        meta = {
            "pts": pts_flat,
            "conf": conf_flat,
            "voxel_ids": voxel_ids,
            "world_to_cam": world_to_cam.detach().cpu() if world_to_cam is not None else None,
            "intrinsic": intrinsic.detach().cpu() if intrinsic is not None else None,
        }
        self.geo_frame_meta[frame_idx] = meta
        self.geo_max_frame_idx = max(self.geo_max_frame_idx, frame_idx)

        # Update global voxel landmark bank (conf/support/stability/recency)
        voxel_to_idx: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        for i in range(voxel_ids.shape[0]):
            key = tuple(int(v) for v in voxel_ids[i].tolist())
            voxel_to_idx[key].append(i)

        for key, idxs in voxel_to_idx.items():
            pts_cur = pts_flat.index_select(0, torch.tensor(idxs, dtype=torch.long))
            conf_cur = conf_flat.index_select(0, torch.tensor(idxs, dtype=torch.long))
            conf_mean = float(conf_cur.mean().item())
            pos_mean = pts_cur.mean(dim=0)

            if key not in self.geo_voxel_bank:
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
        if len(self.geo_voxel_bank) > self.geo_max_voxels:
            items = []
            for key, val in self.geo_voxel_bank.items():
                importance = self._voxel_importance(val, frame_idx)
                items.append((importance, key))

            items.sort(key=lambda x: x[0], reverse=True)
            keep_keys = set(k for _, k in items[: self.geo_max_voxels])
            self.geo_voxel_bank = {k: v for k, v in self.geo_voxel_bank.items() if k in keep_keys}

        if frame_idx % self.geo_anchor_refresh_interval == 0:
            self._refresh_geo_anchor_voxels(frame_idx)

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
        is_anchor = meta.get("is_anchor", torch.zeros_like(is_special))
        if frame_idx.numel() == 0:
            return torch.empty(0, dtype=torch.bool)
        current_frame_idx = int(frame_idx.max().item())
        recent_min = max(0, current_frame_idx - int(recent_frames))
        return is_special | is_anchor | (frame_idx == 0) | (frame_idx >= recent_min)

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
        anchor_keep = is_anchor_all.index_select(0, keep)

        protected = special_keep | anchor_keep | (frame_keep == 0) | (frame_keep >= recent_min)
        prot_idx = keep[protected]
        if prot_idx.numel() >= budget:
            # Keep the most recent protected tokens under strict hard budget.
            prot_frame = frame_idx_all.index_select(0, prot_idx)
            order = torch.argsort(prot_frame, descending=True)
            return torch.unique(prot_idx.index_select(0, order[:budget]), sorted=True)

        remain = budget - int(prot_idx.numel())
        non_prot_idx = keep[~protected]
        if non_prot_idx.numel() > remain:
            non_prot_frame = frame_idx_all.index_select(0, non_prot_idx)
            order = torch.argsort(non_prot_frame, descending=True)
            non_prot_idx = non_prot_idx.index_select(0, order[:remain])

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

        capped_identities = set(self._build_identity_keep_from_meta(meta, idx).tolist())
        out = [int(key) for key in identity_keep.detach().cpu().long().tolist() if int(key) in capped_identities]
        if not out:
            return torch.empty((0,), dtype=torch.long)
        return torch.tensor(out, dtype=torch.long)

    @staticmethod
    def _ensure_identity_lookup(meta: Dict[str, torch.Tensor]):
        gid = meta.get("global_id", torch.empty((0,), dtype=torch.long))
        if gid.numel() == 0:
            meta["_gid_sorted"] = torch.empty((0,), dtype=torch.long)
            meta["_gid_pos"] = torch.empty((0,), dtype=torch.long)
            return
        if "_gid_sorted" in meta and "_gid_pos" in meta and meta["_gid_sorted"].numel() == gid.numel():
            return
        order = torch.argsort(gid)
        meta["_gid_sorted"] = gid.index_select(0, order)
        meta["_gid_pos"] = order

    @staticmethod
    def _identity_keep_to_index(meta: Dict[str, torch.Tensor], identity_keep: torch.Tensor) -> torch.Tensor:
        if identity_keep is None or identity_keep.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        Aggregator._ensure_identity_lookup(meta)
        gid_sorted = meta.get("_gid_sorted", torch.empty((0,), dtype=torch.long))
        gid_pos = meta.get("_gid_pos", torch.empty((0,), dtype=torch.long))
        if gid_sorted.numel() == 0:
            return torch.empty(0, dtype=torch.long)
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
        return gid_pos.index_select(0, where[matched])

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

        current_frame_idx = int(frame_idx.max().item()) if frame_idx.numel() > 0 else 0
        recent_min = max(0, current_frame_idx - int(recent_frames))
        recent_mask = frame_idx >= recent_min

        # Optional fast-path (disabled by default, can be enabled by setting geo_selection_interval>1).
        if self.geo_selection_interval > 1 and (current_frame_idx % self.geo_selection_interval != 0):
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
                keep_frames = uniq.sort().values[-self.geo_max_old_frames_to_score :]
                keep_mask = (frame_idx[candidate_indices].unsqueeze(1) == keep_frames.unsqueeze(0)).any(dim=1)
                candidate_indices = candidate_indices[keep_mask]

        # Acceleration guard 2: cap candidate token count per layer.
        if self.geo_max_candidate_tokens > 0 and candidate_indices.numel() > self.geo_max_candidate_tokens:
            cf = frame_idx[candidate_indices]
            order = torch.argsort(cf)
            candidate_indices = candidate_indices.index_select(0, order)[-self.geo_max_candidate_tokens :]

        # Global per-voxel top-k across all old frames (avoid per-frame duplicates)
        bucket: Dict[Tuple[int, int, int], List[Tuple[float, int]]] = defaultdict(list)
        bucket_conf_proxy: List[float] = []
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
            visible_count = int(visible.sum().item())
            visible_total += visible_count
            for j in range(in_frame.numel()):
                gidx = int(in_frame[j].item())
                key = tuple(int(v) for v in vox[j].tolist())

                bank = self.geo_voxel_bank.get(key)
                if bank is None:
                    bank_conf = float(conf[j].item())
                    bank_support = 1.0
                    bank_var = 0.0
                else:
                    bank_conf = float(bank["conf_ema"])
                    bank_support = float(bank["support"])
                    bank_var = float(bank["pos_var"])
                bucket_conf_proxy.append(bank_conf)

                stability = 1.0 / (1.0 + bank_var)
                support_gain = torch.log1p(torch.tensor(bank_support)).item()
                base_score = max(float(conf[j].item()), 1e-6) * max(bank_conf, 1e-6) * support_gain * stability
                if bool(visible[j].item()):
                    vis_weight = 1.0
                else:
                    vis_weight = (
                        self.geo_anchor_invisible_read_weight
                        if key in self.geo_anchor_voxels
                        else self.geo_invisible_read_weight
                    )
                score = base_score * vis_weight

                # bucket threshold is adaptive to avoid empty/overcrowded bucket.
                bucket[key].append((score, gidx, bank_conf))

        # Adaptive bucket threshold to control bucket size.
        remaining_budget = None
        if max_past_tokens is not None:
            remaining_budget = max(0, int(max_past_tokens) - len(selected))
        tau_bucket = self._compute_dynamic_bucket_threshold(bucket_conf_proxy, remaining_budget or 0)

        # Global anchor quota from ordered anchor list (deterministic).
        anchor_count = 0
        if self.geo_anchor_voxel_list:
            anchor_quota = int(self.geo_anchor_read_quota)
            if max_past_tokens is not None:
                anchor_quota = min(anchor_quota, max(0, int(max_past_tokens * self.geo_anchor_budget_ratio)))
            for vox in self.geo_anchor_voxel_list:
                if vox not in bucket:
                    continue
                entries = [e for e in bucket[vox] if e[2] >= self.geo_anchor_conf_exit]
                if not entries:
                    continue
                entries.sort(key=lambda x: x[0], reverse=True)
                selected.add(entries[0][1])
                anchor_count += 1
                if anchor_count >= anchor_quota:
                    break

        for _, entries in bucket.items():
            entries = [e for e in entries if e[2] >= tau_bucket]
            entries.sort(key=lambda x: x[0], reverse=True)
            for _, gidx, _ in entries[: max(1, int(topk_per_voxel))]:
                selected.add(gidx)

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
