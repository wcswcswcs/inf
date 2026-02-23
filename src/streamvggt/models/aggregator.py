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
        self.reset_geo_cache_state()

    def reset_geo_cache_state(self):
        self.geo_frame_meta: Dict[int, Dict[str, Any]] = {}
        self.geo_max_frame_idx = -1
        self.geo_token_meta: Dict[int, Dict[str, torch.Tensor]] = {
            i: {
                "frame_idx": torch.empty(0, dtype=torch.long),
                "is_special": torch.empty(0, dtype=torch.bool),
                "local_patch_idx": torch.empty(0, dtype=torch.long),
            }
            for i in range(self.depth)
        }

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

        voxel_ids = torch.round(pts_flat / max(voxel_size, 1e-6)).to(torch.int32)
        meta = {
            "pts": pts_flat,
            "conf": conf_flat,
            "voxel_ids": voxel_ids,
            "world_to_cam": world_to_cam.detach().cpu() if world_to_cam is not None else None,
            "intrinsic": intrinsic.detach().cpu() if intrinsic is not None else None,
        }
        self.geo_frame_meta[frame_idx] = meta
        self.geo_max_frame_idx = max(self.geo_max_frame_idx, frame_idx)

    @staticmethod
    def _frustum_mask(pts: torch.Tensor, world_to_cam: torch.Tensor, intrinsic: torch.Tensor, near: float, far: float):
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
        z = cam[:, 2]
        valid_z = (z > near) & (z < far)

        uv_h = cam @ intrinsic.t()         # [N, 3]
        u = uv_h[:, 0] / (uv_h[:, 2].clamp_min(1e-6))
        v = uv_h[:, 1] / (uv_h[:, 2].clamp_min(1e-6))
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
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
        }

    @staticmethod
    def _concat_meta(meta_a: Dict[str, torch.Tensor], meta_b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "frame_idx": torch.cat([meta_a["frame_idx"], meta_b["frame_idx"]], dim=0),
            "is_special": torch.cat([meta_a["is_special"], meta_b["is_special"]], dim=0),
            "local_patch_idx": torch.cat([meta_a["local_patch_idx"], meta_b["local_patch_idx"]], dim=0),
        }

    def _build_current_frame_meta(self, frame_idx: int, tokens_per_frame: int) -> Dict[str, torch.Tensor]:
        special = self.patch_start_idx
        patch_tokens = max(tokens_per_frame - special, 0)

        frame_idx_t = torch.full((tokens_per_frame,), int(frame_idx), dtype=torch.long)
        is_special = torch.zeros((tokens_per_frame,), dtype=torch.bool)
        is_special[:special] = True

        local_patch_idx = torch.full((tokens_per_frame,), -1, dtype=torch.long)
        if patch_tokens > 0:
            local_patch_idx[special:] = torch.arange(patch_tokens, dtype=torch.long)

        return {
            "frame_idx": frame_idx_t,
            "is_special": is_special,
            "local_patch_idx": local_patch_idx,
        }

    def _select_geo_active_indices(
        self,
        meta: Dict[str, torch.Tensor],
        topk_per_voxel: int,
        recent_frames: int,
        near: float,
        far: float,
        current_view: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        total_tokens = int(meta["frame_idx"].numel())
        if total_tokens == 0:
            return None

        selected = set()
        frame_idx = meta["frame_idx"]
        is_special = meta["is_special"]
        local_idx = meta["local_patch_idx"]

        # Always keep special tokens
        special_idx = torch.nonzero(is_special, as_tuple=False).flatten().tolist()
        selected.update(special_idx)

        current_frame_idx = int(frame_idx.max().item()) if frame_idx.numel() > 0 else 0
        recent_min = max(0, current_frame_idx - int(recent_frames))
        recent_mask = frame_idx >= recent_min
        recent_idx = torch.nonzero(recent_mask, as_tuple=False).flatten().tolist()
        selected.update(recent_idx)

        if current_view is None or current_view.get("world_to_cam") is None or current_view.get("intrinsic") is None:
            return torch.tensor(sorted(selected), dtype=torch.long)

        world_to_cam = current_view["world_to_cam"].detach().cpu()
        intrinsic = current_view["intrinsic"].detach().cpu()
        if world_to_cam.ndim == 3:
            world_to_cam = world_to_cam[0]
        if intrinsic.ndim == 3:
            intrinsic = intrinsic[0]

        # Candidates for geometry-based pruning: non-special and non-recent tokens
        candidate_mask = (~is_special) & (~recent_mask) & (local_idx >= 0)
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            return torch.tensor(sorted(selected), dtype=torch.long)

        # Global per-voxel top-k across all old frames (avoid per-frame duplicates)
        bucket: Dict[Tuple[int, int, int], List[Tuple[float, int]]] = defaultdict(list)

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
            )
            if visible.sum().item() == 0:
                continue

            vis_global = in_frame[visible]
            vis_conf = conf[visible]
            vis_vox = vox[visible]

            for j in range(vis_global.numel()):
                gidx = int(vis_global[j].item())
                key = tuple(int(v) for v in vis_vox[j].tolist())
                bucket[key].append((float(vis_conf[j].item()), gidx))

        for _, entries in bucket.items():
            entries.sort(key=lambda x: x[0], reverse=True)
            for _, gidx in entries[: max(1, int(topk_per_voxel))]:
                selected.add(gidx)

        if not selected:
            return None

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
            camera_token_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            camera_token = camera_token_full[-1:, :, :]

            register_token_full = slice_expand_and_flatten(self.register_token, B, S_true)
            register_token = register_token_full[-1:, :, :]
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
                            keep_idx = self._select_geo_active_indices(
                                meta=past_meta,
                                topk_per_voxel=geo_topk_per_voxel,
                                recent_frames=geo_recent_frames,
                                near=geo_near,
                                far=geo_far,
                                current_view=current_view,
                            )
                            if keep_idx is not None and keep_idx.numel() > 0:
                                keep_idx_dev = keep_idx.to(past_kv_block[0].device)
                                past_kv_block = (
                                    torch.index_select(past_kv_block[0], 2, keep_idx_dev),
                                    torch.index_select(past_kv_block[1], 2, keep_idx_dev),
                                )
                                past_meta = self._index_meta(past_meta, keep_idx)

                            # Hard pre-attention cap so this layer cannot exceed budget in current forward.
                            max_past_tokens = max(0, layer_budget - P)
                            if max_past_tokens == 0:
                                past_kv_block = (past_kv_block[0][:, :, :0], past_kv_block[1][:, :, :0])
                                past_meta = self._index_meta(past_meta, torch.empty(0, dtype=torch.long))
                            elif past_kv_block[0].shape[2] > max_past_tokens:
                                start = past_kv_block[0].shape[2] - max_past_tokens
                                pre_keep = torch.arange(start, past_kv_block[0].shape[2], dtype=torch.long, device=past_kv_block[0].device)
                                past_kv_block = (
                                    torch.index_select(past_kv_block[0], 2, pre_keep),
                                    torch.index_select(past_kv_block[1], 2, pre_keep),
                                )
                                past_meta = self._index_meta(past_meta, pre_keep.cpu())

                        tokens, global_idx, global_intermediates, new_kv, current_scores = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_kv_block,
                            use_cache=True,
                            past_frame_idx=past_frame_idx,
                            cache_budget=None if use_geo_kv_prune else layer_budget
                        )

                        if use_geo_kv_prune:
                            current_meta = self._build_current_frame_meta(past_frame_idx, P)
                            merged_meta = self._concat_meta(past_meta, current_meta)

                            # Keep explicit hard cap in geo mode (post-attention safety net).
                            if new_kv[0].shape[2] > layer_budget:
                                start = new_kv[0].shape[2] - layer_budget
                                cap_keep = torch.arange(start, new_kv[0].shape[2], dtype=torch.long, device=new_kv[0].device)
                                new_kv = (
                                    torch.index_select(new_kv[0], 2, cap_keep),
                                    torch.index_select(new_kv[1], 2, cap_keep),
                                )
                                merged_meta = self._index_meta(merged_meta, cap_keep.cpu())
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
