import copy

import torch

from streamvggt.models.aggregator import Aggregator


def _make_agg():
    return Aggregator(depth=2, embed_dim=64, num_heads=4, patch_embed="conv", img_size=28, patch_size=14)


def _make_images():
    return torch.randn(1, 1, 3, 28, 28)


def _make_meta(frame_idx: int):
    return {
        "frame_idx": torch.tensor([frame_idx, frame_idx, frame_idx, frame_idx], dtype=torch.long),
        "is_special": torch.tensor([True, False, False, False], dtype=torch.bool),
        "is_keyframe": torch.tensor([False, False, False, False], dtype=torch.bool),
        "local_patch_idx": torch.tensor([-1, 0, 1, 2], dtype=torch.long),
        "is_anchor": torch.tensor([False, False, False, False], dtype=torch.bool),
        "is_landmark": torch.tensor([False, False, False, False], dtype=torch.bool),
        "is_reference": torch.tensor([False, False, False, False], dtype=torch.bool),
        "identity_local": torch.tensor([-1, 0, 1, 2], dtype=torch.long),
        "global_id": torch.tensor([1, 2, 3, 4], dtype=torch.long),
    }


def test_peek_is_read_only():
    agg = _make_agg()
    before = {
        "mode": agg.geo_selector_mode,
        "ema": (
            agg.geo_maturity_ema,
            agg.geo_instability_ema,
            agg.geo_pressure_ema,
            agg.geo_motion_ema,
            agg.geo_confdrop_ema,
            agg.geo_matched_ema,
            agg.geo_new_voxel_ema,
            agg.geo_ref_overlap_ema,
            agg.geo_selector_overlap_ema,
            agg.geo_selector_visible_ratio_ema,
        ),
        "streak": (
            agg.geo_handover_ready_streak,
            agg.geo_handover_unready_streak,
            agg.geo_recovery_enter_streak,
            agg.geo_recovery_exit_streak,
        ),
    }
    _ = agg._geo_peek_adaptive_policy(
        frame_idx=3,
        total_tokens=128,
        max_past_tokens=256,
        current_view={"pose_delta": 0.1, "conf_drop": 0.2},
        observation={"frame_idx": 2, "matched_ratio": 0.3, "new_voxel_ratio": 0.1, "ref_overlap": 5.0, "trust_score": 0.9},
        selector_diag={"frame_idx": 2, "stable_visible_overlap": 6.0, "stable_visible_ratio": 0.7},
    )
    after = {
        "mode": agg.geo_selector_mode,
        "ema": (
            agg.geo_maturity_ema,
            agg.geo_instability_ema,
            agg.geo_pressure_ema,
            agg.geo_motion_ema,
            agg.geo_confdrop_ema,
            agg.geo_matched_ema,
            agg.geo_new_voxel_ema,
            agg.geo_ref_overlap_ema,
            agg.geo_selector_overlap_ema,
            agg.geo_selector_visible_ratio_ema,
        ),
        "streak": (
            agg.geo_handover_ready_streak,
            agg.geo_handover_unready_streak,
            agg.geo_recovery_enter_streak,
            agg.geo_recovery_exit_streak,
        ),
    }
    assert before == after


def test_commit_once_per_frame():
    agg = _make_agg()
    obs = {"frame_idx": 4, "matched_ratio": 0.4, "new_voxel_ratio": 0.1, "ref_overlap": 8.0, "trust_score": 0.9}
    diag = {"frame_idx": 4, "stable_visible_overlap": 10.0, "stable_visible_ratio": 0.8}
    p1 = agg._geo_commit_adaptive_policy_once(
        frame_idx=5,
        total_tokens=512,
        max_past_tokens=1024,
        current_view={"pose_delta": 0.2, "conf_drop": 0.1},
        observation=obs,
        selector_diag=diag,
    )
    state_after_first = (
        agg.geo_maturity_ema,
        agg.geo_instability_ema,
        agg.geo_pressure_ema,
        agg.geo_selector_mode,
        agg.geo_handover_ready_streak,
    )
    p2 = agg._geo_commit_adaptive_policy_once(
        frame_idx=5,
        total_tokens=100,
        max_past_tokens=200,
        current_view={"pose_delta": 1.0, "conf_drop": 1.0},
        observation={"frame_idx": 0, "matched_ratio": 0.0, "new_voxel_ratio": 1.0, "ref_overlap": 0.0, "trust_score": 0.0},
        selector_diag={"frame_idx": 0, "stable_visible_overlap": 0.0, "stable_visible_ratio": 0.0},
    )
    state_after_second = (
        agg.geo_maturity_ema,
        agg.geo_instability_ema,
        agg.geo_pressure_ema,
        agg.geo_selector_mode,
        agg.geo_handover_ready_streak,
    )
    assert p1 == p2
    assert state_after_first == state_after_second


def test_update_geo_frame_metadata_does_not_advance_controller():
    agg = _make_agg()
    before = (
        agg.geo_maturity_ema,
        agg.geo_instability_ema,
        agg.geo_selector_mode,
        agg.geo_handover_ready_streak,
    )
    pts = torch.randn(1, 28, 28, 3)
    conf = torch.rand(1, 28, 28)
    w2c = torch.eye(4).unsqueeze(0)
    K = torch.eye(3).unsqueeze(0)
    stats = agg.update_geo_frame_metadata(frame_idx=0, pts3d=pts, conf=conf, world_to_cam=w2c, intrinsic=K, voxel_size=0.1)
    after = (
        agg.geo_maturity_ema,
        agg.geo_instability_ema,
        agg.geo_selector_mode,
        agg.geo_handover_ready_streak,
    )
    assert before == after
    assert int(agg.geo_last_observation["frame_idx"]) == 0
    assert abs(float(stats["matched_ratio"]) - float(agg.geo_last_observation["matched_ratio"])) < 1e-6


def test_selector_diag_updates_every_frame_not_log_interval_bound():
    agg = _make_agg()
    agg.geo_console_log_interval = 50
    for f in [1, 2, 3]:
        _ = agg._select_geo_active_indices_legacy_early(
            meta=_make_meta(frame_idx=f),
            topk_per_voxel=1,
            recent_frames=4,
            near=0.1,
            far=10.0,
            current_view=None,
            max_past_tokens=8,
            policy=None,
        )
        assert int(agg.geo_last_selector_diag["frame_idx"]) == f


def test_forward_without_cache_context_does_not_commit():
    agg = _make_agg()
    images = _make_images()
    _ = agg(
        images,
        past_key_values=[None] * agg.depth,
        use_cache=True,
        past_frame_idx=0,
        use_geo_kv_prune=True,
    )
    assert int(agg.geo_last_policy_frame) == -1


def test_forward_with_cache_context_commits():
    agg = _make_agg()
    images = _make_images()
    out = agg(images, past_key_values=[None] * agg.depth, use_cache=True, past_frame_idx=0, use_geo_kv_prune=True, total_budget=512)
    past = out[2]
    _ = agg(images, past_key_values=past, use_cache=True, past_frame_idx=1, use_geo_kv_prune=True, total_budget=512)
    assert int(agg.geo_last_policy_frame) >= 0


def test_causal_observation_delay():
    agg = _make_agg()
    images = _make_images()
    out = agg(images, past_key_values=[None] * agg.depth, use_cache=True, past_frame_idx=0, use_geo_kv_prune=True, total_budget=512)
    past = out[2]

    pts = torch.randn(1, 28, 28, 3)
    conf = torch.rand(1, 28, 28)
    w2c = torch.eye(4).unsqueeze(0)
    K = torch.eye(3).unsqueeze(0)
    _ = agg.update_geo_frame_metadata(frame_idx=0, pts3d=pts, conf=conf, world_to_cam=w2c, intrinsic=K, voxel_size=0.1)

    _ = agg(images, past_key_values=past, use_cache=True, past_frame_idx=1, use_geo_kv_prune=True, total_budget=512)
    assert int(agg.geo_last_policy_inputs.get("observation_frame", -1)) == 0
