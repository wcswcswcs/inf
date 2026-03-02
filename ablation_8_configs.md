# 8组消融实验配置表（仅推理并保存 .pth）

> 基于当前 `run_inference.py` 暴露的 aggregator 参数，下面 8 组用于快速定位主要收益来源。

| 组别 | 目标 | 关键改动（相对默认） | 额外 CLI 参数 |
|---|---|---|---|
| E0_baseline | 默认改进版基线 | 不改动 | 无 |
| E1_no_vis_penalty | 去掉可见性惩罚 | `geo_invisible_read_weight=1.0`, `geo_anchor_invisible_read_weight=1.0` | `--geo_invisible_read_weight 1.0 --geo_anchor_invisible_read_weight 1.0` |
| E2_strong_recent | 加强 recent 保留 | `geo_local_budget_ratio=1.0`, `geo_local_budget_cap_per_frame=2000` | `--geo_local_budget_ratio 1.0 --geo_local_budget_cap_per_frame 2000` |
| E3_weak_recent | 减弱 recent 保留 | `geo_local_budget_ratio=0.25`, `geo_local_budget_cap_per_frame=400` | `--geo_local_budget_ratio 0.25 --geo_local_budget_cap_per_frame 400` |
| E4_weak_frame0 | 弱化 frame0 保护 | `geo_frame0_patch_cap=512` | `--geo_frame0_patch_cap 512` |
| E5_disable_anchor_read | 禁用 anchor read | `geo_anchor_read_quota=0` | `--geo_anchor_read_quota 0` |
| E6_bucket_loose | 放宽 bucket | `geo_bucket_quantile_target=0.35` | `--geo_bucket_quantile_target 0.35` |
| E7_bucket_strict | 收紧 bucket | `geo_bucket_quantile_target=0.8` | `--geo_bucket_quantile_target 0.8` |

默认值来源于 `Aggregator.__init__` 现有参数默认配置。
