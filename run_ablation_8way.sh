#!/usr/bin/env bash
set -uo pipefail
shopt -s nullglob

# ===== 可改参数 =====
GPU_IDS=(0 1 2 3)                # 可用 GPU 列表
CHECKPOINT="./ckpt/checkpoints.pth"
DATA_ROOT="Long3D"               # 目录结构: Long3D/<scene>/images/scan_images
OUT_DIR="./ablation_outs"
LOG_DIR="./ablation_logs"
MAX_VIEWS="-1"                   # -1 表示全部视角
TOTAL_BUDGET="1200000"
SKIP_EXISTING=1

# 基础 geo 参数（所有实验共享）
BASE_GEO_ARGS=(
  --use_geo_kv_prune
  --total_budget "$TOTAL_BUDGET"
  --geo_voxel_size 0.2
  --geo_topk_per_voxel 4
  --geo_recent_frames 2
  --geo_near 0.05
  --geo_far 200.0
)

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: data root not found: $DATA_ROOT"
  exit 1
fi

sanitize_name() {
  local s="$1"
  s="${s// /_}"
  s="${s//\//_}"
  echo "$s"
}

is_alive() {
  local pid="$1"
  [[ "$pid" -ne 0 ]] && kill -0 "$pid" 2>/dev/null
}

log_line() {
  local prefix="$1"
  local log_path="$2"
  local msg="$3"
  printf '%s%s\n' "$prefix" "$msg" | tee -a "$log_path"
}

# ===== 8组实验定义 =====
EXP_NAMES=(
  "E0_baseline"
  "E1_no_vis_penalty"
  "E2_strong_recent"
  "E3_weak_recent"
  "E4_weak_frame0"
  "E5_disable_anchor_read"
  "E6_bucket_loose"
  "E7_bucket_strict"
)

run_job() {
  local gpu="$1"
  local scene_path="$2"
  local scene_name="$3"
  local exp_name="$4"

  local scene_id input_dir out_path log_path prefix
  local exp_args=()

  case "$exp_name" in
    E0_baseline) exp_args=() ;;
    E1_no_vis_penalty) exp_args=(--geo_invisible_read_weight 1.0 --geo_anchor_invisible_read_weight 1.0) ;;
    E2_strong_recent) exp_args=(--geo_local_budget_ratio 1.0 --geo_local_budget_cap_per_frame 2000) ;;
    E3_weak_recent) exp_args=(--geo_local_budget_ratio 0.25 --geo_local_budget_cap_per_frame 400) ;;
    E4_weak_frame0) exp_args=(--geo_frame0_patch_cap 512) ;;
    E5_disable_anchor_read) exp_args=(--geo_anchor_read_quota 0) ;;
    E6_bucket_loose) exp_args=(--geo_bucket_quantile_target 0.35) ;;
    E7_bucket_strict) exp_args=(--geo_bucket_quantile_target 0.8) ;;
    *)
      echo "[GPU${gpu}|${scene_name}|${exp_name}] ERROR: unknown experiment"
      return 3
      ;;
  esac

  scene_id="$(sanitize_name "$scene_name")"
  input_dir="${scene_path%/}/images/scan_images"
  out_path="${OUT_DIR}/${scene_id}_${exp_name}.pth"
  log_path="${LOG_DIR}/${scene_id}_${exp_name}.log"
  prefix="[GPU${gpu}|${scene_id}|${exp_name}] "

  if [[ ! -d "$input_dir" ]]; then
    echo "$prefix ERROR: input dir not found: $input_dir"
    return 2
  fi

  if [[ "$SKIP_EXISTING" -eq 1 && -f "$out_path" ]]; then
    echo "$prefix SKIP (exists): $out_path"
    return 0
  fi

  : > "$log_path"
  log_line "$prefix" "$log_path" "Time: $(date)"
  log_line "$prefix" "$log_path" "Input: $input_dir"
  log_line "$prefix" "$log_path" "Output: $out_path"
  log_line "$prefix" "$log_path" "Args: ${exp_args[*]}"

  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" python -u run_inference.py \
    --input_dir "$input_dir" \
    --checkpoint_path "$CHECKPOINT" \
    --output_path "$out_path" \
    --max_views "$MAX_VIEWS" \
    "${BASE_GEO_ARGS[@]}" \
    "${exp_args[@]}" 2>&1 \
  | while IFS= read -r line; do
      printf '%s%s\n' "$prefix" "$line"
    done \
  | tee -a "$log_path"

  local py_rc tee_rc
  py_rc="${PIPESTATUS[0]}"
  tee_rc="${PIPESTATUS[2]}"

  if [[ "$tee_rc" -ne 0 ]]; then
    log_line "$prefix" "$log_path" "ERROR: tee failed rc=$tee_rc"
    return "$tee_rc"
  fi

  log_line "$prefix" "$log_path" "EXIT rc=$py_rc"
  return "$py_rc"
}

SCENE_DIRS=( "$DATA_ROOT"/*/ )
if [[ ${#SCENE_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: no scenes under $DATA_ROOT"
  exit 1
fi

declare -a JOB_SCENE_PATH JOB_SCENE_NAME JOB_EXP_NAME
for scene_path in "${SCENE_DIRS[@]}"; do
  [[ -d "$scene_path" ]] || continue
  scene_name="$(basename "${scene_path%/}")"

  for idx in "${!EXP_NAMES[@]}"; do
    JOB_SCENE_PATH+=("$scene_path")
    JOB_SCENE_NAME+=("$scene_name")
    JOB_EXP_NAME+=("${EXP_NAMES[idx]}")
  done
done

NUM_JOBS="${#JOB_EXP_NAME[@]}"
echo "Total jobs: $NUM_JOBS"
echo "GPUs: ${GPU_IDS[*]}"
echo "Output: $OUT_DIR"
echo "Logs: $LOG_DIR"

num_gpus="${#GPU_IDS[@]}"
GPU_PID=()
GPU_JOBIDX=()
for ((i=0; i<num_gpus; i++)); do
  GPU_PID[i]=0
  GPU_JOBIDX[i]=-1
done

FAIL_FILE="${LOG_DIR}/failed_jobs.txt"
: > "$FAIL_FILE"

cleanup() {
  echo ""
  echo "[SCHED] terminating running jobs..."
  for ((i=0; i<num_gpus; i++)); do
    pid="${GPU_PID[i]}"
    if [[ "$pid" -ne 0 ]] && is_alive "$pid"; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait || true
  exit 130
}
trap cleanup INT TERM

job_idx=0
while true; do
  for ((i=0; i<num_gpus; i++)); do
    pid="${GPU_PID[i]}"
    if [[ "$pid" -ne 0 ]] && ! is_alive "$pid"; then
      wait "$pid"
      rc=$?
      ended_idx="${GPU_JOBIDX[i]}"
      if [[ "$rc" -ne 0 ]]; then
        printf '%s\n' "idx=${ended_idx} scene=${JOB_SCENE_NAME[ended_idx]} exp=${JOB_EXP_NAME[ended_idx]} rc=${rc}" >> "$FAIL_FILE"
      fi
      GPU_PID[i]=0
      GPU_JOBIDX[i]=-1
    fi
  done

  all_idle=1
  for ((i=0; i<num_gpus; i++)); do
    if [[ "${GPU_PID[i]}" -ne 0 ]]; then
      all_idle=0
    fi
  done

  if [[ "$job_idx" -ge "$NUM_JOBS" && "$all_idle" -eq 1 ]]; then
    break
  fi

  for ((i=0; i<num_gpus; i++)); do
    if [[ "$job_idx" -ge "$NUM_JOBS" ]]; then
      break
    fi
    if [[ "${GPU_PID[i]}" -eq 0 ]]; then
      gpu="${GPU_IDS[i]}"
      s_path="${JOB_SCENE_PATH[job_idx]}"
      s_name="${JOB_SCENE_NAME[job_idx]}"
      e_name="${JOB_EXP_NAME[job_idx]}"
      run_job "$gpu" "$s_path" "$s_name" "$e_name" &
      GPU_PID[i]=$!
      GPU_JOBIDX[i]="$job_idx"
      echo "[SCHED] start idx=$job_idx gpu=$gpu scene=$s_name exp=$e_name pid=${GPU_PID[i]}"
      ((job_idx+=1))
    fi
  done

  sleep 1
done

if [[ -s "$FAIL_FILE" ]]; then
  echo "Done with failures. See: $FAIL_FILE"
  exit 2
fi

echo "All ablations completed successfully."
