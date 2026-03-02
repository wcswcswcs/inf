#!/usr/bin/env bash
set -uo pipefail
shopt -s nullglob

# ===== 可改参数 =====
GPU_IDS=(4 5 6 7)                 # 你有哪些 GPU 就写哪些
CHECKPOINT="./ckpt/checkpoints.pth"
DATA_ROOT="Long3D"

VOXEL_SIZE="0.2"
TOPK="2"
BUDGETS=(400000 800000 1200000)

OUT_DIR="./outs"
LOG_DIR="./logs_long3d"
SKIP_EXISTING=0                   # 1=如果输出已存在就跳过；0=每次都重跑

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ===== 预检查 =====
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: data root not found: $DATA_ROOT"
  exit 1
fi
if [[ ! -f "run_inference.py" ]]; then
  echo "ERROR: run_inference.py not found in current dir: $(pwd)"
  exit 1
fi

# ===== 工具函数 =====
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

run_job() {
  local gpu="$1"
  local scene_path="$2"
  local scene_name="$3"
  local budget="$4"

  local scene_id input_dir out_path log_path tag prefix
  scene_id="$(sanitize_name "$scene_name")"
  input_dir="${scene_path%/}/images/scan_images"

  if [[ ! -d "$input_dir" ]]; then
    echo "[GPU $gpu] ERROR: input_dir not found: $input_dir"
    return 2
  fi

  out_path="${OUT_DIR}/${scene_id}_geo_${budget}.pth"
  log_path="${LOG_DIR}/${scene_id}_geo_${budget}.log"

  if [[ "$SKIP_EXISTING" -eq 1 && -f "$out_path" ]]; then
    echo "[GPU $gpu] SKIP (exists): $out_path"
    return 0
  fi

  tag="GPU${gpu}|${scene_id}|geo|${budget}"
  prefix="[$tag] "

  : > "$log_path"
  log_line "$prefix" "$log_path" "===================================="
  log_line "$prefix" "$log_path" "Time: $(date)"
  log_line "$prefix" "$log_path" "GPU: $gpu"
  log_line "$prefix" "$log_path" "Scene: $scene_name"
  log_line "$prefix" "$log_path" "Kind: geo"
  log_line "$prefix" "$log_path" "Budget: $budget"
  log_line "$prefix" "$log_path" "Input: $input_dir"
  log_line "$prefix" "$log_path" "Output: $out_path"
  log_line "$prefix" "$log_path" "Checkpoint: $CHECKPOINT"
  log_line "$prefix" "$log_path" "===================================="

  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" python -u run_inference.py \
    --input_dir "$input_dir" \
    --checkpoint_path "$CHECKPOINT" \
    --output_path "$out_path" \
    --use_geo_kv_prune \
    --geo_voxel_size "$VOXEL_SIZE" \
    --geo_topk_per_voxel "$TOPK" \
    --total_budget "$budget" 2>&1 \
  | while IFS= read -r line; do
      printf '%s%s\n' "$prefix" "$line"
    done \
  | tee -a "$log_path"

  local py_rc tee_rc
  py_rc="${PIPESTATUS[0]}"
  tee_rc="${PIPESTATUS[2]}"

  if [[ "$tee_rc" -ne 0 ]]; then
    log_line "$prefix" "$log_path" "ERROR: tee failed rc=$tee_rc (disk full?)"
    return "$tee_rc"
  fi

  log_line "$prefix" "$log_path" "EXIT rc=$py_rc"
  return "$py_rc"
}

# ===== 收集场景目录 =====
SCENE_DIRS=( "$DATA_ROOT"/*/ )
if [[ ${#SCENE_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: no scene folders under $DATA_ROOT"
  exit 1
fi

# ===== 生成任务列表（只有 geo）=====
declare -a JOB_SCENE_PATH JOB_SCENE_NAME JOB_BUDGET

for scene_path in "${SCENE_DIRS[@]}"; do
  [[ -d "$scene_path" ]] || continue
  scene_name="$(basename "${scene_path%/}")"

  for b in "${BUDGETS[@]}"; do
    JOB_SCENE_PATH+=("$scene_path")
    JOB_SCENE_NAME+=("$scene_name")
    JOB_BUDGET+=("$b")
  done
done

NUM_JOBS="${#JOB_BUDGET[@]}"
echo "Total jobs: $NUM_JOBS"
echo "GPUs: ${GPU_IDS[*]}"
echo "Logs: $LOG_DIR"
echo ""

# ===== GPU 调度（每张 GPU 同时 1 个任务）=====
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
  echo "[SCHED] Caught signal, terminating running jobs..."
  for ((i=0; i<num_gpus; i++)); do
    pid="${GPU_PID[i]}"
    if [[ "$pid" -ne 0 ]] && is_alive "$pid"; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait || true
  echo "[SCHED] Exit."
  exit 130
}
trap cleanup INT TERM

job_idx=0

while true; do
  # 1) 回收结束任务
  for ((i=0; i<num_gpus; i++)); do
    g="${GPU_IDS[i]}"
    pid="${GPU_PID[i]}"
    if [[ "$pid" -ne 0 ]] && ! is_alive "$pid"; then
      wait "$pid"
      rc=$?
      ended_idx="${GPU_JOBIDX[i]}"

      if [[ "$rc" -ne 0 ]]; then
        scene="${JOB_SCENE_NAME[$ended_idx]}"
        bud="${JOB_BUDGET[$ended_idx]}"
        echo "[SCHED] FAIL rc=$rc | job=$((ended_idx+1)) | gpu=$g | $scene | geo | $bud"
        echo "job=$((ended_idx+1)) gpu=$g rc=$rc scene=\"$scene\" kind=geo budget=$bud" >> "$FAIL_FILE"
      else
        echo "[SCHED] DONE | job=$((ended_idx+1)) | gpu=$g"
      fi

      GPU_PID[i]=0
      GPU_JOBIDX[i]=-1
    fi
  done

  # 2) 派发新任务
  for ((i=0; i<num_gpus; i++)); do
    g="${GPU_IDS[i]}"
    if [[ "${GPU_PID[i]}" -eq 0 && "$job_idx" -lt "$NUM_JOBS" ]]; then
      scene_path="${JOB_SCENE_PATH[$job_idx]}"
      scene_name="${JOB_SCENE_NAME[$job_idx]}"
      bud="${JOB_BUDGET[$job_idx]}"

      echo "[SCHED] START job $((job_idx+1))/$NUM_JOBS | gpu=$g | $scene_name | geo | $bud"

      run_job "$g" "$scene_path" "$scene_name" "$bud" &
      GPU_PID[i]=$!
      GPU_JOBIDX[i]=$job_idx
      job_idx=$((job_idx+1))
    fi
  done

  # 3) 退出条件
  if [[ "$job_idx" -ge "$NUM_JOBS" ]]; then
    any_running=0
    for ((i=0; i<num_gpus; i++)); do
      pid="${GPU_PID[i]}"
      if [[ "$pid" -ne 0 ]] && is_alive "$pid"; then
        any_running=1
        break
      fi
    done
    [[ "$any_running" -eq 0 ]] && break
  fi

  sleep 1
done

echo ""
echo "All tasks finished."
echo "Logs in: $LOG_DIR"
echo "Failed list (if any): $FAIL_FILE"