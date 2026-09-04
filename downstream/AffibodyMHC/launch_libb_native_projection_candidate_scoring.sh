#!/usr/bin/env bash
set -euo pipefail

repo=/fsx/users/dongweij/mint
input_dir=private_data/prospective/libb_existing_targets_selection_missed_scoring_inputs_v1
rde_output=private_data/prospective/libb_all_design_rde_native_projection_chunks_v1
stab_output=private_data/prospective/libb_all_design_stab_native_projection_chunks_v1
log_root=private_data/prospective/libb_native_projection_candidate_scoring_logs_v1
peptides=AF,AH,DP,EA,EL,LL,LV,MW,NF,PH,TL,VV
nodes=(
  gpu-dy-p4d24xlarge-1
  gpu-dy-p4d24xlarge-2
  gpu-dy-p4d24xlarge-3
  gpu-dy-p4d24xlarge-4
  gpu-dy-p4d24xlarge-5
  gpu-dy-p4d24xlarge-6
  gpu-dy-p4d24xlarge-11
  gpu-dy-p4d24xlarge-18
)

cd "$repo"
for path in "$rde_output" "$stab_output" "$log_root"; do
  if [[ -e "$path" ]]; then
    printf 'Refusing to reuse existing launch path: %s\n' "$path" >&2
    exit 1
  fi
done
mkdir -p "$rde_output" "$stab_output" "$log_root"
receipt="$log_root/launch_receipts.tsv"
printf 'family\tslice\tnum_slices\tnode\tgpu\tpid\tlog\n' > "$receipt"

launch_worker() {
  local family=$1
  local slice=$2
  local num_slices=$3
  local node=$4
  local gpu=$5
  local output_dir batch_size checkpoint_args logfile pidfile remote_pid
  if [[ "$family" == rde ]]; then
    output_dir=$rde_output
    batch_size=128
    checkpoint_args='--rde-checkpoint-dir private_data/experiments/rde_libb_native_projection_deployment_v1/final_checkpoints --rde-readout-config downstream/AffibodyMHC/configs/rde_libb_native_projection_readouts_v1.json'
  else
    output_dir=$stab_output
    batch_size=20
    checkpoint_args='--stab-checkpoint-dir private_data/experiments/stab_libb_native_projection_120_full_v1/final_checkpoints --stab-readout-config downstream/AffibodyMHC/configs/stab_libb_native_projection_readouts_v1.json'
  fi
  logfile="$log_root/${family}_slice$(printf '%02d' "$slice")of$(printf '%02d' "$num_slices")_${node}_gpu${gpu}.log"
  pidfile="$log_root/${family}_slice$(printf '%02d' "$slice")of$(printf '%02d' "$num_slices")_${node}_gpu${gpu}.pid"

  # ``nohup ... &`` can keep an SSH session open on these nodes even after all
  # file descriptors are redirected.  ``setsid -f`` performs the detach before
  # SSH returns.  The detached shell records its own PID immediately and then
  # becomes the Python worker via exec.
  ssh "ubuntu@$node" "cd $repo && setsid -f bash -c 'echo \$\$ > $pidfile; exec env CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 venv/bin/python downstream/AffibodyMHC/score_libb_fixed_structure_candidates.py --family $family --input-dir $input_dir --output-dir $output_dir --peptides $peptides --device cuda:0 --batch-size $batch_size --chunk-rows 8192 --slice-index $slice --num-slices $num_slices $checkpoint_args > $logfile 2>&1 < /dev/null'"
  for _ in $(seq 1 100); do
    if [[ -s "$pidfile" ]]; then
      break
    fi
    sleep 0.05
  done
  if [[ ! -s "$pidfile" ]]; then
    printf 'Worker failed to write PID: %s slice %s on %s GPU %s\n' \
      "$family" "$slice" "$node" "$gpu" >&2
    exit 1
  fi
  remote_pid=$(<"$pidfile")
  if ! ssh "ubuntu@$node" "kill -0 $remote_pid"; then
    printf 'Worker exited during launch: %s slice %s on %s GPU %s\n' \
      "$family" "$slice" "$node" "$gpu" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$family" "$slice" "$num_slices" "$node" "$gpu" "$remote_pid" "$logfile" \
    >> "$receipt"
}

# Measured sustained rates are approximately 73.5 rows/s/GPU for RDE and
# 30.2 rows/s/GPU for StaB.  A 19/45 split therefore gives the two equal-sized
# candidate universes similar completion times.
for slice in $(seq 0 18); do
  global_gpu=$slice
  node=${nodes[$((global_gpu / 8))]}
  gpu=$((global_gpu % 8))
  launch_worker rde "$slice" 19 "$node" "$gpu"
done

for slice in $(seq 0 44); do
  global_gpu=$((slice + 19))
  node=${nodes[$((global_gpu / 8))]}
  gpu=$((global_gpu % 8))
  launch_worker stab "$slice" 45 "$node" "$gpu"
done

printf 'Launched 19 RDE and 45 StaB workers. Receipts: %s\n' "$receipt"
