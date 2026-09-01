#!/usr/bin/env bash
# Continue the frozen-ESMFold2 LibB experiment after all eight extraction
# shards have completed.  This script never exposes retention labels to model
# fitting: the retention sidecar is opened only by the final sealed evaluator.

set -Eeuo pipefail
umask 077

PROJECT_ROOT=/fsx/users/dongweij/mint
PYTHON_BIN="$PROJECT_ROOT/private_data/envs/esmfold2_hf_5_16_1_torch291cu126/bin/python"
FEATURE_ROOT="$PROJECT_ROOT/private_data/derived/esmfold2_libb_features_bce015ef_seed20260829_v1"
ROW_MANIFEST="$PROJECT_ROOT/private_data/derived/esmfold2_libb_canonical_rows_v1/rows.json"
TRAINING_LABELS="$PROJECT_ROOT/private_data/derived/esmfold2_libb_training_labels_v1/training_labels.csv"
TRAINING_MANIFEST="$PROJECT_ROOT/private_data/derived/esmfold2_libb_training_labels_v1/manifest.json"
RETENTION_AUDIT="$PROJECT_ROOT/private_data/derived/esmfold2_libb_evaluation_labels_v1/evaluation_labels.csv"
CONFIG="$PROJECT_ROOT/downstream/AffibodyMHC/configs/esmfold2_libb_frozen_readouts_v1.json"

RUN_ROOT="$PROJECT_ROOT/private_data/experiments/esmfold2_libb_postextract_v1"
CV_ROOT="$PROJECT_ROOT/private_data/experiments/esmfold2_libb_readout_cv_v1"
FINAL_ROOT="$PROJECT_ROOT/private_data/experiments/esmfold2_libb_readout_final_v1"
EVALUATION_ROOT="$PROJECT_ROOT/private_data/experiments/esmfold2_libb_readout_evaluation_v1"

export TMPDIR="$PROJECT_ROOT/private_data/tmp/esmfold2"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"

timestamp() {
  date -u +'%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

fail_if_exists() {
  if [[ -e "$1" ]]; then
    log "refusing to overwrite existing output: $1"
    exit 1
  fi
}

trap 'status=$?; log "pipeline failed at line $LINENO with status $status"; exit "$status"' ERR

cd "$PROJECT_ROOT"
install -d -m 700 "$RUN_ROOT" "$TMPDIR" "$PYTHONPYCACHEPREFIX"

fail_if_exists "$CV_ROOT"
fail_if_exists "$FINAL_ROOT"
fail_if_exists "$EVALUATION_ROOT"

log "waiting for all eight frozen-ESMFold2 feature shards"
while true; do
  completed=0
  for shard_index in 0 1 2 3 4 5 6 7; do
    shard_name=$(printf 'shard-%05d-of-00008' "$shard_index")
    completion="$FEATURE_ROOT/$shard_name/shard_complete.json"
    if [[ -f "$completion" ]]; then
      completed=$((completed + 1))
      continue
    fi

    process_pattern="extract_esmfold2_libb_features.py.*--shard-index $shard_index --num-shards 8"
    if ! pgrep -f "$process_pattern" >/dev/null; then
      log "shard $shard_index is incomplete and its extractor is not running"
      exit 1
    fi
  done

  log "completed extraction shards: $completed/8"
  if [[ "$completed" -eq 8 ]]; then
    break
  fi
  sleep 60
done

log "validating and merging the eight label-free feature shards"
"$PYTHON_BIN" downstream/AffibodyMHC/merge_esmfold2_libb_features.py \
  --feature-root "$FEATURE_ROOT" \
  --row-manifest "$ROW_MANIFEST"

log "auditing the merged cache and training-only data contract"
"$PYTHON_BIN" downstream/AffibodyMHC/train_esmfold2_libb_readouts.py \
  --config "$CONFIG" \
  --training-labels "$TRAINING_LABELS" \
  --training-labels-manifest "$TRAINING_MANIFEST" \
  --feature-cache "$FEATURE_ROOT/merged" \
  --mode audit \
  >"$RUN_ROOT/readout_audit.json"

log "selecting training epochs with weak-label double-cold validation only"
"$PYTHON_BIN" downstream/AffibodyMHC/train_esmfold2_libb_readouts.py \
  --config "$CONFIG" \
  --training-labels "$TRAINING_LABELS" \
  --training-labels-manifest "$TRAINING_MANIFEST" \
  --feature-cache "$FEATURE_ROOT/merged" \
  --output-dir "$CV_ROOT" \
  --mode cross_validate \
  --device cuda

log "fitting five final seeds on all weakly labelled training pairs"
"$PYTHON_BIN" downstream/AffibodyMHC/train_esmfold2_libb_readouts.py \
  --config "$CONFIG" \
  --training-labels "$TRAINING_LABELS" \
  --training-labels-manifest "$TRAINING_MANIFEST" \
  --feature-cache "$FEATURE_ROOT/merged" \
  --output-dir "$FINAL_ROOT" \
  --mode fit_final \
  --selected-epochs-json "$CV_ROOT/selected_epochs.json" \
  --device cuda

shopt -s nullglob
prediction_files=("$FINAL_ROOT"/blinded_predictions/*.csv)
if [[ "${#prediction_files[@]}" -ne 25 ]]; then
  log "expected 25 blinded prediction files; found ${#prediction_files[@]}"
  exit 1
fi
prediction_arguments=()
for prediction_file in "${prediction_files[@]}"; do
  prediction_arguments+=(--predictions "$prediction_file")
done

log "opening direct retention measurements for the first time in the sealed evaluator"
"$PYTHON_BIN" downstream/AffibodyMHC/evaluate_libb_readouts_sealed.py \
  "${prediction_arguments[@]}" \
  --retention-audit "$RETENTION_AUDIT" \
  --output-dir "$EVALUATION_ROOT"

log "post-extraction LibB experiment completed"
