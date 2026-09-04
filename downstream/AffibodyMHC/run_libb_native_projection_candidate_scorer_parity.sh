#!/usr/bin/env bash
set -euo pipefail

repo=/fsx/users/dongweij/mint
parity_root="$repo/private_data/experiments/libb_native_projection_candidate_scorer_parity_v1"
input_dir="$parity_root/label_free_inputs"
rde_output="$parity_root/rde_candidate_scorer_output"
stab_output="$parity_root/stab_candidate_scorer_output"
audit_output="$parity_root/audit"
peptides=AF,AH,DP,EA,EL,LL,LV,MW,NF,PH,TL,VV
gpu_index=${PARITY_GPU_INDEX:-0}
canonical_rows=private_data/derived/esmfold2_libb_canonical_rows_provider_revision_120_v1/rows.json
pdb=nyeso_xx133_complex.pdb
residue_mapping=private_data/derived/libb_fixed_crystal_contract_provider_revision_120_v1/residue_mapping.json
rde_root=private_data/vendor/rde-ppi
rde_vendor_checkpoint=private_data/vendor/rde-ppi/trained_models/RDE.pt
rde_network_vendor_checkpoint=private_data/vendor/rde-ppi/trained_models/DDG_RDE_Network_30k.pt
stab_root=private_data/vendor/StaB-ddG
stab_vendor_checkpoint=private_data/vendor/StaB-ddG/model_ckpts/stabddg.pt

cd "$repo"

# A completed merged manifest can only exist after every exhaustive worker for
# that family has finished and its chunks have passed the merge validator.
# This gate prevents the smoke test from competing with the active exhaustive
# jobs for a GPU.
for required in \
  private_data/prospective/libb_all_design_rde_native_projection_merged_scores_v1/manifest.json \
  private_data/prospective/libb_all_design_stab_native_projection_merged_scores_v1/manifest.json
do
  if [[ ! -f "$required" ]]; then
    printf 'Exhaustive native scoring is not fully merged; refusing GPU parity run: %s\n' "$required" >&2
    exit 1
  fi
done

if [[ -e "$parity_root" ]]; then
  printf 'Parity artifact already exists; refusing overwrite: %s\n' "$parity_root" >&2
  exit 1
fi
mkdir -p "$parity_root"
chmod 700 "$parity_root"

venv/bin/python downstream/AffibodyMHC/prepare_libb_panel_scoring_inputs.py \
  --panel private_data/derived/retention_panel_provider_revision_2026-09-03_v2/libb_evaluation_panel.csv \
  --canonical-rows "$canonical_rows" \
  --output-dir "$input_dir"

env CUDA_VISIBLE_DEVICES="$gpu_index" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  venv/bin/python downstream/AffibodyMHC/score_libb_fixed_structure_candidates.py \
    --family rde \
    --input-dir "$input_dir" \
    --output-dir "$rde_output" \
    --peptides "$peptides" \
    --device cuda:0 \
    --batch-size 10 \
    --chunk-rows 10 \
    --canonical-rows "$canonical_rows" \
    --pdb "$pdb" \
    --residue-mapping "$residue_mapping" \
    --rde-root "$rde_root" \
    --rde-checkpoint "$rde_vendor_checkpoint" \
    --rde-network-checkpoint "$rde_network_vendor_checkpoint" \
    --rde-checkpoint-dir private_data/experiments/rde_libb_native_projection_deployment_v1/final_checkpoints \
    --rde-readout-config downstream/AffibodyMHC/configs/rde_libb_native_projection_readouts_v1.json

env CUDA_VISIBLE_DEVICES="$gpu_index" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  venv/bin/python downstream/AffibodyMHC/score_libb_fixed_structure_candidates.py \
    --family stab \
    --input-dir "$input_dir" \
    --output-dir "$stab_output" \
    --peptides "$peptides" \
    --device cuda:0 \
    --batch-size 10 \
    --chunk-rows 10 \
    --canonical-rows "$canonical_rows" \
    --pdb "$pdb" \
    --residue-mapping "$residue_mapping" \
    --stab-root "$stab_root" \
    --stab-checkpoint "$stab_vendor_checkpoint" \
    --stab-checkpoint-dir private_data/experiments/stab_libb_native_projection_120_full_v1/final_checkpoints \
    --stab-readout-config downstream/AffibodyMHC/configs/stab_libb_native_projection_readouts_v1.json

venv/bin/python downstream/AffibodyMHC/audit_libb_native_projection_candidate_scorer_parity.py \
  --input-dir "$input_dir" \
  --canonical-rows "$canonical_rows" \
  --rde-output-dir "$rde_output" \
  --stab-output-dir "$stab_output" \
  --rde-reference-dir private_data/experiments/rde_libb_native_projection_120_ensemble_v1/blinded_predictions \
  --stab-reference-dir private_data/experiments/stab_libb_native_projection_120_full_v1/blinded_predictions \
  --aggregate-reference private_data/experiments/libb_native_projection_ensemble_retention_audit_v1/averaged_score_predictions.csv \
  --retrospective-audit-manifest private_data/experiments/libb_native_projection_ensemble_retention_audit_v1/manifest.json \
  --rde-exhaustive-manifest private_data/prospective/libb_all_design_rde_native_projection_merged_scores_v1/manifest.json \
  --stab-exhaustive-manifest private_data/prospective/libb_all_design_stab_native_projection_merged_scores_v1/manifest.json \
  --pdb "$pdb" \
  --residue-mapping "$residue_mapping" \
  --rde-root "$rde_root" \
  --rde-vendor-checkpoint "$rde_vendor_checkpoint" \
  --rde-network-vendor-checkpoint "$rde_network_vendor_checkpoint" \
  --stab-root "$stab_root" \
  --stab-vendor-checkpoint "$stab_vendor_checkpoint" \
  --absolute-tolerance 1e-6 \
  --output-dir "$audit_output"

printf 'Native candidate-scorer parity audit complete: %s\n' "$audit_output/report.md"
