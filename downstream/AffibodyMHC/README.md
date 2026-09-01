# Affibody--pMHC downstream baselines

This directory contains project-specific downstream code. The first baseline
uses only amino acids at the designed peptide/Affibody positions; it does not
pretend the short codes are complete protein chains and does not call MINT.

The full-sequence workflow uses the provider-confirmed two-chain model input:
smart-HLA--linker--peptide as chain 1 and Affibody as chain 2. The experimental
linker between peptide and Affibody is omitted. No TCR is supplied.

Run the strong-label baseline in the repository's pinned environment:

```bash
venv/bin/python downstream/AffibodyMHC/code_only_baseline.py \
  --input private_data/derived/retention_matrix.csv \
  --output-dir private_data/experiments/code_only_strong_v1 \
  --seed 20260807
```

The command runs LibA and LibB separately. It refuses output paths outside a
Git-ignored child of this repository's `private_data/` tree and creates the run
directory as `0700` and files as `0600`. It evaluates a no-information baseline, a whole-code additive
diagnostic, and a position-wise amino-acid additive model under repeated random,
peptide-cold, Affibody-cold, and strict double-cold splits.

All private inputs and generated predictions/results must remain under the
Git-excluded `private_data/` tree. The reusable code contains no private
measurements or sequence codes.

Row identifiers in private prediction artifacts are deterministic pseudonyms,
not anonymous values; the small code space makes dictionary recovery possible.
For bit-for-bit metric audits, load artifact CSV files with
`pandas.read_csv(path, float_precision="round_trip")`; the default parser in the
pinned pandas version can collapse extremely close floating-point scores.
The manifest records SHA-256 checksums for every generated artifact.

## Full-sequence reconstruction and frozen features

Reconstruct the private sequence table directly from the updated provider deck:

```bash
venv/bin/python downstream/AffibodyMHC/build_sequence_table.py \
  --retention-csv private_data/derived/retention_matrix.csv \
  --sequence-zip 'Affibody coevolution dataset_updated.zip' \
  --output private_data/derived/retention_sequences_v2.csv \
  --manifest private_data/derived/retention_sequences_v2.manifest.json
```

Extract MINT features on a CUDA node. The output contains the recommended
separate-chain means and prespecified mutation-focused sensitivity pools:

```bash
venv/bin/python downstream/AffibodyMHC/extract_mint_features.py \
  --input private_data/derived/retention_sequences_v2.csv \
  --output private_data/derived/mint_features_v2.npz \
  --manifest private_data/derived/mint_features_v2.manifest.json \
  --checkpoint checkpoints/mint.ckpt \
  --config data/esm2_t33_650M_UR50D.json \
  --ppi-head checkpoints/bernett_mlp.pth \
  --device cuda:0 --batch-size 2
```

`extract_esm2_features.py` provides the independent-chain ESM-2 control using
the official `esm2_t33_650M_UR50D.pt` checkpoint. Evaluation scripts fit only a
fold-local lightweight head and keep random-pair, peptide-cold, Affibody-cold,
and double-cold predictions separate. All sequence, embedding, prediction, and
metric artifacts must remain under ignored `private_data/` paths.

## Selection-derived weak supervision

Build the provider-defined binary selection labels from the extracted raw-round
directory. The positive rule is the inclusive pooled R009+R010 top 2% after an
outer join, zero fill and raw-count sum (canonical cutoffs: LibA >=13, LibB >=12).
The negative rule is present in R001 and absent from every R002--R014 round.
Ambiguous pairs and all 228 exact retention-matrix pairs are excluded.

```bash
venv/bin/python downstream/AffibodyMHC/build_selection_weak_labels.py \
  --raw-root private_data/source/<extracted-provider-archive> \
  --retention-csv private_data/derived/retention_matrix.csv \
  --sequence-zip 'Affibody coevolution dataset_updated.zip' \
  --output-dir private_data/derived/selection_weak_labels_v1
```

Run the inexpensive identity-cold position-additive control:

```bash
venv/bin/python downstream/AffibodyMHC/evaluate_selection_weak_baseline.py \
  --weak-label-dir private_data/derived/selection_weak_labels_v1 \
  --retention-csv private_data/derived/retention_matrix.csv \
  --retention-sequences-csv private_data/derived/retention_sequences_v2.csv \
  --raw-root private_data/source/<extracted-provider-archive> \
  --output-dir private_data/experiments/selection_weak_site_v2
```

On a CUDA node, train LibA and LibB separately with the same primary filter and
identity-blocked weak validation. The default run balances the two weak classes;
the retention partner identities define the strict-cold boundary, while retention
labels are used only for the final uncalibrated retrospective metrics.

```bash
venv/bin/python downstream/AffibodyMHC/finetune_mint_selection.py \
  --weak-label-dir private_data/derived/selection_weak_labels_v1 \
  --retention-sequences private_data/derived/retention_sequences_v2.csv \
  --retention-manifest private_data/derived/retention_sequences_v2.manifest.json \
  --sequence-zip 'Affibody coevolution dataset_updated.zip' \
  --checkpoint checkpoints/mint.ckpt \
  --config data/esm2_t33_650M_UR50D.json \
  --output-dir private_data/experiments/mint_selection_weak_liba_v1 \
  --library LibA --device cuda:0 --batch-size 64 --eval-batch-size 64
```

The weak negative label is a conservative assay-derived heuristic, not confirmed
nonbinding. R009/R010 were chosen after inspecting the retention experiment, so
these results require prospective confirmation even though retention labels are
excluded from fitting and checkpoint selection.

Canonical balanced-pilot retention results (Spearman / AUROC / AUPRC) are:

- LibA frozen MINT head: `0.5905 / 0.6850 / 0.4432`; rank-2 LoRA:
  `0.5986 / 0.6940 / 0.4519` (best weak-validation epoch 1).
- LibB frozen MINT head: `0.8298 / 0.8958 / 0.9008`; LoRA is identical because
  weak validation selected epoch 0.

LibA therefore shows a small ranking improvement, while LibB provides no evidence
that adapter updates help. These probabilities are uncalibrated across selection
and retention assays; ranking metrics, not probability error or retention MAE, are
the intended comparison. Conditional paired row-bootstrap intervals for every
LoRA-minus-control ranking delta include zero. The next prespecified sensitivity is LibB-only training
on all eligible candidates with inverse-frequency class weighting, because the
balanced pilot omits 10,630 eligible positives.
