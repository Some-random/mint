# Affibody project recovery manifest

Frozen backup date: 2026-09-04 UTC

This repository preserves the reusable Affibody modeling source, configuration,
tests, and the original MINT history. The frozen recovery branch is
`backup/affibody-work-2026-09-04`; the matching release tag is
`affibody-backup-2026-09-04`.

The backup is private because it contains unpublished experimental material.
Its intended destination is the standalone private repository
`Some-random/mint-affibody-private`. The public `Some-random/mint` fork is not a
backup destination and must not receive the private archives or new reports.

## Reports tracked in Git

The following curated reports are stored as ordinary Git objects:

- `private_data/affibody_modeling_report_public_2026-08-17.md`;
- `private_data/esmfold2_libb_report_public_2026-09-01.md`;
- `private_data/affibody_weak_label_report_public_2026-09-02.md`;
- `private_data/affibody_structure_model_report_public_2026-09-03.md`;
- `private_data/affibody_liba_structure_model_report_public_2026-09-04.md`;
- `private_data/affibody_liba_candidate_selection_report_public_2026-09-04.md`.

`data_revision_audit.md` records the provider-data corrections and identifies
which older results are historical snapshots. In particular, older documents
that say LibB has 119 measured pairs predate the corrected 120-pair matrix.

## Release assets

The private release attached to the matching tag is the recovery unit. Its
`SHA256SUMS` file gives the authoritative checksum for every asset.

| Asset | Purpose |
|---|---|
| `Affibody coevolution dataset.zip` | Original provider source archive, retained byte-for-byte and treated as opaque because it contains an unusual root-directory entry. |
| `affibody_private_materials_2026-09-01.tar.zst` | Verified legacy private snapshot covering the work completed through 2026-09-01. |
| `affibody_private_current_2026-09-04.tar.zst` | Current compact inputs, audits, reports, corrected LibB results, LibA sequence/structure results, trained small readout heads, and final handoffs. |
| `affibody_liba_exhaustive_scores_2026-09-04.tar.zst` | LibA selection-missed universe and the canonical exhaustive score tables used to build the prospective handoff. |
| `mint-affibody-private-2026-09-04.bundle` | Self-contained Git bundle of all refs in the frozen repository. |
| `SHA256SUMS` | SHA-256 checksums for all assets above except itself. |

Both newly assembled compressed archives contain an internal
`BACKUP_CONTENTS.md` and `MANIFEST.sha256`. The archives use a single relative
top-level directory. They are verified by listing, decompression testing,
extraction into fresh temporary directories, and `sha256sum -c` against their
internal manifests. Release assets are also downloaded again after upload and
compared with the local checksums.

## Canonical 2026-09-04 LibA artifacts

- The wet-lab handoff is
  `private_data/prospective/liba_wetlab_candidate_handoff_v3/`. Earlier handoff
  versions and all `.draft.md` files are superseded.
- The generic selector and compiled handoff inputs are the `v2` directories.
- The structure comparison uses
  `esmfold2_liba_replicated_cv_fast_v1`, its aggregate, the 25 final small
  readout heads, the sealed sequence-plus-ESMFold2 comparison, and
  `esmfold2_liba_candidate_scale_gate_v2`. The slow-loader CV and gate `v1` are
  superseded.
- The structure gate failed, so there is no exhaustive LibA structure-model
  score archive. The prospective candidate handoff uses the locked sequence
  models.

## Deliberate omissions

The following are omitted because they are redundant, superseded, temporary,
downloadable, or reproducible from retained inputs, code, configs, model heads,
and provenance receipts:

- every `_DO_NOT_BACKUP*`, `.staging-*`, draft, smoke, aborted, and superseded
  directory;
- full ESMFold2, MINT, RDE-PPI, and StaB-ddG feature tensors and per-shard array
  caches;
- backbone checkpoints, root model checkpoints, and bulk OpenFold outputs;
- virtual environments, package caches, logs, locks, and pytest debris;
- redundant per-peptide score partitions when an identical canonical merged
  table is retained;
- the superseded LibA handoff/selector/compiler versions and the superseded
  LibB single-tolerance release driver;
- the incidental 52-byte `uv.lock` placeholder.

Small trained readout heads are retained where they are part of a final model
contract. Feature-extraction receipts, exact hashes, configs, and merge records
are retained even when multi-gigabyte feature arrays are omitted.

## Restore and verification

Verify the downloaded release in one directory:

```bash
sha256sum -c SHA256SUMS
unzip -tqq 'Affibody coevolution dataset.zip'
zstd -t affibody_private_materials_2026-09-01.tar.zst
zstd -t affibody_private_current_2026-09-04.tar.zst
zstd -t affibody_liba_exhaustive_scores_2026-09-04.tar.zst
git bundle verify mint-affibody-private-2026-09-04.bundle
```

Restore the Git repository with:

```bash
git clone mint-affibody-private-2026-09-04.bundle mint-affibody
```

Extract either compressed archive only into a newly created directory after
inspecting its member list. Do not blindly extract the original provider ZIP.
