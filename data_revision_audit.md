# Provider-data revision audit

**Audit date:** 2026-09-03
**Scope:** Affibody LibA/LibB selection labels, the LibB direct-retention panel,
residue numbering, and downstream result artifacts

## Summary

Three separate corrections were checked, with three different consequences.

1. **The LibB retention panel is now complete.** Adding `AH × LIFTK = 87.94`
   changes LibB from 119 measured pairs to the complete 12-peptide ×
   10-Affibody matrix of 120 pairs. At the existing 75% cutoff, the panel now
   contains 61 binders and 59 non-binders. Every LibB metric based on retention
   must therefore be recomputed on 120 pairs.
2. **The current positive-label code already pools R009 and R010 before taking
   the top 2%.** An independent reconstruction from the available full raw
   round files gives exactly the same positive pair IDs as the current pipeline.
   The strict training sets remain 22,542 LibA pairs and 30,648 LibB pairs.
   Under the current tie-inclusive definition, this finding does not require
   retraining; the unavailable provider exports still need to confirm how ties
   at the 2% boundary were handled.
3. **Existing 58-aa analyses used the intended physical residues under the old
   names.** Every auditable residue mapping in the current and in-progress
   sequence, MINT, ESMFold2, crystal, RDE-PPI, and StaB-ddG code is category A,
   not B. This does not mark the in-progress structural adapters as complete.
   Adding two to the existing Python indices would select the wrong amino acids.
   Whether future model input should remain 58 aa or explicitly include the
   hidden `MA` is still unresolved.

The corrected provider PPT and the two named pooled-count TSV exports are not
present in the workspace. The corrected panel in this audit is therefore the
old locally available matrix plus the provider revision notice's one stated
value. Positive-ID equality was checked by independently pooling the full raw
R009/R010 files, not by comparing with the unavailable provider exports.

## Source files and provenance

Timestamps are UTC filesystem modification times unless identified as an
archive-member timestamp. All relative paths below are rooted at
`/fsx/users/dongweij/mint`.

| Source | SHA-256 | Timestamp | Audit use |
|---|---|---|---|
| `Affibody-MHC data summary.pptx` | `4d7874bdc1c3cdea9202040672ed3b954923498ad8aaa1c87ba5b2e842c39fbb` | 2026-08-17 05:15:58 | Locally available retention deck; still has the blank AH × LIFTK cell and stale N=119 tables |
| `Affibody coevolution dataset_updated.zip` | `68903e7a397874f418ec4b5e94fef13ea5c1408eb7198a7fd3e35874711d2986` | 2026-08-10 03:01:24 | Most recent uploaded provider archive |
| `Affibody coevolution dataset_updated.zip!Affibody coevolution dataset/pep-affibody sequences_updated.pptx` | `35a5c7ec478ab46c871416385e22b90a690dfcdd7eb3e0484b8bf1c790cb433f` | 2026-02-04 20:58:52 (ZIP DOS); 2026-02-05 04:58:52 UTC (UT field) | Available sequence/numbering presentation; still shows the 58-aa Affibody without the hidden `MA` |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibA Raw data/xxylibA_R1_009_count_freq_pvalue.tsv` | `65f7842fb437c813a95d8987670ccc17726c62ce5e7103774e3cb2ea647a998c` | 2026-08-06 19:24:04 | LibA full R009 counts |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibA Raw data/xxylibA_R1_010_count_freq_pvalue.tsv` | `6862f7f87d89f2cbd091f9ed43aed9a077f426ef4eaace71e1f0cc0dce20df9b` | 2026-08-06 19:24:04 | LibA full R010 counts |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibB Raw data/xxylibB_R005_count_freq_pvalue.tsv` | `5e87725e50452ec5ed7321007ad95013ec2f3cf80b3135a98024bec8148be357` | 2026-08-06 19:24:03 | LibB full R005 counts |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibB Raw data/xxylibB_R006_count_freq_pvalue.tsv` | `da4b6ef44bb8cdda203f45a4867326ecfd7ea3f9ee104f4e87fdc95848de492b` | 2026-08-06 19:24:03 | LibB full R006 counts |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibB Raw data/xxylibB_R007_count_freq_pvalue.tsv` | `f46a39df9dc5e04e640ace329a2a62ea16f69683e42721ee036e42ac9814970d` | 2026-08-06 19:24:03 | LibB full R007 counts |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibB Raw data/xxylibB_R008_count_freq_pvalue.tsv` | `e550b2644b6496957c5c602cf3450f429547fa9bcbcba385f81e832973a50340` | 2026-08-06 19:24:03 | LibB full R008 counts |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibB Raw data/xxylibB_R009_count_freq_pvalue.tsv` | `05edcbbf092f7b858e80f1bf5c5e8855de7a3894002ccb33e8285a652f16662a` | 2026-08-06 19:24:03 | LibB full R009 counts |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibB Raw data/xxylibB_R010_count_freq_pvalue.tsv` | `a0363be70692a6f1063ac5df5b8fa7e834f43ce04a074d6a86ae0b2cf65c7dc7` | 2026-08-06 19:24:03 | LibB full R010 counts |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibB Raw data/xxylibB_R011n_count_freq_pvalue.tsv` | `062dd2486f28ffc818541bbd1a37597f4bdf8b037587110f542a58839127dbc9` | 2026-08-06 19:24:04 | LibB full R011 counts |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibB Raw data/xxylibB_R012n_count_freq_pvalue.tsv` | `5201fb3a909975653c23ab797414d8efcc4634022fcfe052ebce537ab3d71664` | 2026-08-06 19:24:04 | LibB full R012 counts |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibB Raw data/xxylibB_R013n_count_freq_pvalue.tsv` | `bf6f2b455ac44c0c0415c7ecc3141972a6ee4f7102b2c3f3a38b07c2b2f94017` | 2026-08-06 19:24:04 | LibB full R013 counts |
| `private_data/source/affibody_coevolution_dataset_d91914a2/LibB Raw data/xxylibB_R014n_count_freq_pvalue.tsv` | `916a81b42c5bcec0dc5ca524415f8a71752688d3d3f9bf4583b02c7152099861` | 2026-08-06 19:24:04 | LibB full R014 counts |
| `private_data/derived/retention_matrix.csv` | `5ab9c505c3aa3b39b5f9f2586da6d5efc02b9fdac28464a93eb816ec0a0f4e5a` | 2026-08-05 23:47:26 | Old 228-design matrix; LibB contains 120 cells but only 119 values |
| `private_data/derived/retention_sequences_v2.csv` | `5645ee19d5b56e4dc33493125ccc2405795348a75b5d3803325321a4faf654cd` | 2026-08-11 03:31:30 | Stable pair IDs and complete sequences; correctly leaves the old missing outcome blank |
| `private_data/provider_revision_2026-09-03/retention_correction.json` | `b6cc3fccf7c57ad09662edec7a67746d3c2c0eb7161058c7ddc1f6b3bc29192d` | 2026-09-03 01:22:50 | Local, hash-bound record of the correction stated in the provider revision notice |
| `private_data/derived/selection_weak_labels_v1/weak_labels.csv` | `186dcd4fbb2d9c933b351beb7e0f4d46f59391cd14ef046db4861d5108230470` | 2026-08-11 07:13:46 | Current selection-derived labels |
| `private_data/derived/selection_weak_labels_v1/holdout_overlap.csv` | `178c733ae9689300d1852efe475c871ebb19521815a1db42fff38159082c3306` | 2026-08-11 07:13:46 | All 228 evaluation designs excluded from training |
| `nyeso_xx133_complex.pdb` | `59d026542cc42006302117cc79ad720497e69c4298f95598413f1d172d407b0e` | 2026-08-20 07:29:50 | LibB crystal mapping |
| `private_data/derived/libb_fixed_crystal_contract_v2/residue_mapping.json` | `a3d08cfd4a4800d3cb2048dab420e50896f36534548988f282dfd184779c2258` | 2026-09-03 00:54:26 | Existing sequence-to-crystal mapping |

The explicit local record of the provider correction is
`private_data/provider_revision_2026-09-03/retention_correction.json`. The
corrected, sequence-aware output is
`private_data/derived/retention_panel_provider_revision_2026-09-03_v2/libb_evaluation_panel.csv`
(SHA-256
`79e6ce2543d9712968c49e2421a28ed28ef0026f94c1b5ddbb2072e0b06c6da8`).
The builder preserved every pre-existing sequence and identifier and filled
only the corrected pair's retention and derived binder label.

Corrected audit and evaluation artifacts generated during this audit are:

| Output | SHA-256 | Timestamp |
|---|---|---|
| `private_data/derived/retention_panel_provider_revision_2026-09-03_v2/libb_evaluation_panel.csv` | `79e6ce2543d9712968c49e2421a28ed28ef0026f94c1b5ddbb2072e0b06c6da8` | 2026-09-03 01:30:12 |
| `private_data/derived/retention_panel_provider_revision_2026-09-03_v2/manifest.json` | `fb88cc658396fc256e5add42afc1644ace75c16cc50bfc8240cf2d5dcf7e1ae1` | 2026-09-03 01:30:12 |
| `private_data/derived/libb_provider_pooled_score_120_metrics_revision_20260903_v1/metrics_by_model_seed.csv` | `be425b2a0b296bf8768fa6c7c5b5de98777283ec4ef7190a9c33c2068d67e1e9` | 2026-09-03 01:30:57 |
| `private_data/experiments/esmfold2_libb_provider_revision_120_metrics_v1/metrics_seed_summary.csv` | `0ce9f4b9466f88d157b4c0b464fb4cd3df79978758d724a19d792dda08386b73` | 2026-09-03 01:32:18 |
| `private_data/experiments/pnu_libb_retention_120_revision_v1/metrics/metrics_by_model_seed.csv` | `b5c648771981c7607ddcf7d4f0ffe79bbb7bab1b9b227038698c8194e614e423` | 2026-09-03 01:41:17 |
| `private_data/experiments/libb_revision_mint_metrics_120_v1/metrics_by_model_seed.csv` | `0755e310d63e4d9b3ad05fcc92e543e26574a7320207aac95d4704316bd85422` | 2026-09-03 01:48:47 |
| `private_data/provider_revision_2026-09-03/residue_numbering_mapping.json` | `89ba0ef2dad7daaa36270211532612bdb7cb9cad869415eab4e6faf9ae6f5844` | 2026-09-03 01:53:55 |

## 1. Corrected LibB direct-retention panel

### What changed

| Panel | Designed cells | Measured | Retention ≥75 | Retention <75 | Missing |
|---|---:|---:|---:|---:|---:|
| LibA, unchanged | 108 | 108 | 38 | 70 | 0 |
| LibB, old artifact | 120 | 119 | 60 | 59 | 1 |
| LibB, corrected | 120 | 120 | 61 | 59 | 0 |
| Both libraries, corrected | 228 | 228 | 99 | 129 | 0 |

The restored pair is:

| Peptide | Affibody | Retention | Binder at ≥75 | Pair UID |
|---|---|---:|---:|---|
| AH | LIFTK | 87.94 | yes | `6e8454b1ba8587952e0d` |

The complete corrected AH row is useful because this is the only peptide whose
candidate set changed from nine measured choices to ten:

| Affibody | MVKNT | FALTA | NNYYF | LIFTK | ANTKV | EFYSV | ATETI | TDIDA | VAKST | NMDKV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AH retention | 76.32 | 100.00 | 31.30 | **87.94** | 36.27 | 56.88 | 8.50 | 21.67 | 15.23 | 3.52 |

### Corrected relationship between selection counts and retention

Each round's score is the raw read count for the same peptide–Affibody pair;
absence from a raw file is assigned count zero. The pooled score is the sum of
the component rounds. This calculation reproduces every old PPT Spearman and
confusion-matrix cell when restricted to the old 119 values, then changes only
the evaluation panel to 120 below.

Spearman measures whether higher selection scores tend to correspond to higher
retention values, based on rank rather than absolute scale. AUROC measures how
well the score separates retention binders from non-binders over all possible
score cutoffs. AP (average precision) summarizes the precision-recall curve and
is more sensitive to the binder/non-binder balance. For the confusion columns,
a count at or above the displayed cutoff is predicted positive and retention
of at least 75% is the measured positive: TP/TN are correct positive/negative
calls and FP/FN are the two error types. The cutoffs are those printed in the
provider deck. R005 and R006 have no confusion rows in the deck, so their
cutoff cells are intentionally left blank.

| Selection score | Spearman | AUROC | AP | Count cutoff | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R005 | 0.259226 | 0.616421 | 0.639092 | — | — | — | — | — |
| R006 | 0.422452 | 0.702973 | 0.712796 | — | — | — | — | — |
| R007 | 0.463966 | 0.728536 | 0.732436 | 4 | 13 | 0 | 59 | 48 |
| R008 | 0.530195 | 0.777438 | 0.785553 | 5 | 16 | 0 | 59 | 45 |
| R009 | 0.714846 | 0.887747 | 0.897032 | 9 | 32 | 1 | 58 | 29 |
| R010 | 0.709665 | 0.869547 | 0.889859 | 8 | 44 | 6 | 53 | 17 |
| R011 | 0.439469 | 0.718116 | 0.721887 | 19 | 38 | 18 | 41 | 23 |
| R012 | 0.436981 | 0.706863 | 0.703058 | 22 | 36 | 16 | 43 | 25 |
| R013 | 0.410128 | 0.692137 | 0.686917 | 36 | 35 | 17 | 42 | 26 |
| R014 | 0.371712 | 0.662545 | 0.668101 | 55 | 23 | 7 | 52 | 38 |
| R009 + R010 | **0.735439** | **0.886913** | **0.903666** | 12 | 45 | 6 | 53 | 16 |
| R011–R014 | 0.438851 | 0.710058 | 0.709399 | 36 | 40 | 19 | 40 | 21 |

AH × LIFTK has R009 count 352 and R010 count 903, so its pooled score is
1,255. It was already above the selection cutoff; the newly supplied retention
value changes it from an unmeasured selection-positive pair to a measured true
positive.

### Candidate-selection metrics for pooled R009 + R010

These metrics rank the ten Affibodies separately for each peptide, then average
over the 12 peptides. Precision@3 is the fraction of the three selected
Affibodies that have retention of at least 75%. Hit@3 asks only whether at
least one of the three is a binder. Best retention@3 is the best measured
retention among the three selections. Regret@3 is the difference between the
best available retention among all ten Affibodies and the best retention among
the selected three.

| AUROC | AP | Global Spearman | Within-peptide Spearman | P@1 | P@3 | Hit@1 | Hit@3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.886913 | 0.903666 | 0.735439 | 0.550675 | 0.916667 | 0.805556 | 0.916667 | 0.916667 |

| Mean best retention@1 | Mean regret@1 | Mean best retention@3 | Mean regret@3 | Distinct top-1 Affibodies |
|---:|---:|---:|---:|---:|
| 83.1875 | 11.9342 | 94.9925 | 0.1292 | 5 |

Exact score ties are resolved by ascending Affibody design code, without using
retention. This matters for pooled-count top-one metrics on EA: FALTA and MVKNT
both have count 6, but their retentions are 77.91 and 6.55. The stated rule
selects FALTA. Reversing only that tie would change P@1 and Hit@1 from 0.9167
to 0.8333, mean best retention@1 from 83.1875 to 77.2408, and mean regret@1
from 11.9342 to 17.8808. P@3 and Hit@3 are unaffected by the relevant
boundary ties because all tied boundary candidates are non-binders. The
retention-based top-three metrics are sensitive, however: on DP, FALTA is one
of three Affibodies tied at count 1 for two remaining slots. The stated rule
includes FALTA; a tie order that excluded it would change mean best
retention@3 from 94.9925 to 89.2650 and mean regret@3 from 0.1292 to 5.8567.

For AH, the pooled-count order begins `LIFTK`, `MVKNT`, `FALTA`. All three are
binders, the best retention among them is 100, and top-three regret is zero.
The complete per-peptide table is in
`private_data/derived/libb_provider_pooled_score_120_metrics_revision_20260903_v1/per_peptide_metrics.csv`.

## 2. Positive-label construction

### Operation actually used by the code

For each library, the current pipeline:

1. takes every distinct pair appearing in full raw R009, full raw R010, or
   both;
2. fills a missing round count with zero;
3. sums `R009 + R010` for that pair;
4. finds the top-2% count boundary in the pooled distribution; and
5. includes every pair tied at that boundary.

It does **not** take the top 2% independently in each round and then combine
the unique IDs appearing in either list. The implementation is
`downstream/AffibodyMHC/build_selection_weak_labels.py` (SHA-256
`32841569f194951dc38b785d83201a43febaf5ddcf415e4612a01d6a16d4d7a7`).

### Current IDs versus independent pool-first reconstruction

| Library | Old/current pipeline positive IDs | Independent pool-first reconstruction using the current tie-inclusive rule | Added IDs | Removed IDs | Intersection | Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| LibA | 19,343 | 19,343 | 0 | 0 | 19,343 | 1.000000 |
| LibB | 31,656 | 31,656 | 0 | 0 | 31,656 | 1.000000 |

Jaccard is the number of IDs shared by both sets divided by the number of
unique IDs present in either set; 1.0 means exact membership equality.

There are no added or removed IDs to list. The canonical SHA-256 set hashes
use sorted `peptide|Affibody` records, one per line with a final newline:

- LibA: `41e4fff30f5fce318b04e024bbc2887cf23ad44435e04b4ea981d1c3d15e1486`
- LibB: `56c0cc148ca9d237509e710e46628f877e413949d507a5972fde88293655f082`

The strict set retains only the declared library amino-acid design and excludes
every peptide or Affibody identity present in the retention panel. A negative
must appear in R001, never appear in any R002–R014 raw round, and have R001
count ≥3. The counts at the later filtering stages are:

| Library | Pooled positives after removing exact evaluation pairs | Strict-train positives | Strict-train negatives | Strict train total |
|---|---:|---:|---:|---:|
| LibA | 19,288 | 11,320 | 11,222 | 22,542 |
| LibB | 31,605 | 23,725 | 6,923 | 30,648 |

The retention matrix already contained AH × LIFTK as a designed but unmeasured
cell, so that pair and both evaluation partner identities were already held
out. Filling its retention value does not change training membership.

### Boundary-tie caveat

The current code includes all ties at the top-2% boundary. This gives 19,343
LibA positives at pooled count ≥13 although the nominal 2% rank is 17,630, and
31,656 LibB positives at pooled count ≥12 although the nominal rank is 29,032.
The unavailable provider pooled TSVs are needed to verify whether the provider
export also includes every boundary tie or instead keeps exactly the first
`k` rows.

For comparison, the incorrect operation of independently taking each round's
top 2% and then combining the IDs would produce only 16,026 LibA and 28,154
LibB positives, with Jaccard overlaps of approximately 0.761 and 0.757 with
the correct pooled sets. The current pipeline did not make this mistake.

## 3. Residue-numbering audit

The provider's revised labels count two preceding Affibody residues, `MA`,
that were hidden from the displayed 58-aa sequence. Existing Python indices
refer to that displayed sequence and therefore must remain unchanged. The
correction applies only to Affibody position names: peptide and HLA positions
were checked and were not shifted.

Displayed positions, revised labels, and PDB residue numbers below are
one-based. Python and model-token indices are zero-based. A code-character
number means its order in the short Affibody mutation code.

The full machine-readable mapping, including separate PDB chain and residue
fields plus model-specific token indices, is
`private_data/provider_revision_2026-09-03/residue_numbering_mapping.json`.

### LibB

| Code character | Reference amino acid | Displayed-sequence position | Python index | Revised label | PDB chain and residue | ESMFold2 token | MINT tensor token | RDE patch token |
|---:|---|---:|---:|---:|---|---:|---:|---:|
| 1 | N | 6 | 5 | 8 | H:8 | 275 | 278 | 0 |
| 2 | N | 10 | 9 | 12 | H:12 | 279 | 282 | 1 |
| 3 | Y | 13 | 12 | 15 | H:15 | 282 | 285 | 2 |
| 4 | Y | 14 | 13 | 16 | H:16 | 283 | 286 | 6 |
| 5 | F | 17 | 16 | 19 | H:19 | 286 | 289 | 3 |

The exact PDB sequence alignment shows that displayed LibB positions 3–57
match chain H residues 5–59, so displayed position `p` maps to PDB residue
`H:(p + 2)`. The five StaB-ddG native-complex tokens are 387, 391, 394, 395,
and 398; its fragment/isolated-chain token numbers differ because those inputs
contain different preceding residues.
These token numbers locate a residue inside each model's particular input
layout; different numbers across MINT, ESMFold2, RDE-PPI, and StaB-ddG do not
refer to different physical residues.

### LibA

There is no LibA-specific crystal in this workspace, so a PDB chain/residue is
not asserted as LibA structural ground truth.

| Code character | Displayed-sequence position | Python index | Provider revised position label | ESMFold2 token | MINT tensor token |
|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 12 | 15 | 282 | 285 |
| 2 | 17 | 16 | 19 | 286 | 289 |
| 3 | 27 | 26 | 29 | 296 | 299 |
| 4 | 31 | 30 | 33 | 300 | 303 |

As a direct guard against an accidental shift, the intended LibB indices read
`NNYYF` from the reference 58-aa Affibody. Treating revised labels
8/12/15/16/19 as one-based positions in that displayed sequence—equivalently,
adding two to the existing zero-based Python indices—would read `EAEIL`, which
is wrong.

### Component-by-component verdict

| Component | Audit result |
|---|---|
| Short-code sequence reconstruction and mutation-position baseline | **A — correct physical residues, old names** |
| MINT local-residue extraction | **A — correct physical residues, old names** |
| MINT whole-chain and intermediate-layer mean pooling | **A — independent of position labels for the current 58-aa input** |
| ESMFold2 input validation and full peptide × Affibody feature extraction | **A — correct currently modeled 58-aa sequences; extraction itself uses all residues** |
| Crystal contacts and mutation masks | **A — correct sequence-aligned PDB residues** |
| RDE-PPI adapter | **A — correct residues through sequence-to-structure alignment** |
| StaB-ddG mutation strings | **A — mapping logic explicitly maps displayed position `p` to PDB `H:(p+2)`; the broader adapter remains in progress and is not yet test-clean** |
| ProteoCraft mutable-position constraints | **Unresolved — no ProteoCraft source/configuration artifact is present** |

No auditable component was classified as **B — wrong physical residues**.
Prior numerical results therefore need revised labels in text and mapping
metadata, not residue-driven retraining.

## 4. Corrected existing-model evaluation

Whenever saved model weights exist, the corrected evaluation leaves them
unchanged and scores AH × LIFTK without reading its retention. The retention
value is joined only after that score is fixed. Where historical classifier
weights were not saved, the approximate reconstruction is labeled explicitly.

### Main sequence models

The original public-report designed-position score from
`selection_weak_site_v2` is exactly recoverable because that model is additive.
It assigns a learned contribution to the amino acid at each designed position
and adds those contributions. The two frozen-MINT rows instead train a small
logistic classifier on MINT sequence features from either the final layer or
layer 5. The LoRA row also updates a limited subset of MINT weights for one
training epoch.

The saved LoRA checkpoint reproduces all 119 archived probabilities exactly.
The two historical frozen-MINT classifiers were not saved, so the same fits
were reconstructed from the unchanged weak-label data. Their old-panel scores
differ from the archived values by at most 0.00131 and 0.00088. If the unseen
AH × LIFTK reconstruction error is no larger than those observed old-row
discrepancies, its gap to neighboring scores is sufficient to leave the shown
rankings unchanged. The two corrected frozen-MINT rows are therefore
sensitivity estimates, not exact recovery of the missing historical
classifiers.

| Model | AH × LIFTK score source | AUROC | AP | Global Spearman | Within-peptide Spearman |
|---|---|---:|---:|---:|---:|
| Original designed-position additive model | Exact additive reconstruction | 0.8972 | 0.9039 | 0.8241 | **0.5804** |
| Frozen MINT final layer + classifier | Archived 119 scores + replayed AH × LIFTK score | 0.8858 | 0.8879 | 0.8094 | 0.5308 |
| Frozen MINT layer 5 + classifier | Archived 119 scores + replayed AH × LIFTK score | **0.9030** | **0.9094** | **0.8210** | 0.5370 |
| MINT after one LoRA epoch | Exact saved-checkpoint inference | 0.8891 | 0.8933 | 0.8155 | 0.5410 |

| Model | P@1 | P@3 | Hit@1 | Hit@3 | Best retention@1 | Regret@1 | Best retention@3 | Regret@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original designed-position additive model | 0.9167 | 0.7778 | 0.9167 | 0.9167 | 94.6417 | 0.4800 | 95.1217 | 0.0000 |
| Frozen MINT final layer + classifier | 0.9167 | 0.7778 | 0.9167 | 0.9167 | 94.6417 | 0.4800 | 95.1217 | 0.0000 |
| Frozen MINT layer 5 + classifier | 0.9167 | 0.7778 | 0.9167 | 0.9167 | 94.6417 | 0.4800 | 95.1217 | 0.0000 |
| MINT after one LoRA epoch | 0.9167 | 0.7778 | 0.9167 | 0.9167 | 94.6417 | 0.4800 | 95.1217 | 0.0000 |

For AH, all four models rank `FALTA`, `EFYSV`, and `LIFTK` as the top three,
although their internal scores differ. AH therefore has P@1 = 1, P@3 = 2/3,
best retention@1 = best retention@3 = 100, and zero regret. Its within-peptide
Spearman is 0.8424 for the additive model, 0.7333 for final-layer frozen MINT
and LoRA, and 0.6970 for layer-5 MINT.

All four models rank FALTA first for every peptide. The DP peptide has no
retention-positive Affibody among the ten measured choices, so 11/12 = 0.9167
is the highest possible P@1 or Hit@1 on this panel. Reaching that ceiling does
not show that these models make peptide-specific top-one choices.

The complete 12-peptide rows for the three MINT models are in
`private_data/experiments/libb_revision_mint_metrics_120_v1/per_peptide_metrics.csv`.
The original designed-position rows are in
`private_data/experiments/pnu_libb_retention_120_revision_v1/metrics/per_peptide_metrics.csv`
under model `site_primary_pn_control`.

Among these four original sequence-model runs, the correction does not reverse
the earlier interpretation: the additive model remains strongest on the
peptide-conditioned ranking metric, while LoRA does not establish an
improvement over its frozen-MINT control. The layer-5 model has the best global
AUROC/AP but not the best within-peptide ranking.

### ESMFold2 frozen-feature readouts

Each ESMFold2 readout is a small classifier trained while the folding model
itself remains frozen. “Predicted distance categories” are the model's
probabilities for residue-pair distance ranges; “pair state” is a richer
residue-pair representation produced inside the folding network; “all tested
features” combines these with residue features; and “pre-folding” uses residue
features created before structural reasoning. All 25 saved classifier
checkpoints were evaluated without retraining. AH × LIFTK feature extraction
used no retention field, and the inference path reproduced the archived 119
scores with a worst absolute difference of `2.98 × 10^-8` before appending the
new prediction. Values are mean ± sample standard deviation over five saved
seeds.

| Readout | AUROC | AP | Global Spearman | Within-peptide Spearman |
|---|---:|---:|---:|---:|
| Predicted distance categories | 0.8002 ± 0.0132 | 0.8133 ± 0.0140 | 0.6138 ± 0.0141 | 0.4173 ± 0.0675 |
| Distance categories + folding pair state | 0.8399 ± 0.0163 | 0.8059 ± 0.0267 | 0.7383 ± 0.0202 | 0.4167 ± 0.0538 |
| All tested ESMFold2 features | 0.8661 ± 0.0193 | 0.8447 ± 0.0318 | 0.7732 ± 0.0259 | 0.4798 ± 0.0732 |
| Folding-derived pair state | 0.8660 ± 0.0131 | 0.8506 ± 0.0295 | 0.7741 ± 0.0166 | 0.4888 ± 0.0402 |
| Pre-folding residue features, not structure | **0.8859 ± 0.0110** | **0.8933 ± 0.0135** | **0.8067 ± 0.0186** | **0.6470 ± 0.0060** |

| Readout | P@1 | P@3 | Hit@1 | Hit@3 | Best retention@1 | Regret@1 | Best retention@3 | Regret@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Predicted distance categories | 0.6667 ± 0.0000 | 0.6667 ± 0.0340 | 0.6667 ± 0.0000 | 0.7333 ± 0.0697 | 70.0017 ± 2.7999 | 25.1200 ± 2.7999 | 82.6605 ± 4.0078 | 12.4612 ± 4.0078 |
| Distance categories + folding pair state | 0.8667 ± 0.0745 | 0.6500 ± 0.0373 | 0.8667 ± 0.0745 | 0.9167 ± 0.0000 | 87.8940 ± 4.7093 | 7.2277 ± 4.7093 | 93.8995 ± 2.5892 | 1.2222 ± 2.5892 |
| All tested ESMFold2 features | 0.8667 ± 0.0745 | 0.6833 ± 0.0639 | 0.8667 ± 0.0745 | 0.9167 ± 0.0000 | 86.3020 ± 6.9465 | 8.8197 ± 6.9465 | 93.8995 ± 2.5892 | 1.2222 ± 2.5892 |
| Folding-derived pair state | 0.8667 ± 0.0456 | 0.6944 ± 0.0651 | 0.8667 ± 0.0456 | 0.9167 ± 0.0000 | 89.0750 ± 5.1566 | 6.0467 ± 5.1566 | 95.0442 ± 0.0707 | 0.0775 ± 0.0707 |
| Pre-folding residue features, not structure | **0.9000 ± 0.0373** | **0.7778 ± 0.0000** | 0.9000 ± 0.0373 | 0.9167 ± 0.0000 | **89.8420 ± 5.1099** | **5.2797 ± 5.1099** | **95.0515 ± 0.1569** | **0.0702 ± 0.1569** |

For AH, the pre-folding readout ranks `FALTA`, `LIFTK`, and `EFYSV` in all
five seeds. Two of the three are binders. The corrected within-peptide
Spearman is 0.8424 in four seeds and 0.7818 in one.

The updated panel does not change the scientific conclusion: the strongest
ESMFold2 result still comes from features created before folding, while the
genuinely folding-derived distance and pair-state features do not beat the
strongest sequence-only ranking. Full outputs are in
`private_data/experiments/esmfold2_libb_provider_revision_120_metrics_v1/`.

### PNU and other weak-label ablations

Here P means selection-positive pairs, N means the filtered negative pairs,
and U means pairs whose status remains unlabeled. The assumed percentage is a
hypothesis about how many U pairs are truly positive, and `eta` controls how
much U contributes to training. The 2%, 5%, 10%, and 15% assumptions were
prespecified sensitivity scenarios; within each scenario, `eta`,
regularization, and training duration were selected using weak-label validation
only, never retention.

The saved models in all 19 LibB PNU/site/MINT prediction groups were left
unchanged; only the missing pair's score was appended and checked against each
complete archived 119-score vector. The table below shows the settings selected
for the weak-label report. `eta=0` means U contributed nothing to the loss. The
“frozen-MINT P+N rebaseline” is a separate weak-validation-selected control
with different regularization, not the same fit as the original frozen-MINT
row above.

| Representation and training setting | AUROC | AP | Global Spearman | Within-peptide Spearman | P@1 | P@3 |
|---|---:|---:|---:|---:|---:|---:|
| Designed-position P+N control | 0.8972 | 0.9039 | 0.8241 | 0.5804 | 0.9167 | 0.7778 |
| Designed-position PNU, assumed 2%, eta 0.25 | 0.8580 | 0.8626 | 0.7523 | 0.4227 | 0.9167 | 0.6389 |
| Designed-position PNU, assumed 5%, eta 0.25 | 0.8683 | 0.8718 | 0.7654 | 0.4833 | 0.9167 | 0.6389 |
| Designed-position PNU, assumed 10%, eta 0.25 | 0.8633 | 0.8659 | 0.7566 | 0.4833 | 0.9167 | 0.6389 |
| Designed-position reweighted P+N, assumed 15%, eta 0 | 0.9128 | 0.9182 | 0.8531 | 0.6614 | 0.9167 | 0.7778 |
| Frozen-MINT P+N rebaseline | 0.9033 | 0.9085 | 0.8459 | 0.5491 | 0.9167 | 0.7778 |
| Frozen-MINT reweighted P+N, assumed 2%, eta 0 | 0.8741 | 0.8787 | 0.7778 | 0.5005 | 0.9167 | 0.7778 |
| Frozen-MINT reweighted P+N, assumed 5%, eta 0 | 0.8808 | 0.8816 | 0.7899 | 0.4357 | 0.9167 | 0.7778 |
| Frozen-MINT PNU, assumed 10%, eta 0.25 | 0.8480 | 0.8562 | 0.7342 | 0.3741 | 0.9167 | 0.6389 |
| Frozen-MINT PNU, assumed 15%, eta 0.25 | 0.8525 | 0.8576 | 0.7443 | 0.3650 | 0.9167 | 0.6389 |

For all displayed PNU rows, Hit@1 and Hit@3 are both 0.9167. The settings that
actually use unlabeled pairs (`eta=0.25`) remain worse than their P+N controls
on within-peptide ranking and P@3. The apparently strong designed-position
15% row has `eta=0`, so it is not evidence that unlabeled data helped.

The complete 19-group tables, including best retention and regret at one and
three choices plus every AH row, are in
`private_data/experiments/pnu_libb_retention_120_revision_v1/metrics/`. In 18
of the 19 groups, every peptide receives FALTA as its top choice, producing
mean best retention@1 of 94.6417 and mean regret@1 of 0.4800. Models that omit
LIFTK from AH's top three now receive AH P@3 of one-third rather than
two-thirds.

## 5. Result disposition

The categories below mean:

- **Unaffected:** the numerical result does not use the changed LibB cell.
- **Reevaluate:** fitted weights remain valid, but the model must score the
  added pair and all retention metrics must be recalculated on 120.
- **Full retraining/rerun:** retention entered training, sample selection,
  validation, early stopping, or hyperparameter choice, so adding the outcome
  can change the fitted result itself.
- **Unresolved:** a required source artifact is unavailable.
- **No existing result:** implementation was in progress, but there is no final
  metric to reinterpret.

| Existing result family | Required treatment | Status in this audit | Reason/action |
|---|---|---|---|
| LibA selection, sequence, MINT, PNU, and retention metrics | Unaffected | Complete | The changed cell is LibB-only; update old residue names in prose where applicable |
| LibB pooled-round correlations/confusion matrices | Reevaluate | Complete | Use the 120-row values in this audit; do not copy the N=119 PPT tables |
| LibB primary designed-position, frozen final-layer MINT, selected layer-5 MINT, and one-epoch LoRA | Reevaluate | Complete for the four models listed above; two frozen rows are approximate replays | Weak-label training membership is unchanged; score AH × LIFTK without its outcome and recompute metrics |
| Full nine-layer MINT ablation | Reevaluate | Pending except for the selected layer-5 row above | The archived ablation tables still evaluate 119 pairs |
| Multi-seed matched and shared-epoch LoRA confirmatory ablations | Reevaluate | Pending | Saved checkpoints exist, but only the separate public-report one-epoch LoRA run was extended here |
| LibB PNU/site/MINT weak-label-report groups | Reevaluate | Complete for all 19 saved groups | P/N/U training and validation do not use retention; the missing score was appended to each saved group |
| Other LibB weak-label cleaning, balance, holdout, and split ablations | Reevaluate | Pending | Their fitted models remain valid, but their old 119-row metric tables are stale |
| LibB ESMFold2 frozen readouts | Reevaluate | Complete for all 25 saved checkpoints | One new feature extraction and inference pass was sufficient; no retraining |
| OpenFold3 MW-only distogram pilot | Unaffected | Complete | It evaluates only peptide MW; update only panel-count and residue-label prose |
| Retention-optimized cutoff, precision/recall, and F1 tables | Reevaluate | Pending; not part of the requested metric set above | The displayed score cutoffs were selected on the old 119-pair retention panel and must not be presented as current |
| RDE-PPI and StaB-ddG production readouts | No existing result | Pending | Existing 30,648 + 119 contracts must become 30,648 + 120 before any fit/evaluation |
| Legacy `code_only_strong_v1`, `mint_frozen_strong_v1`, `esm2_frozen_strong_v1` | Full retraining/rerun | Pending | These classifiers and their regularization choices were trained on direct retention |
| `mint_finetune_pilot_v5` LibB runs | Full retraining/rerun | Pending | Retention supplied train/validation/test data and validation MAE selected the saved epoch |
| LibB `meta_gradient_diagnostic_v1` and `v2_converged` | Full retraining/rerun | Pending | Development-retention gradients choose weak examples and affect the refitted classifier |
| LibA crossed-9 meta-gradient/DataRater artifacts | Unaffected | Complete | They are LibA-only, although retention is deliberately part of their training/selection method |
| ProteoCraft constraints | Unresolved | Pending | No auditable ProteoCraft artifact exists in the workspace |

If the missing provider pooled TSVs later show different positive membership
or different boundary-tie handling, every selection-supervised site, MINT,
LoRA, PNU, ESMFold2, RDE/StaB, and fusion readout must be refitted. Frozen
backbone weights and pure PDB measurements would remain usable.

## 6. Prospective evaluation scope

The corrected 120-pair matrix remains a retrospective holdout: it is useful for
debugging and comparison, but the project has already examined it repeatedly.
The planned progression is:

1. the corrected existing 120-pair holdout;
2. peptide–Affibody pairs allowed by the current library design but not labeled
   positive by the selection screen;
3. combinations at the mutable positions that were absent from training and
   are drawn from the full 20-amino-acid alphabet;
4. new peptide targets organized by experimental priority.

The initial prospective batch can cover 5–10 peptides with no more than ten
Affibody candidates per peptide. The expected cleavage-assay turnaround is
approximately 2–3 weeks after the candidate list is received; this is the
wet-lab assay that produces the direct retention measurement. These new
measurements—not further optimization on the 120 known values—will provide the
actual test of generalization.

## 7. Unresolved inconsistencies

1. **Corrected retention PPT unavailable.** Every locally accessible
   `Affibody-MHC data summary.pptx` has the same old hash, leaves AH × LIFTK
   blank, and contains stale N=119 summaries. The value 87.94 is currently
   attributable to the provider revision notice, not a locally verifiable
   corrected PowerPoint file.
2. **Provider pooled TSVs unavailable.** The requested
   `Affibody coevolution dataset/LibA top2pct/xxylibA_R009R010_pooled_count_freq.tsv`
   and
   `Affibody coevolution dataset/LibB top2pct/xxylibB_R009R010_pooled_count_freq.tsv`
   are absent from both uploaded ZIPs, the extracted source, and the accessible
   `/fsx/users/dongweij` tree. This prevents a literal provider-file ID
   comparison and leaves the boundary-tie convention unverified.
3. **Meaning of hidden `MA` for future model input.** Current evidence supports
   a numbering-only correction: the modeled 58-aa sequence and crystal
   alignment select the correct physical residues. The revised presentation is
   still needed to confirm whether future full-sequence inputs should remain 58
   aa or should explicitly prepend `MA`. If the input sequence itself changes,
   whole-sequence MINT/ESMFold2 features and dependent readouts must be rerun.
4. **Old v1 binder-label defect.**
   `private_data/derived/retention_sequences_v1.csv` incorrectly
   encoded the formerly missing AH × LIFTK row as `target_binder=0` despite a
   blank retention. `private_data/derived/retention_sequences_v2.csv` corrected
   this by leaving both outcome fields blank. Pipelines should use v2 or the new
   corrected panel.
5. **Hard-coded old panel sizes.** Several ESMFold2, fixed-crystal, RDE-PPI,
   StaB-ddG, fusion, sealed-evaluation, and test contracts still assert 119
   evaluation rows or 30,767 total rows. Their corrected contracts are 120 and
   30,768, respectively, before further structural model work resumes.
