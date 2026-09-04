# Testing folding-model intermediate representations for LibA Affibody ranking

Affibodies are small engineered proteins designed here to bind peptide--HLA
targets. This study asks whether information produced inside a protein folding
model helps choose an Affibody for a specified LibA peptide. Every
project-specific classifier is trained on the same 22,542 provisional positive
and negative examples derived from the selection rounds. The automated
ESMFold2 comparison does not use retention for training or model selection;
its outcome sidecar is loaded only after all model scores and settings have
been frozen.

The folding model is ESMFold2. It is kept frozen: its pretrained weights are
not changed. For each peptide--Affibody pair, we save several numerical arrays
that ESMFold2 creates while processing the two protein chains. Small supervised
readouts then learn whether those arrays predict the project's
selection-derived positive or negative label. Their output is a relative
selection score. It is not a predicted retention percentage, a probability of
binding, or a physical binding energy.

The feature-extraction pilot passed its identity, chain-mapping,
repeatability, and sequence-change checks. Full feature extraction for the
22,542 training pairs and 108 measured pairs is complete. The result is
negative under the prespecified decision rule. The best ESMFold2
representation on held-out selection labels was created before the folding
trunk, so it is a sequence-derived control rather than structure evidence. Its
within-peptide average precision was 0.661, below 0.710 for frozen MINT layer
9. Every folding-derived ESMFold2 representation performed worse on that
comparison. Candidate-scale ESMFold2 extraction is therefore not justified.

There is an important limit on what this experiment can establish. The current
project files contain no experimentally determined LibA peptide--Affibody
complex. The only experimental complex in the workspace is the LibB
`MW + NNYYF` crystal. Its geometry is not reused as though it were a LibA
structure. This study can determine whether ESMFold2 intermediate
representations improve prediction, but it cannot check whether ESMFold2
reconstructed the true LibA binding interface.

## What folding-model intermediate representations mean here

A protein chain is a connected sequence of amino acids. Once an amino acid is
part of the chain, it is also called a residue. The written sequence tells us
the residue order. A three-dimensional structure tells us which residues end
up close to one another after the chain folds.

LibA varies two peptide residues and four Affibody residues. A sequence model
can learn that a particular amino acid is generally favorable at one of those
six sites. A folding model may additionally represent how the peptide and
Affibody are arranged, which residues might be close, and what chemical context
surrounds a changed residue.

Predicted proximity is not the same as binding. A folding-derived numerical
representation can also be useful because of sequence patterns learned during
pretraining even when its implied geometry is inaccurate. For this reason, the
experiment includes a pre-folding ESMFold2 representation as a control. A
claim that the folding stage added predictive value requires a folding-derived
representation to improve over that pre-folding control as well as over the
independent sequence-only models. It would still not establish structural
accuracy.

## The LibA structural evidence currently available

There is no LibA PDB or CIF file in the current workspace. The provider
described LibA as having been designed from an RFdiffusion/AlphaFold model, but
the original predicted complex used for that design has not been provided here.
Its binding geometry therefore cannot be audited against a structure file.

The available `nyeso_xx133_complex.pdb` is an experimental LibB complex for
peptide code `MW` and Affibody code `NNYYF`. LibA and LibB vary different
Affibody positions and were designed from different structural starting
points. Placing LibA mutations onto the LibB crystal would create an unverified
template transfer, not a LibA structure model.

For that reason, two methods used in the LibB analysis cannot be carried over
as valid LibA structure predictors:

- RDE-PPI-derived features tied to the LibB crystal; and
- StaB-ddG/ProteinMPNN-derived features tied to that same crystal.

These methods can be revisited if a provider-approved LibA complex model or an
experimental LibA complex is supplied. Until then, their omission is a data
limitation, not a negative result about either architecture.

## Training data from the selection rounds

The LibA R000--R014 sequencing files contain 12,736,262 rows. A row records the
read count for one peptide--Affibody pair in one round. The same pair can
therefore contribute rows in several rounds; the 12.7 million rows are repeated
selection observations, not 12.7 million independently labeled examples.

Positive and negative training examples are defined as follows:

- **Positive:** combine every distinct pair observed in R009, R010, or both;
  treat absence from a round as a count of zero; add the R009 and R010 counts;
  and keep the highest 2% of the pooled distribution, including all ties at the
  boundary. The current LibA boundary is a pooled count of 13.
- **Negative:** require at least three reads in R001 and no observation of that
  exact pair in any round from R002 through R014.
- R000 does not define either class. A pair satisfying neither rule is not used
  for training.

The three-read negative rule was introduced by this analysis to reduce the
influence of pairs with almost no starting observations. It was not
experimentally calibrated by the data provider and is kept fixed here so that
the comparison changes only the representation.

| Data at each stage | LibA observations or pairs |
|---|---:|
| Sequencing rows across R000--R014 | 12,736,262 |
| Positive pairs before measured-panel removal | 19,343 |
| Positive pairs after removing measured pairs, before partner holdout | 19,288 |
| Strict training positives | 11,320 |
| Strict training negatives | 11,222 |
| Strict training pairs used by every model | 22,542 |
| Direct-retention evaluation pairs | 108 |

Before fitting, every complete peptide-side sequence and every complete
Affibody sequence in the retention panel is removed from training. The model
can still learn individual amino-acid effects from other LibA variants, but it
cannot retrieve either complete evaluation partner from the training set. This
is a strict partner holdout within LibA, not a test on a different library,
HLA, scaffold, or assay.

The positive IDs reconstructed independently from the raw R009 and R010 files
match the current pipeline exactly: 19,343 shared IDs, no additions or
removals, and Jaccard overlap 1.0. The separately named provider pooled TSV was
not present in the audited workspace, so the provider's treatment of ties at
the 2% boundary has not been independently confirmed from that export.

## Sequence and residue mapping

The complete model inputs are a 270-residue
SMART--HLA--linker--peptide chain and a 58-residue displayed Affibody chain. No
TCR is included. The experimental linker connecting those two constructs is
not included.

LibA's two-character peptide code occupies peptide positions 4 and 5. Its
four-character Affibody code occupies displayed-sequence positions 13, 17, 27,
and 31, which are Python indices 12, 16, 26, and 30. The revised
crystal-aligned labels for those same physical Affibody residues are 15, 19,
29, and 33. The change in names comes from two preceding residues, `MA`; the
existing code already changes the intended physical residues and must not add
two to its Python indices.

The current full-sequence models still use the provider-displayed 58-residue
Affibody. Whether the hidden leading `MA` should be included in future model
input remains unresolved. Adding it would change the sequence presented to
MINT and ESMFold2 and would require re-extraction and refitting, not just a
cosmetic label change. The detailed mapping is maintained in
`data_revision_audit.md`.

## What was extracted from ESMFold2

The audited checkpoint is the local Hugging Face ESMFold2 revision
`bce015efb23b5dc604842d0ab5c2bbb02c7bd3ee`. Extraction uses Python 3.12.3,
PyTorch 2.9.1 with CUDA 12.6, and Transformers 5.16.1. The complete 270- and
58-residue chains are passed to the folding trunk. The saved arrays retain only
the nine-residue peptide and the 58-residue Affibody because those are the
partners whose relationship is being modeled.

| Saved representation for one pair | Shape | Plain-language meaning |
|---|---:|---|
| `single_inputs` | 67 × 451 | One 451-number vector for each of 9 peptide and 58 Affibody residues, created before the repeated folding calculations |
| `distogram_probabilities` | 9 × 58 × 64 | For every peptide--Affibody residue pair, a distribution over 64 model-specific distance categories |
| `pair_states_symmetric` | 9 × 58 × 256 | A richer 256-number relationship vector for every peptide--Affibody residue pair after the folding trunk |

The public checkpoint metadata does not specify physical angstrom boundaries
for the 64 distance categories. We therefore retain the complete category
distribution and do not convert it into a claimed “probability within 8 Å.”
The final coordinate-diffusion stage is not run; the folding trunk and its
distance head are run because those are what create the saved intermediate
features.

## Small extraction pilot

The pilot made six ESMFold2 calls: one LibA training pair was processed twice
with the same random seed, and four additional pairs changed the peptide,
Affibody, or both. The repeated pair was `TH + IKAA`. This is an extraction
anchor, not an experimental structure reference.

| Pilot check | Result |
|---|---|
| Pair IDs and saved rows map back to the intended full sequences | Passed |
| The repeated pair gives bit-for-bit identical saved arrays | Passed |
| The distance-category array changes when the sequence changes | Passed |
| The residue-pair array changes when the sequence changes | Passed |
| The pre-folding residue array changes when the sequence changes | Passed |
| One-time checkpoint load on an NVIDIA A100 40 GB | 131.6 seconds |
| Warm forward pass per pair | approximately 4.68 seconds |
| Peak allocated GPU memory per forward | approximately 14.6 GB |
| Saved float16 features per pair | approximately 0.39 MB before filesystem overhead |
| Estimated raw feature storage for 22,650 rows | approximately 8.94 GB |

These checks show that the pipeline is aligned and responsive to LibA sequence
changes. They do not show that any feature predicts selection or retention;
that question is answered only by the matched readout comparison below.

## Models in the matched comparison

Every model receives the current pair and produces one binary selection logit.
Higher scores mean “more similar to the selection-derived positives” on that
model's scale.

### Six-position additive sequence control

This model sees a 120-value one-hot encoding: 20 possible amino acids at each
of the six designed sites. Logistic regression learns one contribution for
each amino acid at each site and adds the six chosen contributions plus a
constant. It has 121 trainable parameters and cannot learn a special
peptide--Affibody combination beyond those individual contributions.

### Six-position nonlinear sequence control

This model receives the same 120 values, followed by hidden layers of 64 and
32 units with GELU activations, layer normalization, and 0.1 dropout. It has
10,225 trainable parameters. Unlike the additive control, it can learn
combinations between peptide and Affibody sites.

### Frozen MINT controls

MINT receives the two complete reconstructed protein chains. Its pretrained
weights remain frozen. Residue representations are averaged separately within
the peptide-side and Affibody chains, the two 1,280-number averages are joined,
and a 2,561-parameter logistic classifier is trained. Layer 9 was selected from
the MINT layer ablation using only selection-derived validation labels; final
layer 33 is retained as a reference.

### Frozen ESMFold2 readouts

Five small readouts are predefined:

- distance categories only;
- richer folding-derived pair states only;
- both of those folding-derived arrays together;
- pre-folding residue features only; and
- all saved features together.

Their trainable parameter counts are 7,713, 14,241, 16,417, 35,553, and
46,433, respectively. Pooling is learned but position-aware and is accompanied
by ordinary mean and maximum summaries. The frozen ESMFold2 backbone itself is
not counted as trainable.

All neural readouts use class-weighted binary cross-entropy and AdamW. Training
settings, including a maximum of 50 epochs and early stopping, are chosen on
selection-derived validation only. The final readout is fitted with five fixed
random seeds.

A discarded initial run performed an unnecessary CPU conversion; the clean
rerun moved that exact conversion to the GPU after a parity test showed
identical inputs, losses, and model weights. No scientific setting changed.

## Model selection without retention

The matched comparison uses three validation folds. In each fold, neither
complete partner of a validation pair appears in that fold's training rows;
rows sharing exactly one held partner are set aside. This gives 7,515 common
out-of-fold predictions, including 3,759 selection-derived positives. There
are 189 peptide groups containing both weak-label classes.

The prespecified primary criterion is average precision calculated separately
within each evaluable peptide group and then averaged. This criterion matches
the practical task of ranking Affibodies for a specified peptide. Five fixed
optimization seeds are run through the same three folds. Their out-of-fold
scores are combined by averaging logits before a feature family is selected;
log loss breaks an average-precision tie within the fixed numerical tolerance
of 1e-12. The final epoch
for a family is the median of its 15 fold-specific best epochs. The
direct-retention matrix is not used for any of these choices.

The completed sequence-only comparison is:

| Model | Weak-label within-peptide AP | Weak-label within-peptide AUROC | Pooled log loss | Trainable parameters |
|---|---:|---:|---:|---:|
| Six-position additive | 0.696 | 0.786 | 0.272 | 121 |
| Six-position nonlinear, five-seed mean score | 0.702 | 0.792 | 0.268 | 10,225 |
| Frozen MINT layer 9 | **0.710** | 0.796 | **0.254** | 2,561 |
| Frozen MINT layer 33 | 0.693 | 0.787 | 0.263 | 2,561 |
| ESMFold2 pre-folding residue features | **0.661** | 0.750 | 0.279 | 35,553 |
| ESMFold2 distance categories | 0.546 | 0.584 | 0.498 | 7,713 |
| ESMFold2 folding-derived pair state | 0.597 | 0.659 | 0.328 | 14,241 |
| ESMFold2 distance + pair state | 0.592 | 0.661 | 0.331 | 16,417 |
| ESMFold2 all saved features | 0.603 | 0.672 | 0.321 | 46,433 |

Among the completed controls, MINT layer 9 is the strongest prospective
sequence-only choice. Its within-peptide AP is 0.710, compared with 0.707 for
the best two-model equal-logit combination of the additive model and MINT
layer 9. The single MINT model is therefore the current locked primary rule;
adding models did not improve the weak-label selection criterion. MINT layer
33 later looks slightly better on the already-measured retention panel, but
that retrospective observation cannot replace the weak-label choice.

Among the five ESMFold2 arms, the pre-folding control was selected. Across its
five individual runs, within-peptide AP was 0.648 ± 0.013; its combined
mean-logit score reached 0.661. The individual-run means ± sample standard
deviations were 0.538 ± 0.006 for distance categories, 0.578 ± 0.008 for
pair state, 0.582 ± 0.007 for distance plus pair state, and 0.587 ± 0.007
for all saved features. None of the five seeds for any ESMFold2 arm exceeded
MINT layer 9's fixed AP of 0.710.

## Retrospective evaluation on direct retention

The LibA panel contains nine peptides and twelve Affibodies for each peptide,
giving 108 direct measurements. Retention of at least 75% defines a measured
binder: 38 pairs are binders and 70 are nonbinders.

| Peptide | Measured pairs | Binders | Nonbinders |
|---|---:|---:|---:|
| AF | 12 | 11 | 1 |
| DL | 12 | 0 | 12 |
| DP | 12 | 0 | 12 |
| EA | 12 | 0 | 12 |
| KF | 12 | 10 | 2 |
| LA | 12 | 0 | 12 |
| LL | 12 | 10 | 2 |
| NF | 12 | 4 | 8 |
| TL | 12 | 3 | 9 |
| **Total** | **108** | **38** | **70** |

The headline metrics compare Affibodies within the same peptide:

- **Within-peptide average precision (AP)** asks whether measured binders are
  concentrated near the top of each peptide's ranked list.
- **Within-peptide AUROC** asks how often a binder is scored above a nonbinder
  for the same peptide.
- **Within-peptide Spearman** compares the complete score order with the
  complete numerical-retention order. A negative value means that, on average,
  the order tends to run backward.

Only AF, KF, LL, NF, and TL contain both binders and nonbinders. AP and AUROC
therefore average five peptide rows. Spearman can use all nine rows because it
compares numerical retention rather than only the binary binder label.

All scores and five-seed aggregation rules were frozen before the retention
sidecar was opened. The matched result is:

| Model | Within-peptide AP | Within-peptide AUROC | Within-peptide Spearman |
|---|---:|---:|---:|
| Six-position additive, locked replay | 0.590 | 0.222 | -0.336 |
| Six-position nonlinear, five-seed mean score | 0.599 | 0.272 | -0.344 |
| Frozen MINT layer 9, locked replay | 0.587 | 0.219 | -0.407 |
| Frozen MINT layer 33, locked replay | 0.600 | 0.282 | -0.304 |
| ESMFold2 pre-folding residue features, five-seed mean score | 0.592 | 0.304 | -0.272 |
| ESMFold2 distance categories, five-seed mean score | 0.636 | 0.338 | -0.145 |
| ESMFold2 folding-derived pair state, five-seed mean score | 0.658 | 0.368 | -0.248 |
| ESMFold2 distance + pair state, five-seed mean score | 0.617 | 0.306 | -0.223 |
| ESMFold2 all saved features, five-seed mean score | 0.599 | 0.269 | -0.273 |

Pair state has the highest AP and AUROC in this retrospective table, while the
distance-category readout has the least negative Spearman. This is not a good
ranking result: every within-peptide AUROC remains below 0.5 and every
Spearman remains negative. Moreover, three of the five peptide rows used for
binary ranking contain 10 or 11 binders among only 12 Affibodies. Their high
binder fractions mean that a ranking with little useful information already
has an average-precision reference of about 0.633; pair state's 0.658 is only
slightly higher. The folding-derived models did select four different top
Affibodies across the nine peptides, unlike the pre-folding ESMFold2 control,
which selected `TISN` for every peptide, but those changed rankings were not
reliably correct.

This does not reverse the study conclusion. Pair state had weak-label AP 0.597
and distance categories had weak-label AP 0.546; neither approached the
retention-blind MINT reference of 0.710, and neither won a single one of the
five weak-label replicate comparisons. Promoting either model because it looks
good on the already-measured retention panel would be post-hoc selection on
the evaluation data. These rows are reported as diagnostic observations, not
as evidence of validated structure-based generalization.

The five-seed retention results also varied. For pair state, individual-seed
within-peptide AP was 0.674 ± 0.048 and Spearman was -0.191 ± 0.135. For
distance categories, the corresponding values were 0.652 ± 0.038 and -0.143
± 0.033. The pre-folding control selected by weak labels had AP 0.590 ±
0.014 and Spearman -0.298 ± 0.048 across individual seeds.

Whole-panel and fixed-list diagnostics remain in the machine-readable result
files for continuity but are omitted from the public comparison. They can look
good when a model separates generally strong and weak peptide rows without
choosing the right Affibody for a requested peptide.

## Decision on full candidate-scale folding

Candidate-scale ESMFold2 extraction is justified only if the completed study
passes both a weak-label gate and a retrospective development gate:

1. the pilot mapping and repeatability checks pass;
2. one feature family and its training duration are selected without
   retention;
3. that family is folding-derived, its five-seed combined weak-label AP exceeds
   MINT layer 9, at least four of five seed runs exceed MINT layer 9, and it
   likewise improves over the matched pre-folding ESMFold2 control; and
4. after that family has been locked, its five-seed combined score on the 108
   already-measured pairs exceeds the strongest sequence-only control on both
   within-peptide AP and within-peptide Spearman, with at least four of five
   individual seeds improving each metric.

| Gate | Current status |
|---|---|
| Pair, chain, residue, and feature-change checks | Passed |
| Five-seed held-out weak-label comparison | Completed: 25 runs and 75 fold fits |
| Reproducible gain over MINT layer 9 AP = 0.710 | Failed: best ESMFold2 arm = 0.661; zero of five seeds exceeded 0.710 |
| Folding-derived gain over pre-folding ESMFold2 | Failed: best folding-derived arm = 0.603 versus 0.661 pre-folding |
| Locked-family gain on both retention-panel ranking metrics | Failed: AP 0.592 did not exceed 0.600; Spearman -0.272 exceeded -0.304, but only one of five seeds passed the AP comparison and three of five passed Spearman |
| Candidate-scale runtime and storage decision | **No-go; 447,731-candidate ESMFold2 extraction was not launched** |

**Final decision:** do not launch ESMFold2 over the 447,731 unmeasured
selection-missed candidates. The weak-label-selected arm was the pre-folding
sequence control and it did not beat MINT layer 9. The four-of-five rules are
prespecified stability checks, not statistical-significance tests. The
retention-panel part of this gate is explicitly a retrospective development
decision; it does not turn the 108 known measurements into a fresh validation
set and cannot be used to switch to another ESMFold2 feature family.

## What the completed comparison does and does not show

This result does not show that folding-model intermediates improve LibA
generalization. It also does not prove that an ESMFold2 geometry is correct,
that a particular residue contact is real, or that any score is a binding free
energy. No final coordinates were generated, and without an experimental LibA
complex, structural accuracy cannot be measured directly. Sequence
responsiveness in the pilot is an extraction check, not structural validation.

The direct-retention matrix is retrospective: its outcomes were already known
while the broader project analysis evolved. It can describe how locked models
rank these 108 pairs, but it is not a fresh validation of the complete analysis
process. The actual test is prospective: freeze the model and candidate rule,
select previously unmeasured pairs, and then evaluate new wet-lab retention
measurements.

No result here establishes transfer to a different library, HLA, Affibody
scaffold, assay, or experimental batch. Four of the nine measured peptide rows
contain no binder at the 75% cutoff, which also limits how precisely
within-peptide binder ranking can be summarized.

## Files for scientific review

Completed inputs and pilot artifacts:

- `data_revision_audit.md`: source, positive-label, and residue-numbering audit;
- `private_data/derived/esmfold2_liba_canonical_rows_v1/rows.json`: label-free
  extraction roster;
- `private_data/derived/esmfold2_liba_features_pilot_20260904_v3/manifest.json`:
  checkpoint, tensor, mapping, perturbation, runtime, and storage audit;
- `downstream/AffibodyMHC/configs/esmfold2_liba_frozen_readouts_v1.json`: locked
  readout and validation settings; and
- `private_data/experiments/liba_common_oof_sequence_predictions_v1/summary.json`:
  matched sequence-only controls;
- `private_data/derived/esmfold2_liba_features_bce015ef_seed20260829_64shard_v1/merged/merge_complete.json`:
  full 22,650-row feature-cache receipt;
- `private_data/experiments/esmfold2_liba_replicated_cv_aggregate_v1/`:
  five-seed weak-label metrics, per-seed values, selected epochs, and the
  retention-blind family lock;
- `private_data/experiments/esmfold2_liba_final_25_fast_v1/`: 25 final readout
  checkpoints, target-free 108-pair scores, and run receipts;
- `private_data/experiments/liba_final_model_comparison_sequence_plus_esmfold2_v1/`:
  matched sealed metrics and per-pair rankings; and
- `private_data/experiments/esmfold2_liba_candidate_scale_gate_v2/decision_gate.json`:
  the machine-readable no-go decision, including the exact command,
  environment, producer hash, and input hashes.

The sealed evaluator was
`downstream/AffibodyMHC/finalize_liba_model_comparison.py` at SHA-256
`8325df2b61f684a89a83e98218253cefad12b168814ecda70fd1f1d6dc3e720a`.
It received the already-frozen sequence score file and all 25 ESMFold2 score
files, then applied
`downstream/AffibodyMHC/configs/liba_final_model_evaluation_sequence_plus_esmfold2_v1.json`
before opening the 108-pair sidecar. Exact input and output hashes are recorded
in the evaluation manifest.
