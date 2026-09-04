<!--
Authoring template. Replace every {{PLACEHOLDER}} before publication and
remove this comment. Numerical results must come from the locked LibA
artifacts. Do not copy LibB values, use retention to select a representation,
or describe a pre-folding feature as structural evidence.
-->

# Testing folding-model representations for LibA Affibody ranking

Affibodies are small engineered proteins designed here to bind peptide--HLA
targets. This study asks whether internal representations from a protein
folding model improve the choice of an Affibody for a specified LibA peptide.
Every project-specific classifier is trained on the same 22,542 positive and
negative pairs derived from the selection rounds. It is evaluated
retrospectively on the complete 9-peptide by 12-Affibody matrix of 108 direct
retention measurements.

Retention is the percentage of assay signal remaining after the construct is
cleaved. It is the project's laboratory readout, not a direct measurement of
binding affinity. Retention values are not used to train a model, select an
ESMFold2 feature, choose the number of training passes, or choose a
hyperparameter. Those choices use only selection-derived training and
validation labels.

The folding model is kept frozen. It produces internal numerical
representations for each current peptide--Affibody pair, and a much smaller
supervised classifier learns to map those representations to the project's
selection-derived positive or negative label. The resulting number is a
relative selection score. It is not a predicted retention percentage, contact
probability, physical binding free energy (`Delta G`), or mutation-induced
change in binding free energy (`Delta Delta G`).

`{{ONE_PARAGRAPH_RESULT_SUMMARY}}`

The structural interpretation is deliberately limited. The current project
files contain no experimentally determined LibA peptide--Affibody complex. The
available crystal is the LibB `MW + NNYYF` complex, which has different
designed positions and an experimentally supported binding geometry. It is not
used as if it were a LibA structure. RDE-PPI- and StaB-ddG-derived
fixed-crystal models are therefore not reported as valid LibA structure
models. The present experiment can test whether ESMFold2 representations are
useful predictors, but it cannot verify that ESMFold2 has reconstructed the
true LibA interface.

## Why folding-model representations might help

A protein chain is a connected sequence of amino acids. An amino acid after it
has been incorporated into the chain is also called a residue. The written
sequence gives their order, while folding determines which residues become
neighbors in three-dimensional space.

LibA changes two residues on the peptide side and four on the Affibody side.
A simple sequence model can learn that one amino acid is generally useful at
one of those six positions. A folding model may also represent which peptide
and Affibody residues could interact, as well as the chemical context supplied
by unchanged residues around them.

This is a hypothesis, not an assumption. Residues being predicted as close
does not guarantee binding: their chemistry, direction, flexibility, the rest
of the interface, and experimental behavior also matter. Folding models also
encode extensive sequence information before constructing a geometry. A
useful folding-model feature is evidence for structure only if a feature
created during folding improves reproducibly over both matched sequence-only
controls and the model's own pre-folding representation.

## The structural information available for LibA

There is no experimentally determined LibA peptide--Affibody complex in the
current workspace. The provider described LibA as having been designed from a
predicted RFdiffusion/AlphaFold model, and its actual binding mode remains
unknown. `{{STATE_WHETHER_THE_ORIGINAL_DESIGN_MODEL_WAS_LATER_PROVIDED_AND_HOW_IT_WAS_AUDITED}}`

The file `nyeso_xx133_complex.pdb` is an experimental crystal structure for
LibB, specifically the peptide code `MW` and Affibody code `NNYYF`. Reusing it
for LibA would place LibA amino-acid identities onto a geometry from another
library. A successful classifier could then reflect pretrained sequence
features or an arbitrary template bias rather than the real LibA interface.
For that reason, the following are outside the primary LibA comparison:

- a fixed-crystal RDE-PPI representation built from the LibB crystal;
- a fixed-crystal StaB-ddG/ProteinMPNN representation built from that crystal;
  and
- any claim that a distance in the LibB complex is a LibA ground-truth
  contact.

These methods may be revisited if a provider-approved LibA structural model or
an experimental LibA complex becomes available. A separately labeled
LibB-template-transfer diagnostic could be run for method development, but it
must not be called a validated LibA structure model or used to justify wet-lab
candidates.

## Training data from the selection rounds

The LibA R000--R014 sequencing files contain 12,736,262 rows. Each row reports
how many reads were observed for one peptide--Affibody pair in one selection
round. If the same pair appears in five rounds, it contributes five rows.
These are repeated observations over selection, not 12.7 million independently
labeled examples.

## How positive and negative training examples were defined

All models in this comparison reuse the established LibA rules without
changing the selected rounds, positive cutoff, or negative definition:

- **Positive training example:** make one list of every distinct pair appearing
  in R009, R010, or both. Add each pair's two counts, using zero if it is absent
  from one round. Label the highest 2% of this pooled distribution as positive,
  including all pairs tied at the count boundary. The current LibA boundary is
  a pooled count of 13.
- **Negative training example:** require at least three R001 reads and no
  observation of that exact pair in any round from R002 through R014.
- R000 does not define either class. A pair satisfying neither rule is not used
  for training.

The three-read negative requirement was introduced by this analysis to reduce
the influence of pairs with almost no initial observations. It was not
experimentally calibrated by the data provider. It is held fixed here so the
comparison changes the representation rather than the label definition.

The strict training set contains 22,542 pairs: 11,320 positives and 11,222
negatives. Every complete peptide-side sequence and every complete Affibody
sequence in the 108-pair retention panel is absent from this training set. The
models can still learn about individual amino acids from other LibA variants;
this is strict partner holdout within LibA, not a test on a different library,
HLA, scaffold, or assay.

| Data at each stage | Number of LibA observations or pairs |
|---|---:|
| Sequencing observations across R000--R014 | 12,736,262 |
| Positive pairs before measured-panel removal | 19,343 |
| Positive pairs after measured-panel removal and before strict partner holdout | 19,288 |
| Pairs used for strict training after all filters and partner holdout | 22,542 |
| Direct-retention evaluation pairs | 108 |

The provider-generated pooled R009/R010 LibA file was not present during the
current audit. The positive set was reconstructed directly from the two raw
round files by outer union, zero filling, count addition, and top-2% selection.
`{{REPLACE_WITH_EXACT_PROVIDER_FILE_COMPARISON_IF_THE_FILE_ARRIVES}}`

These are weak labels. Selection sequencing is related to binding, but it can
also reflect expression, amplification, sampling, and other experimental
processes. The 75% retention cutoff is not used to define the training labels.
It is used only to define measured binders during retrospective evaluation.

## Sequence and residue mapping

The LibA shorthand has two peptide characters and four Affibody characters.
For example:

`peptide [site 1 = S, site 2 = Y]; Affibody [site 1 = L, site 2 = A, site 3 = V, site 4 = G]`

The four Affibody sites formerly described as displayed-sequence positions 13,
17, 27, and 31 are now described with crystal-aligned labels 15, 19, 29, and
33. The two-number difference comes from two preceding residues, `MA`. The
existing sequence reconstruction uses the intended displayed-sequence
positions and Python indices; the correction changes their names, not which
residues should be mutated. Peptide and HLA positions must not be shifted.

Before feature extraction, the implementation verifies for every row that:

- the six code characters reconstruct the intended full sequences;
- the peptide and Affibody chains remain distinct;
- each designed residue maps to the intended model token; and
- changing a code character changes the corresponding saved feature.

The detailed character-to-sequence and sequence-to-token mapping belongs in
the data-revision audit rather than the main results table. The unresolved
question of whether the hidden leading `MA` belongs in the complete model
input must be stated here before publication: `{{MA_INPUT_DECISION_AND_EVIDENCE}}`.

## Auditing the ESMFold2 interface

ESMFold2 is a folding model. It builds several internal arrays while converting
sequence information into a proposed three-dimensional structure. Those
arrays can be saved before the final coordinates and used as frozen inputs to
a separate project-specific classifier.

The implementation must document the checkpoint and actual tensors exposed by
the installed interface rather than assuming names or dimensions from Boltz-2
or another folding model.

| Item | Audited value |
|---|---|
| ESMFold2 checkpoint and revision | `{{CHECKPOINT_AND_REVISION}}` |
| Software environment | `{{ENVIRONMENT}}` |
| Complete peptide-side input length | `{{N_RESIDUES}}` |
| Complete Affibody input length | `{{N_RESIDUES}}` |
| Chain separator/index convention | `{{CHAIN_INDEXING}}` |
| Pre-folding per-residue tensor | `{{NAME_AND_SHAPE}}` |
| Folding-derived per-residue tensor, if exposed | `{{NAME_AND_SHAPE_OR_UNAVAILABLE}}` |
| Folding-derived residue-pair tensor, if exposed | `{{NAME_AND_SHAPE_OR_UNAVAILABLE}}` |
| Distance/contact output, if exposed | `{{NAME_SHAPE_AND_DISTANCE_BIN_MEANING_OR_UNAVAILABLE}}` |
| Final-coordinate branch executed | `{{YES_NO_AND_WHY}}` |

A model-specific pair tensor is not automatically a contact probability. If
the checkpoint exposes distance categories without calibrated physical bin
edges, they remain distance-category features and are not converted into a
claim such as “probability within 8 angstroms.” The symbol Å means angstrom,
pronounced approximately “ANG-strum”; one angstrom is 0.1 nanometres.

## Small extraction pilot before scaling

Feature extraction is first run on `{{REFERENCE_LIBA_PAIR}}` and
`{{N_VARIANTS}}` variants that independently change the peptide code, the
Affibody code, and one designed residue at a time. This pilot checks data
plumbing rather than predictive performance.

| Pilot check | Result |
|---|---|
| Every feature row maps back to the expected pair ID | {{PASS_FAIL}} |
| Chain and token mapping reproduced after save/load | {{PASS_FAIL}} |
| Peptide features change when only peptide sequence changes | {{PASS_FAIL_AND_SUMMARY}} |
| Affibody features change when only Affibody sequence changes | {{PASS_FAIL_AND_SUMMARY}} |
| Unchanged-partner mapping remains stable | {{PASS_FAIL_AND_SUMMARY}} |
| Runtime after checkpoint loading | {{SECONDS_PER_PAIR_AND_HARDWARE}} |
| One-time checkpoint loading time | {{SECONDS}} |
| Stored size per pair | {{BYTES_OR_MEGABYTES}} |
| Estimated extraction time and storage for 22,650 training/evaluation rows | {{ESTIMATE}} |

Do not scale to the complete training set if pair IDs, partner chains, designed
residues, or saved tensors cannot be mapped unambiguously.

## The compared models

Every model below receives the current peptide--Affibody pair and returns one
relative selection score. The primary comparison keeps the weak labels,
partner-holdout folds, class weighting, and validation rule fixed.

### Six-position additive sequence control

This control sees only the six designed amino-acid identities. It learns one
number for each amino acid at each position and adds the six selected numbers
plus a constant. It cannot learn that one peptide amino acid works specifically
with one Affibody amino acid.

### Six-position nonlinear sequence control

This model sees the same six identities but passes them through
`{{NONLINEAR_ARCHITECTURE}}`. Its nonlinear hidden layers can learn combinations
between peptide and Affibody sites. It is the closest small neural-network
control for asking whether a richer representation adds value beyond a
trainable interaction model.

### Frozen MINT controls

MINT receives the reconstructed complete peptide-side sequence and Affibody
sequence as two interacting chains. MINT itself is frozen and a new binary
classifier is trained on its representation. The comparison includes
`{{MINT_LAYERS_AND_POOLING}}`. These controls ask whether a pretrained
interaction-aware sequence model already captures any advantage seen in the
folding-model features.

### ESMFold2 pre-folding residue representation

This input is extracted before ESMFold2's repeated folding calculations. It can
contain rich residue identities and sequence context, but it does not yet
describe a predicted fold. A strong result here supports the usefulness of a
pretrained sequence-like representation, not a claim that structure helped.

The readout receives `{{PREFOLD_POOLING_AND_INPUT_DIMENSION}}` and uses
`{{PREFOLD_READOUT_ARCHITECTURE_AND_PARAMETER_COUNT}}` trainable parameters.

### ESMFold2 distance/contact features

This input retains only `{{DISTANCE_OR_CONTACT_FEATURE_DEFINITION}}`. It asks
whether coarse predicted proximity is sufficient for the selection task. The
feature is not itself used as the final binding score. A supervised classifier
learns how the saved values relate to selection-derived labels.

The readout receives `{{DISTANCE_POOLING_AND_INPUT_DIMENSION}}` and uses
`{{DISTANCE_READOUT_ARCHITECTURE_AND_PARAMETER_COUNT}}` trainable parameters.

### ESMFold2 richer folding-derived intermediate representation

This input retains `{{PAIR_OR_RESIDUE_TENSOR_DESCRIPTION}}`, following the
general PreFold-dG idea of learning from frozen folding-model intermediates
rather than reducing them to one contact number. PreFold-dG itself uses Boltz-2
features and predicts physical binding free energy. The present model uses
ESMFold2 features and predicts the project's current-pair selection label; it
is therefore a PreFold-inspired readout, not a reproduction of PreFold-dG.

The readout receives `{{RICH_FEATURE_POOLING_AND_INPUT_DIMENSION}}` and uses
`{{RICH_READOUT_ARCHITECTURE_AND_PARAMETER_COUNT}}` trainable parameters.

### What is trained

| Model family | Information supplied to the classifier | What is trained | Output |
|---|---|---|---|
| Additive sequence control | Six designed amino acids | Additive classifier | Relative selection score |
| Nonlinear sequence control | Six designed amino acids | Small nonlinear classifier | Relative selection score |
| Frozen MINT | Interaction-aware sequence representation | Project-specific classifier | Relative selection score |
| ESMFold2 pre-folding | Per-residue features before folding | Feature pooling/readout only | Relative selection score |
| ESMFold2 distance/contact | Predicted distance/contact features | Feature pooling/readout only | Relative selection score |
| ESMFold2 richer intermediate | Folding-derived residue or residue-pair representation | Feature pooling/readout only | Relative selection score |

`{{LOSS_CLASS_WEIGHTING_OPTIMIZER_AND_EPOCH_SELECTION}}`

## Choosing model settings without retention

The matched weak-label comparison uses three fixed validation splits. In each
split, neither complete partner in a validation pair occurs in that split's
training rows; rows sharing only one held partner are set aside. The common
out-of-fold comparison contains 7,515 rows, including 3,759
selection-derived positives. Each row is scored by a model that did not train
on that fold.

Feature pooling, classifier width, regularization, training duration, and any
choice among ESMFold2 representations are selected only from these
selection-derived folds. The retention matrix remains sealed during this
choice. `{{EXACT_SELECTION_CRITERION_AND_TIE_BREAKS}}`

Each newly trained readout is fitted with five prespecified random seeds.
Reported `+/-` values are the sample standard deviation across those five
fits. This measures sensitivity to initialization and optimization. It is not
biological uncertainty and, because only nine measured peptides are available,
it does not measure variation across future targets. `{{BOOTSTRAP_OR_OTHER_PEPTIDE_LEVEL_UNCERTAINTY_IF_REPORTED}}`

## Evaluation on the 108-pair retention panel

The LibA panel contains nine peptides and twelve Affibodies for every peptide,
giving 108 measurements. Retention of at least 75% defines a measured binder.
There are 38 binders and 70 nonbinders.

| Peptide code | Measured Affibodies | Binders | Nonbinders |
|---|---:|---:|---:|
| KF | 12 | 10 | 2 |
| LL | 12 | 10 | 2 |
| NF | 12 | 4 | 8 |
| AF | 12 | 11 | 1 |
| TL | 12 | 3 | 9 |
| DL | 12 | 0 | 12 |
| LA | 12 | 0 | 12 |
| EA | 12 | 0 | 12 |
| DP | 12 | 0 | 12 |
| **Total** | **108** | **38** | **70** |

Candidate choice is performed separately for each peptide, so the primary
metrics also compare Affibodies within the same peptide:

- **Average within-peptide Spearman** compares the full ordering by model score
  with the full numerical-retention ordering, then gives each peptide equal
  weight. A value of 1 is perfect agreement, 0 is no consistent ordering, and
  a negative value means the order tends to run backward.
- **Average within-peptide AUROC** asks how often a measured binder receives a
  higher score than a measured nonbinder from the same peptide. A value of 0.5
  is chance and 1 is perfect separation.
- **Average within-peptide precision--recall area** asks whether measured
  binders are concentrated near the high-scoring end of each peptide's list.
  Its baseline depends on how many binders that peptide has.

Only KF, LL, NF, AF, and TL contain both binders and nonbinders. Within-peptide
AUROC and average precision therefore average five defined peptide values, not
all nine. Spearman uses every peptide whose numerical retentions have enough
variation to define a rank correlation.

Operational summaries such as retention of the top-ranked candidate, best
retention among the top three, and regret can remain in the machine-readable
results. Precision@3 is not a headline metric: it forces three recommendations
even when all three scores are weak. A separate score-cutoff table may be
included if needed, but any cutoff fitted on these 108 known outcomes is
retrospective and cannot be reused as a prospective candidate rule.

Whole-panel AUROC, average precision, and Spearman may be retained in the
machine-readable output for continuity. They are not emphasized because they
can reward a model for accepting or rejecting whole peptide rows without
choosing Affibodies correctly within a specified peptide.

## Results on matched weak-label validation

This table determines which feature, if any, is eligible to proceed. Every row
uses the identical 7,515 out-of-fold examples and the same partner-holdout
folds. Report both mean and standard deviation across seeds, and retain all
individual seed values in the companion JSON.

| Model | Weak-label within-peptide AP | Weak-label AUROC | Weak-label log loss | Trainable parameters | Runtime per fit |
|---|---:|---:|---:|---:|---:|
| Six-position additive control | {{VALUE}} | {{VALUE}} | {{VALUE}} | {{N}} | {{TIME}} |
| Six-position nonlinear control | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{N}} | {{TIME}} |
| Frozen MINT {{LAYER}} | {{VALUE_OR_MEAN_SD}} | {{VALUE_OR_MEAN_SD}} | {{VALUE_OR_MEAN_SD}} | {{N}} | {{TIME}} |
| ESMFold2 pre-folding residue features | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{N}} | {{TIME}} |
| ESMFold2 distance/contact features | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{N}} | {{TIME}} |
| ESMFold2 richer folding-derived features | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{N}} | {{TIME}} |

`{{PLAIN_LANGUAGE_WEAK_LABEL_RESULT}}`

## Retrospective results on direct retention

All rows use the same 108 measurements. The table describes how the already
trained models rank Affibodies for each peptide; it does not select a feature or
training setting.

| Model | Within-peptide Spearman | Within-peptide AUROC | Within-peptide average precision |
|---|---:|---:|---:|
| Six-position additive control | {{VALUE}} | {{VALUE}} | {{VALUE}} |
| Six-position nonlinear control | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{MEAN +/- SD}} |
| Frozen MINT {{LAYER}} | {{VALUE_OR_MEAN_SD}} | {{VALUE_OR_MEAN_SD}} | {{VALUE_OR_MEAN_SD}} |
| ESMFold2 pre-folding residue features--not structure | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{MEAN +/- SD}} |
| ESMFold2 distance/contact features | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{MEAN +/- SD}} |
| ESMFold2 richer folding-derived features | {{MEAN +/- SD}} | {{MEAN +/- SD}} | {{MEAN +/- SD}} |

`{{PLAIN_LANGUAGE_RETENTION_RESULT_WITHOUT_SELECTING_ON_RETENTION}}`

The pre-folding row is an essential control. If a folding-derived feature does
not outperform it reproducibly, the study has not shown that predicted
structural reasoning adds value. If every ESMFold2 row is weaker than the best
sequence-only control, then ESMFold2 does not improve the current LibA
predictor under this setup.

## Feature ablations and why they were run

These comparisons explain which part of the folding model might be useful.
They must be selected and interpreted using weak-label validation, not by
choosing whichever row looks best on retention.

- **Are distance/contact features sufficient?** Compare the distance-only arm
  with the richer folding-derived arm. This tests whether coarse predicted
  proximity retains enough information, or whether residue chemistry and other
  internal pair features matter.
- **Does folding add value beyond sequence-like information?** Compare every
  folding-derived arm with the pre-folding residue arm. Improvement only over a
  six-position baseline is insufficient because ESMFold2 already supplies rich
  sequence context before folding.
- **Is information concentrated at the designed interface?** Compare the
  prespecified local pooling around the six designed sites with
  `{{MATCHED_GLOBAL_OR_FULL_CHAIN_POOLING}}`. Local pooling can preserve which
  residue changed; global averaging can dilute six changes among hundreds of
  unchanged residues.
- **Does the readout, rather than the representation, explain a gain?** Use a
  matched small readout or report parameter counts explicitly. A longer input
  naturally creates more classifier weights, so this remains a comparison of
  complete representation-plus-readout systems unless parameter counts are
  exactly matched.

`{{ABLATION_RESULTS_IN_PLAIN_LANGUAGE_OR_REMOVE_UNRUN_ITEMS}}`

## Decision gate before exhaustive candidate extraction

Running ESMFold2 for all 455,625 LibA peptide--Affibody assignments is far more
expensive than scoring cached sequence representations. It should happen only
if all of the following checks pass:

1. pair IDs, chain boundaries, designed residues, and saved feature rows map
   correctly in the pilot;
2. the feature and readout were selected without retention labels;
3. the gain over the strongest matched sequence-only baseline is reproducible
   across seeds on held-out weak-label data; and
4. for a claim that predicted structure helps, a folding-derived feature also
   improves over the ESMFold2 pre-folding control.

| Decision item | Prespecified rule | Observed result | Pass? |
|---|---|---|---|
| Mapping and feature-change checks | All required pilot checks pass | {{RESULT}} | {{YES_NO}} |
| Gain over strongest sequence-only control | {{WEAK_LABEL_EFFECT_SIZE_OR_RULE}} | {{RESULT}} | {{YES_NO}} |
| Stability across five seeds | {{STABILITY_RULE}} | {{RESULT}} | {{YES_NO}} |
| Folding-derived gain over pre-folding control | {{EFFECT_SIZE_OR_RULE}} | {{RESULT}} | {{YES_NO}} |
| Candidate-scale runtime and storage | Within declared resource budget | {{ESTIMATE}} | {{YES_NO}} |

**Decision:** `{{PROCEED_OR_STOP_AND_ONE_SENTENCE_REASON}}`

If the gate fails, report the negative result and stop candidate-scale folding.
Do not hide the failure by choosing a different retention metric or calling a
pre-folding representation structural. The candidate pipeline can still use
the strongest valid sequence-only model or ensemble.

If the gate passes, freeze the checkpoint, feature definition, pooling,
readout, seed aggregation, and weak-label score cutoff before scoring the full
candidate universe. The candidate-selection report should then document the
exhaustive scoring and wet-lab handoff separately.

## What the comparison does and does not show

`{{PLAIN_LANGUAGE_SUPPORTED_CONCLUSION}}`

Even a positive result would show only that a frozen ESMFold2 representation
helps a classifier trained on selection-derived LibA labels. It would not prove
that the predicted geometry is correct, that any individual contact is real,
or that the score is a physical binding energy. Without an experimental LibA
complex, structural accuracy cannot be measured directly.

The direct-retention matrix is retrospective: its outcomes were already known
while the broader analysis evolved. It is useful for comparing rankings after
the rules are locked, but it is not a fresh validation of the complete
analysis process. The actual test is prospective: freeze the model and
candidate rule first, send previously unmeasured candidates to the wet lab,
and then evaluate the returned retention values.

No result here establishes transfer to a different library, HLA, Affibody
scaffold, assay, or experimental batch. The current measured panel also has no
binders for four of its nine peptides, which limits how precisely binder
ranking can be summarized.

## Recommendation

`{{FINAL_RECOMMENDATION_IN_PLAIN_LANGUAGE}}`

Use one of the following conclusions, supported by the completed tables:

- **Stop at sequence-only scoring:** no ESMFold2 representation improved the
  held-out weak-label criterion reproducibly.
- **Use the pre-folding representation as a sequence-like model:** it improved
  prediction, but folding-derived features did not add further value.
- **Proceed with candidate-scale folding:** a folding-derived representation
  passed the mapping, held-out weak-label, reproducibility, pre-folding-control,
  runtime, and storage gates.

## Files for scientific review

The final report should link generated artifacts rather than implementation
logs:

- `{{FEATURE_INTERFACE_AUDIT.json}}`: checkpoint, tensor names, shapes, chain
  indexing, and extraction stages;
- `{{PILOT_FEATURE_CHECKS.json}}`: pair mapping, sequence perturbation checks,
  runtime, and storage;
- `{{PER_PAIR_PREDICTIONS.csv}}`: selection-validation and 108-pair scores for
  every model and seed;
- `{{METRICS_BY_SEED.json}}`: all individual values, means, and standard
  deviations;
- `{{MODEL_CONFIGS_AND_HASHES.json}}`: frozen checkpoint, readout settings,
  data hashes, and split hashes; and
- `{{DECISION_GATE.json}}`: the prespecified candidate-scale decision and its
  evidence.
