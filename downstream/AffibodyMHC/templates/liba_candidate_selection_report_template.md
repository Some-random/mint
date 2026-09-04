<!--
Authoring template. Replace every {{PLACEHOLDER}} before publication.
Numerical model results must come from the locked LibA artifacts; do not copy
LibB values or choose a model using the direct-retention panel.
-->

# Selecting LibA Affibody candidates for prospective wet-lab testing

Affibodies are small engineered proteins designed here to bind peptide--HLA
targets. This study asks whether sequencing data collected during experimental
selection can be used to choose new LibA peptide--Affibody pairs for laboratory
testing. The project-specific models are trained on provisional positive and
negative examples derived from the selection rounds. Direct retention values
are used only for the clearly marked retrospective comparison below; they are
not training labels and do not determine the new candidate order.

The primary prospective rule is `{{PRIMARY_RULE}}`. It was chosen using held-out
selection-derived examples, before any new wet-lab result was available.
`{{ONE_PARAGRAPH_RESULT_SUMMARY}}`

LibA has no experimentally determined complex structure in the current project
files. The available `nyeso_xx133_complex.pdb` structure is the LibB
`MW`--`NNYYF` complex and is not treated as LibA geometry. For this reason,
RDE-PPI- and StaB-ddG-derived fixed-crystal models used for LibB are not included
as valid LibA models. A frozen ESMFold2 experiment may be included because it
starts from each LibA sequence pair and does not require a LibA crystal, but its
features are folding-model predictions rather than experimentally verified
LibA structure.

## Data used to train the LibA predictors

The LibA R000--R014 sequencing files contain 12,736,262 rows. A row records the
read count for one peptide--Affibody pair in one selection round. The same pair
can therefore contribute one row in several rounds; the 12.7 million rows are
not 12.7 million independently labeled training examples.

Positive and negative examples are defined as follows:

- **Positive:** combine the distinct pairs observed in R009 or R010, use zero
  when a pair is absent from one of the rounds, and add its two counts. Keep the
  highest 2% of the resulting LibA pooled-count distribution, including every
  tie at the boundary. The boundary is a pooled count of 13. This gives 19,343
  positive pairs before measured-panel removal; 55 overlap the measured panel,
  leaving 19,288 before the strict partner holdout.
- **Negative:** require at least three R001 reads and no observation of the
  exact pair in any round from R002 through R014.
- A pair meeting neither rule is not used for training. R000 does not define
  either class.

The three-read negative rule is an analysis choice intended to reduce the risk
that a pair disappears merely because it had almost no starting observations.
The value three was not experimentally calibrated and remains a limitation.

The final strict LibA training set contains 22,542 pairs: 11,320 positives and
11,222 negatives, covering 216 peptide sequences and 13,590 Affibody sequences.
It was reconstructed directly from the current round files. The provider's
separate pooled R009/R010 LibA file was not present in the workspace at the time
of this audit, so a future copy of that file should be compared by exact pair ID
before release.

## Separation between training and the measured panel

The direct-retention panel is a complete 9-by-12 LibA matrix: nine peptide
targets, twelve Affibodies per target, and 108 measured pairs. Retention of at
least 75% is the project's experimental binder definition. This gives 38
binders and 70 nonbinders.

| Peptide code | Measured Affibodies | Retention at least 75% | Retention below 75% |
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

Before fitting any reported model, every training pair containing one of these
nine complete peptide-side sequences is removed, and every training pair
containing one of these twelve complete Affibody sequences is removed. Thus,
neither complete partner in a measured pair appeared in training. Individual
amino acids and shorter mutation patterns can still occur in other training
sequences; this is not a test on a different library, HLA, or assay.

For the matched weak-label comparison, all models use the same three-fold split
and the same 7,515 out-of-fold rows, including 3,759 selection-derived
positives. A row is scored only by a model that did not train on that fold. The
fold assignment also prevents the held-out peptide and Affibody sequences from
leaking through another pair.

## The LibA models

LibA deliberately varies two amino acids on the peptide side and four on the
Affibody side. For example, a design might be represented as:

`peptide [site 1 = S, site 2 = Y]; Affibody [site 1 = L, site 2 = A, site 3 = V, site 4 = G]`

The detailed mapping from these six code characters to full-sequence indices
is maintained in the data-revision audit. The existing code uses the intended
physical Affibody residues. The newer crystal-aligned names are two higher
because the revised numbering counts two preceding residues, `MA`; peptide and
Python indices must not be shifted by two. Whether `MA` should be present in
the complete model-input sequence remains unresolved and must be confirmed.

### Six-position additive baseline

This model sees only the six amino-acid identities in the example above. It
learns one contribution for each allowed amino acid at each site and adds the
six contributions to obtain a score. It cannot learn that one particular
peptide amino acid works only with one particular Affibody amino acid. It is a
useful reference because it tests how much can be explained by reusable effects
of individual designed residues.

### Six-position nonlinear model

This model receives the same six amino acids but passes them through
`{{NONLINEAR_ARCHITECTURE}}`. Unlike the additive baseline, it can learn
combinations between peptide and Affibody sites. Comparing the two asks whether
explicit sequence combinations improve held-out weak-label prediction and
within-peptide retention ranking.

### Frozen MINT models

MINT receives the complete reconstructed SMART--HLA--linker--peptide sequence
as one chain and the complete provider-displayed Affibody sequence as the other.
No TCR is included, and the experimental linker between the two constructs is
not included. MINT's pretrained weights remain frozen. A new project-specific
classifier is trained on the 22,542 selection-derived labels.

The final MINT layer is retained as a prespecified reference. A separate
ablation evaluates internal MINT layers because an earlier layer may preserve
the six small sequence changes more clearly. Layer 9 was selected using only
held-out weak-label performance; the retention panel did not select the layer.

### Frozen ESMFold2 intermediate features

ESMFold2 is a folding model. During folding it constructs internal arrays that
describe each residue and relationships between residue pairs before producing
final coordinates. The LibA pilot freezes ESMFold2, extracts those arrays for
each current sequence pair, and trains only a small binder/nonbinder readout on
the same selection-derived labels.

The matched comparison should distinguish:

- contact or distance-distribution features, which summarize predicted
  geometric proximity; and
- richer intermediate features, which retain more of the model's internal
  residue and residue-pair information.

These are **folding-model-derived features**, not measured contacts. There is
no experimental LibA structure with which to verify their geometry. If only a
pre-folding single-residue representation helps, it should be described as a
sequence-like feature result, not evidence that predicted structure helps. An
ESMFold2 model is eligible for exhaustive candidate scoring only if its pipeline
passes sequence-to-token checks and its frozen-feature readout improves the
prespecified weak-label criterion reproducibly.

### Ensembles

The candidate comparison includes individual valid models, an equal-logit
average, and a nonnegative logistic stacker. The stacker is trained only on
out-of-fold selection-derived predictions. No component weight, model family,
or candidate rank is chosen by looking for the best retention-panel result.
The selected primary rule and every component it uses must be listed explicitly
in the final version of this section.

The earlier one-epoch LoRA run can be shown as an exploratory ablation after its
exact data and checkpoint are re-audited. A single LoRA run should not determine
the production candidate list without repeatability evidence.

## How performance is summarized

Candidate choice is performed separately for each peptide, so the main metrics
also compare Affibodies within the same peptide. A high global score can arise
from accepting or rejecting whole peptide rows and does not necessarily show
that the model chooses the best Affibody for a requested peptide. Global AUROC
and global average precision may be retained in a machine-readable supplement,
but they are not headline metrics here.

- **Weak-label average precision** asks whether selection-derived positives are
  near the top within each peptide on the matched out-of-fold rows. This is the
  criterion used to choose a prospective model or ensemble.
- **Direct-retention average precision** asks whether measured binders
  (retention at least 75%) are near the top within each peptide.
- **Direct-retention AUROC** is the probability that a measured binder is
  ranked above a measured nonbinder from the same peptide.
- **Direct-retention Spearman** compares the complete model ranking with the
  complete numerical-retention ranking within each peptide. A value of 1 is
  perfect agreement, 0 is no consistent ordering, and a negative value means
  the order tends to run backward.

Average precision and AUROC require both a binder and a nonbinder in a peptide
row. Only KF, LL, NF, AF, and TL meet that condition, so those two reported
retention metrics average five peptide-level values. DL, LA, EA, and DP contain
no binder at the 75% threshold and are not silently treated as ordinary AUROC
or average-precision rows. Spearman uses every peptide row whose numerical
retention values have enough variation to define a rank correlation.

Average precision summarizes a complete ranked list. It is not Precision@3,
not a selected score cutoff, and not an estimate of future wet-lab yield.
Operational top-choice summaries such as the measured retention of the first
ranked candidate, the best retention among the first three, and corresponding
regret may be provided in the detailed supplement, but they do not replace the
cutoff-based prospective rule below.

## Matched model comparison

All rows below must use identical weak-label folds and the same 108-pair
retention panel. `{{N_WEAK_PEPTIDE_GROUPS}}` weak-label peptide groups contribute
to the first column. The retention columns are retrospective and do not select
the primary rule.

| Model | Weak-selection validation AP (used to lock rules) | Direct-retention panel AP (retrospective, within peptide) | Direct-retention panel AUROC (retrospective, within peptide) | Direct-retention panel Spearman (retrospective, within peptide) |
|---|---:|---:|---:|---:|
| Six-position additive baseline | {{VALUE}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |
| Six-position nonlinear model | {{VALUE}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |
| Frozen MINT layer 9 | {{VALUE}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |
| Frozen MINT final layer | {{VALUE}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |
| {{ESMFOLD2_READOUT_OR_REMOVE_ROW}} | {{VALUE}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |
| Equal-logit ensemble: {{COMPONENTS}} | {{VALUE}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |
| Weak-label-fitted ensemble: {{COMPONENTS}} | {{VALUE}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |

`{{PLAIN_LANGUAGE_COMPARISON_AND_PRIMARY_RULE_JUSTIFICATION}}`

No model should be described as better because of a negligible change from one
run. For every newly trained nonlinear or ESMFold2 readout, report the seed
values and their variation in the supporting file, and state whether the model
choice is stable.

## Retrospective score thresholds on the 108 known measurements

This table is fitted to the same 108 already-known outcomes on which it is
scored. For each model, the reported cutoff maximizes F1 on this panel. It is a
descriptive tradeoff, not an independent test, not the cutoff used for new
candidates, and not an estimate of prospective yield.

Precision is the fraction of recommended measured pairs that are binders.
Recall is the fraction of all 38 measured binders that are recommended. F1 is
the harmonic mean of precision and recall. Scores are shown to two decimal
places for readability; exact machine-readable values determine membership.
The score scales differ between models, and a score such as 0.80 is neither 80%
retention nor an 80% chance of binding.

| Model | Retrospective score cutoff (`score >=`) | Recommended measured pairs | Measured binders among recommendations | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Six-position additive baseline | {{VALUE}} | {{N}}/108 | {{N}}/{{N}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |
| Six-position nonlinear model | {{VALUE}} | {{N}}/108 | {{N}}/{{N}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |
| Frozen MINT layer 9 | {{VALUE}} | {{N}}/108 | {{N}}/{{N}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |
| Frozen MINT final layer | {{VALUE}} | {{N}}/108 | {{N}}/{{N}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |
| {{ADDITIONAL_DEPLOYED_MODELS}} | {{VALUE}} | {{N}}/108 | {{N}}/{{N}} | {{VALUE}} | {{VALUE}} | {{VALUE}} |

The measured panel is retrospective: its outcomes were already known while the
analysis evolved, and choices such as using R009/R010 had been informed by it.
Although the model weights do not use retention, this panel cannot serve as a
fresh validation of the complete analysis process.

## Candidate space and the two evidence tiers

Each of the four LibA Affibody sites permits 15 amino acids
(`ADEFHIKLNPQSTVY`). There are therefore `15^4 = 50,625` Affibody codes for each
peptide and `9 x 50,625 = 455,625` assignments across the nine current targets.
This is exhaustive for the stated LibA alphabet and these targets; it is not an
all-20-amino-acid search and does not add new peptide targets.

The unmeasured assignments are kept in two separate pools because they answer
different experimental questions:

1. **Pairs already supported by pooled selection counts:** 7,786 unmeasured
   current-target pairs are in the pooled R009+R010 top-2% positive set. Testing
   them checks whether strong same-pair selection evidence transfers to the
   retention assay. It is not a test of model generalization.
2. **Pairs missed by that positive rule:** after excluding the 108 measured
   pairs and those 7,786 additional selection-positive pairs, 447,731 assignments
   remain (`455,625 - 108 - 7,786`). Ranking these asks whether a model can find
   useful candidates beyond the pairs already favored by pooled selection.

The final handoff must label the source pool for every proposed pair. Results
from the two pools should not be merged into one headline success rate.

## Weak-label cutoffs used for new candidate menus

Prospective cutoffs are selected by maximizing F1 on the matched weak-label
out-of-fold predictions, not on retention. For each peptide, the candidate rule
keeps pairs at or above its locked cutoff, ranks them within that peptide, and
takes at most ten. A peptide is allowed to receive fewer than ten candidates;
below-cutoff rows are not added merely to fill capacity.

| Prospective candidate rule | Weak-label score cutoff | Model-generalization candidates | Targets represented | Missing targets |
|---|---:|---:|---:|---|
| {{PRIMARY_RULE}} | {{VALUE}} | {{N}} | {{N}}/9 | {{TARGETS_OR_NONE}} |
| {{CONTROL_RULE_1}} | {{VALUE}} | {{N}} | {{N}}/9 | {{TARGETS_OR_NONE}} |
| {{CONTROL_RULE_2}} | {{VALUE}} | {{N}} | {{N}}/9 | {{TARGETS_OR_NONE}} |

The full-precision cutoff is applied in code. Values in the public report and
candidate tables may be rounded to two decimal places for readability. Passing
the cutoff means only that the score met a rule learned from the provisional
selection labels; it does not assign a probability of binding.

## Proposed candidate menus

The complete model-generalization menu can contain at most 90 pairs: no more
than ten Affibodies for each of the nine current peptide targets. The first wet-
lab batch should preserve the locked within-peptide ranks and fit the stated
capacity of five to ten peptide targets with no more than ten Affibodies per
target. It should not be constructed by comparing raw scores across peptides,
because those score scales need not be comparable.

### Primary model-generalization menu

Scores below are shown to two decimal places. Exact scores and complete
sequences are retained in the machine-readable handoff.

| Peptide code | Ranked Affibody codes and displayed scores |
|---|---|
| AF | {{CODE (0.00), ...}} |
| DL | {{CODE (0.00), ...}} |
| DP | {{CODE (0.00), ...}} |
| EA | {{CODE (0.00), ...}} |
| KF | {{CODE (0.00), ...}} |
| LA | {{CODE (0.00), ...}} |
| LL | {{CODE (0.00), ...}} |
| NF | {{CODE (0.00), ...}} |
| TL | {{CODE (0.00), ...}} |

This menu uses `{{N_DISTINCT_AFFIBODIES}}` distinct Affibody designs across
`{{N_PAIR_ASSIGNMENTS}}` peptide--Affibody assignments. The most reused design
appears with `{{N_TARGETS}}` targets. Reuse can reduce the number of constructs
required, but heavy reuse can also indicate that the model favors a generally
strong Affibody instead of learning peptide-specific compatibility.

### Direct-selection-evidence controls

`{{DIRECT_SELECTION_CONTROL_DESCRIPTION_AND_TABLE}}`

These pairs should remain visibly separate from the model-generalization menu.
Their pooled R009/R010 counts are direct same-pair selection evidence and can be
useful experimental controls, but they cannot demonstrate recovery of a pair
that selection missed.

## Candidate checks before wet-lab release

For every proposed assignment, the handoff should contain:

- a stable sample and pair identifier;
- the peptide code and Affibody code;
- both complete model-input sequences;
- source pool: pooled-selection-positive control or model-generalization pair;
- within-peptide model rank, full-precision score, and whether it passes the
  locked weak-label cutoff;
- component-model scores for an ensemble;
- whether the exact pair or either complete partner occurred in training;
- its complete R000--R014 observation history;
- sequence flags such as introduced cysteine and N-X-S/T motifs; and
- blank fields for returned retention and assay-quality notes.

The hidden N-terminal `MA`, vector, tag, signal peptide, and physical linker
context must be confirmed with the data provider before constructs are ordered.
The displayed sequences in the computational handoff are model inputs, not a
complete manufacturing specification.

## How the new experiment becomes the actual test

The model, weak-label cutoff, source pools, and within-peptide candidate ranks
must be frozen before any new assay result is received. After testing, report
the number of technically valid pairs, the fraction with retention at least
75%, the numerical retention values, and the resulting within-peptide ranking.
Technical failures should be recorded rather than silently counted as
nonbinders.

This is the prospective evaluation: the model makes choices first and the wet
lab reveals the outcomes later. If two candidate rules are to be compared,
the experimental batch must include prespecified pairs unique to both rules;
performance on overlapping candidates alone cannot show which rule is better.

## Limitations

- The positive and negative labels are inferred from selection sequencing, not
  direct measurements of binding or retention. Expression, amplification, and
  sampling effects can change round counts.
- The separate provider pooled R009/R010 LibA file is not yet available locally
  for an independent exact-ID comparison with the reconstructed positives.
- Four of nine measured peptide rows contain no binder at the 75% threshold,
  so within-peptide AP and AUROC are informed by only five rows.
- The 108-pair panel is retrospective and has already influenced parts of the
  analysis. Its best F1 cutoff must not be called validated.
- The strict split prevents reuse of complete measured-panel partners during
  weak-label fitting, but the prospective search still uses the same nine known
  peptide targets. It does not establish transfer to novel peptide targets or a
  new library.
- LibA has no experimentally verified complex structure in the current files.
  ESMFold2 features cannot substitute for a crystal-based mapping check, and a
  useful ESMFold2 representation would not by itself prove that its predicted
  geometry is correct.
- The search covers the stated 15-amino-acid library alphabet at four designed
  Affibody sites, not arbitrary full-protein sequences.

## Files for scientific review

The final package should link the following generated artifacts rather than
embedding implementation details in this public report:

- `{{PRIMARY_CANDIDATE_MENU.csv}}`: exact ranks, scores, pair IDs, and full
  sequences for the primary model-generalization menu;
- `{{DIRECT_SELECTION_CONTROLS.csv}}`: separately identified candidates with
  direct pooled-selection support;
- `{{ALL_MODEL_MENUS.csv}}`: matched comparator menus;
- `{{PER_PAIR_RETROSPECTIVE_PREDICTIONS.csv}}`: the 108 measured pairs and all
  model scores;
- `{{METRICS_BY_SEED.json}}`: detailed results and run-to-run variability; and
- `{{BLANK_RESULT_ENTRY_TEMPLATE.csv}}`: frozen candidate IDs with empty fields
  for the new laboratory outcomes.
