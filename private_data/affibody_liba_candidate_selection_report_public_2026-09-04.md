# LibA candidate selection for prospective wet-lab testing

This handoff applies already-locked models to nine LibA peptide targets. The
primary discovery model is **Weak-label-selected frozen MINT layer 9**. Because the sequence models have
weak peptide-specific ranking on the measured LibA panel, we made a pragmatic,
retrospectively informed choice not to rely on model rank alone for the batch
composition. The batch targets three unmeasured pairs supported by
direct pooled-R009/R010 sequencing evidence and seven strict-double-cold model discoveries
per peptide when eligible candidates exist. If fewer than three evidence comparators exist,
the unused slots are filled only by model discoveries that pass the locked
threshold. A target with neither kind of candidate is explicitly reported and
omitted from the order; no comparator or discovery is fabricated. Candidate scores were
used only to determine threshold eligibility and rank discoveries within the same peptide.
Retention measurements were not used to train the models, choose the deployment
threshold, or choose exact candidates within either tier; they informed only the
pragmatic decision to mix evidence comparators and model discoveries in this first batch.
Retention is the percentage of assay signal remaining after the construct is
cleaved. It is the project's laboratory readout, not a direct measurement of
binding affinity.

## Training and evaluation data

22,542 strict weak-label pairs: 11,320 positives and 11,222 negatives; all direct evaluation peptide and Affibody identities are excluded.

Weak labels: Positives are selected after summing exact-pair R009 and R010 counts and taking the tie-inclusive top 2%. Negatives have at least three reads in R001 and never appear in any positive-selection round R002-R014. The strict split then excludes evaluation peptide and Affibody identities from training.

Split rule: Three diagonal double-cold folds; neither validation peptide identities nor validation Affibody identities occur in the corresponding training fold.

The table below uses the supplied retrospective comparison of directly measured
retention pairs. Those measurements describe prior performance; they do not
enter the prospective candidate-selection rule.

## What the wet lab should test

`ORDER_THIS_BATCH.csv` gives the exact tiered peptide--Affibody pairs. It is a
list of specified pair assays, not a Cartesian cross between every peptide and
every listed Affibody. Each row has a stable sample ID and a blank matching row
in `RESULT_ENTRY.csv`. `PRIMARY_MODEL_ONLY_BATCH.csv` preserves the original
model-only proposal for comparison, but it is not the recommended first batch.
`NO_CANDIDATE_ABOVE_LOCKED_THRESHOLD.csv` records every model--target case with
no passing candidate, including its maximum score and locked threshold. Such
rows are diagnostic and are not orderable.

The 3+7
allocation is fixed for this prospective handoff: 3
sequencing-evidence comparators provide an immediate check that the assay and
pooled-selection evidence transfer, while
7 discoveries
leave most capacity for testing strict generalization. Discoveries exclude pairs
flagged as high-confidence weak negatives. Here, that means pairs with at least
three R001 reads that never appeared in any positive-selection round R002--R014.
Selection first prefers Affibody codes
that differ at two or more of the four designed positions from comparators and
discoveries already chosen for that peptide, then relaxes only to fill otherwise
available assay slots.

| Peptide | 9-aa sequence | Batch status | Evidence-supported pairs available | Evidence comparators selected | Model discoveries selected | Total assays |
|---|---|---|---|---|---|---|
| AF | SLLAFITQV | ORDERABLE_CANDIDATES_AVAILABLE | 2103 | 3 | 7 | 10 |
| DL | SLLDLITQV | ORDERABLE_CANDIDATES_AVAILABLE | 303 | 3 | 7 | 10 |
| DP | SLLDPITQV | NO_CANDIDATE_ABOVE_LOCKED_THRESHOLD | 0 | 0 | 0 | 0 |
| EA | SLLEAITQV | ORDERABLE_CANDIDATES_AVAILABLE | 25 | 3 | 7 | 10 |
| KF | SLLKFITQV | ORDERABLE_CANDIDATES_AVAILABLE | 482 | 3 | 7 | 10 |
| LA | SLLLAITQV | ORDERABLE_CANDIDATES_AVAILABLE | 6 | 3 | 7 | 10 |
| LL | SLLLLITQV | ORDERABLE_CANDIDATES_AVAILABLE | 2604 | 3 | 7 | 10 |
| NF | SLLNFITQV | ORDERABLE_CANDIDATES_AVAILABLE | 264 | 3 | 7 | 10 |
| TL | SLLTLITQV | ORDERABLE_CANDIDATES_AVAILABLE | 1999 | 3 | 7 | 10 |

The actual mixed batch contains 80 specified pair assays and
39 distinct displayed
Affibody sequences; an Affibody can be paired with more than one peptide.

The compact table below shows the actual mixed batch. Evidence comparators show
their pooled R009+R010 counts; model discoveries show scores rounded to two
decimals. The scores are not binding probabilities or predicted retention
percentages. Values displayed as `1.00` are rounded and need not be exact ties;
the full-precision values in `ORDER_THIS_BATCH.csv` determine the order.

| Peptide | 9-aa sequence | Sequencing-evidence comparators: code (pooled count) | Model discoveries: code (score) |
|---|---|---|---|
| AF | SLLAFITQV | PPVA (count 353), PLKV (count 198), NLLS (count 197) | TQEN (1.00), THDN (1.00), TEQN (1.00), TSVN (1.00), TKHN (1.00), ASEN (1.00), TDKN (1.00) |
| DL | SLLDLITQV | PDAI (count 211), SHAY (count 168), PSIT (count 160) | TSDN (1.00), TQEN (1.00), TEQN (1.00), TNHN (1.00), ASEN (1.00), THIN (1.00), TDVN (1.00) |
| DP | SLLDPITQV | None | None passes the locked threshold |
| EA | SLLEAITQV | TTEA (count 74), PHVN (count 45), TDNH (count 45) | TSDN (0.89), TQEN (0.85), TEQN (0.80), TNHN (0.72), ASEN (0.69), TDVN (0.67), THIN (0.67) |
| KF | SLLKFITQV | EDDH (count 259), LSLL (count 100), PQIN (count 96) | TSDN (1.00), TQEN (1.00), TEQN (1.00), TKHN (1.00), THIN (1.00), ASEN (1.00), TDVN (1.00) |
| LA | SLLLAITQV | KVLI (count 32), PSKV (count 19), AIDS (count 16) | TSDN (0.97), TQEN (0.95), TEQN (0.93), TNHN (0.90), ASEN (0.87), THIN (0.87), TDVN (0.86) |
| LL | SLLLLITQV | YNII (count 208), TLDA (count 181), LAPN (count 170) | TSDN (1.00), TQEN (1.00), TKQN (1.00), TNHN (1.00), TEKN (1.00), THIN (1.00), ASEN (1.00) |
| NF | SLLNFITQV | FYII (count 122), TPVA (count 105), TVST (count 105) | TSDN (1.00), TQEN (1.00), TEQN (1.00), TNHN (1.00), THVN (1.00), ASEN (1.00), TKLN (1.00) |
| TL | SLLTLITQV | HSLH (count 260), LITK (count 254), AAHI (count 250) | TSDN (1.00), TQEN (1.00), TEQN (1.00), TNHN (1.00), THIN (1.00), ASEN (1.00), TDVN (1.00) |

### Secondary model-only comparison

The table below lists every threshold-passing entry in the separate primary
model-only menu. It is retained to show what the model alone would have selected;
it is not the recommended mixed batch.

| Peptide | 9-aa sequence | Primary model candidates in rank order: code (score) |
|---|---|---|
| AF | SLLAFITQV | TSEN (1.00), TQEN (1.00), THEN (1.00), TDEN (1.00), THDN (1.00), TEEN (1.00), TQDN (1.00), TNEN (1.00), TSQN (1.00), THPN (1.00) |
| DL | SLLDLITQV | TTEN (1.00), TSDN (1.00), TSEN (1.00), TTPN (1.00), TSSN (1.00), TSTN (1.00), TSPN (1.00), TQEN (1.00), TTSN (1.00), TSHN (1.00) |
| DP | SLLDPITQV | None passes the lock (maximum 0.21; threshold 0.40) |
| EA | SLLEAITQV | TTDN (0.91), TTEN (0.91), TPPN (0.90), TSDN (0.89), TSEN (0.89), TTPN (0.88), TSSN (0.86), TTTN (0.86), TSTN (0.86), TSPN (0.85) |
| KF | SLLKFITQV | TSDN (1.00), TSEN (1.00), TSTN (1.00), TQEN (1.00), TSHN (1.00), THEN (1.00), TDEN (1.00), THDN (1.00), TTQN (1.00), TEEN (1.00) |
| LA | SLLLAITQV | TTDN (0.97), TTEN (0.97), TSDN (0.97), TSEN (0.96), TTPN (0.96), TSSN (0.96), TSTN (0.95), TTTN (0.95), TSPN (0.95), TTSN (0.95) |
| LL | SLLLLITQV | TSDN (1.00), TQEN (1.00), TNEN (1.00), TSQN (1.00), TQDN (1.00), TKEN (1.00), TNDN (1.00), TDSN (1.00), TVEN (1.00), TESN (1.00) |
| NF | SLLNFITQV | TTDN (1.00), TTEN (1.00), TSDN (1.00), TSEN (1.00), TSPN (1.00), TSTN (1.00), TQEN (1.00), TSHN (1.00), THEN (1.00), THDN (1.00) |
| TL | SLLTLITQV | TSDN (1.00), TQEN (1.00), TSHN (1.00), THEN (1.00), TDEN (1.00), TEEN (1.00), THDN (1.00), TNEN (1.00), TQDN (1.00), TIEN (1.00) |

Each model directory also contains a separately named strict-double-cold menu:
the peptide and the Affibody identity were both absent from that model's strict
training set. It ranks only within each peptide and uses the same outcome-blind
weak-label threshold. The full selection-missed universe contains
330,880 such pairs and
116,851 peptide-cold-only
pairs. The main menu is retained as the yield-oriented view; the double-cold menu
is the stronger generalization view.

This is a computational shortlist, not yet an order-ready construct list. The
displayed 58-aa Affibody sequence is the provider-displayed/model-input sequence,
not a final vendor or cloning construct. The hidden N-terminal `MA`,
vector, and tag context must be confirmed. Likewise, the SMART--HLA--linker--
peptide sequence is a model/assay-side input; signal peptide, tag, linker, and
vector details must be verified before cloning.

## Models that can currently be used

Only models marked scientifically eligible in the supplied weak-OOF lock are
included. No result is silently hard-coded into this report.

MINT layer 9 is primary because its retention-blind weak-label out-of-fold
within-peptide AP was 0.709795, compared with
Equal-logit additive plus MINT layer 9 0.706828. The rounded table shows both values as `0.71`, but the
unrounded locked comparison selected MINT layer 9.

For the MINT model, the two full chain sequences enter a frozen MINT backbone;
layer-9 residue vectors are averaged within each chain and a trained logistic
head scores the concatenated chain averages. The equal-logit comparator averages
the MINT head's logit and the six-position additive model's logit before applying
the sigmoid;
the additive model uses peptide positions 4 and 5 plus the four designed
Affibody positions.

| Model | Prospective role | Candidate pair assays | Unique Affibody sequences |
|---|---|---|---|
| Equal-logit additive plus MINT layer 9 | Separate comparison menu | 80 | 28 |
| Weak-label-selected frozen MINT layer 9 | Primary discovery component of tiered batch | 80 | 28 |

Other sequence experiments were excluded for specific reasons: LoRA showed no
reproducible gain; PNU and later-round labels did not improve the locked
weak-label ranking objective; and the meta-gradient and trajectory-rater models
used retention information, so they are ineligible for retention-blind
prospective selection.

## Retrospective comparison on the existing retention panel

These values describe performance on already-measured pairs. Scores are not
retention percentages or calibrated binding probabilities. All displayed
metrics are rounded to two decimal places; machine-readable CSVs retain the
input precision.

| Model | Measured pairs | Within-peptide AUROC | AUROC groups | Within-peptide AP | AP groups | Within-peptide Spearman | Spearman groups | Weak-OOF within-peptide AP | Weak-OOF groups |
|---|---|---|---|---|---|---|---|---|---|
| Equal-logit additive plus MINT layer 9 | 108 | 0.21 | 5 | 0.59 | 5 | -0.37 | 9 | 0.71 | 189 |
| Weak-label-selected frozen MINT layer 9 | 108 | 0.22 | 5 | 0.59 | 5 | -0.41 | 9 | 0.71 | 189 |

At each retention-optimized descriptive cutoff, the old panel would have
produced the following recommendations. This table is explanatory, not a rule
for the new batch.

| Model | Descriptive cutoff | Pairs above cutoff | Measured binders above cutoff | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Equal-logit additive plus MINT layer 9 | 0.98 | 72 | 38 | 0.53 | 1.00 | 0.69 |
| Weak-label-selected frozen MINT layer 9 | 0.97 | 72 | 38 | 0.53 | 1.00 | 0.69 |

## Two different kinds of threshold

The weak-label deployment threshold was fixed from selection-derived weak
labels and is the only threshold used to create the prospective menus. The
retention-optimized threshold maximizes a stated retrospective criterion on an
already-measured panel; it is descriptive and must not be used to select this
batch or claimed as prospectively validated.

For the primary MINT model, 306,653 of
447,731 scored pairs pass the global weak-label threshold. Every
non-DP target has at least 1,572 passing pairs, so the
within-peptide ranking and ten-candidate cap--not the cutoff--determine those
eight menus. DP is the exception: no pair passes. This sharp peptide-to-peptide
shift is another reason not to interpret the score as a calibrated probability.

| Model | Weak-label deployment threshold | Retention-optimized descriptive threshold | How it may be used |
|---|---|---|---|
| Equal-logit additive plus MINT layer 9 | 0.42 | 0.98 | Deploy weak-label threshold only; retention-optimized threshold is descriptive |
| Weak-label-selected frozen MINT layer 9 | 0.40 | 0.97 | Deploy weak-label threshold only; retention-optimized threshold is descriptive |

## Separate pooled-selection evidence tier

The model-scored universe contains 447,731
selection-missed pairs. It is kept separate from
7,786 unmeasured current-target pairs
that already belong to pooled R009+R010's top 2%. The latter have direct sequencing
evidence but have not been confirmed as binders in the direct retention assay.
They are not model rescues. Their full menu remains separate
from every model menu. The first assay batch intentionally samples both tiers,
with an explicit tier label on every row.

On the previously measured LibA panel, ranking by raw pooled count gave:

| Measured pairs | Pooled-count cutoff | Measured binders above cutoff | Measured pairs above cutoff | Cutoff precision | Cutoff recall |
|---|---|---|---|---|---|
| 108 | 13 | 32 | 55 | 0.58 | 0.84 |

This raw-count result is not evidence of generalization to new sequence space: it
uses direct evidence from the same peptide--Affibody pair. If these pairs have not
already been tested, this separate tier may be more defensible than learned-model
rescues when the immediate goal is wet-lab yield. The candidate menus and supplied
retrospective results are interpreted separately below. No binder-yield estimate
is assigned to the unmeasured menu; the new assay is the test.

The learned-model diagnostics are:

On the already-measured 108-pair panel, both deployed rules ranked `TTHN` first for all nine peptides. That is historical behavior on the old 12-Affibody panel, not the prospective result below.

- **Equal-logit additive plus MINT layer 9:** 4 different top Affibody codes across targets; the most recurrent top-ten code(s) `TSDN` appear for 7 of 8 orderable targets; average within-peptide Spearman -0.37.
- **Weak-label-selected frozen MINT layer 9:** 4 different top Affibody codes across targets; the most recurrent top-ten code(s) `TSDN` appear for 7 of 8 orderable targets; average within-peptide Spearman -0.41.
- **Agreement between the two prospective menus:** they share an average of 6.13 of ten codes across 8 orderable targets, choose the same top code for 3/8 targets, and have the identical ordered top ten for 0/8 targets.

The learned-model recommendations therefore remain a distinct prospective test.

## Structure-model availability

- **ESMFold2: excluded by the locked weak-label gate.** No folding-derived family passed. The pre-folding sequence control scored within-peptide AP 0.660926; the best actually folding-derived family (`full`) scored 0.603381, versus 0.709795 for MINT layer 9. It therefore contributes no candidate scores.

- **RDE-PPI: unavailable.** The only local experimental complex is the LibB MW+NNYYF crystal; using that geometry as LibA ground truth would be an invalid transfer.
- **StaB-ddG: unavailable.** The only local experimental complex is the LibB MW+NNYYF crystal; no validated LibA geometry exists for this structure-derived encoder.

Unavailable structure families are omitted rather than represented by a
different model under the same name.

## How the prospective result will be read

After laboratory results return, prospective yield is the number of valid
pair assays with retention at 30 minutes of at least 75%, divided by the number
of valid returned pair assays. Report this separately for pooled-selection-evidence
comparators and strict-double-cold model discoveries, then for the combined batch.
Technical failures are excluded. The released result template contains no
retention outcomes.
