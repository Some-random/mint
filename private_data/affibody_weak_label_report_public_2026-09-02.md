# Testing whether below-cutoff selection pairs improve Affibody ranking

Affibodies are small engineered proteins designed here to bind peptide--HLA
targets. The existing sequence-only models are trained using two deliberately
conservative groups from the experimental selection rounds: high-count pairs
are provisional positives, and pairs that appear before selection but never
appear again are provisional negatives. Most observed pairs satisfy neither
rule and are discarded.

This study asks whether those discarded R009/R010 pairs can contribute as
**unlabeled** training data. An unlabeled pair is not assumed to be a binder or
a nonbinder. Instead, positive--negative--unlabeled, or PNU, training allows
the group to contain a mixture of both. We tested this first with the simple
model using only the deliberately varied amino acids, and then with frozen
MINT representations.

Every complete peptide sequence and every complete Affibody sequence in the
retention experiment was removed from training. The 108 LibA and 119 LibB
retention values were not used as training labels, for early stopping, or to
choose among the new PNU settings. Retention is the percentage of assay signal
remaining after the construct is cleaved; it is the project's laboratory
readout, not a direct measurement of binding affinity. This is still a
retrospective study because the panel had already been examined in earlier
work, including comparisons that led to the inherited 2% positive cutoff.

Adding the unlabeled pool did not improve peptide-specific ranking. For the
simple model, LibB's average within-peptide Spearman fell from 0.578 to at best
0.484 among the settings that actually used unlabeled pairs. The frozen-MINT
experiment reached the same conclusion: LibA validation rejected unlabeled
data in every tested setting, while the two LibB settings that used it were
worse than both P-versus-N controls.

## How the three training groups were defined

The same R009/R010 positive rule and conservative negative rule used in the
sequence-only report were preserved:

- **Positive, abbreviated P:** make one list of every distinct pair appearing
  in R009, R010, or both. Add the two counts, treating absence from either
  round as zero. Within each library, the highest 2% of these pooled counts,
  including ties at the boundary, are provisional positives. The minimum
  pooled count is 13 for LibA and 12 for LibB.
- **Confident negative, abbreviated N:** require at least three R001 reads and
  no appearance of that pair in any round from R002 through R014.
- **Unlabeled, abbreviated U:** use an observed R009/R010 pair whose pooled
  count is below the positive boundary. These pairs are not relabeled as
  negatives; some may be weak binders or genuine binders with low counts. U
  does not include every unobserved combination that could theoretically be
  made from a peptide and an Affibody.

All three groups use the same library-wide amino-acid-code check: each short
mutation code must contain only amino acids that are allowed somewhere in that
library's stated alphabet. This does not verify the allowed alphabet separately
at each position. We also exclude all 228 designs chosen for retention testing,
including the one LibB design without a retention measurement, and remove any
training pair sharing either complete input sequence with one of those designs.
Finally, we verify that no U pair is already in P or N.

| Library | Positive P | Confident negative N | Eligible unlabeled U | Retention measurements |
|---|---:|---:|---:|---:|
| LibA | 11,320 | 11,222 | 726,532 | 108 |
| LibB | 23,725 | 6,923 | 1,157,625 | 119 |

The U counts are numbers of distinct peptide--Affibody pairs, not observations
across rounds. For example, a pair observed in both R009 and R010 appears once
in U with its two counts added together.

## What PNU training changes

Ordinary P-versus-N training tells the model that every P example is positive
and every N example is negative. PNU training retains those two groups but also
estimates what can be learned from U without declaring each U pair to be a
negative.

Two settings need to be supplied because the true labels inside U are unknown:

- **Assumed positive fraction (`pi`):** the assumed percentage of U that would
  be genuine positives if every pair could be measured. We tested 2%, 5%, 10%,
  and 15%. These are sensitivity settings, not estimates obtained from the
  retention experiment.
- **Unlabeled-data weight (`eta`):** the mixing setting in the training loss.
  We tested `0`, `0.25`, `0.50`, `0.75`, and `1.00`. At `eta=0`, U contributes
  nothing: the result is a reweighted P-versus-N model. At `0.25`, `0.50`, and
  `0.75`, progressively more of the model's estimate of nonbinder behavior
  comes from U and less comes from the confident N group. At `eta=1`, N drops
  out of that part of the loss and training uses P and U; this is called
  positive--unlabeled, or PU, training. `eta` is not the percentage of U rows
  sampled and is not the assumed binder fraction; that separate assumption is
  `pi`.

This distinction matters when reading the results. A high-scoring row with
`eta=0` cannot show that unlabeled data helped, because that model never used
U.

## What we actually compared

To test sensitivity to the unlabeled-data weight, we ablated `eta`: for each
library and each fixed `pi` value, we trained the same model repeatedly while
changing `eta`. We then chose `eta` and the other training settings using only
held-out selection-derived labels, rather than choosing the setting that looked
best on retention.

| Experimental choice | Values compared |
|---|---|
| Assumed binder fraction in U (`pi`) | 2%, 5%, 10%, and 15% |
| Weight given to U in the loss (`eta`) | 0, 0.25, 0.50, 0.75, and 1.00 |
| Designed-position model | all 726,532 LibA or 1,157,625 LibB eligible U pairs; five identity-separated validation folds |
| Frozen-MINT model | five sampled U pairs per P pair; three identity-separated validation folds |

For the designed-position model, a separate pure-PU implementation repeated
the `eta=1` endpoint as a check; its peptide-specific ranking was also weaker
than P-versus-N training. The tables below show the one `eta` chosen by
weak-label validation for each fixed `pi`; they do not show every trained fit.

Weak-label validation chose the following patterns. For the designed-position
model, LibA selected `eta` values `0.25, 0, 0, 0` and LibB selected
`0.25, 0.25, 0.25, 0` as `pi` increased from 2% to 15%. For frozen MINT, LibA
selected `0, 0, 0, 0`, while LibB selected `0, 0, 0.25, 0.25`. Thus validation
never selected `eta=0.50`, `0.75`, or `1.00`, and it frequently chose to ignore
U entirely.

For one concrete example, take the LibB designed-position model with
`pi=5%`. We trained five branches using `eta=0`, `0.25`, `0.50`, `0.75`, and
`1.00`, while also comparing ordinary training settings inside each branch.
Held-out selection data chose `eta=0.25`. We then refitted that chosen setup on
the eligible weak-label training data and scored the 119 retention pairs. Only
at that final step did we compare its predictions with retention: AUROC was
0.872, average precision 0.874, within-peptide Spearman 0.484, and retrospective
cutoff precision/recall/F1 0.707/0.967/0.817. The P + N control remained better
overall.

## The two models tested

### Model using only the designed amino-acid changes

This is the same simple model used in the sequence-only report. For an
illustrative LibA pair, its input could be:

`peptide [position 4 = S, position 5 = Y]; Affibody [position 13 = L, position 17 = A, position 27 = V, position 31 = G]`

The model learns one number for each allowed amino acid at each varied position
and adds the corresponding numbers to score the pair. The PNU version changes
the training objective, not this input or model architecture. LibA has 120
amino-acid/position coefficient slots plus one constant term; LibB has 140
coefficient slots plus one constant term.

This experiment used every eligible U pair: 726,532 for LibA and 1,157,625 for
LibB. Whenever U had nonzero weight, every eligible U pair was visited once per
training epoch. Model settings were evaluated using five validation folds that
also kept complete peptide and Affibody identities separated.

### Classifier using frozen MINT representations

MINT receives the complete reconstructed peptide-side sequence and the complete
Affibody sequence. Its pretrained parameters remain fixed. The final MINT
layer supplies 1,280 numbers for every amino acid; these are averaged separately
over the two input sequences and joined into 2,560 numbers. One linear
classifier with 2,560 weights plus one constant term is trained on top.

Embedding all 1.88 million U pairs with MINT would be substantially more
expensive than fitting the simple model. This pilot therefore used one fixed
random sample of five U pairs per positive pair: 56,600 LibA U pairs and
118,625 LibB U pairs. Those samples cover 7.79% and 10.25% of the eligible U
pools, respectively. The sample and training seed were fixed without reference
to retention outcomes before these PNU metrics were computed; this was a
one-sample, one-seed pilot rather than a complete U experiment.

## Evaluation against direct retention measurements

The evaluation set and strict train/test separation are identical to those in
the sequence-only report. A measured retention of at least 75% is treated as a
binder for AUROC and average precision. The 75% rule is used only to interpret
the retention measurements; it does not define P, N, or U.

- **AUROC** measures how often a measured binder is scored above a measured
  nonbinder when all possible score cutoffs are considered.
- **Average precision** summarizes the precision--recall curve and gives more
  weight to placing binders near the high-scoring end.
- **At one retrospective cutoff**, precision is the fraction of pairs above
  the cutoff that are measured binders, recall is the fraction of all measured
  binders that are above it, and F1 balances those two values. As in the other
  public reports, we show the cutoff that maximizes F1 on this same retention
  panel.
- **Average within-peptide Spearman** first compares the Affibody ranking with
  measured retention separately for each peptide, then averages those
  correlations. This is the most direct reported measure of whether the model
  can choose among Affibodies for one specified peptide.

The F1-maximizing cutoff is a retrospective description, not a validated rule
for new candidates. The reported metrics are AUROC, average precision, average
within-peptide Spearman, and precision/recall/F1 at this descriptive cutoff.

## Results using every eligible unlabeled pair

For each library and each assumed positive fraction, `eta`, regularization, and
training epoch were chosen using selection-derived validation AUROC, followed
by average precision and log loss to break ties. The established P + N control
keeps its earlier model choice, which used selection-derived validation log
loss rather than AUROC as the first criterion. The data and strict split are
the same, but the model-selection rules are therefore not perfectly matched.
All four assumed positive fractions are retained in the table; retention was
not used to choose one.

| Library | Training | Assumed positive fraction in U | `eta` | AUROC | Average precision | Average within-peptide Spearman | Cutoff precision / recall / F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| LibA | Established P + N | -- | -- | 0.670 | 0.428 | -0.336 | 0.535 / 1.000 / 0.697 |
| LibA | P + N + U | 2% | 0.25 | 0.689 | 0.479 | -0.455 | 0.521 / 1.000 / 0.685 |
| LibA | Reweighted P + N; U ignored | 5% | 0.00 | 0.657 | 0.420 | -0.377 | 0.528 / 1.000 / 0.691 |
| LibA | Reweighted P + N; U ignored | 10% | 0.00 | 0.664 | 0.430 | -0.364 | 0.528 / 1.000 / 0.691 |
| LibA | Reweighted P + N; U ignored | 15% | 0.00 | 0.663 | 0.430 | -0.381 | 0.528 / 1.000 / 0.691 |
| LibB | Established P + N | -- | -- | 0.900 | 0.906 | 0.578 | 0.797 / 0.850 / 0.823 |
| LibB | P + N + U | 2% | 0.25 | 0.862 | 0.865 | 0.426 | 0.709 / 0.933 / 0.806 |
| LibB | P + N + U | 5% | 0.25 | 0.872 | 0.874 | 0.484 | 0.707 / 0.967 / 0.817 |
| LibB | P + N + U | 10% | 0.25 | 0.867 | 0.868 | 0.484 | 0.699 / 0.967 / 0.811 |
| LibB | Reweighted P + N; U ignored | 15% | 0.00 | 0.917 | 0.921 | 0.658 | 0.786 / 0.917 / 0.846 |

LibA's only weak-selected setting that uses U improves AUROC and average
precision but makes its already negative within-peptide correlation more
negative, from -0.336 to -0.455. It therefore separates the 75%-retention
classes somewhat better across the full panel while ordering Affibodies worse
within a specified peptide. Its retrospective F1 also falls from 0.697 to
0.685.

Every LibB setting that actually uses U is worse than the established model on
AUROC, average precision, within-peptide Spearman, and retrospective F1. The
best within-peptide result among them is 0.484, compared with 0.578 without U.

Pure PU training, which sets `eta=1` and does not use the confident N group for
its negative-risk estimate, was weaker still. Across the four assumed positive
fractions, LibB's within-peptide Spearman was 0.138 and LibA's ranged from
-0.418 to -0.397. In this setup, removing the confident N contribution in
favor of U did not help.

## Why the high `eta=0` result is not evidence for U

The LibB row with AUROC 0.917 and within-peptide Spearman 0.658 looks better
than the established model, but `eta=0` means that no U example affects its
weights. Its only change is to give P and N different total weights according
to the assumed 15% positive fraction.

Because this row was fitted with a gradient-based optimizer, we repeated the
same weighted P-versus-N comparison with a converged solver for a convex
logistic-regression problem. All regularization settings were again selected
from weak labels before retention was loaded.

| Library | P-versus-N weighting | AUROC | Average precision | Average within-peptide Spearman | Cutoff precision / recall / F1 |
|---|---|---:|---:|---:|---:|
| LibA | Converged solver, equal class influence | 0.664 | 0.423 | -0.322 | 0.528 / 1.000 / 0.691 |
| LibA | Converged solver, assumed 15% positive class | 0.680 | 0.437 | -0.350 | 0.543 / 1.000 / 0.704 |
| LibB | Converged solver, equal class influence | 0.899 | 0.905 | 0.578 | 0.797 / 0.850 / 0.823 |
| LibB | Converged solver, assumed 15% positive class | 0.896 | 0.899 | 0.585 | 0.820 / 0.833 / 0.826 |

The apparent LibB gain did not reproduce in this converged comparison. The
15%-weighted model changes within-peptide Spearman from 0.578 to 0.585 while
slightly reducing AUROC and average precision. A 5%-weighted
LibB fit reached within-peptide Spearman 0.625, but its selection-derived
validation score was lower than the established weighting. Choosing it because
of its retention result would amount to choosing a model on the evaluation
data.

This audit does not change the conclusion about U: none of these `eta=0`
models uses an unlabeled example.

## Results using frozen MINT representations

The historical P-versus-N row is the exact frozen-MINT result from the
sequence-only report, where regularization was selected by selection-derived
log loss. To make the PNU comparison more direct, we also refitted P-versus-N
training while selecting regularization by selection-derived AUROC, the same
primary criterion used for PNU. This second row is a matched rebaseline, not a
replacement for the public sequence-only result.

| Library | Training | Assumed positive fraction in U | `eta` | AUROC | Average precision | Average within-peptide Spearman | Cutoff precision / recall / F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| LibA | Historical P + N control | -- | -- | 0.695 | 0.448 | -0.304 | 0.529 / 0.974 / 0.685 |
| LibA | AUROC-selected P + N rebaseline | -- | -- | 0.695 | 0.448 | -0.304 | 0.529 / 0.974 / 0.685 |
| LibA | Reweighted P + N; U ignored | 2% | 0.00 | 0.724 | 0.484 | -0.216 | 0.535 / 1.000 / 0.697 |
| LibA | Reweighted P + N; U ignored | 5% | 0.00 | 0.728 | 0.485 | -0.222 | 0.544 / 0.974 / 0.698 |
| LibA | Reweighted P + N; U ignored | 10% | 0.00 | 0.697 | 0.451 | -0.201 | 0.528 / 1.000 / 0.691 |
| LibA | Reweighted P + N; U ignored | 15% | 0.00 | 0.691 | 0.442 | -0.317 | 0.536 / 0.974 / 0.692 |
| LibB | Historical P + N control | -- | -- | 0.890 | 0.890 | 0.525 | 0.725 / 0.967 / 0.829 |
| LibB | AUROC-selected P + N rebaseline | -- | -- | 0.906 | 0.911 | 0.542 | 0.775 / 0.917 / 0.840 |
| LibB | Reweighted P + N; U ignored | 2% | 0.00 | 0.878 | 0.881 | 0.492 | 0.716 / 0.967 / 0.823 |
| LibB | Reweighted P + N; U ignored | 5% | 0.00 | 0.884 | 0.884 | 0.428 | 0.720 / 0.983 / 0.831 |
| LibB | P + N + sampled U | 10% | 0.25 | 0.852 | 0.858 | 0.375 | 0.730 / 0.900 / 0.806 |
| LibB | P + N + sampled U | 15% | 0.25 | 0.857 | 0.860 | 0.366 | 0.761 / 0.850 / 0.803 |

LibA validation selected `eta=0` for every assumed positive fraction, so no
LibA row in the PNU grid provides evidence about using U. LibB validation also
selected `eta=0` at 2% and 5%. At 10% and 15%, it selected `eta=0.25`, but both
U-using models were worse than both P-versus-N controls on AUROC, average
precision, within-peptide Spearman, and retrospective F1.

The result is therefore negative for both input representations tested. The
simple model used all eligible U pairs, and frozen MINT used one deterministic
sample. Neither showed that the current below-cutoff pool improves the ranking
needed for this project.

## Why the unlabeled pool may not behave like standard PU data

P and U were created by splitting the same pooled R009/R010 count distribution
at a high-count boundary. P is deliberately the highest-count tail, while U is
the lower-count remainder. Labeled positives are therefore not a random or
representative sample of every genuine binder that might be hidden inside U.

This matters because conventional PU and PNU reasoning commonly assumes that
the labeled positives resemble the full positive population. Here, a binder in
U may differ systematically from a binder in P because of expression,
amplification, sequencing depth, or weaker selection enrichment. N is also a
deliberately conservative group and may be easier than the unknown nonbinders
inside U.

The tested values of the positive fraction are therefore sensitivity analyses,
not calibrated estimates of how many U pairs bind. The model scores are not
retention percentages or binding probabilities.

N is also a weak label rather than a direct binding measurement. A pair can
disappear after R001 because of sampling, sequencing, expression, or
amplification effects as well as poor binding. The three-read rule reduces this
risk but cannot eliminate it.

## Robustness and limitations

- The designed-position model used every eligible U pair and five
  identity-separated validation folds. The frozen-MINT pilot used three folds
  and only one fixed sample of U. The MINT result therefore tests that sample,
  not every possible use of the full unlabeled pool.
- Four frozen-MINT settings with `eta=0` were allowed extra training passes
  after reaching the original boundary, whereas nonzero-`eta` settings were
  not. This favors models that ignore U, so the comparison is not perfectly
  symmetric. It cannot manufacture evidence that U helped, but it limits how
  broadly the negative MINT result should be generalized.
- The strongest-looking `eta=0` result was checked with a separate converged
  P-versus-N fit. That check was necessary because `eta=0` changes class
  weighting but contains no unlabeled-data contribution.
- The retention panel is a retrospective evaluation. R009/R010 and several
  analysis choices were selected after the retention measurements were already
  available. In particular, the established 2% positive cutoff had previously
  been compared with 1% and 5% using this panel. No retention value entered the
  new PNU model fits or their setting choices.
- LibB contains the broadly successful `FALTA` Affibody, while LibA has negative
  within-peptide correlations for every primary sequence model. These properties
  limit how strongly either library can establish peptide-specific
  generalization.

## What the results show

The below-cutoff R009/R010 pool should not be added to the current predictor
based on these experiments. In LibB, every weak-selected full-U simple-model
setting and every sampled-U MINT setting that actually used U reduced
peptide-specific ranking. In LibA, the one full-U setting selected by weak
validation made the within-peptide ranking worse, while frozen-MINT validation
chose to ignore U entirely.

Reweighting the existing P and N examples remains a separate modeling choice.
It can change retrospective results, but it is not evidence that weakly labeled
U examples contributed information. A future weak-label study would need a
more defensible estimate of the positive fraction and a labeling process in
which known positives are more representative of the positives hidden in U.
