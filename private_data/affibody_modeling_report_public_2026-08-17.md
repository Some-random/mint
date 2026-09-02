# Using selection sequencing data to rank Affibodies

Affibodies are small engineered proteins designed here to bind peptide--HLA
targets. This study asks whether sequencing data collected during experimental
selection can help rank peptide--Affibody combinations before laboratory
testing. We converted the selection history into provisional positive and
negative training examples, trained one model using only the deliberately
varied amino acids, another using frozen MINT representations, and a third that
updates a small part of MINT through LoRA. We compared their rankings with 227
laboratory retention measurements. The retention values were not used as
training labels or to optimize model parameters. Retention is the percentage of
assay signal remaining after the construct is cleaved; it is the project's
laboratory readout, not a direct measurement of binding affinity.

In the strict evaluation, every complete peptide-side sequence and every
complete Affibody sequence in the retention experiment was removed from
training. LibB performed substantially better than LibA. All three LibB models
separated binders from nonbinders well when all possible score thresholds were
considered: AUROC ranged from 0.890 to 0.900 and average precision from 0.890
to 0.906. LibA was much weaker, and its negative within-peptide correlations
show that it still did not order Affibodies reliably for a specified peptide.
Frozen MINT and one epoch of LoRA did not improve the overall conclusion over
the simpler model.

One unusually strong LibB Affibody, `FALTA`, is important for interpreting
these results. It had at least 75% retention for 11 of the 12 peptides and was
tied for or achieved the highest measured retention for 10 of them. All three
models ranked it first for every LibB peptide. This is real and useful
experimental performance, so `FALTA` remains in the evaluation. However, it
also leaves little room for the current retrospective panel to distinguish
better models by their top choices.

## Training data from the selection rounds

The R000--R014 sequencing files contain 27,821,822 rows. Each row reports how
many reads were observed for one peptide--Affibody pair in one round. If the
same pair appears in five rounds, it contributes five rows. These are therefore
not 27.8 million independently labeled training examples.

## How positive and negative training examples were defined

The rounds are used differently for the two training classes:

- **Positive training example:** first make a list of every distinct pair that
  appears in R009, R010, or both. Add that pair's two counts, treating absence
  from either round as zero. Within each library, label the highest 2% of these
  sums as positive, including all ties at the boundary. The resulting minimum
  sum is 13 for LibA and 12 for LibB.
- **Negative training example:** require an appearance in R001 and no appearance
  of that pair in any round from R002 through R014.
- R000 is not used to define either class. A pair that satisfies neither rule is
  not used for training.

We also exclude all 228 peptide--Affibody designs in the retention-test matrix,
including the one LibB design without a retention measurement. Before fitting,
we require at least three R001 reads for a negative and keep only short mutation
codes composed of amino acids allowed somewhere in that library's stated
alphabet. This is not a position-by-position check of the library design.

The three-read requirement was added by our analysis, not supplied by the
experimental team. Its rationale was that pairs seen only once or twice in
R001 may disappear later through sequencing or sampling noise. The exact value
three was not scientifically calibrated: it reduces the provider-defined
negative candidates from 1,246,608 to 18,943 before the amino-acid-code check,
a 98.48% reduction. The reported models use this threshold throughout;
sensitivity to the threshold has not yet been measured.

| Data at each stage | LibA | LibB | Total |
|---|---:|---:|---:|
| Raw observations across R000--R014 | 12,736,262 | 15,085,560 | 27,821,822 |
| Pairs that can be called positive or negative using the rules above, after excluding the 228 designs chosen for retention testing | 549,883 | 747,618 | 1,297,501 |
| Pairs left after the amino-acid-code check and the three-read rule for negatives | 31,007 | 38,710 | 69,717 |
| Pairs used to train the primary models after removing every test peptide and test Affibody | 22,542 | 30,648 | 53,190 |

The primary LibA training set contains 11,320 positives and 11,222 negatives.
The primary LibB training set contains 23,725 positives and 6,923 negatives.

These positive and negative labels come only from the selection rounds. They do
not use the laboratory retention values or the 75% retention cutoff.

## The models

Consider an illustrative LibA variant with amino acids S and Y at the two
varied peptide positions and L, A, V, and G at the four varied Affibody
positions. This example shows the input format; it is not one of the reported
experimental rows.

### Model using only the designed amino-acid changes

For the illustrative variant, this model receives:

`peptide [position 4 = S, position 5 = Y]; Affibody [position 13 = L, position 17 = A, position 27 = V, position 31 = G]`

It sees only these six amino acids. It learns a separate number for every amino
acid at every varied position and adds the six corresponding numbers to score
the pair. It does not see the unchanged sequence or explicitly model a special
combination between peptide and Affibody positions.

The LibA version has 120 possible amino-acid/position coefficient slots plus
one constant term. The LibB version has 140 coefficient slots plus one constant
term.

### Model using frozen MINT representations

For the same illustrative variant, MINT receives the complete reconstructed
SMART--HLA--linker--peptide sequence containing S and Y as its first input, and
the complete Affibody sequence containing L, A, V, and G as its second input.
The linker inside the supplied SMART--HLA--linker--peptide construct remains;
the experimental linker connecting that construct to the Affibody is omitted.
No TCR is included.

MINT's 813 million pretrained weights are not changed. For the primary frozen
MINT model, we use its final transformer layer to produce 1,280 numbers for
every amino acid. For each of the two inputs, we average those numbers across
its amino acids; joining the two 1,280-number averages gives 2,560 inputs to a
new binary classifier. That classifier has 2,560 trained weights plus one
constant term and is fitted on this project's positive and negative examples.
MINT's original output layer is not used because it predicts masked amino
acids, not whether a pair meets our selection rule.

The simple model and the frozen MINT model use the same training examples and
the same rules for choosing model settings. The MINT classifier nevertheless
has about 18--21 times as many newly trained weights as the simple classifier.

### Model with one epoch of partial MINT fine-tuning

This model starts from the frozen MINT model and classifier described above.
It adds rank-2 LoRA adapters to the query and value projections in the
cross-chain attention modules of MINT's last two blocks. The classifier and
adapters--23,041 parameters in total--are then updated jointly for exactly one
prespecified epoch on the same complete training set, using the same class
weighting. The other MINT parameters remain fixed.

This is one exploratory run. We did not repeat it or choose the epoch using
retention results, so no run-to-run uncertainty is available. Because the
classifier and adapters were updated together, this comparison measures the
complete one-epoch fine-tuning procedure; it does not isolate the adapters'
contribution.

## Evaluation on peptide and Affibody sequences absent from training

The laboratory selected 228 designs for retention testing. One LibB design has
no retention measurement, leaving 108 measured LibA pairs and 119 measured
LibB pairs. These 227 measurements form the evaluation set, and the primary
analysis uses all of them.

Before fitting the models, we remove every training pair containing any
complete SMART--HLA--linker--peptide sequence or any complete Affibody sequence
found in that library's retention experiment. The models therefore score test
pairs for which neither complete input sequence appeared in training. They can
still learn the effects of individual amino acids from other variants in the
same library; this is not a test on a different library or protein design.

The 108 LibA and 119 LibB retention measurements form the evaluation set. We
use the experimental definition that retention of at least 75% is a binder.
We evaluate the scores in three complementary ways:

- **Across all possible score cutoffs:** AUROC measures how often a binder is
  scored above a nonbinder. Average precision summarizes the precision--recall
  curve and gives more weight to placing binders near the high-scoring end.
  Neither metric requires choosing one cutoff.
- **At one retrospective cutoff:** a pair is classified as a potential binder
  when its model score is at least the cutoff. Precision is the fraction of
  pairs above the cutoff that are measured binders; recall is the fraction of
  all measured binders that are above the cutoff. For a concrete description
  of the current data, we report the cutoff that maximizes F1, the harmonic
  mean of precision and recall.
- **Average within-peptide Spearman:** for each peptide, compare the complete
  ordering of its Affibodies by model score with their ordering by measured
  retention, then average the correlation across peptides. A value of 1 means
  perfect agreement, 0 means no consistent ordering, and a negative value
  means that the ranking tends to run backward.

The F1-maximizing cutoff is selected using these same retention measurements.
It describes the most favorable operating point on this panel, but it is not a
validated cutoff for new candidates. The model scores are not predicted
retention percentages or calibrated binding probabilities. A cutoff of 0.987,
for example, does not mean a 98.7% chance of binding, and numerical cutoffs
cannot be compared across models.

**Retrospective** means that this evaluation uses a retention panel whose
results were already known while parts of the analysis were being developed.
Retention was not used to fit the model weights, but it was used to choose the
displayed score cutoff and had informed earlier choices such as using R009 and
R010 and labeling the highest 2% as positive. The reported numbers may
therefore be optimistic.

The actual test will be **prospective**: first lock the model, score cutoff, and
candidate-selection rule; then apply them to new peptide--Affibody candidates;
finally use new wet-lab retention measurements to determine whether the method
works. Those new measurements, not this reused panel, will be the real test.

## Results when neither test sequence appeared in training

All rows use the same 108 LibA measurements or 119 LibB measurements. All
models use the same 22,542 eligible LibA or 30,648 eligible LibB training pairs
and the same class weighting. The LoRA result is one exploratory fit using one
prespecified training epoch.

| Library | Model | AUROC | Average precision | Average within-peptide Spearman |
|---|---|---:|---:|---:|
| LibA | Model using only varied positions | 0.670 | 0.428 | -0.336 |
| LibA | Frozen MINT representation + classifier | 0.695 | 0.448 | -0.304 |
| LibA | MINT + classifier after one LoRA epoch | 0.697 | 0.447 | -0.301 |
| LibB | Model using only varied positions | 0.900 | 0.906 | 0.578 |
| LibB | Frozen MINT representation + classifier | 0.890 | 0.890 | 0.525 |
| LibB | MINT + classifier after one LoRA epoch | 0.893 | 0.896 | 0.535 |

The following table applies the score directly as a recommendation rule. Each
cutoff was chosen to maximize F1 on the same retention panel shown in the
table.

| Library | Model | Best retrospective score cutoff | Pairs scoring above cutoff | Binders among those pairs | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| LibA | Model using only varied positions | 0.987 | 71/108 | 38/38 | 0.535 | 1.000 | 0.697 |
| LibA | Frozen MINT representation + classifier | 0.979 | 70/108 | 37/38 | 0.529 | 0.974 | 0.685 |
| LibA | MINT + classifier after one LoRA epoch | 0.978 | 70/108 | 37/38 | 0.529 | 0.974 | 0.685 |
| LibB | Model using only varied positions | 0.795 | 64/119 | 51/60 | 0.797 | 0.850 | 0.823 |
| LibB | Frozen MINT representation + classifier | 0.575 | 80/119 | 58/60 | 0.725 | 0.967 | 0.829 |
| LibB | MINT + classifier after one LoRA epoch | 0.572 | 81/119 | 58/60 | 0.716 | 0.967 | 0.823 |

The cutoff rule can retain fewer candidates when scores are low. In LibB, zero
to ten Affibodies per peptide score above the simple model's reported cutoff;
for frozen MINT and LoRA, the range is one to ten. On a larger prospective
candidate pool, the laboratory could apply the cutoff and then test at most its
capacity, such as the ten highest-scoring passing candidates.

LibA illustrates why threshold metrics and within-peptide ranking must be read
together. The simple model recovers all 38 measured binders at its best
retrospective cutoff, but it recommends 71 of 108 pairs and largely accepts or
rejects whole peptide rows. Its negative within-peptide Spearman shows that it
still cannot reliably choose between Affibodies for the same peptide.

In LibB, the simple model has the highest AUROC, average precision, and
precision at its own best cutoff. Frozen MINT recovers more binders and has the
highest F1 at its own best cutoff, but it recommends 80 rather than 64 pairs.
Because every cutoff was optimized on the evaluation labels, these operating
points should be treated as descriptive tradeoffs rather than evidence that
one will transfer better to new peptides.

`FALTA` explains an important part of the strong LibB result. Its direct
retention is at least 75% for 11 of 12 peptides. The exception is `DP`, for
which none of the ten measured Affibodies passes 75%. `FALTA` is therefore a
genuinely useful candidate and should not be removed or penalized. At the same
time, repeatedly recognizing one broadly effective Affibody is easier than
learning which substitutions work specifically for each peptide. The complete
`FALTA` Affibody was absent from the 30,648 training pairs, although the five
individual amino-acid choices and closely related Affibodies did occur in
training.

One epoch of LoRA changes average within-peptide Spearman by only +0.003 in
LibA and +0.010 in LibB relative to frozen MINT. Its AUROC and average
precision changes are also small. Because LoRA was run once, these differences
should not be interpreted as a reproducible improvement. In LibB, the simple
varied-position model still has the highest within-peptide Spearman.

## Why LibB scores much higher than LibA

The unusually strong measured performance of `FALTA` is one reason LibB looks
much better than LibA. Recognizing a broadly effective Affibody is practically
valuable, but it does not by itself show that a model learned peptide-specific
compatibility. LibB's positive within-peptide Spearman indicates that the
full rankings contain useful information, although the metric can still benefit
from Affibodies that work broadly across peptides. The analysis does not
establish why LibB is easier, although three observations make the difference
plausible:

1. The data provider explained that LibB was designed from a crystal structure
   and is expected to preserve the original binding mode. LibA was designed
   from a predictive model, and its actual binding mode is unknown. The varied
   LibB positions may therefore correspond more reliably to the real binding
   interface. This is a biological hypothesis, not something proven by the
   current model.
2. After removing all test sequences, LibB still has 23,725 positive training
   examples, compared with 11,320 for LibA.
3. When R009+R010 counts are used directly to rank Affibodies separately within
   each peptide, their average Spearman correlation with retention is about
   0.55 in LibB and 0.20 in LibA. The selected rounds therefore preserve the
   within-peptide Affibody ordering more clearly in LibB. This same-pair
   comparison is descriptive only: it was calculated after the retention
   results were available and is not the held-out model result above.

The fact that the simple amino-acid model matches or exceeds MINT is consistent
with reusable effects at the deliberately varied LibB positions. It does not
show that LibB performance will transfer to another library, HLA, or assay.

## Ablations and robustness checks

We varied the label rules, class balance, train/test separation, which frozen
MINT layer supplied the representation, and whether the libraries were fitted
together to determine which choices materially affect the conclusion.

- **Could a positive label be caused by only one unusually high round?** We
  repeated training while requiring at least three reads in both R009 and R010.
  We also removed Affibodies labeled positive with many different peptides to
  test whether nonspecific binding or display effects created easy positives.
  Neither change altered the conclusion.
- **Could the class proportions drive the result?** LibB has many more positive
  than negative training examples. We compared giving the two classes equal
  total influence, leaving every example equally weighted, and randomly keeping
  equal numbers of positives and negatives. The conclusion was unchanged;
  forcing equal counts discarded 16,802 LibB positives.
- **Could an earlier MINT layer preserve more useful sequence information?** We
  compared layers 1, 5, 9, 13, 17, 21, 25, 29, and 33, choosing the layer and
  classifier setting by three-fold validation on the selection-derived labels.
  Retention was not used for this choice. LibA selected layer 9, but its AUROC,
  average precision, and within-peptide Spearman were 0.630, 0.394, and -0.407,
  worse than the final layer's 0.695, 0.448, and -0.304. LibB selected layer 5:
  AUROC and average precision rose from 0.890 and 0.890 to 0.907 and 0.912,
  while within-peptide Spearman changed only from 0.525 to 0.530. At each
  model's retrospectively optimized F1 cutoff, precision and recall changed
  from 0.529 and 0.974 to 0.528 and 1.000 in LibA, and from 0.725 and 0.967 to
  0.775 and 0.917 in LibB. The earlier layer therefore did not clearly improve
  peptide-specific ranking.
- **Could the model be memorizing peptides or Affibodies seen during training?**
  We compared an easier test, where the exact pair was absent but both partners
  had appeared separately in training, with the strict test that removed every
  test peptide and test Affibody. On the same 98 LibB measurements, frozen
  MINT's average within-peptide Spearman fell from 0.728 to 0.449 under the
  strict split. This decline is why the strict version is used in the main
  table, where no complete test sequence is present in training. The primary
  LibB result uses all 119 measurements rather than only those 98 shared rows.
- **Could LibA and LibB improve by sharing one model?** The libraries vary
  different Affibody positions and were designed differently. A model forced to
  share the same amino-acid effects did not improve LibB's within-peptide
  ranking or correct LibA's negative within-peptide ranking. The reported
  results therefore fit the two libraries separately.
- **Could performance come from obviously inactive peptides or Affibodies?** We
  kept only negative pairs whose peptide appears in a positive pair with another
  Affibody and whose Affibody appears in a positive pair with another peptide.
  This tests whether the model can distinguish the pairing rather than simply
  reject a component that is never positive. It leaves only 1,764 LibA and 562
  LibB negatives and does not consistently improve the choice within a peptide,
  so it is not used for the primary result.

## What the results do and do not show

The training labels are inferred from sequencing selection rather than directly
measured binding. A pair may disappear because it binds poorly, expresses
poorly, amplifies inefficiently, or is missed during sequencing. R009/R010 and
several analysis settings were also selected after examining the existing
retention results, so the reported performance may be optimistic.

The primary test rules out reuse of the complete peptide and Affibody sequences
in training. It does not test a new library, protein design, HLA, assay, or
experimental batch. The current matrix contains only 9--12 Affibodies per
peptide, so it cannot demonstrate the intended task of selecting approximately
ten candidates from a much larger prospective pool. The best score cutoffs
were also chosen on this same matrix, and `FALTA` performs well for almost every
measured LibB peptide. The actual validation will be a new wet-lab experiment
in which the model and recommendation rule are locked before any new retention
result is known.
