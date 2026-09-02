# Testing ESMFold2 representations for LibB Affibody ranking

This study asks whether information produced inside a protein-folding model can
improve the choice of an Affibody for a peptide--HLA target. We kept ESMFold2
frozen, extracted several kinds of information from the peptide--Affibody
interface, and trained small classifiers on the same 30,648 selection-derived
LibB training pairs used by the existing sequence models. We then compared the
resulting rankings with 119 direct retention measurements. Retention was not
used to train the classifiers, choose their settings, or decide when to stop
training.

The strongest classifier built from an ESMFold2 intermediate was numerically
higher on the average ordering of Affibodies within a peptide: its Spearman
correlation was 0.642 ± 0.006, compared with 0.578 for the existing
designed-position model. However, that intermediate is created *before*
ESMFold2's folding calculations. It is an atom/residue input encoding, not a
predicted three-dimensional structure. This classifier also has 35,553 newly
trained parameters and nonlinear peptide--Affibody combinations, compared with
141 additive coefficients in the control, so the ranking difference cannot yet
be attributed specifically to the ESMFold2 representation. Its AUROC and
average precision were 0.887 ± 0.011 and 0.894 ± 0.014, compared with 0.900
and 0.906 for the existing model.

The features extracted from ESMFold2's folding calculations at the tested
peptide--Affibody block--its predicted distance distribution and its richer
residue-pair representation--did not establish an improvement over the
sequence controls. The current
experiment therefore motivates a matched test of richer nonlinear sequence
models. It does not support the claim that these predicted-structure features
improve LibB candidate selection, and it does not justify scaling immediately
to a separate structure prediction for every variant.

## Why protein structure might help

A protein chain is a connected sequence of amino acids. Each amino acid in the
chain is also called a residue. The sequence tells us the order of those
residues, but a protein normally folds into a three-dimensional shape. Residues
that are far apart in the written sequence can therefore become neighbors in
space.

When two proteins bind, parts of their surfaces meet. The residues near that
meeting surface form an interface. This experiment focuses on possible
contacts between the nine-amino-acid peptide presented by HLA and the
58-amino-acid Affibody. Changing only a few residues can alter the shape,
charge, flexibility, or chemical compatibility of that interface. A folding
model could in principle make those relationships easier for a predictor to
recognize than a model that sees only the seven deliberately varied residue
identities. The reference LibB crystal also shows direct HLA--Affibody
contacts; those HLA--Affibody residue-pair features were not retained in this
experiment.

Proximity alone is not a binding score. Two residues being close does not imply
that the complete pair binds well: their orientation and chemistry also matter,
and an apparently favorable local contact can coexist with an unfavorable
interface elsewhere. A predicted distance is also a model estimate rather than
an experimentally measured structure.

An earlier pilot compressed predicted structure into a few selected contact
values. That asks a narrow question but may discard most of the useful signal.
The present experiment instead tests the folding model's larger internal
representations and lets a supervised classifier learn which parts are useful.
This follows the general motivation of
[PreFold-dG](https://academic.oup.com/bioinformatics/article/42/Supplement_2/btag489/8767279),
which uses frozen folding-model intermediates rather than only final
coordinates. This is not a reproduction of PreFold-dG: that method uses
Boltz-2 representations to predict physical binding free energy, whereas this
study uses ESMFold2 and predicts the project's selection-derived binary label.
No physical binding-energy labels were created or inferred here.

## Training data from the selection rounds

The LibB R000--R014 sequencing files contain 15,085,560 rows. A row reports the
read count for one peptide--Affibody pair in one experimental round. The same
pair can appear in several rounds, so these rows are repeated observations over
time rather than 15.1 million independently labeled training examples. The
27,821,822-row total in the earlier report includes both LibA and LibB; this
experiment is LibB only.

The training labels use exactly the same definitions as the current LibB
sequence comparison:

- **Positive training example:** make one list of the distinct pairs observed
  in R009, R010, or both. For each pair, add its R009 and R010 counts, treating
  absence from one round as zero. Within LibB, call the highest 2% of these
  pooled counts positive, including ties at the boundary. The resulting LibB
  cutoff is a pooled count of 12.
- **Negative training example:** require the pair to appear in R001 and never
  appear in any round from R002 through R014. The analysis also requires at
  least three reads in R001, reducing the chance that a one- or two-read
  observation is treated as reliable evidence of disappearance.
- R000 does not define either class. A pair that meets neither rule is not used
  for training.

The three-read cutoff was introduced by the analysis rather than supplied or
experimentally calibrated by the data provider. It is held fixed here so the
structure-feature comparison does not change the existing label definition.

All designs from the direct-retention matrix are excluded before training. We
also keep only mutation codes made from amino acids allowed by the stated LibB
library design. Finally, the strict split removes every training pair
containing either a complete peptide-side sequence or a complete Affibody
sequence used in the retention experiment.

| Data at each stage | Number of LibB rows or pairs |
|---|---:|
| Sequencing observations across R000--R014 | 15,085,560 |
| Distinct pairs that meet the positive or negative rule after the retention designs are excluded | 747,618 |
| Pairs left after the amino-acid-code check and the three-read rule for negatives | 38,710 |
| Pairs used for strict training after all evaluation peptide and Affibody sequences are removed | 30,648 |
| Direct-retention evaluation pairs | 119 |

The 30,648 training pairs contain 23,725 positives and 6,923 negatives. The 119
evaluation pairs contain 60 pairs with retention of at least 75% and 59 below
75%. Feature extraction therefore covered 30,767 sequence pairs in total, but
only the 30,648 training rows carried labels during model fitting.

These are weak training labels: selection and sequencing are related to
binding, but they can also reflect expression, amplification, experimental
sampling, and other processes. The 75% retention cutoff is not part of the
training-label definition. It is used only when the final retention experiment
is converted into passing and failing pairs for evaluation.

## The complete sequences given to ESMFold2

The updated provider sequence files and numbering presentation were checked
against every one of the 30,767 inputs. Each example contains two separate
chains:

1. a 270-residue SMART--HLA--linker--peptide chain, whose final nine residues
   are the peptide; and
2. a 58-residue Affibody chain.

No T-cell receptor or additional protein chain is included. The experimental
linker that connects the pMHC construct to the Affibody (`GGSLEVLFQGPGSG`) is
omitted, so ESMFold2 treats the pMHC-side construct and Affibody as distinct
partners.

The released input helper prepares one chain at a time, so this project uses a
tested wrapper that combines those official features and assigns the two chains
separate identifiers. It supplies only the two current sequences, without a
paired multiple-sequence alignment--a collection of related sequence pairs
that some folding systems use as additional evolutionary evidence. The result
therefore applies to this particular two-chain, sequence-only ESMFold2
interface rather than every possible way of running a folding model on the
complex.

For a concrete example, consider peptide code `MW` and Affibody code `NNYYF`.
`MW` is not the complete peptide: it states the amino acids at peptide
positions 4 and 5. `NNYYF` states the amino acids at Affibody positions 6, 10,
13, 14, and 17. The remaining residues come from the fixed LibB templates.
Thus, ESMFold2 receives the complete reconstructed 270- and 58-residue
sequences, while the existing designed-position model receives only these
seven letters:

`peptide [position 4 = M, position 5 = W]; Affibody [positions 6, 10, 13, 14, 17 = N, N, Y, Y, F]`

ESMFold2 processes all 328 residues jointly with distinct chain identifiers.
After processing, we retain only the complete 9-by-58 peptide--Affibody block:
522 possible pairs between one peptide residue and one Affibody residue. Pair
features between the other 261 pMHC-side residues and the Affibody are
discarded. The LibB crystal coordinates are not supplied to ESMFold2, and
retention values do not enter feature extraction. The resulting comparison
therefore tests this peptide--Affibody readout, not every structural feature
ESMFold2 calculates for the full pMHC--Affibody complex.

## ESMFold2 as a frozen feature extractor

We used the public `biohub/ESMFold2-hf` checkpoint, which contains approximately
6.58 billion parameters. "Frozen" means that none of those pretrained
parameters changes during this experiment: ESMFold2 performs the same
calculation for every sequence pair, and only a much smaller classifier fitted
afterward learns from the LibB labels. The model and its exposed outputs are
documented in the
[official ESMFold2 documentation](https://huggingface.co/docs/transformers/main/model_doc/esmfold2)
and [Biohub implementation](https://github.com/Biohub/esm).

The relevant calculation is a branch rather than one straight line:

```text
complete sequences → pre-folding features → folding-derived residue-pair features
                                               ├→ predicted distance categories
                                               └→ three-dimensional coordinates
```

The distance and coordinate stages both read folding-model intermediates; the
distance categories are not themselves fed into coordinate generation. We
saved the pre-folding features, folding-derived residue-pair features, and
distance categories, but did not run the coordinate branch. A coordinate would
give an explicit x, y, and z location for each predicted atom.

Three types of frozen information were saved. A tensor below simply means a
multidimensional array of numbers.

- **Predicted distance categories** (`distogram_probabilities` in the saved
  files) give 64 probabilities for each of the 522 residue pairs. The public
  checkpoint does not provide calibrated physical edges for those categories,
  so this study does not convert them into angstroms, apply an 8-angstrom
  contact threshold, or average them and call the result a binding score.
- **Folding-derived residue-pair features** (`pair_states_symmetric`) give 256
  internal values for each residue pair after the repeated folding
  calculations. They are richer than the distance output and can contain
  sequence, partner-context, and structure-related information. The two
  directional peptide-to-Affibody and Affibody-to-peptide arrays are aligned
  and averaged.
- **Pre-folding residue features** (`single_inputs`) give 451 values for each
  individual peptide or Affibody residue before the two partners are refined by
  the folding calculations. They combine amino-acid identity with an atom-level
  input encoder. The project wrapper places the two chains next to each other in
  one combined atom stream without a chain-break mask for that short-range
  encoder. It can therefore mix some information across this artificial array
  boundary; these are not purely partner-independent one-letter features, but
  that mixing should not be interpreted as a learned physical contact. They
  nevertheless contain no predicted coordinates or folding-refined interface.
  We therefore call them pre-folding atom/residue features rather than
  structural evidence.

A six-pair pilot checked reproducibility and mutation sensitivity before
production scaling. ESMFold2 initializes part of this calculation randomly, so
the same random seed was reset immediately before every pair. Under that fixed
seed, repeating `MW + NNYYF` produced identical saved features, while changing
one designed peptide or Affibody residue changed both folding-derived outputs.
This establishes that this fixed extraction notices the sequence changes; it
does not establish that a feature change predicts retention.

## The supervised classifiers

Every ESMFold2 feature set is followed by a newly trained classifier. We do not
use an existing ESMFold2 output layer because ESMFold2 was not originally
trained to output this project's positive/negative selection label.

For the distance-category and folding-derived residue-pair models, each of the
522 interface cells is first converted to 32 learned values. The classifier
learns which peptide and Affibody positions deserve more weight, then combines
a learned weighted average, the ordinary average, and the maximum. A small
two-layer classifier converts that summary into one score. For the pre-folding
residue model, peptide and Affibody features are first converted separately and
multiplied pairwise to form the same 522-cell grid. The output is a relative
binder score for ranking current pairs. It is not a retention percentage, a
contact probability, an affinity, physical binding free energy (ΔG), or a
mutation-induced change in binding free energy (ΔΔG).

| Classifier | Information given to it | Saved feature shape per pair | Newly trained parameters | Training passes selected using weak-label validation |
|---|---|---:|---:|---:|
| Predicted distance categories only | 64 distance-category probabilities for every peptide--Affibody residue pair | 9 × 58 × 64 | 7,713 | 18 |
| Folding-derived residue-pair features only | 256 values after ESMFold2's folding calculations for every residue pair | 9 × 58 × 256 | 14,241 | 9 |
| Distance categories + folding-derived residue-pair features | Both preceding arrays | both | 16,417 | 11 |
| Pre-folding residue features only--not structure | 451 pre-folding values for every residue, converted into peptide--Affibody combinations | 9 × 451 and 58 × 451 | 35,553 | 10 |
| All ESMFold2 features | Pre-folding residue features, folding-derived residue-pair features, and distance categories | all preceding arrays | 46,433 | 7 |

The five rows are feature-source comparisons: they ask where any useful signal
enters the ESMFold2 calculation. They are not perfectly capacity-matched
models. In particular, the pre-folding residue version needs two additional
projections, so parameter counts range from 7,713 to 46,433.

### Sequence controls

The main control is the designed-position additive model from the current LibB
analysis. For `MW + NNYYF`, it learns one number for M at peptide position 4,
one for W at peptide position 5, and one for each of N, N, Y, Y, and F at the
five varied Affibody positions. It adds those seven numbers and a constant to
score the pair. It has 140 possible amino-acid/position coefficient slots plus
one constant. Because it adds independent position effects, it cannot directly
learn that a specific peptide mutation works only with a specific Affibody
mutation.

We also retain the previous frozen MINT and one-epoch LoRA MINT controls. Frozen
MINT averages MINT's per-residue representation separately over the two
complete input chains and fits a small classifier. The LoRA experiment updates
a restricted set of MINT attention parameters and its classifier for one
prespecified epoch. All three controls use the same 30,648 weak training pairs,
strict partner exclusion, class weighting, and 119-row evaluation panel as the
ESMFold2 comparison. Each is represented by one archived fitted model rather
than five new seeds.

## Choosing training settings without retention

LibB has substantially more positive than negative weak labels. All classifiers
therefore use class-weighted binary cross-entropy, a standard training penalty
for wrong yes/no predictions. Each example still appears once, but the loss
gives the positive and negative classes equal total influence. This prevents
the 23,725 positives from dominating simply because they are more numerous.

The number of training passes was selected with three fixed validation splits
made only from the weak training data. In each split, validation pairs contain
peptide and Affibody identities that are both absent from that split's training
rows. A row sharing only one of those held identities is set aside rather than
allowed into either side. This makes the validation problem resemble the final
strict test, in which both complete partners are unseen. We select the number
of passes with weak-label validation loss and take the median choice across the
three splits.

After these settings were fixed, every ESMFold2 classifier was trained five
times on all 30,648 weak-label pairs. ESMFold2 remained frozen, so these runs
vary only the small downstream classifier. No architecture or training setting
was chosen from retention results.

The reported `±` values are the sample standard deviation across those five
classifier-training runs. They measure sensitivity to classifier initialization
and optimization, not to repeated ESMFold2 extraction. They are also not
uncertainty estimates across peptides, experiments, or future libraries. The
three older sequence controls have one saved fit each, so no seed standard
deviation is available for them.

## Evaluation on peptide and Affibody sequences absent from training

The LibB evaluation matrix contains 12 peptide designs and 10 Affibody designs.
One cell was not measured and was not imputed, leaving 119 direct retention
measurements. Retention is the percentage of assay signal remaining after the
construct is cleaved. It is the project's laboratory readout, not a direct
measurement of physical binding affinity.

Before training, every pair containing any of the 12 complete peptide-side
sequences or any of the 10 complete Affibody sequences was removed. Therefore,
neither complete partner in an evaluation pair appeared in training. The model
can still learn how individual amino acids behave from other variants in LibB.
All examples retain the same SMART--HLA and Affibody scaffolds and differ only
at the two designed peptide positions and five designed Affibody positions.
This is therefore not de novo protein-family generalization or a test on
another library, HLA, assay, or experimental batch.

We use the experimental definition that retention of at least 75% is a binder.
We evaluate the model scores in three complementary ways:

- **Across all possible score cutoffs:** AUROC measures how often a binder is
  scored above a nonbinder. Average precision summarizes the precision--recall
  curve and gives more weight to placing binders near the high-scoring end.
  Neither metric requires choosing one cutoff.
- **At one retrospective cutoff:** a pair is classified as a potential binder
  when its score is at least the cutoff. Precision is the fraction of selected
  pairs that are measured binders; recall is the fraction of all measured
  binders that are selected. For a concrete description of the current data,
  we report the cutoff that maximizes F1, the harmonic mean of precision and
  recall.
- **Average within-peptide Spearman:** rank the tested Affibodies separately for
  each peptide by model score and measured retention, then average the
  correlation over the 12 peptides. A value of 1 means the orders agree
  perfectly, 0 means no consistent ordering, and a negative value means the
  order tends to run backward.

The F1-maximizing cutoff is selected using these same retention measurements.
It is therefore a retrospective description of the most favorable operating
point on this panel, not a validated cutoff for new data. It was not used to
train a classifier, choose its architecture or inputs, select its training
settings, or decide when to stop training. The scores are not calibrated
binding probabilities or predicted retention percentages, and numerical
cutoffs cannot be compared across models or training seeds.

## Results when neither complete partner appeared in training

ESMFold2 rows show the mean and sample standard deviation across five seeds;
control rows show one archived fit. All rows were recalculated on the same 119
measured pairs. The sequence controls are one archived fit each, were not
refitted against retention, and reproduce the corresponding sequence-report
values.

| Model | AUROC | Average precision | Average within-peptide Spearman |
|---|---:|---:|---:|
| Designed-position additive control | 0.900 | 0.906 | 0.578 |
| Frozen MINT control | 0.890 | 0.890 | 0.525 |
| One-epoch LoRA MINT control | 0.893 | 0.896 | 0.535 |
| ESMFold2 pre-folding residue features--not structure | 0.887 ± 0.011 | 0.894 ± 0.014 | 0.642 ± 0.006 |
| Folding-derived residue-pair features | 0.871 ± 0.014 | 0.853 ± 0.031 | 0.482 ± 0.043 |
| Folding-derived predicted distance categories | 0.801 ± 0.012 | 0.813 ± 0.014 | 0.416 ± 0.063 |
| Both folding-derived feature types | 0.845 ± 0.015 | 0.808 ± 0.026 | 0.420 ± 0.046 |
| All ESMFold2 features | 0.871 ± 0.019 | 0.847 ± 0.032 | 0.473 ± 0.072 |

The following table applies the score directly as a recommendation rule. Each
cutoff was chosen to maximize F1 on the same retention panel. For an ESMFold2
row, each of its five seeds has a different optimized cutoff; the table reports
the mean and sample standard deviation. The mean cutoff is not a proposed rule
for future data.

| Model | Best retrospective score cutoff | Pairs scoring above cutoff | Binders among those pairs | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Designed-position additive control | 0.795 | 64/119 | 51/60 | 0.797 | 0.850 | 0.823 |
| Frozen MINT control | 0.575 | 80/119 | 58/60 | 0.725 | 0.967 | 0.829 |
| One-epoch LoRA MINT control | 0.572 | 81/119 | 58/60 | 0.716 | 0.967 | 0.823 |
| ESMFold2 pre-folding residue features--not structure | 0.597 ± 0.333 | 69.4 ± 9.5 / 119 | 52.6 ± 3.5 / 60 | 0.764 ± 0.053 | 0.877 ± 0.058 | 0.813 ± 0.012 |
| Folding-derived residue-pair features | 0.677 ± 0.177 | 76.4 ± 4.2 / 119 | 56.6 ± 1.1 / 60 | 0.742 ± 0.030 | 0.943 ± 0.019 | 0.830 ± 0.016 |
| Folding-derived predicted distance categories | 0.253 ± 0.178 | 83.0 ± 16.9 / 119 | 54.2 ± 6.6 / 60 | 0.665 ± 0.075 | 0.903 ± 0.110 | 0.758 ± 0.014 |
| Both folding-derived feature types | 0.498 ± 0.216 | 79.8 ± 4.3 / 119 | 57.8 ± 1.3 / 60 | 0.725 ± 0.024 | 0.963 ± 0.022 | 0.827 ± 0.011 |
| All ESMFold2 features | 0.629 ± 0.152 | 74.6 ± 3.6 / 119 | 56.0 ± 2.3 / 60 | 0.751 ± 0.022 | 0.933 ± 0.039 | 0.832 ± 0.022 |

`FALTA` is important for interpreting these values. It had at least 75%
retention for 11 of the 12 peptides and was tied for or achieved the highest
measured retention for 10 of them. The three sequence controls ranked it first
for all 12 peptides, and the pre-folding model ranked it first in 56 of the 60
seed-by-peptide cases. Recognizing this broadly effective Affibody is useful,
but it is easier than learning peptide-specific compatibility and can make
AUROC and average precision look strong.

The pre-folding residue model is the only candidate that exceeds the additive
control on average within-peptide Spearman: 0.642 instead of 0.578. However,
its AUROC and average precision are both lower. At each model's
own retrospective cutoff, the pre-folding model trades lower precision for
higher recall than the additive control and has slightly lower mean F1. The one
apparent ranking gain therefore does not establish a better candidate-selection
rule, and it comes from a representation created before folding refinement.

The genuinely folding-derived features did not beat the additive control on
AUROC, average precision, or within-peptide Spearman. Their
retrospectively optimized F1 values sometimes look similar, but every cutoff was
chosen on these same retention labels and is not an independent validation.
Predicted distance categories were the weakest input. Combining distance
categories with residue-pair features made the mean result worse rather than
better, consistent with the distance output being a compressed readout of
information already present in the richer residue-pair array. Adding all
available features did not rescue performance.

## What the feature comparisons establish

The feature comparisons were run to separate two questions that would otherwise
be conflated:

- **Does a predicted distance summary help?** No. The distance-category model
  is worse than both the sequence controls and the richer folding-derived
  residue-pair model. This supports the concern that compressing folding output
  to distance or contact values discards relevant information and that
  proximity alone is not a binding score.
- **Do the richer folding-derived residue-pair features help?** They contain
  predictive signal, but they do not establish an improvement over the current
  sequence baseline on peptide-conditioned ranking or wet-lab selection. The
  small differences at retrospectively optimized cutoffs are not independent
  validation.
- **Does combining everything help?** No. Neither the two folding-derived
  feature types together nor all available features improve the result. More
  features do not automatically provide more useful supervision.
- **Where does the one apparent ranking gain come from?** From a residue
  representation created before folding refinement. That result is compatible
  with a richer local amino-acid or chemistry representation and a nonlinear
  peptide--Affibody interaction classifier. It is not evidence that predicted
  geometry caused the gain.

The last point needs a matched follow-up before it can be attributed even to the
ESMFold2 input encoding. The designed-position control is additive and has only
141 coefficient slots, whereas the pre-folding residue model has 35,553
trainable parameters and multiplicatively combines peptide and Affibody
residues. A capacity-matched nonlinear model using only amino-acid one-hot
features, or a
frozen per-residue protein-language-model representation, would test whether
the gain comes from the ESMFold2 inputs or simply from the more expressive
downstream classifier.

## What the results do and do not show

This experiment shows that frozen ESMFold2 outputs change when the designed
LibB residues change and that several of those outputs can support retrospective
binder ranking. It also shows that a richer pre-folding residue representation
was numerically higher on the average order of Affibodies within a peptide than
the current additive model on this reused panel.

It does not show that ESMFold2's predicted structure improves the current LibB
predictor. Under the tested two-chain, sequence-only interface, the extracted
peptide--Affibody distance and residue-pair features do not beat the strongest
sequence-only control. Because HLA--Affibody pair features were discarded, this
is not a test of every structural signal in the full complex. The classifier
scores are not predicted retention percentages, physical affinities, ΔG
values, or mutation ΔΔG values. This analysis also does not test whether the
distance categories match experimentally measured contacts.

The evaluation is limited to 119 retrospective measurements across 12 LibB
peptides, with one missing matrix cell. The five-seed variation describes
optimization variability, not biological uncertainty, and the older controls
have only one fit. No peptide-level bootstrap comparison was used. The weak
training labels can reflect processes other than binding, and R009/R010 was
selected with knowledge of earlier retention analysis. Keeping retention out
of the current fitting procedure prevents direct label leakage, but it does not
turn this reused panel into a prospective experimental validation.

Finally, the strict split tests unseen complete partners *within LibB*. It does
not establish transfer to LibA, another HLA, another protein design, another
assay, or a new experimental batch. Because no folding-derived model improves
the strongest sequence baseline consistently on AUROC, average precision, or
within-peptide Spearman, the prespecified decision gate is not met:
variant-specific coordinate prediction should not be launched on the basis of
these results.
