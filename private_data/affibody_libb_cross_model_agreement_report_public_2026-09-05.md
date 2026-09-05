# Do cross-model agreements make LibB predictions more reliable?

This analysis asks whether a peptide--Affibody pair is more likely to work when
several models independently give it a positive score. It uses the corrected
LibB retention matrix: 12 peptides, 10 Affibodies per peptide, and 120 measured
pairs. A pair is counted as an experimental binder when its retention is at
least 75%; 61 of the 120 pairs meet that definition.

The three model families are frozen MINT, a StaB-ddG-derived representation,
and an RDE-PPI-derived representation. StaB-ddG and RDE-PPI are used here as
feature extractors followed by project-specific classifiers; their original
physical-energy outputs are not being treated as binder predictions. The
three classifiers do not use one another's scores, but they were trained from
the same selection-derived labels and therefore are not independent
experimental replications.

## What “success rate of the overlap” means

Each model first makes its own binder or nonbinder call. For example, the
MINT--StaB overlap contains only pairs called binders by both models. Its
success rate is then:

\[
\frac{\text{pairs in the overlap with measured retention at least 75%}}
{\text{all pairs in the overlap}}.
\]

This is precision at a fixed decision rule. It is different from average
precision, which summarizes a model's ranking across every possible score
cutoff.

The primary cutoffs were fixed from 10,181 out-of-fold weak-label examples,
without using the retention measurements:

| Model | Weak-label cutoff |
|---|---:|
| MINT | 0.1452 |
| StaB-derived model | 0.3009 |
| RDE-derived model | 0.4271 |

The scores are selection-label scores, not predicted retention percentages or
calibrated binding probabilities.

## Results using the retention-blind cutoffs

| Pairs required to be positive | Pairs selected | Measured binders | Success rate | Fraction of all 61 binders recovered |
|---|---:|---:|---:|---:|
| MINT | 90 | 60 | 66.7% | 98.4% |
| StaB | 90 | 60 | 66.7% | 98.4% |
| RDE | 79 | 59 | 74.7% | 96.7% |
| MINT and StaB | 84 | 59 | 70.2% | 96.7% |
| MINT and RDE | 77 | 58 | 75.3% | 95.1% |
| StaB and RDE | 79 | 59 | 74.7% | 96.7% |
| MINT, StaB, and RDE | 77 | 58 | 75.3% | 95.1% |

For the example raised in the discussion, **MINT and StaB agree on 84 positive
pairs, of which 59 are measured binders: a 70.2% success rate**. Each model
alone has a 66.7% success rate at its weak-label cutoff, so their agreement
raises the observed rate by 3.6 percentage points while losing one binder.

The largest three-model number is 75.3%, but RDE alone is already 74.7%.
Moreover, every RDE-positive pair is also StaB-positive at these cutoffs.
Requiring StaB therefore adds no filtering to RDE, and requiring all three
improves success over RDE by only 0.6 percentage points while losing one more
binder.

The MINT--StaB call pattern makes the behavior particularly clear:

| MINT call | StaB call | Number of pairs | Measured binders | Success rate |
|---|---|---:|---:|---:|
| Positive | Positive | 84 | 59 | 70.2% |
| Positive | Negative | 6 | 1 | 16.7% |
| Negative | Positive | 6 | 1 | 16.7% |
| Negative | Negative | 24 | 0 | 0.0% |

Disagreement is therefore a useful warning sign on this panel. It does not by
itself prove that agreement is better than applying a stricter rule to one
model.

## Does agreement help beyond selecting fewer pairs?

An intersection normally selects fewer pairs, and selecting fewer high-scoring
pairs can improve success even if the second model contributes no useful
information. We therefore matched the number selected separately for every
peptide. If MINT--StaB agreement selected *n* Affibodies for one peptide, the
control selected MINT's top *n* Affibodies for that peptide.

| Comparison at the same per-peptide assay counts | Binders from agreement | Binders from one model |
|---|---:|---:|
| MINT and StaB versus MINT | 59/84 | 59/84 |
| All three versus MINT | 58/77 | 58/77 |
| All three versus RDE | 58/77 | 58/77 |

The selected pair identities are not necessarily identical, but the binder
counts are. The current panel therefore does not show that model agreement
beats simply using one model to choose the same number of high-scoring pairs.
These per-peptide budgets are diagnostic and were themselves obtained from
the consensus rule; they are not a new standalone deployment rule.

All three models also agree that 24 pairs are negative, and none is a measured
binder. However, taking each individual model's bottom-scoring pairs at the
same per-peptide rejection counts also excludes 24 nonbinders and no binders.
The current data therefore do not establish a unique benefit for negative
agreement either.

## Is the result caused only by FALTA?

FALTA binds 11 of the 12 measured peptides. It is experimentally useful, but
including it can make any model that selects it look stronger. After removing
all 12 FALTA pairs:

| Rule | Pairs selected | Measured binders | Success rate |
|---|---:|---:|---:|
| RDE | 68 | 48 | 70.6% |
| MINT and StaB | 72 | 48 | 66.7% |
| MINT, StaB, and RDE | 66 | 47 | 71.2% |

The three-model result falls from 75.3% to 71.2%. It remains only 0.6
percentage points above RDE alone, so FALTA is not the sole explanation, but
its broad activity contributes materially to the apparent success.

## Why the votes are less independent than they appear

Although the representations come from different architectures, their scores
are strongly related. Pairwise Spearman correlations are 0.86--0.90 across the
whole matrix and 0.74--0.82 when calculated separately within each peptide and
then averaged. All models also learned from the same weak labels and split.

It is therefore better to call this **cross-model agreement** or **model
consensus**, rather than cross-validation. Conventional cross-validation means
training and evaluating across different data splits.

## Why a 92.7% number also appears

If each model's cutoff is optimized on these same 120 retention outcomes,
MINT--StaB agreement contains 51 binders among 55 selected pairs, or 92.7%.
After removing FALTA it is 40/44, or 90.9%.

Those numbers are useful for defining a stricter hypothesis to test next, but
they are optimistic because the same measurements chose and evaluated the
cutoffs. They are not an estimate of future wet-lab success. The next batch
would be the first prospective test after those rules are frozen.

## Recommendation for the next wet-lab batch

Keep the existing ensemble score for ranking candidates, and add model
agreement as a secondary confidence label rather than replacing the ranking
with a hard consensus filter. The batch should contain both consensus-positive
candidates and a smaller set of deliberate model-disagreement controls. New
retention measurements can then answer whether agreement truly improves yield;
testing only consensus candidates would not provide that comparison.

The present results support testing the consensus hypothesis. They do not yet
support claiming that model consensus is more robust than a single-model rule
at the same experimental budget.
