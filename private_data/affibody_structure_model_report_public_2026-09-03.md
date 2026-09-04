# Testing protein-structure representations for LibB Affibody ranking

Affibodies are small engineered proteins designed here to bind peptide--HLA
targets. This study asks whether information about three-dimensional protein
structure can improve the choice of an Affibody for a specified LibB peptide.
Every project-specific classifier is trained on the same 30,648
selection-derived positive and negative pairs and is evaluated on the corrected
12-peptide by 10-Affibody matrix of 120 direct retention measurements.
Retention is the percentage of assay signal remaining after the construct is
cleaved; it is the project's laboratory readout, not a direct measurement of
binding affinity.

The folding and fixed-structure encoders tested in the new comparison are kept
frozen. Only a smaller classifier is trained to score the current
peptide--Affibody pair. The score is not a predicted retention percentage,
contact probability, binding free energy (`ΔG`), or mutation-induced change in
binding free energy (`ΔΔG`). Retention values were not used to fit the
classifiers, choose the number of training passes, or choose between input
representations. Those choices used only selection-derived training and
validation labels. However, the direct-projection rerun reported here was
designed after earlier versions had already been examined on the retention
panel, so this remains a retrospective analysis rather than an untouched test.
The illustrative score cutoffs reported later were fitted to those retention
outcomes and also require prospective testing.

The structural comparison uses one experimentally determined LibB crystal
structure, for peptide code `MW` and Affibody code `NNYYF`. RDE-PPI-derived
and StaB-ddG-derived models reuse that fixed geometry for every sequence
variant; they do not predict a new structure for each pair. The earlier
ESMFold2 comparison instead used internal features calculated from each
sequence pair. Its best result came from features produced before folding, so
that result is a strong sequence-like control rather than evidence that
predicted structure helped.

Both fixed-crystal representations outperform the nonlinear seven-position
sequence control. Average within-peptide Spearman is 0.6540 ± 0.0289 for RDE
and 0.6126 ± 0.0576 for StaB, compared with 0.5638 ± 0.0291 for the control.
RDE also gives the highest within-peptide AUROC in the comparison, while StaB
gives the highest within-peptide average precision. Each representation is
mapped directly into a small learned classifier.

RDE's complete retention ordering is essentially tied with the ESMFold2
pre-folding result of 0.6470 ± 0.0060. StaB's complete ordering is somewhat
lower, although its binder-focused average precision is higher. The experiment
therefore shows that RDE and StaB representations are useful for screening,
but it does not establish that three-dimensional geometry itself adds value
beyond richer pretrained representations. It does not yet justify folding
every sequence variant separately.

At the retrospectively calibrated cutoffs, StaB recommends 62 of the 120 pairs:
55 are measured binders, giving precision 0.887 and recall 0.902. RDE recommends
77 pairs and recovers 59 of the 61 binders, giving precision 0.766 and recall
0.967. These numbers describe a retrospective precision--workload tradeoff on
outcomes already known. They are not the weak-label cutoffs used to select new
wet-lab candidates.

## Why protein structure might help

A protein chain is a connected sequence of amino acids. An amino acid after it
has been incorporated into a chain is also called a residue. The written
sequence gives their order, but the chain folds into a three-dimensional
shape. Residues that are far apart in the written sequence can therefore become
neighbors in space.

When two proteins bind, part of each surface meets the other. This meeting
surface is called the interface. LibB varies two residues in the nine-amino-acid
peptide and five residues in the Affibody. A sequence-only model can learn that
a particular amino acid is generally helpful at one of those seven positions.
A structure-aware representation could additionally help it learn that a
particular peptide residue and Affibody residue are positioned to interact.

This is only a hypothesis. Close residues do not necessarily make a pair bind:
their charge, chemistry, orientation, flexibility, and the rest of the
interface also matter. A structure representation is useful only if a model
trained without retention labels ranks the measured Affibodies better than the
corresponding sequence-only control.

## The experimental structure available for LibB

The file `nyeso_xx133_complex.pdb` is an X-ray crystal structure. X-ray
diffraction data were used to estimate the position of atoms in one physical
complex, and the deposited model reports a resolution of 1.71 angstroms. The
symbol Å means **angstrom**, pronounced approximately “ANG-strum”; one
angstrom is 0.1 nanometres, a convenient unit for distances between atoms.

The file contains two related crystallographic copies. Chains were assigned by
matching their amino-acid sequences and geometry rather than trusting a
filename. The more completely resolved copy contains HLA chain `A`,
beta-2-microglobulin chain `B`, the complete peptide `SLLMWITQV` in chain `P`,
and the Affibody in chain `H`. `MW` identifies peptide positions 4 and 5;
`NNYYF` identifies the five designed Affibody positions. It does not mean that
either complete protein is only two or five residues long.

Three supplied distance checks were reproduced. Each distance uses the
C-beta atom, the first side-chain carbon attached to a residue's main
backbone; C-alpha would be used for glycine because glycine has no C-beta.

| Peptide residue | Affibody residue using corrected numbering | PDB atoms compared | Expected distance | Reproduced distance |
|---|---|---|---:|---:|
| Peptide position 4 | Affibody position 16 | `P:4 CB` to `H:16 CB` | 6.934 Å | 6.934 Å |
| Peptide position 4 | Affibody position 19 | `P:4 CB` to `H:19 CB` | 7.824 Å | 7.823 Å |
| Peptide position 5 | Affibody position 12 | `P:5 CB` to `H:12 CB` | 5.967 Å | 5.967 Å |

These checks establish that the peptide, Affibody, and designed residues were
mapped to the intended coordinates. They do not show that a model predicts
retention, and the distances themselves are not used as the final binding
score.

The provider corrected the numerical labels of the five Affibody mutation
positions. An audit confirmed that every model used the intended physical
residues, so this naming correction does not require retraining. The full table
linking library-code characters, displayed-sequence positions, Python indices,
PDB residue identifiers, and model-token indices remains in the
[data-revision audit](../data_revision_audit.md).

This crystal is structural evidence for one sequence pair, `MW + NNYYF`. It is
not structural ground truth for the remaining measured pairs. Reusing the
crystal for every variant is a deliberate fixed-backbone approximation: the
current amino-acid identities change, while the measured backbone geometry
does not. The experiment therefore asks whether pretrained representations can
combine current sequence chemistry with a known LibB binding geometry. It does
not test whether each variant would actually keep that geometry.

## Training data from the selection rounds

The LibB R000--R014 sequencing files contain 15,085,560 rows. Each row reports
how many reads were observed for one peptide--Affibody pair in one experimental
round. If the same pair appears in five rounds, it contributes five rows.
These are repeated observations over selection, not 15.1 million independently
labeled training examples. The 27,821,822-row total in the earlier sequence
report includes LibA and LibB; the present study is LibB only.

## How positive and negative training examples were defined

The structural models reuse the established LibB label rules without changing
the selected rounds, cutoff, or negative definition:

- **Positive training example:** first make one list of every distinct pair
  appearing in R009, R010, or both. Add each pair's two counts, treating absence
  from either round as zero. Label the highest 2% of the pooled distribution as
  positive, including every pair tied at the count boundary. The LibB boundary
  is a pooled count of 12.
- **Negative training example:** require an appearance in R001, no appearance
  of that pair in any round from R002 through R014, and at least three R001
  reads.
- R000 does not define either class. A pair satisfying neither rule is not used
  for training.

The three-read requirement was introduced by this analysis rather than
experimentally calibrated by the data provider. It is kept fixed here so that
the structure comparison changes the representation, not the labels.

The current implementation was independently checked against the full raw
R009 and R010 files. It pools the two counts before finding the top 2%, rather
than taking a separate top 2% in each round and merging those lists. The
independent reconstruction produced exactly the same LibB positive IDs. One
remaining source question is whether the unavailable provider-generated pooled
file also includes all ties at the 2% boundary.

| Data at each stage | Number of LibB observations or pairs |
|---|---:|
| Sequencing observations across R000--R014 | 15,085,560 |
| Distinct pairs meeting the positive or negative rule after evaluation designs are excluded | 747,618 |
| Pairs left after the amino-acid-code check and three-read rule for negatives | 38,710 |
| Pairs used for strict training after all evaluation peptides and Affibodies are removed | 30,648 |
| Direct-retention evaluation pairs | 120 |

The strict training set contains 23,725 positives and 6,923 negatives. The
complete feature-extraction roster contains 30,768 rows: the 30,648 training
pairs plus the 120 evaluation pairs. Structure extraction reads sequences and
row identifiers for all of them but no selection labels or retention values.
Only the training rows are joined to the selection-derived labels during
classifier fitting.

These are weak training labels. Selection and sequencing are related to
binding, but they can also reflect expression, amplification, experimental
sampling, and other processes. The 75% retention cutoff is not used to create
the training labels. It is used only to describe success on the direct
retention panel after all model scores have been fixed.

## The models

Consider the reference design `MW + NNYYF`. Its seven-letter shorthand means:

`peptide [positions 4, 5 = M, W]; Affibody [corrected positions 8, 12, 15, 16, 19 = N, N, Y, Y, F]`

All other peptide and Affibody residues come from the fixed LibB templates.
Every model below receives information derived from this current sequence pair
and returns one relative selection score.

### Earlier additive sequence control

For continuity with the sequence-only study, the results also include its
simple designed-position model. It learns one number for each possible amino
acid at each of the seven varied positions and adds the seven numbers plus a
constant. For example, its score for `MW + NNYYF` is the sum of the learned
weights for M and W at the two peptide positions and N, N, Y, Y, and F at the
five Affibody positions. It has 140 amino-acid/position weights plus one
constant and cannot learn that a peptide residue works specifically with an
Affibody residue. It is a useful historical control, but the nonlinear model
below is the closer comparison for the new structural representations.

### Nonlinear seven-position sequence control

This control receives only one-hot identities for the seven designed amino
acids. For `MW + NNYYF`, it knows which of the 20 standard amino acids occupies
each of the two peptide and five Affibody positions. Unlike the earlier
additive baseline, the two-hidden-layer classifier can learn combinations such
as “M at peptide position 4 is helpful specifically with Y at Affibody position
16.” It sees no coordinates, distances, unchanged residues, or pretrained
protein representation.

This is the closest sequence-only neural-network control for the RDE-PPI-derived
and StaB-ddG-derived features. It uses the same two-hidden-layer design,
training labels, validation folds, class weighting, optimizer, and five random
seeds. Its input is much shorter, so its first learned layer contains fewer
weights. The later results state separately when several RDE heads are
combined.

### ESMFold2 intermediate representations

ESMFold2 receives the complete reconstructed 270-residue
SMART--HLA--linker--peptide sequence and the complete 58-residue Affibody
sequence as two chains. It is frozen, and a smaller classifier is trained on
one of several saved intermediates:

- **Pre-folding residue features** describe each residue before ESMFold2's
  repeated folding calculations. They are rich sequence and atom-input
  encodings, but they are not predicted structure.
- **Folding-derived pair states** give 256 internal values for each of the
  9 × 58 possible peptide--Affibody residue pairs after folding calculations.
- **Predicted distance categories** give 64 values for each residue pair. The
  public checkpoint does not expose calibrated physical distance edges, so
  these categories were not converted to angstroms or thresholded into a
  contact score.

The reference crystal is not supplied to ESMFold2. The previous extraction did
not run the final coordinate branch. Its pair states and distance categories
are folding-derived intermediates, while its strongest pre-folding readout is
a sequence-like control.

### RDE-PPI-derived fixed-crystal representations

RDE-PPI was developed for structure-based mutation-effect modeling. Here, its
frozen encoders are used only to describe the **current** peptide--Affibody
pair. The original wild-type-minus-mutant subtraction and physical `ΔΔG`
output are not called.

For each pair, the model receives the same 128-residue region surrounding the
seven designed sites in the LibB crystal. That region contains all nine peptide
residues, 52 resolved Affibody residues, and 67 nearby HLA residues. The
current seven amino-acid identities are inserted into this fixed region. The
reference side-chain coordinates and side-chain angles at those seven sites
are hidden so the model cannot mistake the crystallized `MW + NNYYF` side
chains for the current variant.

The extracted representation gives 128 internal values per residue. An earlier
feature-view screen compared information kept in the original order at the
seven designed sites, a mean-and-maximum summary of those sites, and a summary
of all 128 resolved residues. It also compared three pretrained mutation-model
encoders, keeping their scores separate before averaging them. Using only
selection-derived validation labels, that screen chose the ordered seven-site
input and three-encoder average. The present direct-projection rerun carries
that input choice forward and reports it as
`rde_network_designed_3fold_ensemble_native_projection`; it did not use
retention to reselect among feature views.

### StaB-ddG-derived fixed-crystal representations

StaB-ddG was developed to predict how a mutation changes binding free energy.
This study does not use its `ΔΔG` prediction. It uses the frozen ProteinMPNN
component inside the StaB-ddG checkpoint as a feature extractor for the current
pair.

ProteinMPNN is evaluated in two structural contexts: the assembled complex and
the partners separated. This requires three forward passes per batch--one
for the HLA--peptide--Affibody complex, one for the HLA--peptide fragment, and
one for the Affibody alone. The saved values describe how the model's
internal representation and amino-acid compatibility at the seven designed
sites change between those two contexts. A supervised classifier then learns
whether those values help distinguish the project's selection-derived
positives and negatives. The complex-versus-separated difference is a model
representation, not a measured binding energy or a wild-type-to-mutant change.

This input uses the coordinate-resolved assay fragment: HLA chain `A` residues
1--181, the complete peptide chain `P`, and Affibody chain `H` residues 5--59.
The crystal's beta-2-microglobulin and HLA residues 182--276 are omitted to
better match the assayed construct. The SMART domain and assay linker are also
omitted because they have no coordinates in this crystal. The result is
therefore an approximation of the resolved interface, not a complete
three-dimensional assay construct. HLA and peptide also remain separate model
chains because the linker connecting them has no coordinates.

An earlier feature-view screen compared a 138-value whole-pair summary, the
seven ordered designed-residue representations, a mean-and-maximum summary of
those residues, and the combination of whole-pair and ordered local
information. Using only selection-derived validation labels, that screen chose
the ordered seven-site input. The present direct-projection rerun carries that
choice forward and reports it as `stab_designed_ordered`; it did not use
retention to reselect among feature views.

### The downstream classifier

The reported comparison maps each native feature vector directly to 64 hidden
values, then to 32 hidden values, and finally to one score. For RDE, the direct
projection has a learned `896 x 64` weight matrix; for StaB it has a learned
`1,050 x 64` weight matrix. The following `64 x 32` and `32 x 1` weight matrices are learned
as well. Thus, every hidden value can learn its own weighted combination of the
actual encoder features instead of receiving predetermined copies. Nonlinear
activations between layers allow the score to depend on combinations of those
features.

Training compares the score with the positive or negative selection label and
updates the normalization, projection, classifier weights, and their biases. It
does not update RDE, ProteinMPNN, or the crystal coordinates.

Because the original inputs have different lengths, their first learned layers
also have different numbers of weights. The nonlinear sequence control has
11,545 trainable parameters, each RDE head has 61,441, and the StaB model has
71,605. The direct-projection RDE result averages three separately trained
heads; the averaging itself has no trained weight. All pretrained sequence and
structure encoders in the new structural comparison remain frozen. The
historical LoRA MINT row shown later is the explicit exception.

The earlier ESMFold2 models used feature-specific classifiers ranging from
7,713 to 46,433 trainable parameters. These comparisons therefore test the
complete representations and their small readouts; they are not exact
parameter-count comparisons.

An older exploratory implementation first copied each input into an untrained
common-width adapter before classification. Those results are superseded and
are omitted here. Every RDE and StaB result in this report uses the learned
direct projection described above.

| Model family | Information available to the classifier | What is trained | Physical quantity predicted |
|---|---|---|---|
| Nonlinear sequence control | Seven designed amino-acid identities | Direct 11,545-parameter classifier | None; relative selection score |
| ESMFold2 pre-folding control | Rich residue encodings before folding | Feature-specific classifier | None; relative selection score |
| ESMFold2 folding-derived models | Residue-pair states and/or distance categories | Feature-specific classifier | None; relative selection score |
| RDE-PPI-derived models | Current identities plus one fixed crystal neighborhood | One direct 61,441-parameter classifier per encoder; the reported result averages three heads | None; relative selection score |
| StaB-ddG-derived models | Current-sequence compatibility on fixed complex and separated backbones | Direct 71,605-parameter classifier | None; relative selection score |

## Choosing model settings without retention

The 30,648 weak-label pairs contain many more positives than negatives. The
class-weighted binary loss therefore gives the two classes equal total
influence without discarding positive examples.

Training settings are selected with three fixed validation splits made only
from the selection-derived labels. In each split, neither the complete peptide
nor the complete Affibody in a validation pair occurs in that split's training
rows. Rows sharing exactly one held partner are set aside rather than allowed
into either group. This makes validation resemble the final strict evaluation,
where both complete partners are absent from training.

The RDE and StaB input forms were carried forward from an earlier
weak-label-only feature-view screen. The direct-projection rerun did not repeat
every feature-view comparison. Its number
of training passes was selected by weak-label validation loss, and no retention
outcome entered classifier fitting or that duration choice. The rerun itself
was nevertheless performed after the panel had already been examined. Each
final classifier is fitted five times on all 30,648 training pairs, using the
prespecified seeds. Reported `±` values are the sample standard deviation over
those five classifier fits. They measure sensitivity to classifier
initialization and optimization, not biological uncertainty or uncertainty
across future peptides.

The ESMFold2 pre-folding arm shown later also had the lowest weak-label
validation loss among the tested ESMFold2 feature sets. It was not selected by
looking at retention.

The RDE-PPI and StaB-ddG feature files contain only opaque row identifiers,
split labels, and numerical features. Retention outcomes cannot enter their
feature extraction or classifier fitting. The 120 evaluation scores are fixed
before a separate evaluation step joins them to direct retention.

## Evaluation when neither complete partner appeared in training

The corrected LibB evaluation is a complete matrix of 12 peptides and 10
Affibodies, giving 120 measured pairs. The newly supplied value is
`AH + LIFTK = 87.94`. At the experimental cutoff of retention at least 75%, the
matrix contains 61 binders and 59 non-binders.

Before fitting, every training pair containing any complete evaluation peptide
sequence or any complete evaluation Affibody sequence is removed. The models
can still learn how individual amino acids behave from other variants in LibB,
but neither complete partner in a measured pair appeared in training. This is
strict generalization within one library, not a test on a different scaffold,
HLA, assay, or experimental batch.

The evaluation asks two separate questions.

The first is whether a model orders the ten Affibodies correctly for each
peptide:

- **Average within-peptide Spearman** compares the complete ordering by model
  score with the ordering by measured retention, then gives each peptide equal
  weight. A value of 1 means perfect agreement, 0 means no consistent ordering,
  and a negative value means the order tends to run backward.
- **Average within-peptide AUROC** asks, within one peptide, how often a binder
  receives a higher score than a nonbinder. A value of 0.5 is chance and 1 is
  perfect separation.
- **Average within-peptide precision--recall area** summarizes precision and
  recall across all possible score cutoffs within one peptide. Higher is
  better, but its baseline depends on how many of that peptide's Affibodies are
  binders.

Spearman is calculated for all 12 peptides. The AUROC and precision--recall
summaries use the 11 peptide rows containing both binders and nonbinders; `DP`
has no retention-positive Affibody, so its within-row AUROC is undefined.

The second question is what happens when the model is used as an experimental
screen. Every pair whose model score is at or above a declared cutoff is
recommended. The report then gives:

- **recommended pairs**, the number of experiments that would be sent;
- **precision**, the fraction of recommended pairs whose retention is at least
  75%;
- **recall**, the fraction of all 61 measured binders that were recommended;
- **F1**, the harmonic mean of precision and recall; and
- **peptides with no recommendation**, so empty rows are immediately visible.
  The companion result files retain the exact candidate count for every
  peptide and model.

The 75% retention cutoff defines the observed laboratory outcome. The separate
model-score cutoff decides which pairs would be recommended. A model score is
not a calibrated binding probability, and numerical score cutoffs cannot be
compared between model families.

For the threshold table, one-fit models use their single score. The current
RDE and StaB rows use the deployed score--the sigmoid of the mean logit across
five fits--so they match exhaustive candidate scoring. The other historical
five-fit rows use their mean probability because no corresponding deployed
candidate score was locked. A model-specific cutoff is then chosen to maximize
F1 on the already-known 120-pair panel, breaking an exact tie in favor of
higher precision, fewer experiments, and then the higher threshold. This uses
the retention outcomes and is therefore retrospective calibration, not an
independent test. A cutoff intended for new candidates must instead be frozen
before new wet-lab results are known.

The 120-pair matrix is **retrospective** because its retention results were
already known while parts of the analysis were developed. Retention does not
fit the current models, but the panel has been examined repeatedly. The real
test will be **prospective**: first lock the model and candidate-selection rule,
then score new candidates, and finally use new wet-lab retention measurements
to determine whether the method works.

## Results on the corrected 120-pair panel

The first table compares how well each model ranks all ten Affibodies for a
given peptide. Existing scores were recomputed on all 120 measurements. The
nonlinear control and the ESMFold2, RDE-PPI-derived, and StaB-ddG-derived rows
report the mean and sample standard deviation across five fits. The additive,
MINT, and LoRA rows have one archived fit each. The structural input forms came
from the earlier weak-label-only screen, and training duration was selected
without using retention. The direct-projection rerun and analysis as a whole
are retrospective for the reasons described above.

| Model | Within-peptide Spearman ↑ | Within-peptide AUROC ↑ | Within-peptide average precision ↑ |
|---|---:|---:|---:|
| Earlier designed-position additive control | 0.5804 | 0.8211 | 0.9105 |
| Nonlinear seven-position sequence control | 0.5638 ± 0.0291 | 0.8233 ± 0.0169 | 0.9136 ± 0.0130 |
| Frozen MINT layer 5 + classifier | 0.5370 | 0.8442 | 0.9111 |
| MINT after one LoRA epoch | 0.5410 | 0.8043 | 0.9041 |
| ESMFold2 pre-folding residue features--not structure | 0.6470 ± 0.0060 | 0.8266 ± 0.0081 | 0.9251 ± 0.0211 |
| ESMFold2 folding-derived pair state | 0.4888 ± 0.0402 | 0.7194 ± 0.0366 | 0.8629 ± 0.0236 |
| RDE-PPI-derived direct-projection ensemble | **0.6540 ± 0.0289** | **0.8662 ± 0.0174** | 0.9287 ± 0.0089 |
| StaB-ddG-derived direct-projection model | 0.6126 ± 0.0576 | 0.8412 ± 0.0161 | **0.9353 ± 0.0151** |

The direct-projection fixed-crystal models improve the complete within-peptide
ordering over the nonlinear one-hot control on average. RDE is higher than the
control in all five paired training seeds; StaB is higher in three of five. The
RDE number averages three independently trained encoder-specific scores, whereas
StaB uses one score.

The three columns answer different questions. RDE is strongest at putting a
measured binder above a nonbinder and at reproducing the complete retention
ordering. StaB is strongest at concentrating binders near the beginning of
each peptide's list. RDE's full-ordering result is essentially tied with the
0.6470 obtained from ESMFold2 features created before folding; StaB is lower
on full ordering but higher on average precision. The evidence therefore
supports these pretrained representations for screening, but it does not
isolate fixed crystal geometry as the cause of the improvement.

The second table treats each model as a screen using the aggregation rules
defined above and a model-specific retrospective F1 cutoff. Threshold values
are displayed rounded. The current deployed RDE and StaB values supersede the
older mean-probability cutoff values. Exact deployed values and every selected
measured pair are in the [current ensemble handoff](prospective/libb_native_projection_candidate_selection_v1/wetlab_handoff/ensemble_candidate_selection_report.md).

| Model | Score cutoff ≥ | Recommended pairs | Measured binders recommended | Precision | Recall | F1 | Peptides with no recommendation |
|---|---:|---:|---:|---:|---:|---:|---|
| Earlier designed-position additive control | 0.4244 | 80/120 | 58/61 | 0.725 | 0.951 | 0.823 | `DP` |
| Nonlinear seven-position sequence control | 0.7393 | 73/120 | 56/61 | 0.767 | 0.918 | 0.836 | None |
| Frozen MINT layer 5 + classifier | 0.7680 | 71/120 | 55/61 | 0.775 | 0.902 | 0.833 | None |
| MINT after one LoRA epoch | 0.5720 | 82/120 | 59/61 | 0.720 | **0.967** | 0.825 | None |
| ESMFold2 pre-folding residue features--not structure | 0.7526 | 66/120 | 52/61 | 0.788 | 0.852 | 0.819 | `DP`, `EA`, `PH` |
| ESMFold2 folding-derived pair state | 0.6371 | 77/120 | 57/61 | 0.740 | 0.934 | 0.826 | `DP`, `PH` |
| RDE-PPI-derived direct-projection ensemble | 0.4939 | 77/120 | 59/61 | 0.766 | **0.967** | 0.855 | `DP` |
| StaB-ddG-derived direct-projection model | 0.9072 | **62/120** | 55/61 | **0.887** | 0.902 | **0.894** | `DP` |

At its retrospective operating point, StaB is the most selective: it recommends
62 experiments and 55 are measured binders, giving the highest precision and
F1. RDE recommends 77 pairs and recovers 59 of the 61 measured binders, giving
higher recall at the cost of 18 false-positive experiments. `DP` has no binder
among its ten measured Affibodies, so recommending none for `DP` is correct.
In contrast, the ESMFold2 pre-folding rule recommends nothing for `EA` and
`PH` even though those rows contain measured binders.

These cutoff results describe the best F1 tradeoff found on outcomes we already
know. They are optimistic diagnostics and must not be presented as independent
generalization or silently reused for the prospective candidate menu. The
separately locked weak-label cutoffs used for new candidates are reported in
the current ensemble handoff. Precision at a fixed rank was removed because it
would force a fixed number of candidates through even when their scores are
poor.

## Ablations and why they were run

These checks were used to understand what information might be responsible for
the result, not to choose the model that looked best on retention.

- **Are predicted distances enough by themselves?** No. The ESMFold2
  distance-category readout reaches within-peptide Spearman 0.4173 ± 0.0675,
  well below the 0.6470 ± 0.0060 pre-folding representation. Knowing only broad
  distance categories loses information relevant to binding.
- **Should local designed residues be kept separate?** The earlier
  weak-label-only feature-view screen favored the ordered seven-site RDE and
  StaB inputs carried into the direct-projection rerun. The rerun did not repeat
  all feature-view comparisons with the new classifier. Keeping the sites
  separate is biologically reasonable because averaging can erase which amino
  acid occurs at which peptide or Affibody position.
- **Are the downstream classifiers identical in size?** No. Each native feature
  vector is normalized and projected directly by a learned layer into 64 values
  before the common 64-to-32 classifier. Longer structural representations
  therefore create larger first layers. The parameter counts are reported
  explicitly, and the result should be read as a comparison of complete
  representation-plus-readout systems rather than an exact parameter-count
  experiment.

These ablations rule out distance categories alone as an explanation. They
still do not isolate three-dimensional geometry from other information learned
by the RDE and StaB encoders.

## The unusually strong FALTA Affibody

`FALTA` has retention of at least 75% for 11 of the 12 measured peptides and is
tied for or achieves the highest retention for 10. Several earlier sequence
models rank it first for every peptide. Across the five direct-projection fits,
the nonlinear control ranks it first for 10--12 peptides, RDE for 9--12, and
StaB for 6--9.
This is genuine and useful experimental performance, so `FALTA` remains in all
120 primary evaluation pairs. Its complete Affibody sequence was excluded from
the 30,648 training pairs, even
though its individual amino-acid choices occur in other training variants.

At the same time, repeatedly recommending one broadly effective Affibody is
easier than learning which Affibody works specifically with each peptide. It
can make pooled classification results look strong even when the rest of the
within-peptide ordering is inaccurate. This is why the main table reports
metrics calculated separately within each peptide.

The result is not explained by `FALTA` alone. After removing `FALTA` only for a
secondary sensitivity calculation, within-peptide Spearman remains 0.536 ±
0.038 for RDE and 0.505 ± 0.048 for StaB, compared with 0.412 ± 0.039 for the
nonlinear seven-position control. ESMFold2 pre-folding remains comparable at
0.523 ± 0.009, so this check does not change the structure-specific conclusion.
The primary analysis still contains all 120 pairs.

`FALTA` is not experimentally optimal for two peptides. For `EL`, its retention
is 94.24 while `EFYSV` reaches 98.45. For `MW`, `FALTA` reaches 98.25 while
`LIFTK` reaches 99.80. RDE ranks `FALTA` first in three of five `EL` fits and all
five `MW` fits. StaB does not rank it first for either peptide in any of the five
fits. Under the retrospective score rules above, `EFYSV` for `EL` and `LIFTK`
for `MW` both pass the RDE, StaB, and ESMFold2 pre-folding cutoffs. This
two-peptide observation is only a secondary, post-hoc explanation.

The crystal reference pair `MW + NNYYF` also remains in the primary panel. It
is not removed simply because its geometry supplied the fixed reference. No
pair is removed from the reported analysis.

## Why whole-panel metrics are not emphasized

Global AUROC and average precision pool all 120 pairs together. They can reward
a model for learning that one Affibody is broadly strong or one peptide is
broadly weak, even if the model does not choose well among Affibodies for a
specified peptide. Those values remain in the machine-readable evaluation for
continuity, but the public comparison uses within-peptide AUROC and average
precision instead. Global Spearman and fixed-rank `Hit` metrics are also
omitted.

## What the comparison does and does not show

Both fixed-crystal representations have higher five-seed averages than the
nonlinear model using only the seven designed amino-acid identities. RDE is
higher in all five paired fits; StaB is more variable and is higher in three of
five. Their within-peptide ranking and retrospective cutoff results show that
pretrained RDE and ProteinMPNN representations contain useful information for
the LibB task. RDE is the stronger of the two for reproducing the complete
retention ordering; StaB is the stronger binder-focused screen.

It does not establish that the useful information comes from structure. Both
fixed-crystal models use encoders that learned from more than the seven current
amino-acid identities, and RDE's full-ordering result essentially matches a
pre-folding ESMFold2 representation that has not yet performed structural
reasoning. The one fixed backbone may also be inaccurate for many variants,
the training labels are noisy proxies for binding, and the measured panel is
small and unusually favorable to `FALTA`. The comparison supports richer
pretrained representations, but it has not narrowed their advantage
specifically to crystal geometry.

No result here establishes transfer to LibA, a new HLA, another protein
scaffold, another assay, or a new experimental batch. The current prospective
plan progresses from the corrected 120-pair panel to in-design candidates
missed by selection, combinations absent from training, and new peptide
targets. The laboratory can initially test 5--10 peptides with no more than ten
Affibody candidates per peptide, with an expected cleavage-assay turnaround of
approximately 2--3 weeks after receiving the candidate list.

The current evidence does not pass the decision gate for launching
variant-specific structure prediction. RDE improves binder/nonbinder ordering
over ESMFold2's pre-folding representation but only ties it on the complete
retention ordering; StaB improves average precision but is lower on the
complete ordering. These mixed retrospective gains do not yet show that
predicting a separate geometry for millions of variants would add useful
information. The actual validation will be the prospective experiment in which
the model and candidate-selection rule are locked before new wet-lab
measurements are known.
