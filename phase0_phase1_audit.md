# MINT → Affibody–pMHC: Phase 0/1 audit

Audit date: 2026-08-05 UTC

Upstream repository commit audited: `06694b7606e2d00b76ec58daf5c7aecdaf7cd283`

Compatibility commit exercised by the pinned smoke: `460d6911a3752253fa72f068e5ddcd7dd5c87b84`

Scope: static repository audit, private-data inventory, public model-landscape triage, and public Phase 0 runtime reproduction.

Runtime status: **public smoke passed on the assigned GPU node under both PyTorch 2.9.1 and the repository's pinned PyTorch 1.12.1 environment after one isolated compatibility commit.**

## Executive summary

- The local repository is the public MINT implementation plus an untracked `preview.md`; no model adapter has been added.
- The MINT encoder itself can model more than two interacting physical chains. The GeneralPPI datasets, collators, pooling, and caches impose most of the two-chain assumptions.
- The configured `MutationalPPI` task does not use its mutant sequence columns: it is dispatched through `CSVDataset`, not `MutationalCSVDataset`. This is a static code-path finding and must be reproduced at runtime before any patch.
- The existing GeneralPPI training code is not a valid evaluation harness for the private assay without changes to splitting, preprocessing, evaluation mode, target inversion, artifact provenance, and cache keys.
- No private LibA/LibB raw assay table, full WT sequences, sequence mappings, or cleavage-capture table is present. Phase 1 therefore reaches the explicit stop condition in `preview.md`: implementation must not guess the missing biology or provenance.
- The official public checkpoint, all-five-pair embedding smoke, and Bernett binary head run successfully on an A100 under both a modern PyTorch environment and the exact pinned environment after commit `460d691`. Before the fix, the pinned environment reproduced the post-1.12 `weights_only` incompatibility exactly.
- MINT remains the primary model. TUnA-R and Topsy-Turvy/D-SCRIPT are the lowest-effort public secondary diagnostics. PLM-interact mutation/Gold and RaftPPI are later comparisons. The other candidates are reference-only or deferred for the first milestone.

## 1. Repository architecture map

```text
mint/
├── mint/
│   ├── model/esm2.py             # ESM-2-derived MINT encoder and chain mask
│   ├── modules.py               # intra-chain and multimer attention
│   ├── helpers/extract.py        # checkpoint-based embedding extraction
│   └── utils/wrapper.py          # utility wrapper
├── downstream/
│   ├── GeneralPPI/
│   │   ├── tasks.py             # task metadata and CSV datasets
│   │   ├── embeddings_mint.py   # collators, MINT wrappers, cache generation
│   │   ├── finetune_general.py # cached-embedding heads and evaluation
│   │   ├── mutational-ppi/      # mutation preprocessing/example material
│   │   └── SKEMPI_v2/           # affinity-change preprocessing/example material
│   ├── Antibody/
│   ├── CovidVariants/
│   ├── TCR-Epitope/
│   └── oncoPPI/
├── data/
│   ├── protein_sequences.csv    # five public example pairs
│   └── esm2_t33_650M_UR50D.json
├── train.py
├── environment.yml
├── pyproject.toml
├── setup.py
└── preview.md                    # private-project instructions; untracked
```

### Encoder semantics

`mint/model/esm2.py` constructs a cross-chain mask from unequal chain IDs. `mint/modules.py` applies rotary, within-chain attention separately from parameterized, non-rotary multimer attention. Therefore chain IDs carry physical meaning and must not be erased by concatenating MHC subunits into a single string.

The current GeneralPPI wrapper pools only chain IDs `0` and `1` when separate-chain output is requested. That wrapper restriction should not be mistaken for an encoder restriction.

## 2. Static trace of the GeneralPPI paths

The observable flow is:

```text
hard-coded task CSV
  → dataset selected in tasks.py
  → two-chain or mutation collator in embeddings_mint.py
  → frozen MINT extraction under no_grad
  → unversioned .pt cache
  → sklearn or PyTorch head in finetune_general.py
```

This is a frozen-representation benchmark harness; it does not fine-tune the MINT encoder.

### `downstream/GeneralPPI/tasks.py`

- `clean_seq` removes `*` and every lowercase `f` character. This is surprising preprocessing and is not recorded in cache provenance.
- `CSVDataset` reads two sequence columns plus a target. Its `test_run` subset is an unseeded sample of 20 rows.
- `MutationalCSVDataset` reads four sequences (WT pair and mutant pair) plus the target, but does not call the same sequence cleaning routine.
- `MultiCSVDataset` parses chain identifiers character by character. It is therefore safe only for single-digit chain IDs as written, and it does not assert identifier/sequence cardinality.
- Task paths and column metadata are CWD-relative and hard-coded.
- `SKEMPI` is configured as a four-sequence regression task.
- **Major path issue:** `MutationalPPI` is configured with WT and mutant columns, but dispatches through ordinary `CSVDataset`. The mutant columns are consequently not returned to the mutation collator on that configured path.
- `MutationalPPI_cs` uses four sequences but is marked with method `cv`; its explicit validation/test tables are not used by the ordinary-CV branch.
- There is no schema validation for stable IDs, chain names, mutation consistency, grouping variables, target units, or split provenance.

### `downstream/GeneralPPI/embeddings_mint.py`

- `PPICollateFn` tokenizes exactly two sequences, adds special tokens, maps `J` to `L`, pads each chain, assigns chain IDs `0`/`1`, and does not return the target to embedding extraction.
- `MutationalPPICollateFn` constructs independent WT and mutant two-chain tensors and likewise discards targets during extraction.
- Cropping is random. The WT and mutant groups are cropped independently, so a nominal mutation difference may also encode different crop windows. The crop bounds can remove special tokens or return an unexpectedly short slice.
- Separate-chain pooling only covers IDs `0` and `1`, producing a 2×1280 representation. Joint pooling produces 1280 dimensions. Empty masks are not guarded before division.
- The mutation wrapper returns `WT - mutant` by default. With `--cat`, it returns `[WT, mutant]`. This sign/order must be explicit in any assay manifest.
- Extraction is frozen and wrapped in `no_grad`.
- Cache names omit the checkpoint identity/hash, source-data hash, row order and IDs, split, preprocessing, crop configuration, pooling mode, mutation aggregation, and code commit. `--cat` and default difference can address the same filename; `--test_run` can also populate a production-looking cache. These caches are not safe evidence artifacts.

For absolute retention, the direct mutant/variant complex embedding is the information-preserving primary representation. An aligned `mutant - WT` feature can be an augmentation once authoritative WT mapping and deterministic aligned tokenization are available. A delta alone discards absolute complex context.

### `downstream/GeneralPPI/finetune_general.py`

- Ordinary CV is row-level random K-fold, not grouped or stratified. Applied to an Affibody×peptide matrix, it leaks Affibody and peptide identities across folds.
- The code does not consistently call `model.eval()` during evaluation, leaving dropout active.
- Target transformations are fit and used without converting reported predictions back to original assay units.
- Transformation can occur before inner model selection, leaking fold information.
- The ordinary-CV path consumes the training cache and does not provide the required peptide-row, Affibody-column, or double-cold tests.
- A binary predefined-CV path references `X_test` where it is not defined on that branch.
- Missing split fallbacks can reuse test or training data rather than fail closed.
- `squeeze()` can remove the sample dimension for a one-item batch; `drop_last=True` can silently omit examples.
- The MLP expands input to an input-sized hidden layer, which is unnecessarily large for concatenated embeddings and a small assay.
- The test set is evaluated during every epoch.
- Per-example predictions, IDs, fold assignments, preprocessing manifests, and complete random-state provenance are not emitted.
- External logging is not appropriate for private sequences/labels unless explicitly configured for a private destination; the first adapter should default to local artifacts only.

### Preprocessing notebooks/examples

- The mutational-PPI preprocessing recognizes a single substitution with an unanchored pattern, validates a one-based WT residue, mutates only partner 1, and omits stable mutation/source/split metadata from its final table.
- The SKEMPI preprocessing derives sequences from cleaned PDB coordinates and joins all physical chains belonging to one partner into one sequence. That destroys physical-chain boundaries and is unsuitable as-is for pMHC.
- These observations are static; no notebook has yet been used to transform private project data.

## 3. Environment and runtime report

### Local pinned stack

`environment.yml` requests approximately:

| Component | Pinned version |
|---|---:|
| Python | 3.7.12 |
| PyTorch | 1.12.1 |
| CUDA toolkit | 11.3.1 |
| PyTorch Lightning | 1.9.5 |
| Hugging Face Hub | 0.16.4 |
| pandas | 1.3.5 |
| NumPy | 1.21.2 |
| scikit-learn | 1.0.2 |

This should be isolated from modern shared research environments.

### Official smoke-test target

The repository README's intended sequence is: create the pinned environment, install editable package, run `python -c "import mint; print('Success')"`, download the official `mint.ckpt`, and extract embeddings from `data/protein_sequences.csv`. Documented output sizes are 1280 for joint pooling and 2560 for two separate chain pools.

### Status table

| Check | Status | Evidence/result |
|---|---|---|
| Exact local commit | complete | `06694b7606e2d00b76ec58daf5c7aecdaf7cd283` |
| Static repository audit | complete | Findings above |
| Exact pinned environment | complete | `venv/`: Python 3.7.12, PyTorch 1.12.1, CUDA toolkit 11.3, plus the declared conda/pip dependencies; repository installed editable |
| `import mint` smoke | complete | Printed `Success` under both the pinned environment and the read-only modern inference environment |
| Official checkpoint download/hash | complete | `3,253,773,059` bytes; SHA-256 `84a4016365997cd9f0bccb07d746fa8f076ffd8e45aa0cbcf4e50a037161a342` |
| Bernett MLP download/hash | complete | `26,236,447` bytes; SHA-256 `702849af78e245c7596d8c032391da52ff6657e9173af380e29a8ea3858fdab9` |
| Five-pair embedding smoke | complete on modern runtime | All five public rows, not merely `next(iter(loader))`; finite outputs and batch-size parity verified |
| GPU model/VRAM | complete | `gpu-dy-p4d24xlarge-5`; 8× NVIDIA A100-SXM4-40GB, driver 570.86.15, approximately 40,443 MiB free per GPU before use |
| Process CUDA allocation, batch 1 | complete | 3.100 GiB peak allocated including the 3.095 GiB model baseline; 0.005 GiB incremental on the short public pairs under the pinned runtime |
| Process CUDA allocation, batch 2 | complete | 3.106 GiB peak allocated including the 3.095 GiB model baseline; 0.011 GiB incremental on the short public pairs under the pinned runtime |
| Checkpoint-load allocation | complete | 6.189 GiB process peak allocated and 6.385 GiB peak reserved under both tested runtimes; allocator metrics are not total device use or a minimum hardware claim |
| Joint/separate embedding shapes | complete | `(5, 1280)` joint and `(5, 2560)` with `sep_chains=True`, for batch sizes 1 and 2 |
| Official binary PPI checkpoint | complete under pinned runtime | Five finite probabilities produced: `0.39365727`, `0.75133878`, `0.98835725`, `0.73448479`, `0.53898191` |
| Pinned checkpoint-load compatibility | reproduced and fixed | Pre-fix: `TypeError: 'weights_only' is an invalid keyword argument for Unpickler()`; post-fix: full smoke passes under PyTorch 1.12.1 |

The current official Hugging Face artifact is `3,253,773,059` bytes, not the older 9.76 GB figure observed during the initial metadata audit. It was downloaded from the README source, and its local SHA-256 matches current Hugging Face LFS metadata. The binary head was verified the same way. Checkpoints are stored under ignored `checkpoints/`.

Compatibility findings reproduced before patching:

- Current HEAD uses `torch.load(..., weights_only=False)`, while PyTorch 1.12.1 does not declare that parameter. The exact pinned runtime reaches `torch.serialization._load` and raises `TypeError: 'weights_only' is an invalid keyword argument for Unpickler()`.
- The extraction truncation path calls `random.randint` without importing `random`. A synthetic overlength input reaches `extract.py:62` and raises `NameError: name 'random' is not defined`.
- `mint/utils/wrapper.py` imports `wandb`, which is not declared in the inspected environment, and refers to `.utils.logging` although `logging.py` is a sibling module.
- `train.py` hard-codes eight GPU indices and expects model JSON/PT locations that do not match the inspected config placement.

The first two failures were runtime-confirmed and fixed in separate commit `460d691`: `torch_load_compat` only supplies `weights_only=False` when the installed PyTorch exposes that parameter, and the crop path imports `random` and uses a length-preserving upper bound. Three regression tests pass under the pinned environment. The other observations remain outside the public inference smoke path.

## 4. Phase 1 private-data inventory

### Files found

| Path | Size | Rows/format | Classification |
|---|---:|---|---|
| `data/protein_sequences.csv` | 208 bytes | CSV, header + 5 pairs | Public MINT smoke input; columns `Protein_Sequence_1`, `Protein_Sequence_2` |
| `data/esm2_t33_650M_UR50D.json` | 5,446 bytes | JSON | Public model configuration |
| `downstream/Antibody/in_silico/rcsb_pdb_2G75.fasta` | 572 bytes | FASTA | Public downstream example; not LibA/LibB data |
| `preview.md` | project instructions | Markdown | Aggregate project facts only; not row-level assay data |

No local `.ckpt`, `.pt`, `.pth`, `.safetensors`, or `.bin` model artifact existed during the initial static inventory. The two public checkpoints listed in the runtime report were downloaded afterward into ignored `checkpoints/`.

### Private assay inventory result

No file in the working tree contains a row-level LibA/LibB assay table. A second path/filename search across `/fsx/users/dongweij`, pruning environments, caches, and toolchains, also found no plausible private assay file or directory. Searches for LibA/LibB, R000–R014, retention, cleavage/capture, Affibody, peptide, and pMHC found only the project instructions or unrelated package/compiler artifacts. Therefore none of the following can be inventoried from local evidence:

- nucleotide-level or amino-acid-level assay rows;
- R000–R014 raw/normalized/filtered/pooled columns;
- synonymous-codon collapse status;
- replicate fields and missing values;
- cleavage-capture measurements and retention time points;
- a stable pair identifier joining selection and cleavage-capture data;
- mutation-label-to-full-sequence mappings.

The deck-level totals in `preview.md` (108 LibA pairs and 119 LibB pairs with direct measurements) are aggregate statements, not verified row counts in a supplied table.

### Explicit blockers that must not be guessed

1. Full WT Affibody sequence for each library/context.
2. Full peptide sequence(s), exact WT residues, and the mapping from abbreviated matrix labels to full variants.
3. Whether the canonical MINT input is the 2-chain Affibody+peptide approximation or physical multichain Affibody+MHC-heavy+beta-2-microglobulin+peptide.
4. Full MHC heavy/alpha and beta-2-microglobulin sequences, allele/context, and chain conventions if multichain input is approved.
5. Authoritative raw, normalized, filtered, and pooled R000–R014 values and their provenance.
6. Synonymous-codon collapsing policy.
7. Authoritative retention time point (the materials conflict between 15 and 30 minutes for LibA), replicate policy, and missing-value handling.
8. Stable pair IDs that join selection and cleavage-capture records.
9. The exact R009/R010 and top-2% rule: normalization, pooling operation, tie handling, and denominator.

This is the Phase 1 stop condition. No adapter or training task should be created from the aggregate deck alone.

## 5. Public PPI-model landscape

Classification is for this first Affibody–pMHC milestone, not a ranking of general scientific quality. GPU requirements marked "not established" must be measured or obtained from model documentation before scheduling.

| Screenshot name → resolved model | Official source / repository | Backbone and interaction mechanism | Input / supported task | Checkpoint, license, practical fit | Classification |
|---|---|---|---|---|---|
| ProteoMeLM-S → **ProteomeLM-S** | [ProteomeLM](https://github.com/Bitbol-Lab/ProteomeLM), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13214046/) | ESM-C 600M protein embeddings plus a reported 36.9M proteome-context model; proteins are contextualized as whole-protein tokens | Proteome collections; contextual protein representation, not residue-level mutation scoring | Code reports Apache-2.0; checkpoint licensing/VRAM not established here. Loses residue/interface detail needed for the primary task | `reference-only` |
| PLM-Interact STRING v12 | [PLM-interact](https://github.com/liudan111/PLM-interact), [HF checkpoint](https://huggingface.co/danliu1226/PLM-interact-650M-humanV12) | ESM-2 650M with joint full cross-attention for two proteins | Exactly one protein pair; binary PPI | Public checkpoint; code MIT, checkpoint terms and required VRAM need confirmation. Pair-only and not an absolute-retention model | `reference-only` |
| Topsy-Turvy | [paper](https://academic.oup.com/bioinformatics/article/38/Supplement_1/i264/6617505), [D-SCRIPT repo](https://github.com/samsledje/D-SCRIPT), [checkpoint](https://huggingface.co/samsl/topsy_turvy_human_v1) | Frozen Bepler–Berger residue embeddings followed by a learned contact-map interaction head | Protein pair; binary PPI | Public small task head and MIT code; old dependencies but comparatively cheap. No native pMHC multichain or retention output | `run-now` as a diagnostic |
| TUnA | [TUnA-R paper](https://academic.oup.com/bib/article/25/5/bbae359/7720609), [TUnA-R repo](https://github.com/young-su-ko/TUnA-R), [weights](https://huggingface.co/yk0/tuna-r-tuna) | ESM-2 150M with intra/inter-protein transformers and an uncertainty-aware SNGP head | Protein pair; PPI probability and uncertainty | Public refactor and weights; pair-only and no native mutation/retention task. Practical runtime must be measured | `run-now` as a secondary diagnostic |
| PPI-RIS → **ppIRIS** | [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13159128/), [repo](https://github.com/lupiochi/ppiris) | ESM-C 300M and ProstT5 global representations with Siamese/cross-attention fusion | Protein pair; binary pathogen–host/bacterial PPI | Public code (Apache-2.0 indicated); two large backbones and domain mismatch raise first-milestone cost | `defer` |
| MINT-PPI → **MINT** | [paper](https://www.nature.com/articles/s41467-025-67971-3), [repo](https://github.com/VarunUllanat/mint), [checkpoint](https://huggingface.co/varunullanat2012/mint) | ESM-2 650M with native chain-aware intra-chain and cross-chain attention | Multiple interacting chains; frozen embeddings, PPI tasks, and mutation/affinity examples | MIT repository; verified public checkpoint is 3.253 GB. Public A100 inference and the binary head run successfully under the pinned runtime after the isolated compatibility fix | `run-now`, primary |
| FlashPPI | [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13291599/), [repo](https://github.com/TattaBio/FlashPPI), [weights](https://huggingface.co/tattabio/flashppi) | gLM2 650M dual retrieval plus a contact reranker | Protein pair; large-scale microbial PPI retrieval/reranking | Public artifacts, but repository/model-card licensing needs reconciliation; domain and engineering mismatch | `defer` |
| RaftPPI | [paper](https://openreview.net/pdf/ac61178a0bca360ce214f120e626fc847b044710.pdf), [repo](https://github.com/AndyJZhao/RaftPPI) | Small ESM-2 representation plus Fourier-factorized pair retrieval | Protein pair; binary/retrieval-style PPI | MIT repository and public checkpoints; reported work used A100-class training. Pair-only but a plausible later common-split baseline | `run-now` after primary diagnostics |
| PLM-Interact Gold | [PLM-interact repo](https://github.com/liudan111/PLM-interact), [Gold checkpoint](https://huggingface.co/danliu1226/PLM-interact-650M-Leakage-Free-Dataset), [mutation checkpoint](https://huggingface.co/danliu1226/PLM-interact-650M-Mutation) | Same ESM-2 650M two-protein joint architecture, trained on leakage-controlled or mutation-specific data | Binary PPI; mutation checkpoint predicts binary increase/decrease rather than absolute retention | Public checkpoints; not a distinct architecture. Useful reference only after canonical split/label semantics are aligned | `reference-only` |
| C3PI | [paper](https://academic.oup.com/bib/article/26/6/bbaf685/8383622), [repo](https://github.com/lucian-ilie/C3PI) | ProtT5 residue branches, block permutations, and multiscale convolutions | Protein pair; binary PPI | Public code; license, reproducible checkpoint path, dependency state, and practical VRAM not established in this audit | `defer` |
| ESM2-AMPS → **ESM2_AMPS** | [paper](https://academic.oup.com/bib/article/26/4/bbaf434/8242608), [repo](https://github.com/ywwy-qn/ESM2_AMP) | ESM-2 650M with segmented pooling and a fusion transformer | Protein pair; PPI classification | Public repository with very large reported artifacts; license and portable inference path require confirmation. Pair-only/no native mutation regression | `defer` |

### Staged shortlist

1. **MINT:** reproduce official import and five-pair embedding extraction, then freeze embeddings on the canonical private splits once data blockers are resolved.
2. **TUnA-R:** public weights and a smaller ESM-2 backbone make it a useful probability/uncertainty diagnostic after input-length compatibility is checked.
3. **Topsy-Turvy/D-SCRIPT:** inexpensive binary-PPI sanity baseline; do not interpret it as an affinity model.
4. **PLM-interact mutation/Gold:** compare only after label semantics and two-chain approximation are justified; the mutation model's increase/decrease target is not retention.
5. **RaftPPI:** later retrieval-style comparison on exactly the same canonical examples and group splits.

No model should be allowed to choose its own preprocessing or random split. The shared artifact must determine row IDs, chains, labels, folds, and metrics.

## 6. Minimal future modification map

This is a map, not an implementation authorization. Create it only after the missing private data and biological choices are supplied.

```text
downstream/AffibodyMHC/
├── schema.py / dataset.py
├── collate.py
├── embeddings.py
├── splits.py
├── heads.py
├── evaluate.py
└── README.md
```

### Project-local responsibilities

- **Schema/dataset:** require stable pair ID, library, full WT/mutant sequences, explicit physical-chain names, target provenance, retention time point, raw units, and persisted fold IDs. Validate mutation positions/residues, duplicate IDs, missing targets, sequence alphabet, and maximum length.
- **Collator:** preserve each physical chain. Support the approved two-chain approximation and, separately, explicit multichain pMHC. Do not concatenate MHC physical chains. Reject malformed examples rather than inferring biology.
- **Embeddings:** use direct variant-complex embedding as primary; optionally add deterministic aligned `mutant - WT` and `[WT, mutant]`. Support joint, per-physical-chain, and explicitly documented partner-group pooling.
- **Cache/artifacts:** key by repository commit, checkpoint hash, input-table hash/order, row IDs, chain configuration, preprocessing, max length/crop, pooling, aggregation, and split. Store a manifest beside tensors.
- **Splits:** keep LibA and LibB separate initially; materialize random-pair, peptide-row-held-out, Affibody-column-held-out, and double-cold folds. Group common WT/batch/replicates to prevent leakage.
- **Heads/evaluation:** start with fold-local standardized ridge/logistic regression, then a small MLP. Regression: Pearson, Spearman, MAE, RMSE, and R² in original assay units. Classification: MCC, AUPRC, AUROC, precision, recall, balanced accuracy, threshold, and class counts. Emit local JSON/CSV metrics and per-example predictions; disable network logging by default.

### Shared MINT changes

Prefer no shared change. If runtime reproduction proves a shared compatibility failure, patch only that failure in a separate commit with the failing command, traceback, package/GPU inventory, and a regression smoke test. Do not generalize the GeneralPPI collator when a project-local collator is sufficient.

## 7. Dynamic execution result and next boundary

Steps 1–8 were completed on `gpu-dy-p4d24xlarge-5`. `phase0_smoke_results.json` preserves the initial Python 3.10.19 / PyTorch 2.9.1+cu128 run. The primary final artifact, `phase0_smoke_results_pinned.json`, was generated under Python 3.7.12 / PyTorch 1.12.1+cu113 at compatibility commit `460d691`. Both used physical GPU 7. The artifacts record command arguments, node/runtime versions, Git commit, file hashes, three warmed timing repetitions, allocator measurements, output hashes, and batch-size parity. Timing includes collation and device transfers and is only a smoke measurement on 11–20-residue public inputs, not a pMHC throughput benchmark.

The full declared environment was created unchanged and passed `import mint`. Its first checkpoint load failed exactly at the known `weights_only` incompatibility, and the separate overlength-collator probe reproduced the missing-`random` failure. After the separate compatibility commit, three regression tests and the full pinned GPU smoke pass. The modern and pinned probabilities differ only at low floating-point precision, as expected across PyTorch/CUDA generations.

No private model training or `AffibodyMHC` adapter should start until the Phase 1 blockers are resolved.
