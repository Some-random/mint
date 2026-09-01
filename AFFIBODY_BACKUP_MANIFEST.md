# Affibody project recovery manifest

Backup date: 2026-09-01 UTC

This repository contains the complete reusable Affibody modeling source,
configuration, tests, and the original MINT history. The branch
`backup/affibody-work-2026-09-01` is the recovery branch.

Private experimental material is stored as release assets rather than normal
Git objects:

| Asset | Size | SHA-256 |
|---|---:|---|
| `affibody_private_materials_2026-09-01.tar.zst` | 398,416,113 bytes | `49b8a02515ae5b365ee17634f383dd55ce511461cba0784a041f1b10c9107f22` |
| `Affibody coevolution dataset.zip` | 1,026,267,412 bytes | `d91914a20c02aef78970af833f6443f47355dea8a21972e0c5609f2f2f3a4e3a` |

The private-materials archive contains 1,597 files (1,240,104,960 bytes before
compression): reports, plans, transcripts, presentations, the updated sequence
archive, the reference PDB, all experiment directories, compact derived data,
and structure-analysis manifests/results.

The following bulk objects are deliberately omitted because they are
downloadable or reproducible from the retained code, inputs, manifests, and
configs:

- ESMFold2 checkpoint and root model checkpoints;
- the full ESMFold2, MINT weak-cache, and MINT late-round feature tensors;
- bulk OpenFold3 model outputs;
- virtual environments, package caches, temporary runs, and pytest debris.

To verify downloaded assets:

```bash
sha256sum affibody_private_materials_2026-09-01.tar.zst
sha256sum 'Affibody coevolution dataset.zip'
```

To inspect or restore the compressed private archive:

```bash
tar --zstd -tf affibody_private_materials_2026-09-01.tar.zst
tar --zstd -xf affibody_private_materials_2026-09-01.tar.zst
```

The repository and both release assets contain unpublished experimental
material and must remain private.
