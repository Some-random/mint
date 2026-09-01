#!/usr/bin/env python3
"""Launch OpenFold3 after seeding the in-process feature RNGs.

OpenFold3 featurizes reference conformers before Lightning applies the model
seed. Its default CLI therefore leaves ``ref_pos`` dependent on OS-seeded
PyTorch RNG state. This entrypoint supplies an explicit feature seed while
keeping the model seed in the normal query/runner configuration.
"""

from __future__ import annotations

import argparse
import os
import random
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--global-feature-seed", type=int, required=True)
    parser.add_argument("--ref-conformer-seed", type=int, required=True)
    args, openfold_args = parser.parse_known_args()
    if not openfold_args:
        raise ValueError("OpenFold3 CLI arguments are required")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["OPENFOLD_REF_CONFORMER_SEED"] = str(args.ref_conformer_seed)
    random.seed(args.global_feature_seed)

    import numpy as np

    np.random.seed(args.global_feature_seed % (2**32))

    import torch

    torch.manual_seed(args.global_feature_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.global_feature_seed)

    # Import only after all feature RNGs have been initialized.
    from openfold3.run_openfold import cli

    sys.argv = ["openfold3.run_openfold"] + openfold_args
    cli()


if __name__ == "__main__":
    main()
