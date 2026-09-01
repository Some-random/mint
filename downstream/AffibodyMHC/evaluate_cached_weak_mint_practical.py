#!/usr/bin/env python
"""Run cached weak-label MINT evaluation at sklearn's standard tolerance.

The audited base evaluator deliberately used ``tol=1e-6``.  For nearly
separable 2,560-dimensional MINT features, weakly regularized liblinear fits at
C=10--100 can spend tens of minutes per fold without changing the scientific
decision.  This wrapper changes only the solver stopping tolerance to the
scikit-learn default, ``1e-4``.  It delegates all data construction, cold
splits, cleaning, balancing, metrics, and output creation to the base evaluator
and then records both executable sources and the tolerance in the manifest.
"""

from __future__ import print_function

import json
import os
from pathlib import Path

from downstream.AffibodyMHC import evaluate_cached_weak_mint as base
from downstream.AffibodyMHC.code_only_baseline import sha256_file


SOLVER_TOLERANCE = 1e-4
WRAPPER_PATH = Path(__file__).resolve()
BASE_PATH = Path(base.__file__).resolve()
ORIGINAL_MAKE_LOGISTIC = base._make_logistic


def make_logistic(c_value, balance):
    """Return the base estimator with only its convergence tolerance changed."""
    model = ORIGINAL_MAKE_LOGISTIC(c_value, balance)
    model.set_params(tol=SOLVER_TOLERANCE)
    return model


def rewrite_manifest(output_dir):
    """Bind the completed artifact to this wrapper and its base dependency."""
    manifest_path = Path(output_dir) / "manifest.json"
    with open(str(manifest_path), "r") as handle:
        manifest = json.load(handle)
    manifest["configuration"]["solver_tolerance"] = SOLVER_TOLERANCE
    manifest["configuration"]["solver_tolerance_reason"] = (
        "sklearn default; avoids pathological high-C runtime while retaining "
        "the same liblinear objective and convergence checks"
    )
    manifest["code"] = {
        "path": str(WRAPPER_PATH),
        "sha256": sha256_file(WRAPPER_PATH),
    }
    manifest["dependencies"] = {
        "base_evaluator": {
            "path": str(BASE_PATH),
            "sha256": sha256_file(BASE_PATH),
        }
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    with open(str(temporary), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(temporary), 0o600)
    os.replace(str(temporary), str(manifest_path))
    os.chmod(str(manifest_path), 0o600)
    return manifest


def run(args):
    original = base._make_logistic
    base._make_logistic = make_logistic
    try:
        base.run(args)
    finally:
        base._make_logistic = original
    return rewrite_manifest(args.output_dir)


def main(argv=None):
    run(base.parse_args(argv))


if __name__ == "__main__":
    main()
