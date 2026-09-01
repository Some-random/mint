#!/usr/bin/env python
"""Build the canonical LibA later-round weak-label ablation.

The four positive-label arms are:

* A: inclusive top 2% of the union of R009/R010, ranked by raw-count sum;
* B: inclusive top 2% of the union of R009--R014, ranked by raw-count sum;
* C: inclusive top 2% of the R009--R014 union, ranked by mean count divided by
  that round's read depth (an absent pair contributes zero in that round);
* D: inclusive top 2% of the R011--R014 union, ranked by raw-count sum.

Every arm uses the same negatives: LibA pairs with R001 count >= 3 that are
absent from every R002--R014 table.  Codes outside the declared LibA design
alphabet and every pair sharing either LibA retention peptide or LibA
retention Affibody identity are excluded.  The measured retention values are
never read when defining labels; only the identities of the LibA panel are
used for the prespecified double-identity-cold transfer.

Natural arm membership and deterministic 11,320-positive size-matched
memberships are emitted separately.  Size matching is an unweighted
pseudorandom subset ordered by SHA256("size-match|<seed>|<pair_uid>").
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_selection_weak_labels import (
    discover_raw_round_files,
    key_series,
    load_count_round,
    load_presence,
    within_declared_alphabet,
)
from downstream.AffibodyMHC.code_only_baseline import (
    opaque_id,
    sha256_file,
    validate_private_output_path,
)


LIBRARY = "LibA"
TOP_FRACTION = 0.02
NEGATIVE_MIN_R001_COUNT = 3
SIZE_MATCH_SEEDS = (20260811, 20260812, 20260813)
ARM_ORDER = ("A", "B", "C", "D")
ARM_SPECIFICATIONS = {
    "A": {
        "name": "r009_r010_raw_count",
        "rounds": (9, 10),
        "score_kind": "raw_count_sum",
        "role": "baseline",
    },
    "B": {
        "name": "r009_r014_raw_count",
        "rounds": tuple(range(9, 15)),
        "score_kind": "raw_count_sum",
        "role": "primary_later_round_test",
    },
    "C": {
        "name": "r009_r014_equal_round_frequency",
        "rounds": tuple(range(9, 15)),
        "score_kind": "equal_round_mean_count_over_round_depth",
        "role": "depth_normalized_sensitivity",
    },
    "D": {
        "name": "r011_r014_raw_count",
        "rounds": tuple(range(11, 15)),
        "score_kind": "raw_count_sum",
        "role": "late_only_diagnostic",
    },
}

# These contracts bind this ablation to the already audited provider files and
# the library-local LibA retention exclusion used in the matched-LoRA study.
EXPECTED_BASELINE_UNION_ROWS = 881464
EXPECTED_BASELINE_CUTOFF = 13
EXPECTED_ELIGIBLE_POSITIVES = {"A": 11320, "B": 24925, "C": 23189, "D": 19596}
EXPECTED_ELIGIBLE_NEGATIVES = 11222
EXPECTED_NEGATIVE_MEMBERSHIP_SHA256 = (
    "245a2021a83a08cf3b75b0b72fc183c4bf573041915c7cd01b3a1face113f53a"
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def membership_sha256(frame_or_values):
    """Return an order-independent SHA256 over newline-delimited pair UIDs."""
    if isinstance(frame_or_values, pd.DataFrame):
        values = frame_or_values["pair_uid"].astype(str).tolist()
    else:
        values = [str(value) for value in frame_or_values]
    return hashlib.sha256("\n".join(sorted(values)).encode("ascii")).hexdigest()


def size_match_rank(pair_uid, seed):
    """Canonical deterministic uniform-subsampling order requested by audit."""
    payload = "size-match|{}|{}".format(int(seed), str(pair_uid)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _score_series(frame, value_column):
    _require(not bool(frame[["pep", "aff"]].duplicated().any()), "duplicate round key")
    index = pd.MultiIndex.from_frame(frame[["pep", "aff"]], names=["pep", "aff"])
    values = pd.to_numeric(frame[value_column], errors="raise").to_numpy()
    _require(bool(np.isfinite(values).all()), "non-finite round score")
    _require(bool((values > 0).all()), "round scores must be positive where observed")
    return pd.Series(values, index=index, name=value_column).sort_index()


def sum_round_scores(round_frames, rounds, value_column, divide_by_rounds=False):
    """Outer-union rounds and zero-fill missing pairs before summing.

    For the equal-round-frequency arm, division by the fixed number of rounds
    produces exactly (1/6) sum_i c_ir / D_r because the provider frequency
    column is c_ir / D_r and absent rows contribute zero.
    """
    rounds = tuple(int(value) for value in rounds)
    _require(rounds, "at least one round is required")
    _require(set(rounds).issubset(round_frames), "missing requested round frame")
    score = None
    for round_index in rounds:
        current = _score_series(round_frames[round_index], value_column)
        score = current if score is None else score.add(current, fill_value=0.0)
    score = score.sort_index()
    if divide_by_rounds:
        score = score.astype(np.float64) / float(len(rounds))
    elif value_column == "count":
        numeric = score.to_numpy(dtype=float)
        _require(bool(np.equal(numeric, np.floor(numeric)).all()), "raw count sum is noninteger")
        score = score.astype(np.int64)
    _require(bool((score.to_numpy(dtype=float) > 0).all()), "union score must be positive")
    return score


def build_arm_score_series(round_frames):
    """Return canonical A/B/C/D score series indexed by (pep, aff)."""
    required = set(range(9, 15))
    _require(required.issubset(round_frames), "R009--R014 frames are required")
    normalized_frames = {}
    for round_index in range(9, 15):
        frame = round_frames[round_index]
        depth = int(pd.to_numeric(frame["count"], errors="raise").sum())
        _require(depth > 0, "round depth must be positive")
        normalized = frame[["pep", "aff", "count"]].copy()
        normalized["count_over_depth"] = (
            normalized["count"].to_numpy(dtype=np.float64) / float(depth)
        )
        # The source frequency is useful as an audit field, but derive C from
        # integer count/depth so rounded TSV decimal text cannot affect ties.
        reported = pd.to_numeric(frame["frequency"], errors="raise").to_numpy(dtype=float)
        _require(
            bool(
                np.allclose(
                    normalized["count_over_depth"].to_numpy(dtype=float),
                    reported,
                    rtol=1e-9,
                    atol=1e-15,
                )
            ),
            "provider frequency disagrees with count/round depth in R{:03d}".format(
                round_index
            ),
        )
        normalized_frames[round_index] = normalized
    return {
        "A": sum_round_scores(round_frames, (9, 10), "count"),
        "B": sum_round_scores(round_frames, range(9, 15), "count"),
        "C": sum_round_scores(
            normalized_frames,
            range(9, 15),
            "count_over_depth",
            divide_by_rounds=True,
        ),
        "D": sum_round_scores(round_frames, range(11, 15), "count"),
    }


def inclusive_top_fraction(score, top_fraction=TOP_FRACTION):
    """Select an inclusive top fraction; all pairs tied at the cutoff remain."""
    _require(isinstance(score.index, pd.MultiIndex), "score index must be (pep, aff)")
    _require(0.0 < float(top_fraction) < 1.0, "top fraction must lie in (0,1)")
    values = score.to_numpy(dtype=float)
    _require(len(values) > 0 and bool(np.isfinite(values).all()), "invalid score vector")
    requested_rank = max(1, int(math.ceil(float(top_fraction) * len(values))))
    cutoff = float(np.partition(values, len(values) - requested_rank)[len(values) - requested_rank])
    selected = score.loc[score.ge(cutoff)].rename("label_score").reset_index()
    selected = selected.sort_values(
        ["label_score", "pep", "aff"], ascending=[False, True, True]
    ).reset_index(drop=True)
    _require(len(selected) >= requested_rank, "inclusive selection lost cutoff rows")
    return selected, cutoff, requested_rank, int(len(score))


def fixed_negative_candidates(reference, later_frames, min_r001_count=3):
    """Return R001>=threshold pairs absent from every supplied later round."""
    _require(int(min_r001_count) >= 1, "R001 threshold must be positive")
    base = reference.loc[reference["count"].ge(int(min_r001_count))].copy()
    candidates = set(key_series(base).tolist())
    for later in later_frames:
        candidates.difference_update(key_series(later).tolist())
    result = base.loc[key_series(base).isin(candidates), ["pep", "aff", "count"]].copy()
    return result.sort_values(["count", "pep", "aff"], ascending=[False, True, True]).reset_index(drop=True)


def local_identity_cold_filter(frame, retention):
    """Keep valid LibA codes sharing neither LibA retention partner identity."""
    held = retention.loc[retention["library"].eq(LIBRARY)]
    _require(len(held) == 108, "LibA retention grid must contain 108 pairs")
    held_peptides = set(held["peptide_design_code"].astype(str))
    held_affibodies = set(held["affibody_design_code"].astype(str))
    on_design = [
        within_declared_alphabet(LIBRARY, peptide, affibody)
        for peptide, affibody in zip(frame["pep"].astype(str), frame["aff"].astype(str))
    ]
    keep = (
        np.asarray(on_design, dtype=bool)
        & ~frame["pep"].astype(str).isin(held_peptides).to_numpy()
        & ~frame["aff"].astype(str).isin(held_affibodies).to_numpy()
    )
    return frame.loc[keep].copy().reset_index(drop=True)


def load_liba_retention_identities(path):
    """Read only partner identities; label/retention-value columns stay unread."""
    identities = pd.read_csv(
        path,
        usecols=["library", "peptide_design_code", "affibody_design_code"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    _require(len(identities) == 228, "retention identity table must have 228 rows")
    _require(set(identities["library"]) == {"LibA", "LibB"}, "unexpected library values")
    keys = ["library", "peptide_design_code", "affibody_design_code"]
    _require(not bool(identities.duplicated(keys).any()), "duplicate retention identity")
    _require(int(identities["library"].eq(LIBRARY).sum()) == 108, "LibA identity count changed")
    return identities


def add_pair_identifiers(frame):
    output = frame.copy()
    output["pair_uid"] = [
        opaque_id(LIBRARY, peptide, affibody)
        for peptide, affibody in zip(output["pep"], output["aff"])
    ]
    output["peptide_uid"] = [opaque_id(LIBRARY, "pep", value) for value in output["pep"]]
    output["affibody_uid"] = [opaque_id(LIBRARY, "aff", value) for value in output["aff"]]
    _require(not bool(output["pair_uid"].duplicated().any()), "opaque pair UID collision")
    return output


def deterministic_size_match(positives, target, seed):
    """Uniformly subset positives by the exact prespecified SHA256 ordering."""
    _require(int(target) > 0, "size-match target must be positive")
    _require(len(positives) >= int(target), "positive arm is smaller than target")
    ranked = positives[["pair_uid"]].copy()
    ranked["rank_sha256"] = [size_match_rank(uid, seed) for uid in ranked["pair_uid"]]
    ranked = ranked.sort_values(["rank_sha256", "pair_uid"]).head(int(target)).copy()
    ranked["size_match_rank"] = np.arange(1, len(ranked) + 1, dtype=np.int64)
    _require(len(ranked) == int(target), "size matching produced the wrong row count")
    return ranked.reset_index(drop=True)


def _write_csv(frame, path):
    frame.to_csv(path, index=False)
    os.chmod(str(path), 0o600)


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--retention-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(args.raw_root.is_dir(), "raw root does not exist")
    _require(args.retention_csv.is_file(), "retention CSV does not exist")

    raw_files = discover_raw_round_files(args.raw_root)
    files = raw_files[LIBRARY]
    retention = load_liba_retention_identities(args.retention_csv)

    round_frames = {
        round_index: load_count_round(files[round_index], LIBRARY)
        for round_index in range(9, 15)
    }
    round_depths = {
        "R{:03d}".format(round_index): int(round_frames[round_index]["count"].sum())
        for round_index in range(9, 15)
    }
    scores = build_arm_score_series(round_frames)

    reference = load_count_round(files[1], LIBRARY)
    later_presence = []
    for round_index in range(2, 15):
        if round_index in round_frames:
            later_presence.append(round_frames[round_index][["pep", "aff"]])
        else:
            later_presence.append(load_presence(files[round_index], LIBRARY))
    negatives = fixed_negative_candidates(
        reference, later_presence, min_r001_count=NEGATIVE_MIN_R001_COUNT
    )
    del later_presence
    negatives = local_identity_cold_filter(negatives, retention)
    negatives = negatives.rename(columns={"count": "r001_count"})
    negatives = add_pair_identifiers(negatives)
    negative_hash = membership_sha256(negatives)
    _require(len(negatives) == EXPECTED_ELIGIBLE_NEGATIVES, "eligible negative count changed")
    _require(
        negative_hash == EXPECTED_NEGATIVE_MEMBERSHIP_SHA256,
        "eligible negative membership changed",
    )

    reference_counts = reference[["pep", "aff", "count"]].rename(
        columns={"count": "r001_count"}
    )
    label_blocks = []
    summary_rows = []
    positive_by_arm = {}
    for arm in ARM_ORDER:
        selected, cutoff, requested_rank, union_rows = inclusive_top_fraction(scores[arm])
        selected_before_filter = len(selected)
        selected = local_identity_cold_filter(selected, retention)
        selected = selected.merge(
            reference_counts, on=["pep", "aff"], how="left", validate="one_to_one"
        )
        selected["r001_count"] = selected["r001_count"].fillna(0).astype(np.int64)
        selected = add_pair_identifiers(selected)
        _require(
            len(selected) == EXPECTED_ELIGIBLE_POSITIVES[arm],
            "eligible {} positive count changed".format(arm),
        )
        _require(
            set(selected["pair_uid"]).isdisjoint(set(negatives["pair_uid"])),
            "{} positives overlap fixed negatives".format(arm),
        )
        positive_by_arm[arm] = selected.copy()

        positive = selected.copy()
        positive["weak_label"] = 1
        positive["weak_label_source"] = "{}_inclusive_top_2pct".format(
            ARM_SPECIFICATIONS[arm]["name"]
        )
        negative = negatives.copy()
        negative["label_score"] = 0.0
        negative["weak_label"] = 0
        negative["weak_label_source"] = "r001_count_ge3_absent_r002_r014"
        labels = pd.concat([positive, negative], ignore_index=True, sort=False)
        labels.insert(0, "arm", arm)
        labels.insert(1, "library", LIBRARY)
        labels["within_declared_library_alphabet"] = 1
        labels["strict_liba_retention_identity_cold_eligible"] = 1
        labels = labels[
            [
                "arm",
                "library",
                "pep",
                "aff",
                "r001_count",
                "weak_label",
                "label_score",
                "weak_label_source",
                "within_declared_library_alphabet",
                "strict_liba_retention_identity_cold_eligible",
                "pair_uid",
                "peptide_uid",
                "affibody_uid",
            ]
        ].sort_values(["weak_label", "pair_uid"], ascending=[False, True]).reset_index(drop=True)
        _require(not bool(labels["pair_uid"].duplicated().any()), "duplicate arm pair")
        label_blocks.append(labels)
        summary_rows.append(
            {
                "arm": arm,
                "arm_name": ARM_SPECIFICATIONS[arm]["name"],
                "role": ARM_SPECIFICATIONS[arm]["role"],
                "rounds": ",".join(str(value) for value in ARM_SPECIFICATIONS[arm]["rounds"]),
                "score_kind": ARM_SPECIFICATIONS[arm]["score_kind"],
                "union_rows": int(union_rows),
                "requested_top2_rank": int(requested_rank),
                "inclusive_cutoff": cutoff,
                "positive_before_design_and_identity_filter": int(selected_before_filter),
                "eligible_positive": int(len(selected)),
                "eligible_negative": int(len(negatives)),
                "natural_training_rows": int(len(labels)),
                "positive_membership_sha256": membership_sha256(selected),
                "negative_membership_sha256": negative_hash,
            }
        )

    _require(len(scores["A"]) == EXPECTED_BASELINE_UNION_ROWS, "baseline union size changed")
    baseline_summary = summary_rows[0]
    _require(
        float(baseline_summary["inclusive_cutoff"]) == EXPECTED_BASELINE_CUTOFF,
        "baseline cutoff changed",
    )

    arm_labels = pd.concat(label_blocks, ignore_index=True)
    _require(
        not bool(arm_labels[["arm", "pair_uid"]].duplicated().any()),
        "duplicate arm/pair key",
    )
    size_blocks = []
    target = EXPECTED_ELIGIBLE_POSITIVES["A"]
    for arm in ("B", "C", "D"):
        for seed in SIZE_MATCH_SEEDS:
            membership = deterministic_size_match(positive_by_arm[arm], target, seed)
            membership.insert(0, "arm", arm)
            membership.insert(1, "sampling_seed", int(seed))
            membership.insert(2, "target_positive_rows", int(target))
            size_blocks.append(membership)
    size_membership = pd.concat(size_blocks, ignore_index=True)
    _require(
        not bool(size_membership[["arm", "sampling_seed", "pair_uid"]].duplicated().any()),
        "duplicate size-matched membership key",
    )

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    output_paths = {
        "arm_labels.csv": output_dir / "arm_labels.csv",
        "size_matched_positive_membership.csv": output_dir
        / "size_matched_positive_membership.csv",
        "arm_summary.csv": output_dir / "arm_summary.csv",
    }
    _write_csv(arm_labels, output_paths["arm_labels.csv"])
    _write_csv(size_membership, output_paths["size_matched_positive_membership.csv"])
    _write_csv(pd.DataFrame(summary_rows), output_paths["arm_summary.csv"])

    script_path = Path(__file__).resolve()
    raw_hashes = {
        "R{:03d}".format(round_index): {
            "path": str(files[round_index].resolve()),
            "sha256": sha256_file(files[round_index]),
        }
        for round_index in range(1, 15)
    }
    manifest = {
        "schema_version": "liba-later-round-labels-v1",
        "analysis_status": "retrospective_exploratory",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "code": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
            "dependencies": {
                str(path.resolve()): sha256_file(path)
                for path in (
                    REPO_ROOT / "downstream/AffibodyMHC/build_selection_weak_labels.py",
                    REPO_ROOT / "downstream/AffibodyMHC/code_only_baseline.py",
                )
            },
        },
        "configuration": {
            "library": LIBRARY,
            "arms": ARM_SPECIFICATIONS,
            "top_fraction": TOP_FRACTION,
            "top_fraction_ties": "inclusive",
            "negative_definition": "R001 count >= 3 and absent from every R002-R014 table",
            "design_filter": "declared LibA peptide and Affibody alphabets",
            "retention_exclusion": "library-local LibA: exclude any weak pair sharing either retention peptide or retention Affibody identity",
            "retention_values_used_for_labels": False,
            "round_depths_raw_count_sum": round_depths,
            "size_match_target_positive": target,
            "size_match_arms": ["B", "C", "D"],
            "size_match_seeds": list(SIZE_MATCH_SEEDS),
            "size_match_order": "SHA256('size-match|<seed>|<pair_uid>'); take first target rows",
            "downstream_validation_contract": {
                "folds": 3,
                "split_seed": 17,
                "regime": "double_identity_cold",
                "c_grid": [0.001, 0.01, 0.1, 1.0],
                "primary_metric": "macro Spearman over exactly 9 LibA peptide groups",
            },
        },
        "contracts": {
            "eligible_positive_by_arm": EXPECTED_ELIGIBLE_POSITIVES,
            "eligible_negative": EXPECTED_ELIGIBLE_NEGATIVES,
            "eligible_negative_membership_sha256": EXPECTED_NEGATIVE_MEMBERSHIP_SHA256,
        },
        "sources": {
            "raw_root": str(args.raw_root.resolve()),
            "raw_rounds": raw_hashes,
            "retention_csv": {
                "path": str(args.retention_csv.resolve()),
                "sha256": sha256_file(args.retention_csv),
            },
        },
        "rows": {
            "arm_labels": int(len(arm_labels)),
            "size_matched_positive_membership": int(len(size_membership)),
            "arm_summary": int(len(summary_rows)),
        },
        "outputs": {name: sha256_file(path) for name, path in output_paths.items()},
        "permissions": {"directory": "0700", "files": "0600"},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    _write_json(manifest, output_dir / "manifest.json")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "eligible_positive_by_arm": EXPECTED_ELIGIBLE_POSITIVES,
                "eligible_negative": len(negatives),
                "size_matched_membership_rows": len(size_membership),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run(parse_args())
