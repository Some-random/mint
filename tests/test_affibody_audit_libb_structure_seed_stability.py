import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import audit_libb_structure_seed_stability as audit


def _score_frame(family, peptide="AF", rows=20):
    row_index = np.arange(rows, dtype=np.int64)
    frame = pd.DataFrame({
        "candidate_row_index": row_index,
        "pair_uid": ["%s-pair-%02d" % (peptide, i) for i in row_index],
        "peptide_design_code": peptide,
        "affibody_design_code": ["A%04d" % i for i in row_index],
    })
    # Two seeds prefer rows 0..9, one prefers 5..14, and two prefer 10..19.
    orders = (
        np.arange(rows),
        np.arange(rows),
        np.abs(row_index - 5),
        np.arange(rows)[::-1],
        np.arange(rows)[::-1],
    )
    seed_values = []
    for seed, order in zip(audit.SEEDS, orders):
        probability = 0.99 - np.asarray(order, dtype=float) * 0.02
        frame["%s_probability_seed_%d" % (family, seed)] = probability
        seed_values.append(probability)
    seed_matrix = np.stack(seed_values, axis=1)
    clipped = np.clip(seed_matrix, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    aggregate = 1.0 / (1.0 + np.exp(-logits.mean(axis=1)))
    frame[audit.FAMILY_SPECS[family]["probability"]] = aggregate
    frame[audit.FAMILY_SPECS[family]["seed_sd"]] = seed_matrix.std(axis=1, ddof=1)
    return frame


def _shortlist(frame, family, positions=(0, 15)):
    aggregate_column = audit.FAMILY_SPECS[family]["probability"]
    chosen = frame.iloc[list(positions)].copy()
    chosen = chosen.sort_values(aggregate_column, ascending=False, kind="mergesort")
    return pd.DataFrame({
        "pair_uid": chosen["pair_uid"].astype(str).tolist(),
        "peptide_design_code": chosen["peptide_design_code"].astype(str).tolist(),
        "affibody_design_code": chosen["affibody_design_code"].astype(str).tolist(),
        "wetlab_rank": list(range(1, len(chosen) + 1)),
        "ensemble_score": chosen[aggregate_column].to_numpy(float),
    })


def test_analyze_peptide_calculates_pairwise_and_membership_without_reranking():
    scores = _score_frame("stab")
    shortlist = _shortlist(scores, "stab")

    membership, summary, pairwise = audit.analyze_peptide(
        scores, shortlist, "stab", "AF"
    )

    assert len(pairwise) == 10
    seed_1_2 = next(
        row for row in pairwise
        if row["seed_a"] == audit.SEEDS[0] and row["seed_b"] == audit.SEEDS[1]
    )
    assert seed_1_2["intersection_count"] == 10
    assert seed_1_2["jaccard"] == pytest.approx(1.0)
    assert not summary["all_five_top1_agree"]
    assert membership["pair_uid"].tolist() == shortlist["pair_uid"].tolist()
    assert membership["wetlab_rank"].tolist() == shortlist["wetlab_rank"].tolist()
    assert membership["seed_top10_count"].between(0, 5).all()


def test_stable_top_k_uses_candidate_index_to_break_exact_ties():
    values = np.ones(12, dtype=float)
    immutable_index = np.arange(12, dtype=np.int64)[::-1]
    selected_positions = audit._stable_top_k_indices(values, immutable_index, 10)
    assert set(immutable_index[selected_positions]) == set(range(10))


def test_analyze_peptide_rejects_retention_outcome_column():
    scores = _score_frame("rde")
    scores["measured_retention"] = 88.0
    with pytest.raises(ValueError, match="forbidden outcomes"):
        audit.analyze_peptide(scores, _shortlist(scores, "rde"), "rde", "AF")


def test_shortlist_rejects_fractional_rank(tmp_path):
    scores = _score_frame("stab")
    shortlist = _shortlist(scores, "stab")
    shortlist.loc[0, "wetlab_rank"] = 1.5
    path = tmp_path / "shortlist.csv"
    shortlist.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-integral"):
        audit._read_shortlist(path, "stab")


def _write_family_fixture(root, family, peptides):
    score_dir = root / (family + "_scores")
    score_dir.mkdir()
    partitions = []
    shortlist_blocks = []
    total_rows = 0
    for peptide in peptides:
        frame = _score_frame(family, peptide=peptide)
        path = score_dir / ("peptide_%s.parquet" % peptide)
        frame.to_parquet(path, index=False)
        total_rows += len(frame)
        partitions.append({
            "peptide_design_code": peptide,
            "candidate_rows": len(frame),
            "output_file": path.name,
            "output_bytes": path.stat().st_size,
            "output_sha256": audit.sha256_file(path),
        })
        # Reproduce the real RDE edge case: one peptide has no aggregate
        # candidate because no row crossed the frozen cutoff.
        if not (family == "rde" and peptide == peptides[-1]):
            shortlist_blocks.append(_shortlist(frame, family))
    manifest = {
        "schema_version": audit.MERGED_SCHEMA,
        "family": family,
        "model": audit.FAMILY_SPECS[family]["model"],
        "readout_seeds": list(audit.SEEDS),
        "validation": {key: True for key in audit.REQUIRED_MERGE_VALIDATIONS},
        "outputs": {"candidate_rows": total_rows, "partitions": partitions},
    }
    (score_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    shortlist = pd.concat(shortlist_blocks, ignore_index=True)
    shortlist_path = root / (family + "_shortlist.csv")
    shortlist.to_csv(shortlist_path, index=False)
    return score_dir, shortlist_path, shortlist


def test_cli_publishes_atomic_outcome_blind_audit_and_preserves_shortlists(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(audit, "EXPECTED_PEPTIDES", ("AF", "AH"))
    private = tmp_path / "private_data"
    private.mkdir()
    stab_dir, stab_shortlist_path, stab_shortlist = _write_family_fixture(
        tmp_path, "stab", audit.EXPECTED_PEPTIDES
    )
    rde_dir, rde_shortlist_path, rde_shortlist = _write_family_fixture(
        tmp_path, "rde", audit.EXPECTED_PEPTIDES
    )
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    output = private / "seed_stability"

    manifest = audit.run(SimpleNamespace(
        stab_score_dir=stab_dir,
        rde_score_dir=rde_dir,
        stab_shortlist=stab_shortlist_path,
        rde_shortlist=rde_shortlist_path,
        output_dir=output,
        skip_partition_hashes=False,
    ))

    assert output.is_dir()
    assert (output / "per_peptide_seed_stability.csv").is_file()
    assert (output / "pairwise_seed_top10_jaccard.csv").is_file()
    assert (output / "structure_seed_stability_summary.json").is_file()
    written = pd.read_csv(output / "aggregate_shortlist_seed_membership.csv")
    expected_uids = (
        stab_shortlist["pair_uid"].tolist() + rde_shortlist["pair_uid"].tolist()
    )
    expected_ranks = (
        stab_shortlist["wetlab_rank"].tolist() + rde_shortlist["wetlab_rank"].tolist()
    )
    assert written["pair_uid"].tolist() == expected_uids
    assert written["wetlab_rank"].tolist() == expected_ranks
    assert manifest["invariants"]["retention_outcomes_read"] is False
    assert manifest["invariants"]["aggregate_shortlist_ranks_changed"] is False
    assert manifest["parameters"]["partition_hashes_verified"] is True
    assert manifest["summary"]["rde"]["targets_without_aggregate_shortlist_candidates"] == ["AH"]
    on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["outputs"]["pairwise_seed_top10_jaccard"]["rows"] == 40


def test_run_refuses_to_overwrite_an_existing_output(tmp_path, monkeypatch):
    private = tmp_path / "private_data"
    output = private / "existing"
    output.mkdir(parents=True)
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    with pytest.raises(ValueError, match="refusing overwrite"):
        audit.run(SimpleNamespace(
            output_dir=output,
            stab_score_dir=tmp_path / "unused",
            rde_score_dir=tmp_path / "unused",
            stab_shortlist=tmp_path / "unused.csv",
            rde_shortlist=tmp_path / "unused.csv",
            skip_partition_hashes=False,
        ))
