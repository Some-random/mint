import itertools

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import code_only_baseline as baseline
from downstream.AffibodyMHC import finetune_mint_retention as finetune


def _codes(length, count):
    alphabet = baseline.AA_ALPHABET
    values = []
    for letters in itertools.product(alphabet, repeat=length):
        values.append("".join(letters))
        if len(values) == count:
            return values
    raise AssertionError("insufficient synthetic codes")


def _synthetic_retention_table(path):
    rows = []
    for library, spec in baseline.LIBRARY_SPECS.items():
        peptides = _codes(spec["pep_length"], spec["peptide_codes"])
        # Exercise the missing-value-token parsing guard with a legitimate code.
        peptides[0] = "NA"
        affibodies = _codes(spec["aff_length"], spec["affibody_codes"])
        for peptide_index, peptide in enumerate(peptides):
            for affibody_index, affibody in enumerate(affibodies):
                missing = library == "LibB" and peptide_index == 0 and affibody_index == 0
                retention = 20.0 + 6.0 * peptide_index + 3.0 * affibody_index
                retention = min(retention, 100.0)
                rows.append(
                    {
                        "provisional_matrix_key": "{}|{}|{}".format(
                            library, affibody, peptide
                        ),
                        "library": library,
                        "affibody_design_code": affibody,
                        "peptide_design_code": peptide,
                        "axis_assignment_basis": "synthetic-test",
                        "retention_percent": "" if missing else str(retention),
                        "retention_time_min_as_labeled": "30",
                        "timepoint_status": "synthetic-test",
                        "binder_label_ge_75": "" if missing else str(int(retention >= 75.0)),
                        "measurement_missing": "1" if missing else "0",
                        "source_slide": "0",
                    }
                )
    frame = pd.DataFrame(rows, columns=baseline.EXPECTED_COLUMNS)
    frame.to_csv(path, index=False)


def test_loader_preserves_literal_na_and_expands_position_features(tmp_path):
    path = tmp_path / "retention.csv"
    _synthetic_retention_table(path)

    frame = baseline.load_retention_table(path)

    assert len(frame) == 228
    assert int(frame["target_retention"].notna().sum()) == 227
    assert int(frame["target_binder"].notna().sum()) == 227
    assert bool(frame.loc[frame["target_retention"].isna(), "target_binder"].isna().all())
    assert "NA" in set(frame["peptide_design_code"])
    assert len(baseline.feature_columns("site_residue_additive", "LibA")) == 6
    assert len(baseline.feature_columns("site_residue_additive", "LibB")) == 7
    assert frame["pair_uid"].nunique() == 228


def test_double_cold_outer_split_excludes_test_row_and_column(tmp_path):
    path = tmp_path / "retention.csv"
    _synthetic_retention_table(path)
    frame = baseline.supervised_library_frame(baseline.load_retention_table(path), "LibA")

    splits = baseline.make_outer_splits(frame, "double_cold", seed=7, random_repeats=1)

    assert len(splits) == len(frame)
    for split in splits:
        assert len(split["test"]) == 1
        test = frame.iloc[split["test"][0]]
        train = frame.iloc[split["train"]]
        assert test["peptide_design_code"] not in set(train["peptide_design_code"])
        assert test["affibody_design_code"] not in set(train["affibody_design_code"])
        assert len(split["train"]) + len(split["test"]) + len(split["guarded"]) == len(frame)


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_double_cold_inner_split_covers_every_pair_once(tmp_path, task):
    path = tmp_path / "retention.csv"
    _synthetic_retention_table(path)
    frame = baseline.supervised_library_frame(baseline.load_retention_table(path), "LibA")

    splits, audit = baseline.make_inner_splits(frame, "double_cold", seed=7, task=task)
    validation = np.concatenate([test for _, test in splits])

    assert len(splits) == 9
    assert sorted(validation.tolist()) == list(range(len(frame)))
    assert audit["validation_unique_pairs"] == len(frame)
    assert audit["validation_min_count"] == audit["validation_max_count"] == 1
    assert all(frame.iloc[train]["target_binder"].nunique() == 2 for train, _ in splits)


def test_double_cold_inner_assignment_is_shared_and_deterministic(tmp_path):
    path = tmp_path / "retention.csv"
    _synthetic_retention_table(path)
    frame = baseline.supervised_library_frame(baseline.load_retention_table(path), "LibA")

    regression, _ = baseline.make_inner_splits(frame, "double_cold", seed=19, task="regression")
    classification, _ = baseline.make_inner_splits(
        frame, "double_cold", seed=19, task="classification"
    )
    repeated, _ = baseline.make_inner_splits(frame, "double_cold", seed=19, task="regression")

    for first, second, third in zip(regression, classification, repeated):
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        assert np.array_equal(first[0], third[0])
        assert np.array_equal(first[1], third[1])


def test_double_cold_inner_assignment_retries_atomically(tmp_path):
    path = tmp_path / "retention.csv"
    _synthetic_retention_table(path)
    frame = baseline.supervised_library_frame(baseline.load_retention_table(path), "LibA")
    frame["target_binder"] = 0
    frame.loc[[1, 6, 18, 26, 29, 61, 69, 79], "target_binder"] = 1

    splits, audit = baseline.make_inner_splits(
        frame, "double_cold", seed=7, task="classification"
    )
    validation = np.concatenate([test for _, test in splits])

    assert audit["assignment_attempt"] == 1
    assert len(splits) == 9
    assert sorted(validation.tolist()) == list(range(len(frame)))
    assert all(frame.iloc[train]["target_binder"].nunique() == 2 for train, _ in splits)


def test_site_model_produces_finite_predictions(tmp_path):
    path = tmp_path / "retention.csv"
    _synthetic_retention_table(path)
    frame = baseline.supervised_library_frame(baseline.load_retention_table(path), "LibA")
    matrix = baseline.feature_matrix(frame, "site_residue_additive", "LibA")
    train = np.arange(0, 90)
    test = np.arange(90, len(frame))

    model = baseline.make_estimator(
        "site_residue_additive", "regression", "LibA", parameter=1.0, seed=7
    )
    model.fit(matrix[train], frame.iloc[train]["target_retention"])
    prediction = baseline._predict_score(model, "regression", matrix[test])

    assert prediction.shape == (len(test),)
    assert np.isfinite(prediction).all()


def test_output_path_must_be_private_and_git_ignored():
    repo_root = baseline.Path(baseline.__file__).resolve().parents[2]
    accepted = repo_root / "private_data" / "unit-test-nonexistent-output"
    assert baseline.validate_private_output_path(accepted, repo_root) == accepted
    with pytest.raises(ValueError, match="inside.*private_data"):
        baseline.validate_private_output_path(repo_root / "results" / "unsafe", repo_root)


def test_run_fingerprint_changes_with_configuration():
    first = baseline.run_fingerprint("source", "script", {"seed": 1})
    second = baseline.run_fingerprint("source", "script", {"seed": 2})
    assert len(first) == 12
    assert first != second


def test_finetune_random_split_has_disjoint_three_way_membership(tmp_path):
    path = tmp_path / "retention.csv"
    _synthetic_retention_table(path)
    frame = baseline.supervised_library_frame(baseline.load_retention_table(path), "LibA")

    roles, details = finetune.make_split(frame, "LibA", "random_pair", fold=0, seed=20260807)

    assert details["counts"] == {"train": 64, "validation": 22, "test": 22, "guarded": 0}
    assert set(roles) == {"train", "validation", "test"}
    assert sum(details["counts"].values()) == len(frame)
    repeated, repeated_details = finetune.make_split(
        frame, "LibA", "random_pair", fold=0, seed=20260807
    )
    assert np.array_equal(roles, repeated)
    assert details["membership_sha256"] == repeated_details["membership_sha256"]


def test_finetune_blocked_split_holds_both_partner_identities(tmp_path):
    path = tmp_path / "retention.csv"
    _synthetic_retention_table(path)
    frame = baseline.supervised_library_frame(baseline.load_retention_table(path), "LibA")

    for fold in range(3):
        roles, details = finetune.make_split(
            frame, "LibA", "blocked3", fold=fold, seed=20260807
        )
        development = frame.loc[np.isin(roles, ["train", "validation"])]
        test = frame.loc[roles == "test"]
        assert set(development["peptide_uid"]).isdisjoint(set(test["peptide_uid"]))
        assert set(development["affibody_uid"]).isdisjoint(set(test["affibody_uid"]))
        assert sum(details["counts"].values()) == len(frame)


def test_lora_linear_is_exact_noop_at_initialization():
    torch = pytest.importorskip("torch")
    base = torch.nn.Linear(7, 5)
    original = base.weight.detach().clone(), base.bias.detach().clone()
    adapter = finetune.LoRALinear(base, rank=2, alpha=4.0, dropout=0.05)
    adapter.eval()
    values = torch.randn(3, 7)

    expected = torch.nn.functional.linear(values, original[0], original[1])
    observed = adapter(values)

    assert torch.equal(observed, expected)
    assert not any(parameter.requires_grad for parameter in adapter.base.parameters())
    assert adapter.lora_a.requires_grad and adapter.lora_b.requires_grad


def test_model_names_are_safe_under_default_csv_na_parsing(tmp_path):
    path = tmp_path / "models.csv"
    pd.DataFrame({"model": baseline.MODEL_NAMES}).to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert not bool(loaded["model"].isna().any())
    assert "mean_prior" in set(loaded["model"])


def test_site_coverage_is_aggregated_without_codes(tmp_path):
    path = tmp_path / "retention.csv"
    _synthetic_retention_table(path)
    frame = baseline.supervised_library_frame(baseline.load_retention_table(path), "LibA")
    events = baseline.calculate_site_coverage_events(
        frame, "LibA", "double_cold", seed=7, random_repeats=1
    )
    by_repeat, summary = baseline.summarize_site_coverage(events)

    assert len(events) == len(frame)
    assert int(by_repeat.loc[0, "n"]) == len(frame)
    assert int(summary.loc[0, "repeats"]) == 1
    assert not any("code" in column for column in summary.columns)


def test_generated_summary_tables_have_consistent_column_counts(tmp_path):
    metrics = pd.DataFrame(
        [
            {
                "library": "LibA",
                "task": "regression",
                "split_scheme": "random_pair",
                "model": "mean_prior",
                "repeats": 1,
                "mae": 1.0,
                "spearman": 0.0,
                "pearson": 0.0,
                "r2": 0.0,
                "delta_mae_vs_mean_prior": 0.0,
            },
            {
                "library": "LibA",
                "task": "classification",
                "split_scheme": "random_pair",
                "model": "mean_prior",
                "repeats": 1,
                "brier": 0.2,
                "log_loss": 0.6,
                "average_precision": 0.4,
                "roc_auc": 0.5,
                "mcc": 0.0,
                "delta_average_precision_vs_prevalence": 0.0,
            },
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "library": "LibA",
                "split_scheme": "random_pair",
                "n": 108,
                "repeats": 1,
                "any_unseen_count": 0.0,
                "any_unseen_fraction": 0.0,
                "mean_unseen_site_count": 0.0,
            }
        ]
    )
    path = tmp_path / "summary.md"
    baseline.write_summary(path, metrics, coverage, {"run_id": "test"})
    text = path.read_text()

    regression = text.split("## Regression", 1)[1].split("## Classification", 1)[0]
    classification = text.split("## Classification", 1)[1].split(
        "## Designed-residue coverage", 1
    )[0]
    coverage_section = text.split("## Designed-residue coverage", 1)[1].split(
        "## Interpretation constraints", 1
    )[0]
    assert {line.count("|") for line in regression.splitlines() if line.startswith("|")} == {9}
    assert {
        line.count("|") for line in classification.splitlines() if line.startswith("|")
    } == {10}
    assert {
        line.count("|") for line in coverage_section.splitlines() if line.startswith("|")
    } == {7}
