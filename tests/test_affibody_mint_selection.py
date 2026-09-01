import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from downstream.AffibodyMHC.build_sequence_table import (
    CHAIN1_LENGTH,
    CHAIN2_LENGTH,
    EXPECTED_AFFIBODY_X_POSITIONS,
    EXPECTED_CONSTRUCT_LENGTH,
    EXPECTED_X_POSITIONS,
    OMITTED_LINKER,
)
from downstream.AffibodyMHC.finetune_mint_retention import LoRALinear
from downstream.AffibodyMHC import finetune_mint_selection as selection


def _weak_row(
    pair_uid,
    peptide_uid,
    affibody_uid,
    label,
    r001_count=0,
    on_design=1,
    identity_cold=1,
    library="LibA",
    pep="AA",
    aff="AAAA",
):
    return {
        "library": library,
        "pep": pep,
        "aff": aff,
        "r001_count": r001_count,
        "weak_label": label,
        "within_declared_library_alphabet": on_design,
        "strict_retention_identity_cold_eligible": identity_cold,
        "pair_uid": pair_uid,
        "peptide_uid": peptide_uid,
        "affibody_uid": affibody_uid,
    }


def _uids_for_bin(axis, expected_bin, count, split_seed=19, n_bins=3):
    values = []
    candidate = 0
    while len(values) < count:
        uid = "{}-{}".format(axis, candidate)
        if selection.identity_bin(uid, axis, split_seed, n_bins) == expected_bin:
            values.append(uid)
        candidate += 1
    return values


def _liba_template():
    template = list("A" * EXPECTED_CONSTRUCT_LENGTH)
    template[CHAIN1_LENGTH : CHAIN1_LENGTH + len(OMITTED_LINKER)] = OMITTED_LINKER
    for position in EXPECTED_X_POSITIONS["LibA"]:
        template[position - 1] = "X"
    return "".join(template)


def _lineage_files(tmp_path, mutation=None):
    weak_labels = tmp_path / "weak_labels.csv"
    retention_sequences = tmp_path / "retention_sequences.csv"
    sequence_zip = tmp_path / "provider.zip"
    weak_labels.write_bytes(b"weak-label-table\n")
    retention_sequences.write_bytes(b"retention-sequence-table\n")
    sequence_zip.write_bytes(b"provider-sequence-archive\n")

    weak_label_hash = selection.sha256_file(weak_labels)
    retention_hash = selection.sha256_file(retention_sequences)
    sequence_zip_hash = selection.sha256_file(sequence_zip)
    retention_matrix_hash = "retention-matrix-sha256"
    template_deck_hash = "template-deck-sha256"
    weak_manifest = {
        "outputs": {"weak_labels.csv": weak_label_hash},
        "sources": {
            "sequence_zip": {
                "sha256": sequence_zip_hash,
                "template_deck_sha256": template_deck_hash,
            },
            "retention_csv": {"sha256": retention_matrix_hash},
        },
    }
    retention_manifest = {
        "output": {"sha256": retention_hash},
        "sources": {
            "sequence_zip": {
                "sha256": sequence_zip_hash,
                "deck_sha256": template_deck_hash,
            },
            "retention_csv": {"sha256": retention_matrix_hash},
        },
    }
    if mutation == "weak_output":
        weak_manifest["outputs"]["weak_labels.csv"] = "wrong"
    elif mutation == "retention_output":
        retention_manifest["output"]["sha256"] = "wrong"
    elif mutation == "weak_zip":
        weak_manifest["sources"]["sequence_zip"]["sha256"] = "wrong"
    elif mutation == "retention_zip":
        retention_manifest["sources"]["sequence_zip"]["sha256"] = "wrong"
    elif mutation == "retention_matrix":
        retention_manifest["sources"]["retention_csv"]["sha256"] = "wrong"
    elif mutation == "template_deck":
        retention_manifest["sources"]["sequence_zip"]["deck_sha256"] = "wrong"

    weak_manifest_path = tmp_path / "weak_manifest.json"
    retention_manifest_path = tmp_path / "retention_manifest.json"
    weak_manifest_path.write_text(json.dumps(weak_manifest), encoding="utf-8")
    retention_manifest_path.write_text(
        json.dumps(retention_manifest), encoding="utf-8"
    )
    return {
        "weak_manifest": weak_manifest_path,
        "weak_labels": weak_labels,
        "retention_manifest": retention_manifest_path,
        "retention_sequences": retention_sequences,
        "sequence_zip": sequence_zip,
        "expected": {
            "weak_labels_sha256": weak_label_hash,
            "retention_sequences_sha256": retention_hash,
            "sequence_zip_sha256": sequence_zip_hash,
            "retention_matrix_sha256": retention_matrix_hash,
            "template_deck_sha256": template_deck_hash,
        },
    }


def test_validate_input_lineage_accepts_matching_content_hashes(tmp_path):
    files = _lineage_files(tmp_path)

    observed = selection.validate_input_lineage(
        files["weak_manifest"],
        files["weak_labels"],
        files["retention_manifest"],
        files["retention_sequences"],
        files["sequence_zip"],
    )

    assert observed == files["expected"]


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("weak_output", "weak-label manifest/output hash mismatch"),
        ("retention_output", "retention manifest/output hash mismatch"),
        ("weak_zip", "weak-label manifest uses a different sequence ZIP"),
        ("retention_zip", "retention manifest uses a different sequence ZIP"),
        (
            "retention_matrix",
            "weak labels and retention sequences use different retention matrices",
        ),
        (
            "template_deck",
            "weak labels and retention sequences use different template decks",
        ),
    ],
)
def test_validate_input_lineage_rejects_any_mismatched_hash(tmp_path, mutation, error):
    files = _lineage_files(tmp_path, mutation=mutation)

    with pytest.raises(ValueError, match=error):
        selection.validate_input_lineage(
            files["weak_manifest"],
            files["weak_labels"],
            files["retention_manifest"],
            files["retention_sequences"],
            files["sequence_zip"],
        )


def test_load_weak_primary_applies_on_design_cold_and_negative_count_filters(
    tmp_path, monkeypatch
):
    rows = [
        _weak_row("positive", "pep-1", "aff-1", 1, r001_count=0),
        _weak_row("negative-kept", "pep-2", "aff-2", 0, r001_count=3),
        _weak_row("negative-low", "pep-3", "aff-3", 0, r001_count=2),
        _weak_row(
            "off-design", "pep-4", "aff-4", 1, on_design=0
        ),
        _weak_row(
            "not-cold", "pep-5", "aff-5", 0, r001_count=9, identity_cold=0
        ),
        _weak_row(
            "other-library",
            "pep-6",
            "aff-6",
            1,
            library="LibB",
        ),
    ]
    path = tmp_path / "weak_labels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setitem(
        selection.EXPECTED_PRIMARY,
        "LibA",
        {"positive": 1, "negative": 1},
    )

    primary = selection.load_weak_primary(path, "LibA", min_negative_r001_count=3)

    assert primary["pair_uid"].tolist() == ["negative-kept", "positive"]
    assert primary.set_index("pair_uid")["weak_label"].to_dict() == {
        "negative-kept": 0,
        "positive": 1,
    }


def test_split_and_balance_is_deterministic_balanced_and_identity_disjoint():
    split_seed = 19
    n_bins = 3
    held_peptides = _uids_for_bin("pep", 0, 4, split_seed, n_bins)
    train_peptides = _uids_for_bin("pep", 1, 4, split_seed, n_bins)
    held_affibodies = _uids_for_bin("aff", 0, 4, split_seed, n_bins)
    train_affibodies = _uids_for_bin("aff", 1, 4, split_seed, n_bins)
    peptides = held_peptides + train_peptides
    affibodies = held_affibodies + train_affibodies
    rows = []
    for peptide_index, peptide_uid in enumerate(peptides):
        for affibody_index, affibody_uid in enumerate(affibodies):
            rows.append(
                _weak_row(
                    "pair-{}-{}".format(peptide_index, affibody_index),
                    peptide_uid,
                    affibody_uid,
                    (peptide_index + affibody_index) % 2,
                    r001_count=5,
                )
            )
    frame = pd.DataFrame(rows)

    first = selection.split_and_balance(
        frame,
        "LibA",
        split_seed=split_seed,
        training_seed=23,
        n_bins=n_bins,
        max_train_per_class=3,
        max_validation_per_class=2,
    )
    second = selection.split_and_balance(
        frame.sample(frac=1.0, random_state=7).reset_index(drop=True),
        "LibA",
        split_seed=split_seed,
        training_seed=23,
        n_bins=n_bins,
        max_train_per_class=3,
        max_validation_per_class=2,
    )

    first_roles = first.set_index("pair_uid")["role"].sort_index()
    second_roles = second.set_index("pair_uid")["role"].sort_index()
    pd.testing.assert_series_equal(first_roles, second_roles)
    train = first.loc[first["role"].eq("train")]
    validation = first.loc[first["role"].eq("validation")]
    assert train.groupby("weak_label").size().to_dict() == {0: 3, 1: 3}
    assert validation.groupby("weak_label").size().to_dict() == {0: 2, 1: 2}
    assert set(train["peptide_uid"]).isdisjoint(validation["peptide_uid"])
    assert set(train["affibody_uid"]).isdisjoint(validation["affibody_uid"])
    assert first.loc[first["role"].eq("guarded")].shape[0] == 32
    assert first.loc[first["role"].eq("unused_validation_cap")].shape[0] == 12


def test_reconstruct_weak_sequences_maps_codes_and_omits_construct_linker():
    template = _liba_template()
    source = pd.DataFrame(
        [
            _weak_row(
                "pair-1",
                "pep-1",
                "aff-1",
                1,
                pep="CD",
                aff="EFGH",
            )
        ]
    )

    reconstructed = selection.reconstruct_weak_sequences(source, template)

    row = reconstructed.iloc[0]
    assert len(row["chain1_smart_hla_linker_peptide_sequence"]) == CHAIN1_LENGTH
    assert len(row["chain2_affibody_sequence"]) == CHAIN2_LENGTH
    assert row["chain1_smart_hla_linker_peptide_sequence"][264:266] == "CD"
    affibody_positions = EXPECTED_AFFIBODY_X_POSITIONS["LibA"]
    observed_affibody_code = "".join(
        row["chain2_affibody_sequence"][position - 1]
        for position in affibody_positions
    )
    assert observed_affibody_code == "EFGH"
    assert not row["chain1_smart_hla_linker_peptide_sequence"].endswith(
        OMITTED_LINKER
    )


def test_validate_retention_sequence_reconstruction_detects_table_or_linker_tampering():
    template = _liba_template()
    full = selection.fill_template(template, "CDEFGH")
    frame = pd.DataFrame(
        [
            {
                "library": "LibA",
                "peptide_design_code": "CD",
                "affibody_design_code": "EFGH",
                "chain1_smart_hla_linker_peptide_sequence": full[:CHAIN1_LENGTH],
                "chain2_affibody_sequence": full[-CHAIN2_LENGTH:],
            }
        ]
    )

    selection.validate_retention_sequence_reconstruction(frame, {"LibA": template})

    bad_chain1 = frame.copy()
    bad_chain1.loc[0, "chain1_smart_hla_linker_peptide_sequence"] = (
        "C" + full[1:CHAIN1_LENGTH]
    )
    with pytest.raises(ValueError, match="retention chain-1 does not match"):
        selection.validate_retention_sequence_reconstruction(
            bad_chain1, {"LibA": template}
        )

    bad_chain2 = frame.copy()
    bad_chain2.loc[0, "chain2_affibody_sequence"] = (
        "C" + full[-CHAIN2_LENGTH + 1 :]
    )
    with pytest.raises(ValueError, match="retention chain-2 does not match"):
        selection.validate_retention_sequence_reconstruction(
            bad_chain2, {"LibA": template}
        )

    bad_linker_template = list(template)
    bad_linker_template[CHAIN1_LENGTH] = "A"
    with pytest.raises(ValueError, match="retention reconstruction linker mismatch"):
        selection.validate_retention_sequence_reconstruction(
            frame, {"LibA": "".join(bad_linker_template)}
        )


class _TinyShapeEquivalentLoRAModel(nn.Module):
    """Match MINT's LoRA parameter count without allocating its checkpoint."""

    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList(
            [LoRALinear(nn.Linear(1, 2559), rank=2) for _ in range(4)]
        )
        self.head = nn.Linear(2560, 1)
        self.register_buffer("feature_mean", torch.zeros(2560))
        self.register_buffer("feature_scale", torch.ones(2560))


def test_trainable_audit_allows_only_expected_lora_tensors_and_head():
    model = _TinyShapeEquivalentLoRAModel()

    names, count = selection.trainable_audit(model)

    assert count == selection.EXPECTED_TRAINABLE
    assert set(name for name in names if name.startswith("head.")) == {
        "head.weight",
        "head.bias",
    }
    assert len([name for name in names if "lora_" in name]) == 8
    assert not any(".base." in name for name in names)
    assert "feature_mean" not in names
    assert "feature_scale" not in names

    # Keep the total tensor and parameter counts unchanged while substituting a
    # same-sized rogue encoder parameter; the name-based safety check must fail.
    model.adapters[0].lora_a.requires_grad_(False)
    model.rogue_encoder_weight = nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="non-LoRA encoder tensor trainable"):
        selection.trainable_audit(model)


def test_fit_and_apply_standardization_use_train_only_statistics_and_constant_fallback():
    train = np.asarray(
        [
            [1.0, 10.0, 5.0],
            [3.0, 10.0, 9.0],
            [5.0, 10.0, 13.0],
        ],
        dtype=np.float32,
    )
    holdout = np.asarray([[7.0, 12.0, 17.0]], dtype=np.float32)

    mean, scale = selection.fit_standardization(train)
    train_z = selection.apply_standardization(train, mean, scale)
    holdout_z = selection.apply_standardization(holdout, mean, scale)

    np.testing.assert_allclose(mean, [3.0, 10.0, 9.0])
    np.testing.assert_allclose(
        scale,
        [np.sqrt(8.0 / 3.0), 1.0, np.sqrt(32.0 / 3.0)],
    )
    np.testing.assert_allclose(train_z.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(train_z[:, [0, 2]].std(axis=0), [1.0, 1.0])
    np.testing.assert_allclose(train_z[:, 1], 0.0)
    np.testing.assert_allclose(
        holdout_z,
        [[np.sqrt(6.0), 2.0, np.sqrt(6.0)]],
        rtol=1e-6,
    )
    assert train_z.dtype == np.float32
    assert holdout_z.dtype == np.float32


class _PassthroughFeatureWrapper(nn.Module):
    def forward(self, chains, chain_ids):
        del chain_ids
        return chains


def _checkpoint_free_selection_classifier():
    model = selection.MINTSelectionClassifier.__new__(
        selection.MINTSelectionClassifier
    )
    nn.Module.__init__(model)
    model.wrapper = _PassthroughFeatureWrapper()
    model.head = nn.Linear(2560, 1, bias=False)
    model.register_buffer("feature_mean", torch.zeros(2560, dtype=torch.float32))
    model.register_buffer("feature_scale", torch.ones(2560, dtype=torch.float32))
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.weight[0, 0] = 1.0
    return model


def test_set_feature_standardization_updates_buffers_and_live_forward_without_checkpoint():
    model = _checkpoint_free_selection_classifier()
    mean = np.linspace(-2.0, 2.0, 2560, dtype=np.float64)
    scale = np.linspace(0.5, 3.0, 2560, dtype=np.float64)

    model.set_feature_standardization(mean, scale)

    np.testing.assert_allclose(model.feature_mean.numpy(), mean, rtol=1e-6)
    np.testing.assert_allclose(model.feature_scale.numpy(), scale, rtol=1e-6)
    assert model.feature_mean.dtype == torch.float32
    assert model.feature_scale.dtype == torch.float32
    assert {"feature_mean", "feature_scale"}.issubset(model.state_dict())
    features = torch.from_numpy(mean.astype(np.float32)).reshape(1, -1)
    features[0, 0] += float(scale[0] * 3.0)
    np.testing.assert_allclose(
        model(features, chain_ids=None).detach().numpy(), [3.0], rtol=1e-6
    )


@pytest.mark.parametrize(
    "mean,scale,error",
    [
        (np.zeros(3), np.ones(2560), "feature mean shape mismatch"),
        (np.zeros(2560), np.ones(3), "feature scale shape mismatch"),
        (
            np.concatenate(([np.nan], np.zeros(2559))),
            np.ones(2560),
            "non-finite feature mean",
        ),
        (
            np.zeros(2560),
            np.concatenate(([np.inf], np.ones(2559))),
            "non-finite feature scale",
        ),
        (
            np.zeros(2560),
            np.concatenate(([0.0], np.ones(2559))),
            "nonpositive feature scale",
        ),
    ],
)
def test_set_feature_standardization_rejects_invalid_statistics(mean, scale, error):
    model = _checkpoint_free_selection_classifier()

    with pytest.raises(ValueError, match=error):
        model.set_feature_standardization(mean, scale)
