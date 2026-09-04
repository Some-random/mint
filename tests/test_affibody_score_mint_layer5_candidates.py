import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from downstream.AffibodyMHC.score_mint_layer5_candidates import (
    FEATURE_DIMENSION,
    FEATURE_NAME,
    INPUT_COLUMNS,
    LAYER,
    LIBA_LAYER33_CONTRACT,
    LIBA_LAYER9_CONTRACT,
    LIBB_LAYER5_CONTRACT,
    POOLING_DESCRIPTION,
    EarlyStopLayerExtractor,
    EarlyStopMultiLayerExtractor,
    _load_reference_scores,
    _output_schema,
    _score_record_batch,
    _score_record_batch_dual_liba,
    _validate_candidate_values,
    _validate_parity_receipt,
    _validate_reference_contract,
    load_linear_head,
    resolve_scoring_contract,
    score_features,
    sha256_file,
)
from downstream.AffibodyMHC.code_only_baseline import opaque_id
from mint.helpers.extract import CollateFn


class _FakeLayer(torch.nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = float(increment)
        self.calls = 0

    def forward(
        self,
        values,
        self_attn_padding_mask=None,
        need_head_weights=False,
        self_attn_mask=None,
    ):
        del self_attn_padding_mask, need_head_weights, self_attn_mask
        self.calls += 1
        return values + self.increment, None


class _FakeNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, values):
        self.calls += 1
        return values + 0.25


class _FakeHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, values):
        self.calls += 1
        return values


class _FakeMint(torch.nn.Module):
    def __init__(self, residue_dimension=4, layers=7):
        super().__init__()
        alphabet = CollateFn(512).alphabet
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.padding_idx = alphabet.padding_idx
        self.residue_dimension = int(residue_dimension)
        self.layers = torch.nn.ModuleList(
            [_FakeLayer(index + 1) for index in range(int(layers))]
        )
        self.emb_layer_norm_after = _FakeNorm()
        self.lm_head = _FakeHead()

    def forward(self, tokens, chain_ids=None, repr_layers=None):
        del chain_ids
        requested = set(repr_layers or [])
        values = tokens.float().unsqueeze(-1).expand(
            tokens.shape[0], tokens.shape[1], self.residue_dimension
        )
        values = values.transpose(0, 1)
        representations = {}
        for index, layer in enumerate(self.layers, start=1):
            values, _ = layer(values)
            if index in requested:
                representations[index] = values.transpose(0, 1)
        values = self.emb_layer_norm_after(values).transpose(0, 1)
        if len(self.layers) in requested:
            representations[len(self.layers)] = values
        return {"representations": representations, "logits": self.lm_head(values)}


def _head():
    return {
        "mean": np.linspace(-0.2, 0.2, FEATURE_DIMENSION, dtype=np.float64),
        "scale": np.linspace(0.5, 1.5, FEATURE_DIMENSION, dtype=np.float64),
        "coefficient": np.linspace(-0.01, 0.01, FEATURE_DIMENSION, dtype=np.float64),
        "intercept": -0.25,
    }


def test_linear_head_loader_and_numerical_path(tmp_path: Path):
    source = _head()
    path = tmp_path / "head.npz"
    np.savez(
        path,
        mean=source["mean"],
        scale=source["scale"],
        coef=source["coefficient"].reshape(1, -1),
        intercept=np.asarray([source["intercept"]]),
        layer=np.asarray([LAYER]),
        feature_name=np.asarray(FEATURE_NAME),
        pooling=np.asarray(POOLING_DESCRIPTION),
    )
    loaded = load_linear_head(path)
    assert loaded["keys"] == {
        "mean": "mean",
        "scale": "scale",
        "coefficient": "coef",
        "intercept": "intercept",
    }

    rng = np.random.RandomState(9)
    features = rng.normal(size=(5, FEATURE_DIMENSION)).astype(np.float32)
    logits, scores = score_features(features, loaded)
    standardized = ((features.astype(np.float64) - source["mean"]) / source["scale"]).astype(
        np.float32
    )
    expected_logits = standardized @ source["coefficient"] + source["intercept"]
    expected_scores = 1.0 / (1.0 + np.exp(-expected_logits))
    np.testing.assert_array_equal(logits, expected_logits)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-15, atol=0.0)


def test_loader_auto_selects_canonical_mint_prefix_and_rejects_wrong_layer(tmp_path: Path):
    source = _head()
    path = tmp_path / "deployment_heads.npz"
    np.savez(
        path,
        mint_mean=source["mean"],
        mint_scale=source["scale"],
        mint_coef=source["coefficient"],
        mint_intercept=np.asarray([source["intercept"]]),
        mint_layer=np.asarray([LAYER]),
        mint_feature_key=np.asarray([FEATURE_NAME]),
        additive_coef=np.zeros(140),
        additive_intercept=np.zeros(1),
    )
    loaded = load_linear_head(path)
    assert loaded["prefix"] == "mint"
    assert loaded["metadata"]["layer"] == LAYER
    assert loaded["metadata"]["feature_name"] == FEATURE_NAME

    wrong = tmp_path / "wrong_layer.npz"
    np.savez(
        wrong,
        mint_mean=source["mean"],
        mint_scale=source["scale"],
        mint_coef=source["coefficient"],
        mint_intercept=np.asarray([source["intercept"]]),
        mint_layer=np.asarray([33]),
        mint_feature_key=np.asarray(["mint_layer_33_chain_mean"]),
    )
    try:
        load_linear_head(wrong)
    except ValueError as error:
        assert "feature name" in str(error) or "layer-5" in str(error)
    else:
        raise AssertionError("a layer-33 head was accepted")


def test_early_stop_is_exact_and_never_calls_later_layers():
    model = _FakeMint()
    tokens = torch.arange(24, dtype=torch.long).reshape(2, 12)
    chain_ids = torch.zeros_like(tokens)
    ordinary = model(tokens, chain_ids, repr_layers=[LAYER])["representations"][LAYER]
    ordinary_counts = [layer.calls for layer in model.layers]
    assert ordinary_counts == [1] * 7

    with EarlyStopLayerExtractor(model, LAYER) as extractor:
        early = extractor.extract(tokens, chain_ids)
    assert torch.equal(early, ordinary)
    assert [layer.calls for layer in model.layers] == [2, 2, 2, 2, 2, 1, 1]


def test_record_batch_scores_without_retaining_layer5_features():
    collator = CollateFn(512)
    model = _FakeMint(residue_dimension=1280)
    zeros = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    head = {
        "mean": zeros,
        "scale": np.ones(FEATURE_DIMENSION, dtype=np.float64),
        "coefficient": zeros,
        "intercept": 0.0,
    }
    peptide_code = "AF"
    affibody_codes = ["AAAAA", "AAAAD"]
    peptide = "SLLAFITQV"
    chain1 = "A" * 261 + peptide
    chain2_values = []
    for code in affibody_codes:
        sequence = list("A" * 58)
        for position, residue in zip((5, 9, 12, 13, 16), code):
            sequence[position] = residue
        chain2_values.append("".join(sequence))
    values = {
        "pair_uid": [opaque_id("LibB", peptide_code, code) for code in affibody_codes],
        "library": ["LibB", "LibB"],
        "peptide_design_code": ["AF", "AF"],
        "peptide_full_sequence": [peptide, peptide],
        "affibody_design_code": affibody_codes,
        "peptide_uid": [opaque_id("LibB", "pep", peptide_code)] * 2,
        "affibody_uid": [opaque_id("LibB", "aff", code) for code in affibody_codes],
        "chain1_sha256": [hashlib.sha256(chain1.encode("ascii")).hexdigest()] * 2,
        "chain2_sha256": [
            hashlib.sha256(sequence.encode("ascii")).hexdigest()
            for sequence in chain2_values
        ],
        "chain1_smart_hla_linker_peptide_sequence": [chain1, chain1],
        "chain2_affibody_sequence": chain2_values,
    }
    batch = pa.RecordBatch.from_pydict({name: values[name] for name in INPUT_COLUMNS})
    with EarlyStopLayerExtractor(model, LAYER) as extractor:
        output = _score_record_batch(
            batch,
            7,
            model,
            extractor,
            head,
            torch.device("cpu"),
            collator,
        )
    assert output.column_names[-2:] == ["mint_layer5_logit", "mint_layer5_score"]
    assert output.column("input_row_index").to_pylist() == [7, 8]
    assert output.column("pair_uid").to_pylist() == values["pair_uid"]
    assert output.column("mint_layer5_logit").to_pylist() == [0.0, 0.0]
    assert output.column("mint_layer5_score").to_pylist() == [0.5, 0.5]
    assert [layer.calls for layer in model.layers] == [1, 1, 1, 1, 1, 0, 0]


def _candidate_values(contract, affibody_codes):
    peptide_code = "AF"
    peptide = "SLLAFITQV"
    chain1 = "A" * 261 + peptide
    chain2_values = []
    for code in affibody_codes:
        sequence = list("A" * 58)
        for position, residue in zip(contract.affibody_code_indices, code):
            sequence[position] = residue
        chain2_values.append("".join(sequence))
    values = {
        "pair_uid": [
            opaque_id(contract.library, peptide_code, code) for code in affibody_codes
        ],
        "library": [contract.library] * len(affibody_codes),
        "peptide_design_code": [peptide_code] * len(affibody_codes),
        "peptide_full_sequence": [peptide] * len(affibody_codes),
        "affibody_design_code": list(affibody_codes),
        "peptide_uid": [opaque_id(contract.library, "pep", peptide_code)]
        * len(affibody_codes),
        "affibody_uid": [
            opaque_id(contract.library, "aff", code) for code in affibody_codes
        ],
        "chain1_sha256": [hashlib.sha256(chain1.encode("ascii")).hexdigest()]
        * len(affibody_codes),
        "chain2_sha256": [
            hashlib.sha256(sequence.encode("ascii")).hexdigest()
            for sequence in chain2_values
        ],
        "chain1_smart_hla_linker_peptide_sequence": [chain1] * len(affibody_codes),
        "chain2_affibody_sequence": chain2_values,
    }
    if contract.library == "LibA":
        values.update(
            {
                "observed_in_any_raw_round": [True] * len(affibody_codes),
                "observed_in_r009_or_r010": [False] * len(affibody_codes),
                "affibody_identity_seen_in_strict_training": [True]
                * len(affibody_codes),
                "high_confidence_weak_negative": [False] * len(affibody_codes),
            }
        )
    return values


def test_liba_contract_uses_layer9_and_unshifted_displayed_positions():
    contract = resolve_scoring_contract("LibA")
    assert contract is LIBA_LAYER9_CONTRACT
    assert contract.layer == 9
    assert contract.evaluation_rows == 108
    assert contract.affibody_code_indices == (12, 16, 26, 30)
    assert contract.feature_name == "mint_layer_09_chain_mean"
    assert resolve_scoring_contract("LibB") is LIBB_LAYER5_CONTRACT
    layer33 = resolve_scoring_contract("LibA", 33)
    assert layer33 is LIBA_LAYER33_CONTRACT
    assert layer33.model_id == "mint_l33"
    assert layer33.feature_name == "mint_layer_33_chain_mean"
    assert layer33.affibody_code_indices == contract.affibody_code_indices
    assert contract.token_mapping["peptide_code_mint_token_indices"] == [265, 266]
    assert contract.token_mapping["affibody_code_mint_token_indices"] == [285, 289, 299, 303]

    values = _candidate_values(contract, ["AAAA", "AAAD"])
    _validate_candidate_values(values, contract=contract)
    shifted = dict(values)
    shifted["chain2_affibody_sequence"] = list(values["chain2_affibody_sequence"])
    sequence = list("A" * 58)
    for position, residue in zip((14, 18, 28, 32), "CCCD"):
        sequence[position] = residue
    shifted["affibody_design_code"] = ["AAAA", "CCCD"]
    shifted["chain2_affibody_sequence"][1] = "".join(sequence)
    shifted["chain2_sha256"] = list(values["chain2_sha256"])
    shifted["chain2_sha256"][1] = hashlib.sha256(
        shifted["chain2_affibody_sequence"][1].encode("ascii")
    ).hexdigest()
    shifted["affibody_uid"] = list(values["affibody_uid"])
    shifted["affibody_uid"][1] = opaque_id("LibA", "aff", "CCCD")
    shifted["pair_uid"] = list(values["pair_uid"])
    shifted["pair_uid"][1] = opaque_id("LibA", "AF", "CCCD")
    try:
        _validate_candidate_values(shifted, contract=contract)
    except ValueError as error:
        assert "displayed positions 13/17/27/31" in str(error)
    else:
        raise AssertionError("a +2-shifted LibA mutation code was accepted")


def test_liba_layer9_head_and_108_score_parity_fixture(tmp_path: Path):
    contract = LIBA_LAYER9_CONTRACT
    source = _head()
    head_path = tmp_path / "liba_head.npz"
    np.savez(
        head_path,
        schema_version=np.asarray([contract.head_schema_version]),
        mint_mean=source["mean"],
        mint_scale=source["scale"],
        mint_coef=source["coefficient"],
        mint_intercept=np.asarray([source["intercept"]]),
        mint_layer=np.asarray([9]),
        # The scorer accepts the head agent's explicit alias while preferring
        # the existing LibB-compatible ``mint_feature_key`` spelling.
        mint_feature_name=np.asarray(["mint_layer_09_chain_mean"]),
        mint_pooling=np.asarray([POOLING_DESCRIPTION]),
        training_membership_sha256=np.asarray(
            [contract.training_membership_sha256]
        ),
    )
    loaded = load_linear_head(head_path, contract=contract)
    assert loaded["metadata"]["layer"] == 9
    assert loaded["metadata"]["feature_name"] == contract.feature_name

    rng = np.random.RandomState(19)
    features = rng.normal(size=(108, FEATURE_DIMENSION)).astype(np.float32)
    _, expected = score_features(features, loaded)
    reference_path = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "pair_uid": ["row-{:03d}".format(index) for index in range(108)],
            "library": ["LibA"] * 108,
            "model": [contract.reference_model] * 108,
            "layer": [9] * 108,
            "score": expected,
        }
    ).to_csv(reference_path, index=False)
    observed = _load_reference_scores(
        reference_path, contract.reference_model, contract=contract
    )
    np.testing.assert_allclose(
        [observed["row-{:03d}".format(index)] for index in range(108)],
        expected,
        rtol=1e-14,
        atol=0.0,
    )

    leaked_path = tmp_path / "reference_with_outcomes.csv"
    leaked = pd.read_csv(reference_path)
    leaked["target_retention"] = np.linspace(0.0, 100.0, 108)
    leaked.to_csv(leaked_path, index=False)
    try:
        _load_reference_scores(
            leaked_path, contract.reference_model, contract=contract
        )
    except ValueError as error:
        assert "not target-free" in str(error)
    else:
        raise AssertionError("LibA parity loaded an outcome-bearing reference")

    locked_path, locked_model = _validate_reference_contract(
        contract, contract.reference_scores, contract.reference_model
    )
    assert locked_path == contract.reference_scores.resolve()
    assert locked_model == contract.reference_model
    try:
        _validate_reference_contract(
            contract, reference_path, contract.reference_model
        )
    except ValueError as error:
        assert "locked target-free" in str(error)
    else:
        raise AssertionError("LibA parity accepted an unregistered reference path")


def test_liba_early_stop_and_score_output_are_layer9_and_label_free():
    contract = LIBA_LAYER9_CONTRACT
    model = _FakeMint(residue_dimension=1280, layers=12)
    collator = CollateFn(512)
    zeros = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    head = {
        "mean": zeros,
        "scale": np.ones(FEATURE_DIMENSION, dtype=np.float64),
        "coefficient": zeros,
        "intercept": 0.0,
    }
    values = _candidate_values(contract, ["AAAA", "AAAD"])
    batch = pa.RecordBatch.from_pydict(values)
    with EarlyStopLayerExtractor(model, contract.layer) as extractor:
        output = _score_record_batch(
            batch,
            0,
            model,
            extractor,
            head,
            torch.device("cpu"),
            collator,
            contract=contract,
        )
    assert output.column("model_id").to_pylist() == ["mint_l9", "mint_l9"]
    assert output.column("model_score").to_pylist() == [0.5, 0.5]
    assert output.column("mint_layer9_logit").to_pylist() == [0.0, 0.0]
    assert "mint_layer9_score" not in output.column_names
    assert {
        "peptide_9mer_sequence",
        "provider_displayed_58aa_affibody_sequence",
        "model_input_affibody_sequence",
        "model_input_smart_hla_linker_peptide_sequence",
        "observed_in_any_raw_round",
        "observed_in_r009_or_r010",
        "affibody_identity_seen_in_strict_training",
        "high_confidence_weak_negative",
    }.issubset(output.column_names)
    assert not {
        "retention",
        "target_retention",
        "target_binder",
        "weak_label",
    }.intersection(output.column_names)
    assert _output_schema(contract).metadata[b"library"] == b"LibA"
    assert [layer.calls for layer in model.layers] == [1] * 9 + [0] * 3


def test_final_layer_capture_includes_norm_and_skips_language_model_head():
    model = _FakeMint(residue_dimension=4, layers=33)
    tokens = torch.arange(24, dtype=torch.long).reshape(2, 12)
    chain_ids = torch.zeros_like(tokens)
    ordinary = model(tokens, chain_ids, repr_layers=[33])["representations"][33]
    assert model.emb_layer_norm_after.calls == 1
    assert model.lm_head.calls == 1

    with EarlyStopLayerExtractor(model, 33) as extractor:
        early = extractor.extract(tokens, chain_ids)
    assert torch.equal(early, ordinary)
    assert model.emb_layer_norm_after.calls == 2
    assert model.lm_head.calls == 1
    assert [layer.calls for layer in model.layers] == [2] * 33


def test_dual_extractor_matches_ordinary_layer9_and_normalized_layer33_exactly():
    ordinary_model = _FakeMint(residue_dimension=4, layers=33)
    early_model = _FakeMint(residue_dimension=4, layers=33)
    tokens = torch.arange(24, dtype=torch.long).reshape(2, 12)
    chain_ids = torch.zeros_like(tokens)
    ordinary = ordinary_model(tokens, chain_ids, repr_layers=[9, 33])[
        "representations"
    ]
    with EarlyStopMultiLayerExtractor(early_model, (9, 33)) as extractor:
        captured = extractor.extract(tokens, chain_ids)
    assert torch.equal(captured[9], ordinary[9])
    assert torch.equal(captured[33], ordinary[33])
    assert early_model.lm_head.calls == 0


def test_dual_liba_batch_uses_one_forward_and_emits_two_locked_profiles():
    model = _FakeMint(residue_dimension=1280, layers=33)
    values = _candidate_values(LIBA_LAYER9_CONTRACT, ["AAAA", "AAAD"])
    batch = pa.RecordBatch.from_pydict(values)
    zeros = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    heads = {
        layer: {
            "mean": zeros,
            "scale": np.ones(FEATURE_DIMENSION, dtype=np.float64),
            "coefficient": zeros,
            "intercept": float(layer),
        }
        for layer in (9, 33)
    }
    with EarlyStopMultiLayerExtractor(model, (9, 33)) as extractor:
        outputs = _score_record_batch_dual_liba(
            batch,
            10,
            model,
            extractor,
            heads,
            torch.device("cpu"),
            CollateFn(512),
        )
    assert set(outputs) == {9, 33}
    assert outputs[9].column("model_id").to_pylist() == ["mint_l9"] * 2
    assert outputs[33].column("model_id").to_pylist() == ["mint_l33"] * 2
    assert outputs[9].column("input_row_index").to_pylist() == [10, 11]
    assert outputs[33].column("input_row_index").to_pylist() == [10, 11]
    assert outputs[9].column("mint_layer9_logit").to_pylist() == [9.0, 9.0]
    assert outputs[33].column("mint_layer33_logit").to_pylist() == [33.0, 33.0]
    assert outputs[9].column("pair_uid").to_pylist() == outputs[33].column(
        "pair_uid"
    ).to_pylist()
    assert [layer.calls for layer in model.layers] == [1] * 33
    assert model.emb_layer_norm_after.calls == 1
    assert model.lm_head.calls == 0


def test_candidate_input_with_retention_column_fails_closed():
    contract = LIBA_LAYER9_CONTRACT
    values = _candidate_values(contract, ["AAAA"])
    values["target_retention"] = [91.2]
    batch = pa.RecordBatch.from_pydict(values)
    model = _FakeMint(residue_dimension=1280, layers=10)
    zeros = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    head = {
        "mean": zeros,
        "scale": np.ones(FEATURE_DIMENSION, dtype=np.float64),
        "coefficient": zeros,
        "intercept": 0.0,
    }
    with EarlyStopLayerExtractor(model, contract.layer) as extractor:
        try:
            _score_record_batch(
                batch,
                0,
                model,
                extractor,
                head,
                torch.device("cpu"),
                CollateFn(512),
                contract=contract,
            )
        except ValueError as error:
            assert "forbidden retention/label" in str(error)
        else:
            raise AssertionError("a retention-bearing candidate input was accepted")


def test_liba_score_gate_rejects_a_receipt_for_another_contract(tmp_path: Path):
    contract = LIBA_LAYER9_CONTRACT
    source = _head()
    head_path = tmp_path / "head.npz"
    np.savez(
        head_path,
        schema_version=np.asarray([contract.head_schema_version]),
        mean=source["mean"],
        scale=source["scale"],
        coef=source["coefficient"],
        intercept=np.asarray([source["intercept"]]),
        layer=np.asarray([9]),
        feature_name=np.asarray([contract.feature_name]),
        training_membership_sha256=np.asarray(
            [contract.training_membership_sha256]
        ),
    )
    head = load_linear_head(head_path, contract=contract)
    checkpoint = tmp_path / "checkpoint.bin"
    config = tmp_path / "config.json"
    checkpoint.write_bytes(b"checkpoint fixture")
    config.write_text("{}\n")
    receipt = {
        "schema_version": contract.parity_schema_version,
        "passed": True,
        "contract": {
            "library": contract.library,
            "layer": contract.layer,
            "feature_name": contract.feature_name,
            "token_mapping": contract.token_mapping,
        },
        "rows": {"count": 108},
        "checks": {
            "ordinary_vs_early_features_bitwise_equal": True,
            "head_score_vs_reference_max_abs": 1e-8,
            "score_atol": 5e-5,
            "reference_model": contract.reference_model,
            "reference_score_column": contract.reference_score_column,
        },
        "inputs": {
            "script": {"sha256": sha256_file(
                Path(__file__).parents[1]
                / "downstream/AffibodyMHC/score_mint_layer5_candidates.py"
            )},
            "checkpoint": {"sha256": sha256_file(checkpoint)},
            "config": {"sha256": sha256_file(config)},
            "head_npz": {"sha256": head["sha256"]},
        },
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    _validate_parity_receipt(
        receipt_path, head, checkpoint, config, contract=contract
    )

    receipt["contract"]["library"] = "LibB"
    receipt_path.write_text(json.dumps(receipt))
    try:
        _validate_parity_receipt(
            receipt_path, head, checkpoint, config, contract=contract
        )
    except ValueError as error:
        assert "library does not match" in str(error)
    else:
        raise AssertionError("a LibB parity receipt authorized LibA scoring")
