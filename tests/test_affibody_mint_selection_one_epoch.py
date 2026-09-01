from pathlib import Path

import pytest

from downstream.AffibodyMHC import finetune_mint_selection_one_epoch as one_epoch
from downstream.AffibodyMHC import finetune_mint_selection_matched as matched


def _args(library="LibA"):
    required = [
        "--cache", "cache.npz",
        "--cache-rows", "rows.csv",
        "--cache-rows-manifest", "rows.json",
        "--retention-features", "retention.npz",
        "--reference-run-dir", "reference",
        "--checkpoint", "mint.ckpt",
        "--config", "config.json",
        "--output-dir", "private_data/experiments/output",
        "--library", library,
    ]
    return one_epoch.parse_args(required)


def test_prespecified_contract_is_single_seed_single_epoch_rank_two():
    args = _args()
    one_epoch.validate_one_epoch_contract(args)
    assert args.training_seed == 20260811
    assert args.epochs == 1
    assert args.lora_rank == 2
    assert one_epoch.ARM == "lora_cross"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("training_seed", 2, "seed"),
        ("epochs", 2, "exactly one"),
        ("lora_rank", 4, "rank-2"),
    ],
)
def test_contract_rejects_incomparable_overrides(field, value, message):
    args = _args()
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        one_epoch.validate_one_epoch_contract(args)


def test_canonical_table_contract_is_inherited_without_new_search():
    assert matched.PRIMARY_CONTRACT["LibA"]["rows"] == 22542
    assert matched.PRIMARY_CONTRACT["LibA"]["selected_c"] == 0.01
    assert matched.PRIMARY_CONTRACT["LibA"]["retention_rows"] == 108
    assert matched.PRIMARY_CONTRACT["LibB"]["rows"] == 30648
    assert matched.PRIMARY_CONTRACT["LibB"]["selected_c"] == 0.1
    assert matched.PRIMARY_CONTRACT["LibB"]["retention_rows"] == 119


def test_output_argument_remains_a_path_for_private_path_validation():
    assert isinstance(_args().output_dir, Path)
