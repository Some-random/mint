#!/usr/bin/env python
"""Pilot MINT fine-tuning for Affibody--pMHC retention regression.

The primary fine-tuning regime freezes the pretrained checkpoint and adds small
LoRA updates to the MINT-specific cross-chain attention projections in the last
two transformer layers.  Every run writes its exact split membership and is
restricted to the repository's Git-ignored private_data tree.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import code_only_baseline as code_baseline
from downstream.AffibodyMHC.extract_mint_features import PairCollator, SequencePairDataset
from mint.helpers.extract import MINTWrapper, load_config


LIBRARIES = ("LibA", "LibB")
REGIMES = ("head_only", "lora_cross", "last_multimer", "last_layer")
SCHEMES = ("random_pair", "blocked3")
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
EXPECTED_ROWS = {"LibA": 108, "LibB": 119}
SITE_COLUMNS = {
    "LibA": ("aff_p13", "aff_p17", "aff_p27", "aff_p31", "pep_p4", "pep_p5"),
    "LibB": (
        "aff_p6",
        "aff_p10",
        "aff_p13",
        "aff_p14",
        "aff_p17",
        "pep_p4",
        "pep_p5",
    ),
}
EXPECTED_TRAINABLE = {
    "head_only": 2561,
    "lora_cross": 23041,
    "last_multimer": 4921601,
    "last_layer": 24599041,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError("cannot serialize {}".format(type(value).__name__))


def write_json(path, payload):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def set_deterministic_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_library_frame(path, library):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    required = {
        "library",
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
        "peptide_design_code",
        "affibody_design_code",
        "measurement_missing",
        "target_retention",
        "target_binder",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
    }
    _require(required.issubset(frame.columns), "sequence-table schema mismatch")
    _require(len(frame) == 228, "expected 228 designed pairs")
    _require(frame["pair_uid"].nunique() == len(frame), "duplicate pair UID")
    _require(set(frame["library"]) == set(LIBRARIES), "unexpected library value")
    _require(int(frame["measurement_missing"].eq("0").sum()) == 227, "measured-row mismatch")

    frame = frame.loc[
        frame["library"].eq(library) & frame["measurement_missing"].eq("0")
    ].copy()
    _require(len(frame) == EXPECTED_ROWS[library], "{} measured-row mismatch".format(library))
    frame["target_retention"] = pd.to_numeric(frame["target_retention"], errors="raise")
    frame["target_binder"] = pd.to_numeric(frame["target_binder"], errors="raise").astype(int)
    _require(bool(np.isfinite(frame["target_retention"]).all()), "non-finite retention")
    _require(
        bool(frame["target_retention"].between(0.0, 100.0).all()),
        "retention outside [0,100]",
    )
    expected_binder = frame["target_retention"].ge(75.0).astype(int)
    _require(bool(expected_binder.eq(frame["target_binder"]).all()), "binder threshold mismatch")
    _require(
        bool(frame["chain1_smart_hla_linker_peptide_sequence"].map(len).eq(270).all()),
        "chain-1 length mismatch",
    )
    _require(
        bool(frame["chain2_affibody_sequence"].map(len).eq(58).all()),
        "chain-2 length mismatch",
    )
    return frame.reset_index(drop=True)


def _membership_digest(values):
    joined = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _hash_order(values, seed, library, axis):
    def key(value):
        payload = "{}|blocked3|{}|{}|{}".format(seed, library, axis, value)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return sorted(set(values), key=lambda value: (key(value), value))


def _stratified_holdout(frame, candidate_indices, test_fraction, seed):
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    labels = frame.iloc[candidate_indices]["target_binder"].to_numpy(dtype=int)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(test_fraction),
        random_state=int(seed),
    )
    train_local, test_local = next(splitter.split(candidate_indices, labels))
    return candidate_indices[train_local], candidate_indices[test_local]


def make_split(frame, library, scheme, fold, seed):
    """Return deterministic outer membership plus a development validation split."""
    _require(scheme in SCHEMES, "unknown split scheme")
    roles = np.full(len(frame), "", dtype=object)
    details = {
        "library": library,
        "scheme": scheme,
        "fold": int(fold),
        "seed": int(seed),
    }

    if scheme == "random_pair":
        _require(0 <= int(fold) < 5, "random-pair fold must be in [0,4]")
        ordered = frame.sort_values(
            ["peptide_design_code", "affibody_design_code"]
        ).reset_index()
        split_frame = ordered.drop(columns=["index"])
        outer = code_baseline.make_outer_splits(
            split_frame, "random_pair", seed=int(seed), random_repeats=1
        )
        by_fold = {int(item["outer_fold"].rsplit("fold", 1)[1]): item for item in outer}
        test_original = ordered.iloc[by_fold[int(fold)]["test"]]["index"].to_numpy(dtype=int)
        validation_fold = (int(fold) + 1) % 5
        validation_original = ordered.iloc[by_fold[validation_fold]["test"]][
            "index"
        ].to_numpy(dtype=int)
        all_indices = np.arange(len(frame), dtype=int)
        train_original = np.setdiff1d(
            all_indices,
            np.concatenate((test_original, validation_original)),
            assume_unique=True,
        )
        roles[train_original] = "train"
        roles[validation_original] = "validation"
        roles[test_original] = "test"
        details["validation_outer_fold"] = int(validation_fold)
        details["outer_group_selection_uses_targets"] = True
        details["random_fold_stratification_uses_binder"] = True
    else:
        _require(0 <= int(fold) < 3, "blocked3 fold must be in [0,2]")
        peptide_order = _hash_order(frame["peptide_uid"], seed, library, "peptide_uid")
        affibody_order = _hash_order(frame["affibody_uid"], seed, library, "affibody_uid")
        peptide_assignment = {value: index % 3 for index, value in enumerate(peptide_order)}
        affibody_assignment = {value: index % 3 for index, value in enumerate(affibody_order)}
        held_peptides = {
            value for value, assignment in peptide_assignment.items() if assignment == int(fold)
        }
        held_affibodies = {
            value for value, assignment in affibody_assignment.items() if assignment == int(fold)
        }
        peptide_held = frame["peptide_uid"].isin(held_peptides).to_numpy()
        affibody_held = frame["affibody_uid"].isin(held_affibodies).to_numpy()
        test_indices = np.flatnonzero(peptide_held & affibody_held)
        guarded_indices = np.flatnonzero(peptide_held ^ affibody_held)
        development_indices = np.flatnonzero(~peptide_held & ~affibody_held)
        train_indices, validation_indices = _stratified_holdout(
            frame,
            development_indices,
            test_fraction=0.20,
            seed=int(seed) + 100 + int(fold),
        )
        roles[train_indices] = "train"
        roles[validation_indices] = "validation"
        roles[test_indices] = "test"
        roles[guarded_indices] = "guarded"
        details.update(
            {
                "blocked_folds": 3,
                "held_peptide_uids": sorted(held_peptides),
                "held_affibody_uids": sorted(held_affibodies),
                "held_peptide_codes": sorted(
                    frame.loc[frame["peptide_uid"].isin(held_peptides), "peptide_design_code"].unique()
                ),
                "held_affibody_codes": sorted(
                    frame.loc[
                        frame["affibody_uid"].isin(held_affibodies),
                        "affibody_design_code",
                    ].unique()
                ),
                "outer_group_selection_uses_targets": False,
                "development_validation_stratification_uses_binder": True,
            }
        )

    _require(not bool(pd.Series(roles).eq("").any()), "unassigned split row")
    train = np.flatnonzero(roles == "train")
    validation = np.flatnonzero(roles == "validation")
    test = np.flatnonzero(roles == "test")
    guarded = np.flatnonzero(roles == "guarded")
    _require(len(train) > 0 and len(validation) > 0 and len(test) > 0, "empty split role")
    _require(frame.iloc[train]["target_binder"].nunique() == 2, "one-class training split")
    _require(frame.iloc[validation]["target_binder"].nunique() == 2, "one-class validation split")

    if scheme == "blocked3":
        dev = frame.iloc[np.concatenate((train, validation))]
        held = frame.iloc[test]
        _require(
            set(dev["peptide_uid"]).isdisjoint(set(held["peptide_uid"])),
            "blocked split leaks peptide identity",
        )
        _require(
            set(dev["affibody_uid"]).isdisjoint(set(held["affibody_uid"])),
            "blocked split leaks Affibody identity",
        )

    details["counts"] = {
        "train": int(len(train)),
        "validation": int(len(validation)),
        "test": int(len(test)),
        "guarded": int(len(guarded)),
    }
    details["class_counts"] = {}
    details["membership_sha256"] = {}
    for role in ("train", "validation", "test", "guarded"):
        indices = np.flatnonzero(roles == role)
        labels = frame.iloc[indices]["target_binder"] if len(indices) else pd.Series(dtype=int)
        details["class_counts"][role] = {
            "negative": int(labels.eq(0).sum()),
            "positive": int(labels.eq(1).sum()),
        }
        details["membership_sha256"][role] = _membership_digest(
            frame.iloc[indices]["pair_uid"].tolist()
        )
    return roles, details


def load_feature_matrix(path, frame, feature_name="mint_chain_mean"):
    with np.load(str(path), allow_pickle=False) as archive:
        _require("pair_uid" in archive.files, "feature archive lacks pair_uid")
        _require(feature_name in archive.files, "feature archive lacks {}".format(feature_name))
        uids = archive["pair_uid"].astype(str)
        values = np.asarray(archive[feature_name], dtype=np.float32)
    _require(len(set(uids.tolist())) == len(uids), "duplicate feature pair UID")
    _require(values.ndim == 2 and values.shape[1] == 2560, "feature shape mismatch")
    _require(bool(np.isfinite(values).all()), "non-finite frozen feature")
    uid_to_index = {uid: index for index, uid in enumerate(uids)}
    missing = set(frame["pair_uid"]) - set(uid_to_index)
    _require(not missing, "feature archive is missing source UIDs")
    ordered = np.stack([values[uid_to_index[uid]] for uid in frame["pair_uid"]], axis=0)
    return ordered


def standardization(train_values):
    mean = np.asarray(train_values, dtype=np.float64).mean(axis=0)
    scale = np.asarray(train_values, dtype=np.float64).std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return mean, scale


def tune_dense_ridge(features, targets, train, validation, fixed_alpha=None):
    x_mean, x_scale = standardization(features[train])
    y_mean = float(np.mean(targets[train]))
    y_scale = float(np.std(targets[train], ddof=0))
    _require(y_scale > 0.0, "zero training-target variance")
    x_train = (features[train].astype(np.float64) - x_mean) / x_scale
    x_validation = (features[validation].astype(np.float64) - x_mean) / x_scale
    y_train = (targets[train] - y_mean) / y_scale
    records = []
    candidates = []
    for alpha in ALPHA_GRID:
        ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
        ridge.fit(x_train, y_train)
        prediction = ridge.predict(x_validation) * y_scale + y_mean
        mae = float(mean_absolute_error(targets[validation], prediction))
        records.append({"alpha": float(alpha), "validation_mae": mae})
        candidates.append((mae, -float(alpha), ridge))
    if fixed_alpha is None:
        _, neg_alpha, chosen = min(candidates, key=lambda item: (item[0], item[1]))
        chosen_alpha = float(-neg_alpha)
        selection = "minimum validation MAE; exact ties favor stronger regularization"
    else:
        _require(float(fixed_alpha) in ALPHA_GRID, "fixed ridge alpha is outside the grid")
        chosen_alpha = float(fixed_alpha)
        chosen = Ridge(alpha=chosen_alpha, fit_intercept=True, solver="svd")
        chosen.fit(x_train, y_train)
        selection = "fixed before training"
    return {
        "model": chosen,
        "alpha": chosen_alpha,
        "selection": selection,
        "grid": records,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def tune_site_ridge(frame, library, targets, train, validation, seed):
    del seed
    columns = SITE_COLUMNS[library]
    _require(set(columns).issubset(frame.columns), "site-residue columns are missing")
    matrix = frame.loc[:, list(columns)].to_numpy(dtype=str)
    records = []
    candidates = []
    for alpha in ALPHA_GRID:
        encoder = OneHotEncoder(
            categories=[list(code_baseline.AA_ALPHABET) for _ in columns],
            handle_unknown="error",
            sparse=False,
            dtype=np.float64,
        )
        estimator = Pipeline(
            [
                ("onehot", encoder),
                (
                    "head",
                    Ridge(alpha=float(alpha), fit_intercept=True, solver="svd"),
                ),
            ]
        )
        estimator.fit(matrix[train], targets[train])
        prediction = estimator.predict(matrix[validation])
        mae = float(mean_absolute_error(targets[validation], prediction))
        records.append({"alpha": float(alpha), "validation_mae": mae})
        candidates.append((mae, -float(alpha), estimator))
    _, neg_alpha, chosen = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "model": chosen,
        "matrix": matrix,
        "alpha": float(-neg_alpha),
        "grid": records,
    }


class LoRALinear(nn.Module):
    def __init__(self, base, rank=2, alpha=4.0, dropout=0.05):
        super().__init__()
        _require(isinstance(base, nn.Linear), "LoRA base must be a Linear module")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))
        self.base.requires_grad_(False)

    def forward(self, value):
        base = self.base(value)
        adapted = F.linear(F.linear(self.dropout(value), self.lora_a), self.lora_b)
        return base + self.scaling * adapted


class MINTRegressor(nn.Module):
    def __init__(
        self,
        config_path,
        checkpoint_path,
        device,
        regime,
        x_mean,
        x_scale,
        ridge_model,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.05,
    ):
        super().__init__()
        _require(regime != "head_only", "head-only does not need a live MINT model")
        self.regime = regime
        self.wrapper = MINTWrapper(
            load_config(str(config_path)),
            str(checkpoint_path),
            freeze_percent=1.0,
            use_multimer=True,
            sep_chains=True,
            device=str(device),
        )
        self.wrapper.model.requires_grad_(False)
        self.adapter_modules = []
        if regime == "lora_cross":
            for layer_index in (31, 32):
                attention = self.wrapper.model.layers[layer_index].multimer_attn
                for projection_name in ("q_proj", "v_proj"):
                    module = LoRALinear(
                        getattr(attention, projection_name),
                        rank=lora_rank,
                        alpha=lora_alpha,
                        dropout=lora_dropout,
                    )
                    setattr(attention, projection_name, module)
                    self.adapter_modules.append(module)
        elif regime == "last_multimer":
            self.wrapper.model.layers[-1].multimer_attn.requires_grad_(True)
        elif regime == "last_layer":
            self.wrapper.model.layers[-1].requires_grad_(True)
        else:
            raise ValueError("unsupported live-MINT regime {}".format(regime))

        self.head = nn.Linear(2560, 1)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.from_numpy(np.asarray(ridge_model.coef_, dtype=np.float32)).reshape(1, -1)
            )
            self.head.bias.copy_(
                torch.tensor([float(ridge_model.intercept_)], dtype=torch.float32)
            )
        self.register_buffer("feature_mean", torch.from_numpy(x_mean.astype(np.float32)))
        self.register_buffer("feature_scale", torch.from_numpy(x_scale.astype(np.float32)))

    def forward(self, chains, chain_ids):
        feature = self.wrapper(chains, chain_ids)
        feature = (feature - self.feature_mean) / self.feature_scale
        return self.head(feature).flatten()

    def set_train_mode(self):
        self.wrapper.model.eval()
        self.head.train()
        for module in self.adapter_modules:
            module.train()

    def adapter_parameters(self):
        head_ids = {id(parameter) for parameter in self.head.parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]


class FrozenHeadRegressor(nn.Module):
    def __init__(self, x_mean, x_scale, ridge_model):
        super().__init__()
        self.head = nn.Linear(2560, 1)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.from_numpy(np.asarray(ridge_model.coef_, dtype=np.float32)).reshape(1, -1)
            )
            self.head.bias.copy_(
                torch.tensor([float(ridge_model.intercept_)], dtype=torch.float32)
            )
        self.register_buffer("feature_mean", torch.from_numpy(x_mean.astype(np.float32)))
        self.register_buffer("feature_scale", torch.from_numpy(x_scale.astype(np.float32)))

    def forward(self, feature):
        return self.head((feature - self.feature_mean) / self.feature_scale).flatten()

    def set_train_mode(self):
        self.train()

    def adapter_parameters(self):
        return []


def trainable_parameter_audit(model, regime):
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    _require(count == EXPECTED_TRAINABLE[regime], "unexpected trainable parameter count {}".format(count))
    if regime == "lora_cross":
        encoder_names = [name for name in names if not name.startswith("head.")]
        _require(len(encoder_names) == 8, "unexpected LoRA tensor count")
        _require(all("lora_" in name for name in encoder_names), "non-LoRA backbone tensor is trainable")
    return names, count


def optimizer_for(model, args):
    groups = [
        {
            "params": [model.head.weight],
            "lr": float(args.head_lr),
            "base_lr": float(args.head_lr),
            "weight_decay": float(args.weight_decay),
        },
        {
            "params": [model.head.bias],
            "lr": float(args.head_lr),
            "base_lr": float(args.head_lr),
            "weight_decay": 0.0,
        },
    ]
    adapter = model.adapter_parameters()
    if adapter:
        decay = []
        no_decay = []
        name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
        for parameter in adapter:
            name = name_by_id[id(parameter)]
            if name.endswith("bias") or "layer_norm" in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        if decay:
            groups.append(
                {
                    "params": decay,
                    "lr": float(args.adapter_lr),
                    "base_lr": float(args.adapter_lr),
                    "weight_decay": float(args.weight_decay),
                }
            )
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr": float(args.adapter_lr),
                    "base_lr": float(args.adapter_lr),
                    "weight_decay": 0.0,
                }
            )
    return torch.optim.AdamW(
        groups,
        betas=(0.9, 0.98),
        eps=1e-8,
    )


def learning_rate_multiplier(step, total_steps, warmup_fraction):
    warmup_steps = max(1, int(math.ceil(total_steps * float(warmup_fraction))))
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denominator = max(1, total_steps - warmup_steps)
    progress = min(1.0, float(step - warmup_steps) / float(denominator))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def update_learning_rates(optimizer, step, total_steps, warmup_fraction):
    multiplier = learning_rate_multiplier(step, total_steps, warmup_fraction)
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * multiplier
    return multiplier


def state_for_best(model, regime):
    payload = {"head": copy.deepcopy(model.head.state_dict())}
    if regime != "head_only":
        payload["adapter"] = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("head.")
        }
    return payload


def restore_best(model, payload):
    model.head.load_state_dict(payload["head"])
    if "adapter" in payload:
        named = dict(model.named_parameters())
        _require(set(named).issuperset(payload["adapter"]), "saved adapter name mismatch")
        with torch.no_grad():
            for name, value in payload["adapter"].items():
                named[name].copy_(value.to(named[name].device))


def make_sequence_loader(frame, indices, batch_size, shuffle, seed):
    subset = frame.iloc[np.asarray(indices, dtype=int)].copy().reset_index(drop=True)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        SequencePairDataset(subset),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=PairCollator(),
        num_workers=0,
        generator=generator,
    )


def predict_head(model, features, indices, device, batch_size):
    model.eval()
    predictions = []
    indices = np.asarray(indices, dtype=int)
    with torch.no_grad():
        for start in range(0, len(indices), int(batch_size)):
            batch = torch.from_numpy(features[indices[start : start + batch_size]]).to(device)
            predictions.append(model(batch).detach().cpu().numpy())
    return np.concatenate(predictions).astype(float)


def predict_live(model, loader, device, y_mean, y_scale, use_amp):
    model.eval()
    predictions = []
    observed_uids = []
    with torch.no_grad():
        for chains, chain_ids, _, pair_uids in loader:
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            with torch.cuda.amp.autocast(enabled=bool(use_amp), dtype=torch.float16):
                prediction_z = model(chains, chain_ids)
            predictions.append(prediction_z.float().cpu().numpy() * y_scale + y_mean)
            observed_uids.extend(pair_uids)
    return np.concatenate(predictions).astype(float), observed_uids


def validation_mae_head(model, features, targets, indices, device, batch_size, y_mean, y_scale):
    prediction_z = predict_head(model, features, indices, device, batch_size)
    prediction = prediction_z * y_scale + y_mean
    return float(mean_absolute_error(targets[indices], prediction))


def validation_mae_live(model, loader, targets_by_uid, device, y_mean, y_scale, use_amp):
    prediction, uids = predict_live(model, loader, device, y_mean, y_scale, use_amp)
    target = np.asarray([targets_by_uid[uid] for uid in uids], dtype=float)
    return float(mean_absolute_error(target, prediction))


def train_head_only(model, features, targets, train, validation, args, device, y_mean, y_scale):
    optimizer = optimizer_for(model, args)
    generator = torch.Generator()
    generator.manual_seed(int(args.training_seed))
    train_loader = DataLoader(
        np.asarray(train, dtype=int).tolist(),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    batches_per_epoch = len(train_loader)
    updates_per_epoch = int(math.ceil(float(batches_per_epoch) / float(args.accumulation_steps)))
    total_steps = max(1, updates_per_epoch * int(args.max_epochs))
    best_mae = validation_mae_head(
        model, features, targets, validation, device, args.eval_batch_size, y_mean, y_scale
    )
    best_state = state_for_best(model, "head_only")
    best_epoch = 0
    patience = 0
    global_step = 0
    history = [{"epoch": 0, "train_loss": None, "validation_mae": best_mae}]
    loss_fn = nn.SmoothL1Loss(beta=float(args.huber_beta), reduction="sum")
    target_z = ((targets - y_mean) / y_scale).astype(np.float32)

    for epoch in range(1, int(args.max_epochs) + 1):
        model.set_train_mode()
        optimizer.zero_grad()
        loss_total = 0.0
        examples = 0
        window_examples = 0
        for batch_number, batch_indices in enumerate(train_loader):
            indices = batch_indices.numpy().astype(int, copy=False)
            x = torch.from_numpy(features[indices]).to(device)
            y = torch.from_numpy(target_z[indices]).to(device)
            prediction = model(x)
            raw_loss = loss_fn(prediction.float(), y.float())
            raw_loss.backward()
            loss_total += float(raw_loss.detach().cpu())
            examples += len(indices)
            window_examples += len(indices)
            is_tail = batch_number + 1 == len(train_loader)
            if (batch_number + 1) % int(args.accumulation_steps) == 0 or is_tail:
                update_learning_rates(optimizer, global_step, total_steps, args.warmup_fraction)
                for parameter in model.parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        parameter.grad.div_(float(window_examples))
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(args.clip_norm),
                )
                optimizer.step()
                optimizer.zero_grad()
                window_examples = 0
                global_step += 1
        validation_mae = validation_mae_head(
            model, features, targets, validation, device, args.eval_batch_size, y_mean, y_scale
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(loss_total / max(1, examples)),
                "validation_mae": validation_mae,
            }
        )
        print(
            "epoch {:02d} train_loss {:.6f} validation_mae {:.4f}".format(
                epoch, history[-1]["train_loss"], validation_mae
            ),
            flush=True,
        )
        if validation_mae < best_mae - float(args.min_delta):
            best_mae = validation_mae
            best_epoch = epoch
            best_state = state_for_best(model, "head_only")
            patience = 0
        else:
            patience += 1
        if patience >= int(args.patience):
            break
    restore_best(model, best_state)
    return history, best_state, best_epoch, best_mae


def train_live(
    model,
    frame,
    targets,
    train,
    validation,
    args,
    device,
    y_mean,
    y_scale,
):
    train_loader = make_sequence_loader(
        frame, train, args.batch_size, shuffle=True, seed=args.training_seed
    )
    validation_loader = make_sequence_loader(
        frame, validation, args.eval_batch_size, shuffle=False, seed=args.training_seed
    )
    targets_by_uid = dict(zip(frame["pair_uid"], targets))
    optimizer = optimizer_for(model, args)
    batches_per_epoch = len(train_loader)
    updates_per_epoch = int(math.ceil(float(batches_per_epoch) / float(args.accumulation_steps)))
    total_steps = max(1, updates_per_epoch * int(args.max_epochs))
    use_amp = bool(args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_mae = validation_mae_live(
        model, validation_loader, targets_by_uid, device, y_mean, y_scale, use_amp
    )
    best_state = state_for_best(model, args.regime)
    best_epoch = 0
    patience = 0
    global_step = 0
    history = [{"epoch": 0, "train_loss": None, "validation_mae": best_mae}]
    loss_fn = nn.SmoothL1Loss(beta=float(args.huber_beta), reduction="sum")
    target_z_by_uid = {
        uid: float((target - y_mean) / y_scale) for uid, target in targets_by_uid.items()
    }

    for epoch in range(1, int(args.max_epochs) + 1):
        model.set_train_mode()
        optimizer.zero_grad()
        loss_total = 0.0
        examples = 0
        window_examples = 0
        for batch_number, (chains, chain_ids, _, pair_uids) in enumerate(train_loader):
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            target = torch.tensor(
                [target_z_by_uid[uid] for uid in pair_uids], dtype=torch.float32, device=device
            )
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                prediction = model(chains, chain_ids)
                raw_loss = loss_fn(prediction.float(), target.float())
            scaler.scale(raw_loss).backward()
            loss_total += float(raw_loss.detach().cpu())
            examples += len(pair_uids)
            window_examples += len(pair_uids)
            is_tail = batch_number + 1 == len(train_loader)
            if (batch_number + 1) % int(args.accumulation_steps) == 0 or is_tail:
                update_learning_rates(optimizer, global_step, total_steps, args.warmup_fraction)
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        parameter.grad.div_(float(window_examples))
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(args.clip_norm),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                window_examples = 0
                global_step += 1
        validation_mae = validation_mae_live(
            model, validation_loader, targets_by_uid, device, y_mean, y_scale, use_amp
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(loss_total / max(1, examples)),
                "validation_mae": validation_mae,
            }
        )
        print(
            "epoch {:02d} train_loss {:.6f} validation_mae {:.4f}".format(
                epoch, history[-1]["train_loss"], validation_mae
            ),
            flush=True,
        )
        if validation_mae < best_mae - float(args.min_delta):
            best_mae = validation_mae
            best_epoch = epoch
            best_state = state_for_best(model, args.regime)
            patience = 0
        else:
            patience += 1
        if patience >= int(args.patience):
            break
    restore_best(model, best_state)
    return history, best_state, best_epoch, best_mae


def safe_correlation(function, target, prediction):
    if len(target) < 2 or np.ptp(target) <= 1e-12 or np.ptp(prediction) <= 1e-12:
        return None
    value = function(target, prediction)[0]
    return float(value) if np.isfinite(value) else None


def regression_metrics(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    clipped = np.clip(prediction, 0.0, 100.0)
    return {
        "n": int(len(target)),
        "mae": float(mean_absolute_error(target, prediction)),
        "median_absolute_error": float(np.median(np.abs(target - prediction))),
        "rmse": float(math.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)) if len(target) >= 2 else None,
        "mean_signed_error": float(np.mean(prediction - target)),
        "pearson": safe_correlation(scipy.stats.pearsonr, target, prediction),
        "spearman": safe_correlation(scipy.stats.spearmanr, target, prediction),
        "clipped_mae": float(mean_absolute_error(target, clipped)),
        "predictions_below_0": int(np.sum(prediction < 0.0)),
        "predictions_above_100": int(np.sum(prediction > 100.0)),
    }


def live_feature_parity(model, frame, features, indices, device, tolerance=2e-4):
    sample = np.asarray(indices[: min(2, len(indices))], dtype=int)
    loader = make_sequence_loader(frame, sample, len(sample), shuffle=False, seed=0)
    model.eval()
    with torch.no_grad():
        chains, chain_ids, _, uids = next(iter(loader))
        observed = model.wrapper(chains.to(device), chain_ids.to(device)).float().cpu().numpy()
    expected = features[sample]
    maximum = float(np.max(np.abs(observed - expected)))
    _require(maximum <= float(tolerance), "live/cached MINT feature mismatch: {}".format(maximum))
    _require(uids == frame.iloc[sample]["pair_uid"].tolist(), "parity UID order mismatch")
    return maximum


def prediction_rows(frame, test, targets, predictions_by_model):
    rows = []
    test_frame = frame.iloc[test]
    for model_name, prediction in predictions_by_model.items():
        _require(len(prediction) == len(test), "test prediction length mismatch")
        for row_offset, (_, source) in enumerate(test_frame.iterrows()):
            rows.append(
                {
                    "library": source["library"],
                    "pair_uid": source["pair_uid"],
                    "peptide_uid": source["peptide_uid"],
                    "affibody_uid": source["affibody_uid"],
                    "peptide_design_code": source["peptide_design_code"],
                    "affibody_design_code": source["affibody_design_code"],
                    "model": model_name,
                    "y_true": float(targets[test[row_offset]]),
                    "y_score": float(prediction[row_offset]),
                    "absolute_error": float(
                        abs(targets[test[row_offset]] - prediction[row_offset])
                    ),
                }
            )
    return rows


def summary_markdown(args, split_details, best_epoch, metrics, elapsed, trainable_count):
    lines = [
        "# MINT retention fine-tuning pilot",
        "",
        "- Library: `{}`".format(args.library),
        "- Split: `{}` fold `{}`".format(args.scheme, args.fold),
        "- Regime: `{}`".format(args.regime),
        "- Trainable parameters: `{:,.0f}`".format(trainable_count),
        "- Rows: train `{}`, validation `{}`, test `{}`, guarded `{}`".format(
            split_details["counts"]["train"],
            split_details["counts"]["validation"],
            split_details["counts"]["test"],
            split_details["counts"]["guarded"],
        ),
        "- Selected epoch: `{}`".format(best_epoch),
        "- Elapsed seconds: `{:.1f}`".format(elapsed),
        "",
        "| Model | Test MAE | Test RMSE | Clipped MAE |",
        "|---|---:|---:|---:|",
    ]
    order = ["mean_prior", "site_residue_ridge", "frozen_mint_ridge"]
    if "frozen_mint_ridge_live" in metrics:
        order.append("frozen_mint_ridge_live")
    order.append(args.regime)
    for model_name in order:
        item = metrics[model_name]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} |".format(
                model_name, item["mae"], item["rmse"], item["clipped_mae"]
            )
        )
    lines.append("")
    lines.append("This is a screening result from a small, library-specific grid.")
    if args.scheme == "blocked3":
        lines.extend(
            [
                "The blocked3 folds hold out both partner identities but cover only the",
                "three diagonal hash blocks, not every possible double-cold cell.",
            ]
        )
    else:
        lines.append("Random-pair evaluation is an interpolation check, not an unseen-partner test.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=REPO_ROOT / "private_data/derived/retention_sequences_v2.csv"
    )
    parser.add_argument(
        "--features", type=Path, default=REPO_ROOT / "private_data/derived/mint_features_v1.npz"
    )
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/mint.ckpt")
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "data/esm2_t33_650M_UR50D.json"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--library", choices=LIBRARIES, required=True)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--training-seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.1)
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--adapter-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        help="Fixed frozen-head Ridge alpha; recommended as 1000 for blocked pilot splits.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Opt in to FP16; FP32 is the audited default because feature scaling amplifies rounding.",
    )
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    _require(args.batch_size >= 1, "batch size must be positive")
    _require(args.eval_batch_size >= 1, "evaluation batch size must be positive")
    _require(args.accumulation_steps >= 1, "accumulation steps must be positive")
    _require(args.max_epochs >= 1, "max epochs must be positive")
    _require(args.patience >= 1, "patience must be positive")
    _require(not args.amp, "FP16 AMP is disabled for this pilot; use the audited FP32 path")
    if args.scheme == "blocked3":
        _require(
            args.ridge_alpha is not None,
            "blocked3 requires a pre-fixed --ridge-alpha (the pilot protocol uses 1000)",
        )
    for path, label in (
        (args.input, "input"),
        (args.features, "features"),
        (args.checkpoint, "checkpoint"),
        (args.config, "config"),
    ):
        _require(path.is_file(), "{} file does not exist".format(label))
    script_path = Path(__file__).resolve()
    dependency_path = Path(code_baseline.__file__).resolve()
    immutable_paths = {
        "script": script_path,
        "split_dependency": dependency_path,
        "input": args.input,
        "features": args.features,
        "checkpoint": args.checkpoint,
        "config": args.config,
    }
    initial_hashes = {
        name: code_baseline.sha256_file(path) for name, path in immutable_paths.items()
    }
    output_dir = code_baseline.validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    set_deterministic_seed(args.training_seed)

    frame = load_library_frame(args.input, args.library)
    roles, split_details = make_split(frame, args.library, args.scheme, args.fold, args.seed)
    train = np.flatnonzero(roles == "train")
    validation = np.flatnonzero(roles == "validation")
    test = np.flatnonzero(roles == "test")
    features = load_feature_matrix(args.features, frame)
    targets = frame["target_retention"].to_numpy(dtype=float)

    dense = tune_dense_ridge(
        features, targets, train, validation, fixed_alpha=args.ridge_alpha
    )
    site = tune_site_ridge(frame, args.library, targets, train, validation, args.seed)
    frozen_prediction = (
        dense["model"].predict(
            (features[test].astype(np.float64) - dense["x_mean"]) / dense["x_scale"]
        )
        * dense["y_scale"]
        + dense["y_mean"]
    )
    site_prediction = site["model"].predict(site["matrix"][test])
    mean_prediction = np.full(len(test), float(np.mean(targets[train])), dtype=float)

    device = torch.device("cpu" if args.regime == "head_only" else args.device)
    if args.regime != "head_only":
        _require(torch.cuda.is_available(), "CUDA is required for live MINT fine-tuning")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = MINTRegressor(
            args.config,
            args.checkpoint,
            device,
            args.regime,
            dense["x_mean"],
            dense["x_scale"],
            dense["model"],
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        ).to(device)
        trainable_names, trainable_count = trainable_parameter_audit(model, args.regime)
        parity_error = live_feature_parity(model, frame, features, train, device)
        initial_state = state_for_best(model, args.regime)
        validation_loader = make_sequence_loader(
            frame,
            validation,
            args.eval_batch_size,
            shuffle=False,
            seed=args.training_seed,
        )
        initial_validation_prediction, initial_validation_uids = predict_live(
            model,
            validation_loader,
            device,
            dense["y_mean"],
            dense["y_scale"],
            use_amp=False,
        )
        _require(
            initial_validation_uids == frame.iloc[validation]["pair_uid"].tolist(),
            "initial validation UID order mismatch",
        )
        cached_validation_prediction = (
            dense["model"].predict(
                (features[validation].astype(np.float64) - dense["x_mean"])
                / dense["x_scale"]
            )
            * dense["y_scale"]
            + dense["y_mean"]
        )
        initial_prediction_error = float(
            np.max(np.abs(initial_validation_prediction - cached_validation_prediction))
        )
        _require(
            initial_prediction_error <= 0.1,
            "live/cached epoch-0 prediction mismatch: {}".format(initial_prediction_error),
        )
        history, best_state, best_epoch, best_validation_mae = train_live(
            model,
            frame,
            targets,
            train,
            validation,
            args,
            device,
            dense["y_mean"],
            dense["y_scale"],
        )
        selected_state = state_for_best(model, args.regime)
        test_loader = make_sequence_loader(
            frame, test, args.eval_batch_size, shuffle=False, seed=args.training_seed
        )
        restore_best(model, initial_state)
        frozen_live_prediction, frozen_live_uids = predict_live(
            model,
            test_loader,
            device,
            dense["y_mean"],
            dense["y_scale"],
            use_amp=False,
        )
        _require(
            frozen_live_uids == frame.iloc[test]["pair_uid"].tolist(),
            "frozen-live test UID order mismatch",
        )
        restore_best(model, selected_state)
        neural_prediction, test_uids = predict_live(
            model,
            test_loader,
            device,
            dense["y_mean"],
            dense["y_scale"],
            bool(args.amp),
        )
        _require(test_uids == frame.iloc[test]["pair_uid"].tolist(), "test UID order mismatch")
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        gpu_name = torch.cuda.get_device_name(device)
    else:
        model = FrozenHeadRegressor(
            dense["x_mean"], dense["x_scale"], dense["model"]
        ).to(device)
        trainable_names, trainable_count = trainable_parameter_audit(model, args.regime)
        parity_error = 0.0
        initial_prediction_error = 0.0
        frozen_live_prediction = None
        history, best_state, best_epoch, best_validation_mae = train_head_only(
            model,
            features,
            targets,
            train,
            validation,
            args,
            device,
            dense["y_mean"],
            dense["y_scale"],
        )
        neural_prediction_z = predict_head(
            model, features, test, device, args.eval_batch_size
        )
        neural_prediction = neural_prediction_z * dense["y_scale"] + dense["y_mean"]
        peak_allocated = 0
        gpu_name = None

    predictions_by_model = {
        "mean_prior": mean_prediction,
        "site_residue_ridge": np.asarray(site_prediction, dtype=float),
        "frozen_mint_ridge": np.asarray(frozen_prediction, dtype=float),
    }
    if frozen_live_prediction is not None:
        predictions_by_model["frozen_mint_ridge_live"] = np.asarray(
            frozen_live_prediction, dtype=float
        )
    predictions_by_model[args.regime] = np.asarray(neural_prediction, dtype=float)
    metrics = {
        name: regression_metrics(targets[test], prediction)
        for name, prediction in predictions_by_model.items()
    }
    frozen_reference_name = (
        "frozen_mint_ridge_live"
        if "frozen_mint_ridge_live" in metrics
        else "frozen_mint_ridge"
    )

    membership = frame[
        [
            "library",
            "pair_uid",
            "peptide_uid",
            "affibody_uid",
            "peptide_design_code",
            "affibody_design_code",
        ]
    ].copy()
    membership["role"] = roles
    membership_path = output_dir / "split_membership.csv"
    membership.to_csv(membership_path, index=False)
    os.chmod(str(membership_path), 0o600)

    prediction_path = output_dir / "test_predictions.csv"
    pd.DataFrame(prediction_rows(frame, test, targets, predictions_by_model)).to_csv(
        prediction_path, index=False
    )
    os.chmod(str(prediction_path), 0o600)

    history_path = output_dir / "validation_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    os.chmod(str(history_path), 0o600)

    delta_path = output_dir / "model_delta.pt"
    torch.save(
        {
            "regime": args.regime,
            "base_checkpoint_sha256": initial_hashes["checkpoint"],
            "trainable_parameter_names": trainable_names,
            "best_epoch": int(best_epoch),
            "head_state_dict": {
                name: value.detach().cpu() for name, value in model.head.state_dict().items()
            },
            "adapter_state": best_state.get("adapter", {}),
            "feature_mean": dense["x_mean"].astype(np.float32),
            "feature_scale": dense["x_scale"].astype(np.float32),
            "target_mean": float(dense["y_mean"]),
            "target_scale": float(dense["y_scale"]),
        },
        str(delta_path),
    )
    os.chmod(str(delta_path), 0o600)

    elapsed = time.time() - started
    metrics_path = output_dir / "metrics.json"
    write_json(
        metrics_path,
        {
            "best_epoch": int(best_epoch),
            "best_validation_mae": float(best_validation_mae),
            "models": metrics,
            "paired_test_delta_mae": {
                "fine_tuned_minus_headline_frozen_mint": float(
                    metrics[args.regime]["mae"] - metrics[frozen_reference_name]["mae"]
                ),
                "frozen_reference_model": frozen_reference_name,
                "fine_tuned_minus_site_residue": float(
                    metrics[args.regime]["mae"] - metrics["site_residue_ridge"]["mae"]
                ),
            },
        },
    )

    summary_path = output_dir / "run_summary.md"
    with open(str(summary_path), "w") as handle:
        handle.write(
            summary_markdown(
                args, split_details, best_epoch, metrics, elapsed, trainable_count
            )
        )
    os.chmod(str(summary_path), 0o600)

    artifact_paths = [
        membership_path,
        prediction_path,
        history_path,
        delta_path,
        metrics_path,
        summary_path,
    ]
    for name, path in immutable_paths.items():
        _require(
            code_baseline.sha256_file(path) == initial_hashes[name],
            "{} changed while the run was active".format(name),
        )
    manifest_path = output_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": float(elapsed),
            "hostname": socket.gethostname(),
            "code": {
                "path": str(script_path),
                "sha256": initial_hashes["script"],
                "split_dependency": str(dependency_path),
                "split_dependency_sha256": initial_hashes["split_dependency"],
            },
            "source": {
                "path": str(args.input.resolve()),
                "sha256": initial_hashes["input"],
                "feature_path": str(args.features.resolve()),
                "feature_sha256": initial_hashes["features"],
            },
            "base_model": {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": initial_hashes["checkpoint"],
                "config": str(args.config.resolve()),
                "config_sha256": initial_hashes["config"],
                "chain_order": ["smart-HLA-linker-peptide", "Affibody"],
                "pooling": "separate residue means concatenated to 2560 dimensions",
            },
            "split": split_details,
            "training": {
                "regime": args.regime,
                "training_seed": int(args.training_seed),
                "trainable_parameters": int(trainable_count),
                "trainable_parameter_names": trainable_names,
                "batch_size": int(args.batch_size),
                "eval_batch_size": int(args.eval_batch_size),
                "accumulation_steps": int(args.accumulation_steps),
                "max_epochs": int(args.max_epochs),
                "patience": int(args.patience),
                "min_delta_retention_points": float(args.min_delta),
                "loss": "SmoothL1 on training-fold-standardized retention",
                "huber_beta": float(args.huber_beta),
                "head_lr": float(args.head_lr),
                "adapter_lr": float(args.adapter_lr),
                "weight_decay": float(args.weight_decay),
                "clip_norm": float(args.clip_norm),
                "warmup_fraction": float(args.warmup_fraction),
                "amp_fp16": bool(args.regime != "head_only" and args.amp),
                "lora": {
                    "layers_zero_based": [31, 32],
                    "projections": ["q_proj", "v_proj"],
                    "rank": int(args.lora_rank),
                    "alpha": float(args.lora_alpha),
                    "dropout": float(args.lora_dropout),
                }
                if args.regime == "lora_cross"
                else None,
                "head_initialization": "fold-local frozen MINT ridge ({})".format(
                    dense["selection"]
                ),
                "frozen_ridge_alpha": float(dense["alpha"]),
                "frozen_ridge_selection": dense["selection"],
                "frozen_ridge_validation_grid": dense["grid"],
                "site_ridge_alpha": float(site["alpha"]),
                "site_ridge_validation_grid": site["grid"],
                "best_epoch": int(best_epoch),
                "best_validation_mae": float(best_validation_mae),
                "live_cached_feature_max_abs_error": float(parity_error),
                "epoch0_live_cached_prediction_max_abs_error": float(
                    initial_prediction_error
                ),
            },
            "runtime": {
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": gpu_name,
                "peak_allocated_bytes": int(peak_allocated),
                "python": sys.version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "platform": platform.platform(),
            },
            "artifacts": {
                path.name: code_baseline.sha256_file(path) for path in artifact_paths
            },
        },
    )
    print(str(summary_path), flush=True)
    print(summary_markdown(args, split_details, best_epoch, metrics, elapsed, trainable_count))


if __name__ == "__main__":
    run(parse_args())
