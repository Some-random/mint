#!/usr/bin/env python
"""Fine-tune MINT cross-chain adapters on selection-derived binary labels.

The strong retention matrix is never used for training or checkpoint selection.
It is evaluated once, without calibration, after a frozen-head control and a
matched rank-2 LoRA model have been selected on identity-blocked weak-label
validation data.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_sequence_table import (
    CHAIN1_LENGTH,
    CHAIN2_LENGTH,
    OMITTED_LINKER,
    fill_template,
    load_provider_templates,
)
from downstream.AffibodyMHC.code_only_baseline import sha256_file, validate_private_output_path
from downstream.AffibodyMHC.extract_mint_features import PairCollator
from downstream.AffibodyMHC.finetune_mint_retention import (
    LoRALinear,
    set_deterministic_seed,
    update_learning_rates,
)
from mint.helpers.extract import MINTWrapper, load_config


LIBRARIES = ("LibA", "LibB")
EXPECTED_PRIMARY = {
    "LibA": {"positive": 11320, "negative": 11222},
    "LibB": {"positive": 23725, "negative": 6923},
}
EXPECTED_TRAINABLE = 23041
DEFAULT_C_GRID = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, payload):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def read_json(path):
    with open(str(path)) as handle:
        return json.load(handle)


def sha256_array(value):
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_input_lineage(
    weak_manifest_path,
    weak_labels_path,
    retention_manifest_path,
    retention_sequences_path,
    sequence_zip_path,
):
    """Prove that the derived tables and provider sequence archive belong together."""
    weak = read_json(weak_manifest_path)
    retention = read_json(retention_manifest_path)
    weak_label_hash = sha256_file(weak_labels_path)
    retention_hash = sha256_file(retention_sequences_path)
    sequence_zip_hash = sha256_file(sequence_zip_path)
    _require(
        weak.get("outputs", {}).get("weak_labels.csv") == weak_label_hash,
        "weak-label manifest/output hash mismatch",
    )
    _require(
        retention.get("output", {}).get("sha256") == retention_hash,
        "retention manifest/output hash mismatch",
    )
    weak_sources = weak.get("sources", {})
    retention_sources = retention.get("sources", {})
    _require(
        weak_sources.get("sequence_zip", {}).get("sha256") == sequence_zip_hash,
        "weak-label manifest uses a different sequence ZIP",
    )
    _require(
        retention_sources.get("sequence_zip", {}).get("sha256") == sequence_zip_hash,
        "retention manifest uses a different sequence ZIP",
    )
    _require(
        weak_sources.get("retention_csv", {}).get("sha256")
        == retention_sources.get("retention_csv", {}).get("sha256"),
        "weak labels and retention sequences use different retention matrices",
    )
    weak_deck_hash = weak_sources.get("sequence_zip", {}).get("template_deck_sha256")
    retention_deck_hash = retention_sources.get("sequence_zip", {}).get("deck_sha256")
    _require(
        weak_deck_hash == retention_deck_hash,
        "weak labels and retention sequences use different template decks",
    )
    return {
        "weak_labels_sha256": weak_label_hash,
        "retention_sequences_sha256": retention_hash,
        "sequence_zip_sha256": sequence_zip_hash,
        "retention_matrix_sha256": weak_sources["retention_csv"]["sha256"],
        "template_deck_sha256": weak_deck_hash,
    }


def identity_bin(uid, axis, split_seed, n_bins):
    payload = "{}|weakval{}|{}|{}".format(split_seed, n_bins, axis, uid)
    return int(hashlib.sha256(payload.encode("ascii")).hexdigest(), 16) % int(n_bins)


def balance_order(pair_uid, library, training_seed):
    payload = "{}|balance|{}|{}".format(training_seed, library, pair_uid)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def load_weak_primary(path, library, min_negative_r001_count):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    required = {
        "library",
        "pep",
        "aff",
        "r001_count",
        "weak_label",
        "within_declared_library_alphabet",
        "strict_retention_identity_cold_eligible",
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
    }
    _require(required.issubset(frame.columns), "weak-label schema mismatch")
    for column in (
        "r001_count",
        "weak_label",
        "within_declared_library_alphabet",
        "strict_retention_identity_cold_eligible",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame = frame.loc[frame["library"].eq(library)].copy()
    eligible_label = frame["weak_label"].eq(1) | (
        frame["weak_label"].eq(0)
        & frame["r001_count"].ge(int(min_negative_r001_count))
    )
    frame = frame.loc[
        frame["within_declared_library_alphabet"].eq(1)
        & frame["strict_retention_identity_cold_eligible"].eq(1)
        & eligible_label
    ].copy()
    frame = frame.sort_values("pair_uid").reset_index(drop=True)
    _require(not bool(frame["pair_uid"].duplicated().any()), "duplicate weak pair UID")
    _require(set(frame["weak_label"]) == {0, 1}, "weak primary needs both classes")
    if int(min_negative_r001_count) == 3:
        observed = {
            "positive": int(frame["weak_label"].eq(1).sum()),
            "negative": int(frame["weak_label"].eq(0).sum()),
        }
        _require(observed == EXPECTED_PRIMARY[library], "canonical primary count changed")
    return frame


def split_and_balance(frame, library, split_seed, training_seed, n_bins, max_train_per_class, max_validation_per_class):
    output = frame.copy()
    output["peptide_validation_bin"] = [
        identity_bin(uid, "pep", split_seed, n_bins) for uid in output["peptide_uid"]
    ]
    output["affibody_validation_bin"] = [
        identity_bin(uid, "aff", split_seed, n_bins) for uid in output["affibody_uid"]
    ]
    peptide_held = output["peptide_validation_bin"].eq(0)
    affibody_held = output["affibody_validation_bin"].eq(0)
    validation = peptide_held & affibody_held
    guarded = peptide_held ^ affibody_held
    train_candidate = ~peptide_held & ~affibody_held
    output["role"] = "guarded"
    output.loc[validation, "role"] = "validation"
    output.loc[train_candidate, "role"] = "unused_balance"

    train_candidates = output.loc[train_candidate].copy()
    counts = train_candidates.groupby("weak_label").size().to_dict()
    _require(set(counts) == {0, 1}, "train candidates need both classes")
    per_class = min(int(counts[0]), int(counts[1]))
    if int(max_train_per_class) > 0:
        per_class = min(per_class, int(max_train_per_class))
    selected_train = []
    for label in (0, 1):
        candidates = train_candidates.loc[train_candidates["weak_label"].eq(label)].copy()
        candidates["_balance_order"] = [
            balance_order(uid, library, training_seed) for uid in candidates["pair_uid"]
        ]
        selected_train.extend(
            candidates.sort_values(["_balance_order", "pair_uid"])
            .head(per_class)
            .index.tolist()
        )
    output.loc[selected_train, "role"] = "train"

    if int(max_validation_per_class) > 0:
        validation_rows = output.loc[output["role"].eq("validation")].copy()
        keep_validation = []
        for label in (0, 1):
            candidates = validation_rows.loc[validation_rows["weak_label"].eq(label)].copy()
            candidates["_balance_order"] = [
                balance_order(uid, library + "|validation", training_seed)
                for uid in candidates["pair_uid"]
            ]
            keep_validation.extend(
                candidates.sort_values(["_balance_order", "pair_uid"])
                .head(int(max_validation_per_class))
                .index.tolist()
            )
        discard = output["role"].eq("validation") & ~output.index.isin(keep_validation)
        output.loc[discard, "role"] = "unused_validation_cap"

    train = output.loc[output["role"].eq("train")]
    validation_rows = output.loc[output["role"].eq("validation")]
    _require(set(train["weak_label"]) == {0, 1}, "balanced train lacks a class")
    _require(set(validation_rows["weak_label"]) == {0, 1}, "validation lacks a class")
    _require(
        set(train["peptide_uid"]).isdisjoint(set(validation_rows["peptide_uid"])),
        "weak split leaks peptide identity",
    )
    _require(
        set(train["affibody_uid"]).isdisjoint(set(validation_rows["affibody_uid"])),
        "weak split leaks Affibody identity",
    )
    _require(
        int(train["weak_label"].eq(0).sum()) == int(train["weak_label"].eq(1).sum()),
        "training is not class balanced",
    )
    return output


def load_retention(path, library):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    required = {
        "library",
        "measurement_missing",
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
        "target_retention",
        "target_binder",
        "peptide_design_code",
        "affibody_design_code",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
    }
    _require(required.issubset(frame.columns), "retention sequence schema mismatch")
    frame = frame.loc[
        frame["library"].eq(library) & frame["measurement_missing"].eq("0")
    ].copy()
    frame["target_retention"] = pd.to_numeric(frame["target_retention"], errors="raise")
    frame["target_binder"] = pd.to_numeric(frame["target_binder"], errors="raise").astype(int)
    frame["weak_label"] = frame["target_binder"]
    _require(set(frame["target_binder"]) == {0, 1}, "retention test lacks a class")
    _require(
        bool(frame["chain1_smart_hla_linker_peptide_sequence"].map(len).eq(CHAIN1_LENGTH).all()),
        "retention chain-1 length mismatch",
    )
    _require(
        bool(frame["chain2_affibody_sequence"].map(len).eq(CHAIN2_LENGTH).all()),
        "retention chain-2 length mismatch",
    )
    return frame.sort_values("pair_uid").reset_index(drop=True)


def validate_retention_sequence_reconstruction(frame, templates):
    for row in frame.itertuples(index=False):
        full = fill_template(
            templates[row.library],
            str(row.peptide_design_code) + str(row.affibody_design_code),
        )
        _require(
            full[:CHAIN1_LENGTH] == row.chain1_smart_hla_linker_peptide_sequence,
            "retention chain-1 does not match provider template",
        )
        _require(
            full[-CHAIN2_LENGTH:] == row.chain2_affibody_sequence,
            "retention chain-2 does not match provider template",
        )
        _require(
            full[CHAIN1_LENGTH : CHAIN1_LENGTH + len(OMITTED_LINKER)] == OMITTED_LINKER,
            "retention reconstruction linker mismatch",
        )


def reconstruct_weak_sequences(frame, template):
    records = []
    for row in frame.itertuples(index=False):
        full = fill_template(template, str(row.pep) + str(row.aff))
        omitted = full[CHAIN1_LENGTH : CHAIN1_LENGTH + len(OMITTED_LINKER)]
        _require(omitted == OMITTED_LINKER, "weak reconstructed linker mismatch")
        records.append(
            {
                "library": row.library,
                "pair_uid": row.pair_uid,
                "weak_label": int(row.weak_label),
                "chain1_smart_hla_linker_peptide_sequence": full[:CHAIN1_LENGTH],
                "chain2_affibody_sequence": full[-CHAIN2_LENGTH:],
            }
        )
    return pd.DataFrame(records)


class FastPairDataset(Dataset):
    def __init__(self, frame):
        self.chain1 = frame["chain1_smart_hla_linker_peptide_sequence"].tolist()
        self.chain2 = frame["chain2_affibody_sequence"].tolist()
        self.labels = frame["weak_label"].astype(float).tolist()
        self.uids = frame["pair_uid"].tolist()

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        return self.chain1[index], self.chain2[index], self.labels[index], self.uids[index]


class LabeledPairCollator:
    def __init__(self):
        self.pair = PairCollator()

    def __call__(self, batch):
        chain1, chain2, labels, uids = zip(*batch)
        chains, chain_ids, _, _ = self.pair(
            [(value1, value2, "unused", uid) for value1, value2, uid in zip(chain1, chain2, uids)]
        )
        return chains, chain_ids, torch.tensor(labels, dtype=torch.float32), list(uids)


def make_loader(frame, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        FastPairDataset(frame),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=LabeledPairCollator(),
        num_workers=0,
        generator=generator,
    )


class MINTSelectionClassifier(nn.Module):
    def __init__(self, config, checkpoint, device, rank, alpha, dropout):
        super().__init__()
        self.wrapper = MINTWrapper(
            load_config(str(config)),
            str(checkpoint),
            freeze_percent=1.0,
            use_multimer=True,
            sep_chains=True,
            device=str(device),
        )
        self.wrapper.model.requires_grad_(False)
        self.adapters = []
        for layer_index in (31, 32):
            attention = self.wrapper.model.layers[layer_index].multimer_attn
            for projection in ("q_proj", "v_proj"):
                adapter = LoRALinear(
                    getattr(attention, projection),
                    rank=int(rank),
                    alpha=float(alpha),
                    dropout=float(dropout),
                )
                setattr(attention, projection, adapter)
                self.adapters.append(adapter)
        self.head = nn.Linear(2560, 1)
        self.register_buffer("feature_mean", torch.zeros(2560, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.ones(2560, dtype=torch.float32))

    def forward(self, chains, chain_ids):
        features = self.wrapper(chains, chain_ids)
        features = (features - self.feature_mean) / self.feature_scale
        return self.head(features).flatten()

    def set_feature_standardization(self, mean, scale):
        _require(np.asarray(mean).shape == (2560,), "feature mean shape mismatch")
        _require(np.asarray(scale).shape == (2560,), "feature scale shape mismatch")
        _require(bool(np.isfinite(mean).all()), "non-finite feature mean")
        _require(bool(np.isfinite(scale).all()), "non-finite feature scale")
        _require(bool((np.asarray(scale) > 0.0).all()), "nonpositive feature scale")
        with torch.no_grad():
            self.feature_mean.copy_(torch.from_numpy(np.asarray(mean, dtype=np.float32)))
            self.feature_scale.copy_(torch.from_numpy(np.asarray(scale, dtype=np.float32)))

    def set_train_mode(self):
        self.wrapper.model.eval()
        self.head.train()
        for adapter in self.adapters:
            adapter.train()

    def adapter_parameters(self):
        head_ids = {id(parameter) for parameter in self.head.parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]


def trainable_audit(model, rank=2):
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    expected = 2561 + 10240 * int(rank)
    _require(count == expected, "unexpected trainable parameter count {}".format(count))
    adapter_names = [name for name in names if not name.startswith("head.")]
    _require(len(adapter_names) == 8, "unexpected LoRA tensor count")
    _require(all("lora_" in name for name in adapter_names), "non-LoRA encoder tensor trainable")
    return names, count


def extract_frozen_features(model, loader, device, stage):
    model.eval()
    blocks = []
    labels = []
    uids = []
    with torch.no_grad():
        for batch_number, (chains, chain_ids, target, pair_uids) in enumerate(loader):
            features = model.wrapper(chains.to(device), chain_ids.to(device))
            blocks.append(features.float().cpu().numpy())
            labels.extend(target.numpy().astype(float).tolist())
            uids.extend(pair_uids)
            if (batch_number + 1) % 100 == 0 or batch_number + 1 == len(loader):
                print(
                    "{} frozen features: {}/{} batches".format(
                        stage, batch_number + 1, len(loader)
                    ),
                    flush=True,
                )
    return np.concatenate(blocks).astype(np.float32), np.asarray(labels, dtype=int), uids


def binary_metrics(labels, probability):
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    _require(set(labels) == {0, 1}, "binary metrics require both classes")
    _require(bool(np.isfinite(probability).all()), "non-finite probability")
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probability)),
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
    }


def fit_standardization(train_features):
    values = np.asarray(train_features, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    _require(bool(np.isfinite(mean).all()), "non-finite fitted feature mean")
    _require(bool(np.isfinite(scale).all()), "non-finite fitted feature scale")
    return mean, scale


def apply_standardization(features, mean, scale):
    values = (np.asarray(features, dtype=np.float64) - mean) / scale
    _require(bool(np.isfinite(values).all()), "non-finite standardized feature")
    return values.astype(np.float32)


def retention_metrics(retention, probability, top_k):
    labels = retention["target_binder"].to_numpy(dtype=int)
    target = retention["target_retention"].to_numpy(dtype=float)
    metrics = binary_metrics(labels, probability)
    rho, pvalue = spearmanr(target, probability)
    order = np.argsort(-np.asarray(probability, dtype=float))
    k = min(int(top_k), len(order))
    top = order[:k]
    metrics.update(
        {
            "spearman_rho": float(rho),
            "spearman_pvalue": float(pvalue),
            "top_k": int(k),
            "top_k_binder_fraction": float(labels[top].mean()),
            "top_k_mean_retention": float(target[top].mean()),
        }
    )
    return metrics


def fit_frozen_head(train_features, train_labels, validation_features, validation_labels, c_grid):
    records = []
    candidates = []
    for c_value in c_grid:
        classifier = LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            penalty="l2",
            fit_intercept=True,
            class_weight=None,
            random_state=0,
            max_iter=2000,
            tol=1e-7,
        )
        classifier.fit(train_features, train_labels)
        _require(int(classifier.n_iter_[0]) < classifier.max_iter, "frozen head did not converge")
        probability = classifier.predict_proba(validation_features)[:, 1]
        metrics = binary_metrics(validation_labels, probability)
        records.append(dict({"C": float(c_value)}, **metrics))
        candidates.append((metrics["log_loss"], -metrics["auroc"], float(c_value), classifier))
    _, _, selected_c, selected = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return selected, float(selected_c), records


def load_head(model, classifier):
    with torch.no_grad():
        model.head.weight.copy_(
            torch.from_numpy(np.asarray(classifier.coef_, dtype=np.float32))
        )
        model.head.bias.copy_(
            torch.from_numpy(np.asarray(classifier.intercept_, dtype=np.float32))
        )


def state_for_model(model):
    return {
        "head": copy.deepcopy(model.head.state_dict()),
        "adapter": {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("head.")
        },
    }


def restore_model(model, state):
    model.head.load_state_dict(state["head"])
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state["adapter"].items():
            named[name].copy_(value.to(named[name].device))


def predict_live(model, loader, device):
    model.eval()
    logits = []
    labels = []
    uids = []
    with torch.no_grad():
        for chains, chain_ids, target, pair_uids in loader:
            value = model(chains.to(device), chain_ids.to(device))
            logits.append(value.float().cpu().numpy())
            labels.extend(target.numpy().astype(int).tolist())
            uids.extend(pair_uids)
    logits = np.concatenate(logits).astype(float)
    return logits, 1.0 / (1.0 + np.exp(-logits)), np.asarray(labels, dtype=int), uids


def train_lora(model, train_loader, validation_loader, args, device):
    groups = [
        {"params": [model.head.weight], "lr": float(args.head_lr), "base_lr": float(args.head_lr), "weight_decay": float(args.weight_decay)},
        {"params": [model.head.bias], "lr": float(args.head_lr), "base_lr": float(args.head_lr), "weight_decay": 0.0},
        {"params": model.adapter_parameters(), "lr": float(args.adapter_lr), "base_lr": float(args.adapter_lr), "weight_decay": float(args.weight_decay)},
    ]
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)
    updates_per_epoch = int(
        math.ceil(float(len(train_loader)) / float(args.accumulation_steps))
    )
    total_steps = max(1, updates_per_epoch * int(args.max_epochs))
    global_step = 0
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    _, initial_probability, initial_labels, _ = predict_live(
        model, validation_loader, device
    )
    initial_metrics = binary_metrics(initial_labels, initial_probability)
    best_loss = initial_metrics["log_loss"]
    best_state = state_for_model(model)
    best_epoch = 0
    patience = 0
    history = [dict({"epoch": 0, "train_loss": None}, **initial_metrics)]

    for epoch in range(1, int(args.max_epochs) + 1):
        model.set_train_mode()
        optimizer.zero_grad()
        total_loss = 0.0
        examples = 0
        window_examples = 0
        for batch_number, (chains, chain_ids, target, _) in enumerate(train_loader):
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            target = target.to(device)
            logits = model(chains, chain_ids)
            loss = loss_fn(logits.float(), target.float())
            loss.backward()
            total_loss += float(loss.detach().cpu())
            examples += int(len(target))
            window_examples += int(len(target))
            tail = batch_number + 1 == len(train_loader)
            if (batch_number + 1) % int(args.accumulation_steps) == 0 or tail:
                update_learning_rates(
                    optimizer, global_step, total_steps, args.warmup_fraction
                )
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
            if (batch_number + 1) % 200 == 0 or tail:
                print(
                    "epoch {:02d} training: {}/{} batches".format(
                        epoch, batch_number + 1, len(train_loader)
                    ),
                    flush=True,
                )
        _, probability, labels, _ = predict_live(model, validation_loader, device)
        metrics = binary_metrics(labels, probability)
        history.append(dict({"epoch": int(epoch), "train_loss": float(total_loss / examples)}, **metrics))
        print(
            "epoch {:02d} train_loss {:.6f} val_log_loss {:.6f} val_auroc {:.4f}".format(
                epoch, total_loss / examples, metrics["log_loss"], metrics["auroc"]
            ),
            flush=True,
        )
        if metrics["log_loss"] < best_loss - float(args.min_delta):
            best_loss = metrics["log_loss"]
            best_state = state_for_model(model)
            best_epoch = int(epoch)
            patience = 0
        else:
            patience += 1
        if patience >= int(args.patience):
            break
    restore_model(model, best_state)
    return history, best_state, best_epoch, best_loss


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-label-dir", required=True, type=Path)
    parser.add_argument("--retention-sequences", required=True, type=Path)
    parser.add_argument(
        "--retention-manifest",
        type=Path,
        help="defaults to RETENTION_SEQUENCES with .manifest.json suffix",
    )
    parser.add_argument("--sequence-zip", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--library", choices=LIBRARIES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--training-seed", type=int, default=20260811)
    parser.add_argument("--validation-bins", type=int, default=5)
    parser.add_argument("--min-negative-r001-count", type=int, default=3)
    parser.add_argument("--max-train-per-class", type=int, default=0)
    parser.add_argument("--max-validation-per-class", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--adapter-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--head-c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    _require(args.validation_bins >= 2, "validation bins must be >=2")
    _require(args.batch_size >= 1 and args.eval_batch_size >= 1, "batch sizes must be positive")
    _require(args.accumulation_steps >= 1, "accumulation steps must be positive")
    _require(args.max_epochs >= 1 and args.patience >= 1, "epoch/patience must be positive")
    _require(args.max_train_per_class >= 0, "training cap must be nonnegative")
    _require(args.max_validation_per_class >= 0, "validation cap must be nonnegative")
    _require(args.min_negative_r001_count >= 1, "negative count threshold must be positive")
    _require(args.lora_rank >= 1, "LoRA rank must be positive")
    _require(0.0 <= args.lora_dropout < 1.0, "LoRA dropout must be in [0,1)")
    _require(0.0 <= args.warmup_fraction < 1.0, "warmup fraction must be in [0,1)")
    _require(args.min_delta >= 0.0, "minimum delta must be nonnegative")
    _require(args.weight_decay >= 0.0, "weight decay must be nonnegative")
    _require(args.top_k >= 1, "top-k must be positive")
    for value, label in (
        (args.head_lr, "head learning rate"),
        (args.adapter_lr, "adapter learning rate"),
        (args.clip_norm, "clip norm"),
        (args.lora_alpha, "LoRA alpha"),
    ):
        _require(value > 0.0 and math.isfinite(value), "{} must be finite and positive".format(label))
    retention_manifest_path = args.retention_manifest
    if retention_manifest_path is None:
        retention_manifest_path = args.retention_sequences.with_suffix(".manifest.json")
    for path in (
        args.weak_label_dir / "weak_labels.csv",
        args.weak_label_dir / "manifest.json",
        args.retention_sequences,
        retention_manifest_path,
        args.sequence_zip,
        args.checkpoint,
        args.config,
    ):
        _require(path.is_file(), "missing input {}".format(path))
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(torch.cuda.is_available(), "CUDA is required")

    script_path = Path(__file__).resolve()
    dependencies = {
        "script": script_path,
        "weak_labels": args.weak_label_dir / "weak_labels.csv",
        "weak_manifest": args.weak_label_dir / "manifest.json",
        "retention_sequences": args.retention_sequences,
        "retention_manifest": retention_manifest_path,
        "sequence_zip": args.sequence_zip,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "build_sequence_dependency": REPO_ROOT / "downstream/AffibodyMHC/build_sequence_table.py",
        "lora_dependency": REPO_ROOT / "downstream/AffibodyMHC/finetune_mint_retention.py",
        "extract_dependency": REPO_ROOT / "downstream/AffibodyMHC/extract_mint_features.py",
        "code_baseline_dependency": REPO_ROOT / "downstream/AffibodyMHC/code_only_baseline.py",
        "mint_extract_dependency": REPO_ROOT / "mint/helpers/extract.py",
        "mint_esm2_dependency": REPO_ROOT / "mint/model/esm2.py",
        "mint_modules_dependency": REPO_ROOT / "mint/modules.py",
        "mint_attention_dependency": REPO_ROOT / "mint/multihead_attention.py",
        "mint_data_dependency": REPO_ROOT / "mint/data.py",
        "mint_rotary_dependency": REPO_ROOT / "mint/rotary_embedding.py",
    }
    initial_hashes = {name: sha256_file(path) for name, path in dependencies.items()}
    lineage = validate_input_lineage(
        dependencies["weak_manifest"],
        dependencies["weak_labels"],
        dependencies["retention_manifest"],
        dependencies["retention_sequences"],
        dependencies["sequence_zip"],
    )
    set_deterministic_seed(args.training_seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    primary = load_weak_primary(
        args.weak_label_dir / "weak_labels.csv",
        args.library,
        args.min_negative_r001_count,
    )
    membership = split_and_balance(
        primary,
        args.library,
        args.split_seed,
        args.training_seed,
        args.validation_bins,
        args.max_train_per_class,
        args.max_validation_per_class,
    )
    templates, template_member, template_deck_hash = load_provider_templates(args.sequence_zip)
    _require(
        template_deck_hash == lineage["template_deck_sha256"],
        "loaded provider template deck does not match manifests",
    )
    retention = load_retention(args.retention_sequences, args.library)
    validate_retention_sequence_reconstruction(retention, templates)
    _require(set(primary["peptide_uid"]).isdisjoint(set(retention["peptide_uid"])), "primary weak data shares retention peptide")
    _require(set(primary["affibody_uid"]).isdisjoint(set(retention["affibody_uid"])), "primary weak data shares retention Affibody")
    _require(set(primary["pair_uid"]).isdisjoint(set(retention["pair_uid"])), "primary weak data shares retention pair")

    train_source = membership.loc[membership["role"].eq("train")].copy()
    validation_source = membership.loc[membership["role"].eq("validation")].copy()
    train_frame = reconstruct_weak_sequences(train_source, templates[args.library])
    validation_frame = reconstruct_weak_sequences(validation_source, templates[args.library])
    train_loader_features = make_loader(train_frame, args.eval_batch_size, False, args.training_seed)
    validation_loader = make_loader(validation_frame, args.eval_batch_size, False, args.training_seed)
    retention_loader = make_loader(retention, args.eval_batch_size, False, args.training_seed)

    model = MINTSelectionClassifier(
        args.config,
        args.checkpoint,
        device,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
    ).to(device)
    trainable_names, trainable_count = trainable_audit(model, args.lora_rank)
    train_features, train_labels, train_uids = extract_frozen_features(
        model, train_loader_features, device, "train"
    )
    validation_features, validation_labels, validation_uids = extract_frozen_features(
        model, validation_loader, device, "validation"
    )
    retention_features, retention_labels, retention_uids = extract_frozen_features(
        model, retention_loader, device, "retention"
    )
    _require(train_uids == train_frame["pair_uid"].tolist(), "train feature UID order mismatch")
    _require(validation_uids == validation_frame["pair_uid"].tolist(), "validation feature UID order mismatch")
    _require(retention_uids == retention["pair_uid"].tolist(), "retention feature UID order mismatch")
    feature_mean, feature_scale = fit_standardization(train_features)
    train_features_z = apply_standardization(train_features, feature_mean, feature_scale)
    validation_features_z = apply_standardization(
        validation_features, feature_mean, feature_scale
    )
    retention_features_z = apply_standardization(
        retention_features, feature_mean, feature_scale
    )
    model.set_feature_standardization(feature_mean, feature_scale)

    c_grid = tuple(sorted(set(float(value) for value in args.head_c_grid)))
    _require(c_grid and all(value > 0 and math.isfinite(value) for value in c_grid), "bad head C grid")
    frozen_head, selected_c, head_grid = fit_frozen_head(
        train_features_z,
        train_labels,
        validation_features_z,
        validation_labels,
        c_grid,
    )
    load_head(model, frozen_head)
    frozen_validation_probability = frozen_head.predict_proba(validation_features_z)[:, 1]
    frozen_retention_probability = frozen_head.predict_proba(retention_features_z)[:, 1]
    _, live_frozen_retention_probability, _, live_retention_uids = predict_live(
        model, retention_loader, device
    )
    _require(live_retention_uids == retention_uids, "live frozen retention UID order mismatch")
    epoch0_probability_error = float(
        np.max(np.abs(frozen_retention_probability - live_frozen_retention_probability))
    )
    _require(epoch0_probability_error <= 0.01, "cached/live epoch-0 probability mismatch")

    train_loader = make_loader(train_frame, args.batch_size, True, args.training_seed)
    history, best_state, best_epoch, best_validation_loss = train_lora(
        model, train_loader, validation_loader, args, device
    )
    _, lora_validation_probability, lora_validation_labels, lora_validation_uids = predict_live(
        model, validation_loader, device
    )
    _require(lora_validation_uids == validation_uids, "LoRA validation UID order mismatch")
    _, lora_retention_probability, lora_retention_labels, lora_retention_uids = predict_live(
        model, retention_loader, device
    )
    _require(lora_retention_uids == retention_uids, "LoRA retention UID order mismatch")
    _require(bool(np.array_equal(lora_retention_labels, retention_labels)), "retention labels changed")

    metrics = {
        "weak_validation": {
            "frozen_head": binary_metrics(validation_labels, frozen_validation_probability),
            "lora_cross": binary_metrics(lora_validation_labels, lora_validation_probability),
        },
        "retention_transfer": {
            "frozen_head": retention_metrics(retention, frozen_retention_probability, args.top_k),
            "lora_cross": retention_metrics(retention, lora_retention_probability, args.top_k),
        },
        "best_epoch": int(best_epoch),
        "best_validation_log_loss": float(best_validation_loss),
        "epoch0_cached_live_probability_max_abs_error": epoch0_probability_error,
    }

    _require(not output_dir.exists(), "output directory appeared during run")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)

    membership_output = membership[
        [
            "library",
            "pair_uid",
            "peptide_uid",
            "affibody_uid",
            "weak_label",
            "r001_count",
            "peptide_validation_bin",
            "affibody_validation_bin",
            "role",
        ]
    ].copy()
    membership_path = output_dir / "weak_membership.csv"
    membership_output.to_csv(membership_path, index=False)
    os.chmod(str(membership_path), 0o600)

    prediction_path = output_dir / "retention_predictions.csv"
    predictions = retention[
        ["library", "pair_uid", "peptide_uid", "affibody_uid", "target_retention", "target_binder"]
    ].copy()
    predictions["frozen_mint_head_probability"] = frozen_retention_probability
    predictions["lora_cross_probability"] = lora_retention_probability
    predictions.to_csv(prediction_path, index=False)
    os.chmod(str(prediction_path), 0o600)

    history_path = output_dir / "validation_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    os.chmod(str(history_path), 0o600)

    head_grid_path = output_dir / "frozen_head_grid.csv"
    pd.DataFrame(head_grid).to_csv(head_grid_path, index=False)
    os.chmod(str(head_grid_path), 0o600)

    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)

    delta_path = output_dir / "model_delta.pt"
    torch.save(
        {
            "library": args.library,
            "base_checkpoint_sha256": initial_hashes["checkpoint"],
            "trainable_parameter_names": trainable_names,
            "best_epoch": int(best_epoch),
            "head_state_dict": {
                name: value.detach().cpu() for name, value in model.head.state_dict().items()
            },
            "adapter_state": best_state["adapter"],
            "frozen_head_state_dict": {
                "weight": torch.from_numpy(
                    np.asarray(frozen_head.coef_, dtype=np.float32)
                ),
                "bias": torch.from_numpy(
                    np.asarray(frozen_head.intercept_, dtype=np.float32)
                ),
            },
            "feature_standardization": {
                "mean": torch.from_numpy(np.asarray(feature_mean, dtype=np.float32)),
                "scale": torch.from_numpy(np.asarray(feature_scale, dtype=np.float32)),
            },
            "selected_frozen_head_C": float(selected_c),
        },
        str(delta_path),
    )
    os.chmod(str(delta_path), 0o600)

    summary_path = output_dir / "run_summary.md"
    frozen_transfer = metrics["retention_transfer"]["frozen_head"]
    lora_transfer = metrics["retention_transfer"]["lora_cross"]
    summary = "\n".join(
        [
            "# MINT weak-selection fine-tuning: {}".format(args.library),
            "",
            "- Balanced weak train: {} rows".format(len(train_frame)),
            "- Identity-blocked weak validation: {} rows".format(len(validation_frame)),
            "- Retention transfer test: {} rows; labels never used for fitting/selection".format(len(retention)),
            "- Trainable parameters: {:,} (rank-{} LoRA plus linear head)".format(
                trainable_count, args.lora_rank
            ),
            "- Best LoRA epoch: {}".format(best_epoch),
            "",
            "| Model | Retention Spearman | AUROC >=75 | AUPRC |",
            "|---|---:|---:|---:|",
            "| Frozen MINT head | {:.4f} | {:.4f} | {:.4f} |".format(
                frozen_transfer["spearman_rho"], frozen_transfer["auroc"], frozen_transfer["auprc"]
            ),
            "| MINT LoRA | {:.4f} | {:.4f} | {:.4f} |".format(
                lora_transfer["spearman_rho"], lora_transfer["auroc"], lora_transfer["auprc"]
            ),
            "",
            "Retention partner identities were used a priori to enforce peptide-and-Affibody-cold separation; retention labels were not used for fitting or checkpoint selection.",
            "",
            "This is retrospective selection-to-retention transfer. R009/R010 were chosen after inspecting the retention experiment.",
            "",
        ]
    )
    with open(str(summary_path), "w") as handle:
        handle.write(summary)
    os.chmod(str(summary_path), 0o600)

    artifact_paths = (
        membership_path,
        prediction_path,
        history_path,
        head_grid_path,
        metrics_path,
        delta_path,
        summary_path,
    )
    role_counts = {
        role: int(membership["role"].eq(role).sum())
        for role in sorted(membership["role"].unique())
    }
    _require(sum(role_counts.values()) == len(primary), "membership role counts do not sum")
    for name, path in dependencies.items():
        _require(sha256_file(path) == initial_hashes[name], "{} changed during run".format(name))
    manifest_path = output_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": float(time.time() - started),
            "hostname": socket.gethostname(),
            "argv": list(sys.argv),
            "analysis_status": "retrospective_exploratory",
            "validated_lineage": lineage,
            "code_and_inputs": {
                name: {"path": str(path.resolve()), "sha256": initial_hashes[name]}
                for name, path in dependencies.items()
            },
            "provider_input": {
                "chain_order": ["smart-HLA-linker-peptide", "Affibody"],
                "chain_lengths": [CHAIN1_LENGTH, CHAIN2_LENGTH],
                "omitted_linker": OMITTED_LINKER,
                "template_member": template_member,
                "template_deck_sha256": template_deck_hash,
                "tcr_present": False,
            },
            "configuration": {
                "library": args.library,
                "primary_filter": "on-design, strict retention identity-cold, all positives plus provider negatives R001>={}".format(args.min_negative_r001_count),
                "split_seed": int(args.split_seed),
                "training_seed": int(args.training_seed),
                "validation_bins": int(args.validation_bins),
                "balanced_train": True,
                "max_train_per_class": int(args.max_train_per_class),
                "max_validation_per_class": int(args.max_validation_per_class),
                "batch_size": int(args.batch_size),
                "eval_batch_size": int(args.eval_batch_size),
                "accumulation_steps": int(args.accumulation_steps),
                "max_epochs": int(args.max_epochs),
                "patience": int(args.patience),
                "min_delta": float(args.min_delta),
                "loss": "BCEWithLogitsLoss on raw logits",
                "feature_standardization": {
                    "fit_on": "weak training rows only",
                    "mean_float64_sha256": sha256_array(feature_mean),
                    "scale_float64_sha256": sha256_array(feature_scale),
                    "scale_min": float(np.min(feature_scale)),
                    "scale_median": float(np.median(feature_scale)),
                    "scale_max": float(np.max(feature_scale)),
                },
                "selected_frozen_head_C": float(selected_c),
                "head_C_grid": list(c_grid),
                "head_lr": float(args.head_lr),
                "adapter_lr": float(args.adapter_lr),
                "weight_decay": float(args.weight_decay),
                "warmup_fraction": float(args.warmup_fraction),
                "clip_norm": float(args.clip_norm),
                "top_k": int(args.top_k),
                "lora": {
                    "layers_zero_based": [31, 32],
                    "projections": ["q_proj", "v_proj"],
                    "rank": int(args.lora_rank),
                    "alpha": float(args.lora_alpha),
                    "dropout": float(args.lora_dropout),
                },
                "precision": "FP32",
                "retention_usage": {
                    "labels": "final uncalibrated evaluation only; never used for fitting or checkpoint selection",
                    "partner_identities": "used a priori to enforce peptide-and-Affibody-cold separation",
                },
            },
            "counts": {
                "primary_rows": int(len(primary)),
                "train": int(len(train_frame)),
                "train_negative": int(train_source["weak_label"].eq(0).sum()),
                "train_positive": int(train_source["weak_label"].eq(1).sum()),
                "validation": int(len(validation_frame)),
                "validation_negative": int(validation_source["weak_label"].eq(0).sum()),
                "validation_positive": int(validation_source["weak_label"].eq(1).sum()),
                "membership_roles": role_counts,
                "retention_test": int(len(retention)),
            },
            "training": {
                "trainable_parameters": int(trainable_count),
                "trainable_parameter_names": trainable_names,
                "best_epoch": int(best_epoch),
                "best_validation_log_loss": float(best_validation_loss),
                "epoch0_cached_live_probability_max_abs_error": epoch0_probability_error,
            },
            "metrics": metrics,
            "runtime": {
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": torch.cuda.get_device_name(device),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "python": sys.version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
                "platform": platform.platform(),
            },
            "artifacts": {path.name: sha256_file(path) for path in artifact_paths},
            "permissions": {"directory": "0700", "files": "0600"},
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    run(parse_args())
