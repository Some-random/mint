import json
import os

from downstream.AffibodyMHC import evaluate_cached_weak_mint_practical as practical


def test_practical_estimator_changes_only_tolerance():
    model = practical.make_logistic(0.1, "all_class_weighted")
    assert model.C == 0.1
    assert model.class_weight == "balanced"
    assert model.solver == "liblinear"
    assert model.penalty == "l2"
    assert model.tol == practical.SOLVER_TOLERANCE


def test_practical_estimator_still_works_while_base_is_patched(monkeypatch):
    monkeypatch.setattr(practical.base, "_make_logistic", practical.make_logistic)
    model = practical.base._make_logistic(1.0, "all_unweighted")
    assert model.C == 1.0
    assert model.class_weight is None
    assert model.tol == practical.SOLVER_TOLERANCE


def test_manifest_rewrite_records_wrapper_dependency_and_tolerance(tmp_path):
    manifest = {"configuration": {}, "code": {"path": "old", "sha256": "old"}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    rewritten = practical.rewrite_manifest(tmp_path)
    assert rewritten["configuration"]["solver_tolerance"] == 1e-4
    assert rewritten["code"]["path"] == str(practical.WRAPPER_PATH)
    assert rewritten["dependencies"]["base_evaluator"]["path"] == str(practical.BASE_PATH)
    assert oct(os.stat(str(path)).st_mode & 0o777) == "0o600"
