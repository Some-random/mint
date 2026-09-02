import pandas as pd

from downstream.AffibodyMHC.compare_pnu_site_convergence import (
    build_convergence_comparison,
)


def _tables(epoch, eta_shift=0.0, metric_shift=0.0):
    configs = []
    metrics = []
    for library in ("LibA", "LibB"):
        for risk in ("nnpu", "nnpnu"):
            for pi in (0.02, 0.05, 0.10, 0.15):
                eta = (1.0 if risk == "nnpu" else 0.25) + eta_shift
                configs.append({
                    "library": library,
                    "risk": risk,
                    "class_prior": pi,
                    "eta": eta,
                    "C": 1.0,
                    "selected_epoch": epoch,
                    "config_id": "{}-{}-{}-{}".format(library, risk, pi, eta),
                    "validation_auroc": 0.9,
                    "validation_average_precision": 0.8,
                    "validation_log_loss": 0.5,
                })
                metrics.append({
                    "library": library,
                    "risk": risk,
                    "class_prior": pi,
                    "auroc": 0.7 + metric_shift,
                    "average_precision": 0.6 + metric_shift,
                    "global_spearman": 0.5 + metric_shift,
                    "within_peptide_spearman": 0.4 + metric_shift,
                })
    return pd.DataFrame(configs), pd.DataFrame(metrics)


def test_convergence_comparison_is_matched_and_reports_metric_deltas():
    ref_c, ref_m = _tables(40)
    cur_c, cur_m = _tables(73, eta_shift=0.25, metric_shift=0.03)
    observed = build_convergence_comparison(ref_c, ref_m, cur_c, cur_m)
    assert len(observed) == 16
    assert observed["same_eta"].eq(0).all()
    assert observed["same_C"].eq(1).all()
    assert observed["same_selected_epoch"].eq(0).all()
    assert observed["auroc_change"].round(8).eq(0.03).all()
    assert observed["within_peptide_spearman_change"].round(8).eq(0.03).all()
