#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-hoc revision experiments E1, E2, E3a, E4, E5, E6 for the JES manuscript.
Strictly isolated from locked confirmatory results: reads only existing code/data/CSVs
and writes exclusively to revision_experiments/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import binomtest, wilcoxon
from sklearn.model_selection import GroupKFold

_DEFAULT_ROOT = Path(os.environ.get("SOH_WORK_ROOT", str(Path(__file__).resolve().parent.parent)))
ROOT = _DEFAULT_ROOT
OLD_DATA = ROOT / "data"
NEW_DATA = ROOT / "data"
OUT = ROOT / "revision_experiments"

def _configure_paths(work_root: Path) -> None:
    global ROOT, OLD_DATA, NEW_DATA, OUT
    ROOT = work_root.resolve()
    OLD_DATA = ROOT / "data"
    NEW_DATA = ROOT / "data"
    OUT = ROOT / "revision_experiments"
    OUT.mkdir(parents=True, exist_ok=True)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

if str(ROOT / "src") not in sys.path and (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import reference_models as ref  # noqa: E402
import stage2_run as s2  # noqa: E402


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        v = obj.item()
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_json_sanitize(payload), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_all_datasets() -> Dict[str, List[ref.Cell]]:
    return {
        "NASA": ref.load_cells(str(OLD_DATA / "NASA.npz"), "NASA"),
        "XJTU_B1": ref.load_cells(str(OLD_DATA / "XJTU_B1.npz"), "XJTU_B1"),
        "XJTU_B2": ref.load_cells(str(OLD_DATA / "XJTU_B2.npz"), "XJTU_B2"),
        "XJTU_B3": ref.load_cells(str(OLD_DATA / "XJTU_B3.npz"), "XJTU_B3"),
        "XJTU_B4": ref.load_cells(str(NEW_DATA / "XJTU_B4.npz"), "XJTU_B4"),
        "XJTU_B5": ref.load_cells(str(NEW_DATA / "XJTU_B5.npz"), "XJTU_B5"),
        "XJTU_B6": ref.load_cells(str(NEW_DATA / "XJTU_B6.npz"), "XJTU_B6"),
    }


# =============================================================================
# E1 (M1): Skill Scores vs Trivial Baselines
# =============================================================================
def run_e1() -> Dict[str, Any]:
    print("[E1] Computing skill scores vs trivial baselines...", flush=True)
    # --- Stage 1 ---
    df_nasa = pd.read_csv(ROOT / "results" / "loco_NASA" / "cell_metrics.csv")
    df_nasa = df_nasa.rename(columns={"outer_cell": "cell"})
    df_nasa["dataset"] = "NASA"

    df_b1 = pd.read_csv(ROOT / "results" / "loco_XJTU_B1" / "cell_metrics.csv")
    df_b1 = df_b1.rename(columns={"outer_cell": "cell"})
    df_b1["dataset"] = "XJTU_B1"

    df_dev = pd.read_csv(ROOT / "results" / "development" / "cell_metrics.csv")
    df_conf = pd.read_csv(ROOT / "results" / "confirmation" / "cell_metrics.csv")

    s1_all = pd.concat([df_nasa, df_b1, df_dev, df_conf], ignore_index=True)
    s1_clean = s1_all[s1_all["scenario"] == "clean"].copy()
    s1_clean["mae_pp"] = s1_clean["mae"] * 100.0

    # Add B4_B6_mean pooled cohort (16 confirmation cells)
    conf_clean = s1_clean[s1_clean["dataset"].isin(["XJTU_B4", "XJTU_B6"])].copy()
    conf_clean["dataset"] = "B4_B6_mean"
    s1_with_pool = pd.concat([s1_clean, conf_clean], ignore_index=True)

    # Low/High SOH strata from analysis/low_soh_mae_pp.csv
    df_s1_strata = pd.read_csv(ROOT / "analysis" / "low_soh_mae_pp.csv")
    df_s1_strata = df_s1_strata[df_s1_strata["scenario"] == "clean"].copy()

    s1_rows: List[Dict[str, Any]] = []
    cohorts_s1 = ["NASA", "XJTU_B1", "XJTU_B2", "XJTU_B3", "XJTU_B4", "XJTU_B6", "B4_B6_mean"]
    for ds in cohorts_s1:
        sub = s1_with_pool[s1_with_pool["dataset"] == ds]
        base = sub[sub["model"] == "anchor_constant"].set_index("cell")["mae_pp"]
        const_mean = float(base.mean())
        for model, grp in sub.groupby("model"):
            m_series = grp.set_index("cell")["mae_pp"]
            common = base.index.intersection(m_series.index)
            mae_pp = float(m_series.mean())
            wins = int((m_series.loc[common] < base.loc[common]).sum())
            ties = int((m_series.loc[common] == base.loc[common]).sum())
            n_cells = int(len(common))
            skill = 1.0 - mae_pp / const_mean if const_mean > 0 else float("nan")

            # Strata
            if ds != "B4_B6_mean":
                st_sub = df_s1_strata[df_s1_strata["dataset"] == ds]
                st_base = st_sub[st_sub["model"] == "anchor_constant"].set_index("stratum")["mae_pp"]
                st_m = st_sub[st_sub["model"] == model].set_index("stratum")["mae_pp"]
                high_mae = float(st_m.get("high", float("nan")))
                high_const = float(st_base.get("high", float("nan")))
                low_mae = float(st_m.get("low", float("nan")))
                low_const = float(st_base.get("low", float("nan")))
            else:
                # Equal-weight cell strata across B4 and B6? Note low_soh_mae_pp gives cohort stratum means
                st_b4 = df_s1_strata[(df_s1_strata["dataset"] == "XJTU_B4") & (df_s1_strata["model"] == model)].set_index("stratum")["mae_pp"]
                st_b6 = df_s1_strata[(df_s1_strata["dataset"] == "XJTU_B6") & (df_s1_strata["model"] == model)].set_index("stratum")["mae_pp"]
                bc4 = df_s1_strata[(df_s1_strata["dataset"] == "XJTU_B4") & (df_s1_strata["model"] == "anchor_constant")].set_index("stratum")["mae_pp"]
                bc6 = df_s1_strata[(df_s1_strata["dataset"] == "XJTU_B6") & (df_s1_strata["model"] == "anchor_constant")].set_index("stratum")["mae_pp"]
                high_mae = 0.5 * (float(st_b4.get("high", np.nan)) + float(st_b6.get("high", np.nan)))
                high_const = 0.5 * (float(bc4.get("high", np.nan)) + float(bc6.get("high", np.nan)))
                low_mae = float("nan")
                low_const = float("nan")
            high_skill = 1.0 - high_mae / high_const if (math.isfinite(high_mae) and high_const > 0) else float("nan")
            low_skill = 1.0 - low_mae / low_const if (math.isfinite(low_mae) and low_const > 0) else float("nan")

            s1_rows.append({
                "cohort": ds,
                "model": model,
                "n_cells": n_cells,
                "mae_pp": mae_pp,
                "anchor_constant_mae_pp": const_mean,
                "skill_vs_anchor_constant": skill,
                "wins_vs_anchor_constant": wins,
                "ties_vs_anchor_constant": ties,
                "high_soh_mae_pp": high_mae,
                "high_soh_const_mae_pp": high_const,
                "high_soh_skill": high_skill,
                "low_soh_mae_pp": low_mae,
                "low_soh_const_mae_pp": low_const,
                "low_soh_skill": low_skill,
            })

    df_e1_s1 = pd.DataFrame(s1_rows).sort_values(["cohort", "mae_pp"])
    df_e1_s1.to_csv(OUT / "e1_stage1_skill.csv", index=False)

    # Verification assertions for Stage 1
    def _get_s1(c, m):
        r = df_e1_s1[(df_e1_s1["cohort"] == c) & (df_e1_s1["model"] == m)].iloc[0]
        return float(r["mae_pp"])

    assert abs(_get_s1("XJTU_B4", "primary") - 2.5223) < 1e-3, "B4 primary mismatch"
    assert abs(_get_s1("XJTU_B6", "primary") - 17.7914) < 1e-3, "B6 primary mismatch"
    assert abs(_get_s1("B4_B6_mean", "anchor_constant") - 3.5874) < 1e-3, "B4/B6 anchor_constant mismatch"

    # --- Stage 2 ---
    s2_loco = pd.read_csv(ROOT / "stage2" / "loco" / "loco_cell_metrics.csv")
    s2_dev = pd.read_csv(ROOT / "stage2" / "development" / "development_cell_metrics.csv")
    s2_conf = pd.read_csv(ROOT / "stage2" / "confirmation" / "confirmation_cell_metrics.csv")
    s2_all = pd.concat([s2_loco, s2_dev, s2_conf], ignore_index=True)
    s2_clean = s2_all[s2_all["scenario"] == "clean"].copy()
    s2_clean["mae_pp"] = s2_clean["mae"] * 100.0

    df_s2_strata = pd.read_csv(ROOT / "analysis_stage2" / "low_soh_mae_pp.csv")
    s2_rows: List[Dict[str, Any]] = []
    for ds in ["XJTU_B1", "XJTU_B2", "XJTU_B3", "XJTU_B4", "XJTU_B5", "XJTU_B6"]:
        sub = s2_clean[s2_clean["dataset"] == ds]
        base = sub[sub["model"] == "carry_last"].set_index("cell")["mae_pp"]
        carry_mean = float(base.mean())
        st_sub = df_s2_strata[df_s2_strata["dataset"] == ds]
        st_base = st_sub[st_sub["model"] == "carry_last"].set_index("stratum")["mae_pp"]
        for model, grp in sub.groupby("model"):
            m_series = grp.set_index("cell")["mae_pp"]
            common = base.index.intersection(m_series.index)
            mae_pp = float(m_series.mean())
            wins = int((m_series.loc[common] < base.loc[common]).sum())
            ties = int((m_series.loc[common] == base.loc[common]).sum())
            n_cells = int(len(common))
            skill = 1.0 - mae_pp / carry_mean if carry_mean > 0 else float("nan")

            st_m = st_sub[st_sub["model"] == model].set_index("stratum")["mae_pp"]
            high_mae = float(st_m.get("above_0.85", np.nan))
            high_carry = float(st_base.get("above_0.85", np.nan))
            low_mae = float(st_m.get("low_soh", np.nan))
            low_carry = float(st_base.get("low_soh", np.nan))
            high_skill = 1.0 - high_mae / high_carry if (math.isfinite(high_mae) and high_carry > 0) else float("nan")
            low_skill = 1.0 - low_mae / low_carry if (math.isfinite(low_mae) and low_carry > 0) else float("nan")

            s2_rows.append({
                "cohort": ds,
                "model": model,
                "n_cells": n_cells,
                "mae_pp": mae_pp,
                "carry_last_mae_pp": carry_mean,
                "skill_vs_carry_last": skill,
                "wins_vs_carry_last": wins,
                "ties_vs_carry_last": ties,
                "high_soh_mae_pp": high_mae,
                "high_soh_carry_mae_pp": high_carry,
                "high_soh_skill": high_skill,
                "low_soh_mae_pp": low_mae,
                "low_soh_carry_mae_pp": low_carry,
                "low_soh_skill": low_skill,
            })

    df_e1_s2 = pd.DataFrame(s2_rows).sort_values(["cohort", "mae_pp"])
    df_e1_s2.to_csv(OUT / "e1_stage2_skill.csv", index=False)

    def _get_s2(c, m):
        r = df_e1_s2[(df_e1_s2["cohort"] == c) & (df_e1_s2["model"] == m)].iloc[0]
        return float(r["mae_pp"])

    assert abs(_get_s2("XJTU_B5", "primary") - 5.3929) < 1e-3, "B5 primary mismatch"
    assert abs(_get_s2("XJTU_B5", "carry_last") - 6.0164) < 1e-3, "B5 carry_last mismatch"

    payload = {
        "stage1_b4_b6_mean": df_e1_s1[df_e1_s1["cohort"] == "B4_B6_mean"].to_dict(orient="records"),
        "stage1_b4": df_e1_s1[df_e1_s1["cohort"] == "XJTU_B4"].to_dict(orient="records"),
        "stage1_b6": df_e1_s1[df_e1_s1["cohort"] == "XJTU_B6"].to_dict(orient="records"),
        "stage2_b5": df_e1_s2[df_e1_s2["cohort"] == "XJTU_B5"].to_dict(orient="records"),
    }
    save_json(OUT / "e1_skill_scores.json", payload)
    print("[E1] Done. Verified exact match to locked baselines.", flush=True)
    return payload


# =============================================================================
# E2 (M2): Extended Hyperparameter Grids & Boundary-Optimum Audit
# =============================================================================
def run_e2(datasets: Dict[str, List[ref.Cell]]) -> Dict[str, Any]:
    print("[E2] Running extended Stage-2 & Stage-1 hyperparameter grids...", flush=True)
    b1 = sorted(datasets["XJTU_B1"], key=lambda c: c.name)
    gk = GroupKFold(n_splits=3)
    folds = list(gk.split(np.arange(len(b1)), groups=np.arange(len(b1))))

    # --- Stage 2 Extended Grid ---
    LAMBDAS_S2 = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    TG_EXT = [0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 10000.0, 1e6]
    STRIDES = [1, 6, 11]

    # Cache fit_stats per (fold_idx, lam) and full B1 stats per lam
    fold_stats = {}
    for fi, (tr_i, _) in enumerate(folds):
        tr = [b1[i] for i in tr_i]
        for lam in LAMBDAS_S2:
            fold_stats[(fi, lam)] = s2.fit_stats(tr, lam)
    full_stats = {lam: s2.fit_stats(b1, lam) for lam in LAMBDAS_S2}

    s2_cv_rows: List[Dict[str, Any]] = []
    for kind in ["primary", "isotropic_map", "scalar_map", "source_logridge"]:
        t_list = [np.nan] if kind == "source_logridge" else TG_EXT
        for lam in LAMBDAS_S2:
            for t in t_list:
                scores_by_stride = {s: [] for s in STRIDES}
                cand = {"lambda": float(lam)} if kind == "source_logridge" else {"lambda": float(lam), "t": float(t)}
                for fi, (_, va_i) in enumerate(folds):
                    st = fold_stats[(fi, lam)]
                    for vi in va_i:
                        vc = b1[vi]
                        for stride in STRIDES:
                            out = s2.predict(kind, st, cand, vc, "clean", stride)
                            ytrue = np.asarray(vc.y, float)[out["future_positions"]] / float(vc.ref_y)
                            mae = float(np.mean(np.abs(ytrue - out["pred"])))
                            scores_by_stride[stride].append(mae)
                all_scores = [x for s in STRIDES for x in scores_by_stride[s]]
                s2_cv_rows.append({
                    "model": kind,
                    "lambda": float(lam),
                    "t": float(t) if not math.isnan(t) else None,
                    "cv_mae_pp": 100.0 * float(np.mean(all_scores)),
                    "cv_stride1_pp": 100.0 * float(np.mean(scores_by_stride[1])),
                    "cv_stride6_pp": 100.0 * float(np.mean(scores_by_stride[6])),
                    "cv_stride11_pp": 100.0 * float(np.mean(scores_by_stride[11])),
                    "in_orig_grid": (lam in s2.LAMBDAS) and (kind == "source_logridge" or t in s2.TG),
                })

    df_s2_cv = pd.DataFrame(s2_cv_rows)
    df_s2_cv.to_csv(OUT / "e2_stage2_cv_curve.csv", index=False)

    # Identify original vs extended optima & verify B5 at (1e-4, 10)
    def _eval_s2_target(kind: str, lam: float, t: float | None, ds_name: str) -> Dict[str, Any]:
        st = full_stats[lam]
        beta0 = np.asarray(st["beta0"], float)
        Sigma = np.asarray(st["Sigma"], float)
        sigma2 = float(st["sigma2"])
        cand = {"lambda": float(lam)} if kind == "source_logridge" else {"lambda": float(lam), "t": float(t)}
        maes, q_nus, db_norms = [], [], []
        for c in datasets[ds_name]:
            out = s2.predict(kind, st, cand, c, "clean", 1)
            ytrue = np.asarray(c.y, float)[out["future_positions"]] / float(c.ref_y)
            maes.append(100.0 * float(np.mean(np.abs(ytrue - out["pred"]))))
            beta = np.asarray(out["metadata"]["beta"], float)
            db_norms.append(float(np.linalg.norm(beta - beta0)))
            if t is not None:
                cal_pos = int(out["metadata"]["cal_position"])
                Xtag = np.log(np.asarray(c.D["clean"], float) / np.asarray(c.Dref["clean"], float)[None, :])
                xa = Xtag[cal_pos]
                q_nus.append(float(xa @ Sigma @ xa) / (float(t) * sigma2))
        return {
            "mae_pp": float(np.mean(maes)),
            "mean_q_over_nu": float(np.mean(q_nus)) if q_nus else None,
            "mean_delta_beta_norm": float(np.mean(db_norms)),
        }

    assert abs(_eval_s2_target("primary", 1e-4, 10.0, "XJTU_B5")["mae_pp"] - 5.392867055) < 1e-4
    assert abs(_eval_s2_target("isotropic_map", 1e-4, 10.0, "XJTU_B5")["mae_pp"] - 5.521722769) < 1e-4
    assert abs(_eval_s2_target("scalar_map", 1e-4, 10.0, "XJTU_B5")["mae_pp"] - 5.221428704) < 1e-4
    assert abs(_eval_s2_target("source_logridge", 1e-4, None, "XJTU_B5")["mae_pp"] - 5.370803211) < 1e-4

    s2_optima = {}
    for kind in ["primary", "isotropic_map", "scalar_map", "source_logridge"]:
        sub_orig = df_s2_cv[(df_s2_cv["model"] == kind) & (df_s2_cv["in_orig_grid"])].sort_values("cv_mae_pp")
        sub_ext = df_s2_cv[df_s2_cv["model"] == kind].sort_values("cv_mae_pp")
        orig_best = sub_orig.iloc[0].to_dict()
        ext_best = sub_ext.iloc[0].to_dict()
        # Evaluate target cohorts at ext_best
        ext_t = ext_best["t"] if pd.notna(ext_best["t"]) else None
        ext_targets = {ds: _eval_s2_target(kind, float(ext_best["lambda"]), ext_t, ds)
                       for ds in ["XJTU_B1", "XJTU_B2", "XJTU_B3", "XJTU_B4", "XJTU_B5", "XJTU_B6"]}
        s2_optima[kind] = {
            "orig_best": orig_best,
            "ext_best": ext_best,
            "ext_targets": ext_targets,
        }

    # Sweep t across TG_EXT at lam=1e-4 for primary and scalar_map on B1..B6
    s2_t_sweep = []
    for kind in ["primary", "isotropic_map", "scalar_map"]:
        for t in TG_EXT:
            cv_row = df_s2_cv[(df_s2_cv["model"] == kind) & (df_s2_cv["lambda"] == 1e-4) & (df_s2_cv["t"] == t)].iloc[0]
            rec = {"model": kind, "lambda": 1e-4, "t": t, "cv_mae_pp": float(cv_row["cv_mae_pp"])}
            for ds in ["XJTU_B2", "XJTU_B3", "XJTU_B4", "XJTU_B5", "XJTU_B6"]:
                ev = _eval_s2_target(kind, 1e-4, t, ds)
                rec[f"{ds}_mae_pp"] = ev["mae_pp"]
                if ds == "XJTU_B5":
                    rec["B5_mean_q_over_nu"] = ev["mean_q_over_nu"]
                    rec["B5_mean_delta_beta_norm"] = ev["mean_delta_beta_norm"]
            s2_t_sweep.append(rec)

    # --- Stage 1 Extended Grid ---
    LAMBDAS_S1 = [1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    GAMMAS_S1 = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
    ETAS_S1 = [0.0, 0.1, 1.0, 10.0]
    SCEN_S1 = ["clean", "vp5", "vm5"]

    # Match reference_models.select cell order (ref.load_cells order, which is sorted by name anyway)
    b1_s1 = datasets["XJTU_B1"]
    gk1 = GroupKFold(n_splits=3)
    folds_s1 = list(gk1.split(np.arange(len(b1_s1)), groups=np.arange(len(b1_s1))))

    s1_cv_rows: List[Dict[str, Any]] = []
    for lam in LAMBDAS_S1:
        for gam in GAMMAS_S1:
            for eta in ETAS_S1:
                params = {"lambda": float(lam), "gamma": float(gam), "eta": float(eta)}
                scores = {sc: [] for sc in SCEN_S1}
                failures = 0
                for tr_idx, va_idx in folds_s1:
                    tr_cells = [b1_s1[i] for i in tr_idx]
                    try:
                        m = ref.fit("primary", params, tr_cells)
                    except (RuntimeError, ValueError):
                        failures += len(va_idx)
                        continue
                    for vi in va_idx:
                        vc = b1_s1[vi]
                        rows = ref._rows_for_predict(vc)
                        yt = np.asarray(vc.y[rows], dtype=float)
                        for sc in SCEN_S1:
                            yp = ref.predict(m, vc, sc)
                            scores[sc].append(float(np.mean(np.abs(yt - yp))))
                if failures == 0 and len(scores["clean"]) == len(b1_s1):
                    c_pp = 100.0 * float(np.mean(scores["clean"]))
                    vp_pp = 100.0 * float(np.mean(scores["vp5"]))
                    vm_pp = 100.0 * float(np.mean(scores["vm5"]))
                    rob_pp = max(c_pp, vp_pp, vm_pp)
                else:
                    c_pp = vp_pp = vm_pp = rob_pp = float("inf")
                s1_cv_rows.append({
                    "model": "primary",
                    "lambda": float(lam),
                    "gamma": float(gam),
                    "eta": float(eta),
                    "cv_clean_pp": c_pp,
                    "cv_vp5_pp": vp_pp,
                    "cv_vm5_pp": vm_pp,
                    "cv_robust_pp": rob_pp,
                    "failures": failures,
                    "in_orig_grid": (lam in [1e-6, 1e-4, 0.01, 0.1]) and (gam in [0.0, 1.0, 10.0]) and (eta in [0.0, 1.0, 10.0]),
                })

    for lam in LAMBDAS_S1:
        params = {"lambda": float(lam)}
        scores = {sc: [] for sc in SCEN_S1}
        for tr_idx, va_idx in folds_s1:
            tr_cells = [b1_s1[i] for i in tr_idx]
            m = ref.fit("logridge", params, tr_cells)
            for vi in va_idx:
                vc = b1_s1[vi]
                rows = ref._rows_for_predict(vc)
                yt = np.asarray(vc.y[rows], dtype=float)
                for sc in SCEN_S1:
                    yp = ref.predict(m, vc, sc)
                    scores[sc].append(float(np.mean(np.abs(yt - yp))))
        c_pp = 100.0 * float(np.mean(scores["clean"]))
        vp_pp = 100.0 * float(np.mean(scores["vp5"]))
        vm_pp = 100.0 * float(np.mean(scores["vm5"]))
        s1_cv_rows.append({
            "model": "logridge",
            "lambda": float(lam),
            "gamma": None,
            "eta": None,
            "cv_clean_pp": c_pp,
            "cv_vp5_pp": vp_pp,
            "cv_vm5_pp": vm_pp,
            "cv_robust_pp": max(c_pp, vp_pp, vm_pp),
            "failures": 0,
            "in_orig_grid": lam in ref.LAMBDAS,
        })

    df_s1_cv = pd.DataFrame(s1_cv_rows)
    df_s1_cv.to_csv(OUT / "e2_stage1_cv_curve.csv", index=False)

    # Check Stage 1 original vs extended optima and weight/B4/B6 sensitivity below 1e-6
    def _eval_s1(kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
        m = ref.fit(kind, params, b1_s1)
        w = np.asarray(m.get("w", m.get("beta")), float)
        res = {"w": w.tolist()}
        for ds in ["XJTU_B4", "XJTU_B6"]:
            maes = []
            for c in datasets[ds]:
                rows = ref._rows_for_predict(c)
                yp = ref.predict(m, c, "clean")
                maes.append(100.0 * float(np.mean(np.abs(c.y[rows] - yp))))
            res[f"{ds}_mae_pp"] = float(np.mean(maes))
        return res

    s1_orig_eval = _eval_s1("primary", {"lambda": 1e-6, "gamma": 1.0, "eta": 0.0})
    assert abs(s1_orig_eval["XJTU_B4_mae_pp"] - 2.522337) < 1e-3
    assert abs(s1_orig_eval["XJTU_B6_mae_pp"] - 17.79141) < 1e-3

    s1_tiny_eval = _eval_s1("primary", {"lambda": 1e-12, "gamma": 1.0, "eta": 0.0})
    s1_lam_diff = float(np.linalg.norm(np.array(s1_tiny_eval["w"]) - np.array(s1_orig_eval["w"])))

    df_s1_pri_ext = df_s1_cv[(df_s1_cv["model"] == "primary") & np.isfinite(df_s1_cv["cv_robust_pp"])].sort_values("cv_robust_pp")
    df_s1_pri_orig = df_s1_pri_ext[df_s1_pri_ext["in_orig_grid"]]
    s1_ext_best_row = df_s1_pri_ext.iloc[0].to_dict()
    s1_orig_best_row = df_s1_pri_orig.iloc[0].to_dict()
    s1_ext_eval = _eval_s1("primary", {"lambda": s1_ext_best_row["lambda"], "gamma": s1_ext_best_row["gamma"], "eta": s1_ext_best_row["eta"]})

    df_s1_lr_ext = df_s1_cv[df_s1_cv["model"] == "logridge"].sort_values("cv_robust_pp")
    s1_lr_orig = df_s1_lr_ext[df_s1_lr_ext["in_orig_grid"]].iloc[0].to_dict()
    s1_lr_ext = df_s1_lr_ext.iloc[0].to_dict()

    payload = {
        "stage2_optima": s2_optima,
        "stage2_t_sweep_at_lam1e4": s2_t_sweep,
        "stage1_primary_orig_best": s1_orig_best_row,
        "stage1_primary_orig_eval": s1_orig_eval,
        "stage1_primary_lam1e12_at_gamma1_eta0": s1_tiny_eval,
        "stage1_primary_w_diff_lam1e12_vs_1e6": s1_lam_diff,
        "stage1_primary_ext_best": s1_ext_best_row,
        "stage1_primary_ext_eval": s1_ext_eval,
        "stage1_logridge_orig_best": s1_lr_orig,
        "stage1_logridge_ext_best": s1_lr_ext,
    }
    save_json(OUT / "e2_extended_grids.json", payload)
    print("[E2] Done.", flush=True)
    return payload


# =============================================================================
# E3a (M3 part 1): Cohort-Wide NPZ ICA (dQ/dV) & Segment Evolution
# =============================================================================
def run_e3a(datasets: Dict[str, List[ref.Cell]]) -> Dict[str, Any]:
    print("[E3a] Computing NPZ-level ICA (dQ/dV) and segment evolution across B1..B4, B6...", flush=True)
    npz_map = {
        "XJTU_B1": OLD_DATA / "XJTU_B1.npz",
        "XJTU_B2": OLD_DATA / "XJTU_B2.npz",
        "XJTU_B3": OLD_DATA / "XJTU_B3.npz",
        "XJTU_B4": NEW_DATA / "XJTU_B4.npz",
        "XJTU_B6": NEW_DATA / "XJTU_B6.npz",
    }

    curve_rows: List[Dict[str, Any]] = []
    cohort_summaries: Dict[str, Any] = {}

    for ds, npz_p in npz_map.items():
        cells = datasets[ds]
        with np.load(npz_p, allow_pickle=False) as z:
            raw_name = np.asarray(z["cell_name"]).astype(str)
            raw_cyc = np.asarray(z["cycle"], float)
            raw_soh = np.asarray(z["soh"], float)
            raw_qv = np.asarray(z["qv"], float)
            vgrid = np.asarray(z["vgrid"], float)  # 70 pts 3.50..4.19 step 0.01

        cell_stats = []
        early_dqdv_list, late_dqdv_list, soh90_dqdv_list = [], [], []
        idx_355 = int(np.argmin(np.abs(vgrid - 3.55)))
        idx_380 = int(np.argmin(np.abs(vgrid - 3.80)))
        idx_3877 = int(np.argmin(np.abs(vgrid - 3.877)))
        idx_410 = int(np.argmin(np.abs(vgrid - 4.10)))
        idx_415 = int(np.argmin(np.abs(vgrid - 4.15)))

        def _compute_dqdv(qv_row: np.ndarray) -> np.ndarray:
            out = np.full_like(qv_row, np.nan, dtype=float)
            m = np.isfinite(qv_row)
            idx = np.where(m)[0]
            if idx.size < 7:
                return out
            # Contiguous block
            i0, i1 = idx[0], idx[-1]
            if not np.all(m[i0:i1 + 1]):
                return out
            sub = qv_row[i0:i1 + 1]
            deriv = savgol_filter(sub, window_length=5, polyorder=2, deriv=1, delta=0.01)
            # Mark boundary 2 pts unreliable/central
            out[i0:i1 + 1] = deriv
            out[i0] = np.nan
            out[i1] = np.nan
            return out

        def _extract_metrics(qv_row: np.ndarray, dqdv: np.ndarray, d_segs: np.ndarray, soh_val: float) -> Dict[str, Any]:
            # Peak in [3.55, 4.10] V where finite
            sl = slice(idx_355, idx_410 + 1)
            sub_v = vgrid[sl]
            sub_dq = dqdv[sl]
            valid = np.isfinite(sub_dq)
            if not np.any(valid):
                return {"valid": False}
            pk_i = int(np.nanargmax(sub_dq))
            v_pk = float(sub_v[pk_i])
            h_pk = float(sub_dq[pk_i])

            # Partial-charge capacities in voltage bands (using trapezoidal sum of positive dQ/dV or direct qv diffs where finite)
            # Direct qv diffs are exact where finite; handle left coverage limit if curve starts > 3.55
            q_355_380 = float(qv_row[idx_380] - qv_row[idx_355]) if np.isfinite(qv_row[idx_355]) and np.isfinite(qv_row[idx_380]) else float("nan")
            q_total_355_410 = float(qv_row[idx_410] - qv_row[idx_355]) if np.isfinite(qv_row[idx_355]) and np.isfinite(qv_row[idx_410]) else float("nan")
            d0 = float(d_segs[0])
            d123 = float(np.sum(d_segs[1:]))
            d_sum = float(np.sum(d_segs))
            return {
                "valid": True,
                "soh": float(soh_val),
                "v_peak": v_pk,
                "dqdv_peak": h_pk,
                "q_355_380": q_355_380,
                "q_total_355_410": q_total_355_410,
                "frac_below_380": q_355_380 / q_total_355_410 if (math.isfinite(q_355_380) and q_total_355_410 > 0) else float("nan"),
                "frac_D0_of_355_410": d0 / q_total_355_410 if (math.isfinite(q_total_355_410) and q_total_355_410 > 0) else float("nan"),
                "frac_D0_of_W6": d0 / d_sum if d_sum > 0 else float("nan"),
                "D": d_segs.tolist(),
                "D_sum": d_sum,
            }

        for c in cells:
            # Eligible future rows from c.row_index plus anchor c.ref_row
            rows_fut = ref._rows_for_predict(c)
            if rows_fut.size == 0:
                continue
            r_early = int(c.ref_row)
            r_late = int(c.row_index[rows_fut[-1]])
            # Find row closest to SOH=0.90 among eligible future rows
            pos_90 = int(rows_fut[np.argmin(np.abs(c.y[rows_fut] - 0.90))])
            r_90 = int(c.row_index[pos_90])

            dq_early = _compute_dqdv(raw_qv[r_early])
            dq_late = _compute_dqdv(raw_qv[r_late])
            dq_90 = _compute_dqdv(raw_qv[r_90])

            early_dqdv_list.append(dq_early)
            late_dqdv_list.append(dq_late)
            soh90_dqdv_list.append(dq_90)

            m_early = _extract_metrics(raw_qv[r_early], dq_early, np.asarray(c.Dref["clean"], float), float(c.ref_y))
            m_late = _extract_metrics(raw_qv[r_late], dq_late, np.asarray(c.D["clean"][rows_fut[-1]], float), float(c.y[rows_fut[-1]]))
            m_90 = _extract_metrics(raw_qv[r_90], dq_90, np.asarray(c.D["clean"][pos_90], float), float(c.y[pos_90]))

            d_early = np.asarray(m_early["D"], float)
            d_late = np.asarray(m_late["D"], float)
            d_90 = np.asarray(m_90["D"], float)

            cell_stats.append({
                "cell": c.name,
                "early_soh": m_early["soh"],
                "late_soh": m_late["soh"],
                "soh90_actual": m_90["soh"],
                "early_v_peak": m_early["v_peak"],
                "late_v_peak": m_late["v_peak"],
                "soh90_v_peak": m_90["v_peak"],
                "delta_v_peak_late_minus_early_mV": 1000.0 * (m_late["v_peak"] - m_early["v_peak"]),
                "delta_v_peak_90_minus_early_mV": 1000.0 * (m_90["v_peak"] - m_early["v_peak"]),
                "early_dqdv_peak": m_early["dqdv_peak"],
                "late_dqdv_peak": m_late["dqdv_peak"],
                "early_q_355_380": m_early["q_355_380"],
                "late_q_355_380": m_late["q_355_380"],
                "early_frac_below_380": m_early["frac_below_380"],
                "late_frac_below_380": m_late["frac_below_380"],
                "D_early": d_early.tolist(),
                "D_late": d_late.tolist(),
                "D_90": d_90.tolist(),
                "pct_change_D_early_to_late": (100.0 * (d_late / d_early - 1.0)).tolist(),
                "pct_change_D_early_to_90": (100.0 * (d_90 / d_early - 1.0)).tolist(),
                "log_ratio_early_to_late": np.log(d_late / d_early).tolist(),
                "log_ratio_early_to_90": np.log(d_90 / d_early).tolist(),
            })

        # Cohort mean curves over [3.55, 4.15] V
        mat_e = np.vstack(early_dqdv_list)
        mat_l = np.vstack(late_dqdv_list)
        mat_90 = np.vstack(soh90_dqdv_list)
        for vi in range(idx_355, idx_415 + 1):
            curve_rows.append({
                "dataset": ds,
                "V": round(float(vgrid[vi]), 4),
                "early_dqdv_mean": float(np.nanmean(mat_e[:, vi])) if np.any(np.isfinite(mat_e[:, vi])) else np.nan,
                "early_dqdv_sd": float(np.nanstd(mat_e[:, vi])) if np.any(np.isfinite(mat_e[:, vi])) else np.nan,
                "soh90_dqdv_mean": float(np.nanmean(mat_90[:, vi])) if np.any(np.isfinite(mat_90[:, vi])) else np.nan,
                "late_dqdv_mean": float(np.nanmean(mat_l[:, vi])) if np.any(np.isfinite(mat_l[:, vi])) else np.nan,
                "late_dqdv_sd": float(np.nanstd(mat_l[:, vi])) if np.any(np.isfinite(mat_l[:, vi])) else np.nan,
            })

        # Cohort aggregate summary
        df_cs = pd.DataFrame(cell_stats)
        cohort_summaries[ds] = {
            "n_cells": len(cell_stats),
            "early_soh_mean": float(df_cs["early_soh"].mean()),
            "late_soh_mean": float(df_cs["late_soh"].mean()),
            "early_v_peak_mean": float(df_cs["early_v_peak"].mean()),
            "early_v_peak_sd": float(df_cs["early_v_peak"].std(ddof=0)),
            "late_v_peak_mean": float(df_cs["late_v_peak"].mean()),
            "late_v_peak_sd": float(df_cs["late_v_peak"].std(ddof=0)),
            "delta_v_peak_late_minus_early_mV_mean": float(df_cs["delta_v_peak_late_minus_early_mV"].mean()),
            "delta_v_peak_90_minus_early_mV_mean": float(df_cs["delta_v_peak_90_minus_early_mV"].mean()),
            "early_dqdv_peak_mean": float(df_cs["early_dqdv_peak"].mean()),
            "late_dqdv_peak_mean": float(df_cs["late_dqdv_peak"].mean()),
            "early_q_355_380_mean": float(df_cs["early_q_355_380"].mean()),
            "late_q_355_380_mean": float(df_cs["late_q_355_380"].mean()),
            "D_early_mean": np.mean(np.vstack(df_cs["D_early"].tolist()), axis=0).tolist(),
            "D_late_mean": np.mean(np.vstack(df_cs["D_late"].tolist()), axis=0).tolist(),
            "D_90_mean": np.mean(np.vstack(df_cs["D_90"].tolist()), axis=0).tolist(),
            "pct_change_D_early_to_late_mean": np.mean(np.vstack(df_cs["pct_change_D_early_to_late"].tolist()), axis=0).tolist(),
            "pct_change_D_early_to_90_mean": np.mean(np.vstack(df_cs["pct_change_D_early_to_90"].tolist()), axis=0).tolist(),
            "cells": cell_stats,
        }

    pd.DataFrame(curve_rows).to_csv(OUT / "e3a_ica_curves.csv", index=False)
    save_json(OUT / "e3a_npz_ica_summary.json", cohort_summaries)
    print("[E3a] Done.", flush=True)
    return cohort_summaries


# =============================================================================
# E4 (M5c): Matched-Scale & CV-Tuned Prior Variance for scalar_map
# =============================================================================
def run_e4(datasets: Dict[str, List[ref.Cell]], e2_payload: Dict[str, Any]) -> Dict[str, Any]:
    print("[E4] Running scalar_map prior-scale matching experiment...", flush=True)
    b1 = sorted(datasets["XJTU_B1"], key=lambda c: c.name)
    st = s2.fit_stats(b1, 1e-4)
    beta0 = np.asarray(st["beta0"], float)
    Sigma = np.asarray(st["Sigma"], float)
    sigma2 = float(st["sigma2"])

    tr_Sigma = float(np.trace(Sigma))
    b0_norm2 = float(beta0 @ beta0)
    b0_Sigma_b0 = float(beta0 @ Sigma @ beta0)

    tau_trace = tr_Sigma / b0_norm2
    tau_ray = b0_Sigma_b0 / (b0_norm2 ** 2)

    # Check CV-optimal t_eff for scalar_map from E2 at lambda=1e-4 and overall
    df_s2_cv = pd.read_csv(OUT / "e2_stage2_cv_curve.csv")
    sc_cv_lam1e4 = df_s2_cv[(df_s2_cv["model"] == "scalar_map") & (df_s2_cv["lambda"] == 1e-4)].sort_values("cv_mae_pp").iloc[0].to_dict()
    sc_cv_ext = e2_payload["stage2_optima"]["scalar_map"]["ext_best"]

    t_nom = 10.0
    variants = [
        ("primary", 1e-4, t_nom, None),
        ("source_logridge", 1e-4, None, None),
        ("scalar_map_orig", 1e-4, t_nom, 1.0),
        ("scalar_map_trace", 1e-4, t_nom, tau_trace),
        ("scalar_map_rayleigh", 1e-4, t_nom, tau_ray),
        ("scalar_map_cv_lam1e4", 1e-4, float(sc_cv_lam1e4["t"]), 1.0),
        ("scalar_map_cv_ext", float(sc_cv_ext["lambda"]), float(sc_cv_ext["t"]), 1.0),
    ]

    stats_cache = {1e-4: st}
    if float(sc_cv_ext["lambda"]) not in stats_cache:
        stats_cache[float(sc_cv_ext["lambda"])] = s2.fit_stats(b1, float(sc_cv_ext["lambda"]))

    rows: List[Dict[str, Any]] = []
    cohorts = ["XJTU_B1", "XJTU_B2", "XJTU_B3", "XJTU_B4", "XJTU_B5", "XJTU_B6"]
    for name, lam, t_val, tau_a in variants:
        sti = stats_cache[lam]
        t_eff = (t_val / tau_a) if (t_val is not None and tau_a is not None) else t_val
        b0 = np.asarray(sti["beta0"], float)
        Sig = np.asarray(sti["Sigma"], float)
        sig2 = float(sti["sigma2"])

        for ds in cohorts:
            maes, a_vals, lev_vals = [], [], []
            for c in datasets[ds]:
                if name == "primary":
                    out = s2.predict("primary", sti, {"lambda": lam, "t": t_eff}, c, "clean", 1)
                    cal_pos = int(out["metadata"]["cal_position"])
                    xa = np.log(np.asarray(c.D["clean"], float)[cal_pos] / np.asarray(c.Dref["clean"], float))
                    lev_vals.append(float(xa @ Sig @ xa) / (t_eff * sig2))
                elif name == "source_logridge":
                    out = s2.predict("source_logridge", sti, {"lambda": lam}, c, "clean", 1)
                else:
                    out = s2.predict("scalar_map", sti, {"lambda": lam, "t": t_eff}, c, "clean", 1)
                    cal_pos = int(out["metadata"]["cal_position"])
                    xa = np.log(np.asarray(c.D["clean"], float)[cal_pos] / np.asarray(c.Dref["clean"], float))
                    f = float(xa @ b0)
                    za = math.log(float(c.y[cal_pos]) / float(c.ref_y))
                    a_factor = 1.0 + f * (za - f) / (t_eff * sig2 + f * f)
                    a_vals.append(a_factor)
                    lev_vals.append((f * f) / (t_eff * sig2))
                ytrue = np.asarray(c.y, float)[out["future_positions"]] / float(c.ref_y)
                maes.append(100.0 * float(np.mean(np.abs(ytrue - out["pred"]))))
            rows.append({
                "variant": name,
                "lambda": lam,
                "t": t_val,
                "tau_a": tau_a,
                "t_effective": t_eff,
                "cohort": ds,
                "clean_mae_pp": float(np.mean(maes)),
                "mean_a_factor": float(np.mean(a_vals)) if a_vals else None,
                "mean_leverage": float(np.mean(lev_vals)) if lev_vals else None,
            })

    df_e4 = pd.DataFrame(rows)
    df_e4.to_csv(OUT / "e4_scalar_prior_matching.csv", index=False)

    payload = {
        "prior_scales_at_lam1e4": {
            "tr_Sigma": tr_Sigma,
            "beta0_norm2_sq": b0_norm2,
            "beta0_Sigma_beta0": b0_Sigma_b0,
            "tau_trace": tau_trace,
            "tau_rayleigh": tau_ray,
            "t_eff_trace_at_t10": 10.0 / tau_trace,
            "t_eff_rayleigh_at_t10": 10.0 / tau_ray,
        },
        "pivot_clean_mae_pp": df_e4.pivot(index="variant", columns="cohort", values="clean_mae_pp").to_dict(orient="index"),
        "pivot_mean_leverage": df_e4.pivot(index="variant", columns="cohort", values="mean_leverage").to_dict(orient="index"),
    }
    save_json(OUT / "e4_scalar_prior_matching.json", payload)
    print("[E4] Done.", flush=True)
    return payload


# =============================================================================
# E5 (M6): B5 Anchor-Index Sensitivity & Exact Nonparametric Cell Tests
# =============================================================================
def _predict_custom_anchor(kind: str, stats: Dict[str, Any], params: Dict[str, Any],
                           cell: ref.Cell, cal_full: int, fut_full_start: int | None = None) -> Dict[str, Any]:
    Xtag = np.log(np.asarray(cell.D["clean"], float) / np.asarray(cell.Dref["clean"], float)[None, :])
    nfull = Xtag.shape[0] + 1
    cal_pos = cal_full - 1
    start_full = (cal_full + 1) if fut_full_start is None else fut_full_start
    fut_full = np.arange(start_full, nfull, dtype=int)
    fut_pos = fut_full - 1
    if cal_pos < 0 or cal_pos >= Xtag.shape[0] or fut_pos.size == 0:
        raise ValueError(f"{cell.name}: invalid cal_full={cal_full}, start={start_full}, nfull={nfull}")
    xa = Xtag[cal_pos].copy()
    za = math.log(float(cell.y[cal_pos]) / float(cell.ref_y))
    Xf = Xtag[fut_pos].copy()
    beta0 = np.asarray(stats["beta0"], float)
    Sigma = np.asarray(stats["Sigma"], float)
    sigma2 = float(stats["sigma2"])
    H = np.asarray(stats["H"], float)
    h = np.asarray(stats["h"], float)
    lam = float(params.get("lambda", stats["lambda"]))
    t = float(params.get("t", 10.0))
    segment = int(params.get("segment", 1))

    if kind == "primary":
        den = t * sigma2 + float(xa @ Sigma @ xa)
        beta = beta0 + Sigma @ xa * (za - float(xa @ beta0)) / den
        pred = np.exp(Xf @ beta)
    elif kind == "isotropic_map":
        iso = (float(np.trace(Sigma)) / 4.0) * np.eye(4)
        den = t * sigma2 + float(xa @ iso @ xa)
        beta = beta0 + iso @ xa * (za - float(xa @ beta0)) / den
        pred = np.exp(Xf @ beta)
    elif kind == "scalar_map":
        f = float(xa @ beta0)
        a = 1.0 + f * (za - f) / (t * sigma2 + f * f)
        pred = np.exp(Xf @ (a * beta0))
    elif kind == "target_secant":
        nrm = float(xa @ xa)
        pred = np.full(fut_pos.shape, math.exp(za)) if nrm <= 1e-10 else np.exp(Xf @ (xa * za / nrm))
    elif kind == "endpoint_secant":
        Dc = np.asarray(cell.D["clean"], float)[cal_pos]
        Dr = np.asarray(cell.Dref["clean"], float)
        ua = float(np.log(np.sum(Dc) / np.sum(Dr)))
        if abs(ua) <= 1e-8:
            pred = np.full(fut_pos.shape, math.exp(za))
        else:
            pcoef = za / ua
            Du = np.asarray(cell.D["clean"], float)[fut_pos]
            u_t = np.log(np.sum(Du, axis=1) / float(np.sum(Dr)))
            pred = np.exp(pcoef * u_t)
    elif kind == "source_logridge":
        beta = np.linalg.solve(H + lam * np.eye(4), h)
        pred = np.exp(Xf @ beta)
    elif kind == "carry_last":
        pred = np.full(fut_pos.shape, math.exp(za))
    elif kind == "covariance_exact":
        den = float(xa @ Sigma @ xa)
        pred = np.full(fut_pos.shape, math.exp(za)) if den <= 1e-10 else np.exp(Xf @ (beta0 + Sigma @ xa * (za - float(xa @ beta0)) / den))
    elif kind == "best_segment":
        xaj = float(xa[segment])
        if abs(xaj) <= 1e-8:
            pred = np.full(fut_pos.shape, math.exp(za))
        else:
            beta = np.zeros(4)
            beta[segment] = za / xaj
            pred = np.exp(Xf @ beta)
    else:
        raise ValueError(kind)

    ytrue = np.asarray(cell.y, float)[fut_pos] / float(cell.ref_y)
    return {
        "mae_pp": 100.0 * float(np.mean(np.abs(ytrue - pred))),
        "n_fut": int(fut_pos.size),
        "cal_rel_soh": float(cell.y[cal_pos] / cell.ref_y),
        "cal_cycle": float(cell.cycles[cal_pos]),
    }


def _exact_tests(d_vec: np.ndarray) -> Dict[str, Any]:
    """d_vec = primary - comparator (negative favors primary)."""
    n = d_vec.size
    mean_diff = float(np.mean(d_vec))
    wins_pri = int(np.sum(d_vec < 0.0))
    wins_cmp = int(np.sum(d_vec > 0.0))
    # Exact sign test on non-zero differences
    nz = d_vec[d_vec != 0.0]
    p_sign = float(binomtest(int(np.sum(nz < 0.0)), int(nz.size), 0.5, alternative="two-sided").pvalue) if nz.size > 0 else 1.0
    # Exact 2^n paired sign-flip permutation test
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)), dtype=float)  # 256 x 8
    null_means = np.abs(signs @ d_vec) / float(n)
    p_perm = float(np.mean(null_means >= abs(mean_diff) - 1e-15))
    # Exact Wilcoxon signed-rank test
    p_wilcoxon = float(wilcoxon(d_vec, method="exact", alternative="two-sided").pvalue) if nz.size > 0 else 1.0
    return {
        "n": int(n),
        "mean_diff_pri_minus_cmp_pp": mean_diff,
        "wins_primary": wins_pri,
        "wins_comparator": wins_cmp,
        "p_sign_exact": p_sign,
        "p_perm_exact": p_perm,
        "p_wilcoxon_exact": p_wilcoxon,
    }


def run_e5(datasets: Dict[str, List[ref.Cell]]) -> Dict[str, Any]:
    print("[E5] Running B5 anchor-index sensitivity & exact cell tests...", flush=True)
    cfg = json.loads((ROOT / "stage2" / "source" / "config_freeze.json").read_text())
    b5 = datasets["XJTU_B5"]

    # Inspect B5 trajectories & rebound structure
    b5_diag = []
    post_rebound_anchor = {}
    for c in b5:
        nfull = c.y.size + 1
        rel = np.concatenate([[1.0], np.asarray(c.y, float) / float(c.ref_y)])
        cycs = np.concatenate([[float(c.ref_cycle)], np.asarray(c.cycles, float)])
        peak_idx = int(np.argmax(rel))
        peak_soh = float(rel[peak_idx])
        # First full index >= 1 strictly below 1.0 (post-rebound) with at least 3 future points
        pr_candidates = [i for i in range(1, nfull - 3) if rel[i] < 1.0]
        pr_idx = int(pr_candidates[0]) if pr_candidates else int(min(4, nfull - 4))
        post_rebound_anchor[c.name] = pr_idx
        b5_diag.append({
            "cell": c.name,
            "nfull": int(nfull),
            "ref_cycle": float(c.ref_cycle),
            "ref_y": float(c.ref_y),
            "rebound_peak_rel_soh": peak_soh,
            "rebound_peak_full_idx": peak_idx,
            "rebound_peak_cycle": float(cycs[peak_idx]),
            "first_below_ref_full_idx": int(pr_candidates[0]) if pr_candidates else None,
            "post_rebound_anchor_used": pr_idx,
            "post_rebound_anchor_cycle": float(cycs[pr_idx]),
            "post_rebound_anchor_rel_soh": float(rel[pr_idx]),
            "cal4_cycle": float(cycs[4]),
            "cal4_rel_soh": float(rel[4]),
            "last_rel_soh": float(rel[-1]),
        })

    models = ["primary", "isotropic_map", "scalar_map", "source_logridge",
              "target_secant", "endpoint_secant", "carry_last", "covariance_exact", "best_segment"]
    sweep_rows: List[Dict[str, Any]] = []
    exact_rows: List[Dict[str, Any]] = []

    schemes: List[Tuple[str, Any, int | None]] = (
        [("suffix", int(a), None) for a in [1, 2, 3, 4, 5, 6, 8, 10]]
        + [("suffix", "post_rebound", None)]
        + [("common_fut11", int(a), 11) for a in [1, 2, 3, 4, 5, 6, 8, 10]]
    )

    for scheme, anchor_spec, fut_start in schemes:
        per_model_cells: Dict[str, np.ndarray] = {}
        total_rows = 0
        for kind in models:
            mcfg = cfg["models"][kind]
            st = mcfg["stats"]
            params = mcfg["params"]
            cell_maes = []
            tot_r = 0
            for c in b5:
                cal_full = post_rebound_anchor[c.name] if anchor_spec == "post_rebound" else int(anchor_spec)
                out = _predict_custom_anchor(kind, st, params, c, cal_full, fut_full_start=fut_start)
                cell_maes.append(out["mae_pp"])
                tot_r += out["n_fut"]
            per_model_cells[kind] = np.array(cell_maes, dtype=float)
            total_rows = tot_r
            sweep_rows.append({
                "scheme": scheme,
                "anchor": str(anchor_spec),
                "model": kind,
                "n_scored_rows": int(total_rows),
                "mean_mae_pp": float(np.mean(cell_maes)),
                "sd_mae_pp": float(np.std(cell_maes, ddof=1)),
            })

        # Verify anchor=4 suffix reproduces frozen 5.392867 pp
        if scheme == "suffix" and anchor_spec == 4:
            assert abs(float(np.mean(per_model_cells["primary"])) - 5.392867) < 1e-4
            assert abs(float(np.mean(per_model_cells["scalar_map"])) - 5.221429) < 1e-4

        # Exact paired tests vs primary
        pri_vec = per_model_cells["primary"]
        for kind in models:
            if kind == "primary":
                continue
            t_res = _exact_tests(pri_vec - per_model_cells[kind])
            exact_rows.append({
                "scheme": scheme,
                "anchor": str(anchor_spec),
                "comparator": kind,
                **t_res,
            })

    df_sweep = pd.DataFrame(sweep_rows)
    df_exact = pd.DataFrame(exact_rows)
    df_sweep.to_csv(OUT / "e5_b5_anchor_sweep.csv", index=False)
    df_exact.to_csv(OUT / "e5_b5_exact_tests.csv", index=False)

    payload = {
        "b5_cell_diagnostics": b5_diag,
        "pivot_suffix_mae_pp": df_sweep[df_sweep["scheme"] == "suffix"].pivot(index="anchor", columns="model", values="mean_mae_pp").to_dict(orient="index"),
        "pivot_common_fut11_mae_pp": df_sweep[df_sweep["scheme"] == "common_fut11"].pivot(index="anchor", columns="model", values="mean_mae_pp").to_dict(orient="index"),
        "exact_tests_at_cal4": df_exact[(df_exact["scheme"] == "suffix") & (df_exact["anchor"] == "4")].to_dict(orient="records"),
        "exact_tests_at_post_rebound": df_exact[(df_exact["scheme"] == "suffix") & (df_exact["anchor"] == "post_rebound")].to_dict(orient="records"),
        "exact_tests_all_summary": exact_rows,
    }
    save_json(OUT / "e5_b5_sensitivity_and_exact_tests.json", payload)
    print("[E5] Done.", flush=True)
    return payload


# =============================================================================
# E6 (M9): NASA LOCO Hyperparameters, Weights, & Per-Regime Audit
# =============================================================================
def run_e6(datasets: Dict[str, List[ref.Cell]]) -> Dict[str, Any]:
    print("[E6] Running NASA LOCO hyperparameter, weight, and thermal-regime audit...", flush=True)
    nasa_cells = datasets["NASA"]
    df_cm = pd.read_csv(ROOT / "results" / "loco_NASA" / "cell_metrics.csv")
    df_cm = df_cm[df_cm["scenario"] == "clean"].copy()
    df_cm["mae_pp"] = df_cm["mae"] * 100.0

    # Map cell_name -> (temp, group) from NASA.npz
    with np.load(OLD_DATA / "NASA.npz", allow_pickle=False) as z:
        r_name = np.asarray(z["cell_name"]).astype(str)
        r_temp = np.asarray(z["temp"], float)
        r_grp = np.asarray(z["group"]).astype(str)
    cell_meta = {}
    for nm in sorted(set(r_name)):
        idx = np.where(r_name == nm)[0]
        cell_meta[nm] = {"temp_C": float(r_temp[idx[0]]), "group": str(r_grp[idx[0]])}

    fold_rows: List[Dict[str, Any]] = []
    weights_by_kind: Dict[str, List[np.ndarray]] = {"primary": [], "without_nuisance": [], "logridge": [], "unit_sum": [], "best_segment": []}
    param_counts: Dict[str, Dict[str, int]] = {"primary": {}, "without_nuisance": {}, "logridge": {}, "unit_sum": {}, "best_segment": {}}

    for held in nasa_cells:
        chosen = json.loads((ROOT / "results" / "loco_NASA" / f"chosen_{held.name}.json").read_text())
        train_cells = [c for c in nasa_cells if c.name != held.name]

        row_info = {
            "cell": held.name,
            "temp_C": cell_meta[held.name]["temp_C"],
            "group": cell_meta[held.name]["group"],
        }
        for kind in ["primary", "without_nuisance", "logridge", "unit_sum", "best_segment"]:
            p = chosen[kind]
            p_key = json.dumps(p, sort_keys=True)
            param_counts[kind][p_key] = param_counts[kind].get(p_key, 0) + 1

            m = ref.fit(kind, p, train_cells)
            w = np.asarray(m.get("w", m.get("beta")), float)
            weights_by_kind[kind].append(w)

            # Verify exact reproduction of cell_metrics.csv
            frows = ref._rows_for_predict(held)
            yp = ref.predict(m, held, "clean")
            mae_pp = 100.0 * float(np.mean(np.abs(held.y[frows] - yp)))
            csv_mae_pp = float(df_cm[(df_cm["outer_cell"] == held.name) & (df_cm["model"] == kind)]["mae_pp"].iloc[0])
            assert abs(mae_pp - csv_mae_pp) < 1e-7, f"NASA verify fail {held.name}/{kind}: {mae_pp} vs {csv_mae_pp}"

            row_info[f"{kind}_params"] = p_key
            row_info[f"{kind}_mae_pp"] = mae_pp
            row_info[f"{kind}_w"] = w.tolist()
            row_info[f"{kind}_w_sum"] = float(np.sum(w))
            row_info[f"{kind}_w_l1"] = float(np.sum(np.abs(w)))
            row_info[f"{kind}_w_l2"] = float(np.linalg.norm(w))

        for extra_k in ["histgb", "anchor_constant", "endpoint", "clean_selector"]:
            row_info[f"{extra_k}_mae_pp"] = float(df_cm[(df_cm["outer_cell"] == held.name) & (df_cm["model"] == extra_k)]["mae_pp"].iloc[0])

        fold_rows.append(row_info)

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(OUT / "e6_nasa_folds.csv", index=False)

    # Weight summaries
    weight_summary = {}
    for kind, wlist in weights_by_kind.items():
        W = np.vstack(wlist)
        weight_summary[kind] = {
            "mean_w": np.mean(W, axis=0).tolist(),
            "sd_w": np.std(W, axis=0, ddof=1).tolist(),
            "mean_sum_w": float(np.mean(np.sum(W, axis=1))),
            "sd_sum_w": float(np.std(np.sum(W, axis=1), ddof=1)),
            "mean_l1": float(np.mean(np.sum(np.abs(W), axis=1))),
            "mean_l2": float(np.mean(np.linalg.norm(W, axis=1))),
            "param_frequencies": param_counts[kind],
        }

    # Per-thermal-regime breakdown
    regime_models = ["primary", "without_nuisance", "clean_selector", "logridge", "best_segment", "histgb", "endpoint", "anchor_constant", "unit_sum"]
    regime_rows: List[Dict[str, Any]] = []
    for (temp_c, grp), sub in df_folds.groupby(["temp_C", "group"]):
        rec: Dict[str, Any] = {"temp_C": float(temp_c), "group": str(grp), "n_cells": int(len(sub)), "cells": sub["cell"].tolist()}
        for mname in regime_models:
            rec[f"{mname}_mae_pp"] = float(sub[f"{mname}_mae_pp"].mean())
        regime_rows.append(rec)
    # Pooled NASA row
    pooled_rec: Dict[str, Any] = {"temp_C": None, "group": "ALL_NASA", "n_cells": int(len(df_folds)), "cells": df_folds["cell"].tolist()}
    for mname in regime_models:
        pooled_rec[f"{mname}_mae_pp"] = float(df_folds[f"{mname}_mae_pp"].mean())
    regime_rows.append(pooled_rec)

    df_reg = pd.DataFrame(regime_rows)
    df_reg.to_csv(OUT / "e6_nasa_by_regime.csv", index=False)

    payload = {
        "weight_and_param_summaries": weight_summary,
        "by_thermal_regime": regime_rows,
    }
    save_json(OUT / "e6_nasa_loco_audit.json", payload)
    print("[E6] Done. All 14 NASA LOCO folds verified to <1e-7 pp.", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated post-hoc revision audits E1-E6.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Working root containing data/results/stage2 (default: SOH_WORK_ROOT env or repo parent).",
    )
    args = parser.parse_args()
    _configure_paths(args.root)
    t0 = time.time()
    datasets = load_all_datasets()
    e1 = run_e1()
    e2 = run_e2(datasets)
    e3a = run_e3a(datasets)
    e4 = run_e4(datasets, e2)
    e5 = run_e5(datasets)
    e6 = run_e6(datasets)
    print(f"ALL E1,E2,E3a,E4,E5,E6 COMPLETE in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
