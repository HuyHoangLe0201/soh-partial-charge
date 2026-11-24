#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-hoc Major Revision Audits E7-E15:
  E7:  Label-free oblique-projector Stage-1 compatibility gate across XJTU & external chemistries
  E8:  Chronological online rebound-completion anchor selection & standardized innovation gating on XJTU B5 / B1-B6
  E9:  NASA Leave-One-Temperature-Regime-Out (LOTRO) vs 14-cell LOCO audit
  E10: External multi-chemistry validation on CALCE CS2 (LCO) and Tongji TJU D3 (NCM+NCA)
  E11: Asymmetric one-sided hardware voltage drift (future-only & anchor-only vs symmetric grid shift)
  E12: Stage-2 posterior predictive UQ coverage (sigma^2 vs nu=t*sigma^2) & embedded BMS compute footprint
  E13: OOD / domain-mismatch detector baselines, B4/B6 geometric margin analysis, & fallback control audit
  E14: UQ under distribution shift (conformal vs Bayesian, gate-conditioned coverage, & external CIs)
  E15: SOH knee threshold sensitivity (80%, 85%, 90%) & label-free observable proxy trigger validation

Strictly isolated from frozen confirmatory manifests and accepts datasets via CLI arguments
without encoding local machine paths.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import reference_models as ref  # noqa: E402
import stage2_run as s2  # noqa: E402
from soh_partial_charge import Stage1ResponseFilter, Stage2RankOneUpdater  # noqa: E402


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


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _load_core_datasets(old_dir: Path, new_dir: Path) -> Dict[str, List[ref.Cell]]:
    return {
        "NASA": ref.load_cells(str(old_dir / "NASA.npz"), "NASA"),
        "XJTU_B1": ref.load_cells(str(old_dir / "XJTU_B1.npz"), "XJTU_B1"),
        "XJTU_B2": ref.load_cells(str(old_dir / "XJTU_B2.npz"), "XJTU_B2"),
        "XJTU_B3": ref.load_cells(str(old_dir / "XJTU_B3.npz"), "XJTU_B3"),
        "XJTU_B4": ref.load_cells(str(new_dir / "XJTU_B4.npz"), "XJTU_B4"),
        "XJTU_B5": ref.load_cells(str(new_dir / "XJTU_B5.npz"), "XJTU_B5"),
        "XJTU_B6": ref.load_cells(str(new_dir / "XJTU_B6.npz"), "XJTU_B6"),
    }


def _load_external_datasets(ext_dir: Optional[Path]) -> Dict[str, List[ref.Cell]]:
    out: Dict[str, List[ref.Cell]] = {}
    if ext_dir is None or not ext_dir.exists():
        return out
    calce: List[ref.Cell] = []
    tju_1c: List[ref.Cell] = []
    tju_2c: List[ref.Cell] = []
    tju_4c: List[ref.Cell] = []
    for p in sorted(ext_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".npz":
            continue
        nm = p.name
        if nm.startswith("CALCE_"):
            calce.extend(ref.load_cells(str(p), "CALCE_CS2"))
        elif nm.startswith("TJU_D3_CY25_05_1_") or nm.startswith("TJU_1C"):
            tju_1c.extend(ref.load_cells(str(p), "TJU_1C"))
        elif nm.startswith("TJU_D3_CY25_05_2_") or nm.startswith("TJU_2C"):
            tju_2c.extend(ref.load_cells(str(p), "TJU_2C"))
        elif nm.startswith("TJU_D3_CY25_05_4_") or nm.startswith("TJU_4C"):
            tju_4c.extend(ref.load_cells(str(p), "TJU_4C"))
    if calce:
        out["CALCE_CS2"] = calce
    if tju_1c:
        out["TJU_1C"] = tju_1c
    if tju_2c:
        out["TJU_2C"] = tju_2c
    if tju_4c:
        out["TJU_4C"] = tju_4c
    return out


# =============================================================================
# E7: Label-Free Oblique Projector Domain-Shift & Polarity Gate
# =============================================================================
def run_e7(datasets: Dict[str, List[ref.Cell]], ext_datasets: Dict[str, List[ref.Cell]], out_dir: Path) -> Dict[str, Any]:
    print("[E7] Running label-free Stage-1 oblique-projector online gate...", flush=True)
    f1 = Stage1ResponseFilter.from_pretrained("XJTU_B1")
    b1_cells = datasets["XJTU_B1"]
    cal = f1.calibrate_online_gate(
        [c.D["clean"][ref._future_rows(c)] for c in b1_cells],
        [c.Dref["clean"] for c in b1_cells],
        quantile=0.99,
        reg_floor=1e-4,
    )
    # Check exact agreement with bundled B1 thresholds
    assert abs(cal["tau_d2"] - 8.393022872667578) < 1e-9
    assert abs(cal["tau_zpos"] - 0.051323988898317636) < 1e-9

    # Also compute unbiased 8-fold LOCO gated metrics for XJTU_B1 (threshold calibrated on 7 training cells only)
    b1_loco_rows: List[Dict[str, float]] = []
    for i, held in enumerate(b1_cells):
        tr = [b1_cells[j] for j in range(len(b1_cells)) if j != i]
        p_fold = ref.select("primary", tr)["chosen"]
        f_fold = Stage1ResponseFilter(lam=float(p_fold["lambda"]), gamma=float(p_fold["gamma"]), eta=float(p_fold["eta"])).fit_from_cells(tr)
        rows = ref._future_rows(held)
        g_res = f_fold.predict_gated(held.D["clean"][rows], held.Dref["clean"], float(held.ref_y))
        yt = np.asarray(held.y[rows], dtype=float)
        b1_loco_rows.append({
            "med_d2": float(np.median(g_res["d2_perp"])),
            "alarm_rate": float(g_res["alarm_fraction"]),
            "pol_flip": float(np.mean(g_res["polarity_flip"])),
            "domain_rejected": 1.0 if g_res["domain_incompatible"] else 0.0,
            "rpt_flag": 1.0 if g_res["rpt_recalibration_triggered"] else 0.0,
            "raw_mae_pp": 100.0 * float(np.mean(np.abs(g_res["y_hat_raw"] - yt))),
            "gated_mae_pp": 100.0 * float(np.mean(np.abs(g_res["y_hat_gated"] - yt))),
            "const_mae_pp": 100.0 * float(np.mean(np.abs(float(held.ref_y) - yt))),
        })

    all_cohorts: Dict[str, List[ref.Cell]] = {
        "XJTU_B1": datasets["XJTU_B1"],
        "XJTU_B3": datasets["XJTU_B3"],
        "XJTU_B4": datasets["XJTU_B4"],
        "XJTU_B2": datasets["XJTU_B2"],
        "XJTU_B6": datasets["XJTU_B6"],
        "NASA": datasets["NASA"],
    }
    for k in ("CALCE_CS2", "TJU_1C", "TJU_2C", "TJU_4C"):
        if k in ext_datasets:
            all_cohorts[k] = ext_datasets[k]

    summary_rows: List[Dict[str, Any]] = []
    # XJTU_B1 LOCO entry
    summary_rows.append({
        "cohort": "XJTU_B1_LOCO",
        "regime_type": "Source LOCO (8-fold)",
        "n_cells": len(b1_cells),
        "med_d2_perp": float(np.mean([r["med_d2"] for r in b1_loco_rows])),
        "alarm_pct": 100.0 * float(np.mean([r["alarm_rate"] for r in b1_loco_rows])),
        "polarity_flip_pct": 100.0 * float(np.mean([r["pol_flip"] for r in b1_loco_rows])),
        "domain_rejected_cells": int(round(sum(r["domain_rejected"] for r in b1_loco_rows))),
        "cells_flagged": int(round(sum(r["rpt_flag"] for r in b1_loco_rows))),
        "raw_mae_pp": float(np.mean([r["raw_mae_pp"] for r in b1_loco_rows])),
        "gated_mae_pp": float(np.mean([r["gated_mae_pp"] for r in b1_loco_rows])),
        "anchor_const_mae_pp": float(np.mean([r["const_mae_pp"] for r in b1_loco_rows])),
    })

    for cname, cells in all_cohorts.items():
        per_c = []
        for c in cells:
            rows = ref._future_rows(c)
            g_res = f1.predict_gated(c.D["clean"][rows], c.Dref["clean"], float(c.ref_y))
            yt = np.asarray(c.y[rows], dtype=float)
            per_c.append({
                "med_d2": float(np.median(g_res["d2_perp"])),
                "alarm_rate": float(g_res["alarm_fraction"]),
                "pol_flip": float(np.mean(g_res["polarity_flip"])),
                "domain_rejected": 1 if g_res["domain_incompatible"] else 0,
                "rpt_flag": 1 if g_res["rpt_recalibration_triggered"] else 0,
                "raw_mae_pp": 100.0 * float(np.mean(np.abs(g_res["y_hat_raw"] - yt))),
                "gated_mae_pp": 100.0 * float(np.mean(np.abs(g_res["y_hat_gated"] - yt))),
                "const_mae_pp": 100.0 * float(np.mean(np.abs(float(c.ref_y) - yt))),
            })
        role_map = {
            "XJTU_B1": "Source (in-cohort)",
            "XJTU_B3": "Rate-matched dev.",
            "XJTU_B4": "Stage-1 conf. (R3)",
            "XJTU_B2": "Fast-charge dev. (3C)",
            "XJTU_B6": "Stage-1 conf. (Sat)",
            "NASA": "Cross-archive check (B1->NASA)",
            "CALCE_CS2": "External LCO check (B1->LCO)",
            "TJU_1C": "External NCM+NCA (B1->1C)",
            "TJU_2C": "External NCM+NCA (B1->2C)",
            "TJU_4C": "External NCM+NCA (B1->4C)",
        }
        summary_rows.append({
            "cohort": cname,
            "regime_type": role_map.get(cname, cname),
            "n_cells": len(cells),
            "med_d2_perp": float(np.mean([r["med_d2"] for r in per_c])),
            "alarm_pct": 100.0 * float(np.mean([r["alarm_rate"] for r in per_c])),
            "polarity_flip_pct": 100.0 * float(np.mean([r["pol_flip"] for r in per_c])),
            "domain_rejected_cells": int(sum(r["domain_rejected"] for r in per_c)),
            "cells_flagged": int(sum(r["rpt_flag"] for r in per_c)),
            "raw_mae_pp": float(np.mean([r["raw_mae_pp"] for r in per_c])),
            "gated_mae_pp": float(np.mean([r["gated_mae_pp"] for r in per_c])),
            "anchor_const_mae_pp": float(np.mean([r["const_mae_pp"] for r in per_c])),
        })

    # Verify frozen Stage-1 clean MAEs exactly
    by_c = {r["cohort"]: r for r in summary_rows}
    assert abs(by_c["XJTU_B1_LOCO"]["raw_mae_pp"] - 1.3061) < 1e-3, by_c["XJTU_B1_LOCO"]
    assert abs(by_c["XJTU_B4"]["raw_mae_pp"] - 2.5223) < 1e-3
    assert abs(by_c["XJTU_B6"]["raw_mae_pp"] - 17.7914) < 1e-3
    assert abs(by_c["XJTU_B2"]["raw_mae_pp"] - 14.3977) < 1e-3
    assert abs(by_c["XJTU_B3"]["raw_mae_pp"] - 2.0008) < 1e-3

    # Pooled equal-weighted B4/B6 confirmation entry
    b4, b6 = by_c["XJTU_B4"], by_c["XJTU_B6"]
    summary_rows.append({
        "cohort": "B4_B6_mean",
        "regime_type": "Pooled Stage-1 confirmation (16 cells)",
        "n_cells": int(b4["n_cells"] + b6["n_cells"]),
        "med_d2_perp": 0.5 * (b4["med_d2_perp"] + b6["med_d2_perp"]),
        "alarm_pct": 0.5 * (b4["alarm_pct"] + b6["alarm_pct"]),
        "polarity_flip_pct": 0.5 * (b4["polarity_flip_pct"] + b6["polarity_flip_pct"]),
        "domain_rejected_cells": int(b4["domain_rejected_cells"] + b6["domain_rejected_cells"]),
        "cells_flagged": int(b4["cells_flagged"] + b6["cells_flagged"]),
        "raw_mae_pp": 0.5 * (b4["raw_mae_pp"] + b6["raw_mae_pp"]),
        "gated_mae_pp": 0.5 * (b4["gated_mae_pp"] + b6["gated_mae_pp"]),
        "anchor_const_mae_pp": 0.5 * (b4["anchor_const_mae_pp"] + b6["anchor_const_mae_pp"]),
    })

    df_e7 = pd.DataFrame(summary_rows)
    df_e7.to_csv(out_dir / "e7_online_projector_gate.csv", index=False)
    payload = {"thresholds": cal, "cohorts": summary_rows}
    _save_json(out_dir / "e7_online_projector_gate.json", payload)
    print("[E7] Done.", flush=True)
    return payload


# =============================================================================
# E8: Causal Online Rebound Anchor & Innovation Screening
# =============================================================================
def run_e8(datasets: Dict[str, List[ref.Cell]], out_dir: Path) -> Dict[str, Any]:
    print("[E8] Running causal online rebound-anchor & innovation audits...", flush=True)
    b1_cells = datasets["XJTU_B1"]
    b5_cells = datasets["XJTU_B5"]
    u2 = Stage2RankOneUpdater.from_pretrained()
    b0, Sig, sigma2, nu = u2.beta0_, u2.Sigma_, float(u2.sigma2_), float(u2.t * u2.sigma2_)
    iso = (float(np.trace(Sig)) / 4.0) * np.eye(4)

    def _eval_scheme(name: str, anchor_fn) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        cell_recs: List[Dict[str, Any]] = []
        for c in b5_cells:
            nfull = c.y.size + 1
            D_fut = np.asarray(c.D["clean"], dtype=float)
            Dr = np.asarray(c.Dref["clean"], dtype=float)
            X = np.log(D_fut / Dr[None, :])
            z = np.log(np.asarray(c.y, dtype=float) / float(c.ref_y))
            cal_full = int(anchor_fn(c, D_fut, Dr, z))
            cal_pos = cal_full - 1
            fut_pos = np.arange(cal_pos + 1, c.y.size, dtype=int)
            xa, za = X[cal_pos], float(z[cal_pos])
            Xf, hf = X[fut_pos], np.exp(z[fut_pos])
            q_i = float(xa @ Sig @ xa)
            zeta_a = abs(za - float(xa @ b0)) / math.sqrt(nu + q_i)
            b_pri = b0 + Sig @ xa * (za - float(xa @ b0)) / (nu + q_i)
            b_iso = b0 + iso @ xa * (za - float(xa @ b0)) / (nu + float(xa @ iso @ xa))
            f_s = float(xa @ b0)
            b_scal = (1.0 + f_s * (za - f_s) / (nu + f_s * f_s)) * b0
            b_sec = (xa * (za / float(xa @ xa))) if float(xa @ xa) > 1e-10 else np.zeros(4)
            def _mae(bv):
                pred = np.full(hf.shape, math.exp(za)) if (np.all(bv == 0) and float(xa @ xa) <= 1e-10) else np.exp(Xf @ bv)
                return 100.0 * float(np.mean(np.abs(pred - hf)))
            cell_recs.append({
                "scheme": name,
                "cell": c.name,
                "cal_full": cal_full,
                "cal_cycle": float(c.cycles[cal_pos]),
                "cal_rel_soh": float(math.exp(za)),
                "n_fut": int(hf.size),
                "q_over_nu": float(q_i / nu),
                "zeta_a": float(zeta_a),
                "primary_mae_pp": _mae(b_pri),
                "isotropic_mae_pp": _mae(b_iso),
                "scalar_mae_pp": _mae(b_scal),
                "source_logridge_mae_pp": _mae(b0),
                "target_secant_mae_pp": _mae(b_sec),
                "carry_last_mae_pp": 100.0 * float(np.mean(np.abs(math.exp(za) - hf))),
            })
        agg = {
            "scheme": name,
            "mean_cal_full": float(np.mean([r["cal_full"] for r in cell_recs])),
            "min_cal_full": int(min(r["cal_full"] for r in cell_recs)),
            "max_cal_full": int(max(r["cal_full"] for r in cell_recs)),
            "median_cal_rel_soh": float(np.median([r["cal_rel_soh"] for r in cell_recs])),
            "mean_q_over_nu": float(np.mean([r["q_over_nu"] for r in cell_recs])),
            "mean_zeta_a": float(np.mean([r["zeta_a"] for r in cell_recs])),
            "primary_mae_pp": float(np.mean([r["primary_mae_pp"] for r in cell_recs])),
            "isotropic_mae_pp": float(np.mean([r["isotropic_mae_pp"] for r in cell_recs])),
            "scalar_mae_pp": float(np.mean([r["scalar_mae_pp"] for r in cell_recs])),
            "source_logridge_mae_pp": float(np.mean([r["source_logridge_mae_pp"] for r in cell_recs])),
            "target_secant_mae_pp": float(np.mean([r["target_secant_mae_pp"] for r in cell_recs])),
            "carry_last_mae_pp": float(np.mean([r["carry_last_mae_pp"] for r in cell_recs])),
        }
        return agg, cell_recs

    def _fn_cal4(c, D_fut, Dr, z):
        return 4

    def _fn_causal_prior(c, D_fut, Dr, z):
        # full_idx = pos + 1
        return 1 + u2.select_post_rebound_anchor_online(
            D_fut, Dr, fade_threshold=-0.020, consecutive=2, min_pos=3, max_pos=15, signal="prior"
        )

    def _fn_causal_endp_070(c, D_fut, Dr, z):
        return 1 + u2.select_post_rebound_anchor_online(
            D_fut, Dr, fade_threshold=-0.070, consecutive=2, min_pos=3, max_pos=15, signal="endpoint"
        )

    def _fn_causal_endp_075(c, D_fut, Dr, z):
        return 1 + u2.select_post_rebound_anchor_online(
            D_fut, Dr, fade_threshold=-0.075, consecutive=2, min_pos=3, max_pos=15, signal="endpoint"
        )

    def _fn_oracle_post_rebound(c, D_fut, Dr, z):
        rel = np.concatenate([[1.0], np.exp(z)])
        nfull = rel.size
        cands = [i for i in range(1, nfull - 3) if rel[i] < 1.0]
        return int(cands[0]) if cands else int(min(4, nfull - 4))

    aggs = []
    all_cells: List[Dict[str, Any]] = []
    for sname, sfn in [
        ("locked_a4", _fn_cal4),
        ("causal_prior_m020", _fn_causal_prior),
        ("causal_endpoint_m070", _fn_causal_endp_070),
        ("causal_endpoint_m075", _fn_causal_endp_075),
        ("oracle_post_rebound", _fn_oracle_post_rebound),
    ]:
        ag, recs = _eval_scheme(sname, sfn)
        aggs.append(ag)
        all_cells.extend(recs)

    assert abs(aggs[0]["primary_mae_pp"] - 5.392867) < 1e-3
    assert abs(aggs[0]["target_secant_mae_pp"] - 7.1786) < 1e-3

    # Standardized innovation diagnostic across XJTU B1-B6 at locked a=4
    innov_cohorts = []
    for cname in ["XJTU_B1", "XJTU_B3", "XJTU_B4", "XJTU_B5", "XJTU_B2", "XJTU_B6"]:
        zs, qnus = [], []
        for c in datasets[cname]:
            xa = np.log(c.D["clean"][3] / c.Dref["clean"])
            za = math.log(float(c.y[3]) / float(c.ref_y))
            q_i = float(xa @ Sig @ xa)
            zs.append(abs(za - float(xa @ b0)) / math.sqrt(nu + q_i))
            qnus.append(q_i / nu)
        innov_cohorts.append({
            "cohort": cname,
            "mean_zeta_a4": float(np.mean(zs)),
            "max_zeta_a4": float(np.max(zs)),
            "frac_zeta_gt_2": float(np.mean(np.array(zs) > 2.0)),
            "mean_q_over_nu": float(np.mean(qnus)),
        })

    pd.DataFrame(aggs).to_csv(out_dir / "e8_causal_rebound_anchor.csv", index=False)
    payload = {
        "b5_schemes": aggs,
        "b5_cell_records": all_cells,
        "a4_innovation_across_xjtu": innov_cohorts,
    }
    _save_json(out_dir / "e8_causal_rebound_anchor.json", payload)
    print("[E8] Done.", flush=True)
    return payload


# =============================================================================
# E9: NASA Leave-One-Temperature-Regime-Out (LOTRO) vs 14-Cell LOCO
# =============================================================================
def run_e9(datasets: Dict[str, List[ref.Cell]], old_dir: Path, out_dir: Path) -> Dict[str, Any]:
    print("[E9] Running NASA Leave-One-Temperature-Regime-Out (LOTRO)...", flush=True)
    nasa_cells = datasets["NASA"]
    with np.load(old_dir / "NASA.npz", allow_pickle=False) as z:
        cell_to_group = {str(cn): str(grp) for cn, grp in zip(z["cell_name"], z["group"])}

    kinds = ["primary", "without_nuisance", "logridge", "best_segment", "histgb", "endpoint", "anchor_constant", "unit_sum"]

    # First compute exact 14-fold LOCO as internal reference check
    loco_by_cell: List[Dict[str, Any]] = []
    for i, held in enumerate(nasa_cells):
        tr = [nasa_cells[j] for j in range(len(nasa_cells)) if j != i]
        rec: Dict[str, Any] = {"mode": "LOCO", "cell": held.name, "group": cell_to_group[held.name]}
        for m in kinds:
            ch = ref.select(m, tr)["chosen"]
            mod = ref.fit(m, ch, tr)
            rows = ref._rows_for_predict(held)
            yh = ref.predict(mod, held, "clean")
            rec[m] = 100.0 * float(np.mean(np.abs(held.y[rows] - yh)))
        loco_by_cell.append(rec)

    assert abs(float(np.mean([r["primary"] for r in loco_by_cell])) - 7.0294) < 5e-3
    assert abs(float(np.mean([r["histgb"] for r in loco_by_cell])) - 5.7610) < 5e-3

    # Next compute strict Leave-One-Temperature-Regime-Out (LOTRO)
    lotro_by_cell: List[Dict[str, Any]] = []
    for reg in sorted(set(cell_to_group.values())):
        tr = [c for c in nasa_cells if cell_to_group[c.name] != reg]
        te = [c for c in nasa_cells if cell_to_group[c.name] == reg]
        fitted = {m: ref.fit(m, ref.select(m, tr)["chosen"], tr) for m in kinds}
        for held in te:
            rec = {"mode": "LOTRO", "cell": held.name, "group": reg}
            rows = ref._rows_for_predict(held)
            for m in kinds:
                yh = ref.predict(fitted[m], held, "clean")
                rec[m] = 100.0 * float(np.mean(np.abs(held.y[rows] - yh)))
            lotro_by_cell.append(rec)

    df_all = pd.DataFrame(loco_by_cell + lotro_by_cell)
    summary_rows: List[Dict[str, Any]] = []
    for mode in ("LOCO", "LOTRO"):
        sub_m = df_all[df_all["mode"] == mode]
        for grp in sorted(set(cell_to_group.values())):
            sub_g = sub_m[sub_m["group"] == grp]
            rec = {"mode": mode, "group": grp, "n_cells": int(len(sub_g))}
            for m in kinds:
                rec[f"{m}_mae_pp"] = float(sub_g[m].mean())
            summary_rows.append(rec)
        rec_p = {"mode": mode, "group": "ALL_NASA", "n_cells": int(len(sub_m))}
        for m in kinds:
            rec_p[f"{m}_mae_pp"] = float(sub_m[m].mean())
        summary_rows.append(rec_p)

    pd.DataFrame(summary_rows).to_csv(out_dir / "e9_nasa_lotro.csv", index=False)
    payload = {"summary": summary_rows, "cell_records": loco_by_cell + lotro_by_cell}
    _save_json(out_dir / "e9_nasa_lotro.json", payload)
    print("[E9] Done.", flush=True)
    return payload


# =============================================================================
# E10: External Multi-Chemistry Validation (CALCE CS2 LCO & TJU D3 NCM+NCA)
# =============================================================================
def _cohort_profile_stats(name: str, cells: List[ref.Cell]) -> Dict[str, Any]:
    elig = sum(int(ref._eligibility_mask(c).sum()) for c in cells)
    fut1 = sum(int(ref._future_rows(c).size) for c in cells)
    fut2 = sum(max(0, int(c.y.size - 4)) for c in cells)
    cyc_min = min(float(c.ref_cycle) for c in cells)
    cyc_max = max(float(c.cycles.max()) for c in cells)
    all_y = np.concatenate([np.concatenate([[float(c.ref_y)], np.asarray(c.y, float)]) for c in cells])
    lt80 = int(np.sum(all_y < 0.80))
    le85 = int(np.sum(all_y <= 0.85))
    return {
        "cohort": name,
        "n_cells": len(cells),
        "elig": int(elig),
        "fut1": int(fut1),
        "fut2": int(fut2),
        "cycle_min": int(round(cyc_min)),
        "cycle_max": int(round(cyc_max)),
        "soh_min": float(np.min(all_y)),
        "soh_med": float(np.median(all_y)),
        "soh_max": float(np.max(all_y)),
        "below_080": lt80,
        "le_085": le85,
    }


def run_e10(datasets: Dict[str, List[ref.Cell]], ext_datasets: Dict[str, List[ref.Cell]], out_dir: Path) -> Dict[str, Any]:
    print("[E10] Running external multi-chemistry validation (CALCE CS2 & Tongji TJU D3)...", flush=True)
    if not ext_datasets:
        print("[E10] No external .npz files provided; skipping external validation.", flush=True)
        return {}

    cohort_stats = [_cohort_profile_stats(k, v) for k, v in ext_datasets.items()]
    kinds = ["primary", "without_nuisance", "logridge", "best_segment", "histgb", "endpoint", "anchor_constant", "unit_sum"]
    s1_rows: List[Dict[str, Any]] = []

    def _eval_s1_suite(label: str, folds: List[Tuple[List[ref.Cell], ref.Cell]]) -> None:
        per_m: Dict[str, List[Dict[str, float]]] = {m: [] for m in kinds}
        gam_effs: List[float] = []
        for tr, held in folds:
            rows = ref._rows_for_predict(held)
            yt = np.asarray(held.y[rows], dtype=float)
            high_m = yt > 0.85
            low_m = yt <= 0.85
            for m in kinds:
                ch = ref.select(m, tr)["chosen"]
                mod = ref.fit(m, ch, tr)
                yh = ref.predict(mod, held, "clean")
                err = yh - yt
                per_m[m].append({
                    "mae": 100.0 * float(np.mean(np.abs(err))),
                    "rmse": 100.0 * float(np.sqrt(np.mean(err ** 2))),
                    "bias": 100.0 * float(np.mean(err)),
                    "high_mae": 100.0 * float(np.mean(np.abs(err[high_m]))) if np.any(high_m) else float("nan"),
                    "low_mae": 100.0 * float(np.mean(np.abs(err[low_m]))) if np.any(low_m) else float("nan"),
                })
                if m == "primary":
                    X_te = np.log(held.D["clean"][rows] / held.Dref["clean"][None, :])
                    z_te = np.log(yt / float(held.ref_y))
                    b_te = (X_te * z_te[:, None]).mean(0) / float(np.mean(z_te ** 2))
                    gam_effs.append(float(b_te @ np.asarray(mod["w"], float)))
        geff = float(np.mean(gam_effs)) if gam_effs else float("nan")
        for m in kinds:
            hi_vals = [r["high_mae"] for r in per_m[m] if math.isfinite(r["high_mae"])]
            lo_vals = [r["low_mae"] for r in per_m[m] if math.isfinite(r["low_mae"])]
            s1_rows.append({
                "evaluation": label,
                "model": m,
                "n_cells": len(folds),
                "gamma_eff_primary": geff if m == "primary" else None,
                "mae_pp": float(np.mean([r["mae"] for r in per_m[m]])),
                "rmse_pp": float(np.mean([r["rmse"] for r in per_m[m]])),
                "bias_pp": float(np.mean([r["bias"] for r in per_m[m]])),
                "high_soh_mae_pp": float(np.mean(hi_vals)) if hi_vals else None,
                "low_soh_mae_pp": float(np.mean(lo_vals)) if lo_vals else None,
            })

    if "CALCE_CS2" in ext_datasets:
        cs2 = ext_datasets["CALCE_CS2"]
        _eval_s1_suite("CALCE_CS2_LOCO", [([cs2[j] for j in range(len(cs2)) if j != i], cs2[i]) for i in range(len(cs2))])

    if "TJU_1C" in ext_datasets:
        t1 = ext_datasets["TJU_1C"]
        # Nested 3-fold LOCO uses 2 train cells (GroupKFold capped to 2 folds inside custom call or full TJU_1C train -> 2C/4C)
        # Note ref.select needs >= 3 cells; compute TJU_1C -> TJU_2C and TJU_1C -> TJU_4C transfer using 3-cell TJU_1C source!
        fitted_t1 = {m: ref.fit(m, ref.select(m, t1)["chosen"], t1) for m in kinds}
        for target_k in ("TJU_2C", "TJU_4C"):
            if target_k not in ext_datasets:
                continue
            t_cells = ext_datasets[target_k]
            geffs = []
            for held in t_cells:
                rows = ref._rows_for_predict(held)
                yt = np.asarray(held.y[rows], dtype=float)
                X_te = np.log(held.D["clean"][rows] / held.Dref["clean"][None, :])
                z_te = np.log(yt / float(held.ref_y))
                b_te = (X_te * z_te[:, None]).mean(0) / float(np.mean(z_te ** 2))
                geffs.append(float(b_te @ np.asarray(fitted_t1["primary"]["w"], float)))
            geff_mean = float(np.mean(geffs))
            for m in kinds:
                errs, hi_e, lo_e, r_e, b_e = [], [], [], [], []
                for held in t_cells:
                    rows = ref._rows_for_predict(held)
                    yt = np.asarray(held.y[rows], dtype=float)
                    yh = ref.predict(fitted_t1[m], held, "clean")
                    err = yh - yt
                    errs.append(100.0 * float(np.mean(np.abs(err))))
                    r_e.append(100.0 * float(np.sqrt(np.mean(err ** 2))))
                    b_e.append(100.0 * float(np.mean(err)))
                    if np.any(yt > 0.85):
                        hi_e.append(100.0 * float(np.mean(np.abs(err[yt > 0.85]))))
                    if np.any(yt <= 0.85):
                        lo_e.append(100.0 * float(np.mean(np.abs(err[yt <= 0.85]))))
                s1_rows.append({
                    "evaluation": f"TJU_1C_to_{target_k}",
                    "model": m,
                    "n_cells": len(t_cells),
                    "gamma_eff_primary": geff_mean if m == "primary" else None,
                    "mae_pp": float(np.mean(errs)),
                    "rmse_pp": float(np.mean(r_e)),
                    "bias_pp": float(np.mean(b_e)),
                    "high_soh_mae_pp": float(np.mean(hi_e)) if hi_e else None,
                    "low_soh_mae_pp": float(np.mean(lo_e)) if lo_e else None,
                })

    # Stage 2 on TJU_1C -> TJU_2C & TJU_4C
    s2_rows: List[Dict[str, Any]] = []
    if all(k in ext_datasets for k in ("TJU_1C", "TJU_2C", "TJU_4C")):
        t1 = ext_datasets["TJU_1C"]
        # 3-cell TJU_1C Stage-2 source CV selection
        best_cfg: Dict[str, Dict[str, Any]] = {}
        for knd in ("primary", "isotropic_map", "scalar_map", "source_logridge", "best_segment"):
            best_sc = float("inf")
            best_c: Dict[str, Any] = {}
            for cand in s2._cand_grid(knd):
                ms = []
                for i in range(len(t1)):
                    tr = [t1[j] for j in range(len(t1)) if j != i]
                    st_f = s2.fit_stats(tr, float(cand.get("lambda", 1e-4)))
                    out = s2.predict(knd, st_f, cand, t1[i], "clean", stride=1)
                    fp = np.asarray(out["future_positions"], int)
                    yt = np.asarray(t1[i].y[fp], float) / float(t1[i].ref_y)
                    ms.append(float(np.mean(np.abs(out["pred"] - yt))))
                sc = float(np.mean(ms))
                if sc < best_sc:
                    best_sc = sc
                    best_c = dict(cand)
            best_cfg[knd] = best_c

        for tk in ("TJU_2C", "TJU_4C"):
            tcs = ext_datasets[tk]
            for knd in ("primary", "isotropic_map", "scalar_map", "source_logridge", "target_secant", "endpoint_secant", "best_segment", "carry_last"):
                cand = best_cfg.get(knd, {"lambda": best_cfg["primary"]["lambda"], "t": best_cfg["primary"]["t"]})
                st_full = s2.fit_stats(t1, float(cand.get("lambda", best_cfg["primary"]["lambda"])))
                params = {
                    "lambda": float(cand.get("lambda", best_cfg["primary"]["lambda"])),
                    "t": float(cand.get("t", best_cfg["primary"]["t"])),
                    "segment": int(cand.get("segment", best_cfg["best_segment"]["segment"])),
                }
                c_maes, c_qnu = [], []
                for c in tcs:
                    out = s2.predict(knd, st_full, params, c, "clean", stride=1)
                    fp = np.asarray(out["future_positions"], int)
                    yt = np.asarray(c.y[fp], float) / float(c.ref_y)
                    c_maes.append(100.0 * float(np.mean(np.abs(out["pred"] - yt))))
                    if knd == "primary":
                        xa = np.log(c.D["clean"][3] / c.Dref["clean"])
                        nu_t = float(params["t"]) * float(st_full["sigma2"])
                        c_qnu.append(float(xa @ np.asarray(st_full["Sigma"], float) @ xa) / nu_t)
                s2_rows.append({
                    "evaluation": f"TJU_1C_to_{tk}",
                    "model": knd,
                    "selected_params": json.dumps(params, sort_keys=True),
                    "mae_pp": float(np.mean(c_maes)),
                    "mean_q_over_nu": float(np.mean(c_qnu)) if c_qnu else None,
                })

    pd.DataFrame(s1_rows).to_csv(out_dir / "e10_external_chemistries.csv", index=False)
    payload = {
        "external_cohort_stats": cohort_stats,
        "stage1_external": s1_rows,
        "stage2_external_tju": s2_rows,
    }
    _save_json(out_dir / "e10_external_chemistries.json", payload)
    print("[E10] Done.", flush=True)
    return payload


# =============================================================================
# E11: Asymmetric One-Sided Hardware Voltage Offset Stress
# =============================================================================
def run_e11(datasets: Dict[str, List[ref.Cell]], out_dir: Path) -> Dict[str, Any]:
    print("[E11] Running asymmetric one-sided hardware voltage offset audit...", flush=True)
    kinds = ["primary", "without_nuisance", "logridge", "endpoint", "histgb"]
    tags = ["vm10", "vm5", "vp5", "vp10"]
    b1_models = {m: ref.fit(m, ref.select(m, datasets["XJTU_B1"])["chosen"], datasets["XJTU_B1"]) for m in kinds}

    def _asym_predict(mod: Dict[str, Any], cell: ref.Cell, d_tag: str, r_tag: str) -> float:
        rows = ref._rows_for_predict(cell)
        yt = np.asarray(cell.y[rows], dtype=float)
        kind = mod["kind"]
        if kind in ("response", "logridge"):
            w = np.asarray(mod["w"] if kind == "response" else mod["beta"], dtype=float)
            X = np.log(cell.D[d_tag][rows] / cell.Dref[r_tag][None, :])
            yhat = float(cell.ref_y) * np.exp(X @ w)
        elif kind == "endpoint":
            u = np.log(np.sum(cell.D[d_tag][rows], axis=1) / float(np.sum(cell.Dref[r_tag])))
            yhat = float(cell.ref_y) * np.exp(u)
        elif kind == "histgb":
            est = mod["model"]
            q_fut = np.asarray(cell.Q[d_tag][rows], dtype=float)
            q_ref = np.asarray(cell.Qref[r_tag], dtype=float).reshape(1, -1)
            yhat = float(cell.ref_y) + (est.predict(q_fut) - float(est.predict(q_ref)[0]))
        else:
            raise ValueError(kind)
        return 100.0 * float(np.mean(np.abs(yhat - yt)))

    # Fit outer LOCO fold models for NASA and XJTU_B1 so comparisons match LOCO benchmark faithfully
    loco_fold_mods: Dict[str, List[Tuple[ref.Cell, Dict[str, Dict[str, Any]]]]] = {}
    for cname in ("NASA", "XJTU_B1"):
        cs = datasets[cname]
        folds = []
        for i, held in enumerate(cs):
            tr = [cs[j] for j in range(len(cs)) if j != i]
            mods = {m: ref.fit(m, ref.select(m, tr)["chosen"], tr) for m in kinds}
            folds.append((held, mods))
        loco_fold_mods[cname] = folds

    rows_out: List[Dict[str, Any]] = []
    for cname in ("NASA", "XJTU_B1", "XJTU_B2", "XJTU_B3", "XJTU_B4", "XJTU_B6"):
        for m in kinds:
            if cname in ("NASA", "XJTU_B1"):
                pairs = [(held, mods[m]) for held, mods in loco_fold_mods[cname]]
            else:
                pairs = [(c, b1_models[m]) for c in datasets[cname]]

            clean_mae = float(np.mean([_asym_predict(mod, c, "clean", "clean") for c, mod in pairs]))
            sym_vals = [float(np.mean([_asym_predict(mod, c, tg, tg) for c, mod in pairs])) for tg in tags]
            fut_vals = [float(np.mean([_asym_predict(mod, c, tg, "clean") for c, mod in pairs])) for tg in tags]
            ref_vals = [float(np.mean([_asym_predict(mod, c, "clean", tg) for c, mod in pairs])) for tg in tags]
            rows_out.append({
                "cohort": cname,
                "model": m,
                "clean_mae_pp": clean_mae,
                "sym_min_pp": min(sym_vals),
                "sym_max_pp": max(sym_vals),
                "fut_only_min_pp": min(fut_vals),
                "fut_only_max_pp": max(fut_vals),
                "ref_only_min_pp": min(ref_vals),
                "ref_only_max_pp": max(ref_vals),
                "sym_worst_delta_pp": max(abs(v - clean_mae) for v in sym_vals),
                "asym_worst_delta_pp": max(max(abs(v - clean_mae) for v in fut_vals), max(abs(v - clean_mae) for v in ref_vals)),
            })

    pd.DataFrame(rows_out).to_csv(out_dir / "e11_asymmetric_voltage_drift.csv", index=False)
    payload = {"asymmetric_voltage_drift": rows_out}
    _save_json(out_dir / "e11_asymmetric_voltage_drift.json", payload)
    print("[E11] Done.", flush=True)
    return payload


# =============================================================================
# E12: Stage-2 Posterior Predictive Coverage & Embedded BMS Compute Footprint
# =============================================================================
def run_e12(datasets: Dict[str, List[ref.Cell]], out_dir: Path) -> Dict[str, Any]:
    print("[E12] Running Stage-2 UQ coverage & embedded BMS runtime benchmarks...", flush=True)
    b1_cells = datasets["XJTU_B1"]
    z90 = float(norm.ppf(0.95))
    z95 = float(norm.ppf(0.975))

    def _eval_uq_cells(cohort: str, pairs: List[Tuple[ref.Cell, Stage2RankOneUpdater]]) -> Dict[str, Any]:
        cov90_s2, cov95_s2, cov90_nu, cov95_nu, w95_s2, w95_nu = [], [], [], [], [], []
        for c, upd in pairs:
            D_fut = np.asarray(c.D["clean"], dtype=float)
            Dr = np.asarray(c.Dref["clean"], dtype=float)
            y_all = np.asarray(c.y, dtype=float)
            cal_pos = 3
            fut_pos = np.arange(4, y_all.size, dtype=int)
            hf = y_all[fut_pos] / float(c.ref_y)

            iv_s2 = upd.predict_interval(
                D_fut[fut_pos], Dr, D_cal=D_fut[cal_pos], y_cal=float(y_all[cal_pos]),
                y_ref=float(c.ref_y), level=0.95, include_t_scale=False,
            )
            iv_s2_90 = upd.predict_interval(D_fut[fut_pos], Dr, level=0.90, include_t_scale=False)
            iv_nu = upd.predict_interval(D_fut[fut_pos], Dr, level=0.95, include_t_scale=True)
            iv_nu_90 = upd.predict_interval(D_fut[fut_pos], Dr, level=0.90, include_t_scale=True)

            cov90_s2.append(float(np.mean((hf >= iv_s2_90["lower"]) & (hf <= iv_s2_90["upper"]))))
            cov95_s2.append(float(np.mean((hf >= iv_s2["lower"]) & (hf <= iv_s2["upper"]))))
            cov90_nu.append(float(np.mean((hf >= iv_nu_90["lower"]) & (hf <= iv_nu_90["upper"]))))
            cov95_nu.append(float(np.mean((hf >= iv_nu["lower"]) & (hf <= iv_nu["upper"]))))
            w95_s2.append(100.0 * float(np.mean(iv_s2["upper"] - iv_s2["lower"])))
            w95_nu.append(100.0 * float(np.mean(iv_nu["upper"] - iv_nu["lower"])))

        return {
            "cohort": cohort,
            "n_cells": len(pairs),
            "cov90_sigma2_pct": 100.0 * float(np.mean(cov90_s2)),
            "cov95_sigma2_pct": 100.0 * float(np.mean(cov95_s2)),
            "width95_sigma2_pp": float(np.mean(w95_s2)),
            "cov90_nu_pct": 100.0 * float(np.mean(cov90_nu)),
            "cov95_nu_pct": 100.0 * float(np.mean(cov95_nu)),
            "width95_nu_pp": float(np.mean(w95_nu)),
        }

    # Unbiased 8-fold B1 LOCO UQ check (using frozen lambda=1e-4, t=10 or fold CV selected? Match locked stage2 LOCO fold CV!)
    loco_pairs: List[Tuple[ref.Cell, Stage2RankOneUpdater]] = []
    for i, held in enumerate(b1_cells):
        tr = [b1_cells[j] for j in range(len(b1_cells)) if j != i]
        # Stage-2 CV selection across 7 train cells
        best_sc = float("inf")
        best_lt = (1e-4, 10.0)
        for lam in s2.LAMBDAS:
            for t_val in s2.TG:
                ms = []
                for j in range(len(tr)):
                    sub_tr = [tr[k] for k in range(len(tr)) if k != j]
                    st_f = s2.fit_stats(sub_tr, lam)
                    out = s2.predict("primary", st_f, {"lambda": lam, "t": t_val}, tr[j], "clean", stride=1)
                    fp = np.asarray(out["future_positions"], int)
                    yt = np.asarray(tr[j].y[fp], float) / float(tr[j].ref_y)
                    ms.append(float(np.mean(np.abs(out["pred"] - yt))))
                if np.mean(ms) < best_sc:
                    best_sc = float(np.mean(ms))
                    best_lt = (lam, t_val)
        upd_f = Stage2RankOneUpdater(lam=best_lt[0], t=best_lt[1]).fit_from_cells(tr)
        loco_pairs.append((held, upd_f))

    u2_frozen = Stage2RankOneUpdater.from_pretrained()
    uq_rows = [_eval_uq_cells("XJTU_B1_LOCO", loco_pairs)]
    for cname in ("XJTU_B1", "XJTU_B3", "XJTU_B4", "XJTU_B5", "XJTU_B2", "XJTU_B6"):
        uq_rows.append(_eval_uq_cells(cname, [(c, u2_frozen) for c in datasets[cname]]))

    # Embedded BMS memory, FLOPs & runtime micro-benchmark (50,000 repeats across 20 batches of 2,500)
    f1 = Stage1ResponseFilter.from_pretrained("XJTU_B1")
    sample = datasets["XJTU_B4"][0]
    D1 = sample.D["clean"][0]
    Dr = sample.Dref["clean"]
    yr = float(sample.ref_y)
    Q1 = sample.Q["clean"][0]
    Qr = sample.Qref["clean"]
    P_perp, S_perp_inv = f1.oblique_projector_matrices()
    u2_frozen.update_weights(sample.D["clean"][3], Dr, float(sample.y[3]), yr)
    hgb = ref.fit("histgb", ref.select("histgb", b1_cells)["chosen"], b1_cells)["model"]

    # Pure closed-form float operations matching MCU C firmware + full Python API calls
    w1 = f1.w_.copy()
    td2 = float(f1.gate_thresholds_["tau_d2"])
    tzp = float(f1.gate_thresholds_["tau_zpos"])

    def _bench_us(fn, n_iter: int = 2000, n_batches: int = 25) -> Dict[str, float]:
        samples_us = []
        fn()
        for _ in range(n_batches):
            t0 = time.perf_counter()
            for _ in range(n_iter):
                fn()
            samples_us.append((time.perf_counter() - t0) * 1e6 / n_iter)
        arr = np.array(samples_us, dtype=float)
        return {
            "median_us": float(np.median(arr)),
            "iqr_us": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        }

    # Estimate HistGB node count (200 trees)
    n_hgb_nodes = sum(len(pred_arr[0].nodes) for pred_arr in hgb._predictors)

    b1_s1 = _bench_us(lambda: f1.predict(D1, Dr, yr), n_iter=2000, n_batches=25)
    b1_gate = _bench_us(lambda: f1.predict_gated(D1, Dr, yr), n_iter=1500, n_batches=25)
    b1_c_kernel = _bench_us(lambda: yr * math.exp(float(np.log(D1 / Dr) @ w1)), n_iter=5000, n_batches=25)
    b2_upd = _bench_us(lambda: u2_frozen.update_weights(sample.D["clean"][3], Dr, float(sample.y[3]), yr), n_iter=1500, n_batches=25)
    b2_iv = _bench_us(lambda: u2_frozen.predict_interval(D1, Dr), n_iter=1000, n_batches=25)
    b_hgb = _bench_us(lambda: float(hgb.predict(Q1.reshape(1, -1))[0]), n_iter=200, n_batches=20)

    bms_footprint = [
        {
            "component": "Stage-1 Primary Filter",
            "params_float32": 9,          # D_ref(4) + beta(4) + y_ref(1)
            "flash_bytes_f32": 36,
            "sram_bytes_f32": 64,
            "flops_per_cycle": 17,        # 4 div + 4 log + 4 mul + 3 add + 1 exp + 1 mul
            "python_api_median_us": b1_s1["median_us"],
            "python_api_iqr_us": b1_s1["iqr_us"],
            "vector_kernel_median_us": b1_c_kernel["median_us"],
        },
        {
            "component": "Stage-1 + Projector Gate",
            "params_float32": 43,         # 9 + P_perp(16) + S_perp_inv(16) + thresholds(2)
            "flash_bytes_f32": 172,
            "sram_bytes_f32": 128,
            "flops_per_cycle": 75,        # 17 + P_perp@x (28) + d2 quad (28) + compare (2)
            "python_api_median_us": b1_gate["median_us"],
            "python_api_iqr_us": b1_gate["iqr_us"],
            "vector_kernel_median_us": None,
        },
        {
            "component": "Stage-2 Rank-One MAP (update)",
            "params_float32": 32,         # beta0(4) + Sigma(16) + nu(1) + D_ref(4) + q0(1) + beta*(4) + cal(2)
            "flash_bytes_f32": 128,
            "sram_bytes_f32": 192,
            "flops_per_cycle": 92,        # 1-time at 2nd RPT: Sig@x + dot + rank-1 update + interval
            "python_api_median_us": b2_upd["median_us"],
            "python_api_iqr_us": b2_upd["iqr_us"],
            "vector_kernel_median_us": None,
        },
        {
            "component": "Stage-2 Predict + 95% Interval",
            "params_float32": 32,
            "flash_bytes_f32": 128,
            "sram_bytes_f32": 192,
            "flops_per_cycle": 64,        # x^T beta* + x^T Sigma_post x + sqrt/exp bounds
            "python_api_median_us": b2_iv["median_us"],
            "python_api_iqr_us": b2_iv["iqr_us"],
            "vector_kernel_median_us": None,
        },
        {
            "component": "HistGB (200 trees, 31 leaves)",
            "params_float32": int(n_hgb_nodes * 4),
            "flash_bytes_f32": int(n_hgb_nodes * 16),
            "sram_bytes_f32": 1024,
            "flops_per_cycle": 1200,      # ~5 comparisons/tree * 200 + Q_32 input copy + sum
            "python_api_median_us": b_hgb["median_us"],
            "python_api_iqr_us": b_hgb["iqr_us"],
            "vector_kernel_median_us": None,
        },
    ]

    pd.DataFrame(uq_rows).to_csv(out_dir / "e12_uq_coverage.csv", index=False)
    pd.DataFrame(bms_footprint).to_csv(out_dir / "e12_bms_benchmark.csv", index=False)
    env_info = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "runtime_precision": "float64",
        "memory_accounting": "analytical float32 parameters and FLOP counts",
        "timing_protocol": "time.perf_counter median and IQR across 20-25 batches (200-5000 repeats/batch)",
    }
    payload = {
        "uq_coverage": uq_rows,
        "bms_footprint": bms_footprint,
        "histgb_nodes": int(n_hgb_nodes),
        "benchmark_environment": env_info,
    }
    _save_json(out_dir / "e12_uq_and_bms_benchmark.json", payload)
    print("[E12] Done.", flush=True)
    return payload


# =============================================================================
# E13: OOD Detector Baselines, B4/B6 Geometric Margin, & Fallback Control Audit
# =============================================================================
def _fit_fold_detectors(train_cells: List[ref.Cell], reg_floor: float = 1e-4, quantile: float = 0.99) -> Dict[str, Any]:
    """Fit all Stage-1 unlabelled OOD/domain detectors strictly on training cells."""
    p_prim = ref.select("primary", train_cells)["chosen"]
    f1 = Stage1ResponseFilter(
        lam=float(p_prim["lambda"]),
        gamma=float(p_prim["gamma"]),
        eta=float(p_prim["eta"]),
    ).fit_from_cells(train_cells)
    p_ridge = ref.select("logridge", train_cells)["chosen"]
    m_ridge = ref.fit("logridge", p_ridge, train_cells)
    beta_ridge = np.asarray(m_ridge["beta"], dtype=float)

    X_blocks = []
    for c in train_cells:
        rows = ref._future_rows(c)
        X_blocks.append(np.log(np.asarray(c.D["clean"][rows], float) / np.asarray(c.Dref["clean"], float)[None, :]))
    X_src = np.vstack(X_blocks)
    n_src = X_src.shape[0]

    # 1. Oblique projector P_perp = I - b w^T
    P_perp, S_perp_inv = f1.oblique_projector_matrices(reg_floor=reg_floor)
    r_perp_src = X_src @ P_perp.T
    d2_perp_src = np.sum((r_perp_src @ S_perp_inv) * r_perp_src, axis=1)
    z_hat_src = X_src @ f1.w_
    tau_d2 = float(np.percentile(d2_perp_src, 100.0 * quantile))
    tau_zpos = float(max(np.percentile(z_hat_src, 100.0 * quantile), 0.01))

    # 2. Direct 4D feature Mahalanobis (centered covariance & uncentered second moment)
    mu_X = X_src.mean(axis=0)
    Xc_src = X_src - mu_X[None, :]
    cov_X = (Xc_src.T @ Xc_src) / max(1, n_src - 1) + reg_floor * np.eye(4)
    cov_X_inv = np.linalg.inv(cov_X)
    maha_cov_src = np.sum((Xc_src @ cov_X_inv) * Xc_src, axis=1)
    tau_maha_cov = float(np.percentile(maha_cov_src, 100.0 * quantile))

    H_src = (X_src.T @ X_src) / n_src + reg_floor * np.eye(4)
    H_inv = np.linalg.inv(H_src)
    maha_unc_src = np.sum((X_src @ H_inv) * X_src, axis=1)
    tau_maha_unc = float(np.percentile(maha_unc_src, 100.0 * quantile))

    # 3. Unlabelled top-1 PCA rank-3 orthogonal residual (using uncentered second moment H_src)
    eigvals, eigvecs = np.linalg.eigh((X_src.T @ X_src) / n_src)
    u1 = eigvecs[:, -1]
    P_pca = np.eye(4) - np.outer(u1, u1)
    r_pca_src = X_src @ P_pca.T
    S_pca = (r_pca_src.T @ r_pca_src) / n_src + reg_floor * np.eye(4)
    S_pca_inv = np.linalg.inv(S_pca)
    d2_pca_src = np.sum((r_pca_src @ S_pca_inv) * r_pca_src, axis=1)
    tau_pca = float(np.percentile(d2_pca_src, 100.0 * quantile))

    # 4. Non-idempotent ridge residual P_ridge = I - b beta_ridge^T (Eq. in Prop. 4(b))
    P_ridge = np.eye(4) - np.outer(f1.b_, beta_ridge)
    r_ridge_src = X_src @ P_ridge.T
    S_ridge = P_ridge @ f1.C_ @ P_ridge.T + reg_floor * np.eye(4)
    S_ridge_inv = np.linalg.inv(S_ridge)
    d2_ridge_src = np.sum((r_ridge_src @ S_ridge_inv) * r_ridge_src, axis=1)
    tau_ridge = float(np.percentile(d2_ridge_src, 100.0 * quantile))

    return {
        "f1": f1,
        "beta_ridge": beta_ridge,
        "P_perp": P_perp,
        "S_perp_inv": S_perp_inv,
        "tau_d2": tau_d2,
        "tau_zpos": tau_zpos,
        "mu_X": mu_X,
        "cov_X_inv": cov_X_inv,
        "tau_maha_cov": tau_maha_cov,
        "H_inv": H_inv,
        "tau_maha_unc": tau_maha_unc,
        "P_pca": P_pca,
        "S_pca_inv": S_pca_inv,
        "tau_pca": tau_pca,
        "P_ridge": P_ridge,
        "S_ridge_inv": S_ridge_inv,
        "tau_ridge": tau_ridge,
    }


def _score_cell_detectors(det: Dict[str, Any], cell: ref.Cell) -> Dict[str, Any]:
    rows = ref._future_rows(cell)
    D = np.asarray(cell.D["clean"][rows], float)
    Dr = np.asarray(cell.Dref["clean"], float)
    X = np.log(D / Dr[None, :])

    # 1 & 2: Oblique projector
    r_perp = X @ det["P_perp"].T
    d2_perp = np.sum((r_perp @ det["S_perp_inv"]) * r_perp, axis=1)
    z_hat = X @ det["f1"].w_
    pol_flip = (X[:, 0] > 0.01) & ((X[:, 2] < -0.01) | (X[:, 3] < -0.01))
    alarm_full = (d2_perp > det["tau_d2"]) | (z_hat > det["tau_zpos"]) | pol_flip
    score_full = np.maximum(
        d2_perp / det["tau_d2"],
        np.maximum(z_hat / det["tau_zpos"], pol_flip.astype(float) * 2.0),
    )

    # 3: 4D Mahalanobis (centered & uncentered)
    Xc = X - det["mu_X"][None, :]
    maha_cov = np.sum((Xc @ det["cov_X_inv"]) * Xc, axis=1)
    maha_unc = np.sum((X @ det["H_inv"]) * X, axis=1)

    # 4: PCA rank-3 residual
    r_pca = X @ det["P_pca"].T
    d2_pca = np.sum((r_pca @ det["S_pca_inv"]) * r_pca, axis=1)

    # 5: Non-idempotent ridge residual
    r_ridge = X @ det["P_ridge"].T
    d2_ridge = np.sum((r_ridge @ det["S_ridge_inv"]) * r_ridge, axis=1)

    # 6: Polarity / segment ratio rule continuous score
    score_pol = np.where(pol_flip, 1.0 + np.maximum(X[:, 0], 0.0) + np.maximum(-np.minimum(X[:, 2], X[:, 3]), 0.0), 0.0)

    return {
        "oblique_full_gate": {"alarm": alarm_full, "score": score_full},
        "oblique_d2_only": {"alarm": d2_perp > det["tau_d2"], "score": d2_perp / det["tau_d2"]},
        "maha_4d_cov": {"alarm": maha_cov > det["tau_maha_cov"], "score": maha_cov / det["tau_maha_cov"]},
        "maha_4d_uncentered": {"alarm": maha_unc > det["tau_maha_unc"], "score": maha_unc / det["tau_maha_unc"]},
        "pca_rank3_resid": {"alarm": d2_pca > det["tau_pca"], "score": d2_pca / det["tau_pca"]},
        "ridge_nonidemp_resid": {"alarm": d2_ridge > det["tau_ridge"], "score": d2_ridge / det["tau_ridge"]},
        "polarity_only_rule": {"alarm": pol_flip, "score": score_pol},
    }


def run_e13(datasets: Dict[str, List[ref.Cell]], ext_datasets: Dict[str, List[ref.Cell]], out_dir: Path) -> Dict[str, Any]:
    print("[E13] Running OOD detector baselines, B4 geometric analysis & fallback control audit...", flush=True)
    b1_cells = datasets["XJTU_B1"]
    det_frozen = _fit_fold_detectors(b1_cells, reg_floor=1e-4, quantile=0.99)
    b1_loco_dets: List[Tuple[ref.Cell, Dict[str, Any]]] = []
    for i, held in enumerate(b1_cells):
        tr = [b1_cells[j] for j in range(len(b1_cells)) if j != i]
        b1_loco_dets.append((held, _fit_fold_detectors(tr, reg_floor=1e-4, quantile=0.99)))

    cohort_order = [
        ("XJTU_B1_LOCO", "compatible_xjtu"),
        ("XJTU_B3", "moderate_xjtu"),
        ("XJTU_B4", "compatible_xjtu"),
        ("XJTU_B2", "shifted_xjtu"),
        ("XJTU_B6", "shifted_xjtu"),
        ("NASA", "cross_chem"),
        ("CALCE_CS2", "cross_chem"),
        ("TJU_1C", "cross_chem"),
        ("TJU_2C", "cross_chem"),
        ("TJU_4C", "cross_chem"),
    ]
    det_names = [
        "oblique_full_gate",
        "oblique_d2_only",
        "maha_4d_cov",
        "maha_4d_uncentered",
        "pca_rank3_resid",
        "ridge_nonidemp_resid",
        "polarity_only_rule",
    ]

    det_cohort_rows: List[Dict[str, Any]] = []
    # Collect scores across cycles and cells for global AUROC
    pool_records: Dict[str, Dict[str, List[Any]]] = {
        d: {"cohort": [], "group": [], "cyc_score": [], "cell_alarm_frac": []} for d in det_names
    }

    for cname, grp in cohort_order:
        if cname == "XJTU_B1_LOCO":
            pairs = b1_loco_dets
        elif cname in datasets:
            pairs = [(c, det_frozen) for c in datasets[cname]]
        elif cname in ext_datasets:
            pairs = [(c, det_frozen) for c in ext_datasets[cname]]
        else:
            continue

        per_det: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {d: [] for d in det_names}
        for c, det in pairs:
            res = _score_cell_detectors(det, c)
            for d in det_names:
                al = np.asarray(res[d]["alarm"], dtype=bool)
                sc = np.asarray(res[d]["score"], dtype=float)
                per_det[d].append((al, sc))
                pool_records[d]["cohort"].extend([cname] * sc.size)
                pool_records[d]["group"].extend([grp] * sc.size)
                pool_records[d]["cyc_score"].extend(sc.tolist())
                pool_records[d]["cell_alarm_frac"].append((cname, grp, float(np.mean(al))))

        for d in det_names:
            cell_al_rates = [float(np.mean(al)) for al, _ in per_det[d]]
            cell_med_scores = [float(np.median(sc)) for _, sc in per_det[d]]
            n_rej = int(sum(r > 0.25 for r in cell_al_rates))
            det_cohort_rows.append({
                "cohort": cname,
                "group": grp,
                "detector": d,
                "n_cells": len(pairs),
                "mean_alarm_pct": 100.0 * float(np.mean(cell_al_rates)),
                "cell_rejected_gt25": n_rej,
                "med_normalized_score": float(np.mean(cell_med_scores)),
            })

    # Also test within-chemistry deep-aging transfer: TJU_1C -> TJU_2C / TJU_4C
    tju_deep_rows: List[Dict[str, Any]] = []
    if all(k in ext_datasets for k in ("TJU_1C", "TJU_2C", "TJU_4C")):
        t1 = ext_datasets["TJU_1C"]
        # Regularized 1C detector (3 cells, zero labels for OOD)
        det_t1 = _fit_fold_detectors(t1, reg_floor=1e-4, quantile=0.99)
        for tk in ("TJU_2C", "TJU_4C"):
            tcs = ext_datasets[tk]
            per_det_t = {d: [] for d in det_names}
            for c in tcs:
                res = _score_cell_detectors(det_t1, c)
                for d in det_names:
                    al = np.asarray(res[d]["alarm"], dtype=bool)
                    sc = np.asarray(res[d]["score"], dtype=float)
                    per_det_t[d].append((al, sc))
            for d in det_names:
                al_rates = [float(np.mean(al)) for al, _ in per_det_t[d]]
                tju_deep_rows.append({
                    "suite": f"TJU_1C_to_{tk}_deep_compatible",
                    "detector": d,
                    "n_cells": len(tcs),
                    "mean_alarm_pct": 100.0 * float(np.mean(al_rates)),
                    "cell_rejected_gt25": int(sum(r > 0.25 for r in al_rates)),
                })

    # AUROC & summary TPR/FPR table across all 7 detectors
    detector_summary: List[Dict[str, Any]] = []
    df_dc = pd.DataFrame(det_cohort_rows)
    for d in det_names:
        sub_d = df_dc[df_dc["detector"] == d].set_index("cohort")
        cyc_groups = np.asarray(pool_records[d]["group"])
        cyc_scores = np.asarray(pool_records[d]["cyc_score"], dtype=float)
        mask_xjtu = np.isin(cyc_groups, ["compatible_xjtu", "shifted_xjtu"])
        y_cyc_xjtu = (cyc_groups[mask_xjtu] == "shifted_xjtu").astype(int)
        auroc_cyc_xjtu = float(roc_auc_score(y_cyc_xjtu, cyc_scores[mask_xjtu]))

        mask_all = np.isin(cyc_groups, ["compatible_xjtu", "shifted_xjtu", "cross_chem"])
        y_cyc_all = (cyc_groups[mask_all] != "compatible_xjtu").astype(int)
        auroc_cyc_all = float(roc_auc_score(y_cyc_all, cyc_scores[mask_all]))

        cell_rows = pool_records[d]["cell_alarm_frac"]
        cx_y = [1 if g == "shifted_xjtu" else 0 for _, g, _ in cell_rows if g in ("compatible_xjtu", "shifted_xjtu")]
        cx_s = [r for _, g, r in cell_rows if g in ("compatible_xjtu", "shifted_xjtu")]
        auroc_cell_xjtu = float(roc_auc_score(cx_y, cx_s))

        fpr_matched = 0.5 * (float(sub_d.loc["XJTU_B1_LOCO", "mean_alarm_pct"]) + float(sub_d.loc["XJTU_B4", "mean_alarm_pct"]))
        rej_matched = int(sub_d.loc["XJTU_B1_LOCO", "cell_rejected_gt25"] + sub_d.loc["XJTU_B4", "cell_rejected_gt25"])
        rej_b2 = int(sub_d.loc["XJTU_B2", "cell_rejected_gt25"])
        rej_b6 = int(sub_d.loc["XJTU_B6", "cell_rejected_gt25"])
        rej_cross = int(sum(sub_d.loc[k, "cell_rejected_gt25"] for k in ("NASA", "CALCE_CS2", "TJU_1C", "TJU_2C", "TJU_4C") if k in sub_d.index))
        n_cross = int(sum(sub_d.loc[k, "n_cells"] for k in ("NASA", "CALCE_CS2", "TJU_1C", "TJU_2C", "TJU_4C") if k in sub_d.index))
        tju2_fpr = next((r["mean_alarm_pct"] for r in tju_deep_rows if r["suite"] == "TJU_1C_to_TJU_2C_deep_compatible" and r["detector"] == d), None)
        tju4_fpr = next((r["mean_alarm_pct"] for r in tju_deep_rows if r["suite"] == "TJU_1C_to_TJU_4C_deep_compatible" and r["detector"] == d), None)

        detector_summary.append({
            "detector": d,
            "b1_loco_alarm_pct": float(sub_d.loc["XJTU_B1_LOCO", "mean_alarm_pct"]),
            "b4_alarm_pct": float(sub_d.loc["XJTU_B4", "mean_alarm_pct"]),
            "matched_xjtu_fpr_pct": fpr_matched,
            "matched_xjtu_rej": f"{rej_matched}/16",
            "b2_alarm_pct": float(sub_d.loc["XJTU_B2", "mean_alarm_pct"]),
            "b2_rej": f"{rej_b2}/15",
            "b6_alarm_pct": float(sub_d.loc["XJTU_B6", "mean_alarm_pct"]),
            "b6_rej": f"{rej_b6}/8",
            "cross_chem_rej": f"{rej_cross}/{n_cross}",
            "cycle_auroc_xjtu_comp_vs_shift": auroc_cyc_xjtu,
            "cell_auroc_xjtu_comp_vs_shift": auroc_cell_xjtu,
            "cycle_auroc_comp_vs_all_incompat": auroc_cyc_all,
            "tju_1c_to_2c_compat_alarm_pct": tju2_fpr,
            "tju_1c_to_4c_compat_alarm_pct": tju4_fpr,
        })

    # 13b: Exact Geometric Explanation of XJTU B4 acceptance vs B2/B6 rejection
    f1 = det_frozen["f1"]
    b_S = np.asarray(f1.b_, dtype=float)
    w = np.asarray(f1.w_, dtype=float)
    P_perp, S_perp_inv = det_frozen["P_perp"], det_frozen["S_perp_inv"]
    tau_d2 = det_frozen["tau_d2"]

    geom_rows: List[Dict[str, Any]] = []
    geom_cohorts = [(k, datasets[k]) for k in ("XJTU_B1", "XJTU_B3", "XJTU_B4", "XJTU_B2", "XJTU_B6", "NASA")]
    for k in ("CALCE_CS2", "TJU_1C", "TJU_2C", "TJU_4C"):
        if k in ext_datasets:
            geom_cohorts.append((k, ext_datasets[k]))

    for cname, cells in geom_cohorts:
        X_list, z_list, abs_z_list, r_tilde_list = [], [], [], []
        b_cells = []
        for c in cells:
            rows = ref._future_rows(c)
            Xc = np.log(np.asarray(c.D["clean"][rows], float) / np.asarray(c.Dref["clean"], float)[None, :])
            zc = np.log(np.asarray(c.y[rows], float) / float(c.ref_y))
            b_cells.append((Xc * zc[:, None]).mean(axis=0) / float(np.mean(zc ** 2)))
            X_list.append(Xc)
            z_list.append(zc)
            abs_z_list.extend(np.abs(zc).tolist())
        b_K = np.mean(np.vstack(b_cells), axis=0)
        X_all = np.vstack(X_list)
        z_all = np.concatenate(z_list)
        r_tilde = X_all - z_all[:, None] * b_K[None, :]
        r_tilde_perp = r_tilde @ P_perp.T
        noise_d2 = np.sum((r_tilde_perp @ S_perp_inv) * r_tilde_perp, axis=1)

        gamma_eff = float(b_K @ w)
        delta_perp = P_perp @ b_K
        eta_perp = math.sqrt(max(float(delta_perp @ S_perp_inv @ delta_perp), 0.0))
        med_abs_z = float(np.median(abs_z_list))
        p95_abs_z = float(np.percentile(abs_z_list, 95))
        det_d2_med = float((eta_perp * med_abs_z) ** 2)
        det_d2_p95 = float((eta_perp * p95_abs_z) ** 2)
        z_crit = math.sqrt(tau_d2) / eta_perp if eta_perp > 1e-9 else float("inf")
        soh_rel_crit = math.exp(-z_crit) if math.isfinite(z_crit) and z_crit < 50.0 else 0.0
        geom_rows.append({
            "cohort": cname,
            "slope_euclid_diff": float(np.linalg.norm(b_K - b_S)),
            "gamma_eff": gamma_eff,
            "delta_perp_norm": float(np.linalg.norm(delta_perp)),
            "eta_perp": eta_perp,
            "med_abs_z": med_abs_z,
            "p95_abs_z": p95_abs_z,
            "det_d2_at_med_z": det_d2_med,
            "det_d2_at_p95_z": det_d2_p95,
            "med_noise_d2": float(np.median(noise_d2)),
            "z_crit_tau_d2": z_crit,
            "rel_soh_crit_tau_d2": soh_rel_crit,
        })

    # 13c: Fallback Control Policy Comparison on Warning Alarms
    m_hgb_b1 = ref.fit("histgb", ref.select("histgb", b1_cells)["chosen"], b1_cells)
    m_logr_b1 = ref.fit("logridge", ref.select("logridge", b1_cells)["chosen"], b1_cells)

    loco_fb_models: List[Tuple[ref.Cell, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for i, held in enumerate(b1_cells):
        tr = [b1_cells[j] for j in range(len(b1_cells)) if j != i]
        loco_fb_models.append((
            held,
            b1_loco_dets[i][1],
            ref.fit("logridge", ref.select("logridge", tr)["chosen"], tr),
            ref.fit("histgb", ref.select("histgb", tr)["chosen"], tr),
        ))

    fb_rows: List[Dict[str, Any]] = []
    fb_cohorts = ["XJTU_B1_LOCO", "XJTU_B3", "XJTU_B4", "XJTU_B2", "XJTU_B6", "NASA"]
    for k in ("CALCE_CS2", "TJU_2C", "TJU_4C"):
        if k in ext_datasets:
            fb_cohorts.append(k)

    for cname in fb_cohorts:
        if cname == "XJTU_B1_LOCO":
            quads = loco_fb_models
        elif cname in datasets:
            quads = [(c, det_frozen, m_logr_b1, m_hgb_b1) for c in datasets[cname]]
        else:
            quads = [(c, det_frozen, m_logr_b1, m_hgb_b1) for c in ext_datasets[cname]]

        pol_maes: Dict[str, List[float]] = {
            "raw_primary": [],
            "hold_last_safe": [],
            "alarm_to_yref": [],
            "latch_yref_on_3consec": [],
            "alarm_to_damped_logridge": [],
            "alarm_to_histgb": [],
            "always_anchor_constant": [],
        }
        for c, det, m_lr, m_hg in quads:
            rows = ref._future_rows(c)
            yt = np.asarray(c.y[rows], dtype=float)
            D = np.asarray(c.D["clean"][rows], float)
            Dr = np.asarray(c.Dref["clean"], float)
            yr = float(c.ref_y)
            g_res = det["f1"].predict_gated(
                D, Dr, yr, tau_d2=det["tau_d2"], tau_zpos=det["tau_zpos"], reg_floor=1e-4
            )
            al = np.asarray(g_res["alarm"], dtype=bool)
            y_raw = np.asarray(g_res["y_hat_raw"], dtype=float)
            y_hold = np.asarray(g_res["y_hat_gated"], dtype=float)
            y_const = np.full_like(y_raw, yr)
            y_al_const = np.where(al, yr, y_raw)

            # Latch commissioning y_ref permanently once 3 consecutive alarms occur (RPT requested)
            y_latch = y_raw.copy()
            consec = 0
            latched = False
            for idx in range(y_raw.size):
                if latched:
                    y_latch[idx] = yr
                elif bool(al[idx]):
                    consec += 1
                    y_latch[idx] = yr
                    if consec >= 3:
                        latched = True
                else:
                    consec = 0

            y_lr = ref.predict(m_lr, c, "clean")
            y_lr_damp = yr * ((y_lr / yr) ** 0.5)
            y_al_damp = np.where(al, y_lr_damp, y_raw)
            y_hg = ref.predict(m_hg, c, "clean")
            y_al_hg = np.where(al, y_hg, y_raw)

            pol_maes["raw_primary"].append(100.0 * float(np.mean(np.abs(y_raw - yt))))
            pol_maes["hold_last_safe"].append(100.0 * float(np.mean(np.abs(y_hold - yt))))
            pol_maes["alarm_to_yref"].append(100.0 * float(np.mean(np.abs(y_al_const - yt))))
            pol_maes["latch_yref_on_3consec"].append(100.0 * float(np.mean(np.abs(y_latch - yt))))
            pol_maes["alarm_to_damped_logridge"].append(100.0 * float(np.mean(np.abs(y_al_damp - yt))))
            pol_maes["alarm_to_histgb"].append(100.0 * float(np.mean(np.abs(y_al_hg - yt))))
            pol_maes["always_anchor_constant"].append(100.0 * float(np.mean(np.abs(y_const - yt))))

        row_fb = {"cohort": cname, "n_cells": len(quads)}
        for pkey, vals in pol_maes.items():
            row_fb[f"{pkey}_mae_pp"] = float(np.mean(vals))
        fb_rows.append(row_fb)

    # Pooled B4/B6 fallback mean
    fb_by_c = {r["cohort"]: r for r in fb_rows}
    if "XJTU_B4" in fb_by_c and "XJTU_B6" in fb_by_c:
        mean_row = {"cohort": "B4_B6_mean", "n_cells": 16}
        for k in fb_by_c["XJTU_B4"]:
            if k.endswith("_mae_pp"):
                mean_row[k] = 0.5 * (fb_by_c["XJTU_B4"][k] + fb_by_c["XJTU_B6"][k])
        fb_rows.append(mean_row)

    pd.DataFrame(detector_summary).to_csv(out_dir / "e13_ood_detector_comparison.csv", index=False)
    pd.DataFrame(fb_rows).to_csv(out_dir / "e13_fallback_policies.csv", index=False)
    payload = {
        "detector_summary": detector_summary,
        "detector_by_cohort": det_cohort_rows,
        "tju_deep_compatible_alarm": tju_deep_rows,
        "geometric_margin_analysis": geom_rows,
        "fallback_policies": fb_rows,
    }
    _save_json(out_dir / "e13_ood_baselines_and_fallback.json", payload)
    print("[E13] Done.", flush=True)
    return payload


# =============================================================================
# E14: UQ Under Shift (Conformal vs Bayesian, Gate-Conditioned) & External CIs
# =============================================================================
def _bootstrap_cell_ci(vals: Sequence[float], n_boot: int = 20000, seed: int = 20261107) -> Dict[str, Any]:
    arr = np.asarray(vals, dtype=float)
    m = arr.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n_boot, m))
    boot_means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if m > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_lo": float(np.percentile(boot_means, 2.5)),
        "ci95_hi": float(np.percentile(boot_means, 97.5)),
        "cell_vals": [float(v) for v in arr],
    }


def _conformal_loo_quantiles(train_cells: List[ref.Cell], lam: float, t_val: float, alpha: float = 0.05) -> Tuple[float, float]:
    """Leave-one-cell-out conformal quantiles (unweighted |e| and studentized |e|/sd) for Stage 2."""
    abs_res, norm_res = [], []
    for i, held in enumerate(train_cells):
        tr = [train_cells[j] for j in range(len(train_cells)) if j != i]
        upd = Stage2RankOneUpdater(lam=lam, t=t_val).fit_from_cells(tr)
        D_fut = np.asarray(held.D["clean"], float)
        Dr = np.asarray(held.Dref["clean"], float)
        y_all = np.asarray(held.y, float)
        fut_pos = np.arange(4, y_all.size, dtype=int)
        if fut_pos.size == 0:
            continue
        iv = upd.predict_interval(
            D_fut[fut_pos], Dr, D_cal=D_fut[3], y_cal=float(y_all[3]), y_ref=float(held.ref_y),
            level=1.0 - alpha, include_t_scale=True,
        )
        z_true = np.log(y_all[fut_pos] / float(held.ref_y))
        z_pred = np.log(np.asarray(iv["pred"], float))
        log_sd = np.asarray(iv["log_sd"], float)
        e = np.abs(z_true - z_pred)
        abs_res.extend(e.tolist())
        norm_res.extend((e / log_sd).tolist())

    def _fq(arr_list: List[float]) -> float:
        s = np.sort(np.asarray(arr_list, dtype=float))
        n = s.size
        k = min(n, max(1, int(math.ceil((n + 1) * (1.0 - alpha)))))
        return float(s[k - 1])

    return _fq(norm_res), _fq(abs_res)


def run_e14(datasets: Dict[str, List[ref.Cell]], ext_datasets: Dict[str, List[ref.Cell]], out_dir: Path) -> Dict[str, Any]:
    print("[E14] Running Stage-2 conformal/gated UQ reliability & external bootstrap CIs...", flush=True)
    b1_cells = datasets["XJTU_B1"]
    f1_frozen = Stage1ResponseFilter.from_pretrained("XJTU_B1")
    u2_frozen = Stage2RankOneUpdater.from_pretrained()

    # Build unbiased 8-fold LOCO tuples on XJTU_B1 (matching E12 hyperparameter CV)
    b1_loco_models: List[Tuple[ref.Cell, Stage1ResponseFilter, Stage2RankOneUpdater, float, float]] = []
    for i, held in enumerate(b1_cells):
        tr = [b1_cells[j] for j in range(len(b1_cells)) if j != i]
        p_s1 = ref.select("primary", tr)["chosen"]
        f1_f = Stage1ResponseFilter(lam=float(p_s1["lambda"]), gamma=float(p_s1["gamma"]), eta=float(p_s1["eta"])).fit_from_cells(tr)
        best_sc, best_lt = float("inf"), (1e-4, 10.0)
        for lam in s2.LAMBDAS:
            for tv in s2.TG:
                ms = []
                for j in range(len(tr)):
                    sub_tr = [tr[k] for k in range(len(tr)) if k != j]
                    st_f = s2.fit_stats(sub_tr, lam)
                    out = s2.predict("primary", st_f, {"lambda": lam, "t": tv}, tr[j], "clean", stride=1)
                    fp = np.asarray(out["future_positions"], int)
                    yt = np.asarray(tr[j].y[fp], float) / float(tr[j].ref_y)
                    ms.append(float(np.mean(np.abs(out["pred"] - yt))))
                if np.mean(ms) < best_sc:
                    best_sc = float(np.mean(ms))
                    best_lt = (lam, tv)
        u2_f = Stage2RankOneUpdater(lam=best_lt[0], t=best_lt[1]).fit_from_cells(tr)
        qn, qa = _conformal_loo_quantiles(tr, best_lt[0], best_lt[1], alpha=0.05)
        b1_loco_models.append((held, f1_f, u2_f, qn, qa))

    qn_b1, qa_b1 = _conformal_loo_quantiles(b1_cells, float(u2_frozen.lam), float(u2_frozen.t), alpha=0.05)

    def _eval_uq_suite(label: str, items: List[Tuple[ref.Cell, Stage1ResponseFilter, Stage2RankOneUpdater, float, float]]) -> Dict[str, Any]:
        c_nu, w_nu = [], []
        c_conf_s, w_conf_s = [], []
        c_conf_a, w_conf_a = [], []
        c_gated_nu, c_gated_cs, accept_fracs = [], [], []
        c_hi_nu, c_lo_nu, c_hi_cs, c_lo_cs = [], [], [], []
        n_cells_accepted = 0

        for held, f1_m, u2_m, qn, qa in items:
            D_fut = np.asarray(held.D["clean"], float)
            Dr = np.asarray(held.Dref["clean"], float)
            y_all = np.asarray(held.y, float)
            fut_pos = np.arange(4, y_all.size, dtype=int)
            hf = y_all[fut_pos] / float(held.ref_y)
            D_eval = D_fut[fut_pos]

            iv = u2_m.predict_interval(
                D_eval, Dr, D_cal=D_fut[3], y_cal=float(y_all[3]), y_ref=float(held.ref_y),
                level=0.95, include_t_scale=True,
            )
            pred = np.asarray(iv["pred"], float)
            mu = np.log(pred)
            sd = np.asarray(iv["log_sd"], float)
            zeta_a = float(iv["standardized_innovation_zeta"])

            in_nu = (hf >= iv["lower"]) & (hf <= iv["upper"])
            lo_cs, hi_cs = np.exp(mu - qn * sd), np.exp(mu + qn * sd)
            in_cs = (hf >= lo_cs) & (hf <= hi_cs)
            lo_ca, hi_ca = np.exp(mu - qa), np.exp(mu + qa)
            in_ca = (hf >= lo_ca) & (hf <= hi_ca)

            c_nu.append(float(np.mean(in_nu)))
            w_nu.append(100.0 * float(np.mean(iv["upper"] - iv["lower"])))
            c_conf_s.append(float(np.mean(in_cs)))
            w_conf_s.append(100.0 * float(np.mean(hi_cs - lo_cs)))
            c_conf_a.append(float(np.mean(in_ca)))
            w_conf_a.append(100.0 * float(np.mean(hi_ca - lo_ca)))

            if np.any(hf > 0.85):
                c_hi_nu.append(float(np.mean(in_nu[hf > 0.85])))
                c_hi_cs.append(float(np.mean(in_cs[hf > 0.85])))
            if np.any(hf <= 0.85):
                c_lo_nu.append(float(np.mean(in_nu[hf <= 0.85])))
                c_lo_cs.append(float(np.mean(in_cs[hf <= 0.85])))

            # Gate-conditioned UQ: accept intervals only when Stage-1 projector gate is calm AND zeta_a <= 2.0
            g_res = f1_m.predict_gated(D_eval, Dr, float(held.ref_y))
            cycle_accept = (~np.asarray(g_res["alarm"], bool)) & (zeta_a <= 2.0)
            accept_fracs.append(float(np.mean(cycle_accept)))
            if (not bool(g_res["domain_incompatible"])) and (zeta_a <= 2.0) and np.any(cycle_accept):
                n_cells_accepted += 1
                c_gated_nu.append(float(np.mean(in_nu[cycle_accept])))
                c_gated_cs.append(float(np.mean(in_cs[cycle_accept])))

        return {
            "cohort": label,
            "n_cells": len(items),
            "q95_studentized_conformal": float(np.mean([x[3] for x in items])),
            "cov95_nu_pct": 100.0 * float(np.mean(c_nu)),
            "width95_nu_pp": float(np.mean(w_nu)),
            "cov95_conformal_stud_pct": 100.0 * float(np.mean(c_conf_s)),
            "width95_conformal_stud_pp": float(np.mean(w_conf_s)),
            "cov95_conformal_abs_pct": 100.0 * float(np.mean(c_conf_a)),
            "width95_conformal_abs_pp": float(np.mean(w_conf_a)),
            "accepted_cells": n_cells_accepted,
            "accepted_cycles_pct": 100.0 * float(np.mean(accept_fracs)),
            "gated_cov95_nu_pct": 100.0 * float(np.mean(c_gated_nu)) if c_gated_nu else None,
            "gated_cov95_conformal_stud_pct": 100.0 * float(np.mean(c_gated_cs)) if c_gated_cs else None,
            "hi_h_gt085_cov95_nu_pct": 100.0 * float(np.mean(c_hi_nu)) if c_hi_nu else None,
            "lo_h_le085_cov95_nu_pct": 100.0 * float(np.mean(c_lo_nu)) if c_lo_nu else None,
            "hi_h_gt085_cov95_cs_pct": 100.0 * float(np.mean(c_hi_cs)) if c_hi_cs else None,
            "lo_h_le085_cov95_cs_pct": 100.0 * float(np.mean(c_lo_cs)) if c_lo_cs else None,
        }

    uq_shift_rows: List[Dict[str, Any]] = [_eval_uq_suite("XJTU_B1_LOCO", b1_loco_models)]
    for cname in ("XJTU_B3", "XJTU_B4", "XJTU_B5", "XJTU_B2", "XJTU_B6"):
        uq_shift_rows.append(
            _eval_uq_suite(cname, [(c, f1_frozen, u2_frozen, qn_b1, qa_b1) for c in datasets[cname]])
        )

    # Protocol-matched within-chemistry Tongji TJU_1C -> TJU_2C / TJU_4C UQ
    if all(k in ext_datasets for k in ("TJU_1C", "TJU_2C", "TJU_4C")):
        t1 = ext_datasets["TJU_1C"]
        p_t1 = ref.select("primary", t1)["chosen"]
        f1_t1 = Stage1ResponseFilter(lam=float(p_t1["lambda"]), gamma=float(p_t1["gamma"]), eta=float(p_t1["eta"])).fit_from_cells(t1)
        u2_t1 = Stage2RankOneUpdater(lam=1e-6, t=0.1).fit_from_cells(t1)
        qn_t1, qa_t1 = _conformal_loo_quantiles(t1, 1e-6, 0.1, alpha=0.05)
        for tk in ("TJU_2C", "TJU_4C"):
            uq_shift_rows.append(
                _eval_uq_suite(f"TJU_1C_to_{tk}", [(c, f1_t1, u2_t1, qn_t1, qa_t1) for c in ext_datasets[tk]])
            )

    # 14b: External Cohort Cell-Level Bootstrap CIs & Min-Max Ranges
    ext_ci_rows: List[Dict[str, Any]] = []
    ext_kinds = ["primary", "logridge", "histgb", "best_segment", "endpoint", "anchor_constant"]
    if "CALCE_CS2" in ext_datasets:
        cs2 = ext_datasets["CALCE_CS2"]
        per_m: Dict[str, List[float]] = {m: [] for m in ext_kinds}
        for i, held in enumerate(cs2):
            tr = [cs2[j] for j in range(len(cs2)) if j != i]
            rows = ref._future_rows(held)
            yt = np.asarray(held.y[rows], float)
            for m in ext_kinds:
                mod = ref.fit(m, ref.select(m, tr)["chosen"], tr)
                yh = ref.predict(mod, held, "clean")
                per_m[m].append(100.0 * float(np.mean(np.abs(yh - yt))))
        for m in ext_kinds:
            st = _bootstrap_cell_ci(per_m[m])
            st.update({"evaluation": "CALCE_CS2_LOCO", "model": m, "n_cells": len(cs2)})
            ext_ci_rows.append(st)
        for ctrl in ("endpoint", "anchor_constant"):
            diff = np.asarray(per_m["primary"]) - np.asarray(per_m[ctrl])
            st = _bootstrap_cell_ci(diff)
            st.update({"evaluation": "CALCE_CS2_LOCO", "model": f"primary_minus_{ctrl}", "n_cells": len(cs2)})
            ext_ci_rows.append(st)

    if "TJU_1C" in ext_datasets:
        t1 = ext_datasets["TJU_1C"]
        fitted_t1 = {m: ref.fit(m, ref.select(m, t1)["chosen"], t1) for m in ext_kinds}
        for tk in ("TJU_2C", "TJU_4C"):
            if tk not in ext_datasets:
                continue
            tcs = ext_datasets[tk]
            per_m = {m: [] for m in ext_kinds}
            for held in tcs:
                rows = ref._future_rows(held)
                yt = np.asarray(held.y[rows], float)
                for m in ext_kinds:
                    yh = ref.predict(fitted_t1[m], held, "clean")
                    per_m[m].append(100.0 * float(np.mean(np.abs(yh - yt))))
            for m in ext_kinds:
                st = _bootstrap_cell_ci(per_m[m])
                st.update({"evaluation": f"TJU_1C_to_{tk}", "model": m, "n_cells": len(tcs)})
                ext_ci_rows.append(st)
            for ctrl in ("endpoint", "anchor_constant"):
                diff = np.asarray(per_m["primary"]) - np.asarray(per_m[ctrl])
                st = _bootstrap_cell_ci(diff)
                st.update({"evaluation": f"TJU_1C_to_{tk}", "model": f"primary_minus_{ctrl}", "n_cells": len(tcs)})
                ext_ci_rows.append(st)

    pd.DataFrame(uq_shift_rows).to_csv(out_dir / "e14_uq_conformal_gated.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != "cell_vals"} for r in ext_ci_rows]).to_csv(
        out_dir / "e14_external_bootstrap_cis.csv", index=False
    )
    payload = {"uq_conformal_and_gated": uq_shift_rows, "external_bootstrap_cis": ext_ci_rows}
    _save_json(out_dir / "e14_uq_conformal_and_external_ci.json", payload)
    print("[E14] Done.", flush=True)
    return payload


# =============================================================================
# E15: SOH Knee Threshold Sensitivity (80%, 85%, 90%) & Observable Proxy Check
# =============================================================================
def run_e15(datasets: Dict[str, List[ref.Cell]], ext_datasets: Dict[str, List[ref.Cell]], out_dir: Path) -> Dict[str, Any]:
    print("[E15] Running 80%/85%/90% SOH knee threshold sensitivity & observable proxy triggers...", flush=True)
    b1_cells = datasets["XJTU_B1"]
    f1_frozen = Stage1ResponseFilter.from_pretrained("XJTU_B1")
    u2_frozen = Stage2RankOneUpdater.from_pretrained()
    m_b1_mods = {
        m: ref.fit(m, ref.select(m, b1_cells)["chosen"], b1_cells)
        for m in ("primary", "logridge", "endpoint", "anchor_constant")
    }

    # Prepare held-out LOCO fold models for XJTU_B1, NASA, and CALCE_CS2, and 1C fit for TJU
    b1_loco_s1, b1_loco_s2, b1_loco_f1 = [], [], []
    for i, held in enumerate(b1_cells):
        tr = [b1_cells[j] for j in range(len(b1_cells)) if j != i]
        p1 = ref.select("primary", tr)["chosen"]
        f1_f = Stage1ResponseFilter(lam=float(p1["lambda"]), gamma=float(p1["gamma"]), eta=float(p1["eta"])).fit_from_cells(tr)
        b1_loco_f1.append((held, f1_f))
        b1_loco_s1.append((held, {m: ref.fit(m, ref.select(m, tr)["chosen"], tr) for m in ("primary", "logridge", "endpoint", "anchor_constant")}))
        st_f = s2.fit_stats(tr, 1e-4)
        b1_loco_s2.append((held, st_f))

    nasa_cells = datasets["NASA"]
    nasa_loco_s1 = []
    for i, held in enumerate(nasa_cells):
        tr = [nasa_cells[j] for j in range(len(nasa_cells)) if j != i]
        nasa_loco_s1.append((held, {m: ref.fit(m, ref.select(m, tr)["chosen"], tr) for m in ("primary", "logridge", "endpoint", "anchor_constant")}))

    calce_loco_s1 = []
    if "CALCE_CS2" in ext_datasets:
        cs2 = ext_datasets["CALCE_CS2"]
        for i, held in enumerate(cs2):
            tr = [cs2[j] for j in range(len(cs2)) if j != i]
            calce_loco_s1.append((held, {m: ref.fit(m, ref.select(m, tr)["chosen"], tr) for m in ("primary", "logridge", "endpoint", "anchor_constant")}))

    thresholds = [0.90, 0.85, 0.80]
    strata_rows: List[Dict[str, Any]] = []

    def _collect_s1_strata(cname: str, pairs: List[Tuple[ref.Cell, Dict[str, Any]]]) -> None:
        for thr in thresholds:
            hi_e: Dict[str, List[float]] = {m: [] for m in ("primary", "logridge", "endpoint", "anchor_constant")}
            lo_e: Dict[str, List[float]] = {m: [] for m in ("primary", "logridge", "endpoint", "anchor_constant")}
            n_hi_cells, n_lo_cells = set(), set()
            n_hi_pts, n_lo_pts = 0, 0
            for held, mods in pairs:
                rows = ref._rows_for_predict(held)
                yt = np.asarray(held.y[rows], float)
                m_hi = yt > thr
                m_lo = yt <= thr
                n_hi_pts += int(np.sum(m_hi))
                n_lo_pts += int(np.sum(m_lo))
                if np.any(m_hi):
                    n_hi_cells.add(held.name)
                if np.any(m_lo):
                    n_lo_cells.add(held.name)
                for mname, mod in mods.items():
                    yh = ref.predict(mod, held, "clean")
                    err = np.abs(yh - yt) * 100.0
                    if np.any(m_hi):
                        hi_e[mname].append(float(np.mean(err[m_hi])))
                    if np.any(m_lo):
                        lo_e[mname].append(float(np.mean(err[m_lo])))
            strata_rows.append({
                "stage": "Stage1",
                "cohort": cname,
                "threshold_soh": thr,
                "hi_cells": len(n_hi_cells),
                "hi_profiles": n_hi_pts,
                "lo_cells": len(n_lo_cells),
                "lo_profiles": n_lo_pts,
                "hi_primary_pp": float(np.mean(hi_e["primary"])) if hi_e["primary"] else None,
                "lo_primary_pp": float(np.mean(lo_e["primary"])) if lo_e["primary"] else None,
                "hi_logridge_pp": float(np.mean(hi_e["logridge"])) if hi_e["logridge"] else None,
                "lo_logridge_pp": float(np.mean(lo_e["logridge"])) if lo_e["logridge"] else None,
                "hi_endpoint_pp": float(np.mean(hi_e["endpoint"])) if hi_e["endpoint"] else None,
                "lo_endpoint_pp": float(np.mean(lo_e["endpoint"])) if lo_e["endpoint"] else None,
                "hi_anchor_const_pp": float(np.mean(hi_e["anchor_constant"])) if hi_e["anchor_constant"] else None,
                "lo_anchor_const_pp": float(np.mean(lo_e["anchor_constant"])) if lo_e["anchor_constant"] else None,
            })

    _collect_s1_strata("XJTU_B1_LOCO", b1_loco_s1)
    _collect_s1_strata("XJTU_B3", [(c, m_b1_mods) for c in datasets["XJTU_B3"]])
    _collect_s1_strata("XJTU_B4", [(c, m_b1_mods) for c in datasets["XJTU_B4"]])
    _collect_s1_strata("XJTU_B2", [(c, m_b1_mods) for c in datasets["XJTU_B2"]])
    _collect_s1_strata("XJTU_B6", [(c, m_b1_mods) for c in datasets["XJTU_B6"]])
    _collect_s1_strata("NASA_LOCO", nasa_loco_s1)
    if calce_loco_s1:
        _collect_s1_strata("CALCE_CS2_LOCO", calce_loco_s1)
    if all(k in ext_datasets for k in ("TJU_1C", "TJU_2C", "TJU_4C")):
        t1 = ext_datasets["TJU_1C"]
        m_t1 = {m: ref.fit(m, ref.select(m, t1)["chosen"], t1) for m in ("primary", "logridge", "endpoint", "anchor_constant")}
        _collect_s1_strata("TJU_1C_to_2C", [(c, m_t1) for c in ext_datasets["TJU_2C"]])
        _collect_s1_strata("TJU_1C_to_4C", [(c, m_t1) for c in ext_datasets["TJU_4C"]])

    # Stage 2 knee sensitivity at relative health h_t = y/y_ref thresholds
    st_b1_frozen = s2.fit_stats(b1_cells, 1e-4)
    for cname, s2_pairs in [
        ("XJTU_B1_LOCO", b1_loco_s2),
        ("XJTU_B3", [(c, st_b1_frozen) for c in datasets["XJTU_B3"]]),
        ("XJTU_B4", [(c, st_b1_frozen) for c in datasets["XJTU_B4"]]),
        ("XJTU_B5", [(c, st_b1_frozen) for c in datasets["XJTU_B5"]]),
        ("XJTU_B6", [(c, st_b1_frozen) for c in datasets["XJTU_B6"]]),
    ]:
        for thr in thresholds:
            hi_p, lo_p, hi_s, lo_s = [], [], [], []
            n_lo_c, n_lo_n, n_hi_n = 0, 0, 0
            for held, st in s2_pairs:
                o_p = s2.predict("primary", st, {"lambda": 1e-4, "t": 10.0}, held, "clean", stride=1)
                o_s = s2.predict("target_secant", st, {"lambda": 1e-4, "t": 10.0}, held, "clean", stride=1)
                fp = np.asarray(o_p["future_positions"], int)
                ht = np.asarray(held.y[fp], float) / float(held.ref_y)
                m_hi, m_lo = ht > thr, ht <= thr
                n_hi_n += int(np.sum(m_hi))
                n_lo_n += int(np.sum(m_lo))
                if np.any(m_hi):
                    hi_p.append(100.0 * float(np.mean(np.abs(o_p["pred"] - ht)[m_hi])))
                    hi_s.append(100.0 * float(np.mean(np.abs(o_s["pred"] - ht)[m_hi])))
                if np.any(m_lo):
                    n_lo_c += 1
                    lo_p.append(100.0 * float(np.mean(np.abs(o_p["pred"] - ht)[m_lo])))
                    lo_s.append(100.0 * float(np.mean(np.abs(o_s["pred"] - ht)[m_lo])))
            strata_rows.append({
                "stage": "Stage2",
                "cohort": cname,
                "threshold_soh": thr,
                "hi_cells": len(s2_pairs),
                "hi_profiles": n_hi_n,
                "lo_cells": n_lo_c,
                "lo_profiles": n_lo_n,
                "hi_primary_pp": float(np.mean(hi_p)) if hi_p else None,
                "lo_primary_pp": float(np.mean(lo_p)) if lo_p else None,
                "hi_logridge_pp": float(np.mean(hi_s)) if hi_s else None,
                "lo_logridge_pp": float(np.mean(lo_s)) if lo_s else None,
                "hi_endpoint_pp": None,
                "lo_endpoint_pp": None,
                "hi_anchor_const_pp": None,
                "lo_anchor_const_pp": None,
            })

    # 15b: Label-free observable knee trigger test (without ground-truth y_t!)
    proxy_rows: List[Dict[str, Any]] = []
    proxy_cohorts = [
        ("XJTU_B1_LOCO", b1_loco_f1),
        ("XJTU_B4", [(c, f1_frozen) for c in datasets["XJTU_B4"]]),
        ("XJTU_B6", [(c, f1_frozen) for c in datasets["XJTU_B6"]]),
    ]
    if "CALCE_CS2" in ext_datasets:
        cs2 = ext_datasets["CALCE_CS2"]
        c_pairs = []
        for i, held in enumerate(cs2):
            tr = [cs2[j] for j in range(len(cs2)) if j != i]
            p1 = ref.select("primary", tr)["chosen"]
            f1_c = Stage1ResponseFilter(lam=float(p1["lambda"]), gamma=float(p1["gamma"]), eta=float(p1["eta"])).fit_from_cells(tr)
            c_pairs.append((held, f1_c))
        proxy_cohorts.append(("CALCE_CS2_LOCO", c_pairs))
    if all(k in ext_datasets for k in ("TJU_1C", "TJU_2C", "TJU_4C")):
        t1 = ext_datasets["TJU_1C"]
        p1 = ref.select("primary", t1)["chosen"]
        f1_t = Stage1ResponseFilter(lam=float(p1["lambda"]), gamma=float(p1["gamma"]), eta=float(p1["eta"])).fit_from_cells(t1)
        proxy_cohorts.append(("TJU_1C_to_2C", [(c, f1_t) for c in ext_datasets["TJU_2C"]]))
        proxy_cohorts.append(("TJU_1C_to_4C", [(c, f1_t) for c in ext_datasets["TJU_4C"]]))

    for cname, pairs in proxy_cohorts:
        for tau_hat in thresholds:
            pre_maes, trig_soh = [], []
            n_reach_80, n_caught_before_80 = 0, 0
            for held, f1_m in pairs:
                rows = ref._future_rows(held)
                yt = np.asarray(held.y[rows], float)
                D = np.asarray(held.D["clean"][rows], float)
                Dr = np.asarray(held.Dref["clean"], float)
                yr = float(held.ref_y)
                g_res = f1_m.predict_gated(D, Dr, yr)
                y_raw = np.asarray(g_res["y_hat_raw"], float)
                al = np.asarray(g_res["alarm"], bool)
                # Label-free endpoint partial-charge proxy y_endpt = y_ref * sum(D) / sum(D_ref)
                y_end = yr * (np.sum(D, axis=1) / float(np.sum(Dr)))
                # Recalibration trigger at first cycle where (y_hat_raw <= tau_hat) OR (y_end <= tau_hat) OR 3 consec alarms
                consec = 0
                trig_idx = None
                for idx in range(y_raw.size):
                    consec = (consec + 1) if bool(al[idx]) else 0
                    if (y_raw[idx] <= tau_hat) or (y_end[idx] <= tau_hat) or (consec >= 3):
                        trig_idx = idx
                        break
                has_lt80 = bool(np.any(yt < 0.80))
                if has_lt80:
                    n_reach_80 += 1
                    first_lt80 = int(np.argmax(yt < 0.80))
                    if trig_idx is not None and trig_idx <= first_lt80:
                        n_caught_before_80 += 1
                if trig_idx is not None:
                    trig_soh.append(float(yt[trig_idx]))
                end_slice = trig_idx if (trig_idx is not None and trig_idx > 0) else y_raw.size
                pre_maes.append(100.0 * float(np.mean(np.abs(y_raw[:end_slice] - yt[:end_slice]))))
            proxy_rows.append({
                "cohort": cname,
                "observable_tau_hat": tau_hat,
                "n_cells": len(pairs),
                "n_cells_reaching_lt080": n_reach_80,
                "n_triggered_before_080": n_caught_before_80,
                "med_true_soh_at_trigger": float(np.median(trig_soh)) if trig_soh else None,
                "pre_trigger_mae_pp": float(np.mean(pre_maes)),
            })

    pd.DataFrame(strata_rows).to_csv(out_dir / "e15_soh_knee_sensitivity.csv", index=False)
    pd.DataFrame(proxy_rows).to_csv(out_dir / "e15_observable_proxy_triggers.csv", index=False)
    payload = {"soh_knee_sensitivity": strata_rows, "observable_proxy_triggers": proxy_rows}
    _save_json(out_dir / "e15_knee_sensitivity_and_proxy.json", payload)
    print("[E15] Done.", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-hoc Major Revision audits E7-E15.")
    parser.add_argument("--old-data-dir", type=Path, default=_REPO_ROOT / "data")
    parser.add_argument("--new-data-dir", type=Path, default=None)
    parser.add_argument("--ext-data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "revision_experiments")
    parser.add_argument("--only", type=str, default="e7,e8,e9,e10,e11,e12,e13,e14,e15")
    args = parser.parse_args()

    old_dir = args.old_data_dir.resolve()
    new_dir = args.new_data_dir.resolve() if args.new_data_dir is not None else old_dir
    ext_dir = args.ext_data_dir.resolve() if args.ext_data_dir is not None else old_dir
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {s.strip().lower() for s in args.only.split(",") if s.strip()}

    t0 = time.time()
    datasets = _load_core_datasets(old_dir, new_dir)
    ext_datasets = _load_external_datasets(ext_dir)
    if "e7" in selected:
        run_e7(datasets, ext_datasets, out_dir)
    if "e8" in selected:
        run_e8(datasets, out_dir)
    if "e9" in selected:
        run_e9(datasets, old_dir, out_dir)
    if "e10" in selected:
        run_e10(datasets, ext_datasets, out_dir)
    if "e11" in selected:
        run_e11(datasets, out_dir)
    if "e12" in selected:
        run_e12(datasets, out_dir)
    if "e13" in selected:
        run_e13(datasets, ext_datasets, out_dir)
    if "e14" in selected:
        run_e14(datasets, ext_datasets, out_dir)
    if "e15" in selected:
        run_e15(datasets, ext_datasets, out_dir)
    print(f"MAJOR REVISION AUDITS ({','.join(sorted(selected))}) COMPLETE in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
