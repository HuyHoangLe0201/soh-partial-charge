#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-hoc Major Revision Audits E7-E12:
  E7:  Label-free oblique-projector Stage-1 compatibility gate across XJTU & external chemistries
  E8:  Causal online rebound-completion anchor selection & standardized innovation gating on XJTU B5 / B1-B6
  E9:  NASA Leave-One-Temperature-Regime-Out (LOTRO) vs 14-cell LOCO audit
  E10: External multi-chemistry validation on CALCE CS2 (LCO) and Tongji TJU D3 (NCM+NCA)
  E11: Asymmetric one-sided hardware voltage drift (future-only & anchor-only vs symmetric grid shift)
  E12: Stage-2 posterior predictive UQ coverage (sigma^2 vs nu=t*sigma^2) & embedded BMS compute footprint

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
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

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
    payload = {"uq_coverage": uq_rows, "bms_footprint": bms_footprint, "histgb_nodes": int(n_hgb_nodes)}
    _save_json(out_dir / "e12_uq_and_bms_benchmark.json", payload)
    print("[E12] Done.", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-hoc Major Revision audits E7-E12.")
    parser.add_argument("--old-data-dir", type=Path, default=_REPO_ROOT / "data")
    parser.add_argument("--new-data-dir", type=Path, default=None)
    parser.add_argument("--ext-data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "revision_experiments")
    parser.add_argument("--only", type=str, default="e7,e8,e9,e10,e11,e12")
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
    print(f"MAJOR REVISION AUDITS ({','.join(sorted(selected))}) COMPLETE in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
