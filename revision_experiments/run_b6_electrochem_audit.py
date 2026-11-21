#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment E3b: Raw .mat RPT pre-conditioning & full-voltage ICA (dQ/dV) audit
for XJTU Batch-1 (4A/2A), Batch-4 (4A/2A R3), and Batch-6 (2A/1A Sim_satellite).
Strictly read-only on raw files; outputs exclusively to revision_experiments/.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import savgol_filter

ROOT = Path(os.environ.get("SOH_WORK_ROOT", str(Path(__file__).resolve().parent.parent)))
RAW_BASE = Path(os.environ.get("SOH_RAW_BASE", str(ROOT / "raw_data")))
OUT = ROOT / "revision_experiments"

VFULL = np.round(np.linspace(3.20, 4.18, 99), 4)  # 0.01 V grid


def _contiguous_runs(mask: np.ndarray):
    n = mask.size
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        yield i, j
        i = j


def _trapz_q(i_arr: np.ndarray, t_sec: np.ndarray) -> np.ndarray:
    if i_arr.size < 2:
        return np.zeros(i_arr.size, dtype=float)
    dq = 0.5 * (i_arr[1:] + i_arr[:-1]) * np.diff(t_sec) / 3600.0
    return np.concatenate([[0.0], np.cumsum(dq)])


def _find_cc_charge(v: np.ndarray, i_arr: np.ndarray, t_sec: np.ndarray,
                    dis_s: int, ch_range: Tuple[float, float]) -> Tuple[int, int] | None:
    lo, hi = ch_range
    pre_i = i_arr[:dis_s]
    char_mask = (pre_i >= lo) & (pre_i <= hi)
    best = None
    for s, e in _contiguous_runs(char_mask):
        if (e - s) >= 30 and np.all(np.diff(t_sec[s:e]) > 0):
            mi = float(np.median(pre_i[s:e]))
            if lo <= mi <= hi:
                best = (int(s), int(e))
    return best


def _find_ref_dis(v: np.ndarray, i_arr: np.ndarray, t_sec: np.ndarray, target_dis: float) -> Tuple[int, int] | None:
    neg_mask = i_arr <= -0.2
    tol = 0.15 * abs(target_dis)
    matches = []
    for s, e in _contiguous_runs(neg_mask):
        if (e - s) < 30 or np.any(np.diff(t_sec[s:e]) <= 0):
            continue
        mi = float(np.median(i_arr[s:e]))
        if abs(mi - target_dis) <= tol and float(v[e - 1]) <= 2.55 and float(v[s]) >= 3.80:
            matches.append((int(s), int(e)))
    return matches[0] if len(matches) == 1 else None


def audit_cell_mat(label: str, mat_rel: str, ch_range: Tuple[float, float], target_dis: float,
                   cache_npz: Path | None) -> Dict[str, Any]:
    mat_path = RAW_BASE / mat_rel
    print(f"[E3b] Loading {mat_rel}...", flush=True)
    t0 = time.time()
    mat = sio.loadmat(str(mat_path), simplify_cells=True)
    print(f"[E3b] Loaded {mat_rel} in {time.time() - t0:.1f}s", flush=True)

    data_list = mat["data"]
    summary = mat["summary"]
    dis_caps = np.asarray(summary["discharge_capacity_Ah"], float).ravel()
    cap0 = float(dis_caps[0])
    n_cyc = min(len(data_list), len(dis_caps))

    first_cyc = data_list[0] if isinstance(data_list[0], dict) else (data_list[1] if isinstance(data_list[1], dict) else {})
    cyc_keys = sorted(list(first_cyc.keys())) if isinstance(first_cyc, dict) else []
    summary_keys = sorted(list(summary.keys())) if isinstance(summary, dict) else []

    # Determine kept record indices
    kept_recs: List[Tuple[int, int | None, int | None, int | None, int | None]] = []
    if cache_npz is not None and cache_npz.exists():
        with np.load(cache_npz, allow_pickle=False) as cz:
            ridxs = np.asarray(cz["record_idx"], int)
            cs = np.asarray(cz["charge_start_idx"], int)
            ce = np.asarray(cz["charge_end_idx"], int)
            ds = np.asarray(cz["dis_start_idx"], int)
            de = np.asarray(cz["dis_end_idx"], int)
            for r, a, b, c, d in zip(ridxs, cs, ce, ds, de):
                kept_recs.append((int(r), int(a), int(b), int(c), int(d)))
    else:
        for k in range(1, n_cyc):
            cyc = data_list[k]
            if not isinstance(cyc, dict):
                continue
            soh = float(dis_caps[k] / cap0)
            if not (0.5 <= soh <= 1.05):
                continue
            v = np.asarray(cyc["voltage_V"], float).ravel()
            i_arr = np.asarray(cyc["current_A"], float).ravel()
            t_sec = np.asarray(cyc["relative_time_min"], float).ravel() * 60.0
            dis_se = _find_ref_dis(v, i_arr, t_sec, target_dis)
            if dis_se is None:
                continue
            ch_se = _find_cc_charge(v, i_arr, t_sec, dis_se[0], ch_range)
            if ch_se is None:
                continue
            kept_recs.append((int(k), ch_se[0], ch_se[1], dis_se[0], dis_se[1]))

    # Select early (first 2), mid (2 around middle), late (last 2)
    nk = len(kept_recs)
    pick_pos = sorted(set([0, min(1, nk - 1), max(0, nk // 2 - 1), nk // 2, max(0, nk - 2), nk - 1]))
    stages_map = {
        pick_pos[0]: "early_0",
        pick_pos[min(1, len(pick_pos) - 1)]: "early_1",
    }
    mid_p = nk // 2
    # Clean stage tagging
    stage_entries: List[Tuple[str, Tuple[int, int | None, int | None, int | None, int | None]]] = []
    if nk >= 6:
        stage_entries = [
            ("early_1", kept_recs[0]),
            ("early_2", kept_recs[1]),
            ("mid_1", kept_recs[mid_p - 1]),
            ("mid_2", kept_recs[mid_p]),
            ("late_1", kept_recs[-2]),
            ("late_2", kept_recs[-1]),
        ]
    else:
        for idx, rec in enumerate(kept_recs):
            stage_entries.append((f"rec_{idx}", rec))

    precond_rows: List[Dict[str, Any]] = []
    ica_profiles: List[Dict[str, Any]] = []

    temp_key = None
    for cand_k in ("temperature_C", "temperature", "temp_C", "Temp_C"):
        if cand_k in cyc_keys:
            temp_key = cand_k
            break

    for stage_tag, (k, ch_s, ch_e, dis_s, dis_e) in stage_entries:
        cyc = data_list[k]
        v = np.asarray(cyc["voltage_V"], float).ravel()
        i_arr = np.asarray(cyc["current_A"], float).ravel()
        t_sec = np.asarray(cyc["relative_time_min"], float).ravel() * 60.0
        temp_arr = np.asarray(cyc[temp_key], float).ravel() if temp_key is not None else None
        desc = str(cyc.get("description", ""))[:120]

        # Inspect episode sequence before ch_s
        pre_rest_mask = np.abs(i_arr[:ch_s]) < 0.05 if ch_s > 0 else np.zeros(0, dtype=bool)
        # Trailing rest run immediately before ch_s
        rest_s = ch_s
        while rest_s > 0 and abs(i_arr[rest_s - 1]) < 0.05:
            rest_s -= 1
        rest_s_duration = float(t_sec[ch_s - 1] - t_sec[rest_s]) if ch_s > rest_s else 0.0
        if rest_s > 0 and ch_s > rest_s:
            # include step across boundary
            rest_s_duration = float(t_sec[ch_s] - t_sec[rest_s])

        # Prior discharge run strictly before rest_s (or before ch_s)
        prior_neg = np.where(i_arr[:ch_s] <= -0.2)[0]
        if prior_neg.size > 0:
            v_prior_dis_end = float(v[prior_neg[-1]])
            i_prior_dis_med = float(np.median(i_arr[prior_neg[-min(30, prior_neg.size):]]))
        else:
            v_prior_dis_end = float("nan")
            i_prior_dis_med = float("nan")

        v_rest_end = float(v[ch_s - 1]) if ch_s > 0 else float("nan")
        i_rest_end = float(i_arr[ch_s - 1]) if ch_s > 0 else float("nan")

        v_cc = v[ch_s:ch_e]
        i_cc = i_arr[ch_s:ch_e]
        t_cc = t_sec[ch_s:ch_e]
        q_cc = _trapz_q(i_cc, t_cc)

        i_cc_med = float(np.median(i_cc))
        v_cc_0 = float(v_cc[0])
        v_cc_1 = float(v_cc[1]) if v_cc.size > 1 else float(v_cc[0])
        dv_ohmic_0 = (v_cc_0 - v_rest_end) if math.isfinite(v_rest_end) else float("nan")
        dv_ohmic_1 = (v_cc_1 - v_rest_end) if math.isfinite(v_rest_end) else float("nan")
        r_ohmic_0_mohm = 1000.0 * dv_ohmic_0 / i_cc_med if (math.isfinite(dv_ohmic_0) and i_cc_med > 0) else float("nan")
        r_ohmic_1_mohm = 1000.0 * dv_ohmic_1 / i_cc_med if (math.isfinite(dv_ohmic_1) and i_cc_med > 0) else float("nan")

        # CV phase between ch_e and dis_s (positive current after CC)
        cv_end = dis_s
        cv_mask = i_arr[ch_e:dis_s] > 0.05
        if np.any(cv_mask):
            cv_idx = ch_e + np.where(cv_mask)[0]
            q_cv_Ah = float(np.sum(0.5 * (i_arr[cv_idx[:-1]] + i_arr[cv_idx[1:]]) * np.diff(t_sec[cv_idx]))) / 3600.0 if cv_idx.size > 1 else 0.0
        else:
            q_cv_Ah = 0.0

        # Check all episodes in record up to dis_s
        n_neg_before_ch = int(np.sum([1 for s, e in _contiguous_runs(i_arr[:ch_s] <= -0.2) if (e - s) >= 10]))
        n_pos_before_ch = int(np.sum([1 for s, e in _contiguous_runs(i_arr[:ch_s] >= 0.2) if (e - s) >= 10]))

        # Full-voltage ICA on [3.20, 4.18] V
        v_clean = np.maximum.accumulate(v_cc) + np.arange(v_cc.size) * 1e-9
        keep = np.concatenate([[True], np.diff(v_clean) > 0])
        q_full = np.interp(VFULL, v_clean[keep], q_cc[keep], left=np.nan, right=np.nan)
        dqdv = np.full(VFULL.size, np.nan, dtype=float)
        fin = np.where(np.isfinite(q_full))[0]
        v_peak_full = float("nan")
        dq_peak_full = float("nan")
        if fin.size >= 7:
            i0, i1 = fin[0], fin[-1]
            if np.all(np.isfinite(q_full[i0:i1 + 1])):
                deriv = savgol_filter(q_full[i0:i1 + 1], window_length=5, polyorder=2, deriv=1, delta=0.01)
                dqdv[i0:i1 + 1] = deriv
                dqdv[i0] = np.nan
                dqdv[i1] = np.nan
                if np.any(np.isfinite(dqdv)):
                    pk = int(np.nanargmax(dqdv))
                    v_peak_full = float(VFULL[pk])
                    dq_peak_full = float(dqdv[pk])

        # D0..D3 from q_full at GRID32
        g32 = np.linspace(3.80, 4.10, 32)
        q32 = np.interp(g32, v_clean[keep], q_cc[keep], left=np.nan, right=np.nan)
        d0 = float(q32[8] - q32[0])
        d1 = float(q32[16] - q32[8])
        d2 = float(q32[24] - q32[16])
        d3 = float(q32[31] - q32[24])

        precond_rows.append({
            "batch_cell": label,
            "mat_rel": mat_rel,
            "stage": stage_tag,
            "record_idx": int(k),
            "n_kept_total": int(nk),
            "soh": float(dis_caps[k] / cap0),
            "summary_dis_Ah": float(dis_caps[k]),
            "description": desc,
            "n_neg_runs_before_ch": n_neg_before_ch,
            "n_pos_runs_before_ch": n_pos_before_ch,
            "v_prior_dis_end": v_prior_dis_end,
            "i_prior_dis_med": i_prior_dis_med,
            "rest_duration_before_cc_s": rest_s_duration,
            "v_rest_end": v_rest_end,
            "v_cc_start": v_cc_0,
            "v_cc_second": v_cc_1,
            "dv_ohmic_step0_V": dv_ohmic_0,
            "dv_ohmic_step1_V": dv_ohmic_1,
            "r_ohmic_step0_mOhm": r_ohmic_0_mohm,
            "r_ohmic_step1_mOhm": r_ohmic_1_mohm,
            "v_cc_min": float(np.min(v_cc)),
            "v_cc_max": float(np.max(v_cc)),
            "i_cc_median_A": i_cc_med,
            "cc_duration_s": float(t_cc[-1] - t_cc[0]),
            "q_cc_total_Ah": float(q_cc[-1]),
            "q_cv_Ah": q_cv_Ah,
            "temp_cc_mean_C": float(np.mean(temp_arr[ch_s:ch_e])) if temp_arr is not None else float("nan"),
            "v_peak_full_V": v_peak_full,
            "dqdv_peak_full": dq_peak_full,
            "D0": d0, "D1": d1, "D2": d2, "D3": d3,
        })
        ica_profiles.append({
            "batch_cell": label,
            "stage": stage_tag,
            "record_idx": int(k),
            "soh": float(dis_caps[k] / cap0),
            "VFULL": VFULL.tolist(),
            "q_full": q_full.tolist(),
            "dqdv_full": dqdv.tolist(),
        })

    del mat, data_list, summary
    gc.collect()
    return {
        "batch_cell": label,
        "mat_rel": mat_rel,
        "cyc_keys": cyc_keys,
        "summary_keys": summary_keys,
        "cap0_Ah": cap0,
        "n_cyc": int(n_cyc),
        "n_kept": int(nk),
        "precond_rows": precond_rows,
        "ica_profiles": ica_profiles,
    }


def main() -> None:
    global ROOT, RAW_BASE, OUT
    parser = argparse.ArgumentParser(description="Run E3b raw .mat RPT preconditioning and ICA audit.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Working root (default: SOH_WORK_ROOT env or repo parent).")
    parser.add_argument("--raw-base", type=Path, default=RAW_BASE, help="Raw XJTU Batch-* directory (default: SOH_RAW_BASE).")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    RAW_BASE = args.raw_base.resolve()
    OUT = ROOT / "revision_experiments"
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [
        ("B1_2C_1", "Batch-1/2C_battery-1.mat", (3.0, 4.8), -2.0, None),
        ("B4_R3_1", "Batch-4/R3_battery-1.mat", (3.0, 4.8), -2.0, ROOT / "data" / "cells" / "B4" / "R3_battery-1.mat".replace(".mat", ".npz")),
        ("B6_Sat_1", "Batch-6/Sim_satellite_battery-1.mat", (1.5, 2.4), -1.0, ROOT / "data" / "cells" / "B6" / "Sim_satellite_battery-1.npz"),
        ("B6_Sat_2", "Batch-6/Sim_satellite_battery-2.mat", (1.5, 2.4), -1.0, ROOT / "data" / "cells" / "B6" / "Sim_satellite_battery-2.npz"),
    ]
    all_rows = []
    cell_summaries = {}
    for label, rel, ch_r, t_dis, cpath in targets:
        res = audit_cell_mat(label, rel, ch_r, t_dis, cpath)
        all_rows.extend(res["precond_rows"])
        cell_summaries[label] = {
            "mat_rel": res["mat_rel"],
            "cyc_keys": res["cyc_keys"],
            "summary_keys": res["summary_keys"],
            "cap0_Ah": res["cap0_Ah"],
            "n_cyc": res["n_cyc"],
            "n_kept": res["n_kept"],
            "precond_rows": res["precond_rows"],
            "ica_profiles": res["ica_profiles"],
        }

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "e3b_rpt_preconditioning.csv", index=False)

    def _san(o):
        if isinstance(o, dict): return {str(k): _san(v) for k, v in o.items()}
        if isinstance(o, list): return [_san(v) for v in o]
        if isinstance(o, float): return o if math.isfinite(o) else None
        return o

    (OUT / "e3b_raw_mat_audit.json").write_text(json.dumps(_san(cell_summaries), indent=2), encoding="utf-8")
    print("[E3b] Complete.", flush=True)


if __name__ == "__main__":
    main()
