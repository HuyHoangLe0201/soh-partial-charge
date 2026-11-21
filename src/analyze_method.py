#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Postprocess already-run SOH experiments. Propose only; no execution here."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ANALYSIS = ROOT / "analysis"
FIGDIR = ANALYSIS / "figures"

TO_PP = 100.0
TOL = 2e-8
SEED = 20261107
N_BOOT = 20000

PRIMARY = "primary"
KEY_MODELS = ["primary", "logridge", "rawridge", "absridge", "histgb", "best_segment", "endpoint", "anchor_constant"]
STRESS_MODELS = ["primary", "logridge", "uncalibrated", "without_nuisance"]
COMPARATORS_FOR_PAIRING = ["logridge", "endpoint"]
ALL_COMPARATORS = None  # filled in main from data

PROTOCOLS = ["XJTU_B4", "XJTU_B6"]
COHORT_GROUPS = ["confirmation", "development", "loco_XJTU_B1", "loco_NASA"]

# Exact scenario mapping: name -> voltage offset in mV.
# clean is the zero-offset baselines; vp10/vm10 are stressed at 10mV; ±5 is the +/-5 pairing set.
SCENARIO_OFFSET = {
    "clean": 0,
    "vp5": +5,
    "vm5": -5,
    "vp10": +10,
    "vm10": -10,
}
CLEAN_SCENARIO = "clean"

# All scenarios we expect per cell/model.
EXPECTED_SCENARIOS = set(SCENARIO_OFFSET) | {'gstatp1','gstatm1','gdriftp1','gdriftm1','grefp1','grefm1'}
N_EXPECTED_SCENARIOS = len(EXPECTED_SCENARIOS)
N_EXPECTED_MODELS = 15
N_EXPECTED_CELLS_PER_PROTOCOL = 8


# ----------------------------- helpers -----------------------------
def die(msg: str) -> None:
    raise RuntimeError(msg)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def finite_or_none(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def pp(x) -> Optional[float]:
    v = finite_or_none(x)
    return None if v is None else TO_PP * v


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return finite_or_none(obj)
    if isinstance(obj, float):
        return finite_or_none(obj)
    return obj


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_json(obj), f, indent=2, allow_nan=False)


def safe_save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, na_rep="")


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame]:
    preds, mets = [], []
    for group in COHORT_GROUPS:
        d = RESULTS / group
        if not d.exists():
            die(f"Required results directory missing: {d}. Refusing silent exclusion of a running or failed experiment.")
        p = d / "predictions.csv"
        m = d / "cell_metrics.csv"
        if not p.exists() or not m.exists():
            die(f"Required files missing in {d}: need predictions.csv and cell_metrics.csv.")
        try:
            pdf = pd.read_csv(p)
            mdf = pd.read_csv(m)
        except Exception as e:
            die(f"Failed reading CSVs in {d}: {e}")
        if pdf.empty or mdf.empty:
            die(f"Empty CSV in {d}.")
        pdf["_group"] = group
        mdf["_group"] = group
        if group.startswith('loco_'):
            ds = group.removeprefix('loco_')
            for df in (pdf,mdf):
                df.rename(columns={'outer_cell':'cell'},inplace=True)
                df['dataset']=ds
        preds.append(pdf)
        mets.append(mdf)
    allp = pd.concat(preds, ignore_index=True)
    allm = pd.concat(mets, ignore_index=True)
    need_p = {"dataset", "cell", "cycle", "row_index", "ref_cycle", "model", "scenario", "y_true", "y_pred"}
    need_m = {"dataset", "cell", "model", "scenario", "n", "mae", "rmse", "bias", "ref_cycle"}
    if not need_p.issubset(allp.columns):
        die(f"predictions.csv missing columns: {sorted(need_p - set(allp.columns))}")
    if not need_m.issubset(allm.columns):
        die(f"cell_metrics.csv missing columns: {sorted(need_m - set(allm.columns))}")
    for df in (allp, allm):
        df["dataset"] = df["dataset"].astype(str)
        df["cell"] = df["cell"].astype(str)
        df["model"] = df["model"].astype(str)
        df["scenario"] = df["scenario"].fillna("").astype(str)
    return allp, allm


def validate_uniqueness(df: pd.DataFrame, keys: List[str], name: str) -> None:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys].head(5).to_dict("records")
        die(f"Duplicate keys in {name} for {keys}; examples: {sample}")


def recompute_metrics(preds: pd.DataFrame, saved: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p = preds.copy()
    for c in ["cycle", "row_index", "ref_cycle", "y_true", "y_pred"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    if p[["cycle", "row_index", "ref_cycle", "y_true", "y_pred"]].isna().any().any():
        bad = p[p[["cycle", "row_index", "ref_cycle", "y_true", "y_pred"]].isna().any(axis=1)].head(5)
        die(f"Non-numeric/missing values in predictions: {bad.to_dict('records')}")
    bad_ref = p[p["cycle"] <= p["ref_cycle"]]
    if not bad_ref.empty:
        die(f"Found {len(bad_ref)} prediction rows with cycle<=ref_cycle; ref cycles must be excluded.")
    validate_uniqueness(p, ["dataset", "cell", "model", "scenario", "row_index"], "predictions")
    g = p.groupby(["dataset", "cell", "model", "scenario"], dropna=False)
    rec = g.apply(lambda d: pd.Series({
        "n": int(len(d)),
        "mae": float(np.mean(np.abs(d["y_pred"].to_numpy() - d["y_true"].to_numpy()))),
        "rmse": float(np.sqrt(np.mean((d["y_pred"].to_numpy() - d["y_true"].to_numpy()) ** 2))),
        "bias": float(np.mean(d["y_pred"].to_numpy() - d["y_true"].to_numpy())),
        "ref_cycle": int(d["ref_cycle"].iloc[0]),
    })).reset_index()
    s = saved.copy()
    for c in ["n", "mae", "rmse", "bias", "ref_cycle"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    keys_join = ["dataset", "cell", "model", "scenario"]
    merged = rec.merge(s, on=keys_join, how="outer", suffixes=("_rec", "_sav"), indicator=True)
    if (merged["_merge"] != "both").any():
        bad = merged[merged["_merge"] != "both"].head(10)
        die(f"Cell-metric key mismatch between recomputed and saved: {bad[keys_join + ['_merge']].to_dict('records')}")
    for col in ["n", "mae", "rmse", "bias", "ref_cycle"]:
        diff = np.abs(merged[f"{col}_rec"].to_numpy(dtype=float) - merged[f"{col}_sav"].to_numpy(dtype=float))
        if not np.all(np.isfinite(diff)) or np.any(diff > TOL):
            j = int(np.nanargmax(diff))
            die(f"Recomputed metric mismatch for {col}: max abs diff {diff[j]:.3e} at {merged.iloc[j][keys_join].to_dict()}")
    return rec, p


def cell_mae_table(preds: pd.DataFrame, selector) -> pd.DataFrame:
    p = preds[selector(preds)].copy()
    if p.empty:
        return pd.DataFrame(columns=["dataset", "cell", "model", "scenario", "n", "mae", "rmse", "bias", "ref_cycle"])
    g = p.groupby(["dataset", "cell", "model", "scenario"], dropna=False)
    out = g.apply(lambda d: pd.Series({
        "n": int(len(d)),
        "mae": float(np.mean(np.abs(d["y_pred"].to_numpy() - d["y_true"].to_numpy()))),
        "rmse": float(np.sqrt(np.mean((d["y_pred"].to_numpy() - d["y_true"].to_numpy()) ** 2))),
        "bias": float(np.mean(d["y_pred"].to_numpy() - d["y_true"].to_numpy())),
        "ref_cycle": int(d["ref_cycle"].iloc[0]),
    })).reset_index()
    return out


def scenario_in_scenarios(preds: pd.DataFrame) -> pd.Series:
    return preds["scenario"].isin(SCENARIO_OFFSET)


def scenario_is_clean(preds: pd.DataFrame) -> pd.Series:
    return preds["scenario"] == CLEAN_SCENARIO


def scenario_has_offset(preds: pd.DataFrame, offset_mv: int) -> pd.Series:
    wanted = {k for k, v in SCENARIO_OFFSET.items() if v == offset_mv}
    return preds["scenario"].isin(wanted)


def sanity_check_schema(preds: pd.DataFrame) -> Dict:
    # Assert all 15 models appear and exactly the 11 expected scenarios per cell.
    grp_models = preds.groupby(["dataset", "cell"])["model"].nunique()
    bad_mod = grp_models[grp_models != N_EXPECTED_MODELS]
    if not bad_mod.empty:
        die(f"Expected {N_EXPECTED_MODELS} models per (dataset,cell); offending groups: {bad_mod.head(10).to_dict()}")
    grp_scen = preds.groupby(["dataset", "cell", "model"])["scenario"].nunique()
    bad_scen = grp_scen[grp_scen != N_EXPECTED_SCENARIOS]
    if not bad_scen.empty:
        die(f"Expected {N_EXPECTED_SCENARIOS} scenarios per (dataset,cell); offending groups: {bad_scen.head(10).to_dict()}")
    # Assert the actual scenario set matches expected, per (dataset,cell)
    per_cell = preds.groupby(["dataset", "cell"])["scenario"].agg(lambda s: frozenset(s.unique()))
    for key, s in per_cell.items():
        if s != frozenset(EXPECTED_SCENARIOS):
            die(f"Scenario set mismatch at {key}: got {sorted(s)} expected {sorted(EXPECTED_SCENARIOS)}")
    # Protocol cells: exactly 8 per protocol, distinct
    for proto in PROTOCOLS:
        sub = preds[preds["dataset"] == proto]
        if sub.empty:
            die(f"No predictions for protocol {proto}")
        cells = sorted(sub["cell"].unique().tolist())
        if len(cells) != N_EXPECTED_CELLS_PER_PROTOCOL:
            die(f"Protocol {proto}: expected {N_EXPECTED_CELLS_PER_PROTOCOL} cells, got {len(cells)}: {cells}")
    all_ds_models = preds.groupby("dataset")["model"].nunique()
    if (all_ds_models != N_EXPECTED_MODELS).any():
        die(f"All datasets must have all {N_EXPECTED_MODELS} models; got {all_ds_models.to_dict()}")
    return {
        "n_models": int(preds["model"].nunique()),
        "n_scenarios": int(preds["scenario"].nunique()),
        "scenarios": sorted(preds["scenario"].unique().tolist()),
        "protocol_cells": {
            proto: sorted(preds.loc[preds["dataset"] == proto, "cell"].unique().tolist())
            for proto in PROTOCOLS
        },
    }


def make_figures(cell_clean: pd.DataFrame, cell_stress: pd.DataFrame) -> None:
    ensure_dir(FIGDIR)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    # ---- Figure 1: clean cell-mean MAE, KEY_MODELS x datasets ----
    agg = cell_clean.groupby(["dataset", "model"], dropna=False)["mae"].mean().reset_index()
    datasets = sorted(agg["dataset"].unique().tolist())
    if datasets:
        mat = pd.DataFrame(index=KEY_MODELS, columns=datasets, dtype=float)
        for _, r in agg.iterrows():
            if r["model"] in mat.index and r["dataset"] in mat.columns:
                mat.loc[r["model"], r["dataset"]] = TO_PP * float(r["mae"])
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        vals = mat.to_numpy(dtype=float)
        im = ax.imshow(vals, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(datasets)), datasets, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(KEY_MODELS)), KEY_MODELS)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > np.nanmedian(vals) else "black", fontsize=6)
        ax.set_title("Clean cell-mean MAE (pp). All datasets; key models shown")
        ax.set_xlabel("Dataset")
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("MAE (pp)")
        fig.savefig(FIGDIR / "fig1_clean_cellmean_mae.pdf")
        fig.savefig(FIGDIR / "fig1_clean_cellmean_mae.png")
        plt.close(fig)

    # ---- Figure 2: per protocol, sorted 8 cells, primary vs logridge vs endpoint (clean) ----
    if not cell_clean.empty:
        fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(7.6, 3.6), constrained_layout=True, squeeze=False)
        for j, proto in enumerate(PROTOCOLS):
            ax = axes[0, j]
            sub = cell_clean[(cell_clean["dataset"] == proto) & (cell_clean["scenario"] == CLEAN_SCENARIO)].copy()
            if sub.empty:
                ax.set_title(f"{proto}: no clean rows")
                continue
            piv = sub.pivot_table(index="cell", columns="model", values="mae", aggfunc="mean")
            # sort cells by primary MAE for readability
            if PRIMARY in piv.columns:
                piv = piv.sort_values(PRIMARY)
            cells_sorted = piv.index.tolist()
            x = np.arange(len(cells_sorted))
            series = [PRIMARY, "logridge", "endpoint"]
            markers = {"primary": "o", "logridge": "s", "endpoint": "^"}
            for m in series:
                if m not in piv.columns:
                    continue
                y = TO_PP * piv.loc[cells_sorted, m].to_numpy(dtype=float)
                ax.plot(x, y, marker=markers[m], linestyle="-", linewidth=1.0, markersize=4, label=m)
            ax.set_xticks(x, cells_sorted, rotation=45, ha="right", fontsize=6)
            ax.set_title(f"{proto}: clean cell MAE (n={len(cells_sorted)})")
            ax.set_ylabel("Cell MAE (pp)")
            ax.set_xlabel("Cell")
            ax.legend(frameon=False, loc="best")
        fig.suptitle("Clean per-cell MAE: primary vs logridge vs endpoint (pp); cells sorted by primary", fontsize=9)
        fig.savefig(FIGDIR / "fig2_paired_errors.pdf")
        fig.savefig(FIGDIR / "fig2_paired_errors.png")
        plt.close(fig)

    # ---- Figure 3: voltage stress curves per protocol; filter by dataset == protocol ----
    if not cell_stress.empty:
        fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(7.2, 3.4), constrained_layout=True, squeeze=False, sharey=True)
        stress_levels = sorted(set(SCENARIO_OFFSET.values()))
        for j, proto in enumerate(PROTOCOLS):
            ax = axes[0, j]
            sub_ds = cell_stress[cell_stress["dataset"] == proto]
            for m in STRESS_MODELS:
                xs, ys = [], []
                sub_m = sub_ds[sub_ds["model"] == m]
                for scen_name, lvl in SCENARIO_OFFSET.items():
                    s = sub_m[sub_m["scenario"] == scen_name]
                    if not s.empty:
                        xs.append(lvl)
                        ys.append(TO_PP * float(s["mae"].mean()))
                if xs:
                    order = np.argsort(xs)
                    ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o", label=m)
            ax.set_title(f"{proto}: voltage stress")
            ax.set_xlabel("Voltage offset (mV)")
            ax.set_ylabel("Cell-mean MAE (pp)")
            ax.set_xticks(stress_levels)
            ax.legend(frameon=False, loc="best")
        fig.suptitle("Voltage stress curves (cell means, no CI band)", fontsize=9)
        fig.savefig(FIGDIR / "fig3_voltage_stress.pdf")
        fig.savefig(FIGDIR / "fig3_voltage_stress.png")
        plt.close(fig)


def paired_diffs_for_protocol(cell_df: pd.DataFrame, proto: str, scenario_name: str, comparator: str) -> pd.DataFrame:
    """Return per-cell rows (actual cells within the protocol) with primary and comparator MAE."""
    sub = cell_df[(cell_df["dataset"] == proto) & (cell_df["scenario"] == scenario_name)]
    if sub.empty:
        return pd.DataFrame(columns=["dataset", "cell", "primary_mae", "comparator_mae", "diff"])
    piv = sub.pivot_table(index=["dataset", "cell"], columns="model", values="mae", aggfunc="mean")
    if PRIMARY not in piv.columns or comparator not in piv.columns:
        return pd.DataFrame(columns=["dataset", "cell", "primary_mae", "comparator_mae", "diff"])
    out = piv[[PRIMARY, comparator]].dropna().reset_index()
    out = out.rename(columns={PRIMARY: "primary_mae", comparator: "comparator_mae"})
    out["diff"] = out["primary_mae"] - out["comparator_mae"]
    return out


def bootstrap_protocol_equal(diffs_by_proto: Dict[str, np.ndarray], n_boot: int, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    names = [k for k, v in diffs_by_proto.items() if len(v) > 0]
    if not names:
        return {"mean_pp": None, "ci95_pp": [None, None], "n_protocols": 0,
                "n_cells_by_protocol": {}, "n_boot": n_boot, "seed": seed}
    means = []
    for _ in range(n_boot):
        mm = []
        for k in names:
            arr = diffs_by_proto[k]
            idx = rng.integers(0, len(arr), size=len(arr))
            mm.append(float(np.mean(arr[idx])))
        means.append(float(np.mean(mm)))
    means = np.asarray(means, dtype=float)
    obs = float(np.mean([np.mean(diffs_by_proto[k]) for k in names]))
    return {
        "mean_pp": pp(obs),
        "ci95_pp": [pp(np.percentile(means, 2.5)), pp(np.percentile(means, 97.5))],
        "n_protocols": len(names),
        "n_cells_by_protocol": {k: int(len(diffs_by_proto[k])) for k in names},
        "n_boot": int(n_boot),
        "seed": int(seed),
        "note": "equal-weight protocol means; percentile95 CI descriptive",
    }


def extract_primary_metadata() -> Dict:
    p = RESULTS / "source_XJTU_B1" / "config_freeze.json"
    if not p.exists():
        return {"available": False, "path": str(p), "error": "missing"}
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"available": False, "path": str(p), "error": f"parse failed: {e}"}
    if not isinstance(cfg, dict):
        return {"available": True, "path": str(p), "model_payload": None, "note": "cfg is not a dict"}
    payload = cfg.get("model_payloads", {}).get("primary")
    if payload is None and "primary" in cfg and isinstance(cfg["primary"], dict):
        payload = cfg["primary"]
    if isinstance(payload, dict):
        numeric = {k: finite_or_none(v) for k, v in payload.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    else:
        numeric = {}
    return {"available": True, "path": str(p), "model_payload": payload, "model_numeric": numeric}


def read_confirmation_success_criterion() -> Dict:
    p = RESULTS / "confirmation" / "confirmation_summary.json"
    if not p.exists():
        p = RESULTS / "confirmation" / "summary.json"
    if not p.exists():
        return {"found": False, "note": "No confirmation_summary.json; screen computed independently."}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"found": True, "path": str(p), "error": f"parse failed: {e}"}
    if not isinstance(obj, dict):
        return {"found": True, "path": str(p), "error": "top-level JSON is not an object"}
    return {"found": True, "path": str(p), "success_criterion": obj.get("success_criterion"), "top_level_keys": sorted(obj.keys())}


def main() -> None:
    ensure_dir(ANALYSIS)
    preds, saved = load_all()
    rec, preds = recompute_metrics(preds, saved)
    schema_report = sanity_check_schema(preds)

    # ref_cycle consistency within cell/model/scenario
    ref_chk = preds.groupby(["dataset", "cell", "model", "scenario"], dropna=False)["ref_cycle"].nunique()
    if (ref_chk > 1).any():
        bad = ref_chk[ref_chk > 1].head(5).to_dict()
        die(f"Inconsistent ref_cycle within cells/models/scenarios: {bad}")

    # Clean + stress cell tables (all models, all scenarios)
    cell_clean = cell_mae_table(preds, scenario_is_clean)
    cell_stress = cell_mae_table(preds, scenario_in_scenarios)

    all_models = sorted(preds["model"].unique().tolist())
    all_datasets = sorted(preds["dataset"].unique().tolist())

    # ---- clean_mat: ALL models x ALL datasets (mean over cells within dataset/model, clean scenario) ----
    clean_agg = cell_clean.groupby(["dataset", "model"], dropna=False)["mae"].mean().reset_index()
    clean_mat = pd.DataFrame(index=all_models, columns=all_datasets, dtype=float)
    for _, r in clean_agg.iterrows():
        clean_mat.loc[r["model"], r["dataset"]] = TO_PP * float(r["mae"])
    clean_mat = clean_mat.reset_index().rename(columns={"index": "model"})
    safe_save_csv(clean_mat, ANALYSIS / "clean_mae_pp.csv")

    # ---- stress_mat: dataset/model/scenario, no averaging across scenarios ----
    stress_agg = cell_stress.groupby(["dataset", "model", "scenario"], dropna=False)["mae"].mean().reset_index()
    stress_agg["mae_pp"] = stress_agg["mae"].map(pp)
    stress_agg["offset_mv"] = stress_agg["scenario"].map(SCENARIO_OFFSET)
    stress_mat = stress_agg[["dataset", "model", "scenario", "offset_mv", "mae_pp"]].sort_values(
        ["dataset", "model", "offset_mv"]).reset_index(drop=True)
    safe_save_csv(stress_mat, ANALYSIS / "stress_mae_pp.csv")

    # ---- Low-SOH strata: dataset/model/scenario/stratum, cells aggregated (never scenarios) ----
    low_rows = []
    for (ds, cell, model, scen), d in preds.groupby(["dataset", "cell", "model", "scenario"], dropna=False):
        yt = d["y_true"].to_numpy(dtype=float)
        yp = d["y_pred"].to_numpy(dtype=float)
        for label, mask in (("low", yt <= 0.85), ("high", yt > 0.85)):
            if mask.any():
                low_rows.append({
                    "dataset": ds, "cell": cell, "model": model, "scenario": scen,
                    "stratum": label, "n_rows": int(mask.sum()),
                    "mae": float(np.mean(np.abs(yp[mask] - yt[mask]))),
                })
    low_cell = pd.DataFrame(low_rows)
    low_avg_records: List[Dict] = []
    if not low_cell.empty:
        low_avg = low_cell.groupby(["dataset", "model", "scenario", "stratum"], dropna=False).agg(
            n_cells=("cell", "nunique"),
            n_rows=("n_rows", "sum"),
            mae=("mae", "mean"),
        ).reset_index()
        low_avg["mae_pp"] = low_avg["mae"].map(pp)
        safe_save_csv(low_avg[["dataset", "model", "scenario", "stratum", "n_cells", "n_rows", "mae_pp"]],
                      ANALYSIS / "low_soh_mae_pp.csv")
        low_avg_records = sanitize_json(low_avg[["dataset", "model", "scenario", "stratum", "n_cells", "n_rows", "mae_pp"]].to_dict("records"))

    # ---- Paired diffs primary-minus-comparator within EACH protocol (clean, ±5 mV) for ALL comparators ----
    comparators = [m for m in all_models if m != PRIMARY]
    paired_scenarios = {
        "clean": CLEAN_SCENARIO,
        "minus5mV": "vm5",
        "plus5mV": "vp5",
    }
    paired_rows: List[Dict] = []
    paired_summary: Dict = {
        "protocols": PROTOCOLS,
        "scenario_map": SCENARIO_OFFSET,
        "n_cells_per_protocol_target": N_EXPECTED_CELLS_PER_PROTOCOL,
        "comparisons": {},
        "bootstrap_config": {"n_boot": N_BOOT, "seed": SEED,
                             "note": "resample per protocol only when n==8; equal-weight protocol mean; percentile95 descriptive"},
    }

    for scen_label, scen_name in paired_scenarios.items():
        for comp in comparators:
            diffs_by_proto: Dict[str, np.ndarray] = {}
            protocol_specific: Dict[str, Dict] = {}
            for proto in PROTOCOLS:
                dd = paired_diffs_for_protocol(cell_clean, proto, scen_name, comp)
                if dd.empty:
                    continue
                for _, r in dd.iterrows():
                    paired_rows.append({
                        "scenario": scen_label,
                        "scenario_raw": scen_name,
                        "comparator": comp,
                        "dataset": proto,
                        "cell": r["cell"],
                        "primary_mae_pp": TO_PP * float(r["primary_mae"]),
                        "comparator_mae_pp": TO_PP * float(r["comparator_mae"]),
                        "diff_pp": TO_PP * float(r["diff"]),
                    })
                arr = dd["diff"].to_numpy(dtype=float)
                if len(arr) == N_EXPECTED_CELLS_PER_PROTOCOL:
                    diffs_by_proto[proto] = arr
                protocol_specific[proto] = {
                    "mean_diff_pp": pp(float(np.mean(arr))),
                    "n_cells": int(len(arr)),
                    "wins_primary": int(np.sum(arr < 0)),
                    "in_bootstrap": bool(len(arr) == N_EXPECTED_CELLS_PER_PROTOCOL),
                }
            if not diffs_by_proto:
                paired_summary["comparisons"].setdefault(scen_label, {})[comp] = {
                    "bootstrap": {"mean_pp": None, "ci95_pp": [None, None], "n_protocols": 0},
                    "protocol_specific": protocol_specific,
                    "note": "no protocol with exactly n=8 cells; bootstrap skipped",
                }
                continue
            boot = bootstrap_protocol_equal(diffs_by_proto, N_BOOT, SEED)
            paired_summary["comparisons"].setdefault(scen_label, {})[comp] = {
                "bootstrap": boot,
                "protocol_specific": protocol_specific,
            }

    paired_df = pd.DataFrame(paired_rows)
    if not paired_df.empty:
        safe_save_csv(paired_df, ANALYSIS / "paired_cells.csv")

    write_json(ANALYSIS / "paired_summary.json", paired_summary)

    # ---- Screen: pooled two-protocol mean MAE >=10% reduction vs BOTH endpoint & logridge;
    #      AND no individual protocol >0.25 pp worse than either comparator ----
    SCREEN_COMPARATORS = ["endpoint", "logridge"]
    REDUCTION_THRESHOLD = 0.10
    WORSE_PP_THRESHOLD = 0.25  # in pp
    screen: Dict = {
        "comparators": SCREEN_COMPARATORS,
        "reduction_threshold": REDUCTION_THRESHOLD,
        "worse_threshold_pp": WORSE_PP_THRESHOLD,
        "scope": "pooled two-protocol mean MAE over cells (B4 and B6 pooled); no averaging across scenarios",
        "result": {},
    }
    per_comp_protocol_means: Dict[str, Dict[str, Optional[float]]] = {}
    for ds in PROTOCOLS:
        sub = cell_clean[(cell_clean["dataset"] == ds) & (cell_clean["scenario"] == CLEAN_SCENARIO)]
        if sub.empty:
            per_comp_protocol_means[ds] = {PRIMARY: None, **{c: None for c in SCREEN_COMPARATORS}}
            continue
        piv = sub.pivot_table(index="cell", columns="model", values="mae", aggfunc="mean")
        rows = {PRIMARY: piv[PRIMARY].dropna() if PRIMARY in piv.columns else pd.Series(dtype=float)}
        for c in SCREEN_COMPARATORS:
            rows[c] = piv[c].dropna() if c in piv.columns else pd.Series(dtype=float)
        per_comp_protocol_means[ds] = {k: (float(v.mean()) if len(v) else None) for k, v in rows.items()}

    # pooled primary/comparators over cells across BOTH protocols
    pooled_sub = cell_clean[(cell_clean["dataset"].isin(PROTOCOLS)) & (cell_clean["scenario"] == CLEAN_SCENARIO)]
    pooled_piv = pooled_sub.pivot_table(index=["dataset", "cell"], columns="model", values="mae", aggfunc="mean")
    pooled_primary = pooled_piv[PRIMARY].dropna() if PRIMARY in pooled_piv.columns else pd.Series(dtype=float)

    per_comp: Dict[str, Dict] = {}
    ok_comps: List[str] = []
    for comp in SCREEN_COMPARATORS:
        if comp not in pooled_piv.columns or PRIMARY not in pooled_piv.columns:
            per_comp[comp] = {"available": False}
            continue
        dd = pooled_piv[[PRIMARY, comp]].dropna()
        if dd.empty:
            per_comp[comp] = {"available": False}
            continue
        mae_p = float(dd[PRIMARY].mean())
        mae_c = float(dd[comp].mean())
        rel_red = (1.0 - mae_p / mae_c) if mae_c != 0 else None
        proto_worse: Dict[str, bool] = {}
        for ds in PROTOCOLS:
            sub = cell_clean[(cell_clean["dataset"] == ds) & (cell_clean["scenario"] == CLEAN_SCENARIO)]
            if sub.empty:
                proto_worse[ds] = False
                continue
            p2 = sub.pivot_table(index="cell", columns="model", values="mae", aggfunc="mean")
            if PRIMARY in p2.columns and comp in p2.columns:
                d2 = (p2[PRIMARY] - p2[comp]).dropna()
                proto_worse[ds] = bool(len(d2) and (TO_PP * float(d2.mean()) > WORSE_PP_THRESHOLD))
            else:
                proto_worse[ds] = False
        passed = (rel_red is not None and rel_red >= REDUCTION_THRESHOLD) and (not any(proto_worse.values()))
        if passed:
            ok_comps.append(comp)
        per_comp[comp] = {
            "available": True,
            "n_cells_pooled": int(len(dd)),
            "pooled_primary_pp": pp(mae_p),
            "pooled_comparator_pp": pp(mae_c),
            "relative_reduction": finite_or_none(rel_red),
            "meets_10pct_reduction": bool(rel_red is not None and rel_red >= REDUCTION_THRESHOLD),
            "protocol_worse_than_0p25pp": proto_worse,
            "protocol_means_pp": {ds: per_comp_protocol_means.get(ds, {}).get(comp) for ds in PROTOCOLS},
            "passed": bool(passed),
        }
    screen["result"] = {
        "per_comparator": per_comp,
        "pooled_passed_vs_all": bool(len(ok_comps) == len([c for c in SCREEN_COMPARATORS if per_comp[c].get("available", False)])),
        "comparators_passed": ok_comps,
        "protocol_primary_mean_pp": {ds: per_comp_protocol_means.get(ds, {}).get(PRIMARY) for ds in PROTOCOLS},
    }

    # ---- Existing confirmation success criterion (mechanical inspection only) ----
    existing_criterion = read_confirmation_success_criterion()

    # ---- Figures ----
    make_figures(cell_clean, cell_stress)

    # ---- Summary ----
    summary = {
        "root": str(ROOT),
        "units": "SOH fraction; pp = 100 * fraction",
        "validation": {
            "recomputed_metrics_match_saved_within": TOL,
            "ref_exclusion": "cycle>ref_cycle asserted",
            "uniqueness_keys": ["dataset", "cell", "model", "scenario", "row_index"],
            "schema_report": schema_report,
        },
        "cohorts_loaded": COHORT_GROUPS,
        "protocols": PROTOCOLS,
        "scenario_map": SCENARIO_OFFSET,
        "expected_models": N_EXPECTED_MODELS,
        "models": all_models,
        "n_models": len(all_models),
        "n_scenarios": schema_report["n_scenarios"],
        "satellite_protocol_cells": schema_report["protocol_cells"],
        "clean_mae_pp_all_models": clean_mat.to_dict("records"),
        "stress_mae_pp": stress_mat.to_dict("records"),
        "low_soh_mae_pp": low_avg_records,
        "paired_summary": paired_summary,
        "screening": screen,
        "primary_metadata": extract_primary_metadata(),
        "existing_confirmation_success_criterion": existing_criterion,
        "source_cv": {"path": str(RESULTS / "source_XJTU_B1" / "source_cv_summary.csv"),
                      "note": "selected score, NOT unbiased validation"},
        "figures": [
            "analysis/figures/fig1_clean_cellmean_mae.pdf/.png",
            "analysis/figures/fig2_paired_errors.pdf/.png",
            "analysis/figures/fig3_voltage_stress.pdf/.png",
        ],
    }
    write_json(ANALYSIS / "summary.json", summary)
    print(f"Wrote {ANALYSIS / 'summary.json'} and figures in {FIGDIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
