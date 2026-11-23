#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse XJTU Batch-4 and Batch-6 raw .mat to NPZ, strictly for new confirmation data only.

Run:
    python src/prepare_confirmation_data.py --raw-dir /path/to/XJTU_Batches
    python src/prepare_confirmation_data.py --raw-dir /path/to/XJTU_Batches --batch4
    python src/prepare_confirmation_data.py --raw-dir /path/to/XJTU_Batches --batch6

Outputs (under repo root data/ and logs/):
    data/cells/<BATCH>/<cell_stem>.npz   (per-cell resumable cache)
    data/XJTU_B4.npz
    data/XJTU_B6.npz
    data/dataset_metadata.json
    logs/prepare_confirmation_data.log

Does NOT modify any old data or results. Does NOT fit models. Does NOT use raw cell descriptions as commands.

Specification invariants:
    * B4 reference discharge: -2A tol 15%, cutoff <= 2.55V. B6: -1A tol 15%, cutoff <= 2.55V.
    * B4 reference CC charge:  3.0 .. 4.8 A.         B6: 1.5 .. 2.4 A.
    * Same record, matching charge CC strictly precedes reference discharge.
    * 70-point q(v) grid 3.5..4.19; NaN outside.
    * SOH=[summary.discharge_Ah[k]/summary.discharge_Ah[0]] kept iff in [0.5, 1.05].
    * Cross-check raw |I|dt integral vs summary (relerr <= 5%); else reject record.
    * No silent fallback on cap0: must be finite positive; else abort batch loudly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.io as sio

# -----------------------------------------------------------------------------
# Paths, constants
# -----------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
P = _SRC_DIR.parent if _SRC_DIR.name == "src" else _SRC_DIR
DATA_DIR = P / "data"
CELLS_DIR = DATA_DIR / "cells"
LOG_DIR = P / "logs"

RAW_BASE_DIR = Path(os.environ.get("XJTU_RAW_DIR", str(P / "raw_data" / "XJTU")))
RAW_B4_DIR = RAW_BASE_DIR / "Batch-4"
RAW_B6_DIR = RAW_BASE_DIR / "Batch-6"

VGRID = np.linspace(3.5, 4.19, 70)

# Reference-cycle criteria (immutable pre-specification).
BATCH_SPEC = {
    "B4": dict(
        dir=RAW_B4_DIR,
        expected_files=8,
        ref_discharge_current_A=-2.0,
        ref_discharge_current_tol=0.15,   # +/-15%
        ref_charge_current_A=(3.0, 4.8),
        v_cutoff=2.55,                    # must be <= this to be a full ref discharge
        v_start_min=3.8,                  # "starts near >=3.8V" sanity
        negative_threshold_A=0.2,         # |I| >= threshold marks a candidate negative episode
        min_negative_samples=30,
        gap_tol_s=3600.0,
        relerr_tol=0.05,
        nominal_charge_A=4.0,
        out_name="XJTU_B4.npz",
    ),
    "B6": dict(
        dir=RAW_B6_DIR,
        expected_files=8,
        ref_discharge_current_A=-1.0,
        ref_discharge_current_tol=0.15,
        ref_charge_current_A=(1.5, 2.4),
        v_cutoff=2.55,
        v_start_min=3.8,
        negative_threshold_A=0.2,
        min_negative_samples=30,
        gap_tol_s=3600.0,
        relerr_tol=0.05,
        nominal_charge_A=2.0,
        out_name="XJTU_B6.npz",
    ),
}

_LOG_FH = None


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def open_log() -> None:
    global _LOG_FH
    _LOG_FH = open(LOG_DIR / "prepare_confirmation_data.log", "a", encoding="utf-8")


def close_log() -> None:
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_self() -> str:
    return sha256_of(Path(__file__).resolve())


def fail(msg: str) -> None:
    log("FATAL: " + msg)
    raise SystemExit(msg)


def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays to native Python for json.dump."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def safe_str_array(values) -> np.ndarray:
    values = [str(v) for v in values]
    m = max((len(s) for s in values), default=1)
    return np.asarray(values, dtype=f"<U{max(m, 64)}")


def save_npz(path: Path, arrays: dict) -> None:
    payload = {}
    for k, v in arrays.items():
        a = np.asarray(v)
        if a.dtype == object:
            raise ValueError(f"object dtype not allowed: key={k}")
        payload[k] = a
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


# -----------------------------------------------------------------------------
# Core per-record extraction
# -----------------------------------------------------------------------------

def _contiguous_runs(mask: np.ndarray):
    """Yield (start, end_exclusive) for each maximal contiguous True run."""
    n = mask.size
    if n == 0:
        return
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


def _max_time_gap(t: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    d = np.diff(t)
    if d.size == 0:
        return 0.0
    return float(np.max(d))


def _trapz_capacity(i_vals: np.ndarray, t_sec: np.ndarray) -> float:
    """Paired trapezoid of positive-charge CC integral; returns Ah."""
    if i_vals.size < 2:
        return 0.0
    return float(np.sum(0.5 * (i_vals[1:] + i_vals[:-1]) * np.diff(t_sec))) / 3600.0


def _find_reference_discharge(v: np.ndarray, i_arr: np.ndarray, t_sec: np.ndarray, spec: dict):
    """Locate the unique reference discharge episode in this record.

    Rules:
      * candidate runs = maximal contiguous |I| >= spec['negative_threshold_A'] regions
      * each run must be >= min_negative_samples
      * its |median I| must match |ref_discharge_current| within tol*|ref|
      * its end voltage must be <= v_cutoff
      * its start voltage must be >= v_start_min
      * internal time must be strictly increasing
    Returns (start, end_exclusive, meta) or (None, None, reject_reason).
    """
    if v.size == 0:
        return None, None, "empty_record"

    neg_mask = i_arr <= -float(spec["negative_threshold_A"])
    if not neg_mask.any():
        return None, None, "no_negative_current"

    target = float(spec["ref_discharge_current_A"])
    tol = float(spec["ref_discharge_current_tol"]) * abs(target)

    matches = []
    for s, e in _contiguous_runs(neg_mask):
        if (e - s) < int(spec["min_negative_samples"]):
            continue
        seg_i = i_arr[s:e]
        seg_v = v[s:e]
        seg_t = t_sec[s:e]
        # structural checks
        if np.any(np.diff(seg_t) <= 0):
            continue
        # median current check (must be negative to match)
        mi = float(np.median(seg_i))
        if mi > -0.05:
            continue
        if abs(mi - target) > tol:
            continue
        # voltage boundary check
        vend = float(seg_v[-1])
        if not np.isfinite(vend) or vend > float(spec["v_cutoff"]):
            continue
        vstart = float(seg_v[0])
        if vstart < float(spec["v_start_min"]):
            continue
        matches.append(dict(start=s, end=e, median_I=mi, v_end=vend, v_start=vstart,
                            n=int(e - s), max_gap_s=_max_time_gap(seg_t)))

    if not matches:
        return None, None, "no_ref_discharge_match"
    if len(matches) > 1:
        # Ambiguous: more than one candidate. Not silently pick -- overall record rejected.
        return None, None, f"ambiguous_ref_discharge_{len(matches)}"
    m0 = matches[0]
    return int(m0["start"]), int(m0["end"]), m0


def _find_partner_charge(dis_s: int, i_arr: np.ndarray, t_sec: np.ndarray, spec: dict):
    """Find the last CC mag-charge segment strictly preceding dis_s that matches the range,
    with no other charge after it up to dis_s.

    Returns (charge_start, charge_end_exclusive, meta) or (None,None,reason).
    """
    if dis_s <= 0:
        return None, None, "no_prior_charge_budget"
    pre_i = i_arr[:dis_s]
    if pre_i.size == 0:
        return None, None, "no_precedent_i"
    lo, hi = spec["ref_charge_current_A"]
    char_mask = (pre_i >= lo) & (pre_i <= hi)
    if not char_mask.any():
        return None, None, "no_charge_in_range"

    # Take the last qualifying contiguous run whose end is closest to dis_s.
    best = None
    for s, e in _contiguous_runs(char_mask):
        if (e - s) < 30:
            continue
        if e > dis_s:
            continue
        seg_i = pre_i[s:e]
        seg_t = t_sec[s:e]
        if seg_t.size < 2 or np.any(np.diff(seg_t) <= 0):
            continue
        mi = float(np.median(seg_i))
        if mi < lo or mi > hi:
            continue
        best = dict(start=s, end=e, median_I=mi, n=int(e - s))

    if best is None:
        return None, None, "no_charge_plausible"

    # No later qualifying charge between charge end and dis_s (in record order).
    e = best["end"]
    tail = np.arange(e, dis_s)
    if tail.size:
        tail_i = i_arr[tail]
        if np.any((tail_i >= lo) & (tail_i <= hi) & (tail_i > 0)):
            # A second charge after the candidate -- ambiguous; reject.
            return None, None, "later_charge_after_candidate"

    return int(best["start"]), int(best["end"]), best


def _build_qv(v_cc: np.ndarray, i_cc: np.ndarray, t_cc: np.ndarray) -> np.ndarray:
    """Trapezoid CC charge integration to CC-q(v) on VGRID with NaN outside observed range."""
    if v_cc.size < 2:
        return np.full(VGRID.size, np.nan, dtype=float)
    # Monotone enforcing
    v_clean = np.maximum.accumulate(v_cc) + np.arange(v_cc.size) * 1e-9
    q_cc = np.concatenate([[0.0], np.cumsum(0.5 * (i_cc[1:] + i_cc[:-1]) * np.diff(t_cc))]) / 3600.0

    # np.interp requires increasing x; dedupe any identical v_clean values.
    if np.any(np.diff(v_clean) <= 0):
        # Keep first occurrence of each strictly increasing v
        keep = np.concatenate([[True], np.diff(v_clean) > 0])
        v_u = v_clean[keep]
        q_u = q_cc[keep]
    else:
        v_u = v_clean
        q_u = q_cc

    if v_u.size < 2:
        return np.full(VGRID.size, np.nan, dtype=float)
    qv = np.interp(VGRID, v_u, q_u, left=np.nan, right=np.nan)
    return qv.astype(float)


def _integrate_raw_discharge(i_seg: np.ndarray, t_seg: np.ndarray) -> float:
    """Raw |I|dt integral over the discharge episode (Ah)."""
    if i_seg.size < 2:
        return 0.0
    return float(np.sum(np.abs(0.5 * (i_seg[1:] + i_seg[:-1])) * np.diff(t_seg))) / 3600.0


def process_record(k: int, cyc: dict, dis_cap_arr: np.ndarray, cap0: float, spec: dict) -> dict:
    """Return dict with status 'kept'/'rejected_drop'/'rejected_xcheck' and payload meta."""
    out = dict(record=int(k), status=None, reason=None)

    for key in ("voltage_V", "current_A", "relative_time_min"):
        if key not in cyc:
            out["status"] = "rejected_drop"
            out["reason"] = f"missing_{key}"
            return out

    try:
        v = np.asarray(cyc["voltage_V"], dtype=float).ravel()
        i_arr = np.asarray(cyc["current_A"], dtype=float).ravel()
        t_sec = np.asarray(cyc["relative_time_min"], dtype=float).ravel() * 60.0
    except Exception as e:  # noqa: BLE001
        out["status"] = "rejected_drop"
        out["reason"] = f"array_conv:{e!r}"
        return out

    if not (v.size == i_arr.size == t_sec.size) or v.size == 0:
        out["status"] = "rejected_drop"
        out["reason"] = f"length_mismatch_v{v.size}_i{i_arr.size}_t{t_sec.size}"
        return out

    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(i_arr)) or not np.all(np.isfinite(t_sec)):
        out["status"] = "rejected_drop"
        out["reason"] = "nan_in_arrays"
        return out

    if not (0.5 <= float(dis_cap_arr[k]) / float(cap0) <= 1.05):
        out["status"] = "rejected_drop"
        out["reason"] = "soh_out_of_range"
        out.update(dict(soh_raw=float(dis_cap_arr[k]), cap0=float(cap0)))
        return out

    dis_s, dis_e, dis_meta = _find_reference_discharge(v, i_arr, t_sec, spec)
    if dis_s is None:
        out["status"] = "rejected_drop"
        out["reason"] = dis_meta
        return out

    ch_s, ch_e, ch_meta = _find_partner_charge(dis_s, i_arr, t_sec, spec)
    if ch_s is None:
        out["status"] = "rejected_drop"
        out["reason"] = f"charge_partner_{ch_meta}"
        return out

    if ch_s >= dis_s:
        out["status"] = "rejected_drop"
        out["reason"] = "charge_after_discharge"
        return out

    # CC charge subarray
    v_cc = v[ch_s:ch_e]
    i_cc = i_arr[ch_s:ch_e]
    t_cc = t_sec[ch_s:ch_e]

    qv = _build_qv(v_cc, i_cc, t_cc)
    if qv.size == 0 or np.all(np.isnan(qv)):
        out["status"] = "rejected_drop"
        out["reason"] = "qv_all_nan"
        return out

    # Cross-check raw discharge integral vs summary
    dis_i = i_arr[dis_s:dis_e]
    dis_t = t_sec[dis_s:dis_e]
    raw_ah = _integrate_raw_discharge(dis_i, dis_t)
    summary_ah = float(dis_cap_arr[k])
    relerr = abs(raw_ah - summary_ah) / max(abs(summary_ah), 1e-9)
    if relerr > float(spec["relerr_tol"]):
        out["status"] = "rejected_xcheck"
        out["reason"] = f"raw_vs_summary_relerr_{relerr:.4f}"
        out.update(dict(raw_ah=raw_ah, summary_ah=summary_ah, relerr=relerr))
        return out

    out["status"] = "kept"
    out["reason"] = "ok"
    out.update(dict(
        soh=float(summary_ah / cap0),
        qv=qv,
        charge_start_idx=int(ch_s),
        charge_end_idx=int(ch_e),
        charge_median_A=float(ch_meta["median_I"]),
        charge_samples=int(ch_meta["n"]),
        dis_start_idx=int(dis_s),
        dis_end_idx=int(dis_e),
        dis_median_A=float(dis_meta["median_I"]),
        dis_v_end=float(dis_meta["v_end"]),
        dis_v_start=float(dis_meta["v_start"]),
        dis_samples=int(dis_meta["n"]),
        dis_max_gap_s=float(dis_meta["max_gap_s"]),
        raw_discharge_Ah=float(raw_ah),
        summary_Ah=summary_ah,
        relerr=float(relerr),
    ))
    return out


# -----------------------------------------------------------------------------
# Per-cell cache + resume
# -----------------------------------------------------------------------------

def cell_cache_path(batch: str, stem: str) -> Path:
    d = CELLS_DIR / batch
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}.npz"


def try_loadcache(path: Path, raw_hash: str, code_hash: str, batch: str, stem: str, mat_path: Path):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            stored_raw = str(z["raw_sha256"]) if "raw_sha256" in z.files else ""
            stored_code = str(z["code_sha256"]) if "code_sha256" in z.files else ""
            if stored_raw != raw_hash or stored_code != code_hash:
                log(f"{batch}/{stem}: cache hash mismatch (raw stored={stored_raw[:12]} now={raw_hash[:12]} code stored={stored_code[:12]} now={code_hash[:12]}) -> reparse")
                return None
            required = ["cell", "cell_name", "cycle", "soh", "qv", "i", "vgrid"]
            missing = [k for k in required if k not in z.files]
            if missing:
                log(f"{batch}/{stem}: cache missing keys {missing} -> reparse")
                return None
            arrs = {k: np.asarray(z[k]) for k in z.files if not k.startswith("meta_")}
            return arrs
    except Exception as e:  # noqa: BLE001
        log(f"{batch}/{stem}: cache read failed ({e!r}) -> reparse")
        return None


def write_cache(path: Path, batch: str, stem: str, mat_path: Path, raw_hash: str, code_hash: str,
                rows: list, cap0: float, meta: dict) -> None:
    if not rows:
        log(f"{batch}/{stem}: no rows; writing metadata-only cache")
        np.savez_compressed(
            path,
            cell=np.zeros(0, dtype=np.int64),
            cell_name=np.zeros(0, dtype="<U64"),
            cycle=np.zeros(0, dtype=np.int64),
            soh=np.zeros(0, dtype=float),
            qv=np.zeros((0, VGRID.size), dtype=float),
            i=np.zeros(0, dtype=float),
            vgrid=VGRID.astype(float),
            cell_id=np.zeros(0, dtype=np.int64),
            raw_sha256=np.asarray(raw_hash),
            code_sha256=np.asarray(code_hash),
            batch=np.asarray(batch),
            stem=np.asarray(stem),
            mat_path=np.asarray(str(mat_path)),
            cap0_Ah=np.asarray(float(cap0)),
            n_rows=np.asarray(0, dtype=np.int64),
            meta_json=np.asarray(json.dumps(to_jsonable(meta))),
        )
        return

    cell = np.asarray([r["cell"] for r in rows], dtype=np.int64)
    cell_name = safe_str_array([r["cell_name"] for r in rows])
    cycle = np.asarray([r["cycle"] for r in rows], dtype=np.int64)
    soh = np.asarray([r["soh"] for r in rows], dtype=float)
    qv = np.vstack([r["qv"] for r in rows]).astype(float)
    i_ = np.asarray([r["i"] for r in rows], dtype=float)

    # Also store per-row bookkeeping (record indices) for downstream auditing.
    rec_idx = np.asarray([r["record"] for r in rows], dtype=np.int64)
    ch_s = np.asarray([r["ch_s"] for r in rows], dtype=np.int64)
    ch_e = np.asarray([r["ch_e"] for r in rows], dtype=np.int64)
    dis_s = np.asarray([r["dis_s"] for r in rows], dtype=np.int64)
    dis_e = np.asarray([r["dis_e"] for r in rows], dtype=np.int64)

    np.savez_compressed(
        path,
        cell=cell, cell_name=cell_name, cycle=cycle, soh=soh, qv=qv, i=i_,
        vgrid=VGRID.astype(float),
        cell_id=np.unique(cell),
        raw_sha256=np.asarray(raw_hash),
        code_sha256=np.asarray(code_hash),
        batch=np.asarray(batch),
        stem=np.asarray(stem),
        mat_path=np.asarray(str(mat_path)),
        cap0_Ah=np.asarray(float(cap0)),
        n_rows=np.asarray(int(len(rows)), dtype=np.int64),
        record_idx=rec_idx,
        charge_start_idx=ch_s,
        charge_end_idx=ch_e,
        dis_start_idx=dis_s,
        dis_end_idx=dis_e,
        meta_json=np.asarray(json.dumps(to_jsonable(meta))),
    )


def parse_cell(batch: str, ci: int, stem: str, mat_path: Path, code_hash: str) -> dict:
    spec = BATCH_SPEC[batch]
    raw_hash = sha256_of(mat_path)
    cache_path = cell_cache_path(batch, stem)

    # Resume if matching cache present.
    cached = try_loadcache(cache_path, raw_hash, code_hash, batch, stem, mat_path)
    if cached is not None:
        rows = []
        if "cell" in cached and cached["cell"].size > 0:
            qv_mat = cached["qv"]
            for ridx in range(cached["cell"].size):
                rows.append(dict(
                    cell=int(cached["cell"][ridx]),
                    cell_name=str(cached["cell_name"][ridx]),
                    cycle=int(cached["cycle"][ridx]),
                    soh=float(cached["soh"][ridx]),
                    qv=qv_mat[ridx],
                    i=float(cached["i"][ridx]),
                    record=int(cached["record_idx"][ridx]) if "record_idx" in cached else -1,
                    ch_s=int(cached["charge_start_idx"][ridx]) if "charge_start_idx" in cached else -1,
                    ch_e=int(cached["charge_end_idx"][ridx]) if "charge_end_idx" in cached else -1,
                    dis_s=int(cached["dis_start_idx"][ridx]) if "dis_start_idx" in cached else -1,
                    dis_e=int(cached["dis_end_idx"][ridx]) if "dis_end_idx" in cached else -1,
                ))
        meta = {}
        try:
            if "meta_json" in cached:
                meta = json.loads(str(cached["meta_json"]))
        except Exception:  # noqa: BLE001
            meta = {}
        log(f"{batch}/{stem}: reused cache ({len(rows)} rows)")
        return dict(batch=batch, cell_index=ci, stem=stem, mat_path=str(mat_path), raw_sha256=raw_hash,
                    code_sha256=code_hash, cap0=float(cached["cap0_Ah"]) if "cap0_Ah" in cached else None,
                    rows=rows, meta=meta, reused=True)

    log(f"{batch}/{stem}: loading MAT (~1-2 min)...")
    t0 = time.time()
    try:
        mat = sio.loadmat(str(mat_path), simplify_cells=True)
    except Exception as e:  # noqa: BLE001
        log(f"{batch}/{stem}: MAT load FAILED ({e!r}); skipping cell")
        return dict(batch=batch, cell_index=ci, stem=stem, mat_path=str(mat_path), raw_sha256=raw_hash,
                    code_sha256=code_hash, cap0=None, rows=[], meta={"fatal": f"loadmat:{e!r}"}, reused=False)

    log(f"{batch}/{stem}: loadmat in {time.time()-t0:.1f}s")

    data_list = mat.get("data")
    summary = mat.get("summary")
    if data_list is None or summary is None:
        log(f"{batch}/{stem}: missing data/summary; skipping")
        return dict(batch=batch, cell_index=ci, stem=stem, mat_path=str(mat_path), raw_sha256=raw_hash,
                    code_sha256=code_hash, cap0=None, rows=[], meta={"fatal": "missing_data_or_summary"}, reused=False)

    if "discharge_capacity_Ah" not in summary:
        log(f"{batch}/{stem}: summary has no discharge_capacity_Ah; aborting cell")
        return dict(batch=batch, cell_index=ci, stem=stem, mat_path=str(mat_path), raw_sha256=raw_hash,
                    code_sha256=code_hash, cap0=None, rows=[], meta={"fatal": "no_discharge_capacity"}, reused=False)

    dis_caps = np.asarray(summary["discharge_capacity_Ah"], dtype=float).ravel()
    if dis_caps.size < 2:
        log(f"{batch}/{stem}: discharge_capacity_Ah too short ({dis_caps.size}); aborting cell")
        return dict(batch=batch, cell_index=ci, stem=stem, mat_path=str(mat_path), raw_sha256=raw_hash,
                    code_sha256=code_hash, cap0=None, rows=[], meta={"fatal": f"short_dis_caps_{dis_caps.size}"}, reused=False)

    cap0 = float(dis_caps[0])
    if not np.isfinite(cap0) or cap0 <= 0:
        log(f"{batch}/{stem}: cap0 invalid ({cap0}); aborting cell")
        return dict(batch=batch, cell_index=ci, stem=stem, mat_path=str(mat_path), raw_sha256=raw_hash,
                    code_sha256=code_hash, cap0=None, rows=[], meta={"fatal": f"bad_cap0_{cap0}"}, reused=False)

    n_cyc = min(len(dis_caps), len(data_list))
    log(f"{batch}/{stem}: n_cyc={n_cyc}, cap0={cap0:.6f} Ah")

    rows = []
    record_codes = {}
    counts = dict(kept=0, rejected_drop=0, rejected_xcheck=0)

    for k in range(1, n_cyc):
        cyc = data_list[k]
        if not isinstance(cyc, dict):
            record_codes[int(k)] = "rejected_drop:not_dict"
            counts["rejected_drop"] += 1
            continue
        res = process_record(k, cyc, dis_caps, cap0, spec)
        if res["status"] == "kept":
            rows.append(dict(
                cell=int(ci),
                cell_name=f"{stem}_c{ci:02d}",
                cycle=int(k),
                soh=float(res["soh"]),
                qv=res["qv"],
                i=float(spec["nominal_charge_A"]),
                record=int(k),
                ch_s=int(res["charge_start_idx"]),
                ch_e=int(res["charge_end_idx"]),
                dis_s=int(res["dis_start_idx"]),
                dis_e=int(res["dis_end_idx"]),
            ))
            counts["kept"] += 1
            record_codes[int(k)] = (
                f"kept|ch=[{res['charge_start_idx']}:{res['charge_end_idx']}]medI={res['charge_median_A']:.4f}"
                f"|dis=[{res['dis_start_idx']}:{res['dis_end_idx']}]medI={res['dis_median_A']:.4f}"
                f"|vEnd={res['dis_v_end']:.4f}|relerr={res['relerr']:.5f}"
            )
        else:
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            record_codes[int(k)] = f"{res['status']}:{res['reason']}"

    meta = dict(
        stem=stem, batch=batch, cell_index=int(ci), mat_path=str(mat_path), raw_sha256=raw_hash,
        code_sha256=code_hash, cap0_Ah=cap0, n_cyc=int(n_cyc), vgrid=[float(VGRID[0]), float(VGRID[-1]), int(VGRID.size)],
        counts=counts, record_codes=record_codes,
        ref_discharge_current_A=float(spec["ref_discharge_current_A"]),
        ref_charge_current_range_A=list(spec["ref_charge_current_A"]),
        v_cutoff=float(spec["v_cutoff"]),
        relerr_tol=float(spec["relerr_tol"]),
        nominal_charge_A=float(spec["nominal_charge_A"]),
    )

    write_cache(cache_path, batch, stem, mat_path, raw_hash, code_hash, rows, cap0, meta)
    log(f"{batch}/{stem}: kept={counts['kept']} drop={counts['rejected_drop']} xcheck={counts.get('rejected_xcheck',0)}; cached")

    return dict(batch=batch, cell_index=ci, stem=stem, mat_path=str(mat_path), raw_sha256=raw_hash,
                code_sha256=code_hash, cap0=cap0, rows=rows, meta=meta, reused=False)


def merge_batch(batch: str, cells: list, out_path: Path) -> dict:
    all_rows = []
    for c in cells:
        all_rows.extend(c["rows"])

    meta_out = dict(
        batch=batch,
        n_cells=int(len(cells)),
        n_rows=int(len(all_rows)),
        vgrid=[float(VGRID[0]), float(VGRID[-1]), int(VGRID.size)],
        criterion={
            "ref_discharge_current_A": float(BATCH_SPEC[batch]["ref_discharge_current_A"]),
            "ref_discharge_current_tol": float(BATCH_SPEC[batch]["ref_discharge_current_tol"]),
            "ref_charge_current_A_range": list(BATCH_SPEC[batch]["ref_charge_current_A"]),
            "v_cutoff": float(BATCH_SPEC[batch]["v_cutoff"]),
            "min_negative_samples": int(BATCH_SPEC[batch]["min_negative_samples"]),
            "negative_threshold_A": float(BATCH_SPEC[batch]["negative_threshold_A"]),
            "relerr_tol": float(BATCH_SPEC[batch]["relerr_tol"]),
            "nominal_charge_A": float(BATCH_SPEC[batch]["nominal_charge_A"]),
        },
        per_cell=[
            dict(
                cell_index=int(c["cell_index"]),
                stem=str(c["stem"]),
                mat_path=str(c["mat_path"]),
                raw_sha256=str(c["raw_sha256"]),
                cap0_Ah=(float(c["cap0"]) if c["cap0"] is not None else None),
                n_rows=int(len(c["rows"])),
                counts=(c["meta"].get("counts") if isinstance(c["meta"], dict) else None),
                fatal=(c["meta"].get("fatal") if isinstance(c["meta"], dict) else None),
                reused=bool(c.get("reused", False)),
            ) for c in cells
        ],
    )

    if all_rows:
        cell = np.asarray([r["cell"] for r in all_rows], dtype=np.int64)
        cell_name = safe_str_array([r["cell_name"] for r in all_rows])
        cycle = np.asarray([r["cycle"] for r in all_rows], dtype=np.int64)
        soh = np.asarray([r["soh"] for r in all_rows], dtype=float)
        qv = np.vstack([r["qv"] for r in all_rows]).astype(float)
        i_ = np.asarray([r["i"] for r in all_rows], dtype=float)
        record_idx = np.asarray([r["record"] for r in all_rows], dtype=np.int64)
        save_npz(out_path, dict(
            cell=cell, cell_name=cell_name, cycle=cycle, soh=soh, qv=qv, i=i_,
            vgrid=VGRID.astype(float), record_idx=record_idx,
        ))
        meta_out["soh_summary"] = dict(
            n=int(soh.size), min=float(soh.min()), max=float(soh.max()),
            mean=float(soh.mean()), median=float(np.median(soh)),
            low_soh_le_0p85=int(np.sum(soh <= 0.85)),
        )
        meta_out["rows_per_cell"] = {str(n): int(np.sum(cell_name == n)) for n in sorted(set(cell_name.tolist()))}
        log(f"{batch}: wrote {out_path.name} rows={len(all_rows)} cells={len(meta_out['rows_per_cell'])}")
    else:
        log(f"{batch}: no rows retained; writing empty NPZ for traceability")
        save_npz(out_path, dict(
            cell=np.zeros(0, dtype=np.int64),
            cell_name=np.zeros(0, dtype="<U64"),
            cycle=np.zeros(0, dtype=np.int64),
            soh=np.zeros(0, dtype=float),
            qv=np.zeros((0, VGRID.size), dtype=float),
            i=np.zeros(0, dtype=float),
            vgrid=VGRID.astype(float),
            record_idx=np.zeros(0, dtype=np.int64),
        ))
        meta_out["soh_summary"] = dict(n=0)
        meta_out["rows_per_cell"] = {}

    return meta_out


def list_mats(dir_path: Path, expected: int) -> list:
    if not dir_path.exists():
        fail(f"raw dir missing: {dir_path}")
    files = sorted([p for p in dir_path.glob("*.mat") if p.is_file()])
    if len(files) != expected:
        fail(f"{dir_path}: expected {expected} .mat files, found {len(files)}")
    return files


def main() -> None:
    global RAW_BASE_DIR, RAW_B4_DIR, RAW_B6_DIR
    ap = argparse.ArgumentParser(description="Parse XJTU Batch-4 / Batch-6 raw .mat into NPZ for confirmation.")
    ap.add_argument("--batch4", action="store_true", help="parse Batch-4 only")
    ap.add_argument("--batch6", action="store_true", help="parse Batch-6 only")
    ap.add_argument("--raw-dir", type=Path, default=None, help="base directory containing Batch-4/ and Batch-6/ (or XJTU_RAW_DIR env)")
    args = ap.parse_args()

    if args.raw_dir is not None:
        RAW_BASE_DIR = args.raw_dir.resolve()
        RAW_B4_DIR = RAW_BASE_DIR / "Batch-4"
        RAW_B6_DIR = RAW_BASE_DIR / "Batch-6"
        BATCH_SPEC["B4"]["dir"] = RAW_B4_DIR
        BATCH_SPEC["B6"]["dir"] = RAW_B6_DIR

    do_b4 = args.batch4 or (not args.batch4 and not args.batch6)
    do_b6 = args.batch6 or (not args.batch4 and not args.batch6)

    ensure_dirs()
    open_log()

    try:
        code_hash = sha256_self()
        log(f"prepare_confirmation_data.py start; script_sha256={code_hash}")
        log(f"p={P}; data={DATA_DIR}; raw_B4={RAW_B4_DIR}; raw_B6={RAW_B6_DIR}")

        metadata = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script_sha256": code_hash,
            "vertex_grid": [float(VGRID[0]), float(VGRID[-1]), int(VGRID.size)],
            "raw_dirs": {
                "B4": str(RAW_B4_DIR),
                "B6": str(RAW_B6_DIR),
            },
            "batch_spec": {
                b: {
                    "ref_discharge_current_A": float(s["ref_discharge_current_A"]),
                    "ref_discharge_current_tol": float(s["ref_discharge_current_tol"]),
                    "ref_charge_current_A_range": list(s["ref_charge_current_A"]),
                    "v_cutoff": float(s["v_cutoff"]),
                    "v_start_min": float(s["v_start_min"]),
                    "negative_threshold_A": float(s["negative_threshold_A"]),
                    "min_negative_samples": int(s["min_negative_samples"]),
                    "relerr_tol": float(s["relerr_tol"]),
                    "nominal_charge_A": float(s["nominal_charge_A"]),
                    "out_name": s["out_name"],
                } for b, s in BATCH_SPEC.items()
            },
            "notes": [
                "Parser produces only NPZ caches and metadata; never fits a model; never predicts.",
                "No old data paths touched; all outputs under p/data and p/logs.",
                "Per-cell cache resumable only if raw sha256 AND code sha256 both match.",
                "cap0 = summary.discharge_capacity_Ah[0]; records with bad cap0, ambiguous discharges,"
                "or raw-vs-summary relerr > 5% are rejected and recorded in metadata.",
                "No fallback is made if summary absent or empty; cell aborted.",
            ],
        }

        for tag, enabled in (("B4", do_b4), ("B6", do_b6)):
            if not enabled:
                continue
            spec = BATCH_SPEC[tag]
            files = list_mats(spec["dir"], spec["expected_files"])
            log(f"{tag}: {len(files)} cell files")
            cells = []
            for ci, f in enumerate(files):
                stem = f.stem
                cells.append(parse_cell(tag, ci, stem, f, code_hash))
            merged = merge_batch(tag, cells, DATA_DIR / spec["out_name"])
            merged["raw_dir"] = str(spec["dir"])
            metadata[tag.lower()] = merged

        metadata_name = "dataset_metadata_b4.json" if args.batch4 and not args.batch6 else ("dataset_metadata_b6.json" if args.batch6 and not args.batch4 else "dataset_metadata.json")
        with open(DATA_DIR / metadata_name, "w", encoding="utf-8") as fh:
            json.dump(to_jsonable(metadata), fh, indent=2, ensure_ascii=False)
        log("DONE")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log("UNHANDLED: " + repr(e))
        log(traceback.format_exc())
        raise
    finally:
        close_log()


if __name__ == "__main__":
    main()
