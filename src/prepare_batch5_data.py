#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse XJTU Batch-5 raw .mat to NPZ for new confirmation data only.

B5 invariant: summary.discharge_capacity_Ah[0] is a partial 1.777 Ah record and
MUST NOT be used as the SOH denominator. The first valid FULL-RPT discharge
(found via base._find_reference_discharge with a -2 A partner charge in 1.5..2.4 A)
determines cap0 per cell. All records are then re-scored with that cap0.

Outputs (only under this dir):
    data/cells/B5/<stem>.npz
    data/XJTU_B5.npz
    data/dataset_metadata_b5.json
    logs/prepare_batch5_data.log

Does NOT modify prepare_confirmation_data.py, Batch-4, or Batch-6 code/outputs.
Does NOT fit models. Does NOT read or produce predictions.
"""
from __future__ import annotations

import argparse, json, os, sys, time, traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_confirmation_data as base  # noqa: E402

# --- B5 paths & spec (B5 only; runtime patch allowed) -------------------------
RAW_B5_DIR = Path(r"D:/bách khoa Đà Nẵng/nghiencuukhoahoc/baotri/data/NASABattery/Battery Dataset/Batch-5")

B5 = dict(
    dir=RAW_B5_DIR,
    expected_files=8,
    ref_discharge_current_A=-2.0,
    ref_discharge_current_tol=0.15,
    ref_charge_current_A=(1.5, 2.4),
    v_cutoff=2.55,
    v_start_min=3.8,
    negative_threshold_A=0.2,
    min_negative_samples=30,
    gap_tol_s=3600.0,
    relerr_tol=0.05,
    nominal_charge_A=2.0,
    out_name="XJTU_B5.npz",
)
base.BATCH_SPEC["B5"] = B5  # runtime-only addition; base code untouched.

LOG = base.LOG_DIR / "prepare_batch5_data.log"
LOG_FH = None


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(line + "\n"); LOG_FH.flush()


def find_first_full_rpt(cyc_list, dis_caps, n_cyc, spec):
    """Earliest k>=1 such that base.process_record(k, ..., cap0=summary[k]) is 'kept'.
    Using cap0=summary[k] sets soh=1.0 (i.e., in-range) so only structural + integral
    checks filter, which is exactly the definition of a valid FULL RPT for this cell.
    """
    for k in range(n_cyc):
        cyc = cyc_list[k]
        if not isinstance(cyc, dict):
            continue
        sk = float(dis_caps[k])
        if not np.isfinite(sk) or sk <= 0:
            continue
        res = base.process_record(k, cyc, dis_caps, sk, spec)
        if res["status"] == "kept":
            return k, res
    return None, None


def parse_cell(ci, stem, mat_path, code_hash):
    raw_hash = base.sha256_of(mat_path)
    log(f"B5/{stem}: loading MAT ...")
    t0 = time.time()
    try:
        mat = base.sio.loadmat(str(mat_path), simplify_cells=True)
    except Exception as e:
        raise RuntimeError(f"{stem}: loadmat failed: {e!r}")
    log(f"B5/{stem}: loadmat in {time.time()-t0:.1f}s")

    dl = mat.get("data"); sm = mat.get("summary")
    if dl is None or sm is None or "discharge_capacity_Ah" not in sm:
        raise RuntimeError(f"{stem}: missing data/summary/discharge_capacity_Ah")
    dis_caps = np.asarray(sm["discharge_capacity_Ah"], dtype=float).ravel()
    if dis_caps.size < 2:
        raise RuntimeError(f"{stem}: discharge_capacity_Ah too short ({dis_caps.size})")

    if len(dis_caps) != len(dl):
        raise RuntimeError(f'{stem}: summary/raw record count mismatch')
    n_cyc = len(dl)
    k_first, first_res = find_first_full_rpt(dl, dis_caps, n_cyc, B5)
    if k_first is None:
        raise RuntimeError(f"{stem}: no valid FULL-RPT record found in k=1..{n_cyc-1}")
    cap0 = float(dis_caps[k_first])
    if not np.isfinite(cap0) or cap0 <= 0:
        raise RuntimeError(f"{stem}: first FULL RPT cap0 invalid ({cap0})")
    log(f"B5/{stem}: first FULL RPT at k={k_first} cap0={cap0:.6f} Ah (k=0 excluded; summary[0]={float(dis_caps[0]):.6f})")

    rows, codes = [], {}
    counts = dict(kept=0, rejected_drop=0, rejected_xcheck=0)
    for k in range(n_cyc):
        cyc = dl[k]
        if not isinstance(cyc, dict):
            counts["rejected_drop"] += 1
            codes[int(k)] = "rejected_drop:not_dict"
            continue
        res = base.process_record(k, cyc, dis_caps, cap0, B5)
        if res["status"] == "kept":
            rows.append(dict(cell=int(ci), cell_name=f"{stem}_c{ci:02d}", cycle=int(k+1),
                             soh=float(res["soh"]), qv=res["qv"], i=float(B5["nominal_charge_A"]),
                             record=int(k), ch_s=int(res["charge_start_idx"]), ch_e=int(res["charge_end_idx"]),
                             dis_s=int(res["dis_start_idx"]), dis_e=int(res["dis_end_idx"])))
            counts["kept"] += 1
            codes[int(k)] = (f"kept|ch=[{res['charge_start_idx']}:{res['charge_end_idx']}]medI={res['charge_median_A']:.4f}"
                             f"|dis=[{res['dis_start_idx']}:{res['dis_end_idx']}]medI={res['dis_median_A']:.4f}"
                             f"|vEnd={res['dis_v_end']:.4f}|relerr={res['relerr']:.5f}")
        else:
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            codes[int(k)] = f"{res['status']}:{res['reason']}"
        if (k % 5) == 0:
            log(f"B5/{stem}: k={k}/{n_cyc-1} kept={counts['kept']} drop={counts['rejected_drop']} x={counts.get('rejected_xcheck',0)}")
    return dict(raw_hash=raw_hash, cap0=cap0, k_first=k_first, rows=rows, codes=codes,
                counts=counts, n_cyc=n_cyc)


def main() -> None:
    global LOG_FH
    ap = argparse.ArgumentParser(description="Parse XJTU Batch-5 raw .mat into NPZ.")
    ap.add_argument("--force", action="store_true", help="ignore per-cell cache (still overwrites)")
    args = ap.parse_args()

    base.ensure_dirs()
    base.LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FH = open(LOG, "a", encoding="utf-8")
    try:
        code_hash = base.sha256_of(Path(__file__).resolve())
        base_hash = base.sha256_of(Path(base.__file__).resolve())
        log(f"prepare_batch5_data.py start; code_sha256={code_hash} base_sha256={base_hash}")

        if not RAW_B5_DIR.exists():
            base.fail(f"raw dir missing: {RAW_B5_DIR}")
        files = sorted([p for p in RAW_B5_DIR.glob("*.mat") if p.is_file()])
        if len(files) != B5["expected_files"]:
            base.fail(f"expected {B5['expected_files']} .mat files, found {len(files)} in {RAW_B5_DIR}")
        log(f"B5: {len(files)} cell files; names={[f.name for f in files]}")

        cells_out = []
        n_kept_total = 0
        per_cell_meta = []
        for ci, f in enumerate(files):
            stem = f.stem
            r = parse_cell(ci, stem, f, code_hash)
            if len(r["rows"]) < 6:
                base.fail(f"{stem}: only {len(r['rows'])} kept records (<6); aborting batch")
            cells_out.append(r)
            n_kept_total += len(r["rows"])
            per_cell_meta.append(dict(
                cell_index=int(ci), stem=stem, mat_path=str(f), raw_sha256=r["raw_hash"],
                first_full_rpt_record=int(r["k_first"]), cap0_Ah=float(r["cap0"]),
                n_cyc=int(r["n_cyc"]), n_kept=int(len(r["rows"])),
                counts=r["counts"], record_codes=r["codes"],
            ))
            log(f"B5/{stem}: DONE kept={len(r['rows'])} first_full_rpt=k{r['k_first']} cap0={r['cap0']:.6f}")

        if len(cells_out) != B5["expected_files"]:
            base.fail(f"expected {B5['expected_files']} cells, processed {len(cells_out)}")

        # Merge to XJTU_B5.npz (flat schema identical to base.merge_batch).
        all_rows = [r for c in cells_out for r in c["rows"]]
        if not all_rows:
            base.fail("no rows retained for B5")
        cell = np.asarray([r["cell"] for r in all_rows], dtype=np.int64)
        cell_name = base.safe_str_array([r["cell_name"] for r in all_rows])
        cycle = np.asarray([r["cycle"] for r in all_rows], dtype=np.int64)
        soh = np.asarray([r["soh"] for r in all_rows], dtype=float)
        qv = np.vstack([r["qv"] for r in all_rows]).astype(float)
        i_ = np.asarray([r["i"] for r in all_rows], dtype=float)
        record_idx = np.asarray([r["record"] for r in all_rows], dtype=np.int64)
        out_path = base.DATA_DIR / B5["out_name"]
        base.save_npz(out_path, dict(cell=cell, cell_name=cell_name, cycle=cycle, soh=soh,
                                     qv=qv, i=i_, vgrid=base.VGRID.astype(float), record_idx=record_idx))
        log(f"B5: wrote {out_path.name} rows={len(all_rows)} cells={B5['expected_files']}")

        meta = dict(
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            code_sha256=code_hash, base_code_sha256=base_hash, base_module=str(Path(base.__file__).resolve()),
            raw_dir=str(RAW_B5_DIR), vertex_grid=[float(base.VGRID[0]), float(base.VGRID[-1]), int(base.VGRID.size)],
            batch_spec=dict(ref_discharge_current_A=float(B5["ref_discharge_current_A"]),
                            ref_discharge_current_tol=float(B5["ref_discharge_current_tol"]),
                            ref_charge_current_A_range=list(B5["ref_charge_current_A"]),
                            v_cutoff=float(B5["v_cutoff"]), v_start_min=float(B5["v_start_min"]),
                            negative_threshold_A=float(B5["negative_threshold_A"]),
                            min_negative_samples=int(B5["min_negative_samples"]),
                            relerr_tol=float(B5["relerr_tol"]), nominal_charge_A=float(B5["nominal_charge_A"]),
                            out_name=B5["out_name"]),
            notes=[
                "cap0 is the capacity of the first valid FULL-RPT discharge per cell, NOT summary[0].",
                "First-pass detection uses process_record with cap0=summary[k] to gate full-RPT; final soh uses first-RPT cap0 for every record.",
                "No arbitrary future normalization; no factory SOH; no model fitting.",
                "Per-cell retained-record indices and reject reasons stored under record_codes.",
            ],
            n_rows_total=int(len(all_rows)), n_kept_total=int(n_kept_total),
            n_cells=int(len(cells_out)), kept_min_threshold=6,
            per_cell=per_cell_meta,
        )
        base.save_npz  # keep linter quiet about unused
        with open(base.DATA_DIR / "dataset_metadata_b5.json", "w", encoding="utf-8") as fh:
            json.dump(base.to_jsonable(meta), fh, indent=2, ensure_ascii=False)
        log("DONE")
    except SystemExit:
        raise
    except Exception as e:
        log("UNHANDLED: " + repr(e)); log(traceback.format_exc()); raise
    finally:
        if LOG_FH is not None:
            LOG_FH.close(); LOG_FH = None


if __name__ == "__main__":
    main()
