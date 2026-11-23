import argparse
import hashlib
import json
import math
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from reference_models import (
    load_cells, fit, predict, select, audit_source,
    gain_cancel_assert, anchor_exact_assert,
    KINDS, VOLT_TAGS, GAIN_TAGS,
)


# ---------------- paths & constants ----------------
ROOT_DEFAULT = _SRC_DIR.parent if _SRC_DIR.name == 'src' else Path.cwd()
DS_OLD = "XJTU_B1"
DS_NEW_DEV = ["XJTU_B2", "XJTU_B3"]
DS_NEW_CONF = ["XJTU_B4", "XJTU_B6"]
DS_NASA = "NASA"

VOLT_SCENARIOS = list(VOLT_TAGS)
ALL_SCENARIOS = list(VOLT_TAGS) + list(GAIN_TAGS)

LOW_SOH_THRESHOLD = 0.85
THREADS = 2

SUCCESS_CRITERION = {
    "name": "primary_clean_equivs_cell_mae_vs_endpoint_and_logridge",
    "primary_kind": "primary",
    "comparators": ["endpoint", "logridge"],
    "scenario": "clean",
    "protocols": ["XJTU_B4", "XJTU_B6"],
    "min_reduction_vs_endpoint_frac": 0.10,
    "min_reduction_vs_logridge_frac": 0.10,
    "max_protocol_degradation_pp_vs_either_comparator": 0.25,
    "perturbation_scenarios": ["vp5", "vm5"],
    "note": "Descriptive practical screen; not a hypothesis test; primary must beat BOTH endpoint and logridge on pooled cell-equal clean MAE over B4/B6 by at least 10% each, and neither B4 nor B6 protocol may be >0.25pp worse than either comparator.",
    "declared_before_outcomes": True,
}


# ---------------- json safety ----------------
def _json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not JSON serializable: {type(o)}')


def _finite_or_none(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return x
    if not math.isfinite(xf):
        return None
    return xf


def _sanitize(o):
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return [_sanitize(v) for v in o.tolist()]
    if isinstance(o, float):
        return x if (x := _finite_or_none(o)) is not None else None
    return o


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_sanitize(payload), f, indent=2, allow_nan=False, default=_json_default)
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _code_hash(files: List[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode('utf-8'))
        h.update(b'\x00')
        h.update(p.read_bytes())
        h.update(b'\x00')
    return h.hexdigest()


# ---------------- args ----------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['source', 'development', 'loco', 'confirmation'])
    ap.add_argument('--source', required=True, choices=['XJTU_B1', 'NASA'])
    ap.add_argument('--root', default=None, help='project root; defaults to CWD')
    ap.add_argument('--replace-source', action='store_true',
                    help='source stage only: allow overwriting an existing source freeze if no confirmation started')
    return ap.parse_args()


# ---------------- path helpers ----------------
def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _paths(root: Path) -> Dict[str, Path]:
    src_dir = root / 'src' if (root / 'src').exists() else root
    spec_dir = root / 'specs' if (root / 'specs').exists() else root
    data_dir = root / 'data'
    return {
        'root': root,
        'p': root,
        'old': root,
        'out': root / 'results',
        'data_new': data_dir,
        'data_old': data_dir,
        'library': src_dir / 'reference_models.py',
        'runner': src_dir / 'method_run.py',
        'spec': spec_dir / 'response_method_spec.txt',
        'engine_contract': spec_dir / 'engine_contract.txt',
    }


def _ds_paths(paths: Dict[str, Path], source: str) -> Dict[str, Path]:
    data_dir = paths['data_old']
    return {
        'XJTU_B1': data_dir / 'XJTU_B1.npz',
        'NASA': data_dir / 'NASA.npz',
        'XJTU_B2': data_dir / 'XJTU_B2.npz',
        'XJTU_B3': data_dir / 'XJTU_B3.npz',
        'XJTU_B4': data_dir / 'XJTU_B4.npz',
        'XJTU_B6': data_dir / 'XJTU_B6.npz',
    }


def _results_dir(paths: Dict[str, Path], stage: str, source: str) -> Path:
    target = paths['out']
    if stage == 'source':
        return target / f'source_{source}'
    if stage == 'development':
        return target / 'development'
    if stage == 'loco':
        return target / f'loco_{source}'
    if stage == 'confirmation':
        return target / 'confirmation'
    raise ValueError(stage)


# ---------------- source-stage writes ----------------
def _source_candidate_rows_csv(kind_selections: Dict[str, Dict[str, Any]], out_csv: Path) -> None:
    rows = []
    for kind, sel in kind_selections.items():
        for r in sel.get('candidate_rows', []):
            row = dict(r)
            row['kind'] = kind
            rows.append(row)
    if not rows:
        pd.DataFrame([]).to_csv(out_csv, index=False)
        return
    df = pd.DataFrame(rows)
    cols = ['kind', 'val_cell', 'scenario', 'train_cells', 'mae', 'lambda', 'gamma', 'alpha']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    df.to_csv(out_csv, index=False, float_format='%.10g')


def _model_payload(model):
    return {k: _sanitize(v) for k,v in model.items() if k != 'model'}


def _save_trusted_models(out_dir: Path, models: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    import joblib
    hashes: Dict[str, str] = {}
    for kind, model in models.items():
        fp = out_dir / f'frozen_{kind}.joblib'
        joblib.dump(model, fp)
        hashes[kind] = _sha256_file(fp)
    return hashes


def _load_trusted_model(fp: Path, expected_sha: str) -> Dict[str, Any]:
    import joblib
    if not fp.exists():
        raise FileNotFoundError(f'frozen model missing: {fp}')
    got = _sha256_file(fp)
    if got != expected_sha:
        raise RuntimeError(f'frozen model SHA mismatch for {fp}: {got} != {expected_sha}')
    return joblib.load(fp)


# ---------------- stage: source ----------------
def stage_source(args, paths: Dict[str, Path]) -> Dict[str, Any]:
    out_dir = _results_dir(paths, 'source', args.source)
    freeze_fp = out_dir / 'config_freeze.json'
    confirmation_started = (paths['out'] / 'confirmation' / 'confirmation_started.json').exists()
    if freeze_fp.exists():
        if not args.replace_source:
            raise RuntimeError(f'source freeze exists at {freeze_fp}; pass --replace-source')
        if confirmation_started:
            raise RuntimeError('refusing to overwrite source freeze after confirmation started')
        shutil.move(str(freeze_fp), str(freeze_fp.with_suffix('.superseded.json')))

    out_dir.mkdir(parents=True, exist_ok=True)
    ds_paths = _ds_paths(paths, args.source)
    src_path = ds_paths[args.source]
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    source_code_hash = _code_hash([paths['library'], paths['runner'], paths['spec']])
    source_npz_hash = _sha256_file(src_path)

    cells = load_cells(str(src_path), args.source)
    cells = sorted(cells, key=lambda c: c.name)
    audit = audit_source(cells)

    # gain cancellation invariant is a source-only check; do not score data here
    gain_cancel_assert(cells, tol=1e-12)

    selections: Dict[str, Dict[str, Any]] = {}
    for kind in KINDS:
        t0 = time.time()
        print(f'[source] selecting {kind}',flush=True)
        sel = select(kind, cells)
        sel['elapsed_sec'] = time.time() - t0
        selections[kind] = sel

    models: Dict[str, Dict[str, Any]] = {}
    for kind in KINDS:
        params = dict(selections[kind]['chosen'])
        models[kind] = fit(kind, params, cells)

    # exactness of anchor on source cells (library-level assertion; kept here for audit trail)
    anchor_checks = {}
    for kind in KINDS:
        try:
            for c in cells:
                anchor_exact_assert(models[kind], c, scenarios=['clean', 'vp5', 'vm5'])
            anchor_checks[kind] = 'ok'
        except AssertionError as e:
            raise RuntimeError(f'Anchor check failed for {kind}: {e}') from e

    _source_candidate_rows_csv(selections, out_dir / 'candidate_rows.csv')

    summary_rows = []
    for kind in KINDS:
        sel = selections[kind]
        summary_rows.append({
            'kind': kind,
            'score': _finite_or_none(sel.get('score')),
            'score_clean': _finite_or_none(sel.get('score_clean')),
            'score_robust': _finite_or_none(sel.get('score_robust')),
            'chosen_lambda': _finite_or_none(sel['chosen'].get('lambda')),
            'chosen_gamma': _finite_or_none(sel['chosen'].get('gamma')),
            'chosen_alpha': _finite_or_none(sel['chosen'].get('alpha')),
            'elapsed_sec': _finite_or_none(sel.get('elapsed_sec')),
            'anchor_check': anchor_checks.get(kind, ''),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / 'source_cv_summary.csv', index=False, float_format='%.10g')

    model_hashes = _save_trusted_models(out_dir, models)
    model_payloads = {kind: _model_payload(models[kind]) for kind in KINDS}

    Gscale_meta = {
        'G_scale': 'raw_half_diff_logratio_no_delta_div',
        'v_definition': '0.5*(log(D_vp5/Dref_vp5) - log(D_vm5/Dref_vm5))',
        'scaling_notes': 'no division by delta0; averaged across cells and rows inside source fit'
    }

    freeze = {
        'freeze_schema': 1,
        'stage': 'source',
        'source_dataset': args.source,
        'source_npz_path': _rel(src_path, paths['root']),
        'source_npz_sha256': source_npz_hash,
        'code_files': {p.name: _rel(p, paths['root']) for p in [paths['library'], paths['runner'], paths['spec']]},
        'code_sha256': source_code_hash,
        'library_sha256': _sha256_file(paths['library']),
        'runner_sha256': _sha256_file(paths['runner']),
        'spec_sha256': _sha256_file(paths['spec']) if paths['spec'].exists() else None,
        'engine_contract_sha256': _sha256_file(paths['engine_contract']) if paths['engine_contract'].exists() else None,
        'kinds': list(KINDS),
        'scenarios_voltage': list(VOLT_SCENARIOS),
        'scenarios_gain': list(GAIN_TAGS),
        'chosen_params': {kind: dict(selections[kind]['chosen']) for kind in KINDS},
        'source_cv_scores': {
            kind: {
                'score': _finite_or_none(selections[kind].get('score')),
                'score_clean': _finite_or_none(selections[kind].get('score_clean')),
                'score_robust': _finite_or_none(selections[kind].get('score_robust')),
            } for kind in KINDS
        },
        'model_joblib_sha256': model_hashes,
        'model_payloads': model_payloads,
        'Gscale_meta': Gscale_meta,
        'anchor_cost': 'ONE anchor label per cell (ref_y) available to every model equally',
        'success_criterion': SUCCESS_CRITERION,
        'audit': audit,
        'anchor_checks': anchor_checks,
        'threads': THREADS,
        'written_at_epoch_s': time.time(),
    }
    _dump_json(freeze_fp, freeze)
    return {'freeze': str(freeze_fp), 'n_cells': len(cells)}


# ---------------- target-stage helpers ----------------
def _verify_freeze(freeze_fp: Path, paths: Dict[str, Path]) -> Dict[str, Any]:
    if not freeze_fp.exists():
        raise RuntimeError(f'freeze missing at {freeze_fp}')
    with open(freeze_fp, 'r', encoding='utf-8') as f:
        fz = json.load(f)
    if fz.get('code_sha256') != _code_hash([paths['library'], paths['runner'], paths['spec']]):
        raise RuntimeError('code SHA mismatch vs freeze')
    if fz.get('library_sha256') != _sha256_file(paths['library']):
        raise RuntimeError('library SHA mismatch vs freeze')
    if fz.get('runner_sha256') != _sha256_file(paths['runner']):
        raise RuntimeError('runner SHA mismatch vs freeze')
    if fz.get('spec_sha256') and paths['spec'].exists():
        if fz['spec_sha256'] != _sha256_file(paths['spec']):
            raise RuntimeError('spec SHA mismatch vs freeze')
    return fz


def _load_frozen_models(fz: Dict[str, Any], out_dir: Path) -> Dict[str, Dict[str, Any]]:
    models = {}
    for kind in KINDS:
        fp = out_dir / f'frozen_{kind}.joblib'
        sha = fz['model_joblib_sha256'][kind]
        models[kind] = _load_trusted_model(fp, sha)
    return models


def _rows_for_cell(cell) -> np.ndarray:
    from reference_models import _rows_for_predict  # noqa: internal for output row alignment only
    return _rows_for_predict(cell)


def _predict_all(paths: Dict[str, Path], cells_by_ds: Dict[str, List[Any]],
                 models: Dict[str, Dict[str, Any]], tag: str,
                 row_out_fp: Path, cell_out_fp: Path,
                 extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Predict all (cell, model, scenario) with real per-row outputs; write per-row and per-cell CSVs."""
    row_rows: List[Dict[str, Any]] = []
    cell_metrics: List[Dict[str, Any]] = []

    def _do_one(ds_name, cell, kind, scen):
        rows = _rows_for_cell(cell)
        ypred = predict(models[kind], cell, scen)
        yt = cell.y[rows]
        if ypred.shape != yt.shape:
            raise RuntimeError(f'predict shape mismatch {ds_name}/{cell.name}/{kind}/{scen}')
        if not np.all(np.isfinite(ypred)):
            raise RuntimeError(f'non-finite prediction {ds_name}/{cell.name}/{kind}/{scen}')
        cycles = np.asarray(cell.cycles)[rows]
        row_idx = np.asarray(cell.row_index)[rows]
        ref_cycle = float(cell.ref_cycle) if np.isfinite(float(cell.ref_cycle)) else float('nan')
        dy = ypred - yt
        mae = float(np.mean(np.abs(dy))) if dy.size else float('nan')
        rmse = float(np.sqrt(np.mean(dy * dy))) if dy.size else float('nan')
        bias = float(np.mean(dy)) if dy.size else float('nan')
        return (ds_name, cell.name, kind, scen, rows, cycles, row_idx, ref_cycle, yt, ypred, mae, rmse, bias)

    tasks = []
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        for ds_name, cells in cells_by_ds.items():
            for cell in cells:
                for kind in KINDS:
                    for scen in ALL_SCENARIOS:
                        tasks.append(ex.submit(_do_one, ds_name, cell, kind, scen))
        for fut in tasks:
            try:
                (ds_name, cell_name, kind, scen, rows, cycles, row_idx, ref_cycle,
                 yt, ypred, mae, rmse, bias) = fut.result()
            except Exception as e:
                raise RuntimeError(f'prediction task failed: {e}') from e
            for i in range(rows.size):
                row_rows.append({
                    'dataset': ds_name,
                    'cell': cell_name,
                    'cycle': float(cycles[i]),
                    'row_index': int(row_idx[i]),
                    'ref_cycle': ref_cycle,
                    'model': kind,
                    'scenario': scen,
                    'y_true': float(yt[i]),
                    'y_pred': float(ypred[i]),
                })
            cell_metrics.append({
                'dataset': ds_name, 'cell': cell_name, 'model': kind, 'scenario': scen,
                'n': int(rows.size), 'mae': mae, 'rmse': rmse, 'bias': bias,
                'ref_cycle': ref_cycle,
            })

    dfr = pd.DataFrame(row_rows)
    dfr.to_csv(row_out_fp, index=False, float_format='%.10g')
    dfc = pd.DataFrame(cell_metrics)
    dfc.to_csv(cell_out_fp, index=False, float_format='%.10g')

    # aggregate by (dataset, model, scenario)
    agg = {}
    for (ds, kind, scen), g in dfc.groupby(['dataset', 'model', 'scenario']):
        mae = np.asarray(g['mae'], dtype=float)
        clean_mae = mae[np.isfinite(mae)]
        mean_mae = float(np.mean(clean_mae)) if clean_mae.size else float('nan')
        sd_mae = float(np.std(clean_mae, ddof=1)) if clean_mae.size > 1 else float('nan')
        agg[f'{ds}|{kind}|{scen}'] = {'n_cells': int(clean_mae.size), 'mean_mae': mean_mae, 'sd_mae': sd_mae}

    return {'n_rows': int(dfr.shape[0]), 'n_cell_metrics': int(dfc.shape[0]), 'aggregate': agg}


def _low_soh_strata(cells_by_ds: Dict[str, List[Any]], threshold: float = LOW_SOH_THRESHOLD) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for ds_name, cells in cells_by_ds.items():
        present = 0
        names = []
        for c in cells:
            rows = _rows_for_cell(c)
            if rows.size and np.any(c.y[rows] <= threshold):
                present += 1
                names.append(c.name)
        out[ds_name] = {'n_present_cells': present, 'cells': names}
    return out


def _paired_improvements(cell_df: pd.DataFrame) -> Dict[str, Any]:
    """Descriptive per-cell paired MAE reductions b - primary on the SAME scenario, same dataset.
    Uses clean and vp5/vm5 for protocol; does NOT compute bootstrap CIs."""
    out: Dict[str, Any] = {}
    scen_list = ['clean', 'vp5', 'vm5']
    comparators = ['endpoint', 'uniform', 'rawridge', 'logridge']
    for ds, g in cell_df.groupby('dataset'):
        for scen in scen_list:
            sub = g[g['scenario'] == scen]
            prim = sub[sub['model'] == 'primary'][['cell', 'mae']].set_index('cell')['mae']
            for comp in comparators:
                cm = sub[sub['model'] == comp][['cell', 'mae']].set_index('cell')['mae']
                common = prim.index.intersection(cm.index)
                if common.size == 0:
                    continue
                diff = cm.loc[common].to_numpy() - prim.loc[common].to_numpy()
                key = f'{ds}|{comp}|{scen}'
                out[key] = {
                    'n_cells': int(common.size),
                    'mean_diff_mae': float(np.mean(diff)),
                    'median_diff_mae': float(np.median(diff)),
                    'frac_improved': float(np.mean(diff > 0)),
                    'note': 'positive value means comparator is worse (primary better)',
                }
    return out


# ---------------- stage: development ----------------
def stage_development(args, paths: Dict[str, Path]) -> Dict[str, Any]:
    source = args.source
    out_src = _results_dir(paths, 'source', source)
    freeze_fp = out_src / 'config_freeze.json'
    fz = _verify_freeze(freeze_fp, paths)
    models = _load_frozen_models(fz, out_src)

    ds_paths = _ds_paths(paths, source)
    cells_by_ds: Dict[str, List[Any]] = {}
    for ds in DS_NEW_DEV:
        fp = ds_paths[ds]
        if not fp.exists():
            raise FileNotFoundError(fp)
        cells = sorted(load_cells(str(fp), ds), key=lambda c: c.name)
        cells_by_ds[ds] = cells

    out = _results_dir(paths, 'development', source)
    out.mkdir(parents=True, exist_ok=True)
    pred = _predict_all(paths, cells_by_ds, models, tag='development',
                        row_out_fp=out / 'predictions.csv',
                        cell_out_fp=out / 'cell_metrics.csv')
    paired = _paired_improvements(pd.read_csv(out / 'cell_metrics.csv'))
    low = _low_soh_strata(cells_by_ds)
    meta = {
        'stage': 'development',
        'source': source,
        'freeze_sha256': fz.get('code_sha256'),
        'source_npz_sha256': fz.get('source_npz_sha256'),
        'datasets': DS_NEW_DEV,
        'dataset_sha256': {ds: _sha256_file(ds_paths[ds]) for ds in DS_NEW_DEV},
        'cohorts': {ds: {'n_cells': len(cells_by_ds[ds]),
                          'n_scored_rows_per_cell': {c.name: int(_rows_for_cell(c).size) for c in cells_by_ds[ds]}}
                     for ds in DS_NEW_DEV},
        'low_soh_strata': low,
        'paired_improvements': paired,
        'predictions_summary': {k: v for k, v in pred['aggregate'].items()},
        'n_rows': pred['n_rows'],
        'n_cell_metrics': pred['n_cell_metrics'],
    }
    _dump_json(out / 'development_summary.json', meta)
    return {'summary': str(out / 'development_summary.json'), 'n_rows': pred['n_rows']}


# ---------------- stage: confirmation ----------------
def _criterion_evaluation(cell_df: pd.DataFrame, criterion: Dict[str, Any]) -> Dict[str, Any]:
    scen = criterion['scenario']
    kinds = [criterion['primary_kind']] + list(criterion['comparators'])
    res: Dict[str, Any] = {'per_protocol': {}, 'pooled': {}}
    protocol_means: Dict[str, Dict[str, float]] = {}
    for ds in criterion['protocols']:
        sub = cell_df[(cell_df['dataset'] == ds) & (cell_df['scenario'] == scen)]
        protocol_means[ds] = {}
        for kind in kinds:
            g = sub[sub['model'] == kind]
            vals = np.asarray(g['mae'], dtype=float)
            vals = vals[np.isfinite(vals)]
            protocol_means[ds][kind] = float(np.mean(vals)) if vals.size else float('nan')
        res['per_protocol'][ds] = dict(protocol_means[ds])
    # combined protocol-equal mean across protocols
    for kind in kinds:
        vals = np.array([protocol_means[ds][kind] for ds in criterion['protocols']], dtype=float)
        res['pooled'][kind] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float('nan')

    def _reduction(base, prim):
        if base is None or prim is None:
            return None
        if not (math.isfinite(base) and math.isfinite(prim)) or base <= 0:
            return None
        return float((base - prim) / base)

    pooled = res['pooled']
    red_endpoint = _reduction(pooled.get('endpoint'), pooled.get('primary'))
    red_logridge = _reduction(pooled.get('logridge'), pooled.get('primary'))

    protocol_deg: Dict[str, Dict[str, float]] = {}
    for ds in criterion['protocols']:
        pm = protocol_means[ds]
        deg: Dict[str, float] = {}
        for comp_key in ['endpoint', 'logridge']:
            if math.isfinite(pm.get('primary', float('nan'))) and math.isfinite(pm.get(comp_key, float('nan'))):
                deg[comp_key] = float(pm['primary'] - pm[comp_key])  # positive = worse than comparator
            else:
                deg[comp_key] = float('nan')
        protocol_deg[ds] = deg

    ok_red_e = red_endpoint is not None and red_endpoint >= criterion['min_reduction_vs_endpoint_frac']
    ok_red_l = red_logridge is not None and red_logridge >= criterion['min_reduction_vs_logridge_frac']
    degrad_ok = True
    for ds, deg in protocol_deg.items():
        for ck, val in deg.items():
            if not math.isfinite(val):
                degrad_ok = False
            elif val > criterion['max_protocol_degradation_pp_vs_either_comparator']:
                degrad_ok = False
    criterion_met = bool(ok_red_e and ok_red_l and degrad_ok)

    return {
        'pooled_means': pooled,
        'per_protocol_means': protocol_means,
        'reduction_vs_endpoint_frac': red_endpoint,
        'reduction_vs_logridge_frac': red_logridge,
        'protocol_degradation_pp': protocol_deg,
        'criterion_met': criterion_met,
        'criterion_notes': 'criterion_met is a mechanical AND of frozen thresholds; no automatic claim of practical value beyond the frozen screen',
    }


def stage_confirmation(args, paths: Dict[str, Path]) -> Dict[str, Any]:
    source = args.source
    out_src = _results_dir(paths, 'source', source)
    freeze_fp = out_src / 'config_freeze.json'
    fz = _verify_freeze(freeze_fp, paths)
    models = _load_frozen_models(fz, out_src)

    out = _results_dir(paths, 'confirmation', source)
    out.mkdir(parents=True, exist_ok=True)
    started_fp = out / 'confirmation_started.json'
    complete_fp = out / 'confirmation_complete.json'
    if started_fp.exists() or complete_fp.exists():
        raise RuntimeError(f'refusing: confirmation already started or complete at {out}')

    # credit receipt BEFORE any target file loads
    ds_paths = _ds_paths(paths, source)
    receipt = {
        'stage': 'confirmation',
        'source': source,
        'freeze_json': _rel(freeze_fp, paths['root']),
        'freeze_code_sha256': fz.get('code_sha256'),
        'source_npz_sha256': fz.get('source_npz_sha256'),
        'model_joblib_sha256': fz.get('model_joblib_sha256'),
        'chosen_params': fz.get('chosen_params'),
        'success_criterion': fz.get('success_criterion'),
        'Gscale_meta': fz.get('Gscale_meta'),
        'target_files': {ds: _rel(ds_paths[ds], paths['root']) for ds in DS_NEW_CONF},
        'target_files_exist_precheck': {ds: ds_paths[ds].exists() for ds in DS_NEW_CONF},
        'started_at_epoch_s': time.time(),
    }
    _dump_json(started_fp, receipt)

    cells_by_ds: Dict[str, List[Any]] = {}
    for ds in DS_NEW_CONF:
        fp = ds_paths[ds]
        if not fp.exists():
            raise FileNotFoundError(fp)
        cells = sorted(load_cells(str(fp), ds), key=lambda c: c.name)
        cells_by_ds[ds] = cells

    pred = _predict_all(paths, cells_by_ds, models, tag='confirmation',
                        row_out_fp=out / 'predictions.csv',
                        cell_out_fp=out / 'cell_metrics.csv')
    cell_df = pd.read_csv(out / 'cell_metrics.csv')
    paired = _paired_improvements(cell_df)
    low = _low_soh_strata(cells_by_ds)
    criterion_eval = _criterion_evaluation(cell_df, fz['success_criterion'])

    summary = {
        'stage': 'confirmation',
        'source': source,
        'freeze_json': _rel(freeze_fp, paths['root']),
        'freeze_code_sha256': fz.get('code_sha256'),
        'source_npz_sha256': fz.get('source_npz_sha256'),
        'target_dataset_sha256': {ds: _sha256_file(ds_paths[ds]) for ds in DS_NEW_CONF},
        'cohorts': {ds: {'n_cells': len(cells_by_ds[ds]),
                          'n_scored_rows_per_cell': {c.name: int(_rows_for_cell(c).size) for c in cells_by_ds[ds]}}
                     for ds in DS_NEW_CONF},
        'low_soh_strata': low,
        'paired_improvements': paired,
        'criterion': fz['success_criterion'],
        'criterion_evaluation': criterion_eval,
        'G_scale_meta': fz.get('Gscale_meta'),
        'n_rows': pred['n_rows'],
        'n_cell_metrics': pred['n_cell_metrics'],
        'aggregate': pred['aggregate'],
        'written_at_epoch_s': time.time(),
        'note': 'Descriptive results only; criterion_met is mechanical; no post-hoc retuning; this file is immutable protocol record.',
    }
    _dump_json(out / 'confirmation_summary.json', summary)
    _dump_json(complete_fp, {'complete': True, 'summary': _rel(out / 'confirmation_summary.json', paths['root']),
                             'finished_at_epoch_s': time.time()})
    return {'summary': _rel(out / 'confirmation_summary.json', paths['root']), 'criterion_met': criterion_eval['criterion_met']}


# ---------------- stage: loco ----------------
def stage_loco(args, paths: Dict[str, Path]) -> Dict[str, Any]:
    source = args.source
    ds_paths = _ds_paths(paths, source)
    fp = ds_paths[source]
    if not fp.exists():
        raise FileNotFoundError(fp)
    cells = sorted(load_cells(str(fp), source), key=lambda c: c.name)
    out = _results_dir(paths, 'loco', source)
    out.mkdir(parents=True, exist_ok=True)
    fold_cfg = {
        'stage': 'loco',
        'source': source,
        'source_npz_path': _rel(fp, paths['root']),
        'source_npz_sha256': _sha256_file(fp),
        'code_sha256': _code_hash([paths['library'], paths['runner'], paths['spec']]),
        'kinds': list(KINDS),
        'scenarios': ALL_SCENARIOS,
        'threads': THREADS,
        'n_outer_cells': len(cells),
        'outer_cells': [c.name for c in cells],
        'note': 'per-outer-cell inner selection over other cells; does not read or write global freeze',
        'started_at_epoch_s': time.time(),
    }
    _dump_json(out / 'fold_config.json', fold_cfg)

    per_fold_rows = []
    per_fold_cell_metrics = []
    for outer in cells:
        train_cells = [c for c in cells if c.name != outer.name]
        if not train_cells:
            continue
        models: Dict[str, Dict[str, Any]] = {}
        chosen: Dict[str, Dict[str, Any]] = {}
        for kind in KINDS:
            sel = select(kind, train_cells)
            chosen[kind] = dict(sel['chosen'])
            models[kind] = fit(kind, chosen[kind], train_cells)
        for kind in KINDS:
            for scen in ALL_SCENARIOS:
                rows = _rows_for_cell(outer)
                if rows.size == 0:
                    per_fold_cell_metrics.append({'outer_cell': outer.name, 'model': kind, 'scenario': scen,
                                                  'n': 0, 'mae': float('nan'), 'rmse': float('nan'),
                                                  'bias': float('nan'), 'ref_cycle': float('nan')})
                    continue
                yp = predict(models[kind], outer, scen)
                yt = outer.y[rows]
                if yp.shape != yt.shape:
                    raise RuntimeError(f'LOCO predict mismatch {outer.name}/{kind}/{scen}')
                dy = yp - yt
                mae = float(np.mean(np.abs(dy))) if dy.size else float('nan')
                rmse = float(np.sqrt(np.mean(dy * dy))) if dy.size else float('nan')
                bias = float(np.mean(dy)) if dy.size else float('nan')
                for i in range(rows.size):
                    per_fold_rows.append({
                        'outer_cell': outer.name,
                        'cycle': float(np.asarray(outer.cycles)[rows][i]),
                        'row_index': int(np.asarray(outer.row_index)[rows][i]),
                        'ref_cycle': float(outer.ref_cycle),
                        'model': kind,
                        'scenario': scen,
                        'y_true': float(yt[i]),
                        'y_pred': float(yp[i]),
                    })
                per_fold_cell_metrics.append({
                    'outer_cell': outer.name, 'model': kind, 'scenario': scen,
                    'n': int(rows.size), 'mae': mae, 'rmse': rmse, 'bias': bias,
                    'ref_cycle': float(outer.ref_cycle),
                })
        _dump_json(out / f'chosen_{outer.name}.json', chosen)

    pd.DataFrame(per_fold_rows).to_csv(out / 'predictions.csv', index=False, float_format='%.10g')
    pd.DataFrame(per_fold_cell_metrics).to_csv(out / 'cell_metrics.csv', index=False, float_format='%.10g')

    agg = {}
    dfc = pd.DataFrame(per_fold_cell_metrics)
    for (kind, scen), g in dfc.groupby(['model', 'scenario']):
        mae = np.asarray(g['mae'], dtype=float)
        mae = mae[np.isfinite(mae)]
        agg[f'{kind}|{scen}'] = {'n_cells': int(mae.size),
                                  'mean_mae': float(np.mean(mae)) if mae.size else float('nan'),
                                  'sd_mae': float(np.std(mae, ddof=1)) if mae.size > 1 else float('nan')}
    _dump_json(out / 'loco_summary.json', {'aggregate': agg, 'n_rows': len(per_fold_rows),
                                          'n_cells': len(cells), 'source': source})
    return {'summary': str(out / 'loco_summary.json'), 'n_rows': len(per_fold_rows)}


# ---------------- main ----------------
def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args() if argv is None else _parse_args_or_default(argv)
    root = Path(args.root).resolve() if args.root else ROOT_DEFAULT.resolve()
    paths = _paths(root)

    if args.stage != 'source':
        if not paths['library'].exists():
            raise FileNotFoundError(paths['library'])

    try:
        if args.stage == 'source':
            info = stage_source(args, paths)
        elif args.stage == 'development':
            info = stage_development(args, paths)
        elif args.stage == 'confirmation':
            info = stage_confirmation(args, paths)
        elif args.stage == 'loco':
            info = stage_loco(args, paths)
        else:
            raise ValueError(args.stage)
    except Exception as e:
        sys.stderr.write(f'[ERROR] stage={args.stage} source={args.source}: {e}\n')
        traceback.print_exc()
        return 2

    with open(os.devnull, 'w') as _devnull:
        pass
    print(json.dumps(_sanitize({'ok': True, 'stage': args.stage, 'source': args.source, 'info': info}),
                     indent=2, allow_nan=False, default=_json_default))
    return 0


def _parse_args_or_default(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['source', 'development', 'loco', 'confirmation'])
    ap.add_argument('--source', required=True, choices=['XJTU_B1', 'NASA'])
    ap.add_argument('--root', default=None)
    ap.add_argument('--replace-source', action='store_true')
    return ap.parse_args(argv)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        sys.exit(main())
