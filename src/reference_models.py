import json, hashlib, math, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

# ---------------- constants ----------------
NODE_IDX = np.array([0, 8, 16, 24, 31], dtype=int)  # 3.80,3.84,3.94,4.01,4.10 on 70-pt grid from 3.8..4.1 step=(4.1-3.8)/69
NODE_V = 3.8 + (4.1 - 3.8) * NODE_IDX / 31.0
N32 = 32
GRID32 = np.linspace(3.8, 4.1, 32)
V70 = np.linspace(3.5, 4.19, 70)
VOLT_TAGS = ['clean', 'vp5', 'vm5', 'vp10', 'vm10']
VOLT_DELTA = {'clean': 0.0, 'vp5': 0.005, 'vm5': -0.005, 'vp10': 0.01, 'vm10': -0.01}
GAIN_TAGS = ['gstatp1', 'gstatm1', 'gdriftp1', 'gdriftm1', 'grefp1', 'grefm1']
LAMBDAS = [1e-6, 1e-4, 0.01, 0.1]
GAMMAS = [0.0, 0.1, 1.0, 10.0, 100.0]
ALPHAS_ABS = [0.001, 0.1, 10.0, 1000.0]
SIGNS = np.array([[s0, s1, s2, s3] for s0 in (-1, 1) for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)], dtype=float)  # 16x4
W0 = np.full(4, 0.25)
KAPPA_DEFAULT = 3.0
KAPPA_SIMPLEX = 1.0
SEG_MIN = 1e-10
KKT_TOL = 1e-5
FEAS_TOL = 1e-7
FOLD_GROUP_K = 3
# Columns used by absridge on 32-node cumulative q: indices chosen to match [4.10, 3.84, 3.94, 4.01]
ABS_COLS = np.array([31, 8, 16, 24], dtype=int)

# ---------------- cell loading ----------------

@dataclass
class Cell:
    name: str
    ds: str
    cycles: np.ndarray
    row_index: np.ndarray
    y: np.ndarray
    ref_y: float
    ref_cycle: float
    ref_row: int
    # Future-only disjoint segment charges under 5 voltage scenarios and gain scenarios.
    # D[tag] shape (n,4): charges over disjoint segments j=0..3 across (a, b_j) sub-windows
    D: Dict[str, np.ndarray] = field(default_factory=dict)
    Dref: Dict[str, np.ndarray] = field(default_factory=dict)  # shape (4,)
    # Future-only 32-node cumulative partial charge q(v)-q(a) for v in GRID32, per scenario.
    Q: Dict[str, np.ndarray] = field(default_factory=dict)  # shape (n,32)
    Qref: Dict[str, np.ndarray] = field(default_factory=dict)  # shape (32,)


# Strict np.interp over a strictly ordered 70-point voltage grid, no bridging outside coverage.
def _interp_strict(q, v, vq, tol=0.0):
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    vq = np.asarray(vq, dtype=float)
    if q.shape != vq.shape:
        raise ValueError(f'curve shape mismatch {q.shape} vs {vq.shape}')
    m = np.isfinite(q) & np.isfinite(vq)
    if not np.all(m):
        idx = np.where(m)[0]
        if idx.size == 0:
            raise ValueError('no finite voltage samples')
        i0, i1 = idx[0], idx[-1]
        if not np.all(m[i0:i1 + 1]):
            raise ValueError('non-contiguous finite coverage: refusing to bridge interior gap')
        q = q[i0:i1 + 1]
        vq = vq[i0:i1 + 1]
    if np.any(np.diff(vq) <= tol):
        raise ValueError('vq must be strictly increasing and finite')
    if v.min() < vq[0] - 1e-12 or v.max() > vq[-1] + 1e-12:
        raise ValueError(f'requested v out of family coverage: v=[{v.min()},{v.max()}], curve=[{vq[0]},{vq[-1]}]')
    return np.interp(v, vq, q)


def _load_array(npz, names, cell, what):
    for n in names:
        if n in npz:
            return npz[n]
    raise KeyError(f'{what} missing for cell {cell}; tried {names}')


def load_cells(path: str, ds: str) -> List[Cell]:
    cells: List[Cell] = []
    with np.load(path, allow_pickle=False) as npz:
        for key in ('cell', 'cell_name', 'cycle', 'soh', 'qv', 'vgrid'):
            if key not in npz:
                raise KeyError(f'load_cells: required field {key!r} not found in {path}')
        raw_cell = np.asarray(npz['cell']).reshape(-1)
        raw_name = np.asarray(npz['cell_name']).reshape(-1)
        raw_cyc = np.asarray(npz['cycle']).reshape(-1)
        raw_y = np.asarray(npz['soh'], dtype=float).reshape(-1)
        raw_qv = np.asarray(npz['qv'], dtype=float)
        raw_vq = np.asarray(npz['vgrid'], dtype=float).reshape(-1)
        n = raw_y.size
        for arr, nm in ((raw_cell, 'cell'), (raw_name, 'cell_name'), (raw_cyc, 'cycle')):
            if arr.size != n:
                raise ValueError(f'load_cells: {nm} length {arr.size} != soh length {n}')
        if raw_qv.ndim != 2 or raw_qv.shape[0] != n:
            raise ValueError(f'load_cells: qv shape {raw_qv.shape} incompatible with n={n}')
        if raw_qv.shape[1] != raw_vq.size:
            raise ValueError(f'load_cells: qv columns {raw_qv.shape[1]} != vgrid {raw_vq.size}')
        if raw_vq.size != 70:
            raise ValueError(f'load_cells: vgrid must have 70 points, got {raw_vq.size}')
        if not (np.isclose(raw_vq[0], V70[0], atol=1e-9) and np.isclose(raw_vq[-1], V70[-1], atol=1e-9)):
            raise ValueError(f'load_cells: vgrid endpoints {raw_vq[0]}..{raw_vq[-1]} != expected {V70[0]}..{V70[-1]}')

        groups: Dict[str, List[int]] = {}
        for i in range(n):
            cid = int(raw_cell[i])
            nm = str(raw_name[i])
            groups.setdefault(nm, []).append(i)

        for name in sorted(groups.keys()):
            idxs = np.array(groups[name], dtype=int)
            cyc = raw_cyc[idxs]
            order = np.argsort(cyc, kind='stable')
            cyc_o = raw_cyc[idxs][order]
            rig_o = idxs.astype(np.int64)[order]
            y_o = raw_y[idxs][order]
            qv_o = raw_qv[idxs][order]
            cell = _build_cell_from_rows(
                name=name,
                ds=ds,
                cyc=cyc_o,
                rig=rig_o,
                y=y_o,
                qv2=qv_o,
                vq=raw_vq,
                order=np.arange(idxs.size),
            )
            cells.append(cell)
    if not cells:
        raise ValueError(f'load_cells: no cells found in {path}')
    return cells


def _reference_curve(npz, name, key_pat: List[str], vq, n):
    for k in key_pat:
        if k in npz:
            a = np.asarray(npz[k], dtype=float).reshape(-1)
            if a.size != vq.size:
                raise ValueError(f'{name}: reference curve size {a.size} != v_grid {vq.size}')
            return a
    raise KeyError(f'{name}: reference curve missing; tried {key_pat}')


def _populate_features(cell: Cell, Qbase: np.ndarray, vq: np.ndarray, npz) -> None:
    # For each of the 5 voltage scenarios, compute the 32-node q(v) values via strict interpolation,
    # then subtract anchor q(a) to form cumulative partial charge, and split into 4 disjoint segment charges on 32 nodes.
    n = cell.y.size
    # Anchor curve is first-chronological row by observables only; contract: anchor is the first record.
    # We treat row0 as the anchor curve for Qref (defensible: 'first record' by observables), matching the anchor row.
    a_q = None
    for tag in VOLT_TAGS:
        d = VOLT_DELTA[tag]
        v_query = GRID32 + d
        # strictly within family coverage
        if v_query.min() < vq[0] - 1e-12 or v_query.max() > vq[-1] + 1e-12:
            raise ValueError(f'{cell.name}:{tag}: q(v\u00b1delta) out of coverage {v_query.min()}..{v_query.max()} vs {vq[0]}..{vq[-1]}')
        Qv = np.stack([_interp_strict(Qbase[i], v_query, vq, 1e-14) for i in range(n)], axis=0)  # n x 32
        # reference curve: reference row of the same cell under shifted grid
        ref_row = cell.ref_row
        if ref_row < 0 or ref_row >= n:
            raise ValueError(f'{cell.name}: ref_row {ref_row} out of range')
        qref_v = _interp_strict(Qbase[ref_row], v_query, vq, 1e-14)  # 32
        a_anchor = qref_v[0]  # q(3.80+\u03b4) of reference
        Q = Qv - a_anchor
        Qref = qref_v - a_anchor
        if tag == 'clean':
            a_q = a_anchor
        cell.Q[tag] = Q
        cell.Qref[tag] = Qref
        # Disjoint 4-segment charges on the 32 cumulative grid (Euclidean segmentation of indices).
        # Node breaks at 0..15, 16..23, 24..29, 30..31 not defensible; use the four disjoint node blocks defined by NODE_IDX-adjacent
        # segments on the 32 grid: ([0..7],[8..15],[16..23],[24..31]) for consistency with absridge ABS_COLS in 32 grid.
        breaks = np.array([0, 8, 16, 24, 31], dtype=int)
        seg = np.stack([Q[:, breaks[j] + 1:breaks[j + 1] + 1].sum(axis=1) - 0.0 for j in range(4)], axis=1)
        # For the first segment, indices [1..8]; subsequent [9..16], [17..24], [25..31]; cumulative differences equal
        # Q at break endpoints minus Q at previous breakpoints. This is the disjoint-segment charge (non-overlapping).
        # To make sums equal to cumulative partial charge difference at boundaries, use boundary differences:
        seg = np.stack([
            Q[:, breaks[j + 1]] if (breaks[j + 1] < Q.shape[1]) else Q[:, -1]
            for j in range(4)
        ], axis=1)
        # Correct disjoint-segment charges from cumulative Q with four segment sums:
        seg[:, 0] = Q[:, breaks[1]] - Q[:, breaks[0]]
        for j in range(1, 4):
            lo = breaks[j]
            hi = breaks[j + 1] if j + 1 < 5 else Q.shape[1] - 1
            seg[:, j] = Q[:, hi] - Q[:, lo]
        # Enforce positive segment charges > SEG_MIN; annotate which rows pass.
        bad = ~np.all(seg > SEG_MIN, axis=1)
        if bad.all():
            raise ValueError(f'{cell.name}:{tag}: no future rows with all four positive segments')
        cell.D[tag] = seg
        ref_seg = np.array([Qref[breaks[1]] - Qref[breaks[0]]] +
                           [Qref[(breaks[j + 1] if j + 1 < 5 else 31)] - Qref[breaks[j]] for j in range(1, 4)])
        cell.Dref[tag] = ref_seg.astype(float)
    cell.ref_cycle = float(cell.cycles[cell.ref_row]) if cell.ref_row < cell.cycles.size else float(cell.cycles[0])
    # Remind the caller: eligibility across ALL five scenarios is enforced inside fit/select/predict via same mask.
    # reference SOH y[ref_row] may or may not be finite; anchor y is ref_y by definition, we do not use y[ref_row].

def _assert_anchor_and_positive(cells: List[Cell]) -> None:
    for c in cells:
        for tag in VOLT_TAGS:
            if tag not in c.D or tag not in c.Dref:
                raise AssertionError(f'{c.name}: missing scenario {tag}')
            if not np.all(c.Dref[tag] > SEG_MIN):
                raise AssertionError(f'{c.name}:{tag}: non-positive reference segments')
            if not (c.D[tag] > 0).any():
                raise AssertionError(f'{c.name}:{tag}: no positive future segments')

# ---------------- eligibility, gains, weights ----------------

def _eligibility_mask(cell: Cell) -> np.ndarray:
    n = cell.y.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(n, dtype=bool)
    for tag in VOLT_TAGS:
        D = np.asarray(cell.D[tag], dtype=float)
        Q = np.asarray(cell.Q[tag], dtype=float)
        if D.shape[0] != n or Q.shape[0] != n:
            raise ValueError(f'{cell.name}:{tag}: D/Q row count mismatch in future-only arrays')
        mask &= np.isfinite(D).all(axis=1)
        mask &= np.isfinite(Q).all(axis=1)
        mask &= (D > SEG_MIN).all(axis=1)
    y = np.asarray(cell.y, dtype=float)
    mask &= np.isfinite(y)
    mask &= (y > 0)
    return mask


def _anchor_index(cell: Cell) -> int:
    return -1


def _future_rows(cell: Cell, exclude_anchor: bool = True) -> np.ndarray:
    n = cell.y.size
    if n == 0:
        return np.zeros(0, dtype=int)
    m = _eligibility_mask(cell)
    return np.where(m)[0]


def _gain_multiplier(tag: str) -> Tuple[float, float]:
    # returns (factor on future D/Q, factor on reference Dref/Qref)
    if tag == 'gstatp1':
        return 1.01, 1.01
    if tag == 'gstatm1':
        return 0.99, 0.99
    if tag == 'gdriftp1':
        return 1.01, 1.0
    if tag == 'gdriftm1':
        return 0.99, 1.0
    if tag == 'grefp1':
        return 1.0, 1.01
    if tag == 'grefm1':
        return 1.0, 0.99
    return 1.0, 1.0


def _scenario_arrays(cell: Cell, scenario: str, rows: np.ndarray, train: bool = False) -> Dict[str, np.ndarray]:
    if scenario in VOLT_TAGS:
        tag = scenario
        return {
            'D': cell.D[tag][rows],
            'Dref': cell.Dref[tag],
            'Q': cell.Q[tag][rows],
            'Qref': cell.Qref[tag],
            'y': cell.y[rows] if not train else cell.y[rows],
        }
    if scenario in GAIN_TAGS:
        # base voltage clean; multiply future and/or reference per contract.
        sf, sr = _gain_multiplier(scenario)
        return {
            'D': cell.D['clean'][rows] * sf,
            'Dref': cell.Dref['clean'] * sr,
            'Q': cell.Q['clean'][rows] * sf,
            'Qref': cell.Qref['clean'] * sr,
            'y': cell.y[rows],
        }
    raise ValueError(f'unknown scenario {scenario}')


def _source_row_weights(cells: List[Cell]) -> np.ndarray:
    rows_per = [_future_rows(c, exclude_anchor=True).size for c in cells]
    ncell = sum(1 for n in rows_per if n > 0)
    if ncell == 0:
        raise ValueError('no cells with eligible future rows')
    return np.array([1.0 / (ncell * max(n, 1)) for n in rows_per], dtype=float)


def _log_ratios(cells: List[Cell], scenario: str) -> np.ndarray:
    Xs = []
    for c in cells:
        rows = _future_rows(c, exclude_anchor=True)
        if rows.size == 0:
            Xs.append(np.zeros((0, 4)))
            continue
        A = _scenario_arrays(c, scenario, rows)
        Xs.append(np.log(A['D'] / A['Dref'][None, :]))
    return np.concatenate(Xs, axis=0) if Xs else np.zeros((0, 4))


def _z_targets(cells: List[Cell], scenario: str = 'clean') -> np.ndarray:
    zs = []
    for c in cells:
        rows = _future_rows(c, exclude_anchor=True)
        if rows.size == 0:
            zs.append(np.zeros(0))
            continue
        A = _scenario_arrays(c, 'clean', rows)
        zs.append(np.log(A['y'] / c.ref_y))
    return np.concatenate(zs) if zs else np.zeros(0)


def _row_weights_expanded(cells: List[Cell]) -> np.ndarray:
    rw = _source_row_weights(cells)
    out = []
    for w, c in zip(rw, cells):
        rows = _future_rows(c, exclude_anchor=True)
        out.append(np.full(rows.size, w, dtype=float))
    return np.concatenate(out) if out else np.zeros(0)

# ---------------- H, G, h ----------------

def _H_and_h(cells: List[Cell]) -> Tuple[np.ndarray, np.ndarray]:
    w = _row_weights_expanded(cells)
    X = _log_ratios(cells, 'clean')
    z = _z_targets(cells, 'clean')
    if X.shape[0] != w.size or z.size != w.size:
        raise ValueError('shape mismatch in _H_and_h')
    H = (X * w[:, None]).T @ X
    h = (X * w[:, None]).T @ z
    return H, h


def _G(cells: List[Cell]) -> np.ndarray:
    per_cell_means = []
    for c in cells:
        rows = _future_rows(c, exclude_anchor=True)
        if rows.size == 0:
            continue
        A5 = _scenario_arrays(c, 'vp5', rows)
        Am5 = _scenario_arrays(c, 'vm5', rows)
        xp = np.log(A5['D'] / A5['Dref'][None, :])
        xm = np.log(Am5['D'] / Am5['Dref'][None, :])
        v = 0.5 * (xp - xm)
        if v.size == 0:
            continue
        per_cell_means.append(v.mean(axis=0))
    if not per_cell_means:
        return np.zeros((4, 4))
    Vc = np.asarray(per_cell_means, dtype=float)
    return (Vc.T @ Vc) / Vc.shape[0]

# ---------------- anchored log-linear solver ----------------

def _solve_quadratic(H, h, lam, gam, G, kappa):
    from scipy.optimize import nnls as _nnls

    H = np.asarray(H, dtype=float)
    h = np.asarray(h, dtype=float).reshape(-1)
    G = np.asarray(G, dtype=float)
    n = 4
    if H.shape != (n, n) or G.shape != (n, n) or h.size != n:
        raise ValueError('H/G must be 4x4 and h length 4')
    A = H + float(lam) * np.eye(n) + float(gam) * G
    b = h + float(lam) * W0

    def fun(w):
        w = np.asarray(w, dtype=float).reshape(-1)
        return float(0.5 * w @ A @ w - b @ w)

    def jac(w):
        return A @ np.asarray(w, dtype=float).reshape(-1) - b

    cons = [{'type': 'eq', 'fun': (lambda w: float(np.sum(w) - 1.0)),
             'jac': (lambda w: np.ones(n))}]
    for s in SIGNS:
        cons.append({'type': 'ineq',
                     'fun': (lambda w, s=s: float(kappa - s @ np.asarray(w, dtype=float).reshape(-1))),
                     'jac': (lambda w, s=s: -s)})

    starts = [W0.copy(), np.full(n, 0.25)]
    best = None
    last_res = None
    for w_init in starts:
        if abs(np.sum(w_init) - 1.0) > 1e-12:
            continue
        if np.any(np.abs(SIGNS @ w_init) > float(kappa) + FEAS_TOL):
            continue
        res = minimize(fun, w_init, jac=jac, constraints=cons, method='SLSQP',
                       options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})
        last_res = res
        w = np.asarray(res.x, dtype=float).reshape(-1)
        feas = float(abs(np.sum(w) - 1.0))
        feas = max(feas, float(np.max(np.abs(SIGNS @ w) - float(kappa), initial=0.0)))
        feas_ok = feas <= FEAS_TOL
        if not feas_ok:
            if best is None and bool(res.success):
                pass
            continue
        if (best is None) or (not bool(best[2].success)) or (bool(res.success) and fun(w) < best[1] - 1e-12):
            best = (w, float(fun(w)), res, feas)
        if bool(res.success):
            break

    if best is None:
        raise RuntimeError(
            'SLSQP failed to return a feasible point for '
            f'(lambda={lam}, gamma={gam}, kappa={kappa}); '
            f'last_success={getattr(last_res, "success", None)}, '
            f'last_message={getattr(last_res, "message", None)}'
        )

    w, _, res, feas_val = best

    grad = A @ w - b
    P = np.eye(n) - np.ones((n, n)) / float(n)
    active_mask = (np.abs(SIGNS @ w - float(kappa)) <= 1e-6)
    if np.any(active_mask):
        Cactive = SIGNS[active_mask, :]
    else:
        Cactive = np.zeros((0, n), dtype=float)
    if Cactive.shape[0] > 0:
        mu_raw = _nnls(P @ Cactive.T, -P @ grad)[0]
        mu = np.asarray(mu_raw, dtype=float).reshape(-1)
    else:
        mu = np.zeros(0, dtype=float)
    alpha = -float(np.mean(grad + (Cactive.T @ mu if Cactive.shape[0] > 0 else np.zeros(n))))
    residual = grad + alpha * np.ones(n) + (Cactive.T @ mu if Cactive.shape[0] > 0 else np.zeros(n))
    kkt_norm = float(np.linalg.norm(residual))

    l1_val = float(np.max(np.abs(SIGNS @ w)))
    l1_activity = bool(l1_val >= float(kappa) - 1e-6)
    neg_mass = float(-np.minimum(w, 0.0).sum())
    objective = float(fun(w))

    try:
        eigs = np.linalg.eigvalsh(A)
        eig_min = float(np.min(eigs))
    except np.linalg.LinAlgError:
        eig_min = float('nan')
    try:
        cond = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        cond = float('nan')

    kkt_ok = bool(kkt_norm <= KKT_TOL)
    success = bool(res.success)
    if not (success and feas_val <= FEAS_TOL and kkt_ok):
        raise RuntimeError(
            'solve_quadratic rejected: '
            f'success={success}, feas={feas_val:.3e}, kkt={kkt_norm:.3e}, '
            f'lambda={lam}, gamma={gam}, kappa={kappa}, message={res.message}'
        )

    return {
        'w': w.tolist(),
        'objective': objective,
        'lambda': float(lam),
        'gamma': float(gam),
        'kappa': float(kappa),
        'l1': l1_val,
        'l1_activity': l1_activity,
        'neg_mass': neg_mass,
        'kkt_norm': kkt_norm,
        'kkt_ok': kkt_ok,
        'eig_min': eig_min,
        'cond': cond,
        'start': 'uniform_then_retry' if len(starts) > 1 else 'uniform',
        'success': success,
        'message': str(res.message),
    }

# ---------------- model kinds ----------------

KINDS = ['primary', 'without_nuisance', 'clean_selector', 'no_heterogeneity', 'uncalibrated', 'unit_sum', 'simplex', 'best_segment', 'logridge', 'rawridge', 'absridge', 'histgb', 'endpoint', 'uniform', 'anchor_constant']

def _train_logridge(cells: List[Cell], lam: float) -> Dict[str, Any]:
    H, h = _H_and_h(cells)
    beta = np.linalg.solve(H + lam * np.eye(4), h)
    return {'kind': 'logridge', 'beta': beta.tolist(), 'lambda': float(lam), 'H': H.tolist(), 'h': h.tolist()}


def _train_rawridge(cells: List[Cell], lam: float) -> Dict[str, Any]:
    feas = [c for c in cells if _future_rows(c).size > 0]
    if not feas:
        raise ValueError('rawridge: no eligible training rows')
    Xs, ys, ws = [], [], []
    rw = _source_row_weights(feas)
    for w, c in zip(rw, feas):
        rows = _future_rows(c)
        A = _scenario_arrays(c, 'clean', rows)
        X = A['D'] - A['Dref'][None, :]
        Xs.append(X)
        ys.append(A['y'] - c.ref_y)
        ws.append(np.full(rows.size, w))
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    w = np.concatenate(ws, 0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std > 0, std, 1.0)
    Xs_ = X / std
    H = (Xs_ * w[:, None]).T @ Xs_
    h = (Xs_ * w[:, None]).T @ y
    beta = np.linalg.solve(H + lam * np.eye(4), h)
    return {'kind': 'rawridge', 'beta': beta.tolist(), 'std': std.tolist(), 'lambda': float(lam)}


def _abs_features(Q: np.ndarray) -> np.ndarray:
    return np.asarray(Q, dtype=float)[:, ABS_COLS]


def _train_absridge(cells: List[Cell], alpha: float) -> Dict[str, Any]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    feas = [c for c in cells if _future_rows(c).size > 0]
    if not feas:
        raise ValueError('absridge: no eligible training rows')

    Xs, ys, ws = [], [], []
    for c in feas:
        rows = _future_rows(c)
        n_future = int(rows.size)
        w_row = 1.0 / (len(feas) * (n_future + 1))
        A = _scenario_arrays(c, 'clean', rows)
        Xs.append(_abs_features(A['Q']))
        ys.append(np.asarray(A['y'], dtype=float))
        ws.append(np.full(n_future, w_row, dtype=float))
        Xs.append(_abs_features(A['Qref'][None, :]))
        ys.append(np.array([float(c.ref_y)], dtype=float))
        ws.append(np.array([w_row], dtype=float))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)
    N = float(w.size)
    wn = w / float(w.sum()) * N

    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(X)

    ridge = Ridge(alpha=float(alpha), fit_intercept=True)
    ridge.fit(Z, y, sample_weight=wn)

    mu = np.asarray(scaler.mean_, dtype=float).reshape(-1)
    sd = np.asarray(scaler.scale_, dtype=float).reshape(-1)
    sd = np.where(sd > 0, sd, 1.0)
    beta = np.asarray(ridge.coef_, dtype=float).reshape(-1)
    intercept = float(ridge.intercept_)

    return {
        'kind': 'absridge',
        'mu': mu.tolist(),
        'sd': sd.tolist(),
        'beta': beta.tolist(),
        'intercept': intercept,
        'alpha': float(alpha),
        'feature_cols': ABS_COLS.tolist(),
    }


def _train_histgb(cells: List[Cell]) -> Dict[str, Any]:
    feas = [c for c in cells if _future_rows(c).size > 0]
    if not feas:
        raise ValueError('histgb: no eligible training rows')

    Xs, ys, ws = [], [], []
    for c in feas:
        rows = _future_rows(c)
        n_future = int(rows.size)
        w_row = 1.0 / (len(feas) * (n_future + 1))
        A = _scenario_arrays(c, 'clean', rows)
        Xs.append(np.asarray(A['Q'], dtype=float))
        ys.append(np.asarray(A['y'], dtype=float))
        ws.append(np.full(n_future, w_row, dtype=float))
        Xs.append(np.asarray(A['Qref'][None, :], dtype=float))
        ys.append(np.array([float(c.ref_y)], dtype=float))
        ws.append(np.array([w_row], dtype=float))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)
    N = float(w.size)
    wn = w / float(w.sum()) * N

    m = HistGradientBoostingRegressor(
        loss='squared_error',
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=0,
        early_stopping=False,
    )
    m.fit(X, y, sample_weight=wn)
    return {'kind': 'histgb', 'model': m, 'feature_cols': list(range(32))}

# ---------------- fit dispatch ----------------

def fit(kind: str, params: Dict[str, Any], TRAIN: List[Cell]) -> Dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f'unknown kind {kind}')
    cells = [c for c in TRAIN if _future_rows(c).size > 0]
    if kind in ('primary', 'without_nuisance', 'clean_selector', 'no_heterogeneity', 'uncalibrated'):
        lam = float(params.get('lambda', 0.0))
        gam = float(params.get('gamma', 0.0))
        eta = float(params.get('eta', 0.0))
        if kind == 'without_nuisance':
            gam = 0.0
        elif kind == 'clean_selector':
            gam = 0.0
            eta = 0.0
        elif kind == 'no_heterogeneity':
            eta = 0.0
        mode = 'uncalibrated' if kind == 'uncalibrated' else 'calibrated'
        m = _fit_response(cells, lam, gam, eta, mode)
        m['kind_name'] = kind
        return m
    if kind == 'best_segment':
        seg = int(params.get('segment', 0))
        lam = float(params.get('lambda', 0.0))
        m = _fit_best_segment(cells, seg, lam)
        m['kind_name'] = kind
        return m
    if kind == 'unit_sum':
        H, h = _H_and_h(cells)
        G = _G(cells)
        gam = float(params.get('gamma', 0.0))
        res = _solve_quadratic(H, h, float(params.get('lambda', 0.0)), gam, G, KAPPA_DEFAULT)
        return {'kind': kind, 'w': res['w'], 'lambda': res['lambda'], 'gamma': res['gamma'],
                'kappa': res['kappa'], 'H': H.tolist(), 'G': G.tolist(), 'h': h.tolist(),
                'optimizer': res}
    if kind == 'simplex':
        H, h = _H_and_h(cells)
        G = _G(cells)
        gam = float(params.get('gamma', 0.0))
        res = _solve_quadratic(H, h, float(params.get('lambda', 0.0)), gam, G, KAPPA_SIMPLEX)
        return {'kind': kind, 'w': res['w'], 'lambda': res['lambda'], 'gamma': res['gamma'],
                'kappa': res['kappa'], 'H': H.tolist(), 'G': G.tolist(), 'h': h.tolist(),
                'optimizer': res}
    if kind == 'logridge':
        return _train_logridge(cells, float(params.get('lambda', 0.0)))
    if kind == 'rawridge':
        return _train_rawridge(cells, float(params.get('lambda', 0.0)))
    if kind == 'absridge':
        return _train_absridge(cells, float(params.get('alpha', 1.0)))
    if kind == 'histgb':
        return _train_histgb(cells)
    if kind in ('endpoint', 'uniform', 'anchor_constant'):
        return {'kind': kind}
    raise ValueError(f'unhandled kind {kind}')

# ---------------- predict ----------------

def _rows_for_predict(cell: Cell) -> np.ndarray:
    # all eligible rows after anchor (scoring set); anchor not scored.
    rows = _future_rows(cell, exclude_anchor=False)
    a = _anchor_index(cell)
    rows = rows[rows != a]
    return rows


def predict(model: Dict[str, Any], cell: Cell, scenario: str) -> np.ndarray:
    kind = model.get('kind_name', model.get('kind'))
    rows = _rows_for_predict(cell)
    if rows.size == 0:
        return np.zeros(0)
    A = _scenario_arrays(cell, scenario, rows)
    if kind in ('primary', 'without_nuisance', 'clean_selector', 'no_heterogeneity', 'uncalibrated', 'unit_sum', 'simplex'):
        w = np.asarray(model['w'], dtype=float)
        X = np.log(A['D'] / A['Dref'][None, :])
        return cell.ref_y * np.exp(X @ w)
    if kind == 'best_segment':
        beta = np.asarray(model['beta'], dtype=float)
        X = np.log(A['D'] / A['Dref'][None, :])
        return cell.ref_y * np.exp(X @ beta)
    if kind == 'logridge':
        beta = np.asarray(model['beta'], dtype=float)
        X = np.log(A['D'] / A['Dref'][None, :])
        return cell.ref_y * np.exp(X @ beta)
    if kind == 'endpoint':
        return cell.ref_y * A['D'].sum(axis=1) / A['Dref'].sum()
    if kind == 'uniform':
        X = np.log(A['D'] / A['Dref'][None, :])
        return cell.ref_y * np.exp(X.mean(axis=1))
    if kind == 'anchor_constant':
        return np.full(rows.size, cell.ref_y, dtype=float)
    if kind == 'rawridge':
        beta = np.asarray(model['beta'], dtype=float)
        std = np.asarray(model['std'], dtype=float)
        X = (A['D'] - A['Dref'][None, :]) / std[None, :]
        return cell.ref_y + X @ beta
    if kind == 'absridge':
        mu = np.asarray(model['mu'], dtype=float)
        sd = np.asarray(model['sd'], dtype=float)
        b = np.asarray(model['beta'], dtype=float)
        Z = (_abs_features(A['Q']) - mu[None, :]) / sd[None, :]
        Zr = (_abs_features(A['Qref'][None, :]) - mu[None, :]) / sd[None, :]
        return cell.ref_y + (Z @ b - Zr @ b)
    if kind == 'histgb':
        m = model['model']
        return cell.ref_y + (m.predict(A['Q']) - m.predict(A['Qref'][None, :])[0])
    raise ValueError(f'unknown kind {kind}')

# ---------------- selection ----------------

def _scenario_cell_maes(kind: str, params: Dict[str, Any], train_cells: List[Cell], val_cell: Cell,
                         scenarios: List[str]) -> Dict[str, float]:
    model = fit(kind, params, train_cells)
    rows = _rows_for_predict(val_cell)
    out: Dict[str, float] = {}
    for sc in scenarios:
        if rows.size == 0:
            out[sc] = float('nan')
            continue
        yp = predict(model, val_cell, sc)
        yt = val_cell.y[rows]
        if yp.size != yt.size:
            raise ValueError('predict shape mismatch')
        if not np.all(np.isfinite(yp)):
            raise ValueError('non-finite predictions')
        out[sc] = float(np.mean(np.abs(yt - yp)))
    return out


def _grid_for(kind: str) -> List[Dict[str, Any]]:
    LAMBDAS_R = [1e-6, 1e-4, 0.01, 0.1]
    GAMMAS_R = [0.0, 1.0, 10.0]
    ETAS_R = [0.0, 1.0, 10.0]
    if kind == 'primary':
        return [{'lambda': l, 'gamma': g, 'eta': e} for l in LAMBDAS_R for g in GAMMAS_R for e in ETAS_R]
    if kind == 'without_nuisance':
        return [{'lambda': l, 'gamma': 0.0, 'eta': e} for l in LAMBDAS_R for e in ETAS_R]
    if kind == 'clean_selector':
        return [{'lambda': l, 'gamma': 0.0, 'eta': 0.0} for l in LAMBDAS_R]
    if kind == 'no_heterogeneity':
        return [{'lambda': l, 'gamma': g, 'eta': 0.0} for l in LAMBDAS_R for g in GAMMAS_R]
    if kind == 'uncalibrated':
        return [{'lambda': l, 'gamma': g, 'eta': e} for l in LAMBDAS_R for g in GAMMAS_R for e in ETAS_R]
    if kind == 'best_segment':
        return [{'segment': j, 'lambda': l} for j in range(4) for l in LAMBDAS_R]
    if kind in ('unit_sum',):
        return [{'lambda': l, 'gamma': g} for l in LAMBDAS for g in GAMMAS]
    if kind == 'simplex':
        return [{'lambda': l, 'gamma': g} for l in LAMBDAS for g in GAMMAS[:3]]
    if kind in ('logridge', 'rawridge'):
        return [{'lambda': l} for l in LAMBDAS]
    if kind == 'absridge':
        return [{'alpha': a} for a in ALPHAS_ABS]
    if kind in ('histgb', 'endpoint', 'uniform', 'anchor_constant'):
        return [{}]
    raise ValueError(f'grid unknown for {kind}')


def _params_sort_key(kind, params):
    key = []
    if 'alpha' in params and params['alpha'] is not None:
        key.append(float(params['alpha']))
    else:
        key.append(0.0)
    if 'lambda' in params and params['lambda'] is not None:
        key.append(float(params['lambda']))
    else:
        key.append(-float('inf'))
    if 'gamma' in params and params['gamma'] is not None:
        key.append(float(params['gamma']))
    else:
        key.append(-float('inf'))
    if 'eta' in params and params['eta'] is not None:
        key.append(float(params['eta']))
    else:
        key.append(-float('inf'))
    if 'segment' in params and params['segment'] is not None:
        key.append(float(params['segment']))
    else:
        key.append(-1.0)
    return tuple(key)


def select(kind: str, cells: List[Cell]) -> Dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f'unknown kind {kind}')
    n_cells = len(cells)
    if n_cells < 2:
        raise ValueError('select: need at least 2 cells for GroupKFold')
    n_splits = int(min(FOLD_GROUP_K, n_cells))
    if n_splits < 2:
        raise ValueError('select: need >=2 splits')

    gk = GroupKFold(n_splits=n_splits)
    folds = list(gk.split(np.arange(n_cells), groups=np.arange(n_cells)))
    grid = _grid_for(kind)
    scenarios = ['clean', 'vp5', 'vm5']

    per_candidate: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    for params in grid:
        cell_scores = {sc: [] for sc in scenarios}
        cell_failures = 0
        for tr_idx, va_idx in folds:
            train_cells = [cells[i] for i in tr_idx]
            if not any(_future_rows(c, exclude_anchor=True).size > 0 for c in train_cells):
                continue
            try:
                model = fit(kind, params, train_cells)
            except (RuntimeError, ValueError):
                cell_failures += int(len(va_idx))
                continue
            for vi in va_idx:
                vc = cells[vi]
                rows = _rows_for_predict(vc)
                if rows.size == 0:
                    continue
                for sc in scenarios:
                    yp = predict(model, vc, sc)
                    yt = np.asarray(vc.y[rows], dtype=float)
                    if yp.size != yt.size:
                        raise ValueError(f'select: predict shape mismatch for {vc.name}/{sc}')
                    if not np.all(np.isfinite(yp)):
                        raise ValueError(f'select: non-finite predictions for {vc.name}/{sc}')
                    mae = float(np.mean(np.abs(yt - yp)))
                    cell_scores[sc].append(mae)
                    candidate_rows.append({
                        'kind': kind,
                        'kind_name': kind,
                        'training_cells': [c.name for c in train_cells],
                        'train_cells': [c.name for c in train_cells],
                        'val_cell': vc.name,
                        'scenario': sc,
                        'lambda': params.get('lambda'),
                        'gamma': params.get('gamma'),
                        'eta': params.get('eta'),
                        'alpha': params.get('alpha'),
                        'segment': params.get('segment'),
                        'mae': mae,
                        'n': int(yp.size),
                    })
        scores = {sc: (float(np.mean(cell_scores[sc])) if len(cell_scores[sc]) > 0 else float('nan')) for sc in scenarios}
        scores['robust'] = float(np.nanmax(np.array([scores['clean'], scores['vp5'], scores['vm5']], dtype=float)))
        if kind in ('clean_selector', 'absridge'):
            scores['score'] = float(scores['clean'])
        else:
            scores['score'] = float(scores['robust'])
        if cell_failures or len(cell_scores['clean']) != n_cells:
            scores['score'] = float('inf')
        n_rows = int(len(cell_scores['clean']))
        per_candidate.append({'params': dict(params), 'scores': scores, 'n_rows': n_rows, 'failures': int(cell_failures)})

    if not per_candidate:
        raise RuntimeError('select: no candidates produced')

    def sort_key(entry_):
        score = entry_['scores'].get('score', float('inf'))
        if score is None or not np.isfinite(score):
            score = float('inf')
        return (float(score), int(entry_.get('failures', 0))) + _params_sort_key(kind, entry_['params'])

    ordered = sorted(per_candidate, key=sort_key)
    chosen = ordered[0]
    sel = chosen['params']
    if not np.isfinite(chosen['scores']['score']):
        raise RuntimeError('No complete valid candidate across all source validation cells')
    return {
        'kind': kind,
        'chosen': dict(sel),
        'score': float(chosen['scores']['score']),
        'score_clean': float(chosen['scores']['clean']),
        'score_robust': float(chosen['scores']['robust']),
        'candidate_rows': candidate_rows,
        'per_candidate': per_candidate,
    }

# ---------------- misc ----------------

def gain_cancel_assert(cells: List[Cell], tol: float = 1e-12) -> None:
    w = W0
    for c in cells:
        rows = _future_rows(c)
        if rows.size == 0:
            continue
        base = _scenario_arrays(c, 'clean', rows)
        stat = _scenario_arrays(c, 'gstatp1', rows)
        xb = np.log(base['D'] / base['Dref'][None, :])
        xs = np.log(stat['D'] / stat['Dref'][None, :])
        if not np.allclose(xb, xs, atol=tol, rtol=0.0):
            raise AssertionError('stationary gain did not cancel log ratios')
        drift = _scenario_arrays(c, 'gdriftp1', rows)
        xd = np.log(drift['D'] / drift['Dref'][None, :])
        if not np.allclose(xd, xb + math.log(1.01) * np.ones_like(xb), atol=1e-9, rtol=0.0):
            raise AssertionError('drift gain should shift log ratios by +log(1.01)')
        ref = _scenario_arrays(c, 'grefp1', rows)
        xr = np.log(ref['D'] / ref['Dref'][None, :])
        if not np.allclose(xr, xb - math.log(1.01) * np.ones_like(xb), atol=1e-9, rtol=0.0):
            raise AssertionError('reference-only gain should shift log ratios by -log(1.01)')


def anchor_exact_assert(model: Dict[str, Any], cell: Cell, scenarios: Optional[List[str]] = None) -> None:
    import dataclasses as _dt
    scs = list(scenarios) if scenarios is not None else list(VOLT_TAGS)
    kind = model.get('kind_name', model.get('kind'))

    anchor_cycles = np.array([float(cell.ref_cycle)], dtype=float)
    anchor_row_index = np.array([int(cell.ref_row)], dtype=np.int64)
    anchor_y = np.array([float(cell.ref_y)], dtype=float)
    anchor_D = {tag: np.asarray(cell.Dref[tag], dtype=float).reshape(1, 4).copy() for tag in VOLT_TAGS}
    anchor_Q = {tag: np.asarray(cell.Qref[tag], dtype=float).reshape(1, N32).copy() for tag in VOLT_TAGS}
    anchor_Dref = {tag: np.asarray(cell.Dref[tag], dtype=float).reshape(4).copy() for tag in VOLT_TAGS}
    anchor_Qref = {tag: np.asarray(cell.Qref[tag], dtype=float).reshape(N32).copy() for tag in VOLT_TAGS}

    anchor_cell = _dt.replace(
        cell,
        name=cell.name + '__anchor',
        cycles=anchor_cycles,
        row_index=anchor_row_index,
        y=anchor_y,
        ref_y=float(cell.ref_y),
        ref_cycle=float(cell.ref_cycle),
        ref_row=int(cell.ref_row),
        D=anchor_D,
        Dref=anchor_Dref,
        Q=anchor_Q,
        Qref=anchor_Qref,
    )

    for sc in scs:
        pred = predict(model, anchor_cell, sc)
        if pred.size != 1:
            raise AssertionError(f'anchor_exact_assert: predict returned {pred.size} rows for {kind}/{cell.name}/{sc}, expected 1')
        p = float(pred[0])
        ref = float(cell.ref_y)
        if not np.isfinite(p):
            raise AssertionError(f'anchor_exact_assert: non-finite prediction for {kind}/{cell.name}/{sc}')
        if abs(p - ref) > 1e-9 * max(1.0, abs(ref)):
            raise AssertionError(f'anchor exactness violated for {kind}/{cell.name}/{sc}: pred={p}, ref_y={ref}')

# ---------------- audit helpers ----------------

def audit_source(cells: List[Cell]) -> Dict[str, Any]:
    out = {'cells': []}
    for c in cells:
        rows = _future_rows(c)
        a = _anchor_index(c)
        out['cells'].append({
            'name': c.name,
            'ds': c.ds,
            'n_total': int(c.y.size),
            'n_eligible': int(_eligibility_mask(c).sum()),
            'n_scored': int(rows.size),
            'ref_row': int(c.ref_row),
            'ref_cycle': float(c.ref_cycle),
            'ref_y': float(c.ref_y),
            'anchor_row': int(a),
        })
    return out


def api_reference() -> Dict[str, Any]:
    return {
        'load_cells': "load_cells(path: str, ds: str) -> List[Cell]",
        'fit': "fit(kind: str, params: dict, TRAIN: List[Cell]) -> dict model",
        'predict': "predict(model: dict, cell: Cell, scenario: str) -> np.ndarray",
        'select': "select(kind: str, cells: List[Cell]) -> dict with chosen/candidate_rows/score/score_clean/score_robust/per_candidate",
        'kinds': KINDS,
        'scenarios': VOLT_TAGS + GAIN_TAGS,
        'gain_cancel_assert': "gain_cancel_assert(cells, tol=1e-12)->None (raises AssertionError on failure)",
        'anchor_exact_assert': "anchor_exact_assert(model, cell, scenarios=None)->None",
        'audit_source': "audit_source(cells)->dict eligibility audit"
    }

def _build_cell_from_rows(name, ds, cyc, rig, y, qv2, vq, order):
    n = y.size
    if n == 0:
        raise ValueError(f'{name}: no rows')
    if qv2.shape[1] != vq.size:
        raise ValueError(f'{name}: q_curve columns {qv2.shape[1]} != v_grid {vq.size}')
    if qv2.shape[0] != n:
        raise ValueError(f'{name}: q_curve rows {qv2.shape[0]} != n {n}')
    cyc = cyc[order].astype(np.int64, copy=False)
    rig = rig[order].astype(np.int64, copy=False)
    y = y[order].astype(float, copy=False)
    Qrows = qv2[order].astype(float, copy=False)

    Qs = {}
    Ds = {}
    finite_ok = {}
    pos_ok = {}
    for tag in VOLT_TAGS:
        vq_tag = GRID32 - float(VOLT_DELTA[tag])
        Qraw = np.empty((n, N32), dtype=float)
        for i in range(n):
            try:
                Qraw[i, :] = _interp_strict(Qrows[i], vq_tag, vq, tol=1e-14)
            except ValueError:
                Qraw[i, :] = np.nan
        Qcum = Qraw - Qraw[:, :1]
        Dtag = _disjoint_segments(Qcum)
        Qs[tag] = Qcum
        Ds[tag] = Dtag
        finite_ok[tag] = np.isfinite(Qcum).all(axis=1) & np.isfinite(Dtag).all(axis=1)
        pos_ok[tag] = (Dtag > SEG_MIN).all(axis=1)

    assert np.all(np.isfinite(y) & (y > 0)), "Prepared labels must be finite and positive"
    elig = np.ones(n,dtype=bool)
    for tag in VOLT_TAGS:
        elig &= finite_ok[tag] & pos_ok[tag]
    if not elig.any():
        raise ValueError(f'{name}: no eligible rows')

    a = int(np.where(elig)[0][0])
    future = np.where(elig)[0]
    future = future[future != a]
    if future.size < 1:
        raise ValueError(f'{name}: need at least 2 eligible rows (anchor + 1 future)')

    Dref = {tag: Ds[tag][a].copy() for tag in VOLT_TAGS}
    Qref = {tag: Qs[tag][a].copy() for tag in VOLT_TAGS}
    for tag in VOLT_TAGS:
        if not np.all(Dref[tag] > SEG_MIN) or not np.isfinite(Dref[tag]).all():
            raise AssertionError(f'{name}:{tag}: non-positive/non-finite reference segments')
        if not np.isfinite(Qref[tag]).all():
            raise AssertionError(f'{name}:{tag}: non-finite reference cumulative Q')

    ref_row = int(rig[a])
    ref_cycle = float(cyc[a])
    ref_y = float(y[a])

    cell = Cell(
        name=name,
        ds=ds,
        cycles=cyc[future].copy(),
        row_index=rig[future].copy(),
        y=y[future].copy(),
        ref_y=ref_y,
        ref_cycle=ref_cycle,
        ref_row=ref_row,
        D={tag: Ds[tag][future].copy() for tag in VOLT_TAGS},
        Dref=Dref,
        Q={tag: Qs[tag][future].copy() for tag in VOLT_TAGS},
        Qref=Qref,
    )
    return cell

def _disjoint_segments(Q):
    breaks = np.array([0, 8, 16, 24, 31], dtype=int)
    Q = np.asarray(Q, dtype=float)
    seg = np.empty((Q.shape[0], 4), dtype=float)
    seg[:, 0] = Q[:, breaks[1]] - Q[:, breaks[0]]
    seg[:, 1] = Q[:, breaks[2]] - Q[:, breaks[1]]
    seg[:, 2] = Q[:, breaks[3]] - Q[:, breaks[2]]
    seg[:, 3] = Q[:, breaks[4]] - Q[:, breaks[3]]
    return seg

def _response_qp(A: np.ndarray, b: np.ndarray, kappa: float = KAPPA_DEFAULT) -> Dict[str, Any]:
    """Solve min .5*beta'A*beta s.t. b'beta=1 and ||beta||_1<=kappa. Analytic equality-unconstrained solution returned exactly if L1 satisfied. Otherwise SLSQP with equality + 16 signed L1 constraints, then KKT via NNLS on active L1 normals."""
    from scipy.optimize import nnls as _nnls

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = 4
    if A.shape != (n, n) or b.size != n:
        raise ValueError('A must be 4x4 and b length 4')
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError('A/b must be finite')
    # Positive definiteness check (up to numerical tolerance).
    eigs = np.linalg.eigvalsh(A)
    eig_min = float(np.min(eigs))
    if eig_min <= 0.0:
        raise ValueError(f'A is not positive definite: eig_min={eig_min}')
    # Feasibility of the equality: exists beta with sum|beta|<=kappa and b'beta=1 iff kappa*max|b| >= 1.
    max_abs_b = float(np.max(np.abs(b)))
    if float(kappa) * max_abs_b < 1.0 - FEAS_TOL:
        raise ValueError(f'response-QP infeasible: kappa*max|b|={float(kappa)*max_abs_b:.6f} < 1, b={b.tolist()}')

    Ainv_b = np.linalg.solve(A, b)
    denom = float(b @ Ainv_b)
    if denom <= 0.0:
        raise ValueError(f'b\'A^-1 b = {denom} not positive; cannot solve')
    beta_analytic = Ainv_b / denom
    l1_analytic = float(np.sum(np.abs(beta_analytic)))
    active_l1 = {}

    if l1_analytic <= float(kappa) + 1e-12:
        beta = beta_analytic
        sgn = np.ones(n)
        alpha = -float(denom)  # Lagrange multiplier of equality b'beta=1: stationarity A beta + alpha b = 0 gives alpha = -b'A^-1 b * b'b? no, alpha solves A beta = -alpha b, A^-1 b / (b'A^-1 b) = beta => A beta = b/(b'A^-1 b) = -alpha b -> alpha = -1/(b'A^-1 b).
        alpha = -1.0 / denom
        grad = A @ beta
        P = np.eye(n) - np.outer(b, b) / float(b @ b)
        residual = P @ (grad + alpha * b)
        kkt_norm = float(np.linalg.norm(residual))
        meta = {
            'solver': 'analytic',
            'l1': l1_analytic,
            'l1_activity': False,
            'kkt_norm': kkt_norm,
            'eig_min': eig_min,
            'cond': float(np.linalg.cond(A)),
            'active_l1_normals': [],
        }
    else:
        def fun(w):
            w = np.asarray(w, dtype=float).reshape(-1)
            return 0.5 * float(w @ A @ w)

        def jac(w):
            return A @ np.asarray(w, dtype=float).reshape(-1)

        cons = [{'type': 'eq', 'fun': (lambda w: float(b @ np.asarray(w, dtype=float).reshape(-1) - 1.0)),
                 'jac': (lambda w: b)}]
        for s in SIGNS:
            cons.append({'type': 'ineq',
                         'fun': (lambda w, s=s: float(kappa - s @ np.asarray(w, dtype=float).reshape(-1))),
                         'jac': (lambda w, s=s: -s)})

        # Closed-form feasible start: e_j / b_j at argmax|b|, feasible because kappa*max|b|>=1.
        j = int(np.argmax(np.abs(b)))
        w_init = np.zeros(n, dtype=float)
        w_init[j] = 1.0 / float(b[j])
        if float(np.max(np.abs(SIGNS @ w_init))) > float(kappa) + FEAS_TOL:
            # trivial infeasible due to numerical edge; also try scaled version
            w_init = w_init * (float(kappa) / float(np.max(np.abs(SIGNS @ w_init))))

        res = minimize(fun, w_init, jac=jac, constraints=cons, method='SLSQP',
                       options={'ftol': 1e-12, 'maxiter': 2000, 'disp': False})
        beta = np.asarray(res.x, dtype=float).reshape(-1)
        eq_resid = float(abs(b @ beta - 1.0))
        l1_val = float(np.max(np.abs(SIGNS @ beta)))
        feas = max(eq_resid, max(l1_val - float(kappa), 0.0))
        if not bool(res.success) or feas > FEAS_TOL:
            raise RuntimeError(
                'response_qp: SLSQP failed to return feasible point; '
                f'success={bool(res.success)}, feas={feas:.3e}, lambda_kappa={float(kappa)}, '
                f'eq_resid={eq_resid:.3e}, l1={l1_val:.6f}, message={getattr(res, "message", None)}'
            )

        # KKT check for SLSQP-active solution.
        grad = A @ beta
        P = np.eye(n) - np.outer(b, b) / float(b @ b)
        active_mask = (np.abs(SIGNS @ beta - float(kappa)) <= 1e-6)
        if np.any(active_mask):
            Cactive = SIGNS[active_mask, :]
        else:
            Cactive = np.zeros((0, n), dtype=float)
        if Cactive.shape[0] > 0:
            mu_raw = _nnls(P @ Cactive.T, -P @ grad)[0]
            mu = np.asarray(mu_raw, dtype=float).reshape(-1)
        else:
            mu = np.zeros(0, dtype=float)
        alpha = -float(b @ (grad + (Cactive.T @ mu if Cactive.shape[0] > 0 else np.zeros(n)))) / float(b @ b)
        resid_vec = grad + alpha * b + (Cactive.T @ mu if Cactive.shape[0] > 0 else np.zeros(n))
        kkt_norm = float(np.linalg.norm(resid_vec))
        if not (kkt_norm <= KKT_TOL):
            raise RuntimeError(f'response_qp: KKT residual {kkt_norm:.3e} > tol {KKT_TOL}; kind_params fail')

        meta = {
            'solver': 'slsqp',
            'l1': float(np.max(np.abs(SIGNS @ beta))),
            'l1_activity': True,
            'kkt_norm': kkt_norm,
            'eig_min': eig_min,
            'cond': float(np.linalg.cond(A)),
            'active_l1_normals': Cactive.tolist(),
            'message': str(res.message),
        }

    beta = np.asarray(beta, dtype=float).reshape(-1)
    response_gain = float(b @ beta)
    beta_sum = float(np.sum(beta))
    neg_mass = float(-np.minimum(beta, 0.0).sum())
    return {
        'w': beta.tolist(),
        'objective': float(0.5 * beta @ A @ beta),
        'response_gain': response_gain,
        'beta_sum': beta_sum,
        'neg_mass': neg_mass,
        'kappa': float(kappa),
        **meta,
    }


def _fit_response(cells: List[Cell], lam: float, gam: float, eta: float, mode: str) -> Dict[str, Any]:
    """mode='calibrated': response-constrained QP; mode='uncalibrated': classical log-ridge with nuisance+G+B (no b'beta=1 constraint, only ||beta||_1<=3)."""
    if mode not in ('calibrated', 'uncalibrated'):
        raise ValueError(f'_fit_response: bad mode {mode}')
    stats = _response_stats(cells, 'clean')
    b = np.asarray(stats['b'], dtype=float)
    B = np.asarray(stats['B'], dtype=float)
    C = np.asarray(stats['C'], dtype=float)
    G = np.asarray(_G(cells), dtype=float)
    rho = float(abs(math.log(2.1)))
    A = C + float(lam) * np.eye(4) + float(gam) * G + float(eta) * (rho ** 2) * B
    if mode == 'calibrated':
        qp = _response_qp(A, b, KAPPA_DEFAULT)
        return {
            'kind': 'response',
            'mode': 'calibrated',
            'w': qp['w'],
            'lambda': float(lam),
            'gamma': float(gam),
            'eta': float(eta),
            'rho': rho,
            'b': b.tolist(),
            'b_cs': np.asarray(stats['b_cs'], dtype=float).tolist(),
            'B': B.tolist(),
            'C': C.tolist(),
            'G': G.tolist(),
            'A': A.tolist(),
            'response_stats': {
                'b': b.tolist(),
                'B': B.tolist(),
                'C': C.tolist(),
                'G': G.tolist(),
                'rho': rho,
                'cell_names': list(stats['cell_names']),
                'n_cells': int(stats['n_cells']),
                'szz_min': float(stats['szz_min']),
            },
            'optimizer': qp,
        }
    # uncalibrated: classical log-ridge objective .5*beta'*(H+lam I+gam G+eta rho^2 B)*beta - h'beta, ||beta||_1<=3
    H, h = _H_and_h(cells)
    H = np.asarray(H, dtype=float)
    h = np.asarray(h, dtype=float)
    M = H + float(lam) * np.eye(4) + float(gam) * G + float(eta) * (rho ** 2) * B

    def fun(w):
        w = np.asarray(w, dtype=float).reshape(-1)
        return 0.5 * float(w @ M @ w) - float(h @ w)

    def jac(w):
        return M @ np.asarray(w, dtype=float).reshape(-1) - h

    cons = []
    for s in SIGNS:
        cons.append({'type': 'ineq',
                     'fun': (lambda w, s=s: float(KAPPA_DEFAULT - s @ np.asarray(w, dtype=float).reshape(-1))),
                     'jac': (lambda w, s=s: -s)})
    w_init = np.zeros(4, dtype=float)
    try:
        w_init = np.linalg.solve(M, h)
    except np.linalg.LinAlgError:
        w_init = np.zeros(4, dtype=float)
    if float(np.max(np.abs(SIGNS @ w_init))) > float(KAPPA_DEFAULT):
        w_init = w_init * (float(KAPPA_DEFAULT) / float(np.max(np.abs(SIGNS @ w_init))))
    res = minimize(fun, w_init, jac=jac, constraints=cons, method='SLSQP',
                   options={'ftol': 1e-12, 'maxiter': 2000, 'disp': False})
    beta = np.asarray(res.x, dtype=float).reshape(-1)
    l1_val = float(np.max(np.abs(SIGNS @ beta)))
    if not bool(res.success) or max(l1_val - float(KAPPA_DEFAULT), 0.0) > FEAS_TOL:
        raise RuntimeError(
            f'uncalibrated QP failed: success={bool(res.success)}, l1={l1_val:.6f}, '
            f'message={getattr(res, "message", None)}'
        )
    return {
        'kind': 'response',
        'mode': 'uncalibrated',
        'w': beta.tolist(),
        'lambda': float(lam),
        'gamma': float(gam),
        'eta': float(eta),
        'rho': rho,
        'b': b.tolist(),
        'B': B.tolist(),
        'C': C.tolist(),
        'G': G.tolist(),
        'H': H.tolist(),
        'h': h.tolist(),
        'M': M.tolist(),
        'response_stats': {
            'b': b.tolist(),
            'B': B.tolist(),
            'C': C.tolist(),
            'G': G.tolist(),
            'rho': rho,
            'cell_names': list(stats['cell_names']),
            'n_cells': int(stats['n_cells']),
            'szz_min': float(stats['szz_min']),
        },
        'optimizer': {
            'solver': 'slsqp_l1',
            'l1': l1_val,
            'l1_activity': bool(l1_val >= float(KAPPA_DEFAULT) - 1e-6),
            'neg_mass': float(-np.minimum(beta, 0.0).sum()),
            'message': str(res.message),
        },
    }


def _fit_best_segment(cells: List[Cell], seg: int, lam: float) -> Dict[str, Any]:
    """Single-segment coefficient beta_j = mean_c(mean_rows(x_j z)) / (mean_c(mean_rows(x_j^2)) + lambda), cell-balanced, no intercept. Uses train cells only."""
    seg = int(seg)
    if seg not in (0, 1, 2, 3):
        raise ValueError(f'best_segment: segment {seg} not in 0..3')
    nums = []
    dens = []
    used = []
    for c in cells:
        rows = _future_rows(c, exclude_anchor=True)
        if rows.size == 0:
            continue
        A = _scenario_arrays(c, 'clean', rows)
        X = np.log(A['D'] / A['Dref'][None, :])
        z = np.log(np.asarray(A['y'], dtype=float) / float(c.ref_y))
        xj = X[:, seg]
        if not (np.all(np.isfinite(xj)) and np.all(np.isfinite(z))):
            raise ValueError(f'best_segment: non-finite x or z for {c.name}')
        nums.append(float(np.mean(xj * z)))
        dens.append(float(np.mean(xj * xj)))
        used.append(c.name)
    if not used:
        raise ValueError('best_segment: no usable TRAIN cells')
    num = float(np.mean(nums))
    den = float(np.mean(dens)) + float(lam)
    beta = float(num / den) if den > 0.0 else 0.0
    return {
        'kind': 'best_segment',
        'segment': int(seg),
        'beta': [0.0 if j != seg else beta for j in range(4)],
        'beta_scalar': float(beta),
        'lambda': float(lam),
        'cells_used': used,
        'num_bar': num,
        'den_bar': float(den - float(lam)),
    }


def _response_stats(cells: List[Cell], tag: str = 'clean') -> Dict[str, Any]:
    """Per-TRAIN-cell finite-rate response stats and cell-balanced B, C. All quantities derived from TRAIN folds only."""
    bs = []
    for c in cells:
        rows = _future_rows(c, exclude_anchor=True)
        if rows.size == 0:
            continue
        A = _scenario_arrays(c, tag, rows)
        X = np.log(A['D'] / A['Dref'][None, :])
        z = np.log(np.asarray(A['y'], dtype=float) / float(c.ref_y))
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(z))):
            raise ValueError(f'{c.name}: non-finite X or z in _response_stats')
        szz = float(np.mean(z * z))
        if szz <= 1e-8:
            raise ValueError(f'{c.name}: response direction unidentifiable from near-zero source degradation')
        b_c = (X * z[:, None]).mean(axis=0) / szz
        bs.append((c.name, b_c, X, z))
    if not bs:
        raise ValueError('_response_stats: no usable TRAIN cells (all szz<=1e-8 or no rows)')
    Bc = np.stack([t[1] for t in bs], axis=0)
    b = Bc.mean(axis=0)
    D = Bc - b[None, :]
    B = (D.T @ D) / float(Bc.shape[0])
    Cacc = np.zeros((4, 4), dtype=float)
    ncell = 0
    for name, b_c, X, z in bs:
        R = X - z[:, None] * b_c[None, :]
        Cacc += (R.T @ R) / float(R.shape[0])
        ncell += 1
    C = Cacc / float(ncell)
    return {'b': b, 'B': B, 'C': C, 'b_cs': Bc, 'cell_names': [t[0] for t in bs], 'n_cells': int(len(bs)), 'szz_min': float(min(float(np.mean(t[3]*t[3])) for t in bs))}

