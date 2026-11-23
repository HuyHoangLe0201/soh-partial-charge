"""Public estimators and feature utilities for reference-anchored SOH estimation."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from src import reference_models as _ref
from src import stage2_run as _s2

NODE_V = _ref.NODE_V.copy()
GRID32 = _ref.GRID32.copy()
V70 = _ref.V70.copy()
SEG_MIN = float(_ref.SEG_MIN)

ArrayLike = Union[np.ndarray, Sequence[float]]
_PRETRAINED_PATH = Path(__file__).resolve().parent / "pretrained_xjtu_b1.json"


def _load_pretrained_payload() -> Dict[str, Any]:
    return json.loads(_PRETRAINED_PATH.read_text(encoding="utf-8"))


def _interp_segment_endpoints(v: np.ndarray, q: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    if v.shape != q.shape:
        raise ValueError(f"voltage shape {v.shape} != charge shape {q.shape}")
    mask = np.isfinite(v) & np.isfinite(q)
    v = v[mask]
    q = q[mask]
    if v.size < 2:
        raise ValueError("need at least 2 finite (voltage, charge) samples")
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    q = q[order]
    if np.any(np.diff(v) <= 0.0):
        # Average duplicate voltages cleanly before strict boundary check
        vu, inv = np.unique(v, return_inverse=True)
        qu = np.zeros_like(vu, dtype=float)
        counts = np.zeros_like(vu, dtype=float)
        np.add.at(qu, inv, q)
        np.add.at(counts, inv, 1.0)
        v = vu
        q = qu / counts
    if nodes[0] < v[0] - 1e-12 or nodes[-1] > v[-1] + 1e-12:
        raise ValueError(
            f"voltage coverage [{v[0]:.4f}, {v[-1]:.4f}] V does not span requested nodes "
            f"[{nodes[0]:.4f}, {nodes[-1]:.4f}] V"
        )
    q_nodes = np.interp(nodes, v, q)
    seg = np.diff(q_nodes)
    if np.any(seg <= SEG_MIN):
        raise ValueError(f"non-positive segment capacity encountered: {seg}")
    return seg


def extract_four_segments(
    voltage: ArrayLike,
    charge: ArrayLike,
    voltage_nodes: ArrayLike = NODE_V,
) -> np.ndarray:
    """Extract the four positive partial-charge segment capacities D_0..D_3 (Ah).

    Parameters
    ----------
    voltage : array-like, shape (M,) or (N, M)
        Monotonic charging voltage samples in Volts covering [3.80, 4.10] V.
    charge : array-like, shape (M,) or (N, M)
        Cumulative charging capacity Q(V) in Ah (non-decreasing).
    voltage_nodes : array-like, shape (5,), optional
        Five segment boundary voltages. Defaults to the manuscript nodes on the
        32-point [3.80, 4.10] V grid (NODE_V).

    Returns
    -------
    np.ndarray
        Positive segment charges of shape (4,) for 1D input or (N, 4) for 2D input.
    """
    nodes = np.asarray(voltage_nodes, dtype=float).reshape(-1)
    if nodes.size != 5 or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("voltage_nodes must contain 5 strictly increasing voltages")

    q_arr = np.asarray(charge, dtype=float)
    v_arr = np.asarray(voltage, dtype=float)
    if q_arr.ndim == 1:
        return _interp_segment_endpoints(v_arr, q_arr, nodes)
    if q_arr.ndim != 2:
        raise ValueError(f"charge must be 1D or 2D, got ndim={q_arr.ndim}")

    n_rows = q_arr.shape[0]
    if v_arr.ndim == 1:
        if v_arr.size != q_arr.shape[1]:
            raise ValueError(f"voltage length {v_arr.size} != charge columns {q_arr.shape[1]}")
        return np.stack([_interp_segment_endpoints(v_arr, q_arr[i], nodes) for i in range(n_rows)], axis=0)
    if v_arr.shape != q_arr.shape:
        raise ValueError(f"2D voltage shape {v_arr.shape} != charge shape {q_arr.shape}")
    return np.stack([_interp_segment_endpoints(v_arr[i], q_arr[i], nodes) for i in range(n_rows)], axis=0)


def _as_2d_D(D: ArrayLike, name: str = "D") -> Tuple[np.ndarray, bool]:
    arr = np.asarray(D, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{name} must have shape (4,) or (N, 4), got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= SEG_MIN):
        raise ValueError(f"{name} must contain finite strictly positive (> {SEG_MIN}) segment charges")
    return arr, single


def _as_1d_Dref(D_ref: ArrayLike) -> np.ndarray:
    arr = np.asarray(D_ref, dtype=float).reshape(-1)
    if arr.size != 4 or not np.all(np.isfinite(arr)) or np.any(arr <= SEG_MIN):
        raise ValueError(f"D_ref must be shape (4,) with positive finite entries, got {arr}")
    return arr


def _wrap_cells(
    D_cells: Sequence[ArrayLike],
    y_cells: Sequence[ArrayLike],
    D_ref_cells: Sequence[ArrayLike],
    y_ref_cells: Sequence[float],
    D_vp5_cells: Optional[Sequence[ArrayLike]] = None,
    D_vm5_cells: Optional[Sequence[ArrayLike]] = None,
    D_vp5_ref: Optional[Sequence[ArrayLike]] = None,
    D_vm5_ref: Optional[Sequence[ArrayLike]] = None,
) -> List[_ref.Cell]:
    n_cells = len(D_cells)
    if not (len(y_cells) == len(D_ref_cells) == len(y_ref_cells) == n_cells) or n_cells == 0:
        raise ValueError("D_cells, y_cells, D_ref_cells, and y_ref_cells must be non-empty of equal length")
    if (D_vp5_cells is None) != (D_vm5_cells is None):
        raise ValueError("Provide both D_vp5_cells and D_vm5_cells, or neither")

    cells: List[_ref.Cell] = []
    for idx in range(n_cells):
        Dc, _ = _as_2d_D(D_cells[idx], f"D_cells[{idx}]")
        Dr = _as_1d_Dref(D_ref_cells[idx])
        yc = np.asarray(y_cells[idx], dtype=float).reshape(-1)
        yr = float(y_ref_cells[idx])
        if yc.size != Dc.shape[0] or not np.all(np.isfinite(yc)) or np.any(yc <= 0.0):
            raise ValueError(f"y_cells[{idx}] must be positive finite of length {Dc.shape[0]}")
        if not math.isfinite(yr) or yr <= 0.0:
            raise ValueError(f"y_ref_cells[{idx}] must be positive finite")

        if D_vp5_cells is not None and D_vm5_cells is not None:
            Dp, _ = _as_2d_D(D_vp5_cells[idx], f"D_vp5_cells[{idx}]")
            Dm, _ = _as_2d_D(D_vm5_cells[idx], f"D_vm5_cells[{idx}]")
            Dpr = _as_1d_Dref(D_vp5_ref[idx]) if D_vp5_ref is not None else Dr
            Dmr = _as_1d_Dref(D_vm5_ref[idx]) if D_vm5_ref is not None else Dr
            if Dp.shape != Dc.shape or Dm.shape != Dc.shape:
                raise ValueError(f"Voltage-shifted D_vp5/D_vm5 shape mismatch at cell {idx}")
        else:
            Dp, Dm, Dpr, Dmr = Dc, Dc, Dr, Dr

        D_dict = {"clean": Dc, "vp5": Dp, "vm5": Dm, "vp10": Dp, "vm10": Dm}
        Dref_dict = {"clean": Dr, "vp5": Dpr, "vm5": Dmr, "vp10": Dpr, "vm10": Dmr}
        Q_dummy = {tag: np.zeros((Dc.shape[0], _ref.N32), dtype=float) for tag in _ref.VOLT_TAGS}
        Qref_dummy = {tag: np.zeros(_ref.N32, dtype=float) for tag in _ref.VOLT_TAGS}

        cells.append(
            _ref.Cell(
                name=f"cell_{idx}",
                ds="custom",
                cycles=np.arange(1, Dc.shape[0] + 1, dtype=float),
                row_index=np.arange(1, Dc.shape[0] + 1, dtype=np.int64),
                y=yc,
                ref_y=yr,
                ref_cycle=0.0,
                ref_row=0,
                D=D_dict,
                Dref=Dref_dict,
                Q=Q_dummy,
                Qref=Qref_dummy,
            )
        )
    return cells


@dataclass
class Stage1ResponseFilter:
    """Stage-1 minimum-variance distortionless response filter (Eqs. 5-9).

    Calibrates 4-segment log-ratio weights ``w`` satisfying ``b^T w = 1`` and
    ``||w||_1 <= kappa`` while minimizing residual variance, inter-cell slope
    dispersion, and symmetric +/-5 mV voltage-shift sensitivity.
    """

    lam: float = 1e-6
    gamma: float = 1.0
    eta: float = 0.0
    kappa: float = float(_ref.KAPPA_DEFAULT)
    w_: Optional[np.ndarray] = None
    b_: Optional[np.ndarray] = None
    B_: Optional[np.ndarray] = None
    C_: Optional[np.ndarray] = None
    G_: Optional[np.ndarray] = None
    metadata_: Optional[Dict[str, Any]] = None

    @classmethod
    def from_pretrained(cls, source: str = "XJTU_B1") -> "Stage1ResponseFilter":
        """Load the exact frozen Stage-1 primary filter calibrated on XJTU B1."""
        if source != "XJTU_B1":
            raise ValueError(f"Only 'XJTU_B1' pretrained weights are bundled; got {source!r}")
        p = _load_pretrained_payload()["stage1"]["primary"]
        return cls(
            lam=float(p["lambda"]),
            gamma=float(p["gamma"]),
            eta=float(p["eta"]),
            kappa=float(p["kappa"]),
            w_=np.asarray(p["w"], dtype=float),
            b_=np.asarray(p["b"], dtype=float),
            B_=np.asarray(p["B"], dtype=float),
            C_=np.asarray(p["C"], dtype=float),
            G_=np.asarray(p["G"], dtype=float),
            metadata_={"source": "XJTU_B1_frozen"},
        )

    def fit_from_cells(self, cells: List[_ref.Cell]) -> "Stage1ResponseFilter":
        """Fit directly from a list of ``Cell`` objects loaded via ``load_cells``."""
        gam = 0.0 if (self.gamma == 0.0) else float(self.gamma)
        model = _ref._fit_response(cells, float(self.lam), gam, float(self.eta), mode="calibrated")
        self.w_ = np.asarray(model["w"], dtype=float)
        self.b_ = np.asarray(model["b"], dtype=float)
        self.B_ = np.asarray(model["B"], dtype=float)
        self.C_ = np.asarray(model["C"], dtype=float)
        self.G_ = np.asarray(model["G"], dtype=float)
        self.metadata_ = dict(model["optimizer"])
        return self

    def fit(
        self,
        D_cells: Sequence[ArrayLike],
        y_cells: Sequence[ArrayLike],
        D_ref_cells: Sequence[ArrayLike],
        y_ref_cells: Sequence[float],
        D_vp5_cells: Optional[Sequence[ArrayLike]] = None,
        D_vm5_cells: Optional[Sequence[ArrayLike]] = None,
        D_vp5_ref: Optional[Sequence[ArrayLike]] = None,
        D_vm5_ref: Optional[Sequence[ArrayLike]] = None,
    ) -> "Stage1ResponseFilter":
        """Fit Stage-1 weights from per-cell segment capacities and true SOH labels."""
        cells = _wrap_cells(
            D_cells, y_cells, D_ref_cells, y_ref_cells,
            D_vp5_cells=D_vp5_cells, D_vm5_cells=D_vm5_cells,
            D_vp5_ref=D_vp5_ref, D_vm5_ref=D_vm5_ref,
        )
        return self.fit_from_cells(cells)

    def _require_fitted(self) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Stage1ResponseFilter is not fitted; call fit(), fit_from_cells(), or from_pretrained()")
        return self.w_

    def predict(self, D: ArrayLike, D_ref: ArrayLike, y_ref: float) -> Union[float, np.ndarray]:
        """Predict SOH y_hat = y_ref * exp(log(D / D_ref) @ w)."""
        w = self._require_fitted()
        D_arr, single = _as_2d_D(D, "D")
        Dr = _as_1d_Dref(D_ref)
        yr = float(y_ref)
        if not math.isfinite(yr) or yr <= 0.0:
            raise ValueError("y_ref must be positive and finite")
        X = np.log(D_arr / Dr[None, :])
        y_hat = yr * np.exp(X @ w)
        return float(y_hat[0]) if single else y_hat

    def predict_from_curves(
        self,
        voltage: ArrayLike,
        charge: ArrayLike,
        v_ref: ArrayLike,
        q_ref: ArrayLike,
        y_ref: float,
    ) -> Union[float, np.ndarray]:
        """Extract 4 segments directly from raw Q(V) curves and predict SOH."""
        D = extract_four_segments(voltage, charge)
        D_ref = extract_four_segments(v_ref, q_ref)
        return self.predict(D, D_ref, y_ref)

    def evaluate_diagnostics(
        self,
        D_target: ArrayLike,
        D_ref_target: ArrayLike,
        y_target: Optional[ArrayLike] = None,
        y_ref: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate Stage-1 BMS guardrails and (if labels provided) effective response gain."""
        w = self._require_fitted()
        D_arr, _ = _as_2d_D(D_target, "D_target")
        Dr = _as_1d_Dref(D_ref_target)
        X = np.log(D_arr / Dr[None, :])
        polarity_flip_mask = (X[:, 0] > 0.0) & ((X[:, 2] < 0.0) | (X[:, 3] < 0.0))
        diag: Dict[str, Any] = {
            "net_weight_sum": float(np.sum(w)),
            "drift_sensitivity_abs_sum": float(abs(np.sum(w))),
            "weight_l1_norm": float(np.sum(np.abs(w))),
            "polarity_flip_fraction": float(np.mean(polarity_flip_mask)),
            "mean_log_feature_shift": X.mean(axis=0).tolist(),
        }
        if y_target is not None and y_ref is not None:
            yt = np.asarray(y_target, dtype=float).reshape(-1)
            yr = float(y_ref)
            z = np.log(yt / yr)
            szz = float(np.mean(z * z))
            if szz > 1e-8:
                b_target = (X * z[:, None]).mean(axis=0) / szz
                gamma_eff = float(b_target @ w)
                diag["target_slope_b"] = b_target.tolist()
                diag["effective_response_gain_gamma_eff"] = gamma_eff
                diag["sign_inverted"] = bool(gamma_eff < 0.0)
        return diag


@dataclass
class Stage2RankOneUpdater:
    """Stage-2 two-landmark rank-one Bayesian updater (Eqs. 13-18).

    Adapts a source prior N(beta0, Sigma) with effective observation variance
    nu = t * sigma^2 from one additional calibration landmark (D_cal, y_cal).
    """

    lam: float = 1e-4
    t: float = 10.0
    beta0_: Optional[np.ndarray] = None
    Sigma_: Optional[np.ndarray] = None
    sigma2_: Optional[float] = None
    beta_post_: Optional[np.ndarray] = None
    last_update_diagnostics_: Optional[Dict[str, Any]] = None

    @classmethod
    def from_pretrained(cls) -> "Stage2RankOneUpdater":
        """Load the exact frozen Stage-2 source prior calibrated on XJTU B1."""
        p = _load_pretrained_payload()["stage2"]["primary"]
        return cls(
            lam=float(p["lambda"]),
            t=float(p["t"]),
            beta0_=np.asarray(p["beta0"], dtype=float),
            Sigma_=np.asarray(p["Sigma"], dtype=float),
            sigma2_=float(p["sigma2"]),
        )

    def fit_from_cells(self, cells: List[_ref.Cell]) -> "Stage2RankOneUpdater":
        """Estimate Gaussian prior (beta0, Sigma, sigma^2) from source ``Cell`` objects."""
        stats = _s2.fit_stats(cells, float(self.lam))
        self.beta0_ = np.asarray(stats["beta0"], dtype=float)
        self.Sigma_ = np.asarray(stats["Sigma"], dtype=float)
        self.sigma2_ = float(stats["sigma2"])
        self.beta_post_ = None
        self.last_update_diagnostics_ = None
        return self

    def fit(
        self,
        D_cells: Sequence[ArrayLike],
        y_cells: Sequence[ArrayLike],
        D_ref_cells: Sequence[ArrayLike],
        y_ref_cells: Sequence[float],
    ) -> "Stage2RankOneUpdater":
        """Estimate source prior from lists of per-cell segment capacities and SOH arrays."""
        cells = _wrap_cells(D_cells, y_cells, D_ref_cells, y_ref_cells)
        return self.fit_from_cells(cells)

    def _require_prior(self) -> Tuple[np.ndarray, np.ndarray, float]:
        if self.beta0_ is None or self.Sigma_ is None or self.sigma2_ is None:
            raise RuntimeError("Stage2RankOneUpdater prior is not fitted; call fit() or from_pretrained()")
        return self.beta0_, self.Sigma_, float(self.sigma2_)

    def update_weights(
        self,
        D_cal: ArrayLike,
        D_ref: ArrayLike,
        y_cal: float,
        y_ref: float,
    ) -> Dict[str, Any]:
        """Compute the closed-form rank-one posterior beta* from landmark (D_cal, y_cal)."""
        beta0, Sigma, sigma2 = self._require_prior()
        Dc = _as_1d_Dref(D_cal)
        Dr = _as_1d_Dref(D_ref)
        yc = float(y_cal)
        yr = float(y_ref)
        if not (math.isfinite(yc) and yc > 0.0 and math.isfinite(yr) and yr > 0.0):
            raise ValueError("y_cal and y_ref must be positive finite numbers")

        xa = np.log(Dc / Dr)
        za = math.log(yc / yr)
        nu = float(self.t) * sigma2
        q = float(xa @ Sigma @ xa)
        denom = nu + q
        if denom <= 0.0 or not math.isfinite(denom):
            raise FloatingPointError("Non-positive MAP denominator")

        err_a = za - float(xa @ beta0)
        delta_beta = (Sigma @ xa) * (err_a / denom)
        beta_star = beta0 + delta_beta
        n_eff = float(q / nu) if nu > 0.0 else float("inf")
        maha_norm = float((math.sqrt(max(q, 0.0)) * abs(err_a)) / denom)

        self.beta_post_ = beta_star
        self.last_update_diagnostics_ = {
            "beta_post": beta_star.tolist(),
            "prior_leverage_q": q,
            "noise_variance_nu": nu,
            "n_eff_leverage_ratio": n_eff,
            "prior_residual_za_minus_xa_beta0": float(err_a),
            "step_l2_norm": float(np.linalg.norm(delta_beta)),
            "step_mahalanobis_norm": maha_norm,
            "low_leverage_shrinkage_warning": bool(n_eff < 0.1),
        }
        return dict(self.last_update_diagnostics_)

    def predict_relative(
        self,
        D: ArrayLike,
        D_ref: ArrayLike,
        D_cal: Optional[ArrayLike] = None,
        y_cal: Optional[float] = None,
        y_ref: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Predict relative health q_t / q_0 = exp(x_t^T beta*).

        If ``D_cal``, ``y_cal``, and ``y_ref`` are provided, performs the rank-one
        posterior update first; otherwise uses stored posterior weights (or prior
        mean ``beta0`` if ``update_weights`` has not yet been called).
        """
        beta0, _, _ = self._require_prior()
        if D_cal is not None:
            if y_cal is None or y_ref is None:
                raise ValueError("y_cal and y_ref are required when D_cal is supplied")
            self.update_weights(D_cal, D_ref, y_cal, y_ref)

        beta = self.beta_post_ if self.beta_post_ is not None else beta0
        D_arr, single = _as_2d_D(D, "D")
        Dr = _as_1d_Dref(D_ref)
        X = np.log(D_arr / Dr[None, :])
        rel = np.exp(X @ beta)
        return float(rel[0]) if single else rel

    def predict_soh(
        self,
        D: ArrayLike,
        D_ref: ArrayLike,
        y_ref: float,
        D_cal: Optional[ArrayLike] = None,
        y_cal: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Predict absolute SOH y_hat = y_ref * (q_hat / q_0)."""
        yr = float(y_ref)
        if not math.isfinite(yr) or yr <= 0.0:
            raise ValueError("y_ref must be positive and finite")
        rel = self.predict_relative(D, D_ref, D_cal=D_cal, y_cal=y_cal, y_ref=yr if D_cal is not None else None)
        return float(yr * rel) if isinstance(rel, float) else yr * rel


__all__ = [
    "NODE_V",
    "GRID32",
    "V70",
    "SEG_MIN",
    "extract_four_segments",
    "Stage1ResponseFilter",
    "Stage2RankOneUpdater",
    "Cell",
    "load_cells",
]

Cell = _ref.Cell
load_cells = _ref.load_cells
