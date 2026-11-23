#!/usr/bin/env python3
"""Minimal end-to-end usage example for the ``soh_partial_charge`` library.

Demonstrates:
1. Loading the pre-calibrated Stage-1 distortionless filter and Stage-2 rank-one
   Bayesian updater trained on XJTU B1.
2. Extracting the 4 partial-charge segment capacities directly from voltage-charge
   curves Q(V) on [3.80, 4.10] V.
3. Fitting both Stage-1 and Stage-2 models from scratch on custom NumPy arrays.
4. Verifying exact anchor identity, stationary gain invariance, and rank-one updating.
"""
from __future__ import annotations

import math
import numpy as np

from soh_partial_charge import (
    NODE_V,
    Stage1ResponseFilter,
    Stage2RankOneUpdater,
    extract_four_segments,
)


def _synthetic_cell(rng: np.random.Generator, n_cycles: int = 12, rate_shift: float = 1.0):
    """Generate synthetic partial-charge curves with XJTU-B1-like mid-voltage kinetics."""
    v = np.linspace(3.75, 4.15, 81)
    y = np.linspace(0.95, 0.80, n_cycles)
    z = np.log(y / y[0])
    # Realistic base segment capacities (Ah) on the 5 nodes
    d_base = np.array([0.45, 0.35, 0.25, 0.15])
    # Finite-rate aging response direction similar to XJTU B1
    b_true = np.array([0.40, -1.12, -0.54, 0.07]) * rate_shift

    q_curves = np.zeros((n_cycles, v.size), dtype=float)
    for k in range(n_cycles):
        noise = rng.normal(0.0, 0.002, size=4)
        d_k = d_base * np.exp(z[k] * b_true + noise)
        # Piecewise-linear cumulative Q(V) interpolation passing cleanly through NODE_V
        q_nodes = np.concatenate([[0.0], np.cumsum(d_k)])
        q_curves[k] = np.interp(v, NODE_V, q_nodes, left=0.0, right=q_nodes[-1])
        # Extend linearly outside [3.80, 4.10]
        lo = v < NODE_V[0]
        hi = v > NODE_V[-1]
        q_curves[k, lo] = -0.1 * (NODE_V[0] - v[lo])
        q_curves[k, hi] = q_nodes[-1] + 0.1 * (v[hi] - NODE_V[-1])
    return v, q_curves, y


def main() -> None:
    rng = np.random.default_rng(42)

    # 1. Pre-trained Stage-1 and Stage-2 inference from raw Q(V) curves
    s1_pretrained = Stage1ResponseFilter.from_pretrained("XJTU_B1")
    v_t, q_t, y_t = _synthetic_cell(rng, n_cycles=10)
    y_pred_pre = s1_pretrained.predict_from_curves(
        voltage=v_t,
        charge=q_t[1:],
        v_ref=v_t,
        q_ref=q_t[0],
        y_ref=float(y_t[0]),
    )
    print(f"[Stage-1 Pretrained] MAE on synthetic matched cell: {np.mean(np.abs(y_pred_pre - y_t[1:]))*100:.2f} pp")

    D_t = extract_four_segments(v_t, q_t)
    D_ref = D_t[0]
    s2_pretrained = Stage2RankOneUpdater.from_pretrained()
    rel_pre = s2_pretrained.predict_relative(D_t[5:], D_ref, D_cal=D_t[4], y_cal=float(y_t[4]), y_ref=float(y_t[0]))
    print(f"[Stage-2 Pretrained] rel MAE: {np.mean(np.abs(rel_pre - y_t[5:]/y_t[0]))*100:.2f} pp")

    # 2. Stationary gain invariance, exact anchor, and BMS diagnostic checks
    y_anchor_hat = s1_pretrained.predict(D_ref, D_ref, float(y_t[0]))
    gain_vec = np.array([1.05, 0.96, 1.02, 0.99])
    y_gain_hat = s1_pretrained.predict(D_t[1:] * gain_vec, D_ref * gain_vec, float(y_t[0]))
    assert math.isclose(y_anchor_hat, y_t[0], abs_tol=1e-12)
    assert np.allclose(y_gain_hat, y_pred_pre, atol=1e-12)
    diag = s1_pretrained.evaluate_diagnostics(D_t[1:], D_ref, y_target=y_t[1:], y_ref=float(y_t[0]))
    print(f"[Checks] Anchor/gain checks OK; gamma_eff={diag['effective_response_gain_gamma_eff']:.3f}")

    # 3. Custom training from scratch + Stage-2 rank-one adaptation
    D_train, y_train, D_refs, y_refs = [], [], [], []
    for _ in range(4):
        vc, qc, yc = _synthetic_cell(rng, n_cycles=15, rate_shift=rng.uniform(0.95, 1.05))
        Dc = extract_four_segments(vc, qc)
        D_train.append(Dc[1:])
        y_train.append(yc[1:])
        D_refs.append(Dc[0])
        y_refs.append(float(yc[0]))

    s1_custom = Stage1ResponseFilter(lam=1e-6, gamma=0.0, eta=0.0).fit(D_train, y_train, D_refs, y_refs)
    print(f"[Stage-1 Custom Fit] Weights w = {np.round(s1_custom.w_, 4)}, sum = {s1_custom.w_.sum():.4f}")

    s2_custom = Stage2RankOneUpdater(lam=1e-4, t=10.0).fit(D_train, y_train, D_refs, y_refs)
    report = s2_custom.update_weights(D_cal=D_t[4], D_ref=D_ref, y_cal=float(y_t[4]), y_ref=float(y_t[0]))
    rel_pred = s2_custom.predict_relative(D_t[5:], D_ref)
    rel_true = y_t[5:] / y_t[0]
    print(f"[Stage-2 Custom Fit] N_eff = {report['n_eff_leverage_ratio']:.4f}, rel MAE = {np.mean(np.abs(rel_pred - rel_true))*100:.2f} pp")


if __name__ == "__main__":
    main()
