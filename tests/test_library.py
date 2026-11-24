"""Unit tests for the soh_partial_charge library and manifest integrity."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import unittest

import numpy as np

from soh_partial_charge import (
    NODE_V,
    Stage1ResponseFilter,
    Stage2RankOneUpdater,
    extract_four_segments,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestLibraryAndManifests(unittest.TestCase):
    def test_pretrained_stage1_matches_freeze(self) -> None:
        s1 = Stage1ResponseFilter.from_pretrained("XJTU_B1")
        fz = json.loads((REPO_ROOT / "frozen_manifests" / "stage1_config_freeze.json").read_text(encoding="utf-8"))
        w_expected = np.asarray(fz["model_payloads"]["primary"]["w"], dtype=float)
        self.assertTrue(np.allclose(s1.w_, w_expected, atol=1e-12))

    def test_pretrained_stage2_matches_freeze(self) -> None:
        s2 = Stage2RankOneUpdater.from_pretrained()
        fz = json.loads((REPO_ROOT / "frozen_manifests" / "stage2_config_freeze.json").read_text(encoding="utf-8"))
        st = fz["models"]["primary"]["stats"]
        self.assertTrue(np.allclose(s2.beta0_, np.asarray(st["beta0"], dtype=float), atol=1e-12))
        self.assertTrue(np.allclose(s2.Sigma_, np.asarray(st["Sigma"], dtype=float), atol=1e-12))
        self.assertAlmostEqual(float(s2.sigma2_), float(st["sigma2"]), places=12)

    def test_stage1_invariances_and_drift_exponent(self) -> None:
        s1 = Stage1ResponseFilter.from_pretrained("XJTU_B1")
        D_ref = np.array([0.45, 0.35, 0.25, 0.15])
        D_fut = np.array([[0.42, 0.37, 0.26, 0.14], [0.40, 0.39, 0.27, 0.13]])
        y_ref = 0.94

        # Exact anchor identity
        self.assertAlmostEqual(float(s1.predict(D_ref, D_ref, y_ref)), y_ref, places=12)

        # Stationary componentwise gain cancellation
        gains = np.array([1.08, 0.92, 1.03, 0.97])
        y0 = s1.predict(D_fut, D_ref, y_ref)
        yg = s1.predict(D_fut * gains, D_ref * gains, y_ref)
        self.assertTrue(np.allclose(yg, y0, atol=1e-12))

        # One-sided drift exponent (1^T beta)
        g_drift = 1.02
        yd = s1.predict(D_fut * g_drift, D_ref, y_ref)
        expected_mult = g_drift ** float(np.sum(s1.w_))
        self.assertTrue(np.allclose(yd / y0, expected_mult, atol=1e-12))

    def test_stage2_shrinkage_limits(self) -> None:
        base = Stage2RankOneUpdater.from_pretrained()
        D_ref = np.array([0.45, 0.35, 0.25, 0.15])
        D_cal = np.array([0.43, 0.36, 0.255, 0.145])
        y_ref, y_cal = 0.94, 0.91
        xa = np.log(D_cal / D_ref)
        za = math.log(y_cal / y_ref)

        # High t limit approaches beta0
        hi = Stage2RankOneUpdater(t=1e9, beta0_=base.beta0_.copy(), Sigma_=base.Sigma_.copy(), sigma2_=base.sigma2_)
        hi.update_weights(D_cal, D_ref, y_cal, y_ref)
        self.assertTrue(np.allclose(hi.beta_post_, base.beta0_, atol=1e-6))

        # Small t limit enforces exact calibration point interpolation xa^T beta* = za
        lo = Stage2RankOneUpdater(t=1e-12, beta0_=base.beta0_.copy(), Sigma_=base.Sigma_.copy(), sigma2_=base.sigma2_)
        lo.update_weights(D_cal, D_ref, y_cal, y_ref)
        self.assertAlmostEqual(float(xa @ lo.beta_post_), za, places=8)

    def test_extract_segments_and_custom_fit(self) -> None:
        v = np.sort(np.unique(np.concatenate([np.linspace(3.78, 4.12, 69), NODE_V])))
        y_all = np.array([0.96, 0.93, 0.90, 0.87, 0.84, 0.81])
        d_base = np.array([0.44, 0.34, 0.26, 0.16])
        b_true = np.array([0.35, -1.05, -0.50, 0.10])
        D_list = [d_base * np.exp(math.log(y / y_all[0]) * b_true) for y in y_all]
        q_curves = []
        for d in D_list:
            q_nodes = np.concatenate([[0.0], np.cumsum(d)])
            q = np.interp(v, NODE_V, q_nodes)
            q_curves.append(q)
        D_ext = extract_four_segments(v, np.vstack(q_curves))
        self.assertTrue(np.allclose(D_ext, np.vstack(D_list), atol=1e-12))

        s1 = Stage1ResponseFilter(lam=1e-6, gamma=0.0, eta=0.0).fit(
            [D_ext[1:], D_ext[1:]], [y_all[1:], y_all[1:]], [D_ext[0], D_ext[0]], [float(y_all[0]), float(y_all[0])]
        )
        preds = s1.predict(D_ext[1:], D_ext[0], float(y_all[0]))
        self.assertTrue(np.allclose(preds, y_all[1:], atol=1e-5))

    def test_custom_kappa_bound(self) -> None:
        rng = np.random.default_rng(7)
        y_all = np.linspace(0.96, 0.78, 12)
        z = np.log(y_all[1:] / y_all[0])
        d_base = np.array([0.44, 0.34, 0.26, 0.16])
        b_true = np.array([0.40, -1.10, -0.50, 0.10])
        D_cells = [d_base * np.exp(z[:, None] * b_true[None, :] + rng.normal(0.0, 0.015, size=(z.size, 4))) for _ in range(3)]
        y_cells = [y_all[1:] for _ in range(3)]
        D_refs = [d_base for _ in range(3)]
        y_refs = [float(y_all[0]) for _ in range(3)]

        s1_wide = Stage1ResponseFilter(lam=1e-6, gamma=0.0, eta=0.0, kappa=5.0).fit(D_cells, y_cells, D_refs, y_refs)
        wide_l1 = float(np.sum(np.abs(s1_wide.w_)))
        min_feasible_l1 = 1.0 / float(np.max(np.abs(s1_wide.b_)))
        tight_kappa = 0.5 * (min_feasible_l1 + wide_l1)
        self.assertGreater(wide_l1, tight_kappa)
        self.assertGreater(tight_kappa * float(np.max(np.abs(s1_wide.b_))), 1.0)

        s1_tight = Stage1ResponseFilter(lam=1e-6, gamma=0.0, eta=0.0, kappa=tight_kappa).fit(
            D_cells, y_cells, D_refs, y_refs
        )
        self.assertLessEqual(float(np.sum(np.abs(s1_tight.w_))), tight_kappa + 1e-6)
        self.assertAlmostEqual(float(s1_tight.b_ @ s1_tight.w_), 1.0, places=6)

    def test_freeze_code_hashes_consistent(self) -> None:
        f1 = json.loads((REPO_ROOT / "frozen_manifests" / "stage1_config_freeze.json").read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256((REPO_ROOT / "src" / "reference_models.py").read_bytes()).hexdigest(),
            f1["library_sha256"],
        )
        self.assertEqual(
            hashlib.sha256((REPO_ROOT / "src" / "method_run.py").read_bytes()).hexdigest(),
            f1["runner_sha256"],
        )
        self.assertEqual(
            hashlib.sha256((REPO_ROOT / "specs" / "response_method_spec.txt").read_bytes()).hexdigest(),
            f1["spec_sha256"],
        )
        f2 = json.loads((REPO_ROOT / "frozen_manifests" / "stage2_config_freeze.json").read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256((REPO_ROOT / "src" / "stage2_run.py").read_bytes()).hexdigest(),
            f2["code_sha256"]["stage2_run.py"],
        )
        self.assertEqual(
            hashlib.sha256((REPO_ROOT / "specs" / "stage2_spec.txt").read_bytes()).hexdigest(),
            f2["code_sha256"]["stage2_spec.txt"],
        )

    def test_oblique_projector_duality_and_online_gate(self) -> None:
        s1 = Stage1ResponseFilter.from_pretrained("XJTU_B1")
        P_perp, S_perp_inv = s1.oblique_projector_matrices(reg_floor=1e-4)
        P_par = np.outer(s1.b_, s1.w_)
        # Idempotence and source-slope annihilation
        self.assertTrue(np.allclose(P_par @ P_par, P_par, atol=1e-12))
        self.assertTrue(np.allclose(P_perp @ P_perp, P_perp, atol=1e-12))
        self.assertLess(float(np.linalg.norm(P_perp @ s1.b_)), 1e-12)
        self.assertTrue(np.allclose(S_perp_inv, S_perp_inv.T, atol=1e-10))

        # In-domain synthetic trajectory vs sign-inverted (B6-like) trajectory
        D_ref = np.array([0.45, 0.35, 0.25, 0.15])
        y_ref = 0.96
        z_in = np.linspace(-0.02, -0.15, 6)
        D_in = D_ref[None, :] * np.exp(z_in[:, None] * s1.b_[None, :])
        res_in = s1.predict_gated(D_in, D_ref, y_ref)
        self.assertFalse(bool(res_in["domain_incompatible"]))
        self.assertFalse(bool(res_in["rpt_recalibration_triggered"]))
        self.assertTrue(np.allclose(res_in["y_hat_gated"], res_in["y_hat_raw"], atol=1e-12))

        b_inv = np.array([-2.2, 0.0, 0.9, 1.0])
        D_inv = D_ref[None, :] * np.exp(z_in[:, None] * b_inv[None, :])
        res_inv = s1.predict_gated(D_inv, D_ref, y_ref)
        self.assertTrue(bool(res_inv["domain_incompatible"]))
        self.assertTrue(bool(res_inv["rpt_recalibration_triggered"]))
        self.assertTrue(bool(np.all(res_inv["polarity_flip"])))
        self.assertTrue(bool(np.all(res_inv["alarm"])))
        self.assertTrue(np.allclose(res_inv["y_hat_gated"], y_ref, atol=1e-12))

    def test_stage2_posterior_intervals_and_causal_rebound_selector(self) -> None:
        s2 = Stage2RankOneUpdater.from_pretrained()
        D_ref = np.array([0.45, 0.35, 0.25, 0.15])
        D_cal = np.array([0.43, 0.36, 0.255, 0.145])
        D_fut = np.array([[0.41, 0.37, 0.26, 0.14], [0.39, 0.38, 0.265, 0.135]])
        Sigma_post = s2.posterior_covariance(D_cal, D_ref)
        eigs = np.linalg.eigvalsh(Sigma_post)
        self.assertTrue(bool(np.all(eigs > 0.0)))

        iv = s2.predict_interval(D_fut, D_ref, D_cal=D_cal, y_cal=0.91, y_ref=0.94, level=0.95)
        self.assertTrue(bool(np.all(iv["lower"] < iv["pred"])))
        self.assertTrue(bool(np.all(iv["pred"] < iv["upper"])))
        self.assertIsNotNone(iv["standardized_innovation_zeta"])

        # Rebound followed by fade past thresholds -> causal invariance check (prior & endpoint signals)
        z_traj = np.array([0.02, 0.03, 0.01, -0.01, -0.03, -0.05, -0.07, -0.09])
        step_dir = s2.beta0_ / float(s2.beta0_ @ s2.beta0_)
        D_seq = D_ref[None, :] * np.exp(z_traj[:, None] * step_dir[None, :])
        idx_full = s2.select_post_rebound_anchor_online(
            D_seq, D_ref, fade_threshold=-0.02, consecutive=2, min_pos=3, max_pos=9, signal="prior"
        )
        self.assertEqual(idx_full, 5)
        idx_prefix = s2.select_post_rebound_anchor_online(
            D_seq[:7], D_ref, fade_threshold=-0.02, consecutive=2, min_pos=3, max_pos=9, signal="prior"
        )
        self.assertEqual(idx_prefix, idx_full)

        # Endpoint drop signal causal invariance check
        u_traj = np.array([-0.03, -0.05, -0.06, -0.06, -0.08, -0.09, -0.10, -0.11])
        D_end = D_ref[None, :] * np.exp(u_traj[:, None])
        end_full = s2.select_post_rebound_anchor_online(D_end, D_ref, signal="endpoint", max_pos=15)
        self.assertEqual(end_full, 5)
        self.assertEqual(
            s2.select_post_rebound_anchor_online(D_end[:7], D_ref, signal="endpoint", max_pos=15),
            end_full,
        )


if __name__ == "__main__":
    unittest.main()
