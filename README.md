# Cross-Protocol State-of-Health Estimation of Lithium-Ion Batteries from Reference-Anchored Partial Charging

Official Python library (`soh_partial_charge`) and complete reproducibility repository for the *Journal of Energy Storage* manuscript:

> **Cross-protocol state-of-health estimation of lithium-ion batteries from reference-anchored partial charging**  
> Le Huy Hoang (`lhhoang@dut.udn.vn`)  
> Faculty of Electrical Engineering, University of Science and Technology, The University of Danang, Vietnam

This repository provides two cleanly integrated components:
1. **Reusable Python Library (`soh_partial_charge`)** — Installable package (`pip install -e .`) for extracting the four partial-charge segment capacities on $[3.80, 4.10]\,\mathrm{V}$, fitting or loading the Stage-1 minimum-variance distortionless response filter and Stage-2 rank-one Bayesian updater on custom datasets, and evaluating BMS operational guardrails.
2. **Full Paper Reproducibility Suite (`src/`, `specs/`, `frozen_manifests/`, `revision_experiments/`)** — Self-contained scripts configured to run directly from the repository root with locked hyperparameters, summary tables, post-hoc diagnostics E1–E6, and SHA-256 verification manifests (`SHA256SUMS.txt`).

---

## 1. Installation (Library & Reproducibility Environment)

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <REPO_DIR>

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .                 # Installs the `soh_partial_charge` library in editable mode
```

Verify the library installation with the bundled example and unit tests:

```bash
python examples/quickstart_demo.py
python -m unittest discover -s tests -v
```

---

## 2. Python Library Usage (`soh_partial_charge`)

### Option A: Load Pre-Calibrated XJTU B1 Models Out of the Box

```python
from soh_partial_charge import Stage1ResponseFilter, Stage2RankOneUpdater

# Stage 1: One-anchor minimum-variance distortionless response filter
s1 = Stage1ResponseFilter.from_pretrained("XJTU_B1")

# Predict SOH directly from voltage-charge curves Q(V) covering [3.80, 4.10] V
y_hat = s1.predict_from_curves(
    voltage=v_grid,
    charge=q_curves_future,   # shape (N, M) or (M,)
    v_ref=v_grid,
    q_ref=q_curve_anchor,     # anchor cycle Q(V) at cycle 0
    y_ref=y_anchor,           # initial commissioning capacity/SOH
)

# Check BMS guardrails (e.g., flag D0/D2-D3 polarity inversion & effective gain)
diag = s1.evaluate_diagnostics(D_target, D_ref, y_target=y_true, y_ref=y_anchor)

# Stage 2: Two-landmark rank-one Bayesian relative-health tracking
s2 = Stage2RankOneUpdater.from_pretrained()
report = s2.update_weights(D_cal=D_early, D_ref=D_ref, y_cal=y_early, y_ref=y_anchor)
rel_health = s2.predict_relative(D_future, D_ref)  # q_t / q_0
```

### Option B: Fit Stage 1 or Stage 2 on Your Own Cell Cohort

```python
import numpy as np
from soh_partial_charge import (
    extract_four_segments,
    Stage1ResponseFilter,
    Stage2RankOneUpdater,
)

# Extract 4 disjoint segment capacities D_0..D_3 over [3.80, 4.10] V
D_cell1 = extract_four_segments(v_cell1, Q_cell1)  # shape (N_cycles, 4)
D_ref1, y_ref1 = D_cell1[0], float(y_cell1[0])

# Fit Stage-1 filter across multiple training cells (optionally include +/-5 mV perturbations)
s1_custom = Stage1ResponseFilter(lam=1e-6, gamma=1.0, eta=0.0)
s1_custom.fit(
    D_cells=[D_cell1[1:], D_cell2[1:]],
    y_cells=[y_cell1[1:], y_cell2[1:]],
    D_ref_cells=[D_ref1, D_ref2],
    y_ref_cells=[y_ref1, y_ref2],
)

# Fit Stage-2 Gaussian prior N(beta0, Sigma) and sigma^2
s2_custom = Stage2RankOneUpdater(lam=1e-4, t=10.0)
s2_custom.fit(
    D_cells=[D_cell1[1:], D_cell2[1:]],
    y_cells=[y_cell1[1:], y_cell2[1:]],
    D_ref_cells=[D_ref1, D_ref2],
    y_ref_cells=[y_ref1, y_ref2],
)
```

---

## 3. Repository Structure

- `soh_partial_charge/` — Installable Python library (`Stage1ResponseFilter`, `Stage2RankOneUpdater`, `extract_four_segments`, `pretrained_xjtu_b1.json`)
- `examples/` — Minimal end-to-end demo script (`quickstart_demo.py`)
- `tests/` — Unit tests covering segment extraction, exact anchor identity, stationary gain invariance, drift scaling, and Stage-2 posterior limits (`test_library.py`)
- `src/` — Core estimation & reproducibility runners:
  - `reference_models.py` — Stage-1 estimator suite (`primary`, `endpoint`, `logridge`, `anchor_constant`, ablations, nuisance operators)
  - `method_run.py` — Stage-1 source freeze, LOCO, development, and XJTU B4/B6 confirmation runner
  - `stage2_run.py` — Stage-2 two-landmark Bayesian rank-one update, source freeze, LOCO, development, and XJTU B5 confirmation runner
  - `analyze_method.py`, `analyze_stage2.py` — Paired bootstrap (20,000 resamples, seed `20261107`) and summary metric scripts
  - `prepare_confirmation_data.py`, `prepare_batch5_data.py` — Raw XJTU `.mat` -> `.npz` RPT screening and Coulomb consistency scripts
- `specs/` — Frozen pre-holdout specifications (`response_method_spec.txt`, `engine_contract.txt`, `stage2_spec.txt`)
- `revision_experiments/` — Post-hoc revision audits E1–E6 (`run_revision_experiments.py`, `run_b6_electrochem_audit.py`, and CSV/JSON outputs)
- `frozen_manifests/` — Pre-holdout configuration freezes (`stage1_config_freeze.json`, `stage2_config_freeze.json`) and Stage-1 / Stage-2 summary metric tables (`*.csv`, `*.json`)
- `data/` — Download guide, `.npz` schema, and SHA-256 digests for public NASA PCoE and XJTU Zenodo (`10.5281/zenodo.10963339`) datasets
- `SHA256SUMS.txt` — Cryptographic SHA-256 checksums for tracked files

---

## 4. Locked Confirmatory Headline Results

Both predeclared 10% superiority gates failed without post-hoc model switching:

| Stage & Cohort | Primary Model | Key Comparators (Clean MAE, pp) | Gate Outcome |
| :--- | :--- | :--- | :--- |
| **Stage 1: XJTU B4** (8 cells, R3) | `primary`: **2.522** | `logridge`: 2.443, `anchor_constant`: 3.560, `endpoint`: 3.294 | `+29.1%` skill vs `anchor_constant` |
| **Stage 1: XJTU B6** (8 cells, Sat) | `primary`: **17.791** | `anchor_constant`: **3.615**, `endpoint`: 7.927, `logridge`: 15.340 | Extrapolation inversion (`+17.696` pp bias) |
| **Stage 1: B4/B6 Mean** (16 cells) | `primary`: **10.157** | `anchor_constant`: **3.587**, `endpoint`: 5.610, `logridge`: 8.891 | **Fail** ($\ge 10\%$ reduction missed) |
| **Stage 2: XJTU B5** (8 cells, RW) | `primary`: **5.393** | `scalar_map`: **5.221**, `source_logridge`: 5.371, `isotropic_map`: 5.522, `target_secant`: 7.179 | **Fail** vs `isotropic_map` (`+2.33%` error reduction) |

Post-hoc audits E1–E6 (`revision_experiments/`) show that: (E1) no learned Stage-1 estimator beats `anchor_constant` on the B4/B6 mean; (E2/E4) `scalar_map`'s apparent Stage-2 leverage disappears under matched prior scaling (`t_eff = 165.3` / `956.7`) or extended grid tuning (`t** = 100`); (E3) B6 $D_0$ expansion reflects odd/even partial-discharge RPT preconditioning (`+19.7%` step jump), cell-surface cooling (`4.4–6.2 °C`), CV-charge growth (`+41–58%`), and within-branch positive polarization drift (`+20` to `+50 mV`); and (E5) B5 rankings invert across cells (`0/8` wins for `primary`, exact $p_{\mathrm{perm}}=0.0078$) at cell-specific post-rebound oracle indices (`9–15`, vs. non-significant `2/8–3/8` at fixed `a=10`).

---

## 5. Reproducing Manuscript Experiments

### Step 1: Verify Cryptographic Checksums

```bash
sha256sum -c SHA256SUMS.txt
```

On Windows PowerShell / Python:

```python
import hashlib, pathlib
for line in pathlib.Path("SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    expected, rel = line.split("  ", 1)
    got = hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()
    assert got == expected, (rel, got, expected)
print("All SHA-256 checksums verified.")
```

### Step 2: Prepare Datasets under `data/`

Place the prepared `.npz` files (`NASA.npz`, `XJTU_B1.npz` through `XJTU_B6.npz`) directly inside `data/` (see `data/README.md` for schema, dataset links, and `.mat` conversion commands).

### Step 3: Run Stage 1, Stage 2, and Post-Hoc Audits Directly from Repo Root

```bash
# Stage 1 (XJTU_B1 source freeze, LOCO, development, confirmation, and NASA LOCO)
python src/method_run.py --stage source --source XJTU_B1
python src/method_run.py --stage loco --source XJTU_B1
python src/method_run.py --stage development --source XJTU_B1
python src/method_run.py --stage confirmation --source XJTU_B1
python src/method_run.py --stage loco --source NASA
python src/analyze_method.py

# Stage 2 (two-landmark rank-one Bayesian update)
python src/stage2_run.py --stage source
python src/stage2_run.py --stage loco
python src/stage2_run.py --stage development
python src/stage2_run.py --stage confirmation
python src/analyze_stage2.py

# Post-hoc revision audits E1-E6
python revision_experiments/run_revision_experiments.py
python revision_experiments/run_b6_electrochem_audit.py --raw-base "<PATH_TO_RAW_XJTU_DIR>"
```

*Note on secondary fields in `src/analyze_method.py`:* In its paired-bootstrap loop (line 515), the `plus5mV`/`minus5mV` branches pass `cell_clean` rather than `cell_stress` (leaving those two secondary JSON fields `null`), and `protocol_means_pp` in `stage1_summary.json` stores raw fractions. All clean paired contrasts, CSV tables (`stage1_*.csv`), and manuscript tables are computed directly from cell metrics and are unaffected.

---

## 6. License

Code is released under the MIT License (see `LICENSE`). Underlying experimental battery datasets remain subject to the terms of their original providers (NASA Ames Prognostics Center and XJTU Zenodo DOI `10.5281/zenodo.10963339`).
