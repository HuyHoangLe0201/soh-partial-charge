# Reference-Anchored Partial-Charge Battery Health Estimation

Reproducibility package for the *Journal of Energy Storage* manuscript:

> **Reference-anchored partial-charge battery health estimation: response calibration and two-measurement adaptation**  
> Le Huy Hoang (`lhhoang@dut.udn.vn`)  
> Faculty of Electrical Engineering, University of Science and Technology, The University of Danang, Vietnam

This repository preserves the **exact byte-identical source scripts**, **frozen pre-holdout specifications**, **configuration freeze manifests**, **summary metric tables**, and **post-hoc diagnostic scripts/outputs (`revision_experiments/`)** reported in the manuscript. No external pre-registration or separate code DOI is claimed; verification proceeds against the recorded SHA-256 manifests in `SHA256SUMS.txt`.

---

## Repository Layout

- `src/` — Exact source code verified against `frozen_manifests/`:
  - `reference_models.py` — Stage-1 estimator library (`primary`, `endpoint`, `logridge`, `anchor_constant`, ablations, nuisance operators)
  - `method_run.py` — Stage-1 source freeze, LOCO, development, and XJTU B4/B6 confirmation runner
  - `stage2_run.py` — Stage-2 two-measurement Bayesian rank-one update, source freeze, LOCO, development, and XJTU B5 confirmation runner
  - `analyze_method.py`, `analyze_stage2.py` — Paired bootstrap (20,000 resamples, seed `20261107`) and summary metric scripts
  - `prepare_confirmation_data.py`, `prepare_batch5_data.py` — Raw XJTU `.mat` -> `.npz` RPT screening and Coulomb consistency scripts
- `specs/` — Frozen pre-holdout method specifications (`response_method_spec.txt`, `engine_contract.txt`, `stage2_spec.txt`)
- `revision_experiments/` — Post-hoc revision audits E1–E6 (`run_revision_experiments.py`, `run_b6_electrochem_audit.py`, and all generated CSV/JSON outputs)
- `frozen_manifests/` — Pre-holdout `config_freeze.json` files (`stage1_config_freeze.json`, `stage2_config_freeze.json`), `MANIFEST.json`, and locked Stage-1 / Stage-2 analysis tables (`*.csv`, `*.json`)
- `data/` — Download and `.npz` layout instructions for the public NASA PCoE and XJTU Zenodo (`10.5281/zenodo.10963339`) datasets
- `SHA256SUMS.txt` — SHA-256 checksums for every file in `src/`, `specs/`, `revision_experiments/`, and `frozen_manifests/`

---

## Locked Confirmatory Headline Results (Unaltered Primary Estimators)

Both predeclared 10% superiority gates failed without post-hoc model switching:

| Stage & Cohort | Primary Model | Key Comparators (Clean MAE, pp) | Gate Outcome |
| :--- | :--- | :--- | :--- |
| **Stage 1: XJTU B4** (8 cells, R3) | `primary`: **2.522** | `logridge`: 2.443, `anchor_constant`: 3.560, `endpoint`: 3.294 | `+29.1%` skill vs `anchor_constant` |
| **Stage 1: XJTU B6** (8 cells, Sat) | `primary`: **17.791** | `anchor_constant`: **3.615**, `endpoint`: 7.927, `logridge`: 15.340 | Extrapolation inversion (`+17.696` pp bias) |
| **Stage 1: B4/B6 Mean** (16 cells) | `primary`: **10.157** | `anchor_constant`: **3.587**, `endpoint`: 5.610, `logridge`: 8.891 | **Fail** ($>10\%$ reduction missed) |
| **Stage 2: XJTU B5** (8 cells, RW) | `primary`: **5.393** | `scalar_map`: **5.221**, `source_logridge`: 5.371, `isotropic_map`: 5.522, `target_secant`: 7.179 | **Fail** vs `isotropic_map` (`-2.33%`) |

Post-hoc audits E1–E6 (`revision_experiments/`) demonstrate that: (E1) no learned Stage-1 estimator beats `anchor_constant` on the B4/B6 mean; (E2/E4) `scalar_map`'s apparent Stage-2 leverage disappears under matched prior scaling (`t_eff = 165.3` / `956.7`) or extended grid tuning (`t** = 100`); (E3) B6 $D_0$ inversion reflects alternating partial-discharge RPT preconditioning, `4.4–6.2 °C` cooling, and fivefold $R_{\mathrm{ohmic}}$ growth; and (E5) B5 rankings invert (`0/8` wins for `primary`, exact $p_{\mathrm{perm}}=0.0078$) once anchored past initial capacity rebound.

---

## Verification & Reproduction

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Verify SHA-256 Manifests Against Frozen Confirmation Runs

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

### 3. Re-running Stage 1, Stage 2, and Post-Hoc Audits

See `data/README.md` for dataset downloads and the directory layout expected by the unmodified frozen runners (`<WORKDIR>/work/paper_method_research_20261107/`). From `<WORKDIR>/work/paper_method_research_20261107/`:

```bash
# Stage 1 (XJTU_B1 source freeze, LOCO, development, confirmation, and NASA LOCO sanity check)
python method_run.py --root <WORKDIR> --stage source --source XJTU_B1
python method_run.py --root <WORKDIR> --stage loco --source XJTU_B1
python method_run.py --root <WORKDIR> --stage development --source XJTU_B1
python method_run.py --root <WORKDIR> --stage confirmation --source XJTU_B1
python method_run.py --root <WORKDIR> --stage loco --source NASA
python analyze_method.py

# Stage 2 (two-measurement rank-one covariance update)
python stage2_run.py --stage source
python stage2_run.py --stage loco
python stage2_run.py --stage development
python stage2_run.py --stage confirmation
python analyze_stage2.py

# Post-hoc revision audits E1-E6 (adjust ROOT / RAW_BASE paths if running on another machine)
python revision_experiments/run_revision_experiments.py
python revision_experiments/run_b6_electrochem_audit.py
```

---

## License

Code is released under the MIT License (see `LICENSE`). Underlying experimental battery datasets remain subject to the terms of their original providers (NASA Ames Prognostics Center and XJTU Zenodo DOI `10.5281/zenodo.10963339`).
