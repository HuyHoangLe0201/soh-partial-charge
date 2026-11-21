# Benchmark Datasets and Expected Layout

Raw `.mat` archives and prepared `.npz` files are **not redistributed** in this Git repository due to file size and third-party dataset hosting terms. Download the original experimental archives from their public repositories:

1. **NASA Ames Prognostics Center Battery Data Set** (Saha & Goebel, 2007)
   - 14 cells across four ambient temperature regimes:
     - `24 °C` (`room_24`): `B0005`, `B0006`, `B0007`, `B0018`
     - `22 °C` (`room_22`): `B0042`, `B0043`, `B0044`
     - `43 °C` (`hot_43`): `B0029`, `B0030`, `B0031`, `B0032`
     - `4 °C` (`cold_4`): `B0046`, `B0047`, `B0048`

2. **XJTU Lithium-ion Battery Dataset** (Wang et al., 2024; Zenodo DOI [`10.5281/zenodo.10963339`](https://doi.org/10.5281/zenodo.10963339))
   - 55 NCM523/graphite 18650 cells across six protocols:
     - `Batch-1` (`XJTU_B1`, 8 cells): fixed 2C (`4.0 A`) charge / `2.0 A` discharge (source training & LOCO)
     - `Batch-2` (`XJTU_B2`, 15 cells): `6.0 A` fast charge / `2.0 A` discharge (development transfer)
     - `Batch-3` (`XJTU_B3`, 8 cells): `4.0 A` charge / `R2.5` discharge (development transfer)
     - `Batch-4` (`XJTU_B4`, 8 cells): `R3` cycling with `4.0 A`/`2.0 A` RPTs (Stage-1 fresh confirmation)
     - `Batch-5` (`XJTU_B5`, 8 cells): random walk with `2.0 A`/`2.0 A` RPTs (Stage-2 untouched confirmation)
     - `Batch-6` (`XJTU_B6`, 8 cells): `Sim_satellite` cycling with `2.0 A`/`1.0 A` RPTs (Stage-1 fresh confirmation)

## Expected Directory Layout for Frozen Runners

To preserve exact byte-for-byte SHA-256 parity with `frozen_manifests/stage1_config_freeze.json` and `frozen_manifests/stage2_config_freeze.json`, `src/*.py` are unmodified from the executed confirmation run. When executing from a sandbox root `<WORKDIR>`:

- `<WORKDIR>/work/paper_research_expansion_20261107/data/`:
  - `NASA.npz`
  - `XJTU_B1.npz` (SHA-256 recorded in `stage1_config_freeze.json` & `stage2_config_freeze.json`: `d0299e3de301f67d51d0af34ef0d0de70d17a7780e647e7fce89e9d9857b56e2`)
  - `XJTU_B2.npz`, `XJTU_B3.npz` (for Stage-2; Stage-1 `method_run.py` looks in `data_new` below for `XJTU_B2..B6`)
- `<WORKDIR>/work/paper_method_research_20261107/data/`:
  - `XJTU_B2.npz`, `XJTU_B3.npz`, `XJTU_B4.npz`, `XJTU_B5.npz`, `XJTU_B6.npz`

Use `src/prepare_confirmation_data.py` (for `XJTU_B4` and `XJTU_B6`) and `src/prepare_batch5_data.py` (for `XJTU_B5`), setting the raw XJTU `Batch-4`, `Batch-5`, and `Batch-6` paths on a local working copy if your raw download location differs from the author's local path.
