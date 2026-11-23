# Benchmark Datasets and Expected Layout

Raw `.mat` archives and prepared `.npz` files are **not redistributed** in this Git repository due to file size and third-party dataset hosting terms. Download the original experimental archives from their public repositories:

1. **NASA Ames Prognostics Center Battery Data Set** (Saha & Goebel, 2007)
   - 14 cells across four ambient thermal test regimes (`1.5 A` constant-current charge prior to cutoff):
     - `24 °C` (`room_24`): `B0005`, `B0006`, `B0007`, `B0018`
     - `22 °C` (`room_22`): `B0042`, `B0043`, `B0044`
     - `43 °C` (`hot_43`): `B0029`, `B0030`, `B0031`, `B0032`
     - `4 °C` (`cold_4`): `B0046`, `B0047`, `B0048`

2. **XJTU Lithium-ion Battery Dataset** (Wang et al., 2024; Zenodo DOI [`10.5281/zenodo.10963339`](https://doi.org/10.5281/zenodo.10963339))
   - 55 Lishen NCM523/graphite 18650 cells across six protocols:
     - `Batch-1` (`XJTU_B1`, 8 cells): fixed 2C (`4.0 A`) charge / `2.0 A` discharge (source training & LOCO)
     - `Batch-2` (`XJTU_B2`, 15 cells): `6.0 A` fast charge / `2.0 A` discharge (development transfer)
     - `Batch-3` (`XJTU_B3`, 8 cells): `4.0 A` charge / `R2.5` discharge (development transfer)
     - `Batch-4` (`XJTU_B4`, 8 cells): `R3` cycling with `4.0 A`/`2.0 A` RPTs (Stage-1 fresh confirmation)
     - `Batch-5` (`XJTU_B5`, 8 cells): random walk with `2.0 A`/`2.0 A` RPTs (Stage-2 untouched confirmation)
     - `Batch-6` (`XJTU_B6`, 8 cells): `Sim_satellite` cycling with `2.0 A`/`1.0 A` RPTs (Stage-1 fresh confirmation)

---

## Expected `.npz` Array Schema & SHA-256 Hashes

Every dataset loaded by `src/reference_models.py::load_cells` uses uncompressed/compressed NumPy `.npz` archives containing these arrays (`N` = number of cycles per archive):

- `cell`: `int` `(N,)` — integer cell index (`0..n_cells-1`)
- `cell_name`: `str` `(N,)` — canonical cell identifier (e.g., `2C_battery-1_c00`, `RW_battery-1_c00`)
- `cycle`: `int/float` `(N,)` — chronological cycle/record index
- `soh`: `float` `(N,)` — normalized SOH $q_{\mathrm{dis}} / \mathrm{cap}_0 \in [0.50, 1.05]$
  - **NASA:** `cap_0` is the first valid reference discharge per cell (`reference_models` Stage-1/LOCO only)
  - **XJTU B1–B4, B6:** `cap_0` is the nominal $0.2\,\mathrm{C}$ commissioning discharge capacity (`summary.discharge_capacity_Ah[0]`)
  - **XJTU B5:** `cap_0` is the first verified full-RPT $-2\,\mathrm{A}$ discharge (with a $1.5$–$2.4\,\mathrm{A}$ partner charge, skipping the initial 20–31 partial-discharge records down to `1.777 Ah`)
- `qv`: `float` `(N, 70)` — cumulative constant-current charge curve $Q(V)$ evaluated on `vgrid` (`NaN` outside contiguous monotonic voltage support)
- `vgrid`: `float` `(70,)` — uniform voltage grid `np.linspace(3.5, 4.19, 70)` ($10\,\mathrm{mV}$ spacing)
- **Additional keys in `NASA.npz`** (used by `revision_experiments/run_revision_experiments.py` E6):
  - `temp`: `float` `(N,)` — chamber temperature in $^{\circ}\mathrm{C}$ (`4`, `22`, `24`, `43`)
  - `group`: `str` `(N,)` — regime label (`cold_4`, `room_22`, `room_24`, `hot_43`)

Recorded SHA-256 digests of the 7 prepared `.npz` files used in the study:

| Archive | SHA-256 |
| :--- | :--- |
| `NASA.npz` | `76cb8ca091acc3962cf1b00657edb86def259dc664bdf46ea7e57d244beba760` |
| `XJTU_B1.npz` | `d0299e3de301f67d51d0af34ef0d0de70d17a7780e647e7fce89e9d9857b56e2` |
| `XJTU_B2.npz` | `3b51d668a332029cc573a4b2da0449cffef9a23654f5091b21c2da4f622ac8ce` |
| `XJTU_B3.npz` | `f217c542115477fc5a41f7fd839e1d1264d71509da5f8e161c72a78b98253b32` |
| `XJTU_B4.npz` | `eb2cbaf1e4efb632644ae5d5bcdc1d5cdce72b281e5193d76468d4780d6b072e` |
| `XJTU_B5.npz` | `d4a6315dd2672f617bfb96a6cb58e119a4347d8220ad7523c419476a86e2d1b6` |
| `XJTU_B6.npz` | `25f8acb387707a3e126643eae61b7c3e44b590396037fbd4ed5508fba1874028` |

---

## Expected Directory Layout

Place all seven prepared `.npz` archives directly inside `data/`:

- `data/NASA.npz`
- `data/XJTU_B1.npz`, `data/XJTU_B2.npz`, `data/XJTU_B3.npz`
- `data/XJTU_B4.npz`, `data/XJTU_B5.npz`, `data/XJTU_B6.npz`

To generate `XJTU_B4.npz`, `XJTU_B6.npz`, and `XJTU_B5.npz` from the downloaded raw XJTU Zenodo `.mat` folders (`Batch-4/`, `Batch-5/`, `Batch-6/`), run from the repository root:

```bash
python src/prepare_confirmation_data.py --raw-dir "<PATH_TO_RAW_XJTU_DIR>"
python src/prepare_batch5_data.py --raw-dir "<PATH_TO_RAW_XJTU_DIR>"
```

(Alternatively, set the environment variable `XJTU_RAW_DIR="<PATH_TO_RAW_XJTU_DIR>"` or place the raw `Batch-*` folders under `raw_data/XJTU/`.) For `NASA.npz` and `XJTU_B1.npz`–`XJTU_B3.npz`, construct archives matching the schema above from the original NASA `.mat` files and XJTU Zenodo `Batch-1`–`Batch-3` `.mat` files.
