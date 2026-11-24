# Supplementary Granular Archive Tables

Companion detailed records for *Cross-protocol state-of-health estimation of lithium-ion batteries from reference-anchored partial charging: Transferability limits and online diagnostics* (Le Huy Hoang). All tables below are generated directly from `frozen_manifests/` and `revision_experiments/` JSON/CSV outputs.

---

## S1. Cell-Level Confirmation Results (24 Held-Out XJTU Cells)

### XJTU B4 Per-Cell Clean Stage-1 MAE (pp)

| Cell ID | `primary` | `logridge` | `without_nuisance` | `endpoint` | `histgb` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `R3_battery-1_c00` | 2.287 | 2.381 | 3.021 | 2.254 | 3.780 |
| `R3_battery-2_c01` | 2.515 | 2.535 | 3.097 | 3.665 | 3.880 |
| `R3_battery-3_c02` | 2.061 | 2.280 | 2.828 | 2.660 | 4.356 |
| `R3_battery-4_c03` | 2.267 | 2.276 | 3.023 | 2.691 | 4.217 |
| `R3_battery-5_c04` | 2.267 | 2.243 | 2.960 | 2.471 | 4.004 |
| `R3_battery-6_c05` | 3.359 | 3.148 | 3.606 | 3.052 | 4.027 |
| `R3_battery-7_c06` | 2.603 | 2.085 | 2.630 | 5.257 | 3.969 |
| `R3_battery-8_c07` | 2.820 | 2.596 | 2.928 | 4.297 | 4.256 |
| **Mean** | **2.522** | **2.443** | **3.012** | **3.294** | **4.061** |

### XJTU B6 Per-Cell Clean Stage-1 MAE (pp)

| Cell ID | `primary` | `logridge` | `without_nuisance` | `endpoint` | `histgb` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Sim_satellite_battery-1_c00` | 18.407 | 15.910 | 10.436 | 7.335 | 3.402 |
| `Sim_satellite_battery-2_c01` | 18.156 | 15.706 | 10.310 | 7.936 | 3.234 |
| `Sim_satellite_battery-3_c02` | 17.335 | 14.985 | 9.755 | 7.763 | 3.737 |
| `Sim_satellite_battery-4_c03` | 18.180 | 15.713 | 10.417 | 7.630 | 3.672 |
| `Sim_satellite_battery-5_c04` | 17.843 | 15.430 | 10.129 | 9.033 | 4.575 |
| `Sim_satellite_battery-6_c05` | 18.895 | 16.348 | 10.913 | 7.881 | 3.561 |
| `Sim_satellite_battery-7_c06` | 16.473 | 14.022 | 8.681 | 7.992 | 3.462 |
| `Sim_satellite_battery-8_c07` | 17.042 | 14.604 | 9.299 | 7.849 | 3.276 |
| **Mean** | **17.791** | **15.340** | **9.992** | **7.927** | **3.615** |

### XJTU B5 Untouched Confirmation Per-Cell Clean Stage-2 MAE (pp, Calibrated at Index `a=4`)

| Cell ID | Scored Profiles ($N_c$) | `primary` | `isotropic_map` | `scalar_map` | `source_logridge` | `target_secant` | `carry_last` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RW_battery-1_c00` | 12 | 5.363 | 5.464 | 5.112 | 5.349 | 7.500 | 5.880 |
| `RW_battery-2_c01` | 22 | 4.353 | 4.475 | 4.319 | 4.324 | 6.185 | 5.157 |
| `RW_battery-3_c02` | 24 | 4.971 | 5.172 | 4.684 | 4.935 | 6.868 | 5.860 |
| `RW_battery-4_c03` | 16 | 6.165 | 6.183 | 6.172 | 6.162 | 7.826 | 5.949 |
| `RW_battery-5_c04` | 19 | 5.746 | 5.943 | 5.653 | 5.712 | 7.167 | 6.174 |
| `RW_battery-6_c05` | 14 | 5.383 | 5.497 | 5.048 | 5.363 | 6.915 | 5.952 |
| `RW_battery-7_c06` | 11 | 5.064 | 5.180 | 4.876 | 5.060 | 6.999 | 6.056 |
| `RW_battery-8_c07` | 13 | 6.098 | 6.261 | 5.907 | 6.062 | 7.974 | 7.104 |
| **Mean** | **131** | **5.393** | **5.522** | **5.221** | **5.371** | **7.179** | **6.016** |

---

## S2. Symmetric Digital Voltage-Offset Spans (`{-10, -5, +5, +10}` mV)

### Symmetric Digital Voltage-Offset Spans Across `{-10, -5, +5, +10}` mV (Clean MAE vs. Span, pp)

| Stage | Cohort | Model | Clean MAE (pp) | Voltage-Stress Span (pp) |
| :--- | :--- | :--- | ---: | ---: |
| Stage 1 | `NASA` | `primary` | 7.029 | 6.913–7.112 |
| Stage 1 | `NASA` | `logridge` | 7.360 | 6.849–7.801 |
| Stage 1 | `NASA` | `histgb` | 5.761 | 5.026–6.246 |
| Stage 1 | `NASA` | `endpoint` | 10.873 | 10.866–10.952 |
| Stage 1 | `NASA` | `without_nuisance` | 6.630 | 6.486–6.742 |
| Stage 1 | `XJTU_B1` | `primary` | 1.306 | 1.314–1.851 |
| Stage 1 | `XJTU_B1` | `logridge` | 1.264 | 1.310–1.563 |
| Stage 1 | `XJTU_B1` | `histgb` | 1.382 | 1.270–5.417 |
| Stage 1 | `XJTU_B1` | `endpoint` | 8.656 | 8.585–8.773 |
| Stage 1 | `XJTU_B1` | `without_nuisance` | 1.323 | 1.283–1.928 |
| Stage 1 | `XJTU_B2` | `primary` | 14.398 | 9.931–21.989 |
| Stage 1 | `XJTU_B2` | `logridge` | 12.814 | 8.323–20.748 |
| Stage 1 | `XJTU_B2` | `histgb` | 2.632 | 2.659–2.768 |
| Stage 1 | `XJTU_B2` | `endpoint` | 19.636 | 18.905–20.348 |
| Stage 1 | `XJTU_B2` | `without_nuisance` | 14.017 | 9.735–21.962 |
| Stage 1 | `XJTU_B3` | `primary` | 2.001 | 2.007–2.397 |
| Stage 1 | `XJTU_B3` | `logridge` | 1.996 | 2.003–2.249 |
| Stage 1 | `XJTU_B3` | `histgb` | 6.796 | 4.726–10.153 |
| Stage 1 | `XJTU_B3` | `endpoint` | 10.193 | 9.464–10.986 |
| Stage 1 | `XJTU_B3` | `without_nuisance` | 2.588 | 2.507–2.891 |
| Stage 1 | `XJTU_B4` | `primary` | 2.522 | 2.153–3.041 |
| Stage 1 | `XJTU_B4` | `logridge` | 2.443 | 2.131–2.855 |
| Stage 1 | `XJTU_B4` | `histgb` | 4.061 | 3.772–3.994 |
| Stage 1 | `XJTU_B4` | `endpoint` | 3.294 | 3.019–3.721 |
| Stage 1 | `XJTU_B4` | `without_nuisance` | 3.012 | 2.753–3.267 |
| Stage 1 | `XJTU_B6` | `primary` | 17.791 | 17.567–17.774 |
| Stage 1 | `XJTU_B6` | `logridge` | 15.340 | 14.880–15.419 |
| Stage 1 | `XJTU_B6` | `histgb` | 3.615 | 3.581–3.615 |
| Stage 1 | `XJTU_B6` | `endpoint` | 7.927 | 6.711–9.238 |
| Stage 1 | `XJTU_B6` | `without_nuisance` | 9.992 | 8.923–10.677 |
| Stage 2 | `XJTU_B2` | `primary` | 10.775 | 7.935–12.971 |
| Stage 2 | `XJTU_B2` | `isotropic_map` | 8.770 | 6.339–11.607 |
| Stage 2 | `XJTU_B2` | `scalar_map` | 4.007 | 3.925–4.308 |
| Stage 2 | `XJTU_B3` | `primary` | 2.067 | 2.078–2.325 |
| Stage 2 | `XJTU_B3` | `isotropic_map` | 2.054 | 2.078–2.291 |
| Stage 2 | `XJTU_B3` | `scalar_map` | 2.073 | 2.083–2.304 |
| Stage 2 | `XJTU_B4` | `primary` | 2.627 | 2.283–3.082 |
| Stage 2 | `XJTU_B4` | `isotropic_map` | 2.627 | 2.289–3.078 |
| Stage 2 | `XJTU_B4` | `scalar_map` | 2.714 | 2.250–3.271 |
| Stage 2 | `XJTU_B5` | `primary` | 5.393 | 5.097–5.693 |
| Stage 2 | `XJTU_B5` | `isotropic_map` | 5.522 | 5.232–5.802 |
| Stage 2 | `XJTU_B5` | `scalar_map` | 5.221 | 4.963–5.529 |
| Stage 2 | `XJTU_B6` | `primary` | 15.219 | 14.614–15.393 |
| Stage 2 | `XJTU_B6` | `isotropic_map` | 13.591 | 12.881–13.992 |
| Stage 2 | `XJTU_B6` | `scalar_map` | 6.683 | 6.598–6.762 |

---

## S3. OOD Detector Baselines, TJU Compatible Alarms & Fallback Control Policies (`E13`)

### Complete OOD / Domain-Mismatch Detector Benchmark Across 7 Methods (`E13`, 99th-Percentile Source Thresholds)

| Detector | B1 LOCO Alarm (%) | B4 Alarm (%) | Matched Rej. | B2 Alarm (%) [Rej.] | B6 Alarm (%) [Rej.] | Cross-Chem Rej. | Cycle AUROC (XJTU / All) |
| :--- | ---: | ---: | :---: | ---: | ---: | :---: | :---: |
| `oblique_full_gate` | 3.62 | 5.92 | 0/16 | 84.11 [15/15] | 97.29 [8/8] | 26/27 | 0.961 / 0.963 |
| `oblique_d2_only` | 1.40 | 5.92 | 0/16 | 72.69 [15/15] | 89.78 [8/8] | 26/27 | 0.964 / 0.903 |
| `maha_4d_cov` | 1.30 | 3.93 | 0/16 | 63.71 [13/15] | 89.39 [8/8] | 26/27 | 0.948 / 0.980 |
| `maha_4d_uncentered` | 1.27 | 2.86 | 0/16 | 61.14 [13/15] | 86.82 [8/8] | 26/27 | 0.954 / 0.954 |
| `pca_rank3_resid` | 1.23 | 2.01 | 0/16 | 58.13 [11/15] | 88.20 [8/8] | 26/27 | 0.915 / 0.949 |
| `ridge_nonidemp_resid` | 1.46 | 4.57 | 0/16 | 68.85 [14/15] | 89.09 [8/8] | 26/27 | 0.963 / 0.956 |
| `polarity_only_rule` | 0.71 | 0.00 | 0/16 | 0.72 [0/15] | 91.06 [8/8] | 6/27 | 0.616 / 0.580 |

### Warning vs. Closed-Loop Fallback Control Policies Across Cohorts (`E13c`, Clean MAE, pp)

| Cohort | Raw `primary` | `hold_last_safe` | `alarm_to_yref` | `latch_yref_on_3consec` | `alarm_to_damped_logridge` | `alarm_to_histgb` | `always_anchor_constant` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `XJTU_B1_LOCO` | 1.306 | 1.297 | 1.506 | 2.095 | 1.362 | 1.292 | 4.281 |
| `XJTU_B3` | 2.001 | 1.857 | 3.091 | 3.716 | 2.427 | 2.632 | 4.313 |
| `XJTU_B4` | 2.522 | 2.538 | 2.961 | 2.961 | 2.778 | 2.772 | 3.560 |
| `XJTU_B2` | 14.398 | 3.436 | 3.753 | 3.877 | 5.708 | 2.415 | 3.933 |
| `XJTU_B6` | 17.791 | 5.685 | 3.583 | 3.615 | 8.686 | 3.583 | 3.615 |
| `NASA` | 48.509 | 10.334 | 9.877 | 9.870 | 20.994 | 9.142 | 9.652 |
| `CALCE_CS2` | 23.239 | 16.882 | 16.301 | 16.301 | 22.841 | 16.301 | 16.300 |
| `TJU_2C` | 37.608 | 19.760 | 19.347 | 19.346 | 27.303 | 19.347 | 19.332 |
| `TJU_4C` | 33.998 | 20.056 | 19.138 | 19.137 | 25.666 | 19.138 | 19.097 |
| `B4_B6_mean` | 10.157 | 4.111 | 3.272 | 3.288 | 5.732 | 3.178 | 3.587 |

---

## S4. External Per-Cell Bootstrap CIs (`E14`) & Multi-Threshold SOH Knee Sensitivity (`E15`)

### External Validation Cell-Level Ranges & 20,000-Resample Bootstrap 95% CIs (`E14b`, pp)

| Suite ($m$ cells) | Model / Contrast | Mean MAE (pp) | Cell SD | [Min, Max] Across Cells | Bootstrap 95% CI |
| :--- | :--- | ---: | ---: | :---: | :---: |
| `CALCE_CS2_LOCO` ($m=4$) | `primary` | 1.223 | 0.254 | [0.963, 1.488] | [1.008, 1.437] |
| `CALCE_CS2_LOCO` ($m=4$) | `logridge` | 1.136 | 0.250 | [0.870, 1.442] | [0.940, 1.334] |
| `CALCE_CS2_LOCO` ($m=4$) | `histgb` | 1.483 | 0.310 | [1.133, 1.815] | [1.230, 1.737] |
| `CALCE_CS2_LOCO` ($m=4$) | `best_segment` | 1.543 | 0.248 | [1.222, 1.818] | [1.320, 1.743] |
| `CALCE_CS2_LOCO` ($m=4$) | `endpoint` | 2.082 | 0.259 | [1.705, 2.294] | [1.825, 2.257] |
| `CALCE_CS2_LOCO` ($m=4$) | `anchor_constant` | 16.300 | 0.475 | [15.850, 16.949] | [15.960, 16.729] |
| `CALCE_CS2_LOCO` ($m=4$) | `primary_minus_endpoint` | -0.860 | 0.451 | [-1.242, -0.319] | [-1.231, -0.488] |
| `CALCE_CS2_LOCO` ($m=4$) | `primary_minus_anchor_constant` | -15.078 | 0.452 | [-15.461, -14.464] | [-15.415, -14.690] |
| `TJU_1C_to_TJU_2C` ($m=3$) | `primary` | 0.827 | 0.040 | [0.797, 0.873] | [0.797, 0.873] |
| `TJU_1C_to_TJU_2C` ($m=3$) | `logridge` | 0.815 | 0.057 | [0.779, 0.880] | [0.779, 0.880] |
| `TJU_1C_to_TJU_2C` ($m=3$) | `histgb` | 0.910 | 0.193 | [0.697, 1.073] | [0.697, 1.073] |
| `TJU_1C_to_TJU_2C` ($m=3$) | `best_segment` | 1.531 | 0.470 | [1.146, 2.055] | [1.146, 2.055] |
| `TJU_1C_to_TJU_2C` ($m=3$) | `endpoint` | 9.082 | 0.053 | [9.029, 9.134] | [9.029, 9.134] |
| `TJU_1C_to_TJU_2C` ($m=3$) | `anchor_constant` | 19.332 | 0.095 | [19.231, 19.420] | [19.231, 19.420] |
| `TJU_1C_to_TJU_2C` ($m=3$) | `primary_minus_endpoint` | -8.256 | 0.035 | [-8.287, -8.218] | [-8.287, -8.218] |
| `TJU_1C_to_TJU_2C` ($m=3$) | `primary_minus_anchor_constant` | -18.506 | 0.092 | [-18.610, -18.434] | [-18.610, -18.434] |
| `TJU_1C_to_TJU_4C` ($m=3$) | `primary` | 1.356 | 0.564 | [0.905, 1.988] | [0.905, 1.988] |
| `TJU_1C_to_TJU_4C` ($m=3$) | `logridge` | 1.393 | 0.579 | [0.933, 2.043] | [0.933, 2.043] |
| `TJU_1C_to_TJU_4C` ($m=3$) | `histgb` | 1.309 | 0.087 | [1.216, 1.389] | [1.216, 1.389] |
| `TJU_1C_to_TJU_4C` ($m=3$) | `best_segment` | 1.553 | 0.274 | [1.350, 1.865] | [1.350, 1.865] |
| `TJU_1C_to_TJU_4C` ($m=3$) | `endpoint` | 8.416 | 0.291 | [8.161, 8.733] | [8.161, 8.733] |
| `TJU_1C_to_TJU_4C` ($m=3$) | `anchor_constant` | 19.097 | 0.302 | [18.757, 19.337] | [18.757, 19.337] |
| `TJU_1C_to_TJU_4C` ($m=3$) | `primary_minus_endpoint` | -7.060 | 0.275 | [-7.257, -6.746] | [-7.257, -6.746] |
| `TJU_1C_to_TJU_4C` ($m=3$) | `primary_minus_anchor_constant` | -17.741 | 0.866 | [-18.433, -16.769] | [-18.433, -16.769] |

### SOH Knee Threshold Sensitivity Across 90%, 85%, and 80% (`E15a`, Clean MAE, pp)

| Stage | Cohort | SOH Cutoff | High (`>`) Cells / Prof. | Low (`<=`) Cells / Prof. | `primary` (`>` / `<=`) | Comparator (`logridge` or `secant`) | `anchor_constant` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Stage1 | `XJTU_B1_LOCO` | 0.90 | 8 / 2372 | 8 / 831 | 1.04 / 2.10 | 1.01 / 2.02 | 2.62 / 8.84 |
| Stage1 | `XJTU_B1_LOCO` | 0.85 | 8 / 2803 | 8 / 400 | 1.21 / 2.12 | 1.11 / 2.35 | 3.21 / 11.52 |
| Stage1 | `XJTU_B1_LOCO` | 0.80 | 8 / 3157 | 6 / 46 | 1.29 / 3.42 | 1.23 / 3.94 | 4.12 / 14.56 |
| Stage1 | `XJTU_B3` | 0.90 | 8 / 3715 | 8 / 885 | 1.73 / 3.06 | 1.51 / 3.94 | 2.67 / 11.17 |
| Stage1 | `XJTU_B3` | 0.85 | 8 / 4178 | 8 / 422 | 1.74 / 4.40 | 1.60 / 5.77 | 3.32 / 14.12 |
| Stage1 | `XJTU_B3` | 0.80 | 8 / 4531 | 8 / 69 | 1.95 / 5.50 | 1.92 / 7.10 | 4.12 / 16.96 |
| Stage1 | `XJTU_B4` | 0.90 | 8 / 757 | 8 / 169 | 2.15 / 3.95 | 1.94 / 4.56 | 2.32 / 9.29 |
| Stage1 | `XJTU_B4` | 0.85 | 8 / 850 | 8 / 76 | 2.14 / 6.34 | 1.97 / 7.57 | 2.78 / 12.50 |
| Stage1 | `XJTU_B4` | 0.80 | 8 / 913 | 5 / 13 | 2.43 / 8.70 | 2.34 / 10.15 | 3.40 / 14.89 |
| Stage1 | `XJTU_B2` | 0.90 | 15 / 3091 | 15 / 621 | 13.31 / 18.42 | 11.77 / 16.95 | 3.27 / 7.20 |
| Stage1 | `XJTU_B2` | 0.85 | 15 / 3475 | 15 / 237 | 13.76 / 19.69 | 12.22 / 18.48 | 3.51 / 10.15 |
| Stage1 | `XJTU_B2` | 0.80 | 15 / 3703 | 2 / 9 | 14.36 / 27.05 | 12.78 / 21.93 | 3.91 / 13.35 |
| Stage1 | `XJTU_B6` | 0.90 | 8 / 1088 | 8 / 181 | 15.04 / 34.73 | 12.82 / 30.81 | 2.38 / 11.20 |
| Stage1 | `XJTU_B6` | 0.85 | 8 / 1191 | 8 / 78 | 16.32 / 39.51 | 13.99 / 35.27 | 2.92 / 14.09 |
| Stage1 | `XJTU_B6` | 0.80 | 8 / 1263 | 2 / 6 | 17.67 / 46.47 | 15.22 / 41.54 | 3.55 / 16.60 |
| Stage1 | `NASA_LOCO` | 0.90 | 14 / 416 | 13 / 335 | 5.05 / 10.44 | 5.03 / 10.66 | 4.83 / 15.26 |
| Stage1 | `NASA_LOCO` | 0.85 | 14 / 555 | 9 / 196 | 6.11 / 12.68 | 6.10 / 12.30 | 6.48 / 19.49 |
| Stage1 | `NASA_LOCO` | 0.80 | 14 / 647 | 7 / 104 | 6.90 / 14.34 | 6.84 / 13.95 | 8.20 / 22.25 |
| Stage1 | `CALCE_CS2_LOCO` | 0.90 | 4 / 769 | 4 / 2209 | 1.04 / 1.28 | 1.06 / 1.17 | 7.06 / 19.67 |
| Stage1 | `CALCE_CS2_LOCO` | 0.85 | 4 / 1579 | 4 / 1399 | 0.84 / 1.66 | 0.82 / 1.51 | 9.75 / 23.89 |
| Stage1 | `CALCE_CS2_LOCO` | 0.80 | 4 / 2122 | 4 / 856 | 0.88 / 2.07 | 0.85 / 1.84 | 11.57 / 28.07 |
| Stage1 | `TJU_1C_to_2C` | 0.90 | 3 / 611 | 3 / 2306 | 0.38 / 0.94 | 0.45 / 0.91 | 5.79 / 22.92 |
| Stage1 | `TJU_1C_to_2C` | 0.85 | 3 / 996 | 3 / 1921 | 0.67 / 0.91 | 0.68 / 0.89 | 8.38 / 25.01 |
| Stage1 | `TJU_1C_to_2C` | 0.80 | 3 / 1413 | 3 / 1504 | 0.82 / 0.84 | 0.82 / 0.81 | 11.07 / 27.09 |
| Stage1 | `TJU_1C_to_4C` | 0.90 | 3 / 614 | 3 / 2202 | 0.39 / 1.63 | 0.50 / 1.65 | 5.61 / 22.86 |
| Stage1 | `TJU_1C_to_4C` | 0.85 | 3 / 994 | 3 / 1822 | 0.52 / 1.82 | 0.56 / 1.86 | 8.24 / 25.02 |
| Stage1 | `TJU_1C_to_4C` | 0.80 | 3 / 1410 | 3 / 1406 | 0.82 / 1.91 | 0.85 / 1.96 | 10.98 / 27.25 |
| Stage2 | `XJTU_B1_LOCO` | 0.90 | 8 / 2807 | 8 / 364 | 1.21 / 2.55 | 2.82 / 13.57 | -- / -- |
| Stage2 | `XJTU_B1_LOCO` | 0.85 | 8 / 3119 | 5 / 52 | 1.32 / 3.45 | 3.86 / 16.68 | -- / -- |
| Stage2 | `XJTU_B1_LOCO` | 0.80 | 8 / 3171 | 0 / 0 | 1.35 / -- | 4.08 / -- | -- / -- |
| Stage2 | `XJTU_B3` | 0.90 | 8 / 4022 | 8 / 546 | 1.58 / 5.49 | 3.15 / 16.38 | -- / -- |
| Stage2 | `XJTU_B3` | 0.85 | 8 / 4385 | 8 / 183 | 1.86 / 7.06 | 4.12 / 19.46 | -- / -- |
| Stage2 | `XJTU_B3` | 0.80 | 8 / 4568 | 0 / 0 | 2.07 / -- | 4.74 / -- | -- / -- |
| Stage2 | `XJTU_B4` | 0.90 | 8 / 816 | 8 / 78 | 2.10 / 7.81 | 4.78 / 34.92 | -- / -- |
| Stage2 | `XJTU_B4` | 0.85 | 8 / 879 | 8 / 15 | 2.48 / 10.89 | 6.77 / 43.56 | -- / -- |
| Stage2 | `XJTU_B4` | 0.80 | 8 / 894 | 0 / 0 | 2.63 / -- | 7.39 / -- | -- / -- |
| Stage2 | `XJTU_B5` | 0.90 | 8 / 112 | 8 / 19 | 3.32 / 16.70 | 4.74 / 20.45 | -- / -- |
| Stage2 | `XJTU_B5` | 0.85 | 8 / 122 | 7 / 9 | 4.26 / 19.95 | 5.85 / 23.89 | -- / -- |
| Stage2 | `XJTU_B5` | 0.80 | 8 / 129 | 2 / 2 | 5.08 / 22.82 | 6.80 / 28.05 | -- / -- |
| Stage2 | `XJTU_B6` | 0.90 | 8 / 1129 | 8 / 108 | 13.37 / 33.63 | 3.80 / 15.72 | -- / -- |
| Stage2 | `XJTU_B6` | 0.85 | 8 / 1205 | 8 / 32 | 14.59 / 37.81 | 4.51 / 18.10 | -- / -- |
| Stage2 | `XJTU_B6` | 0.80 | 8 / 1237 | 0 / 0 | 15.22 / -- | 4.87 / -- | -- / -- |

### Label-Free Observable Knee Trigger Audit (`E15b`, Trigger on `y_hat <= tau_hat`, `y_end <= tau_hat`, or 3 Consecutive Gate Alarms)

| Cohort | Observable Cutoff | Caught Before True SOH `< 0.80` | Median True SOH at Trigger | Pre-Trigger Accepted MAE (pp) |
| :--- | :---: | :---: | ---: | ---: |
| `XJTU_B1_LOCO` | 0.90 | 6/6 | 0.9443 | 0.955 |
| `XJTU_B1_LOCO` | 0.85 | 6/6 | 0.9493 | 0.695 |
| `XJTU_B1_LOCO` | 0.80 | 4/6 | 0.9106 | 1.134 |
| `XJTU_B4` | 0.90 | 5/5 | 0.9504 | 1.560 |
| `XJTU_B4` | 0.85 | 5/5 | 0.8640 | 2.192 |
| `XJTU_B4` | 0.80 | 5/5 | 0.8408 | 2.370 |
| `XJTU_B6` | 0.90 | 2/2 | 0.9814 | 9.648 |
| `XJTU_B6` | 0.85 | 2/2 | 0.9814 | 9.648 |
| `XJTU_B6` | 0.80 | 2/2 | 0.9814 | 9.648 |
| `CALCE_CS2_LOCO` | 0.90 | 4/4 | 0.9824 | 1.304 |
| `CALCE_CS2_LOCO` | 0.85 | 4/4 | 0.9824 | 1.304 |
| `CALCE_CS2_LOCO` | 0.80 | 4/4 | 0.9824 | 1.304 |
| `TJU_1C_to_2C` | 0.90 | 3/3 | 0.8941 | 0.409 |
| `TJU_1C_to_2C` | 0.85 | 3/3 | 0.8388 | 0.721 |
| `TJU_1C_to_2C` | 0.80 | 1/3 | 0.7972 | 0.820 |
| `TJU_1C_to_4C` | 0.90 | 3/3 | 0.8994 | 0.389 |
| `TJU_1C_to_4C` | 0.85 | 3/3 | 0.8375 | 0.598 |
| `TJU_1C_to_4C` | 0.80 | 3/3 | 0.8243 | 0.670 |

---

## S5. Host Micro-Benchmark Environment Metadata (`E12`)

### Micro-Benchmark Platform & Accounting Specifications (`E12`)

- **OS:** `Windows-11-10.0.26300-SP0`
- **CPU / Architecture:** `AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD` (`AMD64`)
- **Python / NumPy:** Python `3.12.6`, NumPy `2.2.4`
- **Runtime Precision:** `float64`
- **Flash / SRAM / FLOPs Accounting:** analytical float32 parameters and FLOP counts (no microcontroller C binary was compiled; hardware-in-the-loop MCU testing remains future work)
- **Latency Timing Protocol:** time.perf_counter median and IQR across 20-25 batches (200-5000 repeats/batch)
