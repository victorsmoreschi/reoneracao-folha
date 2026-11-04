# Payroll re-taxation in Brazil (*reoneração da folha*, Lei 14.973/2024): a dose difference-in-differences with Novo CAGED

Replication package for a master's research paper. It estimates how the gradual re-introduction of the employer payroll
contribution for sectors that had been on the revenue-based regime (*desoneração da folha*) affected formal employment flows and hiring wages.
The data are Novo CAGED microdata aggregated to CNAE-4 class × municipality × month, from 2023-01 to 2026-06.

## Design
- **Treatment.** 137 treated CNAE-4 classes: the sectors listed in Lei 14.973/2024 plus the Lei 13.670/2018 codes (`results/descriptives/cnaes_tratados.csv`).
  The **controls** are 248 non-listed classes in the same CNAE sections. Sections A (agriculture) and O (public administration) are excluded.
- **Dose.** The payroll contribution is phased back in at 5 p.p. in 2025 and 10 p.p. in 2026, relative to the 2023–24 baseline. Year coefficients β5 (2025) and β10 (2026) are reported along with Δ = β10 − β5 and Δ/5. β5 and β10 are ATTs relative to 2023–24 under parallel trends. Δ/5 is an average causal response only under strong parallel trends, as in the paper.
- **Outcomes.** Net employment (`saldo`), hires (`admitidos`), separations (`desligados`) and log mean hiring wage (`log_salario`).

## Specification
Estimation uses **pyfixest 0.60.0** (`feols`):

```
Y ~ beta5 + beta10 | cnae4^municipio^mes_cal + municipio^time_id + municipio^ano,  vcov = {"CRV1": "cnae4"}
```
Standard errors are CRV1, clustered by CNAE-4 class.
- **Sample sizes.** Flows: N = 9,219,210 of 9,221,016 cells. Wages: N = 2,222,824 of 2,799,760 cells.
  pyfixest removes singletons iteratively, so N (and the standard errors, slightly) can change across pyfixest versions. Use the pinned version to reproduce exactly.
- **Leaner comparison spec.** `cnae4^mes_cal + municipio + time_id` (wage N = 2,799,714).
- **Wage sample filters.** Hiring wage between 0.3 and 150 times the minimum wage. Cells with ≤ 2 hires and a mean wage above R$ 50,000 are dropped. Aug–Sep 2023 are dropped for wages.
- **Robustness.** Same specification (1) without section F (construction) and (2) restricted to the CNAE-2 divisions that contain a listed class. Event studies come in two forms: assigned dose W (base 2024, or base 2023 with 2024 free as an anticipation check) and binary treated × period (base 2024), with T_max and Wald tests on the 2023 leads, plus a 2024 placebo.

## Repository layout
```
scripts/     exact code used for the paper (00a → 10)
notebook/    notebook_codigo_final_reoneracao.ipynb (code + Portuguese explanations) and a code-only PDF
results/     main/ (principal spec), robustness/ (sem F, CNAE-2), descriptives/, resultados_paper_conferencia.xlsx
figures/     final English figures (PNG)
data/        how to obtain the Novo CAGED data (data itself is not versioned)
paper/       (to be added by a collaborator) manuscript source and PDF
```

## Reproduction (run in order)
| Script | Produces |
|---|---|
| `00a_download_caged.py` | Downloads Novo CAGED 2023-01–2026-06 from the MTE FTP and builds the month × municipality × CNAE-4 flow/wage panel |
| `00b_amostra_estimacao.py` | Estimation sample (treated classes + same-section controls; sections A and O excluded) → `painel_estimacao.parquet` |
| `01_build_panels_run_cristine_att_ano.py` | Balanced flow panel and filtered wage panel (`painel_balanced_flows.parquet`, `painel_wage.parquet`). Its own estimates are preliminary and are not used in the paper |
| `02_helpers_run_cgs_rodada2.py` | Helper module (imported as `run_cgs_rodada2`): dose variables, event-study, T_max and plotting functions, lean FE. Its standalone outputs are not used in the paper |
| `03_run_cgs_sat_main.py` | Principal ATT, lean comparison, linear dose, 2024 placebo, dose-W event studies (base 2024/2023), joint test → `results/main/` |
| `04_resume_es_binario.py` | Binary event study and T_max under the principal FE → `tab_es_binario_base2024.csv`, `tab_tmax.csv` |
| `05_run_cgs_sat_main_rob.py` | ATT robustness without section F and with the CNAE-2 sample → `results/robustness/tab_att_rob_*.csv` |
| `06_resume_es_dose_rob.py` | Dose-W event studies for sem-F and CNAE-2 → `tab_es_dose_{semF,cnae2}_base2024.csv` |
| `07_resume_es_bin_tmax_rob.py` | Binary event studies and T_max for sem-F and CNAE-2 → `tab_es_binario_*`, `tab_tmax_*` |
| `08_descritivas.py` | Table 2 raw descriptives → `tab_descritivas_recalc.csv` |
| `09_figuras_en_replot.py` | English replots of the event-study and raw-mean figures from the CSVs (no re-estimation) |
| `10_figura_es_base2023.py` | Assigned-dose event-study figures with 2023 omitted (2024 free) |

**Paths.** The scripts are kept exactly as they were run and contain hard-coded absolute input/output paths (e.g. `Path("/.../notebook_outputs/...")`).
Edit the `ROOT`/`OUT`/`PANEL_*` constants before running. Scripts 03–07 also expect script 02 to be importable as `run_cgs_rodada2.py` from their `ROOT` directory, so copy or rename it there.
Full re-estimation takes hours and around 30 GB of RAM. Install dependencies with `pip install -r requirements.txt`.

## Paper → script → results
| Paper | Script | File |
|---|---|---|
| Sample (137 treated, 248 controls) | 00a, 00b, 01 | `results/descriptives/cnaes_tratados.csv` |
| Table 2, Fig 1 | 08, 09 | `results/descriptives/tab_descritivas_recalc.csv`, `tab_monthly_means_raw.csv` |
| Table 3 (β5, β10, Δ, Δ/5), abstract | 03 | `results/main/tab_att_principal.csv` (+ `_wide_for_tex`) |
| Table 4 (placebo 2024) | 03 | `results/main/tab_placebo2024.csv` |
| Table 5 (T_max and Wald, 2023 leads) | 03, 04 | `results/main/tab_tmax.csv`, `tab_joint.csv` |
| Table 6 (lean spec) | 03 | `results/main/tab_att_lean_comparacao.csv` |
| Table 7 (linear dose) | 03 | `results/main/tab_reta_comparacao.csv` |
| Fig 2, 7–9 (dose-W, 2024 omitted) | 03 | `results/main/tab_es_dose_base2024.csv` |
| Fig 3, 10–12 (2023 omitted, 2024 free) | 03, 10 | `results/main/tab_es_dose_base2023.csv` |
| Fig 4, 13–15 (binary) | 04 | `results/main/tab_es_binario_base2024.csv` |
| Table 8, Figs 5–6, 16–21 (sem F; CNAE-2) | 05, 06, 07 | `results/robustness/*.csv` |
| English figures | 09, 10 | `figures/*.png` |

## Verification
`results/resultados_paper_conferencia.xlsx` has one sheet per table/figure. Its `Conferencia` sheet lists 228 numbers reported in the final PDF next to the corresponding CSV values, and all 228 match (OK).
The *Conferência* section of the notebook re-runs this check, and recomputes Table 2, in minutes without re-estimation (it needs the panels).
