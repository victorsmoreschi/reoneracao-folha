# Data (not versioned)

Raw data are **not** included in this repository (size and licensing). Everything here except this file is git-ignored.

**Source.** Novo CAGED microdata (movements, `CAGEDMOV*.7z`), Ministério do Trabalho e Emprego, public FTP:
`ftp://ftp.mtps.gov.br/pdet/microdados/NOVO CAGED/`, competences **2023-01 to 2026-06**.

**How to obtain.** `python scripts/00a_download_caged.py --start 202301 --end 202606`
downloads each month via FTP, extracts with `py7zr`, and aggregates to a competence × municipality × CNAE-4 panel
(`caged_movimentos.parquet`: hires, separations, net flow, mean hiring wage). Raw `.7z` files are deleted unless `--keep-raw`.
Script `00b` then builds `painel_estimacao.parquet`, and script `01` the balanced flow and wage panels used for estimation.
These intermediate `.parquet` files are large and are not distributed. Expect tens of GB of disk and around 30 GB of RAM for the saturated-FE event studies.
