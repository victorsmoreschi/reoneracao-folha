"""Monta amostra de estimacao da reoneracao.

Regra (paper):
- mantem CNAEs tratados (lista arts. 7/8 preliminar)
- mantem CNAEs de controle na MESMA secao CNAE dos tratados
- exclui secoes sem vizinho comparavel: A (agro) e O (admin publica)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXCLUDE_SECOES = {"A", "O"}


def load_tratados(path: Path | None = None) -> pd.DataFrame:
    path = path or (ROOT / "cnaes_tratados.csv")
    t = pd.read_csv(path)
    t["cnae4"] = t["cnae4"].astype(str).str.zfill(4)
    return t


def build_estimation_sample(
    painel: pd.DataFrame,
    tratados: pd.DataFrame | None = None,
    exclude_secoes: set[str] | None = None,
) -> pd.DataFrame:
    exclude_secoes = exclude_secoes or EXCLUDE_SECOES
    tratados = tratados if tratados is not None else load_tratados()
    df = painel.copy()
    df["cnae4"] = df["cnae4"].astype(str).str.zfill(4)
    if "secao" not in df.columns:
        raise ValueError("painel sem coluna secao — rode download_caged.py --force")
    df["secao"] = df["secao"].astype(str).str.strip().str.upper().str[:1]
    df["tratado"] = df["cnae4"].isin(set(tratados["cnae4"])).astype(int)

    secoes_com_tratado = set(df.loc[df["tratado"] == 1, "secao"].unique())
    secoes_ok = secoes_com_tratado - set(exclude_secoes)

    out = df[df["secao"].isin(secoes_ok)].copy()
    return out.reset_index(drop=True)


def sample_report(df_full: pd.DataFrame, df_est: pd.DataFrame) -> str:
    lines = [
        f"universo: {len(df_full):,} celulas",
        f"amostra estimacao: {len(df_est):,} celulas",
        f"meses: {df_est['competencia'].nunique()} | munis: {df_est['municipio'].nunique()} | cnae4: {df_est['cnae4'].nunique()}",
        f"tratado=1: {(df_est['tratado']==1).sum():,} | controle: {(df_est['tratado']==0).sum():,}",
        "secoes:\n" + df_est.groupby(["secao", "tratado"]).size().unstack(fill_value=0).to_string(),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    full = pd.read_parquet(ROOT / "caged_movimentos.parquet")
    est = build_estimation_sample(full)
    print(sample_report(full, est))
    est.to_parquet(ROOT / "painel_estimacao.parquet", index=False)
    print("salvou painel_estimacao.parquet")

