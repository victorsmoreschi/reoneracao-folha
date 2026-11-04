"""Download REAL Novo CAGED (FTP MTE) + agrega painel com fluxo e salario.

Painel: competencia x municipio x cnae4 x secao
Cols: admitidos, desligados, saldo, salario_admissao_medio, n_admissao_com_salario
"""
from __future__ import annotations

import argparse
import ftplib
import re
import shutil
import unicodedata
from pathlib import Path

import pandas as pd
import py7zr

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "caged_raw"
SHARDS = ROOT / "caged_shards"
OUT = ROOT / "caged_movimentos.parquet"

FTP_HOST = "ftp.mtps.gov.br"
FTP_BASE = "pdet/microdados/NOVO CAGED"
START = "202301"
END = "202606"


def norm_col(c: str) -> str:
    c = c.replace("\xad", "").replace("\u00ad", "").replace("\xa0", " ")
    c = unicodedata.normalize("NFKD", c)
    c = "".join(ch for ch in c if not unicodedata.combining(ch))
    c = c.strip().lower()
    return re.sub(r"[^a-z0-9]+", "", c)


def map_col(nc: str) -> str | None:
    if nc.startswith("competencia") and "dec" not in nc:
        return "competencia"
    if "municip" in nc:
        return "municipio"
    if nc.startswith("secao") or nc == "secao":
        return "secao"
    if nc.startswith("subclasse"):
        return "subclasse"
    if "saldomoviment" in nc:
        return "saldo_mov"
    if nc == "salario" or (nc.startswith("salario") and "unidade" not in nc and "valor" not in nc and "fixo" not in nc):
        return "salario"
    return None


def month_range(start: str, end: str) -> list[str]:
    out = []
    y, m = int(start[:4]), int(start[4:])
    ye, me = int(end[:4]), int(end[4:])
    while (y < ye) or (y == ye and m <= me):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def ftp_connect() -> ftplib.FTP:
    ftp = ftplib.FTP(FTP_HOST, timeout=180, encoding="latin-1")
    ftp.login()
    return ftp


def download_month(ym: str, dest_7z: Path) -> Path:
    year = ym[:4]
    if dest_7z.exists() and dest_7z.stat().st_size > 1000:
        print(f"[{ym}] ja existe {dest_7z.name} ({dest_7z.stat().st_size:,} bytes)")
        return dest_7z
    ftp = ftp_connect()
    try:
        ftp.cwd(f"{FTP_BASE}/{year}/{ym}")
        names = ftp.nlst()
        match = next((n for n in names if n.upper().startswith("CAGEDMOV") and n.lower().endswith(".7z")), None)
        if not match:
            raise FileNotFoundError(f"CAGEDMOV nao encontrado em {year}/{ym}: {names}")
        print(f"[{ym}] baixando {match} ...")
        with open(dest_7z, "wb") as f:
            ftp.retrbinary(f"RETR {match}", f.write)
        print(f"[{ym}] ok ({dest_7z.stat().st_size:,} bytes)")
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()
    return dest_7z


def extract_txt(archive: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.txt")) + list(out_dir.glob("*.TXT"))
    if existing:
        return existing[0]
    print(f"  extraindo {archive.name} ...")
    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=out_dir)
    txts = list(out_dir.glob("*.txt")) + list(out_dir.glob("*.TXT"))
    if not txts:
        raise FileNotFoundError(f"Nenhum .txt em {out_dir}")
    return txts[0]


def parse_br_number(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")


def aggregate_txt(txt_path: Path, ym: str) -> pd.DataFrame:
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n\r")
    sep = ";" if header.count(";") >= header.count(",") else ","
    raw_cols = header.split(sep)
    mapping: dict[int, str] = {}
    for i, c in enumerate(raw_cols):
        dest = map_col(norm_col(c))
        if dest and dest not in mapping.values():
            mapping[i] = dest
    needed = {"municipio", "subclasse", "saldo_mov", "secao", "salario"}
    if not needed.issubset(set(mapping.values())):
        normed = [norm_col(c) for c in raw_cols]
        raise ValueError(f"Cols faltando. needed={needed} map={mapping} normed={normed[:30]}")

    usecols = sorted(mapping.keys())
    names = [mapping[i] for i in usecols]
    chunks = []
    for chunk in pd.read_csv(
        txt_path,
        sep=sep,
        encoding="utf-8",
        usecols=usecols,
        names=names,
        header=0,
        dtype=str,
        chunksize=400_000,
        low_memory=False,
    ):
        chunk["municipio"] = chunk["municipio"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
        chunk["subclasse"] = chunk["subclasse"].astype(str).str.replace(r"\D", "", regex=True)
        chunk["cnae4"] = chunk["subclasse"].str[:4]
        chunk["secao"] = chunk["secao"].astype(str).str.strip().str.upper().str[:1]
        mov = pd.to_numeric(chunk["saldo_mov"], errors="coerce").fillna(0).astype("int32")
        sal = parse_br_number(chunk["salario"])
        adm = (mov > 0).astype("int32")
        des = (mov < 0).astype("int32")
        sal_adm = sal.where(adm == 1)
        tmp = pd.DataFrame(
            {
                "competencia": ym,
                "municipio": chunk["municipio"],
                "cnae4": chunk["cnae4"],
                "secao": chunk["secao"],
                "admitidos": adm,
                "desligados": des,
                "saldo": mov,
                "sal_sum": sal_adm.fillna(0.0),
                "sal_n": sal_adm.notna().astype("int32"),
            }
        )
        g = tmp.groupby(["competencia", "municipio", "cnae4", "secao"], as_index=False).sum(numeric_only=True)
        chunks.append(g)

    out = pd.concat(chunks, ignore_index=True)
    out = out.groupby(["competencia", "municipio", "cnae4", "secao"], as_index=False).sum(numeric_only=True)
    out["salario_admissao_medio"] = out["sal_sum"] / out["sal_n"].where(out["sal_n"] > 0)
    out = out.drop(columns=["sal_sum"]).rename(columns={"sal_n": "n_admissao_com_salario"})
    for c in ["admitidos", "desligados", "saldo", "n_admissao_com_salario"]:
        out[c] = out[c].astype("int64")
    return out


def process_month(ym: str, keep_raw: bool = False, force: bool = False) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)
    shard = SHARDS / f"caged_{ym}.parquet"
    if shard.exists() and shard.stat().st_size > 1000 and not force:
        cols = set(pd.read_parquet(shard).columns)
        need = {"admitidos", "desligados", "salario_admissao_medio", "secao"}
        if need.issubset(cols):
            print(f"[{ym}] shard completo ja existe")
            return shard
        print(f"[{ym}] shard antigo — reprocessando")
    dest_7z = RAW / f"CAGEDMOV{ym}.7z"
    download_month(ym, dest_7z)
    extract_dir = RAW / ym
    txt = extract_txt(dest_7z, extract_dir)
    df = aggregate_txt(txt, ym)
    df.to_parquet(shard, index=False)
    print(f"[{ym}] shard {shard.name}: {len(df):,} linhas; cols={list(df.columns)}")
    if not keep_raw:
        shutil.rmtree(extract_dir, ignore_errors=True)
        try:
            dest_7z.unlink(missing_ok=True)
        except Exception:
            pass
    return shard


def stack_shards(months: list[str] | None = None) -> Path:
    files = sorted(SHARDS.glob("caged_*.parquet"))
    if months:
        want = {f"caged_{m}.parquet" for m in months}
        files = [f for f in files if f.name in want]
    if not files:
        raise FileNotFoundError("Nenhum shard em caged_shards/")
    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        if "admitidos" not in d.columns:
            raise ValueError(f"Shard antigo sem fluxo: {f.name}. Rode com --force")
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["wsal"] = df["salario_admissao_medio"].fillna(0) * df["n_admissao_com_salario"].fillna(0)
    num = ["admitidos", "desligados", "saldo", "n_admissao_com_salario", "wsal"]
    out = df.groupby(["competencia", "municipio", "cnae4", "secao"], as_index=False)[num].sum()
    out["salario_admissao_medio"] = out["wsal"] / out["n_admissao_com_salario"].where(out["n_admissao_com_salario"] > 0)
    out = out.drop(columns=["wsal"])
    out.to_parquet(OUT, index=False)
    print(
        f"Empilhado {OUT.name}: {len(out):,} linhas, {OUT.stat().st_size:,} bytes, meses={out['competencia'].nunique()}"
    )
    return OUT


def main(start: str = START, end: str = END, keep_raw: bool = False, force: bool = False):
    months = month_range(start, end)
    print(f"Baixando/agregando {len(months)} meses: {months[0]}..{months[-1]} force={force}")
    for ym in months:
        process_month(ym, keep_raw=keep_raw, force=force)
    stack_shards(months)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--only-stack", action="store_true")
    ap.add_argument("--force", action="store_true", help="reprocessa shards antigos")
    args = ap.parse_args()
    if args.only_stack:
        stack_shards(month_range(args.start, args.end))
    else:
        main(args.start, args.end, keep_raw=args.keep_raw, force=args.force)
