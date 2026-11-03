#!/usr/bin/env python3
"""CGS sat-main robustness under FE_MAIN (saturated FE as PRINCIPAL).

Specs (same FE_MAIN, CRV1 cnae4):
  1. Principal — full sample (copy from sat_main tab; fonte marked)
  2. Sem seção F (construção) — drop secao == "F"
  3. Amostra divisões CNAE 2 dígitos com listada — cnae2 in treated divisions

Optional: ES dose-W base2024 saldo (then other Y if RAM) under sem-F and cnae2.

Does NOT overwrite sat_main / paper_v3–v4 / beamer antigos.
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/workspace/notebook_outputs/cgs_rodada2")
SAT = Path("/workspace/notebook_outputs/cgs_rodada2_sat_main")
sys.path.insert(0, str(ROOT))

import run_cgs_rodada2 as r2  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUT = Path("/workspace/notebook_outputs/cgs_rodada2_sat_main_rob")
OUT.mkdir(parents=True, exist_ok=True)
REMODELADO = Path("/workspace/notebook_outputs/REMODELADO_sat_main_rob_v1")
(REMODELADO / "tabs").mkdir(parents=True, exist_ok=True)
(REMODELADO / "figuras").mkdir(parents=True, exist_ok=True)

FE_MAIN = "cnae4^municipio^mes_cal + municipio^time_id + municipio^ano"
FE_SAT_LEAN = "cnae4^mes_cal + municipio^time_id + municipio^ano"
FE_LEAN = "cnae4^mes_cal + municipio + time_id"

OUTCOMES_FLOW = r2.OUTCOMES_FLOW
OUTCOMES_ALL = r2.OUTCOMES_ALL
PANEL_BAL = r2.PANEL_BAL
PANEL_WAGE = r2.PANEL_WAGE
PANEL_META = r2.PANEL_META

FAILURES: list[str] = []
r2.FAILURES = FAILURES
r2.OUT = OUT

ES_FE_USED: dict[str, str] = {}
ES_PRODUCED: list[str] = []


def _mem(tag: str = "") -> None:
    r2._mem(tag)


def fit(fml, data, label=""):
    return r2.fit(fml, data, label)


def wide_att_one(att_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in OUTCOMES_ALL:
        sub = att_df.loc[att_df.outcome == y]
        if sub.empty:
            continue

        def g(term):
            r = sub.loc[sub.term == term]
            return None if r.empty else r.iloc[0]

        r5, r10, rd, rd5 = g("beta5"), g("beta10"), g("delta"), g("delta_over_5")
        if r5 is None:
            continue
        rows.append(
            {
                "outcome": y,
                "b5": r5.coef,
                "se5": r5.se,
                "p5": r5.p,
                "b10": r10.coef,
                "se10": r10.se,
                "p10": r10.p,
                "d": rd.coef,
                "se_d": rd.se,
                "p_d": rd.p,
                "d5": rd5.coef,
                "se_d5": rd5.se,
                "p_d5": rd5.p,
                "N": int(r5.N),
                "fe": r5.fe,
                "spec": r5.spec,
            }
        )
    return pd.DataFrame(rows)


def build_wide_side_by_side(atts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """principal | semF | cnae2 side by side for β5,β10,Δ,Δ/5 × 4 Y."""
    specs_order = ["principal", "semF", "cnae2"]
    terms = ["beta5", "beta10", "delta", "delta_over_5"]
    rows = []
    for y in OUTCOMES_ALL:
        row = {"outcome": y}
        for sp in specs_order:
            df = atts.get(sp)
            if df is None or df.empty:
                for t in terms:
                    row[f"{sp}_{t}"] = np.nan
                    row[f"{sp}_{t}_se"] = np.nan
                    row[f"{sp}_{t}_p"] = np.nan
                row[f"{sp}_N"] = np.nan
                continue
            sub = df.loc[df.outcome == y]
            for t in terms:
                r = sub.loc[sub.term == t]
                if r.empty:
                    row[f"{sp}_{t}"] = np.nan
                    row[f"{sp}_{t}_se"] = np.nan
                    row[f"{sp}_{t}_p"] = np.nan
                else:
                    row[f"{sp}_{t}"] = float(r.coef.iloc[0])
                    row[f"{sp}_{t}_se"] = float(r.se.iloc[0])
                    row[f"{sp}_{t}_p"] = float(r.p.iloc[0])
            r5 = sub.loc[sub.term == "beta5"]
            row[f"{sp}_N"] = int(r5.N.iloc[0]) if not r5.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def try_es(runner, df, outcome, base_year, fe_candidates, tag):
    last_err = None
    for fe in fe_candidates:
        print(f"  Trying {tag}/{outcome} with FE={fe}", flush=True)
        try:
            es_df, m, levels = runner(df, outcome, base_year=base_year, fe=fe)
            if m is not None and es_df is not None and len(es_df):
                if fe != FE_MAIN:
                    FAILURES.append(
                        f"{tag}/{outcome}: FE_MAIN OOM/fail; used leaner FE=`{fe}`"
                    )
                ES_FE_USED[f"{tag}/{outcome}"] = fe
                return es_df, m, levels, fe
            del m
            gc.collect()
            last_err = f"null model under {fe}"
        except MemoryError as e:
            last_err = str(e)
            FAILURES.append(f"{tag}/{outcome} MemoryError under `{fe}`: {e}")
            gc.collect()
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            FAILURES.append(f"{tag}/{outcome} under `{fe}`: {last_err}")
            gc.collect()
    FAILURES.append(f"{tag}/{outcome}: ALL FE candidates failed ({last_err})")
    ES_FE_USED[f"{tag}/{outcome}"] = "FAILED"
    return pd.DataFrame(), None, [], "FAILED"


def plot_es_dose(es_df, outcome, path, title_prefix="", fe_note=""):
    sub = es_df.loc[es_df["outcome"] == outcome].sort_values("k")
    if sub.empty:
        return False
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="gray", ls="--", lw=0.8)
    ax.axvspan(-0.5, 11.5, color="C0", alpha=0.08, label="2025 (dose 5)")
    ax.axvspan(11.5, 18.5, color="C1", alpha=0.10, label="2026 (dose 10)")
    ax.errorbar(
        sub["k"],
        sub["coef"],
        yerr=1.96 * sub["se"],
        fmt="o",
        ms=3.5,
        capsize=2,
        color="black",
        ecolor="gray",
        label=r"$\theta_k$ (per-point)",
    )
    ax.set_xlabel("Event time k (k=0 at Jan/2025)")
    ax.set_ylabel(r"$\theta_k$ (effect per dose point)")
    title = (
        f"ES dose-W ({outcome}) — {title_prefix} | FE saturada; base 2024 omitted\n"
        r"Pre $W{=}5$ = lead scaling (not calendar dose). "
        r"$5\theta_k\approx$ level 2025; $10\theta_k\approx$ level 2026."
    )
    if fe_note:
        title += f"\n[{fe_note}]"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def run_att_block(flows, wage, tag: str) -> pd.DataFrame:
    rows = []
    for y in OUTCOMES_FLOW:
        r, m = r2.run_att(flows, y, FE_MAIN, tag)
        rows.extend(r)
        del m
        gc.collect()
        _mem(f"after ATT {tag}/{y}")
    r, m = r2.run_att(wage, "log_salario", FE_MAIN, tag)
    rows.extend(r)
    del m
    gc.collect()
    return pd.DataFrame(rows)


def write_texts(info, atts, es_note):
    def g(df, term, outcome="saldo"):
        if df is None or df.empty:
            return None
        r = df.loc[(df.outcome == outcome) & (df.term == term)]
        return None if r.empty else r.iloc[0]

    lines = []
    lines.append("# RESUMO — CGS sat-main robustez (FE saturada = FE_MAIN)\n")
    lines.append("## Identificação\n")
    lines.append(
        f"- **FE saturada (FE_MAIN):** `Y ~ t2025 + t2026 | {FE_MAIN}` (CRV1 cnae4)."
    )
    lines.append(
        "- D_s binário (tratado=1 se classe listada). Base omitida: 2023–2024. "
        "2026 só até junho. β5 = 2025 vs base; β10 = 2026 vs **mesma** base."
    )
    lines.append(
        "- Reportamos Δ=β10−β5 e (β10−β5)/5 com EP e **p** clusterizados."
    )
    lines.append(
        "- FE_MAIN **não** inclui classe×ano / classe×time_id / classe×2025–26. "
        "UF×time omitido (colinear). String escrita pode listar também `cnae4^mes_cal` (aninhado).\n"
    )
    lines.append("## Robustez (mesma FE_MAIN)\n")
    lines.append("1. **Principal** — amostra completa (fonte: sat_main `tab_att_principal.csv`).")
    lines.append("2. **Sem seção F (construção)** — dropar `secao == \"F\"` em flows e wage.")
    lines.append(
        "3. **Amostra divisões CNAE 2 dígitos com listada** — células cujo "
        "`cnae2 = cnae4[:2]` está em alguma divisão que tem pelo menos uma classe listada "
        "(`tratado==1`).\n"
    )

    lines.append("## Painel\n")
    lines.append(f"- Pares: **{info.get('n_pairs')}**; N flows (completo): **{info.get('N_flows')}**")
    lines.append(f"- Competências: {info.get('comps_first')}–{info.get('comps_last')}")
    lines.append(f"- NEVER_DROP: {info.get('never_drop')}")
    lines.append(f"- Wage drop months: {info.get('salary_drop_months')}\n")

    labels = {
        "principal": "Principal (amostra completa)",
        "semF": "Sem seção F (construção)",
        "cnae2": "Amostra divisões CNAE 2 dígitos com listada",
    }
    for sp, lab in labels.items():
        df = atts.get(sp)
        lines.append(f"## ATT — {lab}\n")
        lines.append("| outcome | β5 | se | p | β10 | se | p | Δ | se | p | Δ/5 | se | p | N |")
        lines.append("|---------|----|----|---|-----|----|---|---|----|---|-----|----|---|---|")
        if df is None or df.empty:
            lines.append("| _(vazio)_ | | | | | | | | | | | | | |")
            lines.append("")
            continue
        for y in OUTCOMES_ALL:
            r5, r10, rd, rd5 = g(df, "beta5", y), g(df, "beta10", y), g(df, "delta", y), g(df, "delta_over_5", y)
            if r5 is None:
                continue
            lines.append(
                f"| {y} | {r5.coef:.4f} | {r5.se:.4f} | {r5.p:.4f} | "
                f"{r10.coef:.4f} | {r10.se:.4f} | {r10.p:.4f} | "
                f"{rd.coef:.4f} | {rd.se:.4f} | {rd.p:.4f} | "
                f"{rd5.coef:.4f} | {rd5.se:.4f} | {rd5.p:.4f} | {int(r5.N)} |"
            )
        lines.append("")

    if es_note:
        lines.append("## ES dose-W (opcional)\n")
        lines.append(es_note)
        lines.append("")

    lines.append("## Caveats\n")
    lines.append("1. Mesma FE saturada em todas as colunas de robustez.")
    lines.append("2. (β10−β5)/5 sob hipótese forte seria ACRT; com dinâmica 2026 **não** declaramos ACRT limpo.")
    lines.append("3. Wage: pre-trends frágeis — caveat mantido.")
    lines.append("4. Não sobrescrever packs antigos (sat_main / paper_v3–v4 / beamer).\n")

    if FAILURES:
        lines.append("## Failures / FE fallbacks\n")
        for f in FAILURES:
            lines.append(f"- {f}")
        lines.append("")

    (OUT / "RESUMO.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote RESUMO.md", flush=True)

    # TEXTO_RESULTADOS
    def fmt_saldo(sp):
        df = atts.get(sp)
        s5, s10 = g(df, "beta5"), g(df, "beta10")
        if s5 is None:
            return f"**{labels[sp]}:** (falhou / vazio)."
        return (
            f"**{labels[sp]}:** β5 = {float(s5.coef):.4f} (EP = {float(s5.se):.4f}, p = {float(s5.p):.3f}), "
            f"β10 = {float(s10.coef):.4f} (EP = {float(s10.se):.4f}, p = {float(s10.p):.3f}), "
            f"N = {int(s5.N)}."
        )

    texto = f"""# TEXTO_RESULTADOS — CGS sat-main robustez (FE saturada)

## 1. Especificação

Estimamos sob a **FE saturada (FE_MAIN)**
`Y ~ t2025 + t2026 | {FE_MAIN}`
com erros CRV1 por cnae4. D_s = 1 se a classe CNAE está listada (invariante no tempo).
Base omitida: 2023–2024; 2026 só até junho. β5 = ATT 2025 vs base; β10 = ATT 2026 vs a **mesma** base.
Também reportamos Δ = β10 − β5 e (β10 − β5)/5 com EP e p.

**Não** incluímos classe×ano, classe×time_id nem classe×2025/2026 (absorveriam a dose).
UF×time omitido (colinear). A string escrita pode listar também `cnae4^mes_cal` (aninhado em `cnae4^municipio^mes_cal`).

## 2. Robustez (mesma FE_MAIN)

Três amostras:
1. **Principal** — amostra completa (reutilizada de sat_main; fonte marcada).
2. **Sem seção F (construção)** — exclusão da construção.
3. **Amostra divisões CNAE 2 dígitos com listada** — restringe a divisões CNAE-2 que contêm ao menos uma classe listada.

### Saldo (destaque)

{fmt_saldo("principal")}
{fmt_saldo("semF")}
{fmt_saldo("cnae2")}

## 3. ES dose-W (opcional no deck)

{es_note if es_note else "ES dose-W não produzido nesta rodada (prioridade: ATT completo)."}

## 4. Interpretação de (β10 − β5)/5

Sob a hipótese forte de tendências paralelas em doses (CGS Thm 3.3(d)), (β10−β5)/5 identificaria o ACRT.
Com dinâmica de 2026 (já um ano em dose 5) **não** declaramos ACRT limpo nem validamos a hipótese forte.
Não usamos β5/5 = β10/10 como teste da hipótese forte.

## 5. Artefatos

Pasta: `{OUT}`
Stage Overleaf: `{REMODELADO}`
Não editar Overleaf .tex nesta rodada; não sobrescrever packs antigos.
"""
    (OUT / "TEXTO_RESULTADOS.md").write_text(texto, encoding="utf-8")
    print("Wrote TEXTO_RESULTADOS.md", flush=True)


def stage_remodelado():
    tabs = [
        "tab_att_principal.csv",
        "tab_att_rob_semF.csv",
        "tab_att_rob_cnae2.csv",
        "tab_att_rob_wide_for_tex.csv",
        "tab_es_dose_semF_base2024.csv",
        "tab_es_dose_cnae2_base2024.csv",
    ]
    figs = list(OUT.glob("fig_*.png"))
    for name in tabs:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, REMODELADO / "tabs" / name)
    for src in figs:
        shutil.copy2(src, REMODELADO / "figuras" / src.name)
    for name in [
        "RESUMO.md",
        "TEXTO_RESULTADOS.md",
        "key_numbers.json",
        "run_cgs_sat_main_rob.py",
    ]:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, REMODELADO / name)
    nb = OUT / "notebook_reoneracao_cgs_sat_main_rob.ipynb"
    if nb.exists():
        shutil.copy2(nb, REMODELADO / nb.name)
    readme = (
        "REMODELADO_sat_main_rob_v1 — robustez sob FE saturada (FE_MAIN)\n\n"
        f"FE_MAIN = `{FE_MAIN}`\n\n"
        "Specs:\n"
        "  - Principal (amostra completa; fonte sat_main)\n"
        "  - Sem seção F (construção)\n"
        "  - Amostra divisões CNAE 2 dígitos com listada\n\n"
        "tabs/: tab_att_principal, tab_att_rob_semF, tab_att_rob_cnae2,\n"
        "       tab_att_rob_wide_for_tex (+ ES tabs if produced)\n"
        "figuras/: ES dose-W PNG if produced\n"
        "RESUMO + TEXTO + key_numbers.json\n\n"
        "Naming: FE saturada / sem seção F / CNAE 2 dígitos com listada\n"
        "(sem letras A/B/F de specs antigas).\n"
        "Do NOT overwrite sat_main / paper_v3–v4 / beamer. Do not edit Overleaf live.\n"
    )
    (REMODELADO / "README_ASSETS.txt").write_text(readme, encoding="utf-8")
    print(f"Staged to {REMODELADO}", flush=True)


def build_notebook():
    """Create dedicated robustness notebook via nbformat."""
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Reoneração CGS — robustez sob FE saturada (FE_MAIN)\n\n"
            f"**FE_MAIN** = `{FE_MAIN}`\n\n"
            "Specs: principal (amostra completa) | sem seção F (construção) | "
            "amostra divisões CNAE 2 dígitos com listada.\n\n"
            "ATT: `Y ~ t2025 + t2026 | FE_MAIN` (CRV1 cnae4). "
            "Não sobrescreve packs antigos."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """\
import gc, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/workspace/notebook_outputs/cgs_rodada2")
SAT = Path("/workspace/notebook_outputs/cgs_rodada2_sat_main")
OUT = Path("/workspace/notebook_outputs/cgs_rodada2_sat_main_rob")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
import run_cgs_rodada2 as r2

FE_MAIN = "cnae4^municipio^mes_cal + municipio^time_id + municipio^ano"
OUTCOMES_FLOW = r2.OUTCOMES_FLOW
OUTCOMES_ALL = r2.OUTCOMES_ALL
print("FE_MAIN =", FE_MAIN)
print("OUT =", OUT)
"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """\
# Carregar painéis (mesmos PANEL_BAL / PANEL_WAGE do rodada2)
flow_cols = [
    "cnae4", "municipio", "secao", "competencia", "ano", "mes_cal", "time_id",
    "k", "tratado", "dose", "t2025", "t2026", "saldo", "admitidos", "desligados",
]
flows = pd.read_parquet(r2.PANEL_BAL, columns=flow_cols)
wage = pd.read_parquet(r2.PANEL_WAGE)
print(f"flows N={len(flows)}; wage N={len(wage)}")
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Principal — amostra completa\n\n"
            "Copia de `cgs_rodada2_sat_main/tab_att_principal.csv` (fonte marcada) "
            "ou reestima sob FE_MAIN."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """\
src = SAT / "tab_att_principal.csv"
if src.exists():
    att_principal = pd.read_csv(src)
    att_principal["fonte"] = "copied_from_sat_main"
    print("Principal ATT copied from sat_main")
else:
    rows = []
    for y in OUTCOMES_FLOW:
        r, m = r2.run_att(flows, y, FE_MAIN, "att_principal")
        rows.extend(r); del m; gc.collect()
    r, m = r2.run_att(wage, "log_salario", FE_MAIN, "att_principal")
    rows.extend(r); del m; gc.collect()
    att_principal = pd.DataFrame(rows)
    att_principal["fonte"] = "reestimated"
att_principal.to_csv(OUT / "tab_att_principal.csv", index=False)
att_principal.loc[att_principal.outcome=="saldo", ["term","coef","se","p","N"]]
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Sem seção F (construção)\n\nDropar `secao == \"F\"` em flows e wage."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """\
flows_semF = flows.loc[flows["secao"].astype(str) != "F"]
wage_semF = wage.loc[wage["secao"].astype(str) != "F"]
print(f"semF flows N={len(flows_semF)}; wage N={len(wage_semF)}")
rows = []
for y in OUTCOMES_FLOW:
    r, m = r2.run_att(flows_semF, y, FE_MAIN, "att_rob_semF")
    rows.extend(r); del m; gc.collect()
r, m = r2.run_att(wage_semF, "log_salario", FE_MAIN, "att_rob_semF")
rows.extend(r); del m; gc.collect()
att_semF = pd.DataFrame(rows)
att_semF.to_csv(OUT / "tab_att_rob_semF.csv", index=False)
att_semF.loc[att_semF.outcome=="saldo", ["term","coef","se","p","N"]]
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Amostra divisões CNAE 2 dígitos com listada\n\n"
            "`cnae2 = cnae4[:2]` deve estar em alguma divisão que tem pelo menos uma classe listada."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """\
flows = flows.copy()
wage = wage.copy()
flows["cnae2"] = flows["cnae4"].astype(str).str[:2]
wage["cnae2"] = wage["cnae4"].astype(str).str[:2]
treated_div = set(flows.loc[flows["tratado"] == 1, "cnae2"].astype(str).unique())
flows_c2 = flows.loc[flows["cnae2"].astype(str).isin(treated_div)]
wage_c2 = wage.loc[wage["cnae2"].astype(str).isin(treated_div)]
print(f"cnae2 divs com listada: {len(treated_div)}; flows N={len(flows_c2)}; wage N={len(wage_c2)}")
rows = []
for y in OUTCOMES_FLOW:
    r, m = r2.run_att(flows_c2, y, FE_MAIN, "att_rob_cnae2")
    rows.extend(r); del m; gc.collect()
r, m = r2.run_att(wage_c2, "log_salario", FE_MAIN, "att_rob_cnae2")
rows.extend(r); del m; gc.collect()
att_cnae2 = pd.DataFrame(rows)
att_cnae2.to_csv(OUT / "tab_att_rob_cnae2.csv", index=False)
att_cnae2.loc[att_cnae2.outcome=="saldo", ["term","coef","se","p","N"]]
"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. Wide side-by-side (principal | semF | cnae2)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """\
# Prefer CSVs already written by run_cgs_sat_main_rob.py if present
def _load(name, fallback):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else fallback

att_principal = _load("tab_att_principal.csv", att_principal)
att_semF = _load("tab_att_rob_semF.csv", att_semF)
att_cnae2 = _load("tab_att_rob_cnae2.csv", att_cnae2)

specs = {"principal": att_principal, "semF": att_semF, "cnae2": att_cnae2}
terms = ["beta5", "beta10", "delta", "delta_over_5"]
rows = []
for y in OUTCOMES_ALL:
    row = {"outcome": y}
    for sp, df in specs.items():
        sub = df.loc[df.outcome == y]
        for t in terms:
            r = sub.loc[sub.term == t]
            row[f"{sp}_{t}"] = float(r.coef.iloc[0]) if not r.empty else np.nan
            row[f"{sp}_{t}_se"] = float(r.se.iloc[0]) if not r.empty else np.nan
            row[f"{sp}_{t}_p"] = float(r.p.iloc[0]) if not r.empty else np.nan
        r5 = sub.loc[sub.term == "beta5"]
        row[f"{sp}_N"] = int(r5.N.iloc[0]) if not r5.empty else np.nan
    rows.append(row)
wide = pd.DataFrame(rows)
wide.to_csv(OUT / "tab_att_rob_wide_for_tex.csv", index=False)
wide
"""
        )
    )

    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python (.venv_cgs)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    path = OUT / "notebook_reoneracao_cgs_sat_main_rob.ipynb"
    nbf.write(nb, path)
    print(f"Wrote notebook {path}", flush=True)
    return path


def main():
    print("=== CGS sat-main ROB under FE_MAIN ===", flush=True)
    print(f"FE_MAIN = {FE_MAIN}", flush=True)
    info = json.loads(PANEL_META.read_text()) if PANEL_META.exists() else {}

    flow_cols = [
        "cnae4",
        "municipio",
        "secao",
        "competencia",
        "ano",
        "mes_cal",
        "time_id",
        "k",
        "tratado",
        "dose",
        "t2025",
        "t2026",
        "saldo",
        "admitidos",
        "desligados",
    ]
    print("Loading panels...", flush=True)
    flows = pd.read_parquet(PANEL_BAL, columns=flow_cols)
    wage = pd.read_parquet(PANEL_WAGE)
    print(f"flows N={len(flows)}; wage N={len(wage)}", flush=True)
    _mem("loaded")

    atts: dict[str, pd.DataFrame] = {}

    # ---- 1. Principal: copy from sat_main ----
    print("=== 1. Principal ATT (copy from sat_main) ===", flush=True)
    src = SAT / "tab_att_principal.csv"
    if src.exists():
        att_p = pd.read_csv(src)
        att_p["fonte"] = "copied_from_sat_main"
        # ensure fe column matches
        if "fe" not in att_p.columns:
            att_p["fe"] = FE_MAIN
        att_p.to_csv(OUT / "tab_att_principal.csv", index=False)
        atts["principal"] = att_p
        print(f"  Copied {len(att_p)} rows from {src}", flush=True)
    else:
        FAILURES.append("sat_main tab_att_principal.csv missing; re-estimating principal")
        att_p = run_att_block(flows, wage, "att_principal")
        att_p["fonte"] = "reestimated"
        att_p.to_csv(OUT / "tab_att_principal.csv", index=False)
        atts["principal"] = att_p

    # ---- 2. Sem seção F ----
    print("=== 2. Sem seção F (construção) ===", flush=True)
    flows_semF = flows.loc[flows["secao"].astype(str) != "F"].copy()
    wage_semF = wage.loc[wage["secao"].astype(str) != "F"].copy()
    print(f"  semF flows N={len(flows_semF)}; wage N={len(wage_semF)}", flush=True)
    # assign W for possible ES
    flows_semF["W"] = r2.assign_W(flows_semF["tratado"], flows_semF["ano"])
    wage_semF["W"] = r2.assign_W(wage_semF["tratado"], wage_semF["ano"])
    att_sf = run_att_block(flows_semF, wage_semF, "att_rob_semF")
    att_sf.to_csv(OUT / "tab_att_rob_semF.csv", index=False)
    atts["semF"] = att_sf
    print(wide_att_one(att_sf).to_string(index=False), flush=True)
    _mem("after ATT semF")

    # ---- 3. CNAE 2 dígitos com listada ----
    print("=== 3. Amostra divisões CNAE 2 dígitos com listada ===", flush=True)
    flows = flows.copy()
    wage = wage.copy()
    flows["cnae2"] = flows["cnae4"].astype(str).str[:2]
    wage["cnae2"] = wage["cnae4"].astype(str).str[:2]
    treated_div = set(flows.loc[flows["tratado"] == 1, "cnae2"].astype(str).unique())
    flows_c2 = flows.loc[flows["cnae2"].astype(str).isin(treated_div)].copy()
    wage_c2 = wage.loc[wage["cnae2"].astype(str).isin(treated_div)].copy()
    flows_c2["W"] = r2.assign_W(flows_c2["tratado"], flows_c2["ano"])
    wage_c2["W"] = r2.assign_W(wage_c2["tratado"], wage_c2["ano"])
    print(
        f"  treated_divs={len(treated_div)}; flows N={len(flows_c2)}; wage N={len(wage_c2)}",
        flush=True,
    )
    att_c2 = run_att_block(flows_c2, wage_c2, "att_rob_cnae2")
    att_c2.to_csv(OUT / "tab_att_rob_cnae2.csv", index=False)
    atts["cnae2"] = att_c2
    print(wide_att_one(att_c2).to_string(index=False), flush=True)
    _mem("after ATT cnae2")

    # ---- Wide ----
    wide = build_wide_side_by_side(atts)
    wide.to_csv(OUT / "tab_att_rob_wide_for_tex.csv", index=False)
    print("=== WIDE ===", flush=True)
    print(wide.to_string(index=False), flush=True)

    # Free full panels before ES
    del flows, wage
    gc.collect()
    _mem("freed full panels")

    # ---- Optional ES dose-W ----
    es_cands = [FE_MAIN, FE_SAT_LEAN, FE_LEAN]
    es_notes = []

    print("=== ES dose-W base2024 saldo | sem seção F ===", flush=True)
    es_sf_all = []
    try:
        es_df, m, levels, fe_u = try_es(
            r2.run_es_dose_W, flows_semF, "saldo", 2024, es_cands, "ES_dose_semF"
        )
        if len(es_df):
            es_df = es_df.copy()
            es_df["fe"] = fe_u
            es_df["rob_spec"] = "semF"
            es_sf_all.append(es_df)
            note = "" if fe_u == FE_MAIN else f"FE used: {fe_u}"
            ok = plot_es_dose(
                es_df,
                "saldo",
                OUT / "fig_es_dose_semF_base2024_saldo.png",
                title_prefix="sem seção F (construção)",
                fe_note=note,
            )
            if ok:
                ES_PRODUCED.append("fig_es_dose_semF_base2024_saldo.png")
            es_notes.append(f"ES dose-W saldo sem-F sob FE=`{fe_u}` (N model rows in tab).")
        del m
        gc.collect()
        # try other Y if RAM allows
        for y in ["admitidos", "desligados"]:
            _mem(f"before ES semF/{y}")
            try:
                es_df, m, _, fe_u = try_es(
                    r2.run_es_dose_W, flows_semF, y, 2024, es_cands, "ES_dose_semF"
                )
                if len(es_df):
                    es_df = es_df.copy()
                    es_df["fe"] = fe_u
                    es_df["rob_spec"] = "semF"
                    es_sf_all.append(es_df)
                    note = "" if fe_u == FE_MAIN else f"FE used: {fe_u}"
                    ok = plot_es_dose(
                        es_df,
                        y,
                        OUT / f"fig_es_dose_semF_base2024_{y}.png",
                        title_prefix="sem seção F (construção)",
                        fe_note=note,
                    )
                    if ok:
                        ES_PRODUCED.append(f"fig_es_dose_semF_base2024_{y}.png")
                del m
                gc.collect()
            except MemoryError as e:
                FAILURES.append(f"ES semF/{y} OOM: {e}; skipping remaining semF ES")
                gc.collect()
                break
        # wage ES if RAM
        try:
            es_df, m, _, fe_u = try_es(
                r2.run_es_dose_W, wage_semF, "log_salario", 2024, es_cands, "ES_dose_semF"
            )
            if len(es_df):
                es_df = es_df.copy()
                es_df["fe"] = fe_u
                es_df["rob_spec"] = "semF"
                es_sf_all.append(es_df)
                note = "" if fe_u == FE_MAIN else f"FE used: {fe_u}"
                ok = plot_es_dose(
                    es_df,
                    "log_salario",
                    OUT / "fig_es_dose_semF_base2024_log_salario.png",
                    title_prefix="sem seção F (construção)",
                    fe_note=note,
                )
                if ok:
                    ES_PRODUCED.append("fig_es_dose_semF_base2024_log_salario.png")
            del m
            gc.collect()
        except MemoryError as e:
            FAILURES.append(f"ES semF/log_salario OOM: {e}")
            gc.collect()
    except MemoryError as e:
        FAILURES.append(f"ES semF/saldo OOM: {e}; prioritize ATT (done)")
        es_notes.append("ES dose-W sem-F: OOM — não produzido.")
        gc.collect()

    if es_sf_all:
        pd.concat(es_sf_all, ignore_index=True).to_csv(
            OUT / "tab_es_dose_semF_base2024.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(OUT / "tab_es_dose_semF_base2024.csv", index=False)

    del flows_semF, wage_semF
    gc.collect()
    _mem("after ES semF")

    print("=== ES dose-W base2024 saldo | cnae2 ===", flush=True)
    es_c2_all = []
    try:
        es_df, m, levels, fe_u = try_es(
            r2.run_es_dose_W, flows_c2, "saldo", 2024, es_cands, "ES_dose_cnae2"
        )
        if len(es_df):
            es_df = es_df.copy()
            es_df["fe"] = fe_u
            es_df["rob_spec"] = "cnae2"
            es_c2_all.append(es_df)
            note = "" if fe_u == FE_MAIN else f"FE used: {fe_u}"
            ok = plot_es_dose(
                es_df,
                "saldo",
                OUT / "fig_es_dose_cnae2_base2024_saldo.png",
                title_prefix="divisões CNAE 2 dígitos com listada",
                fe_note=note,
            )
            if ok:
                ES_PRODUCED.append("fig_es_dose_cnae2_base2024_saldo.png")
            es_notes.append(f"ES dose-W saldo cnae2 sob FE=`{fe_u}`.")
        del m
        gc.collect()
        for y in ["admitidos", "desligados"]:
            try:
                es_df, m, _, fe_u = try_es(
                    r2.run_es_dose_W, flows_c2, y, 2024, es_cands, "ES_dose_cnae2"
                )
                if len(es_df):
                    es_df = es_df.copy()
                    es_df["fe"] = fe_u
                    es_df["rob_spec"] = "cnae2"
                    es_c2_all.append(es_df)
                    note = "" if fe_u == FE_MAIN else f"FE used: {fe_u}"
                    ok = plot_es_dose(
                        es_df,
                        y,
                        OUT / f"fig_es_dose_cnae2_base2024_{y}.png",
                        title_prefix="divisões CNAE 2 dígitos com listada",
                        fe_note=note,
                    )
                    if ok:
                        ES_PRODUCED.append(f"fig_es_dose_cnae2_base2024_{y}.png")
                del m
                gc.collect()
            except MemoryError as e:
                FAILURES.append(f"ES cnae2/{y} OOM: {e}; skipping remaining cnae2 ES")
                gc.collect()
                break
        try:
            es_df, m, _, fe_u = try_es(
                r2.run_es_dose_W, wage_c2, "log_salario", 2024, es_cands, "ES_dose_cnae2"
            )
            if len(es_df):
                es_df = es_df.copy()
                es_df["fe"] = fe_u
                es_df["rob_spec"] = "cnae2"
                es_c2_all.append(es_df)
                note = "" if fe_u == FE_MAIN else f"FE used: {fe_u}"
                ok = plot_es_dose(
                    es_df,
                    "log_salario",
                    OUT / "fig_es_dose_cnae2_base2024_log_salario.png",
                    title_prefix="divisões CNAE 2 dígitos com listada",
                    fe_note=note,
                )
                if ok:
                    ES_PRODUCED.append("fig_es_dose_cnae2_base2024_log_salario.png")
            del m
            gc.collect()
        except MemoryError as e:
            FAILURES.append(f"ES cnae2/log_salario OOM: {e}")
            gc.collect()
    except MemoryError as e:
        FAILURES.append(f"ES cnae2/saldo OOM: {e}")
        es_notes.append("ES dose-W cnae2: OOM — não produzido.")
        gc.collect()

    if es_c2_all:
        pd.concat(es_c2_all, ignore_index=True).to_csv(
            OUT / "tab_es_dose_cnae2_base2024.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(OUT / "tab_es_dose_cnae2_base2024.csv", index=False)

    del flows_c2, wage_c2
    gc.collect()

    es_note = " ".join(es_notes) if es_notes else "Nenhum ES dose-W produzido."
    if ES_PRODUCED:
        es_note += " Figs: " + ", ".join(ES_PRODUCED) + "."

    write_texts(info, atts, es_note)

    # key_numbers
    def grab(df, term, outcome="saldo"):
        r = df.loc[(df.outcome == outcome) & (df.term == term)]
        if r.empty:
            return None, None, None, None
        return float(r.coef.iloc[0]), float(r.se.iloc[0]), float(r.p.iloc[0]), int(r.N.iloc[0])

    summary = {
        "fe_principal": FE_MAIN,
        "naming": {
            "principal": "FE saturada (FE_MAIN) — amostra completa",
            "semF": "sem seção F (construção)",
            "cnae2": "amostra divisões CNAE 2 dígitos com listada",
        },
        "principal_fonte": "copied_from_sat_main" if src.exists() else "reestimated",
        "es_produced": ES_PRODUCED,
        "es_fe_used": ES_FE_USED,
        "failures": FAILURES,
        "out_dir": str(OUT),
        "stage_dir": str(REMODELADO),
        "treated_divs_n": len(treated_div),
    }
    for sp, df in atts.items():
        for term in ["beta5", "beta10", "delta", "delta_over_5"]:
            c, s, p, n = grab(df, term)
            summary[f"{sp}_saldo_{term}"] = c
            summary[f"{sp}_saldo_{term}_se"] = s
            summary[f"{sp}_saldo_{term}_p"] = p
        _, _, _, n = grab(df, "beta5")
        summary[f"{sp}_saldo_N"] = n
        # all outcomes β5/β10/p briefly
        for y in OUTCOMES_ALL:
            c5, s5, p5, n = grab(df, "beta5", y)
            c10, s10, p10, _ = grab(df, "beta10", y)
            summary[f"{sp}_{y}_beta5"] = c5
            summary[f"{sp}_{y}_beta5_se"] = s5
            summary[f"{sp}_{y}_beta5_p"] = p5
            summary[f"{sp}_{y}_beta10"] = c10
            summary[f"{sp}_{y}_beta10_se"] = s10
            summary[f"{sp}_{y}_beta10_p"] = p10
            summary[f"{sp}_{y}_N"] = n

    (OUT / "key_numbers.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(
        "KEY saldo:",
        json.dumps(
            {k: summary[k] for k in summary if "saldo_beta" in k or k.endswith("_saldo_N")},
            indent=2,
        ),
        flush=True,
    )

    build_notebook()
    stage_remodelado()
    print("DONE", flush=True)
    print("FAILURES:", FAILURES, flush=True)
    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "run_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
