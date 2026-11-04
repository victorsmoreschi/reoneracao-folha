#!/usr/bin/env python3
"""Resume: ES dose-W principal under FE_MAIN for semF + cnae2 (4 outcomes each).

ATT already done in tab_att_*.csv. Does NOT touch old packs.
One sample in memory at a time to avoid OOM (~12GB peak FE_MAIN ES).
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
# Prefer FE_MAIN only; leaner only if MemoryError (document)
FE_FALLBACKS = [
    FE_MAIN,
    "cnae4^mes_cal + municipio^time_id + municipio^ano",
    "cnae4^mes_cal + municipio + time_id",
]

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


def _mem(tag=""):
    r2._mem(tag)


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
        sub["k"], sub["coef"], yerr=1.96 * sub["se"],
        fmt="o", ms=3.5, capsize=2, color="black", ecolor="gray",
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


def run_one_es(df, outcome, tag, title_prefix, fig_stem):
    """Run ES dose-W base2024; prefer FE_MAIN. Return es_df or empty."""
    last_err = None
    for fe in FE_FALLBACKS:
        print(f"  Trying {tag}/{outcome} FE={fe}", flush=True)
        try:
            es_df, m, levels = r2.run_es_dose_W(df, outcome, base_year=2024, fe=fe)
            if m is not None and es_df is not None and len(es_df):
                if fe != FE_MAIN:
                    FAILURES.append(f"{tag}/{outcome}: FE_MAIN fail/OOM; used `{fe}`")
                ES_FE_USED[f"{tag}/{outcome}"] = fe
                es_df = es_df.copy()
                es_df["fe"] = fe
                es_df["rob_spec"] = tag
                note = "" if fe == FE_MAIN else f"FE used: {fe}"
                fig_path = OUT / f"fig_es_dose_{fig_stem}_base2024_{outcome}.png"
                if plot_es_dose(es_df, outcome, fig_path, title_prefix=title_prefix, fe_note=note):
                    ES_PRODUCED.append(fig_path.name)
                del m
                gc.collect()
                _mem(f"after ES {tag}/{outcome}")
                return es_df
            del m
            gc.collect()
            last_err = f"null under {fe}"
        except MemoryError as e:
            last_err = str(e)
            FAILURES.append(f"{tag}/{outcome} MemoryError under `{fe}`: {e}")
            gc.collect()
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            FAILURES.append(f"{tag}/{outcome} under `{fe}`: {last_err}")
            gc.collect()
    FAILURES.append(f"{tag}/{outcome}: ALL FE failed ({last_err})")
    ES_FE_USED[f"{tag}/{outcome}"] = "FAILED"
    return pd.DataFrame()


def load_semF():
    flow_cols = [
        "cnae4", "municipio", "secao", "competencia", "ano", "mes_cal", "time_id",
        "k", "tratado", "dose", "t2025", "t2026", "saldo", "admitidos", "desligados",
    ]
    flows = pd.read_parquet(PANEL_BAL, columns=flow_cols)
    wage = pd.read_parquet(PANEL_WAGE)
    flows = flows.loc[flows["secao"].astype(str) != "F"].copy()
    wage = wage.loc[wage["secao"].astype(str) != "F"].copy()
    flows["W"] = r2.assign_W(flows["tratado"], flows["ano"])
    wage["W"] = r2.assign_W(wage["tratado"], wage["ano"])
    print(f"semF flows={len(flows)} wage={len(wage)}", flush=True)
    return flows, wage


def load_cnae2():
    flow_cols = [
        "cnae4", "municipio", "secao", "competencia", "ano", "mes_cal", "time_id",
        "k", "tratado", "dose", "t2025", "t2026", "saldo", "admitidos", "desligados",
    ]
    flows = pd.read_parquet(PANEL_BAL, columns=flow_cols)
    wage = pd.read_parquet(PANEL_WAGE)
    flows["cnae2"] = flows["cnae4"].astype(str).str[:2]
    wage["cnae2"] = wage["cnae4"].astype(str).str[:2]
    treated_div = set(flows.loc[flows["tratado"] == 1, "cnae2"].astype(str).unique())
    flows = flows.loc[flows["cnae2"].astype(str).isin(treated_div)].copy()
    wage = wage.loc[wage["cnae2"].astype(str).isin(treated_div)].copy()
    flows["W"] = r2.assign_W(flows["tratado"], flows["ano"])
    wage["W"] = r2.assign_W(wage["tratado"], wage["ano"])
    print(f"cnae2 divs={len(treated_div)} flows={len(flows)} wage={len(wage)}", flush=True)
    return flows, wage, len(treated_div)


def g(df, term, outcome="saldo"):
    if df is None or df.empty:
        return None
    r = df.loc[(df.outcome == outcome) & (df.term == term)]
    return None if r.empty else r.iloc[0]


def write_texts(info, atts, es_note, treated_divs_n):
    labels = {
        "principal": "Principal (amostra completa)",
        "semF": "Sem seção F (construção)",
        "cnae2": "Amostra divisões CNAE 2 dígitos com listada",
    }
    lines = []
    lines.append("# RESUMO — CGS sat-main robustez (FE saturada = FE_MAIN)\n")
    lines.append("## Identificação\n")
    lines.append(
        f"- **FE saturada (FE_MAIN):** `Y ~ t2025 + t2026 | {FE_MAIN}` (CRV1 cnae4)."
    )
    lines.append(
        "- ES principal: `i(k_es)×W_sk` | FE_MAIN (W=0/5/10), base 2024 omitida. "
        "Mesma especificação do ES principal da sat_main."
    )
    lines.append(
        "- D_s binário (tratado). Base omitida ATT: 2023–2024. β5, β10, Δ, Δ/5 com se e **p**."
    )
    lines.append(
        "- FE_MAIN **não** inclui classe×ano / classe×time_id / classe×2025–26. "
        "UF×time omitido (colinear).\n"
    )
    lines.append("## Robustez (mesma FE_MAIN)\n")
    lines.append("1. **Principal** — amostra completa (fonte: sat_main `tab_att_principal.csv`).")
    lines.append("2. **Sem seção F (construção)** — dropar `secao == \"F\"`.")
    lines.append(
        "3. **Amostra divisões CNAE 2 dígitos com listada** — "
        f"`cnae2` em divisão com ≥1 classe listada (n_divs={treated_divs_n}).\n"
    )
    lines.append("## Painel\n")
    lines.append(f"- Pares: **{info.get('n_pairs')}**; N flows: **{info.get('N_flows')}**")
    lines.append(f"- Competências: {info.get('comps_first')}–{info.get('comps_last')}")
    lines.append(f"- NEVER_DROP: {info.get('never_drop')}")
    lines.append(f"- Wage drop months: {info.get('salary_drop_months')}\n")

    for sp, lab in labels.items():
        df = atts.get(sp)
        lines.append(f"## ATT — {lab}\n")
        lines.append("| outcome | β5 | se | p | β10 | se | p | Δ | se | p | Δ/5 | se | p | N |")
        lines.append("|---------|----|----|---|-----|----|---|---|----|---|-----|----|---|---|")
        if df is None or df.empty:
            lines.append("| _(vazio)_ | | | | | | | | | | | | | |\n")
            continue
        for y in OUTCOMES_ALL:
            r5 = g(df, "beta5", y)
            r10 = g(df, "beta10", y)
            rd = g(df, "delta", y)
            rd5 = g(df, "delta_over_5", y)
            if r5 is None:
                continue
            lines.append(
                f"| {y} | {r5.coef:.4f} | {r5.se:.4f} | {r5.p:.4f} | "
                f"{r10.coef:.4f} | {r10.se:.4f} | {r10.p:.4f} | "
                f"{rd.coef:.4f} | {rd.se:.4f} | {rd.p:.4f} | "
                f"{rd5.coef:.4f} | {rd5.se:.4f} | {rd5.p:.4f} | {int(r5.N)} |"
            )
        lines.append("")

    lines.append("## ES dose-W (principal; base 2024)\n")
    lines.append(es_note)
    lines.append("")
    lines.append("ES FE por outcome:")
    for k, v in sorted(ES_FE_USED.items()):
        lines.append(f"- {k} → `{v}`")
    lines.append("")
    if ES_PRODUCED:
        lines.append("Figs: " + ", ".join(ES_PRODUCED))
        lines.append("")

    lines.append("## Caveats\n")
    lines.append("1. Mesma FE saturada em ATT e ES de robustez.")
    lines.append("2. (β10−β5)/5 sob hipótese forte seria ACRT; com dinâmica 2026 **não** declaramos ACRT limpo.")
    lines.append("3. Wage: pre-trends frágeis — caveat mantido.")
    lines.append("4. Não sobrescrever packs antigos.\n")
    if FAILURES:
        lines.append("## Failures / FE fallbacks\n")
        for f in FAILURES:
            lines.append(f"- {f}")
        lines.append("")
    (OUT / "RESUMO.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote RESUMO.md", flush=True)

    def fmt_saldo(sp):
        df = atts.get(sp)
        s5, s10 = g(df, "beta5"), g(df, "beta10")
        if s5 is None:
            return f"**{labels[sp]}:** (vazio)."
        return (
            f"**{labels[sp]}:** β5 = {float(s5.coef):.4f} (EP = {float(s5.se):.4f}, p = {float(s5.p):.3f}), "
            f"β10 = {float(s10.coef):.4f} (EP = {float(s10.se):.4f}, p = {float(s10.p):.3f}), "
            f"N = {int(s5.N)}."
        )

    texto = f"""# TEXTO_RESULTADOS — CGS sat-main robustez (FE saturada)

## 1. Especificação

**FE saturada (FE_MAIN):** `Y ~ t2025 + t2026 | {FE_MAIN}` (CRV1 cnae4).
D_s binário; base omitida 2023–2024; β5, β10, Δ, Δ/5 com se e p.
ES principal: `i(k_es)×W_sk` | FE_MAIN, base 2024 omitida (igual sat_main).

## 2. Robustez

{fmt_saldo("principal")}
{fmt_saldo("semF")}
{fmt_saldo("cnae2")}

## 3. ES dose-W

{es_note}

Figs: {", ".join(ES_PRODUCED) if ES_PRODUCED else "(nenhuma)"}

## 4. Artefatos

Pasta: `{OUT}`
Stage: `{REMODELADO}`
Não editar Overleaf .tex; não sobrescrever packs antigos.
"""
    (OUT / "TEXTO_RESULTADOS.md").write_text(texto, encoding="utf-8")
    print("Wrote TEXTO_RESULTADOS.md", flush=True)


def build_notebook():
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    cells = []
    cells.append(nbf.v4.new_markdown_cell(
        "# Reoneração CGS — robustez sob FE saturada (FE_MAIN)\n\n"
        f"**FE_MAIN** = `{FE_MAIN}`\n\n"
        "ATT + ES dose-W (`i(k)×W_sk`, base 2024) para principal / sem seção F / "
        "divisões CNAE 2 dígitos com listada. 4 outcomes."
    ))
    cells.append(nbf.v4.new_code_cell("""\
import gc, sys
from pathlib import Path
import pandas as pd
ROOT = Path("/workspace/notebook_outputs/cgs_rodada2")
OUT = Path("/workspace/notebook_outputs/cgs_rodada2_sat_main_rob")
SAT = Path("/workspace/notebook_outputs/cgs_rodada2_sat_main")
sys.path.insert(0, str(ROOT))
import run_cgs_rodada2 as r2
FE_MAIN = "cnae4^municipio^mes_cal + municipio^time_id + municipio^ano"
OUTCOMES_FLOW, OUTCOMES_ALL = r2.OUTCOMES_FLOW, r2.OUTCOMES_ALL
print(FE_MAIN)
"""))
    cells.append(nbf.v4.new_code_cell("""\
# Carregar painel
flow_cols = ["cnae4","municipio","secao","competencia","ano","mes_cal","time_id",
             "k","tratado","dose","t2025","t2026","saldo","admitidos","desligados"]
flows = pd.read_parquet(r2.PANEL_BAL, columns=flow_cols)
wage = pd.read_parquet(r2.PANEL_WAGE)
print(len(flows), len(wage))
"""))
    cells.append(nbf.v4.new_markdown_cell("## ATT principal / sem F / cnae2 (CSVs já gerados pelo script)"))
    cells.append(nbf.v4.new_code_cell("""\
att_principal = pd.read_csv(OUT / "tab_att_principal.csv")
att_semF = pd.read_csv(OUT / "tab_att_rob_semF.csv")
att_cnae2 = pd.read_csv(OUT / "tab_att_rob_cnae2.csv")
wide = pd.read_csv(OUT / "tab_att_rob_wide_for_tex.csv")
wide
"""))
    cells.append(nbf.v4.new_markdown_cell(
        "## ES dose-W sob FE_MAIN (sem F e cnae2)\n\n"
        "`Y ~ i(k_es, ref):W | FE_MAIN`, base 2024. Ver `resume_es_dose_rob.py` / tabs ES."
    ))
    cells.append(nbf.v4.new_code_cell("""\
# Exemplo: reestimar ES dose-W saldo sem F (pode exigir ~12GB)
# flows_semF = flows.loc[flows.secao.astype(str)!="F"].copy()
# flows_semF["W"] = r2.assign_W(flows_semF["tratado"], flows_semF["ano"])
# es, m, lv = r2.run_es_dose_W(flows_semF, "saldo", base_year=2024, fe=FE_MAIN)
es_semF = pd.read_csv(OUT / "tab_es_dose_semF_base2024.csv") if (OUT/"tab_es_dose_semF_base2024.csv").exists() else None
es_cnae2 = pd.read_csv(OUT / "tab_es_dose_cnae2_base2024.csv") if (OUT/"tab_es_dose_cnae2_base2024.csv").exists() else None
print("ES semF rows:", 0 if es_semF is None else len(es_semF))
print("ES cnae2 rows:", 0 if es_cnae2 is None else len(es_cnae2))
es_semF.head() if es_semF is not None and len(es_semF) else None
"""))
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python (.venv_cgs)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "pygments_lexer": "ipython3"}}
    path = OUT / "notebook_reoneracao_cgs_sat_main_rob.ipynb"
    nbf.write(nb, path)
    print(f"Wrote {path}", flush=True)
    return path


def stage():
    tabs = [
        "tab_att_principal.csv", "tab_att_rob_semF.csv", "tab_att_rob_cnae2.csv",
        "tab_att_rob_wide_for_tex.csv",
        "tab_es_dose_semF_base2024.csv", "tab_es_dose_cnae2_base2024.csv",
    ]
    for name in tabs:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, REMODELADO / "tabs" / name)
    for src in OUT.glob("fig_*.png"):
        shutil.copy2(src, REMODELADO / "figuras" / src.name)
    for name in ["RESUMO.md", "TEXTO_RESULTADOS.md", "key_numbers.json",
                 "run_cgs_sat_main_rob.py", "resume_es_dose_rob.py",
                 "notebook_reoneracao_cgs_sat_main_rob.ipynb"]:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, REMODELADO / name)
    readme = (
        "REMODELADO_sat_main_rob_v1 — robustez sob FE saturada (FE_MAIN)\n\n"
        f"FE_MAIN = `{FE_MAIN}`\n"
        "ATT: principal | sem seção F | divisões CNAE 2 dígitos com listada\n"
        "ES dose-W principal (i(k)×W, base2024) sob FE_MAIN para semF e cnae2 × 4 Y\n\n"
        "tabs/ + figuras/ + RESUMO + TEXTO + key_numbers + notebook\n"
        "Do NOT overwrite sat_main / paper_v3–v4 / beamer. Do not edit Overleaf live.\n"
    )
    (REMODELADO / "README_ASSETS.txt").write_text(readme, encoding="utf-8")
    print(f"Staged {REMODELADO}", flush=True)


def main():
    print("=== RESUME ES dose-W under FE_MAIN (semF + cnae2, 4Y each) ===", flush=True)
    info = json.loads(PANEL_META.read_text()) if PANEL_META.exists() else {}
    atts = {
        "principal": pd.read_csv(OUT / "tab_att_principal.csv"),
        "semF": pd.read_csv(OUT / "tab_att_rob_semF.csv"),
        "cnae2": pd.read_csv(OUT / "tab_att_rob_cnae2.csv"),
    }
    print("ATT tabs loaded", flush=True)

    # ---- semF ES ----
    print("=== ES dose-W sem seção F (4 outcomes) ===", flush=True)
    flows, wage = load_semF()
    _mem("semF loaded")
    es_sf = []
    for y in OUTCOMES_FLOW:
        es_df = run_one_es(flows, y, "semF", "sem seção F (construção)", "semF")
        if len(es_df):
            es_sf.append(es_df)
        gc.collect()
    del flows
    gc.collect()
    es_df = run_one_es(wage, "log_salario", "semF", "sem seção F (construção)", "semF")
    if len(es_df):
        es_sf.append(es_df)
    del wage
    gc.collect()
    if es_sf:
        pd.concat(es_sf, ignore_index=True).to_csv(OUT / "tab_es_dose_semF_base2024.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT / "tab_es_dose_semF_base2024.csv", index=False)
    print(f"semF ES outcomes done: {len(es_sf)}", flush=True)
    _mem("after semF ES freed")

    # ---- cnae2 ES ----
    print("=== ES dose-W cnae2 (4 outcomes) ===", flush=True)
    flows, wage, n_div = load_cnae2()
    _mem("cnae2 loaded")
    es_c2 = []
    for y in OUTCOMES_FLOW:
        es_df = run_one_es(flows, y, "cnae2", "divisões CNAE 2 dígitos com listada", "cnae2")
        if len(es_df):
            es_c2.append(es_df)
        gc.collect()
    del flows
    gc.collect()
    es_df = run_one_es(wage, "log_salario", "cnae2", "divisões CNAE 2 dígitos com listada", "cnae2")
    if len(es_df):
        es_c2.append(es_df)
    del wage
    gc.collect()
    if es_c2:
        pd.concat(es_c2, ignore_index=True).to_csv(OUT / "tab_es_dose_cnae2_base2024.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT / "tab_es_dose_cnae2_base2024.csv", index=False)
    print(f"cnae2 ES outcomes done: {len(es_c2)}", flush=True)

    es_note = (
        f"ES dose-W base2024 sob FE saturada para semF ({len(es_sf)}/4 Y) e "
        f"cnae2 ({len(es_c2)}/4 Y). Spec: i(k_es)×W_sk | FE_MAIN."
    )
    write_texts(info, atts, es_note, n_div)

    # key_numbers
    def grab(df, term, outcome="saldo"):
        r = df.loc[(df.outcome == outcome) & (df.term == term)]
        if r.empty:
            return None, None, None, None
        return float(r.coef.iloc[0]), float(r.se.iloc[0]), float(r.p.iloc[0]), int(r.N.iloc[0])

    summary = {
        "fe_principal": FE_MAIN,
        "principal_fonte": "copied_from_sat_main",
        "es_produced": ES_PRODUCED,
        "es_fe_used": ES_FE_USED,
        "failures": FAILURES,
        "out_dir": str(OUT),
        "stage_dir": str(REMODELADO),
        "treated_divs_n": n_div,
        "es_semF_n_outcomes": len(es_sf),
        "es_cnae2_n_outcomes": len(es_c2),
    }
    for sp, df in atts.items():
        for term in ["beta5", "beta10", "delta", "delta_over_5"]:
            c, s, p, n = grab(df, term)
            summary[f"{sp}_saldo_{term}"] = c
            summary[f"{sp}_saldo_{term}_se"] = s
            summary[f"{sp}_saldo_{term}_p"] = p
        _, _, _, n = grab(df, "beta5")
        summary[f"{sp}_saldo_N"] = n
    (OUT / "key_numbers.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    build_notebook()
    stage()
    print("DONE", flush=True)
    print("ES_PRODUCED:", ES_PRODUCED, flush=True)
    print("ES_FE_USED:", ES_FE_USED, flush=True)
    print("FAILURES:", FAILURES, flush=True)
    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "resume_es_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
