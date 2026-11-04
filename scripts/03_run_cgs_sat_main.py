#!/usr/bin/env python3
"""CGS rodada2 sat-main — PROMOTE saturated FE to PRINCIPAL.

FE_MAIN = cnae4^municipio^mes_cal + municipio^time_id + municipio^ano
FE_LEAN = cnae4^mes_cal + municipio + time_id  (old principal → comparison only)

Reuses panels + helpers from cgs_rodada2/run_cgs_rodada2.py.
OUT = cgs_rodada2_sat_main/ (does NOT destroy cgs_rodada2).
vcov CRV1 cnae4. If FE_MAIN OOMs on ES, fall back to leaner and document.
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

OUT = Path("/workspace/notebook_outputs/cgs_rodada2_sat_main")
OUT.mkdir(parents=True, exist_ok=True)
REMODELADO = Path("/workspace/notebook_outputs/REMODELADO_expand_sat")
(REMODELADO / "tabs").mkdir(parents=True, exist_ok=True)
(REMODELADO / "figuras").mkdir(parents=True, exist_ok=True)

FE_MAIN = "cnae4^municipio^mes_cal + municipio^time_id + municipio^ano"
FE_LEAN = "cnae4^mes_cal + municipio + time_id"  # old principal → comparison
# Intermediate leaner saturated (no triple): for ES OOM fallback
FE_SAT_LEAN = "cnae4^mes_cal + municipio^time_id + municipio^ano"

OUTCOMES_FLOW = r2.OUTCOMES_FLOW
OUTCOMES_ALL = r2.OUTCOMES_ALL
PANEL_BAL = r2.PANEL_BAL
PANEL_WAGE = r2.PANEL_WAGE
PANEL_META = r2.PANEL_META

# Redirect rodada2 FAILURES list usage
FAILURES: list[str] = []
r2.FAILURES = FAILURES
r2.OUT = OUT  # so any leftover writes go here if called

ES_FE_USED: dict[str, str] = {}  # outcome -> fe actually used for ES


def _mem(tag: str = "") -> None:
    r2._mem(tag)


def fit(fml, data, label=""):
    return r2.fit(fml, data, label)


def pre_names_from_model(m, k_lo=-24, k_hi=-13):
    names = []
    if m is None:
        return names
    for n in m.coef().index:
        s = str(n)
        if "k_es::" not in s:
            continue
        try:
            kk = int(s.split("k_es::")[1].split(":")[0])
        except Exception:
            continue
        if k_lo <= kk <= k_hi:
            names.append(s)
    return names


def try_es(runner, df, outcome, base_year, fe_candidates, tag):
    """Try FE candidates in order; return (es_df, m, levels, fe_used)."""
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


def plot_es_dose(es_df, outcome, path, fe_note=""):
    sub = es_df.loc[es_df["outcome"] == outcome].sort_values("k")
    if sub.empty:
        return
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
        f"ES dose-W ({outcome}) — PRINCIPAL | FE_MAIN; base 2024 omitted\n"
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


def plot_es_binary(es_df, outcome, path, fe_note=""):
    sub = es_df.loc[es_df["outcome"] == outcome].sort_values("k")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="gray", ls="--", lw=0.8)
    ax.axvspan(-0.5, 11.5, color="C0", alpha=0.08)
    ax.axvspan(11.5, 18.5, color="C1", alpha=0.10)
    ax.errorbar(
        sub["k"], sub["coef"], yerr=1.96 * sub["se"],
        fmt="o", ms=3, capsize=2, color="C0",
    )
    ax.set_xlabel("Event time k (k=0 at Jan/2025)")
    ax.set_ylabel("ATT (binary treated × month)")
    title = (
        f"ES binary ({outcome}) | FE_MAIN; base 2024\n"
        "Exposure timing; used for usual PT / T_max on 2023 leads."
    )
    if fe_note:
        title += f"\n[{fe_note}]"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def wide_att(att_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in OUTCOMES_ALL:
        sub = att_df.loc[att_df.outcome == y]
        if sub.empty:
            continue

        def g(term):
            r = sub.loc[sub.term == term]
            if r.empty:
                return None
            return r.iloc[0]

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


def write_texts(info, att_main, att_lean, reta_df, placebo_df, tmax_rows, joint_rows, es_fe_note):
    def g(df, term, outcome="saldo"):
        r = df.loc[(df.outcome == outcome) & (df.term == term)]
        return None if r.empty else r.iloc[0]

    s5 = g(att_main, "beta5")
    s10 = g(att_main, "beta10")
    sd = g(att_main, "delta")
    sd5 = g(att_main, "delta_over_5")
    tm = next((r for r in tmax_rows if r.get("outcome") == "saldo"), {})
    jt = next((r for r in joint_rows if r.get("outcome") == "saldo"), {})

    lines = []
    lines.append("# RESUMO — CGS rodada2 sat-main (FE_MAIN = PRINCIPAL)\n")
    lines.append("## Identificação\n")
    lines.append(
        f"- **PRINCIPAL** ATT: `Y ~ t2025 + t2026 | {FE_MAIN}` (CRV1 cnae4)."
    )
    lines.append(
        f"- Comparação lean FE (antiga principal): `{FE_LEAN}`."
    )
    lines.append(
        "- D_s = 1 se classe listada (time-invariant). Base omitida: 2023–2024. "
        "2026 só até junho. β5 = 2025 vs base; β10 = 2026 vs **mesma** base (não vs 2025)."
    )
    lines.append(
        "- Reportamos Δ=β10−β5 e (β10−β5)/5 com EP clusterizado do contraste."
    )
    lines.append(
        "- FE_MAIN **não** inclui class×year / class×mês civil / class×2025–26 (absorveriam dose)."
    )
    lines.append(
        "- Reta linear em Dose | FE_MAIN: **coluna de comparação apenas**."
    )
    lines.append(
        "- ES **principal**: `i(k_es)×W_sk` | FE_MAIN (W=0/5/10). Pré W=5 = escala do lead."
    )
    lines.append(
        "- ES binário | FE_MAIN: timing de exposição; T_max nos leads 2023 = PT usual."
    )
    lines.append(
        "- Joint θ_k=0 em 2023 no ES dose-W = implicação comum; não-rejeição ≠ hipótese forte.\n"
    )
    if es_fe_note:
        lines.append(f"- **ES FE note:** {es_fe_note}\n")

    lines.append("## Painel\n")
    lines.append(f"- Pares: **{info.get('n_pairs')}**; N flows: **{info.get('N_flows')}**")
    lines.append(f"- Competências: {info.get('comps_first')}–{info.get('comps_last')}")
    lines.append(f"- NEVER_DROP: {info.get('never_drop')}")
    lines.append(f"- Wage drop months: {info.get('salary_drop_months')}\n")

    lines.append("## Contrastes ATT PRINCIPAL (FE_MAIN)\n")
    lines.append("| outcome | β5 | se | p | β10 | se | p | Δ | se | p | Δ/5 | se | p | N |")
    lines.append("|---------|----|----|---|-----|----|---|---|----|---|-----|----|---|---|")
    for y in OUTCOMES_ALL:
        r5, r10, rd, rd5 = g(att_main, "beta5", y), g(att_main, "beta10", y), g(att_main, "delta", y), g(att_main, "delta_over_5", y)
        if r5 is None:
            continue
        lines.append(
            f"| {y} | {r5.coef:.4f} | {r5.se:.4f} | {r5.p:.4f} | "
            f"{r10.coef:.4f} | {r10.se:.4f} | {r10.p:.4f} | "
            f"{rd.coef:.4f} | {rd.se:.4f} | {rd.p:.4f} | "
            f"{rd5.coef:.4f} | {rd5.se:.4f} | {rd5.p:.4f} | {int(r5.N)} |"
        )

    lines.append("\n## Comparação lean FE (antiga principal)\n")
    lines.append("| outcome | β5 | se | p | β10 | se | p | Δ | se | p | Δ/5 | se | p | N |")
    lines.append("|---------|----|----|---|-----|----|---|---|----|---|-----|----|---|---|")
    for y in OUTCOMES_ALL:
        r5, r10, rd, rd5 = g(att_lean, "beta5", y), g(att_lean, "beta10", y), g(att_lean, "delta", y), g(att_lean, "delta_over_5", y)
        if r5 is None:
            continue
        lines.append(
            f"| {y} | {r5.coef:.4f} | {r5.se:.4f} | {r5.p:.4f} | "
            f"{r10.coef:.4f} | {r10.se:.4f} | {r10.p:.4f} | "
            f"{rd.coef:.4f} | {rd.se:.4f} | {rd.p:.4f} | "
            f"{rd5.coef:.4f} | {rd5.se:.4f} | {rd5.p:.4f} | {int(r5.N)} |"
        )

    lines.append("\n## Reta comparação (Dose linear | FE_MAIN)\n")
    for _, r in reta_df.iterrows():
        lines.append(
            f"- {r['outcome']}: coef={r['coef']:.6g} se={r['se']:.4g} p={r['p']:.4g} "
            "(comparison only; not preferred)"
        )

    lines.append("\n## Placebo 2024 | FE_MAIN (sample ≤202412; Y ~ t2024)\n")
    lines.append("| outcome | coef | se | p | N |")
    lines.append("|---------|------|----|---|---|")
    for y in OUTCOMES_ALL:
        r = placebo_df.loc[placebo_df.outcome == y]
        if r.empty:
            continue
        rr = r.iloc[0]
        lines.append(f"| {y} | {rr.coef:.6g} | {rr.se:.6g} | {rr.p:.4g} | {int(rr.N)} |")

    lines.append("\n## T_max PT usual (ES binário base2024, k∈2023)\n")
    for r in tmax_rows:
        lines.append(
            f"- {r.get('outcome')}: T_max={r.get('T_max')} p={r.get('p')} "
            f"crit5%={r.get('crit_5pct')} n_pre={r.get('n_pre')} fe={r.get('fe','')} — PT usual, não SPT."
        )

    lines.append("\n## Joint θ_pre dose-W (2023)\n")
    for r in joint_rows:
        lines.append(
            f"- {r.get('outcome')}: Wald={r.get('wald_stat')} p={r.get('p')} "
            f"df={r.get('df')} fe={r.get('fe','')} — implicação comum; não-rejeição ≠ aceitar hipótese forte."
        )

    lines.append("\n## Caveats\n")
    lines.append(
        "1. W_sk=5 no pré é **escala do lead**, não dose real."
    )
    lines.append(
        "2. (β10−β5)/5 sob hipótese forte (CGS Thm 3.3(d)) seria ACRT, mas com dinâmica de 2026 "
        "(já um ano em dose 5) **não** declaramos ACRT limpo / hipótese validada."
    )
    lines.append(
        "3. T_max no ES binário = PT usual; joint θ_2023 no ES dose-W = implicação conjunta, não aceite da forte."
    )
    lines.append("4. Não usamos β5/5 = β10/10 como teste da hipótese forte.")
    lines.append("5. Wage: pre-trends já fracos — manter caveat.\n")

    if FAILURES:
        lines.append("## Failures / FE fallbacks\n")
        for f in FAILURES:
            lines.append(f"- {f}")
        lines.append("")

    (OUT / "RESUMO.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote RESUMO.md", flush=True)

    # TEXTO
    if s5 is not None and sd5 is not None:
        b5, se5, p5 = float(s5.coef), float(s5.se), float(s5.p)
        b10, se10, p10 = float(s10.coef), float(s10.se), float(s10.p)
        dlt, sed, pdlt = float(sd.coef), float(sd.se), float(sd.p)
        d5, se_d5, p_d5 = float(sd5.coef), float(sd5.se), float(sd5.p)
        N = int(s5.N)
        tmax_p = tm.get("p", float("nan"))
        tmax_v = tm.get("T_max", float("nan"))
        joint_p = jt.get("p", float("nan"))
        joint_w = jt.get("wald_stat", float("nan"))
    else:
        b5 = se5 = p5 = b10 = se10 = p10 = dlt = sed = pdlt = d5 = se_d5 = p_d5 = N = float("nan")
        tmax_p = tmax_v = joint_p = joint_w = float("nan")

    texto = f"""# TEXTO_RESULTADOS — CGS sat-main (FE_MAIN = PRINCIPAL)

## 1. Especificação principal (contrastes anuais)

**Mudança de principal:** a FE saturada passa a ser a especificação **principal** (não mais robustez).

Estimamos
Y_smt = FE_MAIN + β5 (D_s × 1{{2025}}) + β10 (D_s × 1{{2026}}) + ε,
com FE_MAIN = `{FE_MAIN}` e erros CRV1 por cnae4. D_s = 1 se a classe CNAE está listada (invariante no tempo). Base omitida: 2023–2024; 2026 só até junho. β5 = ATT 2025 vs base; β10 = ATT 2026 vs a **mesma** base. Também reportamos Δ = β10 − β5 e (β10 − β5)/5.

**Não** incluímos class×year, class×mês civil nem class×2025/2026 (absorveriam a dose).

Para **saldo** (principal): β5 = {b5:.4f} (EP = {se5:.4f}, p = {p5:.3f}), β10 = {b10:.4f} (EP = {se10:.4f}, p = {p10:.3f}), Δ = {dlt:.4f} (EP = {sed:.4f}, p = {pdlt:.3f}), (β10−β5)/5 = {d5:.4f} (EP = {se_d5:.4f}, p = {p_d5:.3f}), N = {N}.

A FE lean `{FE_LEAN}` (antiga principal) permanece apenas como **coluna de comparação**. A reta Y ~ Dose | FE_MAIN é também comparação (impõe linearidade; não preferida).

## 2. Estudo de eventos

ES **principal**: `i(k_es)×W_sk` | FE_MAIN, com W=0 (controle), W=5 (listada 2023–2025), W=10 (listada 2026). Base 2024 omitida. θ_k = efeito por ponto; 5θ ≈ nível 2025; 10θ ≈ nível 2026. W=5 no pré = **escala do lead**, não dose calendário.

ES binário | FE_MAIN: timing de exposição (não principal). Base-2023 dose-W: checagem de antecipação.
{('ES FE note: ' + es_fe_note) if es_fe_note else ''}

## 3. Interpretação de (β10 − β5)/5 sob a hipótese forte (obrigatório)

Sob a hipótese forte de tendências paralelas em doses (CGS, Teorema 3.3(d)), a quantidade (β10 − β5)/5 = {d5:.4f} (EP = {se_d5:.4f}) identifica o ACRT — o efeito causal médio de um ponto adicional de dose entre 5 e 10. Sem a hipótese forte, o mesmo objeto é a soma de um termo de incremento de dose e um termo de seleção (Teorema 3.2). A razão para invocar a interpretação forte neste desenho é institucional: a Lei 14.973/2024 atribui mecanicamente dose 5 e depois dose 10 às **mesmas** classes listadas; não há duas subpopulações listadas escolhendo doses diferentes no mesmo mês civil — o termo de seleção, portanto, não tem um grupo observável em que se apoiar. **Caveat decisivo:** 2026 já carrega um ano inteiro sob dose 5; dinâmicas de ajuste, estoque e composição permanecem. Por isso reportamos (β10 − β5)/5 com o parágrafo da hipótese forte, mas **não** declaramos ACRT limpo nem validamos a hipótese forte / SPT. Em particular, **não** usamos o teste β5/5 = β10/10 como prova da hipótese forte.

## 4. Checagens de pré-tendência

- **Nível (PT usual):** T_max studentizado nos leads de 2023 do ES **binário** (base 2024). Para saldo, T_max = {tmax_v} (p = {tmax_p}). Isso é o teste usual de parallel trends em Y(0), não SPT.
- **Inclinação (implicação conjunta):** Wald conjunto θ_k = 0 em 2023 no ES dose-W. Para saldo, Wald = {joint_w} (p = {joint_p}). Rejeitar conta contra uma implicação comum; **não rejeitar não confirma** a hipótese forte.
- **Placebo 2024:** amostra ≤202412; Y ~ t2024 | FE_MAIN (timing falso; não valida SPT).

## 5. Avisos finais

Wage: pre-trends já frágeis — manter cautela. Não destruir `cgs_rodada2/`. Não editar Overleaf nesta rodada; artefatos copiados para `REMODELADO_expand_sat/` para o Overleaf Guy.
"""
    (OUT / "TEXTO_RESULTADOS.md").write_text(texto, encoding="utf-8")
    print("Wrote TEXTO_RESULTADOS.md", flush=True)


def stage_remodelado():
    tabs = [
        "tab_att_principal.csv",
        "tab_att_principal_wide_for_tex.csv",
        "tab_att_lean_comparacao.csv",
        "tab_placebo2024.csv",
        "tab_tmax.csv",
        "tab_joint.csv",
        "tab_reta_comparacao.csv",
        "tab_es_dose_base2024.csv",
        "tab_es_binario_base2024.csv",
        "tab_es_dose_base2023.csv",
    ]
    figs = list(OUT.glob("fig_*.png"))
    for name in tabs:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, REMODELADO / "tabs" / name)
    for src in figs:
        shutil.copy2(src, REMODELADO / "figuras" / src.name)
    for name in ["RESUMO.md", "TEXTO_RESULTADOS.md", "key_numbers.json", "run_cgs_sat_main.py"]:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, REMODELADO / name)
    readme = (
        "REMODELADO_expand_sat — FE_MAIN = PRINCIPAL (saturated FE promoted)\n\n"
        f"FE_MAIN = `{FE_MAIN}`\n"
        f"FE_LEAN (comparison) = `{FE_LEAN}`\n\n"
        "tabs/: att_principal (+ wide_for_tex), att_lean_comparacao, placebo2024,\n"
        "       tmax, joint, reta_comparacao, es_dose/binario base2024 (+ base2023)\n"
        "figuras/: ES dose-W + binary under FE_MAIN (or documented leaner fallback)\n"
        "RESUMO + TEXTO + key_numbers.json\n\n"
        "Do NOT destroy cgs_rodada2/. Do not edit Overleaf live from here.\n"
    )
    (REMODELADO / "README_ASSETS.txt").write_text(readme, encoding="utf-8")
    print(f"Staged to {REMODELADO}", flush=True)


def main():
    print("=== CGS sat-main: FE_MAIN as PRINCIPAL ===", flush=True)
    print(f"FE_MAIN = {FE_MAIN}", flush=True)
    print(f"FE_LEAN (comparison) = {FE_LEAN}", flush=True)
    info = json.loads(PANEL_META.read_text()) if PANEL_META.exists() else {}

    flow_cols = [
        "cnae4", "municipio", "secao", "competencia", "ano", "mes_cal", "time_id",
        "k", "tratado", "dose", "t2025", "t2026", "saldo", "admitidos", "desligados",
    ]
    print("Loading panels...", flush=True)
    flows = pd.read_parquet(PANEL_BAL, columns=flow_cols)
    wage = pd.read_parquet(PANEL_WAGE)
    flows["W"] = r2.assign_W(flows["tratado"], flows["ano"])
    wage["W"] = r2.assign_W(wage["tratado"], wage["ano"])
    print(f"flows N={len(flows)}; wage N={len(wage)}", flush=True)
    _mem("loaded")

    # ---- 1. ATT PRINCIPAL under FE_MAIN ----
    print("=== ATT PRINCIPAL (FE_MAIN) ===", flush=True)
    att_rows = []
    for y in OUTCOMES_FLOW:
        rows, m = r2.run_att(flows, y, FE_MAIN, "att_principal")
        att_rows.extend(rows)
        del m
        gc.collect()
    rows, m = r2.run_att(wage, "log_salario", FE_MAIN, "att_principal")
    att_rows.extend(rows)
    del m
    gc.collect()
    att_main = pd.DataFrame(att_rows)
    att_main.to_csv(OUT / "tab_att_principal.csv", index=False)
    wide = wide_att(att_main)
    wide.to_csv(OUT / "tab_att_principal_wide_for_tex.csv", index=False)
    print(wide.to_string(index=False), flush=True)
    _mem("after ATT main")

    # ---- 2. Lean FE comparison ----
    print("=== ATT lean comparison (old principal) ===", flush=True)
    lean_rows = []
    for y in OUTCOMES_FLOW:
        rows, m = r2.run_att(flows, y, FE_LEAN, "att_lean_comparacao")
        lean_rows.extend(rows)
        del m
        gc.collect()
    rows, m = r2.run_att(wage, "log_salario", FE_LEAN, "att_lean_comparacao")
    lean_rows.extend(rows)
    del m
    gc.collect()
    att_lean = pd.DataFrame(lean_rows)
    att_lean.to_csv(OUT / "tab_att_lean_comparacao.csv", index=False)

    # ---- 3. Linear dose | FE_MAIN (comparison) ----
    print("=== Reta Dose | FE_MAIN (comparison) ===", flush=True)
    reta_rows = []
    for y in OUTCOMES_FLOW:
        reta_rows.extend(r2.run_reta(flows, y, FE_MAIN))
        gc.collect()
    reta_rows.extend(r2.run_reta(wage, "log_salario", FE_MAIN))
    # retag note
    for r in reta_rows:
        r["note"] = "comparison column only; Dose linear | FE_MAIN; NOT preferred"
        r["spec"] = "reta_comparacao_FE_MAIN"
    reta_df = pd.DataFrame(reta_rows)
    reta_df.to_csv(OUT / "tab_reta_comparacao.csv", index=False)

    # ---- 8. Placebo 2024 | FE_MAIN ----
    print("=== Placebo 2024 | FE_MAIN ===", flush=True)

    def prep_pre(df):
        if "competencia" in df.columns:
            comp = df["competencia"].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
            d = df.loc[comp <= 202412].copy()
        else:
            d = df.loc[df["ano"] <= 2024].copy()
        d["t2024"] = ((d["ano"].astype(int) == 2024) & (d["tratado"].astype(int) == 1)).astype(np.int8)
        return d

    flows_pre = prep_pre(flows)
    wage_pre = prep_pre(wage)
    plac_rows = []
    for y in OUTCOMES_FLOW:
        m = fit(f"{y} ~ t2024 | {FE_MAIN}", flows_pre, f"placebo/{y}")
        if m is not None:
            t = m.tidy()
            if "t2024" in t.index:
                r = t.loc["t2024"]
                plac_rows.append({
                    "outcome": y, "spec": "placebo2024", "fe": FE_MAIN, "term": "t2024",
                    "coef": float(r["Estimate"]), "se": float(r["Std. Error"]),
                    "t": float(r["t value"]), "p": float(r["Pr(>|t|)"]),
                    "ci_lo": float(r["2.5%"]), "ci_hi": float(r["97.5%"]),
                    "N": r2.model_N(m),
                    "note": "timing placebo ordinary PT; sample<=202412; NOT SPT",
                })
        del m
        gc.collect()
    m = fit(f"log_salario ~ t2024 | {FE_MAIN}", wage_pre, "placebo/log_salario")
    if m is not None:
        t = m.tidy()
        if "t2024" in t.index:
            r = t.loc["t2024"]
            plac_rows.append({
                "outcome": "log_salario", "spec": "placebo2024", "fe": FE_MAIN, "term": "t2024",
                "coef": float(r["Estimate"]), "se": float(r["Std. Error"]),
                "t": float(r["t value"]), "p": float(r["Pr(>|t|)"]),
                "ci_lo": float(r["2.5%"]), "ci_hi": float(r["97.5%"]),
                "N": r2.model_N(m),
                "note": "timing placebo ordinary PT; sample<=202412; NOT SPT",
            })
    del m
    gc.collect()
    del flows_pre, wage_pre
    gc.collect()
    placebo_df = pd.DataFrame(plac_rows)
    placebo_df.to_csv(OUT / "tab_placebo2024.csv", index=False)

    # ---- ES candidates: MAIN → SAT_LEAN → LEAN ----
    es_cands = [FE_MAIN, FE_SAT_LEAN, FE_LEAN]

    # ---- 4. ES dose-W principal | FE_MAIN ----
    print("=== ES dose-W PRINCIPAL base2024 | FE_MAIN (w/ fallback) ===", flush=True)
    es_dose_all = []
    joint_rows = []
    for y in OUTCOMES_FLOW:
        es_df, m, levels, fe_u = try_es(r2.run_es_dose_W, flows, y, 2024, es_cands, "ES_dose")
        if len(es_df):
            es_df = es_df.copy()
            es_df["fe"] = fe_u
            es_dose_all.append(es_df)
        pre = pre_names_from_model(m)
        jt = r2.joint_theta_pre(m, pre)
        jt["outcome"] = y
        jt["spec"] = "ES_dose_W_base2024"
        jt["fe"] = fe_u
        joint_rows.append(jt)
        del m
        gc.collect()
        _mem(f"after ES dose {y}")

    es_df, m, levels, fe_u = try_es(r2.run_es_dose_W, wage, "log_salario", 2024, es_cands, "ES_dose")
    if len(es_df):
        es_df = es_df.copy()
        es_df["fe"] = fe_u
        es_dose_all.append(es_df)
    pre = pre_names_from_model(m)
    jt = r2.joint_theta_pre(m, pre)
    jt["outcome"] = "log_salario"
    jt["spec"] = "ES_dose_W_base2024"
    jt["fe"] = fe_u
    joint_rows.append(jt)
    del m
    gc.collect()

    es_dose = pd.concat([e for e in es_dose_all if len(e)], ignore_index=True) if es_dose_all else pd.DataFrame()
    es_dose.to_csv(OUT / "tab_es_dose_base2024.csv", index=False)
    joint_out = []
    for r in joint_rows:
        rr = {k: v for k, v in r.items() if k != "pre_names"}
        rr["pre_names"] = json.dumps(r.get("pre_names", []))
        joint_out.append(rr)
    pd.DataFrame(joint_out).to_csv(OUT / "tab_joint.csv", index=False)

    for y in OUTCOMES_ALL:
        fe_u = ES_FE_USED.get(f"ES_dose/{y}", "")
        note = "" if fe_u == FE_MAIN else f"FE used: {fe_u}"
        plot_es_dose(es_dose, y, OUT / f"fig_es_dose_base2024_{y}.png", fe_note=note)

    # ---- 9. ES dose-W base2023 anticipation ----
    print("=== ES dose-W base2023 anticipation | FE_MAIN (w/ fallback) ===", flush=True)
    es23 = []
    for y in OUTCOMES_FLOW:
        es_df, m, _, fe_u = try_es(r2.run_es_dose_W, flows, y, 2023, es_cands, "ES_dose23")
        if len(es_df):
            es_df = es_df.copy()
            es_df["fe"] = fe_u
            es23.append(es_df)
        del m
        gc.collect()
    es_df, m, _, fe_u = try_es(r2.run_es_dose_W, wage, "log_salario", 2023, es_cands, "ES_dose23")
    if len(es_df):
        es_df = es_df.copy()
        es_df["fe"] = fe_u
        es23.append(es_df)
    del m
    gc.collect()
    if es23:
        pd.concat(es23, ignore_index=True).to_csv(OUT / "tab_es_dose_base2023.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT / "tab_es_dose_base2023.csv", index=False)

    # ---- 5-6. ES binary + T_max ----
    print("=== ES binary base2024 + T_max | FE_MAIN (w/ fallback) ===", flush=True)
    es_bin_all = []
    tmax_rows = []
    for y in OUTCOMES_FLOW:
        es_df, m, levels, fe_u = try_es(r2.run_es_binary, flows, y, 2024, es_cands, "ES_bin")
        if len(es_df):
            es_df = es_df.copy()
            es_df["fe"] = fe_u
            es_bin_all.append(es_df)
        pre = pre_names_from_model(m)
        tm = r2.studentized_max_test(m, pre)
        tm["outcome"] = y
        tm["spec"] = "ES_binary_base2024"
        tm["fe"] = fe_u
        tmax_rows.append(tm)
        del m
        gc.collect()

    es_df, m, levels, fe_u = try_es(r2.run_es_binary, wage, "log_salario", 2024, es_cands, "ES_bin")
    if len(es_df):
        es_df = es_df.copy()
        es_df["fe"] = fe_u
        es_bin_all.append(es_df)
    pre = pre_names_from_model(m)
    tm = r2.studentized_max_test(m, pre)
    tm["outcome"] = "log_salario"
    tm["spec"] = "ES_binary_base2024"
    tm["fe"] = fe_u
    tmax_rows.append(tm)
    del m
    gc.collect()

    es_bin = pd.concat([e for e in es_bin_all if len(e)], ignore_index=True) if es_bin_all else pd.DataFrame()
    es_bin.to_csv(OUT / "tab_es_binario_base2024.csv", index=False)
    tmax_out = []
    for r in tmax_rows:
        rr = {k: v for k, v in r.items() if k not in ("pre_names", "pre_t")}
        rr["pre_names"] = json.dumps(r.get("pre_names", []))
        tmax_out.append(rr)
    pd.DataFrame(tmax_out).to_csv(OUT / "tab_tmax.csv", index=False)

    for y in OUTCOMES_ALL:
        fe_u = ES_FE_USED.get(f"ES_bin/{y}", "")
        note = "" if fe_u == FE_MAIN else f"FE used: {fe_u}"
        plot_es_binary(es_bin, y, OUT / f"fig_es_binario_base2024_{y}.png", fe_note=note)

    # ES FE summary note
    uniq_fe = sorted(set(ES_FE_USED.values()))
    if uniq_fe == [FE_MAIN]:
        es_fe_note = f"All ES ran under FE_MAIN=`{FE_MAIN}`."
    else:
        es_fe_note = "ES FE by outcome: " + "; ".join(f"{k}→{v}" for k, v in sorted(ES_FE_USED.items()))

    write_texts(info, att_main, att_lean, reta_df, placebo_df, tmax_rows, joint_rows, es_fe_note)

    # key_numbers from PRINCIPAL (saldo)
    def grab(term):
        r = att_main.loc[(att_main.outcome == "saldo") & (att_main.term == term)]
        return float(r.coef.iloc[0]), float(r.se.iloc[0]), float(r.p.iloc[0])

    b5, se5, p5 = grab("beta5")
    b10, se10, p10 = grab("beta10")
    dlt, sed, pdlt = grab("delta")
    d5, sed5, pd5 = grab("delta_over_5")
    tm = next(r for r in tmax_rows if r.get("outcome") == "saldo")
    jt = next(r for r in joint_rows if r.get("outcome") == "saldo")
    plac_s = placebo_df.loc[placebo_df.outcome == "saldo"]
    summary = {
        "saldo_beta5": b5,
        "saldo_beta5_se": se5,
        "saldo_beta5_p": p5,
        "saldo_beta10": b10,
        "saldo_beta10_se": se10,
        "saldo_beta10_p": p10,
        "saldo_delta": dlt,
        "saldo_delta_se": sed,
        "saldo_delta_p": pdlt,
        "saldo_delta_over_5": d5,
        "saldo_delta_over_5_se": sed5,
        "saldo_delta_over_5_p": pd5,
        "saldo_N": int(att_main.loc[(att_main.outcome == "saldo") & (att_main.term == "beta5"), "N"].iloc[0]),
        "tmax_saldo": tm.get("T_max"),
        "tmax_p_saldo": tm.get("p"),
        "joint_theta_pre_wald_saldo": jt.get("wald_stat"),
        "joint_theta_pre_p_saldo": jt.get("p"),
        "fe_principal": FE_MAIN,
        "fe_lean_comparison": FE_LEAN,
        "es_fe_used": ES_FE_USED,
        "failures": FAILURES,
        "out_dir": str(OUT),
    }
    if not plac_s.empty:
        summary["placebo_saldo_coef"] = float(plac_s.coef.iloc[0])
        summary["placebo_saldo_se"] = float(plac_s.se.iloc[0])
        summary["placebo_saldo_p"] = float(plac_s.p.iloc[0])
        summary["placebo_saldo_N"] = int(plac_s.N.iloc[0])
    for y in OUTCOMES_ALL:
        r = placebo_df.loc[placebo_df.outcome == y]
        if not r.empty:
            summary[f"placebo_{y}_coef"] = float(r.coef.iloc[0])
            summary[f"placebo_{y}_se"] = float(r.se.iloc[0])
            summary[f"placebo_{y}_p"] = float(r.p.iloc[0])
            summary[f"placebo_{y}_N"] = int(r.N.iloc[0])

    (OUT / "key_numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("KEY saldo principal:", json.dumps({k: summary[k] for k in summary if k.startswith("saldo_") or k.startswith("tmax_") or k.startswith("joint_") or k.startswith("fe_")}, indent=2), flush=True)

    stage_remodelado()
    print("DONE", flush=True)
    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "run_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
