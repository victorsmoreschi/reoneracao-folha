#!/usr/bin/env python3
"""Resume: ES binary + T_max under FE_MAIN, then texts + stage REMODELADO_sat_main_v1."""
from __future__ import annotations

import gc
import json
import shutil
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/workspace/notebook_outputs/cgs_rodada2")
sys.path.insert(0, str(ROOT))
import run_cgs_rodada2 as r2

OUT = Path("/workspace/notebook_outputs/cgs_rodada2_sat_main")
REMODELADO = Path("/workspace/notebook_outputs/REMODELADO_sat_main_v1")
(REMODELADO / "tabs").mkdir(parents=True, exist_ok=True)
(REMODELADO / "figuras").mkdir(parents=True, exist_ok=True)

FE_MAIN = "cnae4^municipio^mes_cal + municipio^time_id + municipio^ano"
FE_LEAN = "cnae4^mes_cal + municipio + time_id"
FE_SAT_LEAN = "cnae4^mes_cal + municipio^time_id + municipio^ano"
OUTCOMES_FLOW = r2.OUTCOMES_FLOW
OUTCOMES_ALL = r2.OUTCOMES_ALL
FAILURES: list[str] = []
r2.FAILURES = FAILURES
ES_FE: dict[str, str] = {}


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


def try_es_bin(df, outcome, cands):
    for fe in cands:
        print(f"  Trying ES_bin/{outcome} FE={fe}", flush=True)
        try:
            es_df, m, levels = r2.run_es_binary(df, outcome, base_year=2024, fe=fe)
            if m is not None and es_df is not None and len(es_df):
                if fe != FE_MAIN:
                    FAILURES.append(f"ES_bin/{outcome}: FE_MAIN fail; used `{fe}`")
                ES_FE[outcome] = fe
                return es_df, m, fe
            del m
            gc.collect()
        except MemoryError as e:
            FAILURES.append(f"ES_bin/{outcome} MemoryError under `{fe}`: {e}")
            gc.collect()
        except Exception as e:
            FAILURES.append(f"ES_bin/{outcome} under `{fe}`: {type(e).__name__}: {e}")
            gc.collect()
    ES_FE[outcome] = "FAILED"
    return pd.DataFrame(), None, "FAILED"


def plot_es_binary(es_df, outcome, path, fe_note=""):
    sub = es_df.loc[es_df["outcome"] == outcome].sort_values("k")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="gray", ls="--", lw=0.8)
    ax.axvspan(-0.5, 11.5, color="C0", alpha=0.08)
    ax.axvspan(11.5, 18.5, color="C1", alpha=0.10)
    ax.errorbar(sub["k"], sub["coef"], yerr=1.96 * sub["se"], fmt="o", ms=3, capsize=2, color="C0")
    ax.set_xlabel("Event time k (k=0 at Jan/2025)")
    ax.set_ylabel("ATT (binary treated × month)")
    title = f"ES binary ({outcome}) | FE_MAIN; base 2024\nExposure timing; T_max on 2023 leads."
    if fe_note:
        title += f"\n[{fe_note}]"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_texts():
    info = json.loads(r2.PANEL_META.read_text()) if r2.PANEL_META.exists() else {}
    att_main = pd.read_csv(OUT / "tab_att_principal.csv")
    att_lean = pd.read_csv(OUT / "tab_att_lean_comparacao.csv")
    reta_df = pd.read_csv(OUT / "tab_reta_comparacao.csv")
    placebo_df = pd.read_csv(OUT / "tab_placebo2024.csv")
    tmax_df = pd.read_csv(OUT / "tab_tmax.csv")
    joint_df = pd.read_csv(OUT / "tab_joint.csv")
    tmax_rows = tmax_df.to_dict("records")
    joint_rows = joint_df.to_dict("records")

    def g(df, term, outcome="saldo"):
        r = df.loc[(df.outcome == outcome) & (df.term == term)]
        return None if r.empty else r.iloc[0]

    s5, s10, sd, sd5 = g(att_main, "beta5"), g(att_main, "beta10"), g(att_main, "delta"), g(att_main, "delta_over_5")
    tm = next((r for r in tmax_rows if r.get("outcome") == "saldo"), {})
    jt = next((r for r in joint_rows if r.get("outcome") == "saldo"), {})

    es_fe_note = "All ES dose-W ran under FE_MAIN. Binary: " + "; ".join(f"{k}→{v}" for k, v in sorted(ES_FE.items()))

    lines = []
    lines.append("# RESUMO — CGS rodada2 sat-main (FE_MAIN = PRINCIPAL)\n")
    lines.append("## Identificação\n")
    lines.append(f"- **PRINCIPAL** ATT: `Y ~ t2025 + t2026 | {FE_MAIN}` (CRV1 cnae4).")
    lines.append(f"- Comparação lean FE (antiga principal): `{FE_LEAN}`.")
    lines.append("- D_s = 1 se classe listada. Base omitida: 2023–2024. 2026 só até junho.")
    lines.append("- Reportamos Δ=β10−β5 e (β10−β5)/5 com EP clusterizado.")
    lines.append("- FE_MAIN **não** inclui class×year / class×mês civil / class×2025–26.")
    lines.append("- Reta Dose | FE_MAIN: comparação apenas.")
    lines.append("- ES principal: i(k_es)×W_sk | FE_MAIN. ES binário: timing; T_max = PT usual.")
    lines.append("- Joint θ_2023 dose-W = implicação comum; não-rejeição ≠ hipótese forte.\n")
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
        lines.append(f"- {r['outcome']}: coef={r['coef']:.6g} se={r['se']:.4g} p={r['p']:.4g} (comparison only)")

    lines.append("\n## Placebo 2024 | FE_MAIN\n")
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
            f"df={r.get('df')} fe={r.get('fe','')} — implicação comum; não-rejeição ≠ hipótese forte."
        )

    lines.append("\n## Caveats\n")
    lines.append("1. W_sk=5 no pré = escala do lead, não dose real.")
    lines.append("2. (β10−β5)/5 sob hipótese forte seria ACRT, mas com dinâmica 2026 **não** declaramos ACRT limpo.")
    lines.append("3. T_max binário = PT usual; joint θ_2023 = implicação conjunta, não aceite da forte.")
    lines.append("4. Não usamos β5/5 = β10/10 como teste da hipótese forte.")
    lines.append("5. Wage: pre-trends fracos — caveat mantido.\n")
    if FAILURES:
        lines.append("## Failures / FE fallbacks\n")
        for f in FAILURES:
            lines.append(f"- {f}")
        lines.append("")
    (OUT / "RESUMO.md").write_text("\n".join(lines), encoding="utf-8")

    b5, se5, p5 = float(s5.coef), float(s5.se), float(s5.p)
    b10, se10, p10 = float(s10.coef), float(s10.se), float(s10.p)
    dlt, sed, pdlt = float(sd.coef), float(sd.se), float(sd.p)
    d5, se_d5, p_d5 = float(sd5.coef), float(sd5.se), float(sd5.p)
    N = int(s5.N)
    tmax_p, tmax_v = tm.get("p", float("nan")), tm.get("T_max", float("nan"))
    joint_p, joint_w = jt.get("p", float("nan")), jt.get("wald_stat", float("nan"))

    texto = f"""# TEXTO_RESULTADOS — CGS sat-main (FE_MAIN = PRINCIPAL)

## 1. Especificação principal (contrastes anuais)

**Mudança de principal:** a FE saturada passa a ser a especificação **principal** (não mais robustez).

Estimamos Y_smt = FE_MAIN + β5 (D_s × 1{{2025}}) + β10 (D_s × 1{{2026}}) + ε,
com FE_MAIN = `{FE_MAIN}` e erros CRV1 por cnae4. Base omitida: 2023–2024; 2026 só até junho.
**Não** incluímos class×year, class×mês civil nem class×2025/2026 (absorveriam a dose).

Para **saldo** (principal): β5 = {b5:.4f} (EP = {se5:.4f}, p = {p5:.3f}), β10 = {b10:.4f} (EP = {se10:.4f}, p = {p10:.3f}), Δ = {dlt:.4f} (EP = {sed:.4f}, p = {pdlt:.3f}), (β10−β5)/5 = {d5:.4f} (EP = {se_d5:.4f}, p = {p_d5:.3f}), N = {N}.

A FE lean `{FE_LEAN}` (antiga principal) permanece apenas como **coluna de comparação**. A reta Y ~ Dose | FE_MAIN é também comparação (não preferida).

## 2. Estudo de eventos

ES **principal**: `i(k_es)×W_sk` | FE_MAIN (W=0/5/10). Base 2024 omitida. θ_k = efeito por ponto. W=5 no pré = escala do lead.
ES binário | FE_MAIN: timing de exposição. Base-2023 dose-W: checagem de antecipação.
ES FE note: {es_fe_note}

## 3. Interpretação de (β10 − β5)/5 sob a hipótese forte (obrigatório)

Sob a hipótese forte de tendências paralelas em doses (CGS, Teorema 3.3(d)), a quantidade (β10 − β5)/5 = {d5:.4f} (EP = {se_d5:.4f}) identifica o ACRT — o efeito causal médio de um ponto adicional de dose entre 5 e 10. Sem a hipótese forte, o mesmo objeto é a soma de um termo de incremento de dose e um termo de seleção (Teorema 3.2). A razão para invocar a interpretação forte neste desenho é institucional: a Lei 14.973/2024 atribui mecanicamente dose 5 e depois dose 10 às **mesmas** classes listadas; não há duas subpopulações listadas escolhendo doses diferentes no mesmo mês civil — o termo de seleção, portanto, não tem um grupo observável em que se apoiar. **Caveat decisivo:** 2026 já carrega um ano inteiro sob dose 5; dinâmicas de ajuste, estoque e composição permanecem. Por isso reportamos (β10 − β5)/5 com o parágrafo da hipótese forte, mas **não** declaramos ACRT limpo nem validamos a hipótese forte / SPT. Em particular, **não** usamos o teste β5/5 = β10/10 como prova da hipótese forte.

## 4. Checagens de pré-tendência

- **Nível (PT usual):** T_max studentizado nos leads de 2023 do ES **binário** (base 2024). Para saldo, T_max = {tmax_v} (p = {tmax_p}). Isso é o teste usual de parallel trends em Y(0), não SPT.
- **Inclinação (implicação conjunta):** Wald conjunto θ_k = 0 em 2023 no ES dose-W. Para saldo, Wald = {joint_w} (p = {joint_p}). Rejeitar conta contra uma implicação comum; **não rejeitar não confirma** a hipótese forte.
- **Placebo 2024:** amostra ≤202412; Y ~ t2024 | FE_MAIN (timing falso; não valida SPT).

## 5. Avisos finais

Wage: pre-trends frágeis — manter cautela. Não destruir `cgs_rodada2/`. Não editar Overleaf. Pack staged em `REMODELADO_sat_main_v1/`.
"""
    (OUT / "TEXTO_RESULTADOS.md").write_text(texto, encoding="utf-8")
    print("Wrote RESUMO + TEXTO", flush=True)

    # key_numbers
    summary = {
        "saldo_beta5": b5, "saldo_beta5_se": se5, "saldo_beta5_p": p5,
        "saldo_beta10": b10, "saldo_beta10_se": se10, "saldo_beta10_p": p10,
        "saldo_delta": dlt, "saldo_delta_se": sed, "saldo_delta_p": pdlt,
        "saldo_delta_over_5": d5, "saldo_delta_over_5_se": se_d5, "saldo_delta_over_5_p": p_d5,
        "saldo_N": N,
        "tmax_saldo": tmax_v, "tmax_p_saldo": tmax_p,
        "joint_theta_pre_wald_saldo": joint_w, "joint_theta_pre_p_saldo": joint_p,
        "fe_principal": FE_MAIN, "fe_lean_comparison": FE_LEAN,
        "es_binary_fe_used": ES_FE,
        "es_dose_fe": FE_MAIN,
        "failures": FAILURES,
        "out_dir": str(OUT),
        "stage_dir": str(REMODELADO),
    }
    for y in OUTCOMES_ALL:
        r = placebo_df.loc[placebo_df.outcome == y]
        if not r.empty:
            summary[f"placebo_{y}_coef"] = float(r.coef.iloc[0])
            summary[f"placebo_{y}_se"] = float(r.se.iloc[0])
            summary[f"placebo_{y}_p"] = float(r.p.iloc[0])
            summary[f"placebo_{y}_N"] = int(r.N.iloc[0])
    (OUT / "key_numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("KEY saldo:", json.dumps({k: summary[k] for k in summary if "saldo" in k or k.startswith("fe_") or k.startswith("tmax") or k.startswith("joint")}, indent=2), flush=True)
    return summary


def stage():
    tabs = [
        "tab_att_principal.csv", "tab_att_principal_wide_for_tex.csv",
        "tab_att_lean_comparacao.csv", "tab_placebo2024.csv",
        "tab_tmax.csv", "tab_joint.csv", "tab_reta_comparacao.csv",
        "tab_es_dose_base2024.csv", "tab_es_binario_base2024.csv", "tab_es_dose_base2023.csv",
    ]
    for name in tabs:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, REMODELADO / "tabs" / name)
    for src in OUT.glob("fig_*.png"):
        shutil.copy2(src, REMODELADO / "figuras" / src.name)
    for name in ["RESUMO.md", "TEXTO_RESULTADOS.md", "key_numbers.json", "run_cgs_sat_main.py", "resume_es_binario.py"]:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, REMODELADO / name)
    (REMODELADO / "README_ASSETS.txt").write_text(
        "REMODELADO_sat_main_v1 — FE_MAIN = PRINCIPAL (saturated FE promoted)\n\n"
        f"FE_MAIN = `{FE_MAIN}`\n"
        f"FE_LEAN (comparison) = `{FE_LEAN}`\n\n"
        "Source: /workspace/notebook_outputs/cgs_rodada2_sat_main/\n"
        "Does NOT overwrite cgs_rodada2 / REMODELADO_paper_v2 / REMODELADO_expand.\n"
        "Do not edit Overleaf live from here.\n",
        encoding="utf-8",
    )
    print(f"Staged to {REMODELADO}", flush=True)


def main():
    print("=== RESUME ES binary + T_max under FE_MAIN ===", flush=True)
    cands = [FE_MAIN, FE_SAT_LEAN, FE_LEAN]
    flow_cols = [
        "cnae4", "municipio", "secao", "competencia", "ano", "mes_cal", "time_id",
        "k", "tratado", "dose", "t2025", "t2026", "saldo", "admitidos", "desligados",
    ]
    flows = pd.read_parquet(r2.PANEL_BAL, columns=flow_cols)
    wage = pd.read_parquet(r2.PANEL_WAGE)
    flows["W"] = r2.assign_W(flows["tratado"], flows["ano"])
    wage["W"] = r2.assign_W(wage["tratado"], wage["ano"])
    r2._mem("loaded")

    es_bin_all = []
    tmax_rows = []
    for y in OUTCOMES_FLOW:
        es_df, m, fe_u = try_es_bin(flows, y, cands)
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
        r2._mem(f"after bin {y}")

    # free flows before wage if needed
    del flows
    gc.collect()

    es_df, m, fe_u = try_es_bin(wage, "log_salario", cands)
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
    del m, wage
    gc.collect()

    es_bin = pd.concat(es_bin_all, ignore_index=True) if es_bin_all else pd.DataFrame()
    es_bin.to_csv(OUT / "tab_es_binario_base2024.csv", index=False)
    tmax_out = []
    for r in tmax_rows:
        rr = {k: v for k, v in r.items() if k not in ("pre_names", "pre_t")}
        rr["pre_names"] = json.dumps(r.get("pre_names", []))
        tmax_out.append(rr)
    pd.DataFrame(tmax_out).to_csv(OUT / "tab_tmax.csv", index=False)

    for y in OUTCOMES_ALL:
        fe_u = ES_FE.get(y, "")
        note = "" if fe_u == FE_MAIN else f"FE used: {fe_u}"
        plot_es_binary(es_bin, y, OUT / f"fig_es_binario_base2024_{y}.png", fe_note=note)

    write_texts()
    stage()
    print("DONE resume", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "resume_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
