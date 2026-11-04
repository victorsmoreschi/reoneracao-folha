#!/usr/bin/env python3
"""CGS rodada2 — Moreschi reoneração Lei 14.973/2024.

Main: Y ~ t2025 + t2026 | FE  (β5, β10 vs base 2023–24; report Δ=β10−β5 and Δ/5)
Saturated FE side-by-side (FEs that do NOT absorb dose).
Linear Dose = comparison column only.
ES PRINCIPAL: i(k_es)×W_sk with W=0 control; W=5 listed 2023–25; W=10 listed 2026.
Binary tratado×k ES = exposure timing (beside, not principal).
T_max on binary ES 2023 leads = usual PT.
Joint θ_k=0 in 2023 on dose-W ES = common implication of both hyps (not SPT acceptance).

FE_BASE = cnae4^mes_cal + municipio + time_id; vcov CRV1 cnae4.
"""
from __future__ import annotations

import gc
import json
import shutil
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyfixest.estimation import feols

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUT = Path("/workspace/notebook_outputs/cgs_rodada2")
OUT.mkdir(parents=True, exist_ok=True)
STAGING = OUT / "drive_staging"
STAGING.mkdir(parents=True, exist_ok=True)
CRISTINE = Path("/workspace/notebook_outputs/cristine_callaway")
PANEL_BAL = CRISTINE / "painel_balanced_flows.parquet"
PANEL_WAGE = CRISTINE / "painel_wage.parquet"
PANEL_META = CRISTINE / "panel_meta.json"

FE_BASE = "cnae4^mes_cal + municipio + time_id"
# Lean saturated first (brief OOM fallback order). Skip UF×time when muni×time present.
FE_SAT_LEAN = "cnae4^mes_cal + municipio^time_id + municipio^ano"
FE_SAT_FULL = "cnae4^municipio^mes_cal + municipio^time_id + municipio^ano"
VCOV = {"CRV1": "cnae4"}
RNG = np.random.default_rng(20250301)
OUTCOMES_FLOW = ["saldo", "admitidos", "desligados"]
OUTCOMES_ALL = OUTCOMES_FLOW + ["log_salario"]

FAILURES: list[str] = []
FE_SAT_USED: str | None = None


def _mem(tag: str = "") -> None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"[mem] {tag}: maxrss={rss/1e6:.2f} GB", flush=True)
    except Exception:
        pass


def make_kes(k: pd.Series, base_year: int) -> tuple[pd.Series, int]:
    """Omit FULL base year by collapsing its months to a single ref k."""
    if base_year == 2024:
        lo, hi, ref = -12, -1, -12
    elif base_year == 2023:
        lo, hi, ref = -24, -13, -24
    else:
        raise ValueError(base_year)
    kes = k.astype(np.int16).copy()
    kes = kes.where(~((kes >= lo) & (kes <= hi)), other=np.int16(ref))
    return kes, ref


def assign_W(tratado: pd.Series, ano: pd.Series) -> pd.Series:
    """Assigned dose W_sk for dose-attributed ES.
    controls: 0; listed 2023–2025: 5; listed 2026: 10.
    Pre W=5 is LEAD SCALING (normalization), not calendar dose.
    """
    t = tratado.astype(np.int8)
    a = ano.astype(np.int16)
    w = np.where(t == 0, 0, np.where(a >= 2026, 10, 5)).astype(np.int8)
    return pd.Series(w, index=tratado.index, name="W")


def fit(fml: str, data: pd.DataFrame, label: str = ""):
    print(f"  FIT {label}: {fml}", flush=True)
    _mem(label + " before")
    try:
        return feols(
            fml,
            data=data,
            vcov=VCOV,
            lean=True,
            copy_data=False,
            store_data=False,
        )
    except Exception as e:
        msg = f"FAIL {label}: {type(e).__name__}: {e}"
        print(f"  {msg}", flush=True)
        FAILURES.append(msg)
        return None


def model_N(m) -> int:
    if m is None:
        return 0
    for attr in ("_N", "N", "nobs"):
        if hasattr(m, attr):
            try:
                return int(getattr(m, attr))
            except Exception:
                pass
    return 0


def contrast_delta(m, name5: str = "t2025", name10: str = "t2026") -> dict:
    """β5, β10, Δ=β10−β5, Δ/5 with CRV1 SEs from VCOV."""
    if m is None:
        return {}
    coef = m.coef()
    idx = list(coef.index)
    if name5 not in idx or name10 not in idx:
        return {"error": f"missing {name5}/{name10} in {idx}"}
    i5, i10 = idx.index(name5), idx.index(name10)
    b5 = float(coef[name5])
    b10 = float(coef[name10])
    V = np.asarray(m._vcov, dtype=float)
    se5 = float(np.sqrt(max(V[i5, i5], 0.0)))
    se10 = float(np.sqrt(max(V[i10, i10], 0.0)))
    var_d = V[i10, i10] + V[i5, i5] - 2.0 * V[i10, i5]
    se_d = float(np.sqrt(max(var_d, 0.0)))
    d = b10 - b5
    d5 = d / 5.0
    se_d5 = se_d / 5.0

    def _tp(c, s):
        if s <= 0:
            return float("nan"), float("nan")
        t = c / s
        # normal approx two-sided
        from math import erfc, sqrt

        p = float(erfc(abs(t) / sqrt(2.0)))
        return float(t), p

    t5, p5 = _tp(b5, se5)
    t10, p10 = _tp(b10, se10)
    td, pd_ = _tp(d, se_d)
    td5, pd5 = _tp(d5, se_d5)
    tidy = m.tidy()
    # prefer pyfixest p/t for the two main coefs when available
    if name5 in tidy.index:
        t5 = float(tidy.loc[name5, "t value"])
        p5 = float(tidy.loc[name5, "Pr(>|t|)"])
    if name10 in tidy.index:
        t10 = float(tidy.loc[name10, "t value"])
        p10 = float(tidy.loc[name10, "Pr(>|t|)"])
    return {
        "beta5": b5,
        "se5": se5,
        "t5": t5,
        "p5": p5,
        "beta10": b10,
        "se10": se10,
        "t10": t10,
        "p10": p10,
        "delta": d,
        "se_delta": se_d,
        "t_delta": td,
        "p_delta": pd_,
        "delta_over_5": d5,
        "se_delta_over_5": se_d5,
        "t_delta_over_5": td5,
        "p_delta_over_5": pd5,
        "N": model_N(m),
    }


def rows_from_contrast(c: dict, outcome: str, spec: str, fe: str) -> list[dict]:
    if not c or "beta5" not in c:
        return []
    out = []
    mapping = [
        ("beta5", "beta5", "se5", "t5", "p5"),
        ("beta10", "beta10", "se10", "t10", "p10"),
        ("delta", "delta", "se_delta", "t_delta", "p_delta"),
        ("delta_over_5", "delta_over_5", "se_delta_over_5", "t_delta_over_5", "p_delta_over_5"),
    ]
    for term, ck, sk, tk, pk in mapping:
        out.append(
            {
                "outcome": outcome,
                "spec": spec,
                "fe": fe,
                "term": term,
                "coef": c[ck],
                "se": c[sk],
                "t": c[tk],
                "p": c[pk],
                "N": c["N"],
            }
        )
    return out


def run_att(df: pd.DataFrame, outcome: str, fe: str, tag: str) -> tuple[list[dict], object]:
    fml = f"{outcome} ~ t2025 + t2026 | {fe}"
    m = fit(fml, df, f"{tag}/{outcome}")
    c = contrast_delta(m)
    return rows_from_contrast(c, outcome, tag, fe), m


def run_reta(df: pd.DataFrame, outcome: str, fe: str) -> list[dict]:
    fml = f"{outcome} ~ dose | {fe}"
    m = fit(fml, df, f"reta/{outcome}")
    rows = []
    if m is not None:
        t = m.tidy()
        if "dose" in t.index:
            r = t.loc["dose"]
            rows.append(
                {
                    "outcome": outcome,
                    "spec": "reta_comparacao",
                    "fe": fe,
                    "term": "dose",
                    "coef": float(r["Estimate"]),
                    "se": float(r["Std. Error"]),
                    "t": float(r["t value"]),
                    "p": float(r["Pr(>|t|)"]),
                    "N": model_N(m),
                    "note": "comparison column only; imposes linearity; NOT preferred",
                }
            )
    del m
    gc.collect()
    return rows


def run_es_dose_W(df: pd.DataFrame, outcome: str, base_year: int = 2024, fe: str = FE_BASE):
    """Principal ES: θ_k per-point via 1{t=t*+k}·W_sk."""
    kes, ref = make_kes(df["k"], base_year)
    d = df.copy(deep=False)
    d["k_es"] = kes
    if "W" not in d.columns:
        d["W"] = assign_W(d["tratado"], d["ano"])
    fml = f"{outcome} ~ i(k_es, ref={ref}):W | {fe}"
    m = fit(fml, d, f"ES_dose_W_b{base_year}/{outcome}")
    rows = []
    levels = []
    if m is not None:
        t = m.tidy()
        for name in t.index:
            s = str(name)
            if "k_es::" not in s:
                continue
            try:
                kk = int(s.split("k_es::")[1].split(":")[0])
            except Exception:
                continue
            levels.append(kk)
            r = t.loc[name]
            rows.append(
                {
                    "outcome": outcome,
                    "spec": f"ES_dose_W_base{base_year}",
                    "k": kk,
                    "term": s,
                    "coef": float(r["Estimate"]),
                    "se": float(r["Std. Error"]),
                    "t": float(r["t value"]),
                    "p": float(r["Pr(>|t|)"]),
                    "ci_lo": float(r["2.5%"]),
                    "ci_hi": float(r["97.5%"]),
                    "ref_k": ref,
                    "base_year": base_year,
                    "N": model_N(m),
                    "note": "θ_k per-point; pre W=5 is lead scaling not calendar dose; 5θ≈level 2025; 10θ≈level 2026",
                }
            )
    return pd.DataFrame(rows), m, sorted(levels)


def run_es_binary(df: pd.DataFrame, outcome: str, base_year: int = 2024, fe: str = FE_BASE):
    kes, ref = make_kes(df["k"], base_year)
    d = df.copy(deep=False)
    d["k_es"] = kes
    fml = f"{outcome} ~ i(k_es, tratado, ref={ref}) | {fe}"
    m = fit(fml, d, f"ES_bin_b{base_year}/{outcome}")
    rows = []
    levels = []
    if m is not None:
        t = m.tidy()
        for name in t.index:
            s = str(name)
            if "k_es::" not in s:
                continue
            try:
                kk = int(s.split("k_es::")[1].split(":")[0])
            except Exception:
                continue
            levels.append(kk)
            r = t.loc[name]
            rows.append(
                {
                    "outcome": outcome,
                    "spec": f"ES_binary_base{base_year}",
                    "k": kk,
                    "term": s,
                    "coef": float(r["Estimate"]),
                    "se": float(r["Std. Error"]),
                    "t": float(r["t value"]),
                    "p": float(r["Pr(>|t|)"]),
                    "ci_lo": float(r["2.5%"]),
                    "ci_hi": float(r["97.5%"]),
                    "ref_k": ref,
                    "base_year": base_year,
                    "N": model_N(m),
                    "note": "exposure timing; not principal",
                }
            )
    return pd.DataFrame(rows), m, sorted(levels)


def studentized_max_test(m, pre_names: list[str], n_draws: int = 10_000) -> dict:
    if m is None or not pre_names:
        return {"error": "no model/pre"}
    t = m.tidy()
    names = [n for n in pre_names if n in t.index]
    if len(names) < 1:
        return {"error": "no pre coefs", "looked_for": pre_names[:5]}
    tvals = np.array([float(t.loc[n, "t value"]) for n in names], dtype=float)
    T_obs = float(np.max(np.abs(tvals)))
    coef_idx = list(m.coef().index)
    ix = [coef_idx.index(n) for n in names]
    V = np.asarray(m._vcov, dtype=float)
    Vsub = V[np.ix_(ix, ix)]
    se = np.sqrt(np.clip(np.diag(Vsub), 1e-18, None))
    corr = Vsub / np.outer(se, se)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)
    try:
        Z = RNG.multivariate_normal(mean=np.zeros(len(names)), cov=corr, size=n_draws)
    except np.linalg.LinAlgError:
        corr2 = corr + np.eye(len(names)) * 1e-8
        Z = RNG.multivariate_normal(mean=np.zeros(len(names)), cov=corr2, size=n_draws)
    max_abs = np.max(np.abs(Z), axis=1)
    p = float(np.mean(max_abs >= T_obs))
    crit = float(np.quantile(max_abs, 0.95))
    return {
        "T_max": T_obs,
        "p": p,
        "crit_5pct": crit,
        "n_pre": len(names),
        "pre_names": names,
        "note": "Usual PT on binary tratado×k ES (Y(0)). Not SPT.",
    }


def joint_theta_pre(m, pre_names: list[str]) -> dict:
    """Wald joint test θ_k=0 for 2023 leads on dose-W ES."""
    if m is None or not pre_names:
        return {"error": "no model/pre"}
    coef_idx = list(m.coef().index)
    names = [n for n in pre_names if n in coef_idx]
    if len(names) < 1:
        return {"error": "no pre coefs", "looked_for": pre_names[:5]}
    k = len(coef_idx)
    m_h = len(names)
    R = np.zeros((m_h, k))
    for i, n in enumerate(names):
        R[i, coef_idx.index(n)] = 1.0
    try:
        res = m.wald_test(R=R)
        stat = float(res["statistic"]) if "statistic" in res.index else float(res.iloc[0])
        pval = float(res["pvalue"]) if "pvalue" in res.index else float(res.iloc[1])
    except Exception as e:
        # manual Wald
        V = np.asarray(m._vcov, dtype=float)
        b = np.asarray(m.coef(), dtype=float)
        ix = [coef_idx.index(n) for n in names]
        Rb = b[ix]
        RVRT = V[np.ix_(ix, ix)]
        try:
            inv = np.linalg.inv(RVRT)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(RVRT)
        stat = float(Rb @ inv @ Rb)
        # chi2 p
        from math import gamma

        # survival of chi2(m_h): use scipy if available else erfc approx for large
        try:
            from scipy.stats import chi2

            pval = float(chi2.sf(stat, m_h))
        except Exception:
            pval = float("nan")
            FAILURES.append(f"joint_theta_pre fallback p failed: {e}")
    return {
        "wald_stat": stat,
        "p": pval,
        "n_pre": len(names),
        "df": len(names),
        "pre_names": names,
        "note": (
            "Joint θ_k=0 in 2023 on dose-W ES: common implication of both hyps. "
            "Rejecting counts against that implication. Non-reject ≠ confirm strong hyp / SPT."
        ),
    }


def plot_es_dose(es_df: pd.DataFrame, outcome: str, path: Path):
    sub = es_df.loc[es_df["outcome"] == outcome].sort_values("k")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="gray", ls="--", lw=0.8)
    # shade 2025 (k=0..11) and 2026 (k=12..17)
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
    ax.set_title(
        f"ES dose-W ({outcome}) — principal; base 2024 omitted\n"
        r"Pre $W{=}5$ = lead scaling (not calendar dose). "
        r"$5\theta_k\approx$ level 2025; $10\theta_k\approx$ level 2026."
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_es_binary(es_df: pd.DataFrame, outcome: str, path: Path):
    sub = es_df.loc[es_df["outcome"] == outcome].sort_values("k")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="gray", ls="--", lw=0.8)
    ax.axvspan(-0.5, 11.5, color="C0", alpha=0.08)
    ax.axvspan(11.5, 18.5, color="C1", alpha=0.10)
    ax.errorbar(
        sub["k"],
        sub["coef"],
        yerr=1.96 * sub["se"],
        fmt="o",
        ms=3,
        capsize=2,
        color="C0",
    )
    ax.set_xlabel("Event time k (k=0 at Jan/2025)")
    ax.set_ylabel("ATT (binary treated × month)")
    ax.set_title(
        f"ES binary ({outcome}) — exposure timing (not principal); base 2024\n"
        "Used for usual PT / T_max on 2023 leads."
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def try_saturated_fe(df: pd.DataFrame, outcome: str = "saldo") -> str:
    """Pick saturated FE that fits; document what ran."""
    global FE_SAT_USED
    for fe, label in [
        (FE_SAT_LEAN, "lean: cnae4^mes_cal + municipio^time_id + municipio^ano"),
        (FE_SAT_FULL, "full: cnae4^municipio^mes_cal + municipio^time_id + municipio^ano"),
    ]:
        print(f"Trying saturated FE ({label})...", flush=True)
        _mem("sat try")
        m = fit(f"{outcome} ~ t2025 + t2026 | {fe}", df, f"sat_probe/{label}")
        if m is not None:
            FE_SAT_USED = fe
            del m
            gc.collect()
            print(f"Saturated FE selected: {fe}", flush=True)
            # Prefer lean if it works; only upgrade to full if lean worked AND full works
            # Brief: try lean first, then add triple if feasible.
            if fe == FE_SAT_LEAN:
                # try full upgrade
                print("Lean OK; probing full triple...", flush=True)
                m2 = fit(
                    f"{outcome} ~ t2025 + t2026 | {FE_SAT_FULL}",
                    df,
                    "sat_probe/full",
                )
                if m2 is not None:
                    FE_SAT_USED = FE_SAT_FULL
                    del m2
                    gc.collect()
                    print(f"Upgraded to full saturated: {FE_SAT_USED}", flush=True)
                else:
                    FAILURES.append(
                        "Full saturated FE (cnae4^municipio^mes_cal + ...) failed; using lean."
                    )
            return FE_SAT_USED
        gc.collect()
    # last resort: base only marked as failed sat
    FAILURES.append("All saturated FE probes failed; saturated table will be empty/partial.")
    FE_SAT_USED = None
    return FE_SAT_USED


def write_texts(info, att_base, att_sat, reta_df, tmax_rows, joint_rows, fe_sat):
    # pull saldo numbers
    def g(df, term, outcome="saldo"):
        r = df.loc[(df.outcome == outcome) & (df.term == term)]
        return None if r.empty else r.iloc[0]

    s5 = g(att_base, "beta5")
    s10 = g(att_base, "beta10")
    sd = g(att_base, "delta")
    sd5 = g(att_base, "delta_over_5")
    tm = next((r for r in tmax_rows if r.get("outcome") == "saldo"), {})
    jt = next((r for r in joint_rows if r.get("outcome") == "saldo"), {})

    # ---- RESUMO ----
    lines = []
    lines.append("# RESUMO_CGS_RODADA2 — Contrastes β5/β10 + ES dose-W\n")
    lines.append("## Identificação\n")
    lines.append(
        f"- Principal ATT: `Y ~ t2025 + t2026 | {FE_BASE}` (CRV1 cnae4)."
    )
    lines.append(
        "- D_s = 1 se classe listada (time-invariant). Base omitida: 2023–2024. "
        "2026 só até junho. β5 = 2025 vs base; β10 = 2026 vs **mesma** base (não vs 2025)."
    )
    lines.append(
        "- Reportamos também Δ=β10−β5 e (β10−β5)/5 com EP clusterizado do contraste."
    )
    lines.append(
        f"- FE saturada (lado a lado): `{fe_sat}` "
        "(FEs que não absorvem dose; sem class×year / class×civil month / class×2025–26). "
        "UF×time omitido quando municipio^time_id está presente."
    )
    lines.append(
        "- Reta linear em Dose: **coluna de comparação apenas** (não preferida)."
    )
    lines.append(
        "- ES **principal**: `i(k_es)×W_sk` com W=0 (controle); W=5 (listada 2023–2025); "
        "W=10 (listada 2026). Pré W=5 = **escala do lead** (normalização), não dose calendário."
    )
    lines.append(
        "- ES binário tratado×k: ao lado, timing de exposição (não principal). "
        "T_max nos leads 2023 do binário = PT usual."
    )
    lines.append(
        "- Teste conjunto θ_k=0 em 2023 no ES dose-W = implicação comum das duas hipóteses; "
        "não-rejeição ≠ confirmação da hipótese forte / SPT.\n"
    )
    lines.append("## Painel\n")
    lines.append(f"- Pares: **{info.get('n_pairs')}**; N flows: **{info.get('N_flows')}**")
    lines.append(f"- Competências: {info.get('comps_first')}–{info.get('comps_last')}")
    lines.append(f"- NEVER_DROP: {info.get('never_drop')}")
    lines.append(f"- Wage drop months: {info.get('salary_drop_months')}\n")

    lines.append("## Contrastes ATT (FE base)\n")
    lines.append("| outcome | β5 | se | β10 | se | Δ | se | Δ/5 | se | N |")
    lines.append("|---------|----|----|-----|----|---|----|-----|----|---|")
    for y in OUTCOMES_ALL:
        r5, r10, rd, rd5 = g(att_base, "beta5", y), g(att_base, "beta10", y), g(att_base, "delta", y), g(att_base, "delta_over_5", y)
        if r5 is None:
            continue
        lines.append(
            f"| {y} | {r5.coef:.4f} | {r5.se:.4f} | {r10.coef:.4f} | {r10.se:.4f} | "
            f"{rd.coef:.4f} | {rd.se:.4f} | {rd5.coef:.4f} | {rd5.se:.4f} | {int(r5.N)} |"
        )

    lines.append("\n## Contrastes ATT (FE saturada)\n")
    if att_sat is not None and len(att_sat):
        lines.append(f"FE usada: `{fe_sat}`\n")
        lines.append("| outcome | β5 | se | β10 | se | Δ | se | Δ/5 | se | N |")
        lines.append("|---------|----|----|-----|----|---|----|-----|----|---|")
        for y in OUTCOMES_ALL:
            r5, r10, rd, rd5 = g(att_sat, "beta5", y), g(att_sat, "beta10", y), g(att_sat, "delta", y), g(att_sat, "delta_over_5", y)
            if r5 is None:
                continue
            lines.append(
                f"| {y} | {r5.coef:.4f} | {r5.se:.4f} | {r10.coef:.4f} | {r10.se:.4f} | "
                f"{rd.coef:.4f} | {rd.se:.4f} | {rd5.coef:.4f} | {rd5.se:.4f} | {int(r5.N)} |"
            )
    else:
        lines.append("_Saturada não estimada (ver failures)._\n")

    lines.append("\n## Reta comparação (Dose linear)\n")
    for _, r in reta_df.iterrows():
        lines.append(
            f"- {r['outcome']}: coef={r['coef']:.6g} se={r['se']:.4g} p={r['p']:.4g} "
            "(comparison only; not preferred)"
        )

    lines.append("\n## T_max PT usual (ES binário base2024, k∈2023)\n")
    for r in tmax_rows:
        lines.append(
            f"- {r.get('outcome')}: T_max={r.get('T_max')} p={r.get('p')} "
            f"crit5%={r.get('crit_5pct')} n_pre={r.get('n_pre')} — PT usual, não SPT."
        )

    lines.append("\n## Joint θ_pre dose-W (2023)\n")
    for r in joint_rows:
        lines.append(
            f"- {r.get('outcome')}: Wald={r.get('wald_stat')} p={r.get('p')} "
            f"df={r.get('df')} — implicação comum; não-rejeição ≠ aceitar hipótese forte."
        )

    lines.append("\n## Caveats (linguagem)\n")
    lines.append(
        "1. W_sk=5 no pré é **escala do lead**, não dose real; θ_k pré é efeito por ponto sob essa normalização."
    )
    lines.append(
        "2. (β10−β5)/5 sob hipótese forte (CGS Thm 3.3(d)) seria ACRT, mas com dinâmica de 2026 "
        "(já um ano em dose 5) **não** declaramos ACRT limpo / hipótese validada."
    )
    lines.append(
        "3. T_max no ES binário = PT usual; joint θ_2023 no ES dose-W = implicação conjunta, não aceite da forte."
    )
    lines.append(
        "4. Não usamos β5/5 = β10/10 como teste da hipótese forte."
    )
    lines.append(
        "5. Wage: pre-trends já fracos na rodada anterior — manter caveat.\n"
    )

    if FAILURES:
        lines.append("## Failures\n")
        for f in FAILURES:
            lines.append(f"- {f}")
        lines.append("")

    (OUT / "RESUMO_CGS_RODADA2.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote RESUMO_CGS_RODADA2.md", flush=True)

    # ---- TEXTO RESULTADOS (PT) with mandatory §3 paragraph ----
    if s5 is not None and sd5 is not None:
        b5, se5 = float(s5.coef), float(s5.se)
        b10, se10 = float(s10.coef), float(s10.se)
        dlt, sed = float(sd.coef), float(sd.se)
        d5, se_d5 = float(sd5.coef), float(sd5.se)
        p_d5 = float(sd5.p)
        N = int(s5.N)
        tmax_p = tm.get("p", float("nan"))
        tmax_v = tm.get("T_max", float("nan"))
        joint_p = jt.get("p", float("nan"))
        joint_w = jt.get("wald_stat", float("nan"))
    else:
        b5 = se5 = b10 = se10 = dlt = sed = d5 = se_d5 = p_d5 = N = float("nan")
        tmax_p = tmax_v = joint_p = joint_w = float("nan")

    texto = f"""# TEXTO_RESULTADOS_RODADA2 — Projeto Reoneração Folha (Lei 14.973/2024)

## 1. Especificação principal (contrastes anuais)

Estimamos
Y_smt = α_{{s,c(t)}} + μ_m + λ_t + β5 (D_s × 1{{2025}}) + β10 (D_s × 1{{2026}}) + ε,
com D_s = 1 se a classe CNAE está listada (invariante no tempo) e 0 caso contrário. A base omitida é 2023–2024; 2026 entra só até junho. Assim β5 é o ATT da listada em 2025 versus a base zero, e β10 é o ATT em 2026 versus a **mesma** base (não versus 2025). FE base: `{FE_BASE}`; erros CRV1 por cnae4. Também reportamos o contraste Δ = β10 − β5 e (β10 − β5)/5 com erro-padrão clusterizado do contraste.

Para **saldo**, β5 = {b5:.4f} (EP = {se5:.4f}), β10 = {b10:.4f} (EP = {se10:.4f}), Δ = {dlt:.4f} (EP = {sed:.4f}), (β10−β5)/5 = {d5:.4f} (EP = {se_d5:.4f}, p = {p_d5:.3f}), N = {N}. Análogos para admitidos, desligados e log do salário de admissão (amostra salarial com filtro 0,3–150 SM e exclusão de 202308–09; pre-trends de salário já fracos — caveat mantido).

A reta Y ~ Dose | FE é mantida apenas como **coluna de comparação** (impõe linearidade; não é a especificação preferida).

FE saturada lado a lado (sem absorver dose): `{fe_sat}`. Não incluímos class×year, class×mês civil nem dummies class×2025/2026. Com municipio^time_id presente, UF×time é omitido por colinearidade.

## 2. Estudo de eventos

O ES **principal** multiplica o mês de evento pela dose atribuída W_sk: controles W=0; listadas em 2023–2025 W=5; listadas em 2026 W=10. Base B = ano civil 2024 omitido por completo. Os coeficientes θ_k são efeitos **por ponto** daquele mês: 5θ_k ≈ nível em 2025 e 10θ_k ≈ nível em 2026. Importante: W=5 no período pré-tratamento é **escala do lead** (normalização para o lead não zerar), **não** dose calendário real — o θ_k pré interpreta-se como efeito por ponto sob essa normalização.

O ES binário tratado×k permanece ao lado como descrição do **timing de exposição**, não como especificação principal. A base-2023 com W=5 nas listadas serve de checagem de antecipação (sem afirmar que antecipação “não paga”).

## 3. Interpretação de (β10 − β5)/5 sob a hipótese forte (obrigatório)

Sob a hipótese forte de tendências paralelas em doses (CGS, Teorema 3.3(d)), a quantidade (β10 − β5)/5 = {d5:.4f} (EP = {se_d5:.4f}) identifica o ACRT — o efeito causal médio de um ponto adicional de dose entre 5 e 10. Sem a hipótese forte, o mesmo objeto é a soma de um termo de incremento de dose e um termo de seleção (Teorema 3.2). A razão para invocar a interpretação forte neste desenho é institucional: a Lei 14.973/2024 atribui mecanicamente dose 5 e depois dose 10 às **mesmas** classes listadas; não há duas subpopulações listadas escolhendo doses diferentes no mesmo mês civil — o termo de seleção, portanto, não tem um grupo observável em que se apoiar. **Caveat decisivo:** 2026 já carrega um ano inteiro sob dose 5; dinâmicas de ajuste, estoque e composição permanecem. Por isso reportamos (β10 − β5)/5 com o parágrafo da hipótese forte, mas **não** declaramos ACRT limpo nem validamos a hipótese forte / SPT. Em particular, **não** usamos o teste β5/5 = β10/10 como prova da hipótese forte.

## 4. Checagens de pré-tendência

- **Nível (PT usual):** T_max studentizado nos leads de 2023 do ES **binário** (base 2024). Para saldo, T_max = {tmax_v} (p = {tmax_p}). Isso é o teste usual de parallel trends em Y(0), não SPT.
- **Inclinação (implicação conjunta):** teste de Wald conjunto θ_k = 0 em 2023 no ES dose-W (análogo à Fig. 10). Para saldo, Wald = {joint_w} (p = {joint_p}). Rejeitar conta contra uma implicação comum das duas hipóteses; **não rejeitar não confirma** a hipótese forte.

## 5. Avisos finais

Wage: pre-trends já frágeis — manter cautela. Não destruir artefatos das rodadas anteriores (`cgs_saturado/`, `cristine_callaway/`). Não editar Overleaf nesta rodada.
"""
    (OUT / "TEXTO_RESULTADOS_RODADA2.md").write_text(texto, encoding="utf-8")
    print("Wrote TEXTO_RESULTADOS_RODADA2.md", flush=True)


def stage_drive():
    keep = [
        "tab_att_contrastes.csv",
        "tab_att_saturada.csv",
        "tab_reta_comparacao.csv",
        "tab_es_dose_base2024.csv",
        "tab_es_binario_base2024.csv",
        "tab_tmax_binario.csv",
        "tab_joint_theta_pre_dose.csv",
        "RESUMO_CGS_RODADA2.md",
        "TEXTO_RESULTADOS_RODADA2.md",
        "key_numbers.json",
        "run_cgs_rodada2.py",
        "notebook_reoneracao_cgs_rodada2.ipynb",
        "fig_es_dose_base2024_saldo.png",
        "fig_es_dose_base2024_admitidos.png",
        "fig_es_dose_base2024_desligados.png",
        "fig_es_dose_base2024_log_salario.png",
        "fig_es_binario_base2024_saldo.png",
    ]
    for name in keep:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, STAGING / name)
    print(f"Staged {len(list(STAGING.iterdir()))} files to drive_staging/", flush=True)


def main():
    print("=== CGS rodada2 pipeline ===", flush=True)
    info = json.loads(PANEL_META.read_text()) if PANEL_META.exists() else {}
    print(json.dumps({k: info.get(k) for k in ("n_pairs", "N_flows", "comps_first", "comps_last", "never_drop")}, indent=2), flush=True)

    flow_cols = [
        "cnae4", "municipio", "secao", "competencia", "ano", "mes_cal", "time_id",
        "k", "tratado", "dose", "t2025", "t2026", "saldo", "admitidos", "desligados",
    ]
    print("Loading flows...", flush=True)
    flows = pd.read_parquet(PANEL_BAL, columns=flow_cols)
    wage = pd.read_parquet(PANEL_WAGE)
    flows["W"] = assign_W(flows["tratado"], flows["ano"])
    wage["W"] = assign_W(wage["tratado"], wage["ano"])
    # sanity: dose calendar still 0 in pre; W is 5 for listed pre
    pre_mask = flows["ano"].isin([2023, 2024])
    assert (flows.loc[pre_mask, "dose"] == 0).all(), "calendar dose nonzero in pre"
    listed_pre = pre_mask & (flows["tratado"] == 1)
    assert (flows.loc[listed_pre, "W"] == 5).all(), "W!=5 for listed pre"
    assert (flows.loc[flows["tratado"] == 0, "W"] == 0).all()
    print(f"flows N={len(flows)}; wage N={len(wage)}; W unique={sorted(flows['W'].unique())}", flush=True)
    _mem("loaded")

    # ---- ATT base contrasts ----
    print("=== ATT base contrasts ===", flush=True)
    att_rows = []
    for y in OUTCOMES_FLOW:
        rows, m = run_att(flows, y, FE_BASE, "att_base")
        att_rows.extend(rows)
        del m
        gc.collect()
    rows, m = run_att(wage, "log_salario", FE_BASE, "att_base")
    att_rows.extend(rows)
    del m
    gc.collect()
    att_base = pd.DataFrame(att_rows)
    att_base.to_csv(OUT / "tab_att_contrastes.csv", index=False)
    print(att_base.to_string(index=False), flush=True)

    # ---- Saturated FE ----
    print("=== Saturated FE probe + ATT ===", flush=True)
    fe_sat = try_saturated_fe(flows, "saldo")
    att_sat_rows = []
    if fe_sat:
        for y in OUTCOMES_FLOW:
            rows, m = run_att(flows, y, fe_sat, "att_saturada")
            att_sat_rows.extend(rows)
            del m
            gc.collect()
        rows, m = run_att(wage, "log_salario", fe_sat, "att_saturada")
        att_sat_rows.extend(rows)
        del m
        gc.collect()
    att_sat = pd.DataFrame(att_sat_rows)
    att_sat.to_csv(OUT / "tab_att_saturada.csv", index=False)

    # ---- Reta comparação ----
    print("=== Reta comparação ===", flush=True)
    reta_rows = []
    for y in OUTCOMES_FLOW:
        reta_rows.extend(run_reta(flows, y, FE_BASE))
        gc.collect()
    reta_rows.extend(run_reta(wage, "log_salario", FE_BASE))
    reta_df = pd.DataFrame(reta_rows)
    reta_df.to_csv(OUT / "tab_reta_comparacao.csv", index=False)

    # ---- ES dose-W PRINCIPAL base 2024 + joint θ_pre ----
    print("=== ES dose-W principal base2024 + joint θ_pre ===", flush=True)
    es_dose_all = []
    joint_rows = []
    for y in OUTCOMES_FLOW:
        es_df, m, levels = run_es_dose_W(flows, y, base_year=2024, fe=FE_BASE)
        es_dose_all.append(es_df)
        pre_names = [f"k_es::{kk}:W" for kk in levels if -24 <= kk <= -13]
        # also try alternate name patterns from pyfixest
        if m is not None:
            idx = list(m.coef().index)
            alt = [n for n in idx if "k_es::" in str(n) and any(f"::{kk}:" in str(n) or str(n).endswith(f"::{kk}") for kk in range(-24, -12))]
            # build pre_names from actual index
            pre_names = []
            for n in idx:
                s = str(n)
                if "k_es::" not in s:
                    continue
                try:
                    kk = int(s.split("k_es::")[1].split(":")[0])
                except Exception:
                    continue
                if -24 <= kk <= -13:
                    pre_names.append(s)
        jt = joint_theta_pre(m, pre_names)
        jt["outcome"] = y
        jt["spec"] = "ES_dose_W_base2024"
        joint_rows.append(jt)
        del m
        gc.collect()

    es_df, m, levels = run_es_dose_W(wage, "log_salario", base_year=2024, fe=FE_BASE)
    es_dose_all.append(es_df)
    pre_names = []
    if m is not None:
        for n in m.coef().index:
            s = str(n)
            if "k_es::" not in s:
                continue
            try:
                kk = int(s.split("k_es::")[1].split(":")[0])
            except Exception:
                continue
            if -24 <= kk <= -13:
                pre_names.append(s)
    jt = joint_theta_pre(m, pre_names)
    jt["outcome"] = "log_salario"
    jt["spec"] = "ES_dose_W_base2024"
    joint_rows.append(jt)
    del m
    gc.collect()

    es_dose = pd.concat([e for e in es_dose_all if e is not None and len(e)], ignore_index=True)
    es_dose.to_csv(OUT / "tab_es_dose_base2024.csv", index=False)
    joint_out = []
    for r in joint_rows:
        rr = {k: v for k, v in r.items() if k != "pre_names"}
        rr["pre_names"] = json.dumps(r.get("pre_names", []))
        joint_out.append(rr)
    pd.DataFrame(joint_out).to_csv(OUT / "tab_joint_theta_pre_dose.csv", index=False)

    for y in OUTCOMES_ALL:
        plot_es_dose(es_dose, y, OUT / f"fig_es_dose_base2024_{y}.png")

    # ES dose-W base 2023 anticipation (table only)
    print("=== ES dose-W base2023 (anticipation check) ===", flush=True)
    es23 = []
    for y in OUTCOMES_FLOW:
        es_df, m, _ = run_es_dose_W(flows, y, base_year=2023, fe=FE_BASE)
        es23.append(es_df)
        del m
        gc.collect()
    es_df, m, _ = run_es_dose_W(wage, "log_salario", base_year=2023, fe=FE_BASE)
    es23.append(es_df)
    del m
    gc.collect()
    pd.concat([e for e in es23 if e is not None and len(e)], ignore_index=True).to_csv(
        OUT / "tab_es_dose_base2023.csv", index=False
    )

    # ---- ES binary (timing) + T_max ----
    print("=== ES binary base2024 + T_max ===", flush=True)
    es_bin_all = []
    tmax_rows = []
    for y in OUTCOMES_FLOW:
        es_df, m, levels = run_es_binary(flows, y, base_year=2024, fe=FE_BASE)
        es_bin_all.append(es_df)
        pre_names = []
        if m is not None:
            for n in m.coef().index:
                s = str(n)
                if "k_es::" not in s:
                    continue
                try:
                    kk = int(s.split("k_es::")[1].split(":")[0])
                except Exception:
                    continue
                if -24 <= kk <= -13:
                    pre_names.append(s)
        tm = studentized_max_test(m, pre_names)
        tm["outcome"] = y
        tm["spec"] = "ES_binary_base2024"
        tmax_rows.append(tm)
        del m
        gc.collect()
    es_df, m, levels = run_es_binary(wage, "log_salario", base_year=2024, fe=FE_BASE)
    es_bin_all.append(es_df)
    pre_names = []
    if m is not None:
        for n in m.coef().index:
            s = str(n)
            if "k_es::" not in s:
                continue
            try:
                kk = int(s.split("k_es::")[1].split(":")[0])
            except Exception:
                continue
            if -24 <= kk <= -13:
                pre_names.append(s)
    tm = studentized_max_test(m, pre_names)
    tm["outcome"] = "log_salario"
    tm["spec"] = "ES_binary_base2024"
    tmax_rows.append(tm)
    del m
    gc.collect()

    es_bin = pd.concat([e for e in es_bin_all if e is not None and len(e)], ignore_index=True)
    es_bin.to_csv(OUT / "tab_es_binario_base2024.csv", index=False)
    tmax_out = []
    for r in tmax_rows:
        rr = {k: v for k, v in r.items() if k not in ("pre_names", "pre_t")}
        rr["pre_names"] = json.dumps(r.get("pre_names", []))
        tmax_out.append(rr)
    pd.DataFrame(tmax_out).to_csv(OUT / "tab_tmax_binario.csv", index=False)

    # at least saldo binary fig; also others if cheap
    plot_es_binary(es_bin, "saldo", OUT / "fig_es_binario_base2024_saldo.png")
    for y in ["admitidos", "desligados", "log_salario"]:
        plot_es_binary(es_bin, y, OUT / f"fig_es_binario_base2024_{y}.png")

    write_texts(info, att_base, att_sat, reta_df, tmax_rows, joint_rows, fe_sat or "NONE")

    # key numbers
    def grab(term):
        r = att_base.loc[(att_base.outcome == "saldo") & (att_base.term == term)]
        return float(r.coef.iloc[0]), float(r.se.iloc[0]), float(r.p.iloc[0])

    b5, se5, p5 = grab("beta5")
    b10, se10, p10 = grab("beta10")
    dlt, sed, pdlt = grab("delta")
    d5, sed5, pd5 = grab("delta_over_5")
    tm = next(r for r in tmax_rows if r.get("outcome") == "saldo")
    jt = next(r for r in joint_rows if r.get("outcome") == "saldo")
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
        "saldo_N": int(att_base.loc[(att_base.outcome == "saldo") & (att_base.term == "beta5"), "N"].iloc[0]),
        "tmax_saldo": tm.get("T_max"),
        "tmax_p_saldo": tm.get("p"),
        "joint_theta_pre_wald_saldo": jt.get("wald_stat"),
        "joint_theta_pre_p_saldo": jt.get("p"),
        "fe_base": FE_BASE,
        "fe_sat_used": fe_sat,
        "failures": FAILURES,
        "out_dir": str(OUT),
    }
    (OUT / "key_numbers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("KEY", json.dumps(summary, indent=2), flush=True)

    stage_drive()
    print("DONE", flush=True)
    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "run_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
