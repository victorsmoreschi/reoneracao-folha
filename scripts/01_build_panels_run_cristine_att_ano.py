#!/usr/bin/env python3
"""Cristine / Callaway–Goodman-Bacon–Sant'Anna DiD for reoneração (Lei 14.973/2024).

S1: ATT by year (t2025, t2026) — MAIN
S2: dose×year interactions — check vs S1×dose levels
S3: ES with EVENTUAL dose D̄_s∈{0,10} × 1{k} (so pre PT is testable)
S4: binary ES (annex)
S5/S6: extra FE robustness
Tests: studentized max T (Roth), linearity β2026=2*β2025, placebo 2024
"""
from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyfixest.estimation import feols
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUT = Path("/workspace/notebook_outputs/cristine_callaway")
OUT.mkdir(parents=True, exist_ok=True)
PANEL_SRC = Path("/workspace/sdid/painel_estimacao.parquet")
PANEL_BAL = OUT / "painel_balanced_flows.parquet"
PANEL_WAGE = OUT / "painel_wage.parquet"
NEVER_DROP = {"1521", "1529", "4911", "5212", "5231", "5510"}
SM = {2023: 1320.0, 2024: 1412.0, 2025: 1518.0, 2026: 1518.0}
FE_PREF = "cnae4^mes_cal + municipio + time_id"
VCOV = {"CRV1": "cnae4"}
RNG = np.random.default_rng(20250301)
PANEL_VERSION = "never_drop_v1"


def _mem(tag: str = "") -> None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KB
        print(f"[mem] {tag}: maxrss={rss/1e6:.2f} GB", flush=True)
    except Exception:
        pass


def make_kes(k: pd.Series, base_year: int) -> tuple[pd.Series, int]:
    """Map all k belonging to base_year to a single reference value."""
    # k=0 at 202501; year from k: 2025 + floor(k/12), month = k%12 + 1
    # For base 2024: k in [-12, -1]; base 2023: k in [-24, -13]
    if base_year == 2024:
        lo, hi, ref = -12, -1, -12
    elif base_year == 2023:
        lo, hi, ref = -24, -13, -24
    else:
        raise ValueError(base_year)
    kes = k.astype(np.int16).copy()
    kes = kes.where(~((kes >= lo) & (kes <= hi)), other=np.int16(ref))
    return kes, ref


def build_panels() -> dict:
    """Build preferred balanced flow panel + wage sample. Save to disk."""
    if PANEL_BAL.exists() and PANEL_WAGE.exists():
        meta_path = OUT / "panel_meta.json"
        if meta_path.exists():
            cached = json.loads(meta_path.read_text())
            if cached.get("panel_version") == PANEL_VERSION and cached.get("never_drop_applied"):
                print("Using cached balanced panels", flush=True)
                return cached
            print("Cached balanced panels predate NEVER_DROP filter; rebuilding", flush=True)

    print("Loading raw panel...", flush=True)
    cols = [
        "competencia",
        "municipio",
        "cnae4",
        "secao",
        "admitidos",
        "desligados",
        "saldo",
        "n_admissao_com_salario",
        "salario_admissao_medio",
        "tratado",
    ]
    raw = pd.read_parquet(PANEL_SRC, columns=cols)
    raw["cnae4"] = raw["cnae4"].astype(str).str.zfill(4)
    raw["municipio"] = raw["municipio"].astype(str)
    raw["competencia"] = raw["competencia"].astype(str)
    raw["secao"] = raw["secao"].astype(str)

    # Drop sections A/O and NEVER_DROP before pair metadata and balanced grid.
    n0 = len(raw)
    never_before = sorted(NEVER_DROP & set(raw["cnae4"].unique()))
    keep = (~raw["secao"].isin(["A", "O"])) & (~raw["cnae4"].isin(NEVER_DROP))
    raw = raw.loc[keep].copy()
    print(f"Drop A/O + NEVER_DROP: {n0} -> {len(raw)} (removed {n0 - len(raw)})", flush=True)
    print(f"NEVER_DROP removed from flows: {never_before}", flush=True)

    comps = sorted(raw["competencia"].unique())
    assert len(comps) == 42, len(comps)
    assert comps[0] == "202301" and comps[-1] == "202606"

    # Pair-level meta
    meta = (
        raw.groupby(["cnae4", "municipio"], sort=False)
        .agg(secao=("secao", "first"), tratado=("tratado", "max"))
        .reset_index()
    )
    n_pairs = len(meta)
    print(f"Pairs: {n_pairs}", flush=True)

    # Observed flow keys
    raw["admitidos"] = raw["admitidos"].fillna(0).astype(np.int32)
    raw["desligados"] = raw["desligados"].fillna(0).astype(np.int32)
    raw["saldo"] = raw["saldo"].fillna(0).astype(np.int32)

    flow = raw[
        ["cnae4", "municipio", "competencia", "admitidos", "desligados", "saldo"]
    ].copy()
    del raw
    gc.collect()

    print("Building full grid (zeros for silent)...", flush=True)
    # Efficient: MultiIndex from product via merge
    meta["key"] = 1
    comp_df = pd.DataFrame({"competencia": comps, "key": 1})
    grid = meta.merge(comp_df, on="key", how="inner").drop(columns="key")
    # Expected N
    N = len(grid)
    print(f"Grid N={N}; N%42==0 -> {N % 42 == 0}; pairs*42={n_pairs * 42}", flush=True)

    bal = grid.merge(flow, on=["cnae4", "municipio", "competencia"], how="left")
    del grid, flow
    gc.collect()
    for c in ("admitidos", "desligados", "saldo"):
        bal[c] = bal[c].fillna(0).astype(np.int32)

    # Calendar features
    bal["ano"] = bal["competencia"].str[:4].astype(np.int16)
    bal["mes_cal"] = bal["competencia"].str[4:6].astype(np.int8)
    # time_id from sorted competencias
    comp_to_tid = {c: i for i, c in enumerate(comps)}
    bal["time_id"] = bal["competencia"].map(comp_to_tid).astype(np.int16)
    # k relative to 202501 (k=0)
    bal["k"] = ((bal["ano"] - 2025) * 12 + (bal["mes_cal"] - 1)).astype(np.int16)

    # Dose calendar (S1/S2/post): 0 in 2023-24; 5 in 2025 if tratado; 10 in 2026+ if tratado
    bal["tratado"] = bal["tratado"].astype(np.int8)
    bal["dose"] = np.int8(0)
    bal.loc[(bal["tratado"] == 1) & (bal["ano"] == 2025), "dose"] = np.int8(5)
    bal.loc[(bal["tratado"] == 1) & (bal["ano"] >= 2026), "dose"] = np.int8(10)

    # Eventual dose for S3 PT-testable ES: D̄_s ∈ {0, 10}
    bal["dose_bar"] = np.where(bal["tratado"] == 1, np.int8(10), np.int8(0)).astype(np.int8)

    # S1 indicators
    bal["t2025"] = ((bal["ano"] == 2025) & (bal["tratado"] == 1)).astype(np.int8)
    bal["t2026"] = ((bal["ano"] == 2026) & (bal["tratado"] == 1)).astype(np.int8)
    # S2: dose × year (dose already 0 outside treated×year, but explicit)
    bal["dose_2025"] = np.where(bal["ano"] == 2025, bal["dose"], 0).astype(np.int8)
    bal["dose_2026"] = np.where(bal["ano"] == 2026, bal["dose"], 0).astype(np.int8)

    # Lean cats
    for c in ("cnae4", "municipio", "secao", "competencia"):
        bal[c] = bal[c].astype("category")

    print(f"Balanced flows: N={len(bal)}, mem≈{bal.memory_usage(deep=True).sum()/1e9:.2f} GB", flush=True)
    bal.to_parquet(PANEL_BAL, index=False)
    print(f"Saved {PANEL_BAL}", flush=True)

    # Wage sample from ORIGINAL (nonzero salary rows) — re-read raw salary cols
    print("Building wage sample...", flush=True)
    wraw = pd.read_parquet(
        PANEL_SRC,
        columns=[
            "competencia",
            "municipio",
            "cnae4",
            "secao",
            "n_admissao_com_salario",
            "salario_admissao_medio",
            "tratado",
        ],
    )
    wraw["cnae4"] = wraw["cnae4"].astype(str).str.zfill(4)
    wraw["municipio"] = wraw["municipio"].astype(str)
    wraw["competencia"] = wraw["competencia"].astype(str)
    wraw["secao"] = wraw["secao"].astype(str)
    wraw = wraw.loc[
        (~wraw["secao"].isin(["A", "O"])) & (~wraw["cnae4"].isin(NEVER_DROP))
    ].copy()
    wraw["ano"] = wraw["competencia"].str[:4].astype(np.int16)
    wraw["mes_cal"] = wraw["competencia"].str[4:6].astype(np.int8)
    wraw["sm"] = wraw["ano"].map(SM).astype(float)
    wraw["sal"] = pd.to_numeric(wraw["salario_admissao_medio"], errors="coerce")
    wraw["n_adm"] = pd.to_numeric(wraw["n_admissao_com_salario"], errors="coerce").fillna(0)
    # Keep 0.3*SM < sal ≤ 150*SM
    ok = (wraw["sal"] > 0.3 * wraw["sm"]) & (wraw["sal"] <= 150 * wraw["sm"])
    # if n_admissao≤2 and sal>50k → missing
    bad_tiny = (wraw["n_adm"] <= 2) & (wraw["sal"] > 50_000)
    wraw = wraw.loc[ok & ~bad_tiny].copy()
    wraw["log_salario"] = np.log(wraw["sal"].astype(float))
    # Drop 202308–202309 ONLY for salary
    n_before = len(wraw)
    wraw = wraw.loc[~wraw["competencia"].isin(["202308", "202309"])].copy()
    print(
        f"Wage: kept after SM filter then drop 202308-09: final N={len(wraw)} (dropped {n_before - len(wraw)} months 08-09)",
        flush=True,
    )
    wraw["time_id"] = wraw["competencia"].map(comp_to_tid).astype(np.int16)
    wraw["k"] = ((wraw["ano"] - 2025) * 12 + (wraw["mes_cal"] - 1)).astype(np.int16)
    wraw["tratado"] = wraw["tratado"].astype(np.int8)
    wraw["dose"] = np.int8(0)
    wraw.loc[(wraw["tratado"] == 1) & (wraw["ano"] == 2025), "dose"] = np.int8(5)
    wraw.loc[(wraw["tratado"] == 1) & (wraw["ano"] >= 2026), "dose"] = np.int8(10)
    wraw["dose_bar"] = np.where(wraw["tratado"] == 1, np.int8(10), np.int8(0)).astype(np.int8)
    wraw["t2025"] = ((wraw["ano"] == 2025) & (wraw["tratado"] == 1)).astype(np.int8)
    wraw["t2026"] = ((wraw["ano"] == 2026) & (wraw["tratado"] == 1)).astype(np.int8)
    wraw["dose_2025"] = np.where(wraw["ano"] == 2025, wraw["dose"], 0).astype(np.int8)
    wraw["dose_2026"] = np.where(wraw["ano"] == 2026, wraw["dose"], 0).astype(np.int8)
    for c in ("cnae4", "municipio", "secao", "competencia"):
        wraw[c] = wraw[c].astype("category")
    keep_w = [
        "competencia",
        "municipio",
        "cnae4",
        "secao",
        "ano",
        "mes_cal",
        "time_id",
        "k",
        "tratado",
        "dose",
        "dose_bar",
        "t2025",
        "t2026",
        "dose_2025",
        "dose_2026",
        "log_salario",
    ]
    wraw[keep_w].to_parquet(PANEL_WAGE, index=False)
    del wraw
    gc.collect()

    info = {
        "panel_version": PANEL_VERSION,
        "never_drop_applied": True,
        "never_drop": sorted(NEVER_DROP),
        "n_pairs": int(n_pairs),
        "N_flows": int(N),
        "N_mod_42": int(N % 42),
        "n_comps": 42,
        "comps_first": comps[0],
        "comps_last": comps[-1],
        "never_drop_present": sorted(NEVER_DROP & set(meta["cnae4"].astype(str))),
        "never_drop_missing": sorted(NEVER_DROP - set(meta["cnae4"].astype(str))),
        "salary_drop_months": ["202308", "202309"],
        "s3_note": (
            "S3 uses time-invariant eventual dose dose_bar∈{0,10} × 1{k_es}; "
            "NOT calendar Dose (which is 0 in 2023-24 and would make pre interactions identically zero)."
        ),
    }
    (OUT / "panel_meta.json").write_text(json.dumps(info, indent=2))
    del bal, meta
    gc.collect()
    return info


def load_flows(cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(PANEL_BAL, columns=cols)


def load_wage(cols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(PANEL_WAGE, columns=cols)


def fit(fml: str, data: pd.DataFrame, label: str = ""):
    print(f"  FIT {label}: {fml}", flush=True)
    _mem(label + " before")
    try:
        m = feols(
            fml,
            data=data,
            vcov=VCOV,
            lean=True,
            copy_data=False,
            store_data=False,
        )
        return m
    except Exception as e:
        print(f"  FAIL {label}: {type(e).__name__}: {e}", flush=True)
        return None


def tidy_row(m, names: list[str], outcome: str, spec: str) -> list[dict]:
    if m is None:
        return []
    t = m.tidy()
    rows = []
    for name in names:
        if name not in t.index:
            # try alternate index forms
            matches = [i for i in t.index if name in str(i)]
            if not matches:
                continue
            name = matches[0]
        r = t.loc[name]
        rows.append(
            {
                "outcome": outcome,
                "spec": spec,
                "term": name,
                "coef": float(r["Estimate"]),
                "se": float(r["Std. Error"]),
                "t": float(r["t value"]),
                "p": float(r["Pr(>|t|)"]),
                "ci_lo": float(r["2.5%"]),
                "ci_hi": float(r["97.5%"]),
            }
        )
    return rows


def run_S1(df: pd.DataFrame, outcome: str, fe: str = FE_PREF, tag: str = "S1") -> list[dict]:
    fml = f"{outcome} ~ t2025 + t2026 | {fe}"
    m = fit(fml, df, f"{tag}/{outcome}")
    rows = tidy_row(m, ["t2025", "t2026"], outcome, tag)
    # stash model briefly for linearity via returned dict
    return rows, m


def run_S2(df: pd.DataFrame, outcome: str) -> list[dict]:
    fml = f"{outcome} ~ dose_2025 + dose_2026 | {FE_PREF}"
    m = fit(fml, df, f"S2/{outcome}")
    return tidy_row(m, ["dose_2025", "dose_2026"], outcome, "S2"), m


def prepare_es(df: pd.DataFrame, base_year: int) -> pd.DataFrame:
    kes, ref = make_kes(df["k"], base_year)
    out = df.copy()
    out["k_es"] = kes
    out.attrs["ref"] = ref
    out.attrs["base_year"] = base_year
    return out


def run_S3_dose_bar(df: pd.DataFrame, outcome: str, base_year: int = 2024) -> tuple[pd.DataFrame, object]:
    """Event study with eventual dose_bar × i(k_es). PT-testable in pre."""
    kes, ref = make_kes(df["k"], base_year)
    # avoid full copy: attach temporary columns
    df = df.copy(deep=False)
    df["k_es"] = kes
    fml = f"{outcome} ~ i(k_es, ref={ref}):dose_bar | {FE_PREF}"
    m = fit(fml, df, f"S3_b{base_year}/{outcome}")
    rows = []
    levels = []
    if m is not None:
        t = m.tidy()
        for name in t.index:
            # name like k_es::-24:dose_bar
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
                    "spec": f"S3_dose_bar_base{base_year}",
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
                    "scaler": "dose_bar_0_10",
                }
            )
    cols_add = [r["term"] for r in rows]
    return pd.DataFrame(rows), m, sorted(levels), cols_add


def run_S4_binary(df: pd.DataFrame, outcome: str, base_year: int = 2024) -> pd.DataFrame:
    kes, ref = make_kes(df["k"], base_year)
    df = df.copy(deep=False)
    df["k_es"] = kes
    fml = f"{outcome} ~ i(k_es, ref={ref}):tratado | {FE_PREF}"
    m = fit(fml, df, f"S4_b{base_year}/{outcome}")
    rows = []
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
            r = t.loc[name]
            rows.append(
                {
                    "outcome": outcome,
                    "spec": f"S4_binary_base{base_year}",
                    "k": kk,
                    "term": s,
                    "coef": float(r["Estimate"]),
                    "se": float(r["Std. Error"]),
                    "t": float(r["t value"]),
                    "p": float(r["Pr(>|t|)"]),
                    "ci_lo": float(r["2.5%"]),
                    "ci_hi": float(r["97.5%"]),
                    "ref_k": ref,
                }
            )
    return pd.DataFrame(rows)


def studentized_max_test(m, pre_names: list[str], n_draws: int = 10_000) -> dict:
    """Roth §4.4.1 studentized max pre-trend test using coef VCOV correlation."""
    if m is None or not pre_names:
        return {"error": "no model/pre"}
    t = m.tidy()
    names = [n for n in pre_names if n in t.index]
    if len(names) < 1:
        return {"error": "no pre coefs", "looked_for": pre_names[:5]}
    tvals = np.array([float(t.loc[n, "t value"]) for n in names], dtype=float)
    T_obs = float(np.max(np.abs(tvals)))
    # correlation from VCOV
    coef_idx = list(m.coef().index)
    ix = [coef_idx.index(n) for n in names]
    V = np.asarray(m._vcov, dtype=float)
    Vsub = V[np.ix_(ix, ix)]
    se = np.sqrt(np.clip(np.diag(Vsub), 1e-18, None))
    corr = Vsub / np.outer(se, se)
    # stabilize
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)
    # draws
    try:
        Z = RNG.multivariate_normal(mean=np.zeros(len(names)), cov=corr, size=n_draws)
    except np.linalg.LinAlgError:
        # jitter
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
        "pre_t": tvals.tolist(),
    }


def linearity_test_S1(m) -> dict:
    """H0: t2026 = 2*t2025 from S1 (dose 10 vs 5 ⇒ ATT_2026 = 2*ATT_2025 under linearity)."""
    if m is None:
        return {"error": "no model"}
    coef = m.coef()
    if "t2025" not in coef.index or "t2026" not in coef.index:
        return {"error": "missing coefs"}
    b5 = float(coef["t2025"])
    b6 = float(coef["t2026"])
    idx = list(coef.index)
    i5, i6 = idx.index("t2025"), idx.index("t2026")
    V = np.asarray(m._vcov, dtype=float)
    # delta: g = b6 - 2*b5; Var = [ -2, 1 ] V [ -2; 1 ]
    R = np.zeros(len(idx))
    R[i5] = -2.0
    R[i6] = 1.0
    g = b6 - 2.0 * b5
    vg = float(R @ V @ R)
    se = float(np.sqrt(max(vg, 0.0)))
    if se > 0:
        z = g / se
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
    else:
        z, p = np.nan, np.nan
    # also try Wald F
    wald_p = np.nan
    try:
        Rm = R.reshape(1, -1)
        w = m.wald_test(R=Rm, q=np.array([0.0]), distribution="chi2")
        wald_p = float(w.get("p_value", w.iloc[-1] if hasattr(w, "iloc") else np.nan))
    except Exception as e:
        wald_p = f"fail:{e}"
    return {
        "b2025": b5,
        "b2026": b6,
        "g_b6_minus_2b5": g,
        "se_g": se,
        "z": z,
        "p_delta": p,
        "wald": wald_p,
        "ratio_b6_over_b5": (b6 / b5) if b5 != 0 else np.nan,
    }


def plot_es(es_df: pd.DataFrame, outcome: str, title_suffix: str, path: Path):
    sub = es_df.loc[es_df["outcome"] == outcome].sort_values("k")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="gray", ls="--", lw=0.8)  # between k=-1 and 0
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
    ax.set_ylabel("Coefficient (per unit eventual dose)")
    ax.set_title(
        f"S3 ES ({outcome}) — eventual dose D̄∈{{0,10}}\n"
        f"ITT of listed class, not ACRT unless linearity+SPT — {title_suffix}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_forest_S1(s1_df: pd.DataFrame, path: Path):
    # γ2025/γ2026 for 4 outcomes
    fig, ax = plt.subplots(figsize=(8, 5))
    outcomes = ["saldo", "admitidos", "desligados", "log_salario"]
    ylabels = []
    y = 0
    ys, coefs, ses, colors = [], [], [], []
    for o in outcomes:
        for term, col in [("t2025", "C0"), ("t2026", "C1")]:
            r = s1_df.loc[(s1_df["outcome"] == o) & (s1_df["term"] == term)]
            if r.empty:
                continue
            ys.append(y)
            coefs.append(float(r["coef"].iloc[0]))
            ses.append(float(r["se"].iloc[0]))
            colors.append(col)
            ylabels.append(f"{o} {term}")
            y += 1
    ax.errorbar(coefs, ys, xerr=[1.96 * s for s in ses], fmt="o", color="black", ecolor="gray", capsize=3)
    for yi, c, col in zip(ys, coefs, colors):
        ax.plot(c, yi, "o", color=col)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("ATT (binary treated × year)")
    ax.set_title("S1 forest — ITT of listed class, not ACRT unless linearity+SPT")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    print("=== Cristine ATT-by-year pipeline ===", flush=True)
    info = build_panels()
    print(json.dumps(info, indent=2), flush=True)

    flow_cols = [
        "cnae4",
        "municipio",
        "secao",
        "ano",
        "mes_cal",
        "time_id",
        "k",
        "tratado",
        "dose",
        "dose_bar",
        "t2025",
        "t2026",
        "dose_2025",
        "dose_2026",
        "saldo",
        "admitidos",
        "desligados",
    ]

    # Resume: skip S1/S2 if already saved
    s1_path = OUT / "tab_S1_year_att.csv"
    SKIP_S1S2 = s1_path.exists() and (OUT / "tab_S2_dose_year.csv").exists() and (OUT / "tab_linearidade.csv").exists()

    print("Loading balanced flows...", flush=True)
    flows = load_flows(flow_cols)
    print(f"flows N={len(flows)}", flush=True)
    _mem("flows loaded")

    outcomes_flow = ["saldo", "admitidos", "desligados"]
    print("Loading wage...", flush=True)
    wage = load_wage()
    print(f"wage N={len(wage)}", flush=True)

    if SKIP_S1S2:
        print("SKIP S1/S2 — loading existing CSVs", flush=True)
        s1_df = pd.read_csv(OUT / "tab_S1_year_att.csv")
        s2_df = pd.read_csv(OUT / "tab_S2_dose_year.csv")
        s2_check = pd.read_csv(OUT / "tab_S2_vs_S1_check.csv").to_dict("records")
        lin_rows = pd.read_csv(OUT / "tab_linearidade.csv").to_dict("records")
        s1_models = {}
    else:
        s1_rows = []
        s1_models = {}
        s2_rows = []
        lin_rows = []
        for y in outcomes_flow:
            rows, m = run_S1(flows, y)
            s1_rows.extend(rows)
            s1_models[y] = m
            lin = linearity_test_S1(m)
            lin["outcome"] = y
            lin["spec"] = "S1"
            lin_rows.append(lin)
            rows2, m2 = run_S2(flows, y)
            s2_rows.extend(rows2)
            del m2
            gc.collect()
        rows, m = run_S1(wage, "log_salario")
        s1_rows.extend(rows)
        s1_models["log_salario"] = m
        lin = linearity_test_S1(m)
        lin["outcome"] = "log_salario"
        lin["spec"] = "S1"
        lin_rows.append(lin)
        rows2, m2 = run_S2(wage, "log_salario")
        s2_rows.extend(rows2)
        del m2
        gc.collect()
        s1_df = pd.DataFrame(s1_rows)
        s2_df = pd.DataFrame(s2_rows)
        s1_df.to_csv(OUT / "tab_S1_year_att.csv", index=False)
        s2_df.to_csv(OUT / "tab_S2_dose_year.csv", index=False)
        s2_check = []
        for y in ["saldo", "admitidos", "desligados", "log_salario"]:
            b5 = s1_df.loc[(s1_df.outcome == y) & (s1_df.term == "t2025"), "coef"]
            b6 = s1_df.loc[(s1_df.outcome == y) & (s1_df.term == "t2026"), "coef"]
            g5 = s2_df.loc[(s2_df.outcome == y) & (s2_df.term == "dose_2025"), "coef"]
            g6 = s2_df.loc[(s2_df.outcome == y) & (s2_df.term == "dose_2026"), "coef"]
            if b5.empty or g5.empty:
                s2_check.append({"outcome": y, "ok": False, "note": "missing coefs"})
                continue
            b5, b6, g5, g6 = map(float, (b5.iloc[0], b6.iloc[0], g5.iloc[0], g6.iloc[0]))
            ok5 = np.isclose(b5, 5 * g5, rtol=1e-4, atol=1e-6)
            ok6 = np.isclose(b6, 10 * g6, rtol=1e-4, atol=1e-6)
            s2_check.append(
                {
                    "outcome": y,
                    "b2025": b5,
                    "g2025": g5,
                    "5g": 5 * g5,
                    "match_b_eq_5g": bool(ok5),
                    "b2026": b6,
                    "g2026": g6,
                    "10g": 10 * g6,
                    "match_b_eq_10g": bool(ok6),
                    "note": "Identity is β=level*γ (ATT). User text γ≈5β is inverted; we check β≈5γ / β≈10γ.",
                }
            )
            if not (ok5 and ok6):
                print(f"S2 mismatch for {y}: b5={b5} vs 5g={5*g5}; b6={b6} vs 10g={10*g6}", flush=True)
        pd.DataFrame(s2_check).to_csv(OUT / "tab_S2_vs_S1_check.csv", index=False)
        pd.DataFrame(lin_rows).to_csv(OUT / "tab_linearidade.csv", index=False)

    # ---- S3 eventual-dose ES base 2024 (MAIN) ----
    print("=== S3 eventual-dose ES base 2024 ===", flush=True)
    es_all = []
    tmax_rows = []
    for y in outcomes_flow:
        # Need dose_bar, k, FEs, y
        es_df, m, levels, cols_add = run_S3_dose_bar(flows, y, base_year=2024)
        es_all.append(es_df)
        # pre = k in 2023 => original k in [-24,-13]; after make_kes base2024 those stay
        pre_names = [f"k_es::{kk}:dose_bar" for kk in levels if -24 <= kk <= -13]
        tm = studentized_max_test(m, pre_names)
        tm["outcome"] = y
        tm["spec"] = "S3_dose_bar_base2024"
        tm["note"] = "Tests PT in Y(0), NOT SPT. Uses eventual dose_bar so pre interactions ≠ 0."
        tmax_rows.append(tm)
        del m
        gc.collect()

    es_df_w, m_w, levels_w, cols_w = run_S3_dose_bar(wage, "log_salario", base_year=2024)
    es_all.append(es_df_w)
    pre_w = [f"k_es::{kk}:dose_bar" for kk in levels_w if -24 <= kk <= -13]
    tm = studentized_max_test(m_w, pre_w)
    tm["outcome"] = "log_salario"
    tm["spec"] = "S3_dose_bar_base2024"
    tm["note"] = "Tests PT in Y(0), NOT SPT. Wage sample drops 202308-09."
    tmax_rows.append(tm)
    del m_w
    gc.collect()

    es2024 = pd.concat(es_all, ignore_index=True)
    es2024.to_csv(OUT / "tab_es_dose_base2024.csv", index=False)
    # serialize tmax (drop long lists partially)
    tmax_out = []
    for r in tmax_rows:
        rr = {k: v for k, v in r.items() if k not in ("pre_names", "pre_t")}
        rr["pre_names"] = json.dumps(r.get("pre_names", []))
        tmax_out.append(rr)
    pd.DataFrame(tmax_out).to_csv(OUT / "tab_tmax_S3.csv", index=False)

    for y in ["saldo", "admitidos", "desligados", "log_salario"]:
        plot_es(es2024, y, "base 2024", OUT / f"fig_es_S3_base2024_{y}.png")
    plot_forest_S1(s1_df, OUT / "fig_forest_S1.png")

    # S3 base 2023 (also)
    print("=== S3 base 2023 ===", flush=True)
    es23 = []
    for y in outcomes_flow:
        es_df, m, _, _ = run_S3_dose_bar(flows, y, base_year=2023)
        es23.append(es_df)
        del m
        gc.collect()
    es_df, m, _, _ = run_S3_dose_bar(wage, "log_salario", base_year=2023)
    es23.append(es_df)
    del m
    gc.collect()
    pd.concat(es23, ignore_index=True).to_csv(OUT / "tab_es_dose_base2023.csv", index=False)

    # S4 annex binary ES base 2024 — saldo only to save time/mem (or all 4)
    print("=== S4 binary ES annex ===", flush=True)
    s4 = []
    for y in outcomes_flow:
        s4.append(run_S4_binary(flows, y, 2024))
        gc.collect()
    s4.append(run_S4_binary(wage, "log_salario", 2024))
    pd.concat(s4, ignore_index=True).to_csv(OUT / "tab_es_binary_base2024_S4.csv", index=False)
    gc.collect()

    # S5: S1 + municipio^mes_cal
    print("=== S5 municipio^mes_cal ===", flush=True)
    s5_rows = []
    fe5 = "cnae4^mes_cal + municipio^mes_cal + time_id"
    try:
        for y in outcomes_flow:
            rows, m = run_S1(flows, y, fe=fe5, tag="S5")
            s5_rows.extend(rows)
            del m
            gc.collect()
        rows, m = run_S1(wage, "log_salario", fe=fe5, tag="S5")
        s5_rows.extend(rows)
        del m
        s5_note = "ok"
    except Exception as e:
        s5_note = f"OOM/skip: {e}"
        print(s5_note, flush=True)
    if s5_rows:
        pd.DataFrame(s5_rows).to_csv(OUT / "tab_S5.csv", index=False)
    (OUT / "S5_status.txt").write_text(s5_note)

    # S6: S1 + municipio^ano
    print("=== S6 municipio^ano ===", flush=True)
    s6_rows = []
    fe6 = "cnae4^mes_cal + municipio^ano + time_id"
    for y in outcomes_flow:
        rows, m = run_S1(flows, y, fe=fe6, tag="S6")
        s6_rows.extend(rows)
        del m
        gc.collect()
    rows, m = run_S1(wage, "log_salario", fe=fe6, tag="S6")
    s6_rows.extend(rows)
    del m
    pd.DataFrame(s6_rows).to_csv(OUT / "tab_S6.csv", index=False)
    gc.collect()

    # Placebo: through 202412; fake treated×2024
    print("=== Placebo 2024 ===", flush=True)
    pre = flows.loc[flows["ano"] <= 2024].copy()
    pre["t_placebo"] = ((pre["ano"] == 2024) & (pre["tratado"] == 1)).astype(np.int8)
    # ref 2023: only estimate placebo year vs 2023
    placebo_rows = []
    for y in outcomes_flow:
        fml = f"{y} ~ t_placebo | {FE_PREF}"
        # restrict to 2023-2024 already
        m = fit(fml, pre, f"placebo/{y}")
        placebo_rows.extend(tidy_row(m, ["t_placebo"], y, "placebo2024"))
        del m
        gc.collect()
    # wage placebo (wage already dropped 08-09 2023)
    wp = wage.loc[wage["ano"] <= 2024].copy()
    wp["t_placebo"] = ((wp["ano"] == 2024) & (wp["tratado"] == 1)).astype(np.int8)
    m = fit(f"log_salario ~ t_placebo | {FE_PREF}", wp, "placebo/log_salario")
    placebo_rows.extend(tidy_row(m, ["t_placebo"], "log_salario", "placebo2024"))
    del m, pre, wp
    pd.DataFrame(placebo_rows).to_csv(OUT / "tab_placebo2024.csv", index=False)
    gc.collect()

    # Robustness S1 + T_max of S3
    print("=== Robustness ===", flush=True)
    # treated cnae2 divisions
    flows["cnae2"] = flows["cnae4"].astype(str).str[:2]
    treated_div = set(flows.loc[flows["tratado"] == 1, "cnae2"].astype(str).unique())
    rob_specs = {
        "R1_same_cnae2": flows["cnae2"].astype(str).isin(treated_div),
        "R2_drop_F": flows["secao"].astype(str) != "F",
        "R3_drop_FCN": ~flows["secao"].astype(str).isin(["F", "C", "N"]),
    }
    rob_s1 = []
    rob_tmax = []
    for rname, mask in rob_specs.items():
        sub = flows.loc[mask].copy()
        print(f"  {rname} N={len(sub)}", flush=True)
        for y in outcomes_flow:
            rows, m = run_S1(sub, y, tag=rname)
            rob_s1.extend(rows)
            del m
            gc.collect()
        # T_max on saldo S3
        es_df, m, levels, cols_add = run_S3_dose_bar(sub, "saldo", base_year=2024)
        pre_names = [f"k_es::{kk}:dose_bar" for kk in levels if -24 <= kk <= -13]
        tm = studentized_max_test(m, pre_names)
        tm["outcome"] = "saldo"
        tm["spec"] = rname
        rob_tmax.append({k: v for k, v in tm.items() if k not in ("pre_names", "pre_t")})
        del m, sub
        gc.collect()

    # wage robustness R1-R3 for S1 only
    wage["cnae2"] = wage["cnae4"].astype(str).str[:2]
    w_rob = {
        "R1_same_cnae2": wage["cnae2"].astype(str).isin(treated_div),
        "R2_drop_F": wage["secao"].astype(str) != "F",
        "R3_drop_FCN": ~wage["secao"].astype(str).isin(["F", "C", "N"]),
    }
    for rname, mask in w_rob.items():
        sub = wage.loc[mask]
        rows, m = run_S1(sub, "log_salario", tag=rname)
        rob_s1.extend(rows)
        del m
        gc.collect()

    pd.DataFrame(rob_s1).to_csv(OUT / "tab_robustez_S1.csv", index=False)
    pd.DataFrame(rob_tmax).to_csv(OUT / "tab_robustez_tmax.csv", index=False)

    # Optional discarded old anchor
    print("=== Old anchor (discarded) ===", flush=True)
    try:
        rows, m = run_S1(flows, "saldo", fe="cnae4 + municipio + time_id", tag="OLD_anchor_discarded")
        pd.DataFrame(rows).to_csv(OUT / "tab_old_anchor_discarded.csv", index=False)
        del m
    except Exception as e:
        print("old anchor fail", e, flush=True)

    # RESUMO
    write_resumo(info, s1_df, s2_df, s2_check, lin_rows, tmax_rows, placebo_rows, s5_note)
    print("DONE", flush=True)
    print("Outputs in", OUT, flush=True)


def write_resumo(info, s1_df, s2_df, s2_check, lin_rows, tmax_rows, placebo_rows, s5_note):
    lines = []
    lines.append("# RESUMO — Cristine / Callaway ATT-por-ano (Lei 14.973/2024)\n")
    lines.append("## Correção S3 (Econometrics Guy)\n")
    lines.append("Dose calendário = 0 para todos em 2023–24 ⇒ `1{k}·Dose ≡ 0` no pré ⇒ PT/`T_max` impossíveis.")
    lines.append("**S3 usa dose eventual** `dose_bar = 10·1{tratado}` (time-invariant) × `i(k_es)` (base 2024 principal; também base 2023).")
    lines.append("`T_max` testa **PT em Y(0), não SPT**. Não reivindicar ACRT a partir de β único em Dose.\n")
    lines.append("## Painel\n")
    lines.append(f"- Pares cnae4×municipio: **{info['n_pairs']}**")
    lines.append(f"- N flows (balanced zeros): **{info['N_flows']}** (N%42={info['N_mod_42']})")
    lines.append(f"- Competências: {info['comps_first']}–{info['comps_last']} (42)")
    lines.append(f"- NEVER_DROP removidos antes do painel: {info['never_drop']}; remanescentes: {info.get('never_drop_present', [])}")
    lines.append(f"- Salário: drop meses {info['salary_drop_months']} **apenas** em regressões de salário")
    lines.append(f"- S3: {info['s3_note']}\n")

    lines.append("## S1 MAIN — ATT por ano (t2025, t2026)\n")
    lines.append("| outcome | t2025 | se | p | t2026 | se | p |")
    lines.append("|---------|-------|----|---|-------|----|---|")
    for y in ["saldo", "admitidos", "desligados", "log_salario"]:
        r5 = s1_df.loc[(s1_df.outcome == y) & (s1_df.term == "t2025")]
        r6 = s1_df.loc[(s1_df.outcome == y) & (s1_df.term == "t2026")]
        if r5.empty:
            continue
        lines.append(
            f"| {y} | {r5.coef.iloc[0]:.4f} | {r5.se.iloc[0]:.4f} | {r5.p.iloc[0]:.4g} | "
            f"{r6.coef.iloc[0]:.4f} | {r6.se.iloc[0]:.4f} | {r6.p.iloc[0]:.4g} |"
        )

    lines.append("\n## Após dropar NEVER\n")
    old_2025, old_2026 = 0.0216, 0.0148
    s5 = s1_df.loc[(s1_df.outcome == "saldo") & (s1_df.term == "t2025")]
    s6 = s1_df.loc[(s1_df.outcome == "saldo") & (s1_df.term == "t2026")]
    if not s5.empty and not s6.empty:
        new_2025, new_2026 = float(s5.coef.iloc[0]), float(s6.coef.iloc[0])
        d5, d6 = new_2025 - old_2025, new_2026 - old_2026
        material = "movimento material" if max(abs(d5), abs(d6)) >= 0.005 else "sem movimento material (|Δ| < 0.005 em ambos)"
        lines.append(f"- saldo: γ_2025={new_2025:.6f}, γ_2026={new_2026:.6f}; antes 0.0216/0.0148; Δ={d5:+.6f}/{d6:+.6f} — {material}.")
    else:
        lines.append("- saldo: coeficientes S1 não disponíveis para comparação.")

    lines.append("\n## S2 vs S1×dose\n")
    lines.append("Identidade correta: β_YYYY ≈ dose_level × γ_YYYY (5 em 2025, 10 em 2026).")
    lines.append("Texto invertido γ≈5β **não** é a identidade; checamos β≈5γ / β≈10γ.\n")
    for c in s2_check:
        lines.append(
            f"- {c['outcome']}: match_b=5g={c.get('match_b_eq_5g')} match_b=10g={c.get('match_b_eq_10g')} "
            f"(b5={c.get('b2025')}, 5g={c.get('5g')}, b6={c.get('b2026')}, 10g={c.get('10g')})"
        )

    lines.append("\n## Linearidade (H0: β_2026 = 2·β_2025 em S1)\n")
    for r in lin_rows:
        lines.append(
            f"- {r.get('outcome')}: g={r.get('g_b6_minus_2b5'):.4g} se={r.get('se_g'):.4g} "
            f"z={r.get('z'):.4g} p_delta={r.get('p_delta')} ratio={r.get('ratio_b6_over_b5')}"
        )

    lines.append("\n## T_max (S3 dose_bar base2024, pre=k∈2023)\n")
    lines.append("Testa PT em Y(0), **não** SPT. Eventual dose D̄∈{0,10} garante 1{k}·D̄ ≠ 0 no pré.\n")
    for r in tmax_rows:
        lines.append(
            f"- {r.get('outcome')}: T_max={r.get('T_max')} p={r.get('p')} crit5%={r.get('crit_5pct')} n_pre={r.get('n_pre')}"
        )

    lines.append("\n## Placebo 2024 (amostra ≤202412; fake tratado×2024)\n")
    pdf = pd.DataFrame(placebo_rows)
    for _, r in pdf.iterrows():
        lines.append(f"- {r['outcome']}: coef={r['coef']:.4f} se={r['se']:.4f} p={r['p']:.4g}")

    lines.append(f"\n## S5 status: {s5_note}\n")
    lines.append("## Avisos\n")
    lines.append("- Não usar β único em Dose como principal.")
    lines.append("- Não reivindicar ACRT/Callaway contínuo sem SPT+linearidade.")
    lines.append("- S3 NÃO usa Dose calendário no pré (seria identicamente zero).")
    lines.append("- FE preferido: cnae4^mes_cal + municipio + time_id. Nunca cnae4^time_id.\n")

    (OUT / "RESUMO_CRISTINE.md").write_text("\n".join(lines))
    print("Wrote RESUMO_CRISTINE.md", flush=True)


if __name__ == "__main__":
    main()
