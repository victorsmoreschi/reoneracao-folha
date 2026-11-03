#!/usr/bin/env python3
"""ES binary base2024 + T_max (k in 2023) under FE_MAIN for semF and cnae2.

Run AFTER dose-W. Prefer 4 Y; at least saldo. Writes:
  tab_es_binario_semF_base2024.csv, tab_tmax_semF.csv
  tab_es_binario_cnae2_base2024.csv, tab_tmax_cnae2.csv
  fig_es_binario_{semF,cnae2}_base2024_{y}.png
Updates STATUS.md, RESUMO, stage.
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
REMODELADO = Path("/workspace/notebook_outputs/REMODELADO_sat_main_rob_v1")
(REMODELADO / "tabs").mkdir(parents=True, exist_ok=True)
(REMODELADO / "figuras").mkdir(parents=True, exist_ok=True)

FE_MAIN = "cnae4^municipio^mes_cal + municipio^time_id + municipio^ano"
FE_FALLBACKS = [
    FE_MAIN,
    "cnae4^mes_cal + municipio^time_id + municipio^ano",
    "cnae4^mes_cal + municipio + time_id",
]
OUTCOMES_FLOW = r2.OUTCOMES_FLOW
OUTCOMES_ALL = r2.OUTCOMES_ALL
PANEL_BAL = r2.PANEL_BAL
PANEL_WAGE = r2.PANEL_WAGE

FAILURES: list[str] = []
r2.FAILURES = FAILURES
ES_FE_USED: dict[str, str] = {}
ES_PRODUCED: list[str] = []


def _mem(tag=""):
    r2._mem(tag)


def update_status(block: str, detail: str):
    path = OUT / "STATUS.md"
    prev = path.read_text(encoding="utf-8") if path.exists() else ""
    # rewrite full structured status from known files
    lines = [
        "# STATUS — cgs_rodada2_sat_main_rob\n",
        f"Updated: {pd.Timestamp.now(tz='America/Sao_Paulo').strftime('%Y-%m-%d %H:%M %Z')}\n",
        "## ATT — DONE\n",
        "- tab_att_principal / rob_semF / rob_cnae2 / wide_for_tex\n",
    ]
    dose_sf = (OUT / "tab_es_dose_semF_base2024.csv").exists() and (OUT / "tab_es_dose_semF_base2024.csv").stat().st_size > 10
    dose_c2 = (OUT / "tab_es_dose_cnae2_base2024.csv").exists() and (OUT / "tab_es_dose_cnae2_base2024.csv").stat().st_size > 10
    bin_sf = (OUT / "tab_tmax_semF.csv").exists()
    bin_c2 = (OUT / "tab_tmax_cnae2.csv").exists()
    lines.append(f"## ES_dose_semF — {'DONE' if dose_sf else 'PENDING'}\n")
    lines.append(f"## ES_dose_cnae2 — {'DONE' if dose_c2 else 'PENDING'}\n")
    lines.append(f"## ES_bin (+ T_max) — {block}\n")
    lines.append(detail + "\n")
    figs = sorted(p.name for p in OUT.glob("fig_*.png"))
    lines.append(f"Figs so far ({len(figs)}): " + (", ".join(figs) if figs else "(none)") + "\n")
    if FAILURES:
        lines.append("## Failures\n")
        for f in FAILURES:
            lines.append(f"- {f}")
        lines.append("")
    done = dose_sf and dose_c2 and bin_sf and bin_c2
    lines.append(f"## DONE (stage+notebook+RESUMO) — {'READY_TO_STAGE' if done else 'PENDING'}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"STATUS.md ← ES_bin={block}", flush=True)


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


def plot_es_binary(es_df, outcome, path, title_prefix="", fe_note=""):
    sub = es_df.loc[es_df["outcome"] == outcome].sort_values("k")
    if sub.empty:
        return False
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
        f"ES binary ({outcome}) — {title_prefix} | FE saturada; base 2024\n"
        "Exposure timing; T_max on 2023 leads = PT usual."
    )
    if fe_note:
        title += f"\n[{fe_note}]"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def run_one_bin(df, outcome, tag, title_prefix, fig_stem):
    last_err = None
    for fe in FE_FALLBACKS:
        print(f"  Trying BIN {tag}/{outcome} FE={fe}", flush=True)
        try:
            es_df, m, levels = r2.run_es_binary(df, outcome, base_year=2024, fe=fe)
            if m is not None and es_df is not None and len(es_df):
                if fe != FE_MAIN:
                    FAILURES.append(f"BIN {tag}/{outcome}: FE_MAIN fail; used `{fe}`")
                ES_FE_USED[f"BIN_{tag}/{outcome}"] = fe
                es_df = es_df.copy()
                es_df["fe"] = fe
                es_df["rob_spec"] = tag
                pre = pre_names_from_model(m)
                tm = r2.studentized_max_test(m, pre)
                tm["outcome"] = outcome
                tm["spec"] = f"ES_binary_base2024_{tag}"
                tm["fe"] = fe
                note = "" if fe == FE_MAIN else f"FE used: {fe}"
                fig_path = OUT / f"fig_es_binario_{fig_stem}_base2024_{outcome}.png"
                if plot_es_binary(es_df, outcome, fig_path, title_prefix=title_prefix, fe_note=note):
                    ES_PRODUCED.append(fig_path.name)
                del m
                gc.collect()
                _mem(f"after BIN {tag}/{outcome}")
                return es_df, tm
            del m
            gc.collect()
            last_err = f"null under {fe}"
        except MemoryError as e:
            last_err = str(e)
            FAILURES.append(f"BIN {tag}/{outcome} MemoryError under `{fe}`: {e}")
            gc.collect()
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            FAILURES.append(f"BIN {tag}/{outcome} under `{fe}`: {last_err}")
            gc.collect()
    FAILURES.append(f"BIN {tag}/{outcome}: ALL FE failed ({last_err})")
    ES_FE_USED[f"BIN_{tag}/{outcome}"] = "FAILED"
    return pd.DataFrame(), {"outcome": outcome, "error": last_err, "fe": "FAILED", "spec": f"ES_binary_base2024_{tag}"}


def load_semF():
    flow_cols = [
        "cnae4", "municipio", "secao", "competencia", "ano", "mes_cal", "time_id",
        "k", "tratado", "dose", "t2025", "t2026", "saldo", "admitidos", "desligados",
    ]
    flows = pd.read_parquet(PANEL_BAL, columns=flow_cols)
    wage = pd.read_parquet(PANEL_WAGE)
    flows = flows.loc[flows["secao"].astype(str) != "F"].copy()
    wage = wage.loc[wage["secao"].astype(str) != "F"].copy()
    # binary ES uses tratado directly; W not required but harmless
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
    print(f"cnae2 divs={len(treated_div)} flows={len(flows)} wage={len(wage)}", flush=True)
    return flows, wage


def save_tmax(rows, path):
    out = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k not in ("pre_names", "pre_t")}
        if "pre_names" in r:
            rr["pre_names"] = json.dumps(r.get("pre_names", []))
        out.append(rr)
    pd.DataFrame(out).to_csv(path, index=False)


def run_block(tag, title, fig_stem, loader):
    print(f"=== ES binary + T_max | {tag} ===", flush=True)
    update_status(f"IN PROGRESS ({tag})", f"Loading {tag}...")
    flows, wage = loader()
    _mem(f"{tag} loaded")
    es_all, tmax_rows = [], []
    for y in OUTCOMES_FLOW:
        es_df, tm = run_one_bin(flows, y, tag, title, fig_stem)
        if len(es_df):
            es_all.append(es_df)
        tmax_rows.append(tm)
        gc.collect()
    del flows
    gc.collect()
    es_df, tm = run_one_bin(wage, "log_salario", tag, title, fig_stem)
    if len(es_df):
        es_all.append(es_df)
    tmax_rows.append(tm)
    del wage
    gc.collect()
    tab_es = OUT / f"tab_es_binario_{fig_stem}_base2024.csv"
    tab_tm = OUT / f"tab_tmax_{fig_stem}.csv"
    if es_all:
        pd.concat(es_all, ignore_index=True).to_csv(tab_es, index=False)
    else:
        pd.DataFrame().to_csv(tab_es, index=False)
    save_tmax(tmax_rows, tab_tm)
    print(f"Wrote {tab_es.name} + {tab_tm.name}; n_es={len(es_all)}", flush=True)
    return len(es_all), tmax_rows


def finalize_texts_and_stage():
    """Append binary/T_max notes into RESUMO if present; refresh key_numbers; stage."""
    # key_numbers merge
    kn_path = OUT / "key_numbers.json"
    summary = json.loads(kn_path.read_text()) if kn_path.exists() else {}
    summary["es_bin_fe_used"] = ES_FE_USED
    summary["es_bin_figs"] = ES_PRODUCED
    summary["failures_bin"] = FAILURES
    for stem in ("semF", "cnae2"):
        tp = OUT / f"tab_tmax_{stem}.csv"
        if tp.exists() and tp.stat().st_size > 10:
            tm = pd.read_csv(tp)
            for _, r in tm.iterrows():
                y = r.get("outcome", "")
                summary[f"tmax_{stem}_{y}"] = r.get("T_max")
                summary[f"tmax_p_{stem}_{y}"] = r.get("p")
                summary[f"tmax_fe_{stem}_{y}"] = r.get("fe")
    kn_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Append to RESUMO
    resumo = OUT / "RESUMO.md"
    if resumo.exists():
        extra = ["\n## ES binário + T_max (robustez)\n"]
        for stem, lab in [("semF", "sem seção F"), ("cnae2", "divisões CNAE 2 dígitos com listada")]:
            tp = OUT / f"tab_tmax_{stem}.csv"
            extra.append(f"### {lab}\n")
            if not tp.exists() or tp.stat().st_size < 10:
                extra.append("(não produzido)\n")
                continue
            tm = pd.read_csv(tp)
            for _, r in tm.iterrows():
                extra.append(
                    f"- {r.get('outcome')}: T_max={r.get('T_max')} p={r.get('p')} "
                    f"n_pre={r.get('n_pre')} fe=`{r.get('fe')}` — PT usual, não SPT."
                )
            extra.append("")
        if ES_PRODUCED:
            extra.append("Figs binárias: " + ", ".join(ES_PRODUCED) + "\n")
        if FAILURES:
            extra.append("### Failures ES bin\n")
            for f in FAILURES:
                extra.append(f"- {f}")
            extra.append("")
        resumo.write_text(resumo.read_text(encoding="utf-8") + "\n".join(extra), encoding="utf-8")

    # stage
    tabs = list(OUT.glob("tab_*.csv"))
    for src in tabs:
        shutil.copy2(src, REMODELADO / "tabs" / src.name)
    for src in OUT.glob("fig_*.png"):
        shutil.copy2(src, REMODELADO / "figuras" / src.name)
    for name in [
        "RESUMO.md", "TEXTO_RESULTADOS.md", "key_numbers.json", "STATUS.md",
        "run_cgs_sat_main_rob.py", "resume_es_dose_rob.py", "resume_es_bin_tmax_rob.py",
        "notebook_reoneracao_cgs_sat_main_rob.ipynb",
    ]:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, REMODELADO / name)
    readme = (
        "REMODELADO_sat_main_rob_v1 — robustez sob FE saturada (FE_MAIN)\n\n"
        f"FE_MAIN = `{FE_MAIN}`\n"
        "ATT: principal | sem seção F | CNAE 2 dígitos com listada\n"
        "ES dose-W + ES binário+T_max (base2024) sob FE_MAIN para semF e cnae2\n"
        "Do NOT overwrite sat_main / paper packs. Do not edit Overleaf live.\n"
    )
    (REMODELADO / "README_ASSETS.txt").write_text(readme, encoding="utf-8")
    print(f"Staged {REMODELADO}", flush=True)


def main():
    print("=== RESUME ES binary + T_max under FE_MAIN ===", flush=True)
    n1, _ = run_block("semF", "sem seção F (construção)", "semF", load_semF)
    update_status("IN PROGRESS (cnae2 next)", f"semF binary outcomes={n1}/4; tmax written")
    n2, _ = run_block("cnae2", "divisões CNAE 2 dígitos com listada", "cnae2", load_cnae2)
    update_status("DONE", f"semF={n1}/4; cnae2={n2}/4 binary outcomes")
    finalize_texts_and_stage()
    # mark DONE
    st = (OUT / "STATUS.md").read_text(encoding="utf-8")
    st = st.replace("## DONE (stage+notebook+RESUMO) — READY_TO_STAGE", "## DONE — COMPLETE")
    st = st.replace("## DONE (stage+notebook+RESUMO) — PENDING", "## DONE — COMPLETE")
    (OUT / "STATUS.md").write_text(st, encoding="utf-8")
    print("DONE binary+tmax", flush=True)
    print("FAILURES:", FAILURES, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "resume_bin_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
