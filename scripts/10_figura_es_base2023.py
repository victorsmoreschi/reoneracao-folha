#!/usr/bin/env python3
"""Assigned-dose ES with 2023 omitted (2024 free = anticipation check). English. FE_MAIN, CRV1 cnae4.
Source: cgs_rodada2_sat_main/tab_es_dose_base2023.csv. No re-estimation."""
from pathlib import Path
import matplotlib.pyplot as plt, pandas as pd
OUT=Path("/workspace/notebook_outputs/figuras_en_v3")
df=pd.read_csv("/workspace/notebook_outputs/cgs_rodada2_sat_main/tab_es_dose_base2023.csv")
EN={"saldo":"net employment","admitidos":"hires","desligados":"separations","log_salario":"log starting wages"}
for o,en in EN.items():
    s=df[df.outcome==o].sort_values("k")
    fig,ax=plt.subplots(figsize=(9.5,4.8))
    ax.axhline(0,color="black",lw=0.8)
    ax.axvline(-0.5,color="gray",ls="--",lw=0.8)
    ax.axvspan(-12.5,-0.5,color="0.5",alpha=0.10,label="2024 (anticipation)")
    ax.axvspan(-0.5,11.5,color="C0",alpha=0.08,label="2025 (dose 5)")
    ax.axvspan(11.5,18.5,color="C1",alpha=0.10,label="2026 (dose 10)")
    ax.errorbar(s.k,s.coef,yerr=1.96*s.se,fmt="o",ms=3.5,capsize=2,color="black",ecolor="gray")
    ax.set_xlim(-13.5,18.5)
    ax.set_xlabel("Event time k (k=0 at Jan/2025)")
    ax.set_ylabel(r"$\theta_k$ (effect per dose point)")
    ax.set_title(f"Assigned-dose event study, {en}, 2023 as reference year")
    ax.legend(loc="best",fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/f"fig_es_dose_base2023_{o}.png",dpi=140); plt.close(fig)
print("ok")
