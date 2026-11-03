#!/usr/bin/env python3
"""Table 2 (raw descriptives) recomputed from the estimation panels. No estimation."""
import pandas as pd
BAL="/workspace/notebook_outputs/cristine_callaway/painel_balanced_flows.parquet"
WAGE="/workspace/notebook_outputs/cristine_callaway/painel_wage.parquet"
f=pd.read_parquet(BAL,columns=["cnae4","municipio","tratado","saldo","admitidos","desligados"])
w=pd.read_parquet(WAGE)
wcol=[c for c in w.columns if "log" in c and "sal" in c][0]
rows=[]
for g,lab in [(1,"listed"),(0,"control")]:
    s=f[f.tratado==g]; ws=w[w.tratado==g]
    rows.append(dict(group=lab,classes=s.cnae4.nunique(),pairs=len(s[["cnae4","municipio"]].drop_duplicates()),
        flow_cells=len(s),mean_net_employment=s.saldo.mean(),mean_hires=s.admitidos.mean(),
        mean_separations=s.desligados.mean(),mean_log_starting_wage=ws[wcol].mean(),wage_cells=len(ws)))
out=pd.DataFrame(rows); print(out.T); out.to_csv("tab_descritivas_recalc.csv",index=False)
