#!/usr/bin/env python3
"""Replot CURRENT-model figures with cleaner English titles into figuras_en_v2/.
Does NOT overwrite sources; does NOT re-estimate; does NOT touch the paper.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("/workspace/notebook_outputs/figuras_en_v2")
MAIN = Path("/workspace/notebook_outputs/cgs_rodada2_sat_main")
ROB = Path("/workspace/notebook_outputs/cgs_rodada2_sat_main_rob")
REV = Path("/workspace/notebook_outputs/reviewer_pack_v1")

# Match paper English outcome names (main.tex): net employment, hires,
# separations, log starting wages. User map aligned where identical.
OUTCOME_EN = {
    "saldo": "net employment",
    "admitidos": "hires",
    "desligados": "separations",
    "log_salario": "log starting wages",
}

# Robustness title prefixes (English)
PREFIX_EN = {
    "semF": "without construction",
    "cnae2": "2-digit CNAE divisions with a listed class",
}

OUTCOMES = ["saldo", "admitidos", "desligados", "log_salario"]

made = []
skipped = []


def plot_es_dose(es_df: pd.DataFrame, outcome: str, path: Path, title_prefix: str | None = None):
    sub = es_df.loc[es_df["outcome"] == outcome].sort_values("k")
    if sub.empty:
        return False
    en = OUTCOME_EN[outcome]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="gray", ls="--", lw=0.8)
    ax.axvspan(-0.5, 11.5, color="C0", alpha=0.08, label="2025 (dose 5)")
    ax.axvspan(11.5, 18.5, color="C1", alpha=0.10, label="2026 (dose 10)")
    ax.errorbar(
        sub["k"], sub["coef"], yerr=1.96 * sub["se"],
        fmt="o", ms=3.5, capsize=2, color="black", ecolor="gray",
    )
    ax.set_xlabel("Event time k (k=0 at Jan/2025)")
    ax.set_ylabel(r"$\theta_k$ (effect per dose point)")
    if title_prefix is None:
        title = f"Assigned-dose event study, {en}"
    else:
        title = f"Assigned-dose event study, {en}, {title_prefix}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def plot_es_binary(es_df: pd.DataFrame, outcome: str, path: Path, title_prefix: str | None = None, principal: bool = False):
    sub = es_df.loc[es_df["outcome"] == outcome].sort_values("k")
    if sub.empty:
        return False
    en = OUTCOME_EN[outcome]
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
    if principal:
        title = f"Binary event study, {en}"
    else:
        title = f"Binary event study, {en}, {title_prefix}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def plot_raw_means_two_panel(mm: pd.DataFrame, path: Path, outcomes: list[str]):
    """Two-panel listed vs controls raw means (same style as original)."""
    n = len(outcomes)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    # map group labels
    treated = mm.loc[mm["group"] == "treated_listed"].sort_values("competencia")
    control = mm.loc[mm["group"] == "control"].sort_values("competencia")
    x_t = treated["competencia"].astype(str)
    x_c = control["competencia"].astype(str)
    # tick every 3 months
    all_comp = sorted(mm["competencia"].unique())
    tick_comp = [c for i, c in enumerate(all_comp) if i % 3 == 0]
    tick_labels = [str(c) for c in tick_comp]

    for ax, ycol in zip(axes, outcomes):
        en = OUTCOME_EN[ycol]
        ax.plot(
            x_t, treated[ycol].values,
            "o-", color="C0", ms=3.5, lw=1.4, label="listed (treated)",
        )
        ax.plot(
            x_c, control[ycol].values,
            "o--", color="C1", ms=3.5, lw=1.4, label="controls",
        )
        # vertical lines at 202501 and 202601
        for v in ("202501", "202601"):
            ax.axvline(v, color="gray", ls=":", lw=1.0)
        ax.set_ylabel(en)
        ax.set_title(f"Mean {en}, listed versus controls")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_xticks([str(c) for c in tick_comp])
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("month")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_raw_means_single(mm: pd.DataFrame, path: Path, ycol: str):
    """Extra single-panel raw-means figure (same style)."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    treated = mm.loc[mm["group"] == "treated_listed"].sort_values("competencia")
    control = mm.loc[mm["group"] == "control"].sort_values("competencia")
    x_t = treated["competencia"].astype(str)
    x_c = control["competencia"].astype(str)
    en = OUTCOME_EN[ycol]
    ax.plot(x_t, treated[ycol].values, "o-", color="C0", ms=3.5, lw=1.4, label="listed (treated)")
    ax.plot(x_c, control[ycol].values, "o--", color="C1", ms=3.5, lw=1.4, label="controls")
    for v in ("202501", "202601"):
        ax.axvline(v, color="gray", ls=":", lw=1.0)
    all_comp = sorted(mm["competencia"].unique())
    tick_comp = [c for i, c in enumerate(all_comp) if i % 3 == 0]
    ax.set_xticks([str(c) for c in tick_comp])
    ax.set_xticklabels([str(c) for c in tick_comp], rotation=45, ha="right")
    ax.set_ylabel(en)
    ax.set_xlabel("month")
    ax.set_title(f"Mean {en}, listed versus controls")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def require_src(png: Path) -> bool:
    if not png.exists():
        skipped.append(f"MISSING source png: {png}")
        return False
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Principal dose ----
    csv_dose = MAIN / "tab_es_dose_base2024.csv"
    es_dose = pd.read_csv(csv_dose)
    for y in OUTCOMES:
        src = MAIN / f"fig_es_dose_base2024_{y}.png"
        dst = OUT / f"fig_es_dose_base2024_{y}.png"
        if not require_src(src):
            continue
        if plot_es_dose(es_dose, y, dst, title_prefix=None):
            made.append(str(dst))
        else:
            skipped.append(f"empty CSV rows for {y} in {csv_dose.name}")

    # ---- Principal binary ----
    csv_bin = MAIN / "tab_es_binario_base2024.csv"
    es_bin = pd.read_csv(csv_bin)
    for y in OUTCOMES:
        src = MAIN / f"fig_es_binario_base2024_{y}.png"
        dst = OUT / f"fig_es_binario_base2024_{y}.png"
        if not require_src(src):
            continue
        if plot_es_binary(es_bin, y, dst, principal=True):
            made.append(str(dst))
        else:
            skipped.append(f"empty CSV rows for {y} in {csv_bin.name}")

    # ---- Robustness dose + binary ----
    for stem in ("semF", "cnae2"):
        prefix = PREFIX_EN[stem]
        csv_d = ROB / f"tab_es_dose_{stem}_base2024.csv"
        csv_b = ROB / f"tab_es_binario_{stem}_base2024.csv"
        es_d = pd.read_csv(csv_d)
        es_b = pd.read_csv(csv_b)
        for y in OUTCOMES:
            src = ROB / f"fig_es_dose_{stem}_base2024_{y}.png"
            dst = OUT / f"fig_es_dose_{stem}_base2024_{y}.png"
            if not require_src(src):
                continue
            if plot_es_dose(es_d, y, dst, title_prefix=prefix):
                made.append(str(dst))
            else:
                skipped.append(f"empty CSV rows for {y} in {csv_d.name}")

            srcb = ROB / f"fig_es_binario_{stem}_base2024_{y}.png"
            dstb = OUT / f"fig_es_binario_{stem}_base2024_{y}.png"
            if not require_src(srcb):
                continue
            if plot_es_binary(es_b, y, dstb, title_prefix=prefix):
                made.append(str(dstb))
            else:
                skipped.append(f"empty CSV rows for {y} in {csv_b.name}")

    # ---- Raw means (two panels: saldo + admitidos) ----
    src_raw = REV / "figuras" / "fig_raw_means_listed_vs_controls.png"
    mm_path = REV / "tabs" / "tab_monthly_means_raw.csv"
    if not require_src(src_raw):
        pass
    else:
        mm = pd.read_csv(mm_path)
        dst_raw = OUT / "fig_raw_means_listed_vs_controls.png"
        plot_raw_means_two_panel(mm, dst_raw, ["saldo", "admitidos"])
        made.append(str(dst_raw))

        # Extra figures when series exist in CSV but not in original two-panel
        if "desligados" in mm.columns and mm["desligados"].notna().any():
            dst_sep = OUT / "fig_raw_means_listed_vs_controls_separations.png"
            plot_raw_means_single(mm, dst_sep, "desligados")
            made.append(str(dst_sep))
        # log wage not in monthly means CSV — do not invent

    # ---- README ----
    readme = OUT / "README.md"
    lines = [
        "# figuras_en_v2 — cleaner English-title remakes",
        "",
        "Paper not updated; v1 kept as backup.",
        "",
        "Replotted from the same existing CSVs (no re-estimation). Same figure filenames as v1.",
        "",
        "## Title and label notes",
        "",
        "- Event-study titles use the requested exact patterns; 2024 remains the omitted visual gap.",
        "- Dose-band legend entries are exactly `2025 (dose 5)` and `2026 (dose 10)`; binary figures have no dose-band legend labels.",
        "- Raw-means figures use `listed (treated)` and `controls`, with month on the x-axis.",
        "",
        "## old path → new path",
        "",
    ]
    mapping = []
    for y in OUTCOMES:
        mapping.append(
            (f"{MAIN}/fig_es_dose_base2024_{y}.png", f"{OUT}/fig_es_dose_base2024_{y}.png")
        )
        mapping.append(
            (f"{MAIN}/fig_es_binario_base2024_{y}.png", f"{OUT}/fig_es_binario_base2024_{y}.png")
        )
    for stem in ("semF", "cnae2"):
        for y in OUTCOMES:
            mapping.append(
                (
                    f"{ROB}/fig_es_dose_{stem}_base2024_{y}.png",
                    f"{OUT}/fig_es_dose_{stem}_base2024_{y}.png",
                )
            )
            mapping.append(
                (
                    f"{ROB}/fig_es_binario_{stem}_base2024_{y}.png",
                    f"{OUT}/fig_es_binario_{stem}_base2024_{y}.png",
                )
            )
    mapping.append(
        (
            f"{REV}/figuras/fig_raw_means_listed_vs_controls.png",
            f"{OUT}/fig_raw_means_listed_vs_controls.png",
        )
    )
    mapping.append(
        (
            "(extra from tab_monthly_means_raw.csv desligados; no prior png)",
            f"{OUT}/fig_raw_means_listed_vs_controls_separations.png",
        )
    )
    for old, new in mapping:
        lines.append(f"- `{old}` → `{new}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- No log-wage raw-means series in `tab_monthly_means_raw.csv`; no wage raw-means figure invented.")
    lines.append("- Sources (png/csv/tex) and Overleaf were not modified.")
    lines.append("")
    readme.write_text("\n".join(lines), encoding="utf-8")

    print(f"MADE {len(made)} files:")
    for p in made:
        print(" ", p)
    if skipped:
        print(f"SKIPPED {len(skipped)}:")
        for s in skipped:
            print(" ", s)

if __name__ == "__main__":
    main()
