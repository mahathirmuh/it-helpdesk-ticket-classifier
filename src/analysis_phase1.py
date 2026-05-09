"""Tahap 1 analysis untuk paper Q2:
1. Per-class F1 per model (SVM vs Hybrid Fusion) — lihat kelas mana yang dapat improvement
2. Paired t-test (per-fold) untuk significance test
3. Confusion matrix Hybrid Fusion vs SVM (top-N kelas yang most-confused)
4. Save semua ke results/analysis_phase1.xlsx + plot
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix


N_FOLDS = 5
FOLD_FILES = [f"results/cobacek_filtered_kfold_fold{i}.xlsx" for i in range(N_FOLDS)]
OUTPUT = "results/analysis_phase1.xlsx"
FIGURE_DIR = Path("results/figures_phase1")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_all_folds() -> List[pd.DataFrame]:
    """Load Predictions_Compare dari semua fold."""
    dfs = []
    for path in FOLD_FILES:
        df = pd.read_excel(path, sheet_name="Predictions_Compare")
        dfs.append(df)
    return dfs


def per_class_f1_table(folds: List[pd.DataFrame], label: str = "category") -> pd.DataFrame:
    """Aggregate per-class F1 across folds untuk tiap model."""
    pred_cols = {
        "SVM":     f"svm_{label}",
        "RF":      f"rf_{label}",
        "LR":      f"lr_{label}",
        "Fusion":  f"fusion_{label}",
    }

    rows = []
    for fold_idx, df in enumerate(folds):
        for model, col in pred_cols.items():
            if col not in df.columns:
                continue
            report = classification_report(df[label], df[col], zero_division=0, output_dict=True)
            for cls, m in report.items():
                if cls in ("accuracy", "macro avg", "weighted avg"):
                    continue
                rows.append({
                    "fold":      fold_idx,
                    "model":     model,
                    "class":     cls,
                    "support":   int(m["support"]),
                    "precision": m["precision"],
                    "recall":    m["recall"],
                    "f1":        m["f1-score"],
                })
    long = pd.DataFrame(rows)

    # Aggregate: mean ± std per (model, class)
    agg = (
        long.groupby(["model", "class"])
        .agg(
            f1_mean=("f1",        "mean"),
            f1_std=("f1",         "std"),
            prec_mean=("precision", "mean"),
            rec_mean=("recall",   "mean"),
            support=("support",   "first"),
        )
        .reset_index()
    )

    # Pivot: rows=class, cols=model F1 mean
    pivot = agg.pivot(index="class", columns="model", values="f1_mean")
    pivot["support"] = agg.groupby("class")["support"].first()
    pivot = pivot.sort_values("support", ascending=False)

    # Tambah kolom delta Fusion - SVM
    if "Fusion" in pivot.columns and "SVM" in pivot.columns:
        pivot["F1_gain_Fusion_vs_SVM"] = pivot["Fusion"] - pivot["SVM"]

    return long, agg, pivot


def paired_ttest(folds: List[pd.DataFrame], label: str = "category") -> pd.DataFrame:
    """Paired t-test antar model per-fold pada Acc dan macro F1."""
    from sklearn.metrics import accuracy_score, f1_score

    metrics_per_fold = []
    pred_cols = {
        "SVM":    f"svm_{label}",
        "RF":     f"rf_{label}",
        "LR":     f"lr_{label}",
        "Fusion": f"fusion_{label}",
    }
    for fold_idx, df in enumerate(folds):
        row = {"fold": fold_idx}
        for model, col in pred_cols.items():
            if col not in df.columns:
                continue
            row[f"{model}_acc"] = accuracy_score(df[label], df[col])
            row[f"{model}_f1"]  = f1_score(df[label], df[col], average="macro", zero_division=0)
        metrics_per_fold.append(row)
    per_fold_df = pd.DataFrame(metrics_per_fold)

    # Paired t-test Fusion vs SVM (Acc & F1)
    tests = []
    for metric_short in ["acc", "f1"]:
        a = per_fold_df[f"Fusion_{metric_short}"].values
        b = per_fold_df[f"SVM_{metric_short}"].values
        t_stat, p_val = stats.ttest_rel(a, b)
        tests.append({
            "comparison": f"Fusion vs SVM ({metric_short.upper()})",
            "fusion_mean": a.mean(),
            "svm_mean":    b.mean(),
            "diff_mean":   (a - b).mean(),
            "t_statistic": t_stat,
            "p_value":     p_val,
            "significant_p<0.05": p_val < 0.05,
            "significant_p<0.01": p_val < 0.01,
        })

    # Compare Fusion vs RF dan Fusion vs LR juga
    for other in ["RF", "LR"]:
        for metric_short in ["acc", "f1"]:
            a = per_fold_df[f"Fusion_{metric_short}"].values
            b = per_fold_df[f"{other}_{metric_short}"].values
            t_stat, p_val = stats.ttest_rel(a, b)
            tests.append({
                "comparison": f"Fusion vs {other} ({metric_short.upper()})",
                "fusion_mean": a.mean(),
                "svm_mean":    b.mean(),
                "diff_mean":   (a - b).mean(),
                "t_statistic": t_stat,
                "p_value":     p_val,
                "significant_p<0.05": p_val < 0.05,
                "significant_p<0.01": p_val < 0.01,
            })
    return per_fold_df, pd.DataFrame(tests)


def plot_per_class_f1(pivot: pd.DataFrame, output_path: str):
    """Bar chart per-class F1 SVM vs Fusion (sorted by support)."""
    classes = pivot.index.tolist()
    svm_f1    = pivot["SVM"].values
    fusion_f1 = pivot["Fusion"].values
    support   = pivot["support"].values

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")

    x = np.arange(len(classes))
    w = 0.35
    ax.bar(x - w/2, svm_f1,    w, label="SVM",    color="#5b9bd5")
    ax.bar(x + w/2, fusion_f1, w, label="Hybrid Fusion", color="#3aa857")

    # Annotate support count di atas
    for i, s in enumerate(support):
        ax.text(i, max(svm_f1[i], fusion_f1[i]) + 0.02, f"n={s}",
                ha="center", fontsize=8, color="#cdd6e0")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", color="#cdd6e0", fontsize=10)
    ax.set_ylabel("F1 Score", color="#cdd6e0", fontsize=12)
    ax.set_title("Per-Class F1: SVM vs Hybrid Fusion (sorted by support)",
                 color="#a5e8c4", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower left", fontsize=10)
    ax.tick_params(colors="#cdd6e0")
    for spine in ax.spines.values():
        spine.set_color("#3a4a5c")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#3a4a5c")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


def plot_minority_class_gain(pivot: pd.DataFrame, output_path: str):
    """Highlight kelas mana yang gain F1 paling besar dari Fusion vs SVM."""
    if "F1_gain_Fusion_vs_SVM" not in pivot.columns:
        return
    gain_df = pivot[["F1_gain_Fusion_vs_SVM", "support"]].copy()
    gain_df = gain_df.sort_values("F1_gain_Fusion_vs_SVM")
    gain_df = gain_df[gain_df["support"] > 0]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#1a2332")

    colors = ["#3aa857" if g >= 0 else "#c1444a" for g in gain_df["F1_gain_Fusion_vs_SVM"]]
    bars = ax.barh(gain_df.index, gain_df["F1_gain_Fusion_vs_SVM"], color=colors)

    for bar, support in zip(bars, gain_df["support"]):
        ax.text(bar.get_width() + (0.005 if bar.get_width() >= 0 else -0.015),
                bar.get_y() + bar.get_height() / 2,
                f"n={support}", va="center", fontsize=8, color="#cdd6e0")

    ax.set_xlabel("F1 Gain (Fusion - SVM)", color="#cdd6e0", fontsize=12)
    ax.set_title("Per-Class F1 Gain: Hybrid Fusion vs SVM\n(positive = Fusion better)",
                 color="#a5e8c4", fontsize=13, fontweight="bold", pad=20)
    ax.axvline(0, color="#cdd6e0", linewidth=1, linestyle="-")
    ax.tick_params(colors="#cdd6e0")
    ax.set_yticklabels(gain_df.index, color="#cdd6e0", fontsize=10)
    for spine in ax.spines.values():
        spine.set_color("#3a4a5c")
    ax.grid(axis="x", linestyle="--", alpha=0.3, color="#3a4a5c")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


def confusion_difference_analysis(folds: List[pd.DataFrame], label: str = "category") -> pd.DataFrame:
    """Cari kelas mana yang paling sering dimisclassify SVM tapi Fusion benar (dan sebaliknya)."""
    rows = []
    for df in folds:
        true = df[label]
        svm_pred    = df[f"svm_{label}"]
        fusion_pred = df[f"fusion_{label}"]

        # Kasus: SVM salah, Fusion benar (Fusion mengkoreksi SVM)
        svm_wrong_fusion_right = (svm_pred != true) & (fusion_pred == true)
        # Kasus: Fusion salah, SVM benar (Fusion merusak)
        svm_right_fusion_wrong = (svm_pred == true) & (fusion_pred != true)

        for cls in true.unique():
            mask_cls = true == cls
            corrected = (svm_wrong_fusion_right & mask_cls).sum()
            broken    = (svm_right_fusion_wrong & mask_cls).sum()
            rows.append({
                "class":         cls,
                "support":       mask_cls.sum(),
                "fusion_corrects_svm":   corrected,
                "fusion_breaks_svm":     broken,
                "net_gain":              corrected - broken,
            })
    df = pd.DataFrame(rows)
    agg = df.groupby("class").agg(
        support=("support", "sum"),
        fusion_corrects_svm=("fusion_corrects_svm", "sum"),
        fusion_breaks_svm=("fusion_breaks_svm", "sum"),
        net_gain=("net_gain", "sum"),
    ).reset_index()
    agg = agg.sort_values("net_gain", ascending=False)
    return agg


def main():
    print("Loading 5 fold predictions...")
    folds = load_all_folds()
    print(f"Loaded {len(folds)} folds, {sum(len(f) for f in folds)} total predictions")

    # === Category analysis ===
    print("\n[Category] Per-class F1 analysis...")
    long_cat, agg_cat, pivot_cat = per_class_f1_table(folds, label="category")

    print("\n[Category] Paired t-test...")
    perfold_cat, ttest_cat = paired_ttest(folds, label="category")
    print(ttest_cat.to_string(index=False))

    print("\n[Category] Confusion difference analysis...")
    confdiff_cat = confusion_difference_analysis(folds, label="category")

    # === Priority analysis ===
    print("\n[Priority] Per-class F1 analysis...")
    long_pri, agg_pri, pivot_pri = per_class_f1_table(folds, label="priority")

    print("\n[Priority] Paired t-test...")
    perfold_pri, ttest_pri = paired_ttest(folds, label="priority")
    print(ttest_pri.to_string(index=False))

    # === Plot ===
    print("\nGenerating plots...")
    plot_per_class_f1(pivot_cat, str(FIGURE_DIR / "per_class_f1_category.png"))
    plot_minority_class_gain(pivot_cat, str(FIGURE_DIR / "f1_gain_category.png"))
    plot_per_class_f1(pivot_pri, str(FIGURE_DIR / "per_class_f1_priority.png"))

    # === Save Excel ===
    print(f"\nSaving to {OUTPUT}...")
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as w:
        # Category sheets
        pivot_cat.to_excel(w,    sheet_name="cat_per_class_f1")
        confdiff_cat.to_excel(w, sheet_name="cat_confusion_diff", index=False)
        ttest_cat.to_excel(w,    sheet_name="cat_paired_ttest", index=False)
        perfold_cat.to_excel(w,  sheet_name="cat_per_fold_metrics", index=False)
        long_cat.to_excel(w,     sheet_name="cat_long_format", index=False)
        # Priority sheets
        pivot_pri.to_excel(w,    sheet_name="pri_per_class_f1")
        ttest_pri.to_excel(w,    sheet_name="pri_paired_ttest", index=False)
        perfold_pri.to_excel(w,  sheet_name="pri_per_fold_metrics", index=False)

    print(f"\nDone. Output: {OUTPUT}")
    print(f"Figures: {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
