"""Tahap 2 analysis untuk paper Q2:
Compare 3 hybrid architectures: SVM (baseline), Hybrid Fusion, Hybrid Voting.

Input: results/voting_<model>.xlsx (output dari run --enable-voting)
Output: results/analysis_phase2.xlsx + figure
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


FIGURE_DIR = Path("results/figures_phase2")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_voting_predictions(voting_xlsx: str) -> pd.DataFrame:
    return pd.read_excel(voting_xlsx, sheet_name="Predictions_Compare")


def detect_voting_models(df: pd.DataFrame) -> List[str]:
    """Detect model keys dari kolom hybrid_voting_category_<model_key>."""
    pattern = re.compile(r"^hybrid_voting_category_(.+)$")
    return [m.group(1) for col in df.columns if (m := pattern.match(col))]


def per_class_f1(df: pd.DataFrame, true_col: str, pred_col: str) -> Dict[str, float]:
    return classification_report(df[true_col], df[pred_col], zero_division=0, output_dict=True)


def overall_metrics(df: pd.DataFrame, true_col: str, pred_col: str) -> Dict[str, float]:
    return {
        "accuracy":    accuracy_score(df[true_col], df[pred_col]),
        "macro_f1":    f1_score(df[true_col], df[pred_col], average="macro", zero_division=0),
        "weighted_f1": f1_score(df[true_col], df[pred_col], average="weighted", zero_division=0),
    }


def build_comparison_table(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Tabel comparison: SVM, Fusion, GenAI Voter, Voting (per model)."""
    rows = []
    for label in ["category", "priority"]:
        # Baselines
        rows.append({"approach": "SVM",     "label": label, **overall_metrics(df, label, f"svm_{label}")})
        rows.append({"approach": "Fusion",  "label": label, **overall_metrics(df, label, f"fusion_{label}")})
        # Per voting model
        for m in models:
            voter_col = f"genai_voter_{label}_{m}"
            vote_col  = f"hybrid_voting_{label}_{m}"
            if voter_col in df.columns:
                rows.append({"approach": f"GenAI Voter ({m})", "label": label,
                             **overall_metrics(df, label, voter_col)})
            if vote_col in df.columns:
                rows.append({"approach": f"Hybrid Voting ({m})", "label": label,
                             **overall_metrics(df, label, vote_col)})
    return pd.DataFrame(rows)


def per_class_breakdown(df: pd.DataFrame, models: List[str], label: str = "category") -> pd.DataFrame:
    """Per-class F1 untuk SVM, Fusion, dan setiap voting variant."""
    pred_cols = {
        "SVM":    f"svm_{label}",
        "Fusion": f"fusion_{label}",
    }
    for m in models:
        if f"genai_voter_{label}_{m}" in df.columns:
            pred_cols[f"GenAI Voter ({m})"] = f"genai_voter_{label}_{m}"
        if f"hybrid_voting_{label}_{m}" in df.columns:
            pred_cols[f"Hybrid Voting ({m})"] = f"hybrid_voting_{label}_{m}"

    rows = []
    for cls in sorted(df[label].unique()):
        mask = df[label] == cls
        row = {"class": cls, "support": int(mask.sum())}
        for approach, col in pred_cols.items():
            sub = df[mask]
            f1 = f1_score(sub[label], sub[col], average="macro", zero_division=0,
                          labels=[cls])  # F1 untuk kelas ini saja
            row[approach] = round(f1, 4)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)
    return out


def voting_breakdown(df: pd.DataFrame, models: List[str], label: str = "category") -> pd.DataFrame:
    """Per voting model: berapa kasus dimana voting beda dari SVM/Fusion."""
    rows = []
    for m in models:
        vote_col = f"hybrid_voting_{label}_{m}"
        voter_col = f"genai_voter_{label}_{m}"
        if vote_col not in df.columns:
            continue

        same_as_svm    = (df[vote_col] == df[f"svm_{label}"]).sum()
        same_as_fusion = (df[vote_col] == df[f"fusion_{label}"]).sum()
        same_as_voter  = (df[vote_col] == df[voter_col]).sum() if voter_col in df.columns else 0

        # Vote results: agree vs disagree
        agree_all = ((df[f"svm_{label}"] == df[f"fusion_{label}"]) &
                     (df[f"fusion_{label}"] == df[voter_col])).sum() if voter_col in df.columns else 0
        agree_2of3 = (((df[f"svm_{label}"] == df[f"fusion_{label}"]) |
                       (df[f"svm_{label}"] == df[voter_col]) |
                       (df[f"fusion_{label}"] == df[voter_col])) &
                       ~((df[f"svm_{label}"] == df[f"fusion_{label}"]) &
                         (df[f"fusion_{label}"] == df[voter_col]))).sum() if voter_col in df.columns else 0
        all_disagree = (~((df[f"svm_{label}"] == df[f"fusion_{label}"]) |
                          (df[f"svm_{label}"] == df[voter_col]) |
                          (df[f"fusion_{label}"] == df[voter_col]))).sum() if voter_col in df.columns else 0

        rows.append({
            "model":            m,
            "label":            label,
            "total":            len(df),
            "all_3_agree":      agree_all,
            "2_of_3_agree":     agree_2of3,
            "all_3_disagree":   all_disagree,
            "vote_eq_svm":      same_as_svm,
            "vote_eq_fusion":   same_as_fusion,
            "vote_eq_voter":    same_as_voter,
        })
    return pd.DataFrame(rows)


def plot_comparison(table: pd.DataFrame, output_path: str):
    """Bar chart Acc vs F1 untuk semua approach."""
    cat = table[table["label"] == "category"]
    pri = table[table["label"] == "priority"]

    approaches = cat["approach"].tolist()
    cat_acc = cat["accuracy"].values
    cat_f1  = cat["macro_f1"].values
    pri_acc = pri["accuracy"].values if len(pri) else np.zeros(len(cat))
    pri_f1  = pri["macro_f1"].values if len(pri) else np.zeros(len(cat))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0f1419")

    for ax, (acc, f1, title) in zip(axes,
                                     [(cat_acc, cat_f1, "Category"),
                                      (pri_acc, pri_f1, "Priority")]):
        ax.set_facecolor("#1a2332")
        x = np.arange(len(approaches))
        w = 0.35
        ax.bar(x - w/2, acc, w, label="Accuracy", color="#5b9bd5")
        ax.bar(x + w/2, f1,  w, label="Macro F1", color="#3aa857")
        ax.set_xticks(x)
        ax.set_xticklabels(approaches, rotation=30, ha="right", color="#cdd6e0", fontsize=10)
        ax.set_title(f"Comparison — {title}",
                     color="#a5e8c4", fontsize=13, fontweight="bold")
        ax.legend(loc="lower left")
        ax.tick_params(colors="#cdd6e0")
        for spine in ax.spines.values():
            spine.set_color("#3a4a5c")
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#3a4a5c")
        ax.set_ylim(0, 1.0)
        # Annotate values
        for i, (a, f) in enumerate(zip(acc, f1)):
            ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8, color="#cdd6e0")
            ax.text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8, color="#cdd6e0")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analysis Tahap 2: Hybrid Voting comparison")
    parser.add_argument("--input",  default="results/voting_gpt41mini.xlsx",
                        help="Excel hasil run --enable-voting")
    parser.add_argument("--output", default="results/analysis_phase2.xlsx")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = load_voting_predictions(args.input)
    print(f"Loaded {len(df)} predictions")

    models = detect_voting_models(df)
    if not models:
        raise ValueError(f"Tidak ada kolom hybrid_voting_category_* di {args.input}. "
                         "Pastikan run pakai --enable-voting.")
    print(f"Detected voting models: {models}")

    print("\n[1/3] Building comparison table...")
    comparison = build_comparison_table(df, models)
    print(comparison.round(4).to_string(index=False))

    print("\n[2/3] Per-class F1 breakdown (category)...")
    per_class_cat = per_class_breakdown(df, models, label="category")
    print(per_class_cat.head(10).to_string(index=False))

    print("\n[3/3] Voting agreement breakdown...")
    vote_breakdown = voting_breakdown(df, models, label="category")
    print(vote_breakdown.to_string(index=False))

    print(f"\nSaving to {args.output}...")
    with pd.ExcelWriter(args.output, engine="openpyxl") as w:
        comparison.to_excel(w,    sheet_name="comparison",    index=False)
        per_class_cat.to_excel(w, sheet_name="per_class_cat", index=False)
        per_class_breakdown(df, models, label="priority").to_excel(
            w, sheet_name="per_class_pri", index=False)
        vote_breakdown.to_excel(w, sheet_name="voting_breakdown_cat", index=False)
        voting_breakdown(df, models, label="priority").to_excel(
            w, sheet_name="voting_breakdown_pri", index=False)

    plot_comparison(comparison, str(FIGURE_DIR / "comparison_3_architectures.png"))

    print(f"\nDone. Output: {args.output}")
    print(f"Figures: {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
