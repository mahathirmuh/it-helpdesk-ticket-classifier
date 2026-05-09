"""Bikin heatmap tabel comparison hasil Hybrid SVM-GenAI experiment.
Output: PNG dengan tampilan mirip tabel comparison di paper.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def load_metrics(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name="Metrics")
    return df


def pivot_for_heatmap(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pivot ke shape: rows=approach, cols=(metric, label).
    Kolom akhir: Acc Cat, Acc Pri, Prec Cat, Prec Pri, ..., Samples
    """
    pivots = []
    for metric_col, short in [
        ("accuracy",        "Acc"),
        ("macro_precision", "Prec"),
        ("macro_recall",    "Rec"),
        ("macro_f1",        "F1"),
        ("weighted_f1",     "wF1"),
    ]:
        p = metrics.pivot(index="approach", columns="label", values=metric_col)
        p.columns = [f"{short} {c[:3].capitalize()}" for c in p.columns]
        pivots.append(p)
    samples = metrics.pivot(index="approach", columns="label", values="samples").iloc[:, 0]
    samples.name = "Samples"
    out = pd.concat(pivots + [samples], axis=1)
    # Re-order kolom: Acc Cat, Acc Pri, Prec Cat, Prec Pri, Rec Cat, Rec Pri, F1 Cat, F1 Pri, wF1 Cat, wF1 Pri, Samples
    desired = []
    for short in ["Acc", "Prec", "Rec", "F1", "wF1"]:
        for label in ["Cat", "Pri"]:
            col = f"{short} {label}"
            if col in out.columns:
                desired.append(col)
    desired.append("Samples")
    out = out[desired]
    return out


def plot_heatmap(table: pd.DataFrame, output_path: str, title: str = "PERBANDINGAN SEMUA SKEMA") -> None:
    """Bikin heatmap dengan colorscale red-yellow-green (low-mid-high).
    Samples kolom tidak diwarnai (gelap netral).
    """
    metric_cols = [c for c in table.columns if c != "Samples"]
    samples_col = "Samples"

    n_rows = len(table)
    n_cols = len(table.columns)

    fig, ax = plt.subplots(figsize=(max(11, n_cols * 1.0), max(2.5, n_rows * 0.55)))
    fig.patch.set_facecolor("#0f1419")
    ax.set_facecolor("#0f1419")

    # Custom red→yellow→green
    cmap = LinearSegmentedColormap.from_list(
        "rygreen",
        ["#c1444a", "#e0a93b", "#3aa857"],
    )

    # Per-kolom normalisasi (min-max within column)
    cell_colors = np.zeros((n_rows, n_cols, 4))
    for j, col in enumerate(table.columns):
        vals = table[col].values.astype(float)
        if col == samples_col:
            cell_colors[:, j] = (0.18, 0.22, 0.28, 1.0)
        else:
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            if vmax > vmin:
                norm_vals = (vals - vmin) / (vmax - vmin)
            else:
                norm_vals = np.full_like(vals, 0.5)
            cell_colors[:, j] = cmap(norm_vals)

    ax.imshow(cell_colors, aspect="auto")

    # Tulis angka di tiap cell
    for i in range(n_rows):
        for j, col in enumerate(table.columns):
            val = table.iloc[i, j]
            if col == samples_col:
                txt = f"{int(val)}"
            else:
                txt = f"{val:.4f}"
            ax.text(j, i, txt, ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(table.columns, color="#cdd6e0", fontsize=10, rotation=0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(table.index, color="#cdd6e0", fontsize=11)

    # Hapus axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)

    # Title bar
    ax.set_title(title, color="#a5e8c4", fontsize=14, fontweight="bold",
                 loc="left", pad=20, family="monospace")

    plt.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.05)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Heatmap disimpan: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualisasi heatmap hasil comparison")
    parser.add_argument("--input",  default="results/cobacek_filtered_compare.xlsx",
                        help="Path Excel hasil run (sheet 'Metrics')")
    parser.add_argument("--output", default="results/heatmap_comparison.png",
                        help="Path output PNG")
    parser.add_argument("--title",  default="PERBANDINGAN SEMUA SKEMA",
                        help="Title heatmap")
    args = parser.parse_args()

    metrics = load_metrics(args.input)
    table = pivot_for_heatmap(metrics)
    print("\n=== Tabel comparison ===")
    print(table.to_string())
    print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_heatmap(table, args.output, title=args.title)


if __name__ == "__main__":
    main()
