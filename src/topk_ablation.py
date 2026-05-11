"""Tahap 3b: validate hypothesis "label space size matters".

Run V5 prompt (top-K constrained) dengan K bervariasi (3, 5, 7, 10).
Kalau correction_rate menurun seiring K naik, itu kuat-kan claim:
constraining LLM ke smaller candidate set adalah real lever, BUKAN
anchor framing.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))
from compare_svm_genai import _parse_json_response


def get_topk_svm(test_df: pd.DataFrame, train_df: pd.DataFrame, k: int) -> List[List[str]]:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("svm",   LinearSVC()),
    ])
    pipe.fit(train_df["description"], train_df["category"])
    scores = pipe.decision_function(test_df["description"].tolist())
    classes = pipe.named_steps["svm"].classes_
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    sorted_idx = np.argsort(-scores, axis=1)
    topk_idx = sorted_idx[:, :k]
    return [[str(classes[j]) for j in row] for row in topk_idx]


def build_topk_prompt(text: str, topk_cats: List[str], allowed_pris: List[str]) -> str:
    return f"""You are an IT helpdesk ticket classifier.
Pick the most appropriate category from the candidate list and the most appropriate priority.
Return strict JSON only with keys: category, priority.

Category candidates (pick exactly one from this shortlist):
{json.dumps(topk_cats, ensure_ascii=True)}

Priority values (pick exactly one):
{json.dumps(allowed_pris, ensure_ascii=True)}

Ticket description:
\"\"\"{text[:3000]}\"\"\""""


def call_llm(client: OpenAI, model: str, prompt: str,
             allowed_cats: List[str], allowed_pris: List[str],
             default_cat: str, default_pri: str,
             retries: int = 2):
    for attempt in range(retries):
        try:
            resp = client.responses.create(model=model, input=prompt)
            raw = (resp.output_text or "").strip()
            parsed = _parse_json_response(raw)
            cat = str(parsed.get("category", default_cat)).strip()
            pri = str(parsed.get("priority", default_pri)).strip()
            if cat not in allowed_cats:
                cat = default_cat
            if pri not in allowed_pris:
                pri = default_pri
            return cat, pri, True
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return default_cat, default_pri, False
    return default_cat, default_pri, False


def run_topk_sweep(
    voting_xlsx: str = "results/voting_gpt41mini_v2.xlsx",
    input_data:  str = "data/cobacek_filtered.xlsx",
    category_col: str = "category_filtered",
    output_xlsx: str = "results/topk_ablation.xlsx",
    n_samples: int = 300,
    model: str = "gpt-4.1-mini",
    seed: int = 42,
    topk_list: List[int] = (3, 5, 7, 10, 19),
):
    load_dotenv()
    client = OpenAI()

    print(f"Loading {voting_xlsx}...")
    test_df = pd.read_excel(voting_xlsx, sheet_name="Predictions_Compare")
    if category_col != "category" and category_col in test_df.columns:
        test_df["category"] = test_df[category_col]
        test_df = test_df.drop(columns=[category_col])

    df = pd.read_excel(input_data)
    if category_col != "category" and category_col in df.columns:
        df["category"] = df[category_col]
        df = df.drop(columns=[category_col])

    cat_counts = df["category"].value_counts()
    rare_cats = set(cat_counts[cat_counts < 2].index)
    strat_key = df["category"].where(~df["category"].isin(rare_cats), "_rare_")
    train_df, _ = train_test_split(df, test_size=0.2, random_state=seed, stratify=strat_key)
    train_df = train_df.reset_index(drop=True)

    # Filter SVM-wrong & sample (sama seed dengan anchor_bias_ablation supaya
    # comparable dengan V5 K=3 dari ablation utama)
    svm_wrong = test_df[
        (test_df["svm_category"] != test_df["category"]) |
        (test_df["svm_priority"] != test_df["priority"])
    ].reset_index(drop=True)
    print(f"SVM-wrong rows: {len(svm_wrong)}")

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(svm_wrong), size=min(n_samples, len(svm_wrong)),
                            replace=False)
    sample = svm_wrong.iloc[sample_idx].reset_index(drop=True)
    print(f"Sampled {len(sample)} rows untuk top-K ablation\n")

    allowed_cats = sorted(test_df["category"].dropna().unique().tolist())
    allowed_pris = sorted(test_df["priority"].dropna().unique().tolist())
    default_cat  = allowed_cats[0]
    default_pri  = allowed_pris[0]

    n_classes_total = len(allowed_cats)
    print(f"Total kelas tersedia: {n_classes_total}")

    # Compute top-K for max K, lalu trim
    max_k = max(topk_list)
    effective_max = min(max_k, n_classes_total)
    print(f"Computing top-{effective_max} kandidat SVM untuk semua test rows...")
    topk_full = get_topk_svm(test_df, train_df, k=effective_max)
    # Cache by sample idx (sample.iloc[i] originally from svm_wrong, originally from test_df)
    sample_test_indices = svm_wrong.index[sample_idx]
    # But we already reset_index on sample — need to map differently
    # Solution: compute top-K for sample directly using their text
    print("Computing top-K for sample rows...")
    sample_topk_full = get_topk_svm(sample, train_df, k=effective_max)

    all_records = []
    for k in topk_list:
        k_eff = min(k, n_classes_total)
        print(f"\n=== Top-{k_eff} (target K={k}) ===")
        for i, row in sample.iterrows():
            cats_topk = sample_topk_full[i][:k_eff]
            # Untuk K=19 (atau k>=n_classes), pakai semua allowed_cats (sorted)
            if k_eff >= n_classes_total:
                cats_topk = allowed_cats
            prompt = build_topk_prompt(row["description"], cats_topk, allowed_pris)
            llm_cat, llm_pri, success = call_llm(
                client, model, prompt,
                allowed_cats, allowed_pris, default_cat, default_pri,
            )
            all_records.append({
                "k":                  k_eff,
                "sample_idx":         i,
                "true_cat":           row["category"],
                "svm_cat":            row["svm_category"],
                "llm_cat":            llm_cat,
                "parse_ok":           success,
                "cat_override":       int(llm_cat != row["svm_category"]),
                "cat_correction":     int(llm_cat != row["svm_category"]
                                          and llm_cat == row["category"]),
                "cat_correct_llm":    int(llm_cat == row["category"]),
            })
            if (i + 1) % 50 == 0:
                print(f"  K={k_eff} processed {i + 1}/{len(sample)}")

    detail = pd.DataFrame(all_records)

    print("\n=== AGGREGATE ===")
    summary = (
        detail.groupby("k")
        .agg(
            n=("sample_idx", "count"),
            cat_override_rate=("cat_override", "mean"),
            cat_correction_rate=("cat_correction", "mean"),
            cat_correct_llm_rate=("cat_correct_llm", "mean"),
            parse_success_rate=("parse_ok", "mean"),
        )
        .round(4)
        .reset_index()
    )
    print(summary.to_string(index=False))

    print(f"\nSaving to {output_xlsx}...")
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary",       index=False)
        detail.to_excel(w,  sheet_name="per_row_detail", index=False)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voting-xlsx", default="results/voting_gpt41mini_v2.xlsx")
    parser.add_argument("--input-data",  default="data/cobacek_filtered.xlsx")
    parser.add_argument("--category-col", default="category_filtered")
    parser.add_argument("--output-xlsx", default="results/topk_ablation.xlsx")
    parser.add_argument("--n-samples",   type=int, default=300)
    parser.add_argument("--model",       default="gpt-4.1-mini")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--topk-list",   default="3,5,7,10,19",
                        help="Comma-separated K values (e.g., 3,5,7,10,19)")
    args = parser.parse_args()

    topk_list = [int(x) for x in args.topk_list.split(",")]
    run_topk_sweep(
        voting_xlsx=args.voting_xlsx,
        input_data=args.input_data,
        category_col=args.category_col,
        output_xlsx=args.output_xlsx,
        n_samples=args.n_samples,
        model=args.model,
        seed=args.seed,
        topk_list=topk_list,
    )
