"""Tahap 3 untuk paper Q2: anchor bias ablation study.

Run 5 prompt variants pada subset baris yang SVM salah, ukur:
- override_rate: % LLM beda dari SVM
- correction_rate: % LLM benar (sama dengan ground truth)
- regression_rate: % LLM merusak prediksi yang awalnya benar

Hipotesis: prompt yang mention ML prediction sebagai context (V3 DEFER_ML)
membuat LLM anchored ke jawaban SVM, mengurangi correction rate.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Pastikan stdout/stderr bisa nge-print karakter non-ASCII di Windows console.
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

# Reuse helper dari main module
sys.path.insert(0, str(Path(__file__).parent))
from compare_svm_genai import _parse_json_response


# ============================================================
# Prompt Variants
# ============================================================

def prompt_v1_no_ml(text: str, allowed_cats: List[str], allowed_pris: List[str],
                    svm_cat: str, svm_pri: str, top3_cats: List[str]) -> str:
    """V1 NO_ML: Control — no mention of ML prediction."""
    return f"""You are an IT helpdesk ticket classifier.
Read the ticket and assign the most appropriate category and priority.
Return strict JSON only with keys: category, priority.

Allowed category values (pick exactly one):
{json.dumps(allowed_cats, ensure_ascii=True)}

Allowed priority values (pick exactly one):
{json.dumps(allowed_pris, ensure_ascii=True)}

Ticket description:
\"\"\"{text[:3000]}\"\"\""""


def prompt_v2_neutral_ml(text: str, allowed_cats: List[str], allowed_pris: List[str],
                         svm_cat: str, svm_pri: str, top3_cats: List[str]) -> str:
    """V2 NEUTRAL_ML: Mention ML prediction neutrally."""
    return f"""You are an IT helpdesk ticket classifier.
Classify the ticket. An ML model previously predicted: category='{svm_cat}', priority='{svm_pri}'. Reach your own conclusion based on the ticket text.
Return strict JSON only with keys: category, priority.

Allowed category values (pick exactly one):
{json.dumps(allowed_cats, ensure_ascii=True)}

Allowed priority values (pick exactly one):
{json.dumps(allowed_pris, ensure_ascii=True)}

Ticket description:
\"\"\"{text[:3000]}\"\"\""""


def prompt_v3_defer_ml(text: str, allowed_cats: List[str], allowed_pris: List[str],
                       svm_cat: str, svm_pri: str, top3_cats: List[str]) -> str:
    """V3 DEFER_ML: Strong anchor (replicates original Hybrid Correction prompt)."""
    return f"""You validate and refine ML predictions.
Return strict JSON only with keys: category, priority.

Allowed category values:
{json.dumps(allowed_cats, ensure_ascii=True)}

Allowed priority values:
{json.dumps(allowed_pris, ensure_ascii=True)}

Current ML prediction: category='{svm_cat}', priority='{svm_pri}'. Improve only if clearly needed.
Text:
\"\"\"{text[:3000]}\"\"\""""


def prompt_v4_challenge_ml(text: str, allowed_cats: List[str], allowed_pris: List[str],
                           svm_cat: str, svm_pri: str, top3_cats: List[str]) -> str:
    """V4 CHALLENGE_ML: Anti-anchor — ML may be wrong."""
    return f"""You are an IT helpdesk ticket classifier with quality assurance role.
An ML model predicted: category='{svm_cat}', priority='{svm_pri}'. The ML model is often wrong; verify the classification independently from the ticket text. Override the ML if the text clearly suggests a different label.
Return strict JSON only with keys: category, priority.

Allowed category values (pick exactly one):
{json.dumps(allowed_cats, ensure_ascii=True)}

Allowed priority values (pick exactly one):
{json.dumps(allowed_pris, ensure_ascii=True)}

Ticket description:
\"\"\"{text[:3000]}\"\"\""""


def prompt_v5_top3_choices(text: str, allowed_cats: List[str], allowed_pris: List[str],
                           svm_cat: str, svm_pri: str, top3_cats: List[str]) -> str:
    """V5 TOP3_CHOICES: Multiple-choice from top-3 SVM (constrained)."""
    return f"""You are an IT helpdesk ticket classifier.
Pick the most appropriate category from the candidate list and the most appropriate priority.
Return strict JSON only with keys: category, priority.

Category candidates (pick exactly one from this shortlist):
{json.dumps(top3_cats, ensure_ascii=True)}

Priority values (pick exactly one):
{json.dumps(allowed_pris, ensure_ascii=True)}

Ticket description:
\"\"\"{text[:3000]}\"\"\""""


VARIANTS: Dict[str, Tuple[str, Callable]] = {
    "V1_NO_ML":         ("No mention of ML prediction",   prompt_v1_no_ml),
    "V2_NEUTRAL_ML":    ("ML mentioned neutrally",        prompt_v2_neutral_ml),
    "V3_DEFER_ML":      ("Strong defer (original)",       prompt_v3_defer_ml),
    "V4_CHALLENGE_ML":  ("ML may be wrong (anti-anchor)", prompt_v4_challenge_ml),
    "V5_TOP3_CHOICES":  ("Multiple-choice from top-3",    prompt_v5_top3_choices),
}


# ============================================================
# Top-3 SVM kandidat (gunakan decision_function dari pipeline yang sudah dilatih)
# ============================================================

def get_top3_svm_for_test(test_df: pd.DataFrame, train_df: pd.DataFrame, k: int = 3) -> List[List[str]]:
    """Train SVM ulang di train_df, prediksi top-k untuk tiap baris test_df.
    Pakai pipeline yang sama dengan compare_svm_genai (TF-IDF + LinearSVC)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

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


# ============================================================
# Run ablation
# ============================================================

def call_llm(client: OpenAI, model: str, prompt: str,
             allowed_cats: List[str], allowed_pris: List[str],
             default_cat: str, default_pri: str,
             retries: int = 2) -> Tuple[str, str, bool]:
    """Call LLM dan parse hasil. Return (cat, pri, success_flag)."""
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


def run_ablation(
    voting_xlsx: str = "results/voting_gpt41mini_v2.xlsx",
    input_data:  str = "data/cobacek_filtered.xlsx",
    category_col: str = "category_filtered",
    output_xlsx: str = "results/anchor_bias_ablation.xlsx",
    n_samples: int = 300,
    model: str = "gpt-4.1-mini",
    seed: int = 42,
):
    load_dotenv()
    client = OpenAI()

    print(f"Loading test predictions dari {voting_xlsx}...")
    test_df = pd.read_excel(voting_xlsx, sheet_name="Predictions_Compare")

    # Setup target column (sama dengan main pipeline)
    if category_col != "category" and category_col in test_df.columns:
        test_df["category"] = test_df[category_col]
        test_df = test_df.drop(columns=[category_col])

    # Train ulang SVM untuk dapat top-3 (test_df adalah hasil dari random_state=42 split)
    df = pd.read_excel(input_data)
    if category_col != "category" and category_col in df.columns:
        df["category"] = df[category_col]
        df = df.drop(columns=[category_col])
    from sklearn.model_selection import train_test_split
    cat_counts = df["category"].value_counts()
    rare_cats = set(cat_counts[cat_counts < 2].index)
    strat_key = df["category"].where(~df["category"].isin(rare_cats), "_rare_")
    train_df, _ = train_test_split(df, test_size=0.2, random_state=seed, stratify=strat_key)
    train_df = train_df.reset_index(drop=True)

    print(f"Computing top-3 SVM kandidat untuk {len(test_df)} baris test...")
    top3_list = get_top3_svm_for_test(test_df, train_df, k=3)
    test_df["top3_svm_categories"] = top3_list

    # Filter ke baris yang SVM SALAH (untuk fair comparison: setiap variant punya
    # something to correct)
    svm_wrong = test_df[
        (test_df["svm_category"] != test_df["category"]) |
        (test_df["svm_priority"] != test_df["priority"])
    ].reset_index(drop=True)
    print(f"Baris SVM salah: {len(svm_wrong)}/{len(test_df)} "
          f"({100 * len(svm_wrong) / len(test_df):.1f}%)")

    # Sample N rows (deterministic via seed)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(svm_wrong), size=min(n_samples, len(svm_wrong)),
                            replace=False)
    sample = svm_wrong.iloc[sample_idx].reset_index(drop=True)
    print(f"Sampled {len(sample)} baris untuk ablation\n")

    allowed_cats = sorted(test_df["category"].dropna().unique().tolist())
    allowed_pris = sorted(test_df["priority"].dropna().unique().tolist())
    default_cat  = allowed_cats[0]
    default_pri  = allowed_pris[0]

    # Run setiap variant
    all_records = []  # baris-level result, satu per (sample, variant)
    for var_id, (var_desc, prompt_fn) in VARIANTS.items():
        print(f"\n=== {var_id}: {var_desc} ===")
        for i, row in sample.iterrows():
            prompt = prompt_fn(
                text=row["description"],
                allowed_cats=allowed_cats,
                allowed_pris=allowed_pris,
                svm_cat=row["svm_category"],
                svm_pri=row["svm_priority"],
                top3_cats=row["top3_svm_categories"],
            )
            llm_cat, llm_pri, success = call_llm(
                client, model, prompt,
                allowed_cats, allowed_pris, default_cat, default_pri,
            )
            all_records.append({
                "variant":      var_id,
                "sample_idx":   i,
                "true_cat":     row["category"],
                "true_pri":     row["priority"],
                "svm_cat":      row["svm_category"],
                "svm_pri":      row["svm_priority"],
                "llm_cat":      llm_cat,
                "llm_pri":      llm_pri,
                "parse_ok":     success,
                # Per-row indikator (Cat fokus utama)
                "cat_override":          int(llm_cat != row["svm_category"]),
                "cat_correction":        int(llm_cat != row["svm_category"]
                                             and llm_cat == row["category"]),
                "cat_wrong_llm":         int(llm_cat != row["category"]),
                # Priority indikator
                "pri_override":          int(llm_pri != row["svm_pri"] if "svm_pri" in row else 0),
                "pri_correction":        int(llm_pri != row["svm_priority"]
                                             and llm_pri == row["priority"]),
                "pri_wrong_llm":         int(llm_pri != row["priority"]),
            })
            if (i + 1) % 50 == 0:
                print(f"  {var_id} processed {i + 1}/{len(sample)}")

    detail_df = pd.DataFrame(all_records)

    # Aggregate per variant
    print("\n=== AGGREGATE per variant ===")
    summary_rows = []
    for var_id in VARIANTS:
        sub = detail_df[detail_df["variant"] == var_id]
        n = len(sub)
        # Cat metrics
        cat_override     = sub["cat_override"].sum()
        cat_correction   = sub["cat_correction"].sum()
        cat_wrong_llm    = sub["cat_wrong_llm"].sum()
        cat_correct_llm  = n - cat_wrong_llm
        # Pri metrics
        pri_correction   = sub["pri_correction"].sum()

        summary_rows.append({
            "variant":               var_id,
            "description":           VARIANTS[var_id][0],
            "n":                     n,
            "cat_override_rate":     round(cat_override / n, 4),
            "cat_correction_rate":   round(cat_correction / n, 4),
            "cat_correct_llm":       cat_correct_llm,
            "cat_correct_llm_rate":  round(cat_correct_llm / n, 4),
            "parse_success_rate":    round(sub["parse_ok"].mean(), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Save
    print(f"\nSaving to {output_xlsx}...")
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="summary",       index=False)
        detail_df.to_excel(w,  sheet_name="per_row_detail", index=False)

    print("Done.")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voting-xlsx", default="results/voting_gpt41mini_v2.xlsx",
                        help="Excel hasil voting v2 (untuk ambil test_df + svm predictions)")
    parser.add_argument("--input-data",  default="data/cobacek_filtered.xlsx")
    parser.add_argument("--category-col", default="category_filtered")
    parser.add_argument("--output-xlsx", default="results/anchor_bias_ablation.xlsx")
    parser.add_argument("--n-samples",   type=int, default=300)
    parser.add_argument("--model",       default="gpt-4.1-mini")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    run_ablation(
        voting_xlsx=args.voting_xlsx,
        input_data=args.input_data,
        category_col=args.category_col,
        output_xlsx=args.output_xlsx,
        n_samples=args.n_samples,
        model=args.model,
        seed=args.seed,
    )
