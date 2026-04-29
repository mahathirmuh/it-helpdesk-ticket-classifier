import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bert_classifier import BertClassifier


def safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def pick_model(client: OpenAI, candidates: List[str]) -> str:
    try:
        models = client.models.list()
        available = {m.id for m in models.data}
        for model in candidates:
            if model in available:
                return model
    except Exception:
        pass
    return candidates[0]


def get_available_models(client: OpenAI) -> List[str]:
    models = client.models.list()
    return sorted({m.id for m in models.data})


def build_svm_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("svm", LinearSVC()),
    ])


def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])


def build_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def classify_with_genai(
    client: OpenAI,
    model: str,
    text: str,
    allowed_categories: List[str],
    allowed_priorities: List[str],
    ml_category: Optional[str] = None,
    ml_priority: Optional[str] = None,
    retries: int = 2,
) -> Tuple[str, str]:
    default_category = ml_category or allowed_categories[0]
    default_priority = ml_priority or allowed_priorities[0]

    context_line = (
        f"Current ML prediction: category='{ml_category}', priority='{ml_priority}'. "
        "Improve only if clearly needed."
    )

    prompt = f"""
You validate and refine ML predictions.
Return strict JSON only with keys: category, priority.

Allowed category values:
{json.dumps(allowed_categories, ensure_ascii=True)}

Allowed priority values:
{json.dumps(allowed_priorities, ensure_ascii=True)}

{context_line}
Text:
\"\"\"{text[:3000]}\"\"\"
""".strip()

    for attempt in range(retries):
        try:
            response = client.responses.create(model=model, input=prompt)
            raw = (response.output_text or "").strip()
            parsed = json.loads(raw)
            category = str(parsed.get("category", default_category)).strip()
            priority = str(parsed.get("priority", default_priority)).strip()
            if category not in allowed_categories:
                category = default_category
            if priority not in allowed_priorities:
                priority = default_priority
            return category, priority
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return default_category, default_priority

    return default_category, default_priority


def metrics_dict(y_true: List[str], y_pred: List[str], label_name: str) -> Dict[str, float]:
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return {
        "label": label_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "samples": len(y_true),
    }


def run_pipeline(
    input_file: str = "cobacek.xlsx",
    output_file: str = "cobacek_compare.xlsx",
    max_genai_rows: int = 100,
    forced_model: Optional[str] = None,
    multi_models: Optional[List[str]] = None,
    bert_model_name: str = "distilbert-base-multilingual-cased",
    bert_epochs: int = 3,
    skip_bert: bool = False,
    skip_lr: bool = False,
) -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY tidak ditemukan di .env")

    df = pd.read_excel(input_file)
    required = ["description", "category", "priority"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada: {missing}")

    df["description"] = df["description"].apply(safe_str)
    df["category"] = df["category"].apply(safe_str)
    df["priority"] = df["priority"].apply(safe_str)

    # --- SVM ---
    svm_cat_model = build_svm_pipeline()
    svm_pri_model = build_svm_pipeline()
    svm_cat_model.fit(df["description"], df["category"])
    svm_pri_model.fit(df["description"], df["priority"])
    df["svm_category"] = svm_cat_model.predict(df["description"])
    df["svm_priority"] = svm_pri_model.predict(df["description"])
    df["svm_match"] = (
        (df["svm_category"] == df["category"]) &
        (df["svm_priority"] == df["priority"])
    )
    print(f"SVM Category Acc: {accuracy_score(df['category'], df['svm_category']):.4f} | "
          f"Priority Acc: {accuracy_score(df['priority'], df['svm_priority']):.4f}")

    # --- Random Forest ---
    rf_cat_model = build_rf_pipeline()
    rf_pri_model = build_rf_pipeline()
    rf_cat_model.fit(df["description"], df["category"])
    rf_pri_model.fit(df["description"], df["priority"])
    df["rf_category"] = rf_cat_model.predict(df["description"])
    df["rf_priority"] = rf_pri_model.predict(df["description"])
    df["rf_match"] = (
        (df["rf_category"] == df["category"]) &
        (df["rf_priority"] == df["priority"])
    )
    print(f"RF  Category Acc: {accuracy_score(df['category'], df['rf_category']):.4f} | "
          f"Priority Acc: {accuracy_score(df['priority'], df['rf_priority']):.4f}")

    # --- Logistic Regression ---
    lr_available = not skip_lr
    if lr_available:
        lr_cat_model = build_lr_pipeline()
        lr_pri_model = build_lr_pipeline()
        lr_cat_model.fit(df["description"], df["category"])
        lr_pri_model.fit(df["description"], df["priority"])
        df["lr_category"] = lr_cat_model.predict(df["description"])
        df["lr_priority"] = lr_pri_model.predict(df["description"])
        df["lr_match"] = (
            (df["lr_category"] == df["category"]) &
            (df["lr_priority"] == df["priority"])
        )
        print(f"LR  Category Acc: {accuracy_score(df['category'], df['lr_category']):.4f} | "
              f"Priority Acc: {accuracy_score(df['priority'], df['lr_priority']):.4f}")

    # --- BERT ---
    bert_available = not skip_bert
    if bert_available:
        print(f"\nTraining BERT ({bert_model_name}) — bisa memakan beberapa menit di CPU...")
        bert_cat_model = BertClassifier(model_name=bert_model_name, epochs=bert_epochs)
        bert_pri_model = BertClassifier(model_name=bert_model_name, epochs=bert_epochs)

        print("[BERT] Training category model...")
        bert_cat_model.fit(df["description"].tolist(), df["category"].tolist())
        print("[BERT] Training priority model...")
        bert_pri_model.fit(df["description"].tolist(), df["priority"].tolist())

        df["bert_category"] = bert_cat_model.predict(df["description"].tolist())
        df["bert_priority"] = bert_pri_model.predict(df["description"].tolist())
        df["bert_match"] = (
            (df["bert_category"] == df["category"]) &
            (df["bert_priority"] == df["priority"])
        )
        print(f"BERT Category Acc: {accuracy_score(df['category'], df['bert_category']):.4f} | "
              f"Priority Acc: {accuracy_score(df['priority'], df['bert_priority']):.4f}")

    allowed_categories = sorted(df["category"].dropna().unique().tolist())
    allowed_priorities = sorted(df["priority"].dropna().unique().tolist())

    client = OpenAI(api_key=api_key)
    available_models = get_available_models(client)
    model_candidates = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-5-mini", "gpt-4.1", "gpt-4o"]
    selected_models: List[str] = []
    if multi_models:
        selected_models = [m.strip() for m in multi_models if m.strip()]
    elif forced_model:
        selected_models = [forced_model]
    elif os.getenv("OPENAI_MODELS"):
        selected_models = [m.strip() for m in os.getenv("OPENAI_MODELS").split(",") if m.strip()]
    else:
        selected_models = [pick_model(client, model_candidates)]

    for selected_model in selected_models:
        if selected_model not in available_models:
            raise ValueError(
                f"Model '{selected_model}' tidak tersedia. "
                f"Model tersedia: {', '.join(available_models)}"
            )

    svm_mismatch_mask = (
        (df["svm_category"] != df["category"]) |
        (df["svm_priority"] != df["priority"])
    )
    svm_mismatch_indices = df[svm_mismatch_mask].index[
        : min(max_genai_rows, svm_mismatch_mask.sum())
    ]

    # Metrik baseline ML
    metrics_rows: List[Dict] = [
        {"approach": "SVM",           **metrics_dict(df["category"].tolist(), df["svm_category"].tolist(), "category")},
        {"approach": "SVM",           **metrics_dict(df["priority"].tolist(),  df["svm_priority"].tolist(),  "priority")},
        {"approach": "Random Forest", **metrics_dict(df["category"].tolist(), df["rf_category"].tolist(),  "category")},
        {"approach": "Random Forest", **metrics_dict(df["priority"].tolist(),  df["rf_priority"].tolist(),   "priority")},
    ]
    if bert_available:
        metrics_rows += [
            {"approach": "BERT", **metrics_dict(df["category"].tolist(), df["bert_category"].tolist(), "category")},
            {"approach": "BERT", **metrics_dict(df["priority"].tolist(),  df["bert_priority"].tolist(),  "priority")},
        ]
    if lr_available:
        metrics_rows += [
            {"approach": "Logistic Regression", **metrics_dict(df["category"].tolist(), df["lr_category"].tolist(), "category")},
            {"approach": "Logistic Regression", **metrics_dict(df["priority"].tolist(),  df["lr_priority"].tolist(),  "priority")},
        ]

    for selected_model in selected_models:
        model_key = re.sub(r"[^a-zA-Z0-9]+", "_", selected_model).strip("_").lower()
        hybrid_svm_cat_col = f"hybrid_svm_category_{model_key}"
        hybrid_svm_pri_col = f"hybrid_svm_priority_{model_key}"

        df[hybrid_svm_cat_col] = df["svm_category"]
        df[hybrid_svm_pri_col] = df["svm_priority"]

        # Hybrid SVM: SVM prediksi, GenAI koreksi mismatch
        print(f"\n[Hybrid SVM] {selected_model}")
        for processed, idx in enumerate(svm_mismatch_indices, start=1):
            new_cat, new_pri = classify_with_genai(
                client=client, model=selected_model,
                text=df.at[idx, "description"],
                allowed_categories=allowed_categories,
                allowed_priorities=allowed_priorities,
                ml_category=df.at[idx, "svm_category"],
                ml_priority=df.at[idx, "svm_priority"],
            )
            df.at[idx, hybrid_svm_cat_col] = new_cat
            df.at[idx, hybrid_svm_pri_col] = new_pri
            if processed % 20 == 0:
                print(f"  processed {processed}/{len(svm_mismatch_indices)}")

        metrics_rows.append({"approach": f"Hybrid SVM ({selected_model})",
            **metrics_dict(df["category"].tolist(), df[hybrid_svm_cat_col].tolist(), "category")})
        metrics_rows.append({"approach": f"Hybrid SVM ({selected_model})",
            **metrics_dict(df["priority"].tolist(),  df[hybrid_svm_pri_col].tolist(),  "priority")})

    metrics_df = pd.DataFrame(metrics_rows)

    summary_rows = [
        {"key": "selected_models",             "value": ", ".join(selected_models)},
        {"key": "bert_model",                  "value": bert_model_name if bert_available else "skipped"},
        {"key": "total_rows",                  "value": len(df)},
        {"key": "svm_mismatch_rows_corrected", "value": len(svm_mismatch_indices)},
        {"key": "max_genai_rows",              "value": max_genai_rows},
    ]
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer,         index=False, sheet_name="Predictions_Compare")
        metrics_df.to_excel(writer, index=False, sheet_name="Metrics")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")

    print(f"\nDone. Output: {output_file} ({len(df)} baris total)")
    print(f"Models: {', '.join(selected_models)} | BERT: {bert_model_name if bert_available else 'skipped'} | LR: {'enabled' if lr_available else 'skipped'}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bandingkan SVM vs RF vs LR vs BERT vs Hybrid SVM")
    parser.add_argument("--input",          default="cobacek.xlsx",         help="Path file input Excel")
    parser.add_argument("--output",         default="cobacek_compare.xlsx", help="Path file output Excel")
    parser.add_argument("--max-genai-rows", type=int, default=100,          help="Batas baris untuk inferensi GenAI (Hybrid SVM)")
    parser.add_argument("--model",          default=None,                   help="Paksa model GenAI tertentu")
    parser.add_argument("--models",         default=None,
        help="Daftar model dipisah koma (contoh: gpt-4.1-mini,gpt-4o-mini)")
    parser.add_argument("--bert-model",     default="distilbert-base-multilingual-cased",
        help="HuggingFace model ID untuk BERT (default: distilbert-base-multilingual-cased)")
    parser.add_argument("--bert-epochs",    type=int, default=3,            help="Jumlah epoch fine-tuning BERT")
    parser.add_argument("--skip-bert",      action="store_true",            help="Lewati BERT")
    parser.add_argument("--skip-lr",        action="store_true",            help="Lewati Logistic Regression")
    args = parser.parse_args()

    models_arg = None
    if args.models:
        models_arg = [m.strip() for m in args.models.split(",") if m.strip()]

    run_pipeline(
        input_file=args.input,
        output_file=args.output,
        max_genai_rows=args.max_genai_rows,
        forced_model=args.model,
        multi_models=models_arg,
        bert_model_name=args.bert_model,
        bert_epochs=args.bert_epochs,
        skip_bert=args.skip_bert,
        skip_lr=args.skip_lr,
    )
